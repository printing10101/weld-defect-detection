"""本地大模型（llama.cpp）的模型发现、清单与选中。

端点
----
- ``GET  /api/v1/llm/status``  引擎状态 + 当前选中 + 扫描进度 + GPU 快照（不探测端点，快）；
- ``GET  /api/v1/llm/models``  **全部**本机可用模型（本地 GGUF + 已有服务上的模型）；
- ``GET  /api/v1/llm/services`` 探测本机已运行的 llama 兼容端点（含不可达项，便于排查）；
- ``POST /api/v1/llm/scan``         触发后台全盘扫描（sysadmin）；
- ``POST /api/v1/llm/scan/cancel``  取消扫描（sysadmin，保留已扫到的结果）；
- ``POST /api/v1/llm/models/{id}/select`` 选中模型并热应用（sysadmin，审计留痕）；
- ``POST /api/v1/llm/dirs`` / ``DELETE /api/v1/llm/dirs`` 增删模型目录（sysadmin，审计留痕）。

设计口径
--------
- **不做可行性过滤**：16GB 显存下 30B MoE 权重放不下，但列表照列，只在
  ``vram.verdict`` 上标注 ``infeasible``/``partial_offload``/``tight``/``full_gpu``
  （由 :func:`backend.infra.gguf_meta.estimate_vram` 依据实测空闲显存判定）。
  隐藏不可行项会让用户以为"本机没有这个模型"，与事实不符；
- **同模型多副本**：全部列出，主条目带 ``duplicate_count``、副本带 ``duplicate_of``，
  由前端决定如何分组，不在服务端折叠掉；
- **不可用条目**（GGUF 头解析失败 / 端点要求鉴权）照列，``available=false`` +
  ``error`` 说明原因；**选中它们会被拒绝**（409），不把"选了个跑不了的"当成功；
- ``model_id`` 是不透明串（``gguf:<sha1>`` / ``svc:<sha1>``），客户端只回传不解析。

安全与审计
----------
- 全部端点要求已登录（由 main.py 统一挂 ``get_principal``）；变更类端点另加
  ``require_role("sysadmin")``——选中模型等价于切换全系统的推理后端，属投产类操作；
- 扫描与选中、目录增删均写主审计链（``llm_*`` 动作）；
- 扫描只读（不移动/删除文件），端点探测仅限回环，API Key 只从环境变量读取。
"""

from __future__ import annotations

import logging
import threading
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from backend.app.auth import require_role
from backend.app.dependencies import Registry, get_operator_name, get_registry
from backend.infra.llm_registry import (
    SOURCE_LOCAL,
    SOURCE_SERVICE,
    LlmModelEntry,
    LlmRegistry,
)

router = APIRouter(tags=["llm"])

_LOG = logging.getLogger("scandetection")

_registry_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# 响应模型
# --------------------------------------------------------------------------- #


class LlmModelInfo(BaseModel):
    """单个可选模型（字段与 :class:`LlmModelEntry` 一一对应）。"""

    id: str
    name: str
    display_name: str
    source: str  # local | service
    path: str = ""
    endpoint: str = ""
    size_bytes: int = 0
    architecture: str = ""
    quant: str = ""
    max_ctx: int | None = None
    is_embedding: bool = False
    available: bool = True
    error: str | None = None
    active: bool = False
    primary: bool = True
    duplicate_of: str | None = None
    duplicate_count: int = 0
    vram: dict | None = None
    notes: list[str] = []


class LlmModelsResponse(BaseModel):
    active_id: str | None = None
    selection: dict
    gpu: dict | None = None
    scan: dict
    counts: dict
    services: list[dict] = []
    models: list[LlmModelInfo] = []


class LlmStatusResponse(BaseModel):
    engine: dict
    selection: dict
    scan: dict
    gpu: dict | None = None
    model_dirs: list[str] = []
    counts: dict


class SelectResponse(BaseModel):
    ok: bool
    active: str | None
    mode: str
    endpoint: str = ""
    model_path: str = ""
    reloaded: bool = False
    engine: dict
    warning: str | None = None  # 显存可行性提醒（不阻断，只是如实告知）


class ScanResponse(BaseModel):
    ok: bool
    started: bool
    status: dict


class ServicesResponse(BaseModel):
    services: list[dict]


class DirRequest(BaseModel):
    path: str


class DirsResponse(BaseModel):
    ok: bool
    model_dirs: list[str]
    config_model_dirs: list[str]


# --------------------------------------------------------------------------- #
# 依赖
# --------------------------------------------------------------------------- #


def _llm_registry(reg: Registry) -> LlmRegistry:
    """取 LLM 注册表；lifespan 尚未装配时按当前配置懒建（端点不应因装配时序 5xx）。"""
    registry = reg.llm_registry
    if registry is None:
        with _registry_lock:
            if reg.llm_registry is None:
                reg.llm_registry = LlmRegistry(reg.config.llm)
            registry = reg.llm_registry
    return registry


def _engine_status(reg: Registry) -> dict:
    """引擎状态：未装配时显式呈现（而不是缺字段让前端猜）。"""
    if reg.llm_manager is None:
        return {
            "enabled": reg.config.llm.enabled,
            "mode": reg.config.llm.mode,
            "state": "disabled",
            "endpoint": "",
            "adopted": False,
            "error": None,
            "advice": None,
        }
    return reg.llm_manager.status()


def _entry_to_info(entry: LlmModelEntry) -> LlmModelInfo:
    return LlmModelInfo(**entry.to_dict())


def _vram_warning(entry: LlmModelEntry) -> str | None:
    """显存判定不乐观时给出提醒（不阻断选择——用户明确要求"跑不动的也要显示"）。

    提醒只是把 ``estimate_vram`` 的结论如实转述，不代替用户做决定。
    """
    vram = entry.vram or {}
    verdict = vram.get("verdict")
    if verdict in {"infeasible", "partial_offload", "tight"}:
        advice = str(vram.get("advice") or "")
        return f"{entry.display_name or entry.name}：{advice}" if advice else verdict
    return None


def _audit(
    reg: Registry,
    *,
    actor: str,
    action: str,
    object_id: str,
    after: dict,
    note: str,
    before: dict | None = None,
) -> None:
    """写主审计链；审计失败不掩盖主流程（与 models.py 同类处理）。

    ``before`` 默认 None；"取消选中/移除目录"这类**可回退**动作应带上原值，
    否则审计链里只剩"改成什么"，事后无法判断改之前是什么。
    """
    try:
        reg.repository.append_audit(
            actor=actor,
            action=action,
            object_type="llm_model",
            object_id=object_id,
            before=before,
            after=after,
            note=note,
        )
    except Exception as exc:  # noqa: BLE001 - 审计不可用不阻断设置变更
        _LOG.warning("LLM 审计写入失败 action=%s: %s", action, exc)


# --------------------------------------------------------------------------- #
# 查询
# --------------------------------------------------------------------------- #


@router.get("/llm/status", response_model=LlmStatusResponse)
def llm_status(reg: Annotated[Registry, Depends(get_registry)]) -> LlmStatusResponse:
    """引擎 + 选中 + 扫描 + GPU 汇总（**不探测端点**，保证响应快）。"""
    registry = _llm_registry(reg)
    gpu = registry.gpu_info()
    models = registry.list_models(include_services=False)
    return LlmStatusResponse(
        engine=_engine_status(reg),
        selection=registry.selection(),
        scan=registry.scan_status(),
        gpu=gpu.to_dict() if gpu is not None else None,
        model_dirs=registry.all_model_dirs(),
        counts={
            "local": sum(1 for m in models if m.source == SOURCE_LOCAL),
            "service": 0,
            "available": sum(1 for m in models if m.available),
            "total": len(models),
        },
    )


@router.get("/llm/models", response_model=LlmModelsResponse)
def llm_models(
    reg: Annotated[Registry, Depends(get_registry)],
    include_services: bool = True,
) -> LlmModelsResponse:
    """全部本机可用模型（本地 GGUF + 已有服务）。``include_services=false`` 时零网络调用。"""
    registry = _llm_registry(reg)
    probes = registry.probe_all() if include_services else []
    entries = registry.list_models(
        probes=probes if include_services else [],
        include_services=include_services,
    )
    gpu = registry.gpu_info()
    local = sum(1 for e in entries if e.source == SOURCE_LOCAL)
    service = sum(1 for e in entries if e.source == SOURCE_SERVICE)
    return LlmModelsResponse(
        active_id=registry.active_id,
        selection=registry.selection(),
        gpu=gpu.to_dict() if gpu is not None else None,
        scan=registry.scan_status(),
        counts={
            "total": len(entries),
            "local": local,
            "service": service,
            "available": sum(1 for e in entries if e.available),
            "infeasible": sum(1 for e in entries if (e.vram or {}).get("verdict") == "infeasible"),
        },
        services=[p.to_dict() for p in probes],
        models=[_entry_to_info(e) for e in entries],
    )


@router.get("/llm/services", response_model=ServicesResponse)
def llm_services(reg: Annotated[Registry, Depends(get_registry)]) -> ServicesResponse:
    """探测预设端点（并发）；不可达项照列，便于排查"为什么连不上"。"""
    registry = _llm_registry(reg)
    return ServicesResponse(services=[p.to_dict() for p in registry.probe_all()])


# --------------------------------------------------------------------------- #
# 扫描
# --------------------------------------------------------------------------- #


@router.post("/llm/scan", response_model=ScanResponse)
async def llm_scan_start(
    reg: Annotated[Registry, Depends(get_registry)],
    operator: Annotated[str, Depends(get_operator_name)],
    principal: Annotated[object, Depends(require_role("sysadmin"))] = None,
) -> ScanResponse:
    """触发后台全盘扫描（sysadmin）。已在扫描中返回 ``started=false``（不重复起线程）。"""
    del principal  # 仅作角色门控依赖
    registry = _llm_registry(reg)
    started = registry.start_scan()
    status = registry.scan_status()
    _audit(
        reg,
        actor=operator,
        action="llm_scan_start",
        object_id="scan",
        after={"started": started, "roots": status.get("roots", [])},
        note="触发本地模型扫描（只读）",
    )
    return ScanResponse(ok=True, started=started, status=status)


@router.post("/llm/scan/cancel", response_model=ScanResponse)
async def llm_scan_cancel(
    reg: Annotated[Registry, Depends(get_registry)],
    operator: Annotated[str, Depends(get_operator_name)],
    principal: Annotated[object, Depends(require_role("sysadmin"))] = None,
) -> ScanResponse:
    """取消正在进行的扫描；已扫到的部分结果保留（取消不等于丢数据）。"""
    del principal
    registry = _llm_registry(reg)
    cancelled = await run_in_threadpool(registry.cancel_scan)
    status = registry.scan_status()
    _audit(
        reg,
        actor=operator,
        action="llm_scan_cancel",
        object_id="scan",
        after={"cancelled": cancelled},
        note="取消本地模型扫描（保留已扫到部分结果）",
    )
    return ScanResponse(ok=True, started=cancelled, status=status)


# --------------------------------------------------------------------------- #
# 选中 / 目录
# --------------------------------------------------------------------------- #


@router.post("/llm/models/{model_id}/select", response_model=SelectResponse)
async def llm_select_model(
    model_id: str,
    reg: Annotated[Registry, Depends(get_registry)],
    operator: Annotated[str, Depends(get_operator_name)],
    principal: Annotated[object, Depends(require_role("sysadmin"))] = None,
) -> SelectResponse:
    """选中模型并把生效配置热应用到引擎（sysadmin）。

    - 模型不存在 → 404；
    - 条目不可用（头部解析失败 / 端点要求鉴权）→ 409，不静默接受；
    - 判定显存不乐观时**仍允许选中**，但在 ``warning`` 里如实提示。
    """
    del principal  # 仅作角色门控依赖
    registry = _llm_registry(reg)
    try:
        entry = await run_in_threadpool(registry.select, model_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail={"code": "LLM_MODEL_NOT_FOUND", "message": f"未找到模型: {model_id}"},
        ) from None
    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "LLM_MODEL_UNAVAILABLE", "message": str(exc)},
        ) from None

    applied = await run_in_threadpool(reg.apply_llm_selection)
    _audit(
        reg,
        actor=operator,
        action="llm_model_select",
        object_id=entry.id,
        after={
            "name": entry.display_name or entry.name,
            "source": entry.source,
            "path": entry.path,
            "endpoint": entry.endpoint,
            "applied": applied,
        },
        note="切换本地大模型（热应用生效配置）",
    )
    return SelectResponse(
        ok=True,
        active=entry.id,
        mode=str(applied.get("mode") or ""),
        endpoint=str(applied.get("endpoint") or ""),
        model_path=entry.path if entry.source == SOURCE_LOCAL else "",
        reloaded=bool(applied.get("reloaded")),
        engine=_engine_status(reg),
        warning=_vram_warning(entry),
    )


@router.post("/llm/dirs", response_model=DirsResponse)
async def llm_add_dir(
    body: DirRequest,
    reg: Annotated[Registry, Depends(get_registry)],
    operator: Annotated[str, Depends(get_operator_name)],
    principal: Annotated[object, Depends(require_role("sysadmin"))] = None,
) -> DirsResponse:
    """添加模型目录（sysadmin）。路径须真实存在，非目录 → 409。"""
    del principal
    registry = _llm_registry(reg)
    try:
        resolved = await run_in_threadpool(registry.add_dir, body.path)
    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "LLM_DIR_INVALID", "message": str(exc)},
        ) from None
    _audit(
        reg,
        actor=operator,
        action="llm_dir_add",
        object_id=resolved,
        after={"path": resolved},
        note="添加本地模型目录",
    )
    return DirsResponse(
        ok=True,
        model_dirs=registry.extra_model_dirs(),
        config_model_dirs=registry.config_model_dirs(),
    )


@router.delete("/llm/dirs", response_model=DirsResponse)
async def llm_remove_dir(
    reg: Annotated[Registry, Depends(get_registry)],
    operator: Annotated[str, Depends(get_operator_name)],
    path: str = "",
    principal: Annotated[object, Depends(require_role("sysadmin"))] = None,
) -> DirsResponse:
    """移除用户添加的模型目录（sysadmin）。配置文件里的内置目录不可删。

    只删"注册表里的记录"，**不动磁盘上的任何文件**。返回 ``ok=false`` 表示该目录
    本就不在列表中（幂等友好，不报 404）。
    """
    del principal
    if not (path or "").strip():
        raise HTTPException(
            status_code=422,
            detail={"code": "LLM_DIR_INVALID", "message": "缺少 path 查询参数"},
        )
    registry = _llm_registry(reg)
    dirs_before = registry.extra_model_dirs()
    removed = await run_in_threadpool(registry.remove_dir, path)
    _audit(
        reg,
        actor=operator,
        action="llm_dir_remove",
        object_id=path,
        before={"model_dirs": dirs_before},
        after={"path": path, "removed": removed},
        note="移除本地模型目录（仅移除注册记录，不删除磁盘文件）",
    )
    return DirsResponse(
        ok=removed,
        model_dirs=registry.extra_model_dirs(),
        config_model_dirs=registry.config_model_dirs(),
    )


@router.delete("/llm/selection", response_model=SelectResponse)
async def llm_clear_selection(
    reg: Annotated[Registry, Depends(get_registry)],
    operator: Annotated[str, Depends(get_operator_name)],
    principal: Annotated[object, Depends(require_role("sysadmin"))] = None,
) -> SelectResponse:
    """放弃显式选中，回落到配置文件默认（sysadmin）。

    没有这个入口，选中就是**单向操作**——用户一旦选过模型，就再也回不到
    ``default.yaml`` 里那套默认值，只能手改 ``data/llm_registry.json``。
    生效配置热应用，与 ``select`` 同一路径。
    """
    del principal
    registry = _llm_registry(reg)
    before = registry.selection()
    await run_in_threadpool(registry.clear_selection)
    applied = await run_in_threadpool(reg.apply_llm_selection)
    _audit(
        reg,
        actor=operator,
        action="llm_selection_clear",
        object_id=str(before.get("active_id") or ""),
        before=before,
        after=registry.selection(),
        note="放弃显式选中，回落到配置文件默认",
    )
    return SelectResponse(
        ok=True,
        active=None,
        mode=str(applied.get("mode") or ""),
        endpoint=str(applied.get("endpoint") or ""),
        model_path="",
        reloaded=bool(applied.get("reloaded")),
        engine=_engine_status(reg),
        warning="已回落到配置文件默认（default.yaml 的 llm 段）。",
    )
