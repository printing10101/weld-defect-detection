"""模型列表与热切换。

- ``GET /api/v1/models``：列出权重目录中所有可用模型（id/名称/版本指纹/大小/是否活跃）。
- ``POST /api/v1/models/{id}/activate``：运行时热切换检测器权重（重载 ONNX 会话），
  成功持久化活跃指针；失败（模型不存在/加载错误）返回 404/500 且**不改变**当前检测器。
- E-14 投产门禁（modelgate.enabled=true）：activate 不再直接切换——先跑 Golden 评估，
  达标进入 candidate 状态，须 sysadmin 经 ``POST /models/{id}/approve`` 审批后才真正
  投产（审计留痕）；评估未达标 → 拒绝并告警+审计。默认 false 保持旧行为（直接切换）。
- ``POST /api/v1/models/{id}/evaluate``：对指定模型跑 Golden 评估（不切换）。

热切换互斥：检测器推理与切换之间由 ResilientDetector 内置读写锁协调——
``load``（含回退重载）持写锁、``infer``/``infer_tta`` 持读锁，切换等在途
推理排空后执行、期间新推理排队，不会打到半初始化会话上。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from backend.app.auth import require_role
from backend.app.dependencies import (
    EVAL_GATE,
    Registry,
    _resolve_path,
    get_operator_name,
    get_registry,
)
from backend.domain.detect import get_detector
from backend.domain.errors import ModelUnavailableError
from backend.evaluation.run_eval import run_golden_evaluation

router = APIRouter(tags=["models"])


class ModelInfo(BaseModel):
    id: str
    name: str
    version: str
    size_bytes: int
    active: bool
    uri: str
    metric_map: dict | None = None  # 评估指标 + Golden Set 指纹（§7.4 模型卡；无评估为 None）


class ModelsResponse(BaseModel):
    active_id: str | None
    models: list[ModelInfo]


class ActivateResponse(BaseModel):
    ok: bool
    active: str
    # E-14 门禁状态机字段（旧行为下恒为 "active"/False，兼容既有消费方）。
    status: str = "active"  # active=已投产 | candidate=待审批 | rejected=评估未过
    approval_required: bool = False  # true=需 sysadmin POST /models/{id}/approve
    evaluation: dict | None = None  # candidate/rejected 时的门禁评估结果


class ApproveResponse(BaseModel):
    ok: bool
    active: str
    approved_by: str


class EvaluateResponse(BaseModel):
    model_id: str
    metrics: dict
    golden_fingerprint: str
    drift: dict
    experiment_run_id: str | None = None


def _preprocess_factory(reg: Registry):
    """复现生产增强链路（与 run_inspection 一致），使 Golden Set mAP 可信。

    preprocess 关闭时返回 None（在原始 gray 上推理）。
    """
    if reg.config.preprocess.enabled:
        pp = reg.preprocessor
        gamma = reg.config.preprocess.gamma

        def _fn(gray):
            return pp.enhance(pp.denoise(gray), gamma)

        return _fn
    return None


@router.get("/models", response_model=ModelsResponse)
def list_models(reg: Annotated[Registry, Depends(get_registry)]) -> ModelsResponse:
    entries = reg.model_registry.scan()
    return ModelsResponse(
        active_id=reg.model_registry.active_id,
        models=[
            ModelInfo(
                id=e.id,
                name=e.name,
                version=e.version,
                size_bytes=e.size_bytes,
                active=e.active,
                uri=e.uri,
                metric_map=reg.eval_report(e.id),
            )
            for e in entries
        ],
    )


@router.post("/models/{model_id}/activate", response_model=ActivateResponse)
async def activate_model(
    model_id: str,
    reg: Annotated[Registry, Depends(get_registry)],
    operator: Annotated[str, Depends(get_operator_name)],
    principal: Annotated[object, Depends(require_role("sysadmin"))] = None,
) -> ActivateResponse:
    """热切换生产检测权重（sysadmin 专属，与 approve 同一权限位）。

    切换对象是全系统的推理权重——审计员/操作员不应能触发；三员权限矩阵
    下投产类操作（activate/approve/evaluate）统一收口 sysadmin。
    """
    del principal  # 仅作角色门控依赖（require_role("sysadmin")）
    # E-14 门禁分支：enabled=true 时 activate 语义变为"提交候选并先评估"，
    # 不直接切换（默认 false 走下方原有直切链路，保持既有测试/行为不变）。
    if reg.config.modelgate.enabled:
        return await _activate_with_gate(model_id, reg, operator)
    try:
        # registry 锁内重载检测器会话（DefectDetector.load 已预热校验）；失败抛出→保持原检测器。
        # actor 记录为请求头操作员（X-Operator-Name）。
        entry = await run_in_threadpool(reg.activate_model, model_id, operator)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail={"code": "MODEL_NOT_FOUND", "message": f"未找到模型: {model_id}"},
        )
    except (RuntimeError, OSError) as exc:  # 加载失败：保持原检测器不变
        # C-21 审计完整性：模型切换失败事件入主审计链（fail-safe 事件的追溯要素）
        reg.repository.append_audit(
            actor=operator,
            action="model_activate_failed",
            object_type="model",
            object_id=model_id,
            before=None,
            after={"error": str(exc)[:200]},
            note="模型热切换失败，原检测器保持不变（fail-safe）",
        )
        raise HTTPException(
            status_code=500,
            detail={"code": "MODEL_ACTIVATE_FAILED", "message": str(exc)},
        )
    return ActivateResponse(ok=True, active=entry.id)


async def _activate_with_gate(model_id: str, reg: Registry, operator: str) -> ActivateResponse:
    """E-14 门禁激活链：先 Golden 评估 → 达标进 candidate（待审批）/未达标拒绝。

    评估在当前请求内同步完成（投产门禁必须"先评估后允许提交"，异步会留下
    未评估窗口）；Golden Set 缺失 → 409（评估不了就不允许投产）。
    """
    if reg.model_registry.get(model_id) is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "MODEL_NOT_FOUND", "message": f"未找到模型: {model_id}"},
        )
    try:
        record = await run_in_threadpool(reg.evaluate_activation_gate, model_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail={"code": "MODEL_NOT_FOUND", "message": f"未找到模型: {model_id}"},
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "NO_GOLDEN_SET", "message": str(exc)},
        ) from exc
    except (RuntimeError, OSError, ModelUnavailableError) as exc:
        raise HTTPException(
            status_code=500,
            detail={"code": "MODEL_LOAD_FAILED", "message": str(exc)},
        ) from exc
    active_id = reg.model_registry.active_id or ""
    if not record["passed"]:
        # 评估未过：拒绝并告警 + 审计（诚实门禁：不允许"先切了再说"）。
        try:
            reg.security_store.raise_alert(
                kind="model_gate_reject",
                level="warn",
                message=f"模型 {model_id} 投产门禁评估未通过，已拒绝激活",
                detail={"reason": record["reason"], "metrics": record["metrics"]},
            )
        except Exception:  # noqa: BLE001, S110 - 告警失败不掩盖 422
            pass
        reg.repository.append_audit(
            actor=operator,
            action="model_gate_reject",
            object_type="model",
            object_id=model_id,
            before=None,
            after={"reason": record["reason"], "metrics": record["metrics"]},
            note="E-14 投产门禁：评估未达标，拒绝激活",
        )
        return ActivateResponse(
            ok=False,
            active=active_id,
            status="rejected",
            approval_required=False,
            evaluation=record,
        )
    return ActivateResponse(
        # ok=True：candidate 是一次"受理成功"（进入待审批），不是请求失败——
        # 门禁已启用时本端点不再直接切换，是否投产由 /approve 决定。
        ok=True,
        active=active_id,
        status="candidate",
        approval_required=True,
        evaluation=record,
    )


@router.post("/models/{model_id}/approve", response_model=ApproveResponse)
async def approve_model(
    model_id: str,
    reg: Annotated[Registry, Depends(get_registry)],
    operator: Annotated[str, Depends(get_operator_name)],
    principal: Annotated[object, Depends(require_role("sysadmin"))] = None,
) -> ApproveResponse:
    """E-14 投产审批（sysadmin 专属）：candidate 评估通过后执行真正热切换。

    - 门禁未启用 → 409（不存在"绕过状态机的审批"）；
    - 无通过评估记录 / 评估未达标 → 409/422；
    - 审批与切换均留审计（model_approve / model_activate）。
    """
    del principal  # 仅作角色门控依赖（require_role("sysadmin")）
    try:
        entry = await run_in_threadpool(reg.approve_model, model_id, operator)
    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "APPROVE_NOT_ALLOWED", "message": str(exc)},
        ) from exc
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail={"code": "MODEL_NOT_FOUND", "message": f"未找到模型: {model_id}"},
        )
    return ApproveResponse(ok=True, active=entry.id, approved_by=operator)


@router.post("/models/{model_id}/evaluate", response_model=EvaluateResponse)
async def evaluate_model(
    model_id: str,
    reg: Annotated[Registry, Depends(get_registry)],
    principal: Annotated[object, Depends(require_role("sysadmin"))] = None,
) -> EvaluateResponse:
    """在固定 Golden Set 上评估指定模型并闭环写报告/漂移/模型卡/实验（sysadmin 专属）。

    - 模型不存在 → 404；
    - Golden Set 未准备（data/eval/golden 缺失）→ 409，提示先建立评估集；
    - 评估在请求线程池内**同步**执行（CPU 密集、可能数十秒到分钟级）；
      进程级信号量串行化并发评估，防止多份 ONNX 会话叠加常驻内存。
    """
    del principal  # 仅作角色门控依赖（require_role("sysadmin")）
    entry = reg.model_registry.get(model_id)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "MODEL_NOT_FOUND", "message": f"未找到模型: {model_id}"},
        )

    # 评估该模型权重：经注册表加载独立检测器（不干扰当前活跃检测器）；
    # tiling 参数与生产同参，保证评估口径与线上推理一致。检测器构建与
    # 评估都在 EVAL_GATE 闸内——会话构建放闸外时，等闸的请求仍各持一份
    # 数百 MB 的 ONNX 会话，闸门形同虚设。
    def _run_gated() -> dict:
        with EVAL_GATE:
            det = get_detector(
                "trained_yolo",
                model_uri=entry.uri,
                backend=reg.config.model.backend,
                providers=reg.config.model.providers,
                tile_size=reg.config.detect.tile_size,
                tile_overlap=reg.config.detect.tile_overlap,
                tile_trigger_side=reg.config.detect.tile_trigger_side,
                tile_max_count=reg.config.detect.tile_max_count,
                tile_merge_iou=reg.config.detect.tile_merge_iou,
            )
            return run_golden_evaluation(
                model_id,
                det,
                golden_dir=_resolve_path(reg.config.eval.golden_dir),
                eval_dir=reg.eval_dir,
                experiments_dir=_resolve_path(reg.config.eval.experiments_dir),
                drift_baseline_path=_resolve_path(reg.config.eval.drift_baseline_path),
                conf=reg.config.detect.infer_conf,
                iou=reg.config.detect.infer_iou,
                class_conf=reg.config.detect.class_conf,
                preprocess_fn=_preprocess_factory(reg),
            )

    try:
        summary = await run_in_threadpool(_run_gated)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "NO_GOLDEN_SET", "message": str(exc)},
        ) from exc
    except (RuntimeError, OSError, ModelUnavailableError) as exc:
        raise HTTPException(
            status_code=500,
            detail={"code": "MODEL_LOAD_FAILED", "message": str(exc)},
        ) from exc
    return EvaluateResponse(
        model_id=summary["model_id"],
        metrics=summary["metrics"],
        golden_fingerprint=summary["golden_fingerprint"],
        drift=summary["drift"],
        experiment_run_id=summary.get("experiment_run_id"),
    )
