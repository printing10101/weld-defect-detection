"""LLM 路由（``app.routers.llm``）：错误映射、审计留痕、清单与状态契约。

以「假 Registry」直接调用路由函数，不启动 FastAPI 应用：
- 路由是薄壳，真正要验的是**映射语义**——404/409/422 的边界、
  变更动作必须留审计、以及"不可行模型照样返回"这条产品口径；
- 全流程不联网：``service_presets=[]`` 使探测立即返回空。
"""

from __future__ import annotations

import asyncio
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from fastapi import HTTPException

from backend.app.dependencies import Registry
from backend.app.routers import llm as llm_router
from backend.app.routers.llm import (
    DirRequest,
    llm_add_dir,
    llm_models,
    llm_remove_dir,
    llm_scan_cancel,
    llm_scan_start,
    llm_select_model,
    llm_status,
)
from backend.infra.config import LlmCfg
from backend.infra.llm_registry import LlmModelEntry, LlmRegistry

_GIB = 1024**3
_T_UINT32, _T_STRING = 4, 8


def _enc(vtype: int, val: object) -> bytes:
    if vtype == _T_STRING:
        raw = str(val).encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw
    return struct.pack("<I", int(val))  # type: ignore[arg-type]


def _gguf_bytes() -> bytes:
    kvs = [
        ("general.architecture", _T_STRING, "qwen3"),
        ("general.name", _T_STRING, "Qwen3 4B"),
        ("general.file_type", _T_UINT32, 15),
        ("qwen3.context_length", _T_UINT32, 40960),
        ("qwen3.block_count", _T_UINT32, 36),
        ("qwen3.embedding_length", _T_UINT32, 2560),
        ("qwen3.attention.head_count", _T_UINT32, 32),
        ("qwen3.attention.head_count_kv", _T_UINT32, 8),
    ]
    buf = bytearray(b"GGUF")
    buf += struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kvs))
    for key, vtype, val in kvs:
        raw = key.encode("utf-8")
        buf += struct.pack("<Q", len(raw)) + raw + struct.pack("<I", vtype) + _enc(vtype, val)
    return bytes(buf)


class _FakeRepo:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def append_audit(self, **kw) -> int:
        self.calls.append(kw)
        return len(self.calls)


class _FakeRegistry:
    """只实现路由真正用到的成员（薄壳 + 真实 LlmRegistry）。"""

    def __init__(self, tmp_path: Path, cfg: LlmCfg) -> None:
        self.config = SimpleNamespace(llm=cfg, paths=SimpleNamespace(data_dir="data"))
        self.llm_registry = LlmRegistry(
            cfg,
            state_file=tmp_path / "state" / "llm_registry.json",
            cache_file=tmp_path / "state" / "scan_cache.json",
        )
        self.llm_manager = None
        self.repository = _FakeRepo()
        self.apply_calls = 0

    def apply_llm_selection(self) -> dict:
        self.apply_calls += 1
        return {"reloaded": True, "mode": "managed", "endpoint": ""}


def _as_reg(fake: _FakeRegistry) -> Registry:
    """测试薄壳只实现路由所需成员；边界处 cast（与 test_sync_adapter 同惯例）。"""
    return cast(Registry, fake)


def _detail(ex: HTTPException) -> dict:
    """starlette 1.6 把 detail 标注为 ``str | None``；路由按 FastAPI 惯例传 dict，测试侧收敛。"""
    return cast(dict, ex.detail)


def _setup(tmp_path: Path, *, with_model: bool = True) -> tuple[_FakeRegistry, Path | None]:
    models_dir = tmp_path / "models"
    model: Path | None = None
    dirs: list[str] = []
    if with_model:
        models_dir.mkdir(parents=True, exist_ok=True)
        model = models_dir / "Qwen3-4B-Q4_K_M.gguf"
        model.write_bytes(_gguf_bytes())
        dirs = [str(models_dir)]
    empty_root = tmp_path / "_empty"
    empty_root.mkdir(exist_ok=True)
    cfg = LlmCfg(model_dirs=dirs, scan_roots=[str(empty_root)], service_presets=[])
    return _FakeRegistry(tmp_path, cfg), model


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# 状态 / 清单
# --------------------------------------------------------------------------- #


def test_status_reports_engine_selection_and_counts(tmp_path) -> None:
    reg, _ = _setup(tmp_path)
    resp = llm_status(_as_reg(reg))
    assert resp.engine["state"] == "disabled"  # 未装配 manager → 显式降级，不是缺字段
    assert resp.selection["active_id"] is None
    assert resp.scan["state"] == "idle"
    assert resp.counts["local"] == 1
    assert resp.counts["available"] == 1
    assert resp.model_dirs  # 配置里的内置目录


def test_models_endpoint_lists_everything(tmp_path) -> None:
    """清单不做可行性过滤：infeasible 的模型也必须出现在响应里。"""
    reg, model = _setup(tmp_path)
    assert model is not None
    resp = llm_models(_as_reg(reg), include_services=False)
    assert len(resp.models) == 1
    m = resp.models[0]
    assert m.available is True
    assert m.architecture == "qwen3"
    assert m.quant == "Q4_K_M"
    assert m.vram is not None  # 显存判定随条目返回，由前端标注而非隐藏
    assert resp.counts["total"] == 1


def test_models_endpoint_returns_broken_files_too(tmp_path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "broken.gguf").write_bytes(b"definitely not gguf")
    empty = tmp_path / "_empty"
    empty.mkdir(exist_ok=True)
    reg = _FakeRegistry(
        tmp_path, LlmCfg(model_dirs=[str(models_dir)], scan_roots=[str(empty)], service_presets=[])
    )
    resp = llm_models(_as_reg(reg), include_services=False)
    assert len(resp.models) == 1
    assert resp.models[0].available is False
    assert resp.models[0].error == "not_gguf"
    assert resp.counts["available"] == 0


# --------------------------------------------------------------------------- #
# 选中：错误映射 + 审计 + 热应用
# --------------------------------------------------------------------------- #


def test_select_local_model_applies_and_audits(tmp_path) -> None:
    reg, model = _setup(tmp_path)
    model_id = llm_models(_as_reg(reg), include_services=False).models[0].id

    resp = _run(llm_select_model(model_id, _as_reg(reg), "tester"))
    assert resp.ok is True
    assert resp.mode == "managed"
    assert resp.model_path == str(model)
    assert resp.reloaded is True
    assert reg.apply_calls == 1
    actions = [c["action"] for c in reg.repository.calls]
    assert actions == ["llm_model_select"]
    assert reg.repository.calls[0]["actor"] == "tester"


def test_select_unknown_model_maps_to_404(tmp_path) -> None:
    reg, _ = _setup(tmp_path)
    with pytest.raises(HTTPException) as ei:
        _run(llm_select_model("gguf:deadbeef0000", _as_reg(reg), "tester"))
    assert ei.value.status_code == 404
    assert _detail(ei.value)["code"] == "LLM_MODEL_NOT_FOUND"
    assert reg.repository.calls == []  # 失败不写审计（无状态变更）
    assert reg.apply_calls == 0  # 也不热应用


def test_select_unavailable_model_maps_to_409(tmp_path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "broken.gguf").write_bytes(b"not gguf")
    empty = tmp_path / "_empty"
    empty.mkdir(exist_ok=True)
    reg = _FakeRegistry(
        tmp_path, LlmCfg(model_dirs=[str(models_dir)], scan_roots=[str(empty)], service_presets=[])
    )
    bad_id = llm_models(_as_reg(reg), include_services=False).models[0].id
    with pytest.raises(HTTPException) as ei:
        _run(llm_select_model(bad_id, _as_reg(reg), "tester"))
    assert ei.value.status_code == 409
    assert _detail(ei.value)["code"] == "LLM_MODEL_UNAVAILABLE"
    assert reg.apply_calls == 0


# --------------------------------------------------------------------------- #
# 扫描
# --------------------------------------------------------------------------- #


def test_scan_start_and_cancel_are_audited(tmp_path) -> None:
    reg, _ = _setup(tmp_path)
    resp = _run(llm_scan_start(_as_reg(reg), "tester"))
    assert resp.ok is True
    assert resp.status["state"] in {"scanning", "done"}
    reg.llm_registry.cancel_scan()  # 让后台线程尽快收敛，避免测试悬挂
    reg.llm_registry._ensure_discovery().join(timeout=10)

    resp2 = _run(llm_scan_cancel(_as_reg(reg), "tester"))
    assert resp2.ok is True
    actions = [c["action"] for c in reg.repository.calls]
    assert actions == ["llm_scan_start", "llm_scan_cancel"]


# --------------------------------------------------------------------------- #
# 目录
# --------------------------------------------------------------------------- #


def test_add_and_remove_dir_are_audited(tmp_path) -> None:
    reg, _ = _setup(tmp_path, with_model=False)
    target = tmp_path / "extra"
    target.mkdir()

    resp = _run(llm_add_dir(DirRequest(path=str(target)), _as_reg(reg), "tester"))
    assert resp.ok is True
    assert any(Path(d) == target for d in resp.model_dirs)

    resp2 = _run(llm_remove_dir(_as_reg(reg), "tester", path=str(target)))
    assert resp2.ok is True
    assert resp2.model_dirs == []

    actions = [c["action"] for c in reg.repository.calls]
    assert actions == ["llm_dir_add", "llm_dir_remove"]


def test_add_dir_invalid_maps_to_409(tmp_path) -> None:
    reg, _ = _setup(tmp_path, with_model=False)
    with pytest.raises(HTTPException) as ei:
        _run(llm_add_dir(DirRequest(path=str(tmp_path / "nope")), _as_reg(reg), "tester"))
    assert ei.value.status_code == 409
    assert _detail(ei.value)["code"] == "LLM_DIR_INVALID"


def test_remove_dir_requires_path(tmp_path) -> None:
    reg, _ = _setup(tmp_path, with_model=False)
    with pytest.raises(HTTPException) as ei:
        _run(llm_remove_dir(_as_reg(reg), "tester", path="  "))
    assert ei.value.status_code == 422


def test_model_dirs_include_config_and_extra(tmp_path) -> None:
    reg, _ = _setup(tmp_path, with_model=False)
    extra = tmp_path / "extra"
    extra.mkdir()
    _run(llm_add_dir(DirRequest(path=str(extra)), _as_reg(reg), "tester"))
    resp = _run(llm_add_dir(DirRequest(path=str(extra)), _as_reg(reg), "tester"))
    assert len(resp.config_model_dirs) == 0
    assert len(resp.model_dirs) == 1  # 幂等：重复添加不产生重复项


# --------------------------------------------------------------------------- #
# 显存提醒
# --------------------------------------------------------------------------- #


def test_vram_warning_only_for_unfavourable_verdicts() -> None:
    entry = LlmModelEntry(
        id="gguf:x", name="X", display_name="X", vram={"verdict": "full_gpu", "advice": "ok"}
    )
    assert llm_router._vram_warning(entry) is None
    entry.vram = {"verdict": "infeasible", "advice": "权重超显存"}
    assert "权重超显存" in (llm_router._vram_warning(entry) or "")
    entry.vram = {"verdict": "tight", "advice": "余量小"}
    assert llm_router._vram_warning(entry) is not None
    entry.vram = None  # 服务模型无显存判定
    assert llm_router._vram_warning(entry) is None
