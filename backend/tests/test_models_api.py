"""：模型注册表与热切换 API 测试。

- 单元：ModelRegistry 扫描权重目录、列出条目、activate 调 loader 并重载、活跃指针持久化、未知 id 抛 KeyError。
- 集成：通过 TestClient 验证 GET /models 与 POST /models/{id}/activate（依赖覆盖指向临时权重目录）。
"""

from __future__ import annotations

import os
import shutil

import pytest
from fastapi.testclient import TestClient

from backend.app.dependencies import get_registry
from backend.app.main import app
from backend.infra.model_registry import ModelRegistry

# 备选用以拷贝的本地 onnx（任意两个不同训练产物即可，保证 id 不同）
_CANDIDATES = [
    "_pkg/ScanDetection/models/weights/best.onnx",
    "data/real_label/runs/real_synth2/weights/best.onnx",
    "data/real_label/runs/yolo11n_real_rare/train/weights/best.onnx",
    "data/real_label/runs/yolo8n_real_rare/train/weights/best.onnx",
]


def _two_distinct_onnx(tmp_path) -> str:
    found = [p for p in _CANDIDATES if os.path.exists(p)]
    if len(found) < 2:
        pytest.skip("需要两个可加载的 onnx 权重用于热切换测试")
    wd = tmp_path / "weights"
    wd.mkdir()
    shutil.copy(found[0], wd / "bestA.onnx")
    shutil.copy(found[1], wd / "bestB.onnx")
    return str(wd)


def test_registry_scan_and_activate(tmp_path) -> None:
    wd = _two_distinct_onnx(tmp_path)
    state = str(tmp_path / "state.json")
    reg = ModelRegistry(wd, state)
    entries = reg.scan()
    assert len(entries) == 2
    assert reg.active_id is None

    calls: list[str] = []
    reg.activate(entries[0].id, loader=lambda uri: calls.append(uri))
    assert calls == [entries[0].uri]  # loader 用条目 uri 重载
    assert reg.active_id == entries[0].id

    # 持久化：新实例能读回活跃指针
    reg2 = ModelRegistry(wd, state)
    assert reg2.active_id == entries[0].id

    # 未知 id → KeyError（由 API 层转 404）
    with pytest.raises(KeyError):
        reg.activate("nope::deadbeefdead", lambda u: None)


def test_models_api_list_and_activate(tmp_path) -> None:
    wd = _two_distinct_onnx(tmp_path)
    state = str(tmp_path / "state.json")

    reg = get_registry()
    original = reg.model_registry
    reg.model_registry = ModelRegistry(wd, state)
    app.dependency_overrides[get_registry] = lambda: reg
    try:
        with TestClient(app) as client:
            r = client.get("/api/v1/models")
            assert r.status_code == 200
            body = r.json()
            assert body["active_id"] is None
            ids = [m["id"] for m in body["models"]]
            assert len(ids) == 2

            target = ids[1]
            a = client.post(f"/api/v1/models/{target}/activate")
            assert a.status_code == 200
            assert a.json()["active"] == target

            r2 = client.get("/api/v1/models")
            active = [m for m in r2.json()["models"] if m["active"]]
            assert len(active) == 1 and active[0]["id"] == target

            # 未知 id → 404
            bad = client.post("/api/v1/models/doesnotexist::000000000000/activate")
            assert bad.status_code == 404
    finally:
        app.dependency_overrides.pop(get_registry, None)
        reg.model_registry = original


def test_models_api_approve_state_machine(tmp_path, monkeypatch) -> None:
    """E-14 投产审批状态机（此端点此前零测试零调用，审计 A-1 补链）：

    - 门禁开启后未评估直接 approve → 409（无 candidate 记录）；
    - evaluate 达标 → approve → 真正热切换（active）；
    - 重复 approve → 409（candidate 记录已消费，防重复切换）。
    评估以 monkeypatch 替身免真实 Golden 依赖，专注状态机接线。
    """
    wd = _two_distinct_onnx(tmp_path)
    state = str(tmp_path / "state.json")

    reg = get_registry()
    original = reg.model_registry
    reg.model_registry = ModelRegistry(wd, state)
    monkeypatch.setattr(reg.config.modelgate, "enabled", True)
    monkeypatch.setattr(reg.config.modelgate, "min_map", 0.0)
    monkeypatch.setattr(reg.config.modelgate, "min_recall", 0.0)

    import backend.evaluation.run_eval as re_mod

    def fake_eval(model_id, *a, **k):
        return {
            "model_id": model_id,
            "metrics": {"mAP50": 0.99, "recall": 0.99},
            "golden_fingerprint": "fp",
            "experiment_run_id": "r1",
        }

    monkeypatch.setattr(re_mod, "run_golden_evaluation", fake_eval)
    app.dependency_overrides[get_registry] = lambda: reg
    try:
        ids = [m.id for m in reg.model_registry.scan()]
        target = ids[1]
        with TestClient(app) as client:
            r0 = client.post(f"/api/v1/models/{target}/approve")
            assert r0.status_code == 409

            r1 = client.post(f"/api/v1/models/{target}/evaluate")
            assert r1.status_code == 200

            r2 = client.post(f"/api/v1/models/{target}/approve")
            assert r2.status_code == 200
            assert r2.json()["active"] == target

            active = [m for m in client.get("/api/v1/models").json()["models"] if m["active"]]
            assert len(active) == 1 and active[0]["id"] == target

            r3 = client.post(f"/api/v1/models/{target}/approve")
            assert r3.status_code == 409
    finally:
        app.dependency_overrides.pop(get_registry, None)
        reg.model_registry = original
