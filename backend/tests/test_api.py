"""API 集成测试（§T5 / §T7）：health 端点必须 200 并返回模型状态。"""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from backend.app.main import app


def test_health_ok() -> None:
    with TestClient(app) as client:
        # registry 已改为后台线程装配（启动提速）：health 先返回 starting，
        # 轮询至装配完成后必须为 ok。
        deadline = time.monotonic() + 30.0
        body: dict = {}
        while time.monotonic() < deadline:
            resp = client.get("/api/v1/health")
            assert resp.status_code == 200
            body = resp.json()
            if body.get("status") == "ok":
                break
            time.sleep(0.1)
        else:
            raise AssertionError(f"registry 未在超时内就绪: {body}")
    assert body["status"] == "ok"
    assert "active_version" in body
    # §7.6 端边云 v1：健康检查暴露本地同步适配器状态（M6）
    assert body["sync"]["adapter"] == "local"
    assert "pending" in body["sync"]


def test_batch_requires_images() -> None:
    """§12.1 批量队列已实现（M6 去 501 桩）：空提交须显式 422，而非 501。"""
    with TestClient(app) as client:
        resp = client.post("/api/v1/batch")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "VALIDATION_ERROR"
