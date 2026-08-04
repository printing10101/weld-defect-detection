"""API 集成测试（§T5 / §T7）：health 端点必须 200 并返回模型状态。"""
from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.main import app


def test_health_ok() -> None:
    with TestClient(app) as client:
        resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "active_version" in body


def test_unimplemented_returns_501() -> None:
    with TestClient(app) as client:
        resp = client.post("/api/v1/detect")
    assert resp.status_code == 501
    assert resp.json()["error"]["code"] == "NOT_IMPLEMENTED"
