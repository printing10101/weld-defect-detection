"""P2-9 测试：安全响应头 + 基础限流。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.security import RateLimitMiddleware


def test_security_headers_present() -> None:
    """所有响应附加安全头（防 XSS/点击劫持/嗅探/泄露 referrer）。"""
    with TestClient(app) as client:
        resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.headers["content-security-policy"].startswith("default-src 'self'")
    assert resp.headers["x-frame-options"] == "DENY"
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["referrer-policy"] == "no-referrer"
    assert "camera=()" in resp.headers["permissions-policy"]


def test_rate_limit_429() -> None:
    """超过限流阈值 → 429 RATE_LIMITED。"""

    # 构造一个 limit=3 的独立应用，避免污染全局 app 的限流计数
    from fastapi import FastAPI

    test_app = FastAPI()

    @test_app.get("/ping")
    def _ping() -> dict:
        return {"ok": True}

    test_app.add_middleware(RateLimitMiddleware, limit=3, window_s=60.0)
    with TestClient(test_app) as client:
        for _ in range(3):
            r = client.get("/ping")
            assert r.status_code == 200, r.text
        r4 = client.get("/ping")
        assert r4.status_code == 429
        assert r4.json()["error"]["code"] == "RATE_LIMITED"


def test_rate_limit_window_resets() -> None:
    """滑动窗口过期后计数清零（限流不永久阻塞）。"""
    from fastapi import FastAPI

    test_app = FastAPI()

    @test_app.get("/ping")
    def _ping() -> dict:
        return {"ok": True}

    test_app.add_middleware(RateLimitMiddleware, limit=2, window_s=0.05)
    with TestClient(test_app) as client:
        assert client.get("/ping").status_code == 200
        assert client.get("/ping").status_code == 200
        assert client.get("/ping").status_code == 429
        import time

        time.sleep(0.08)
        assert client.get("/ping").status_code == 200  # 窗口过期恢复
