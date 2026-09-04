"""P2-9 测试：安全响应头 + 基础限流。"""

from __future__ import annotations

import pytest
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


def test_rate_limit_window_resets(monkeypatch: pytest.MonkeyPatch) -> None:
    """滑动窗口过期后计数清零（限流不永久阻塞）。

    注入确定性单调时钟：旧实现靠 sleep(0.08s) 跨过 0.05s 窗口，全量套件高负载
    下第 3 个请求可能已被挤到窗口过期之后（预期 429 实得 200）造成偶发失败；
    改为手动推时钟，断言与墙钟彻底解耦。
    """
    from fastapi import FastAPI

    import backend.app.security as security_mod

    now = {"t": 1000.0}
    monkeypatch.setattr(security_mod.time, "monotonic", lambda: now["t"])

    test_app = FastAPI()

    @test_app.get("/ping")
    def _ping() -> dict:
        return {"ok": True}

    test_app.add_middleware(RateLimitMiddleware, limit=2, window_s=0.05)
    with TestClient(test_app) as client:
        assert client.get("/ping").status_code == 200
        now["t"] += 0.01
        assert client.get("/ping").status_code == 200
        now["t"] += 0.01
        assert client.get("/ping").status_code == 429
        now["t"] += 1.0  # 跨过窗口
        assert client.get("/ping").status_code == 200  # 窗口过期恢复
