"""未处理异常的 500 必须带 CORS 头（回归）。

Starlette 的 ServerErrorMiddleware 固定在最外层直接回 500 纯文本、不经 CORS
中间件；跨源页面（app:// 壳）会把该响应拦成 TypeError，前端据此把真实的服务
端故障误报成「无法连接本地推理服务」（images.film_no 迁移事故的实际表象）。
内层 UnhandledExceptionMiddleware 把异常转统一错误包后照常过 CORS 出栈。
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from backend.app.security import UnhandledExceptionMiddleware


def _app() -> FastAPI:
    """最小复刻 create_app 的中间件顺序：兜底异常先注册（内层），CORS 后注册。"""
    app = FastAPI()
    app.add_middleware(UnhandledExceptionMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["app://scandetection"],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/api/v1/boom")
    def boom() -> None:
        raise RuntimeError("sqlite3.OperationalError: table images has no column named film_no")

    return app


def test_unhandled_exception_returns_500_envelope_with_cors_headers() -> None:
    with TestClient(_app()) as client:
        resp = client.get("/api/v1/boom", headers={"Origin": "app://scandetection"})
    assert resp.status_code == 500
    body = resp.json()
    assert body["error"]["code"] == "INTERNAL_ERROR"
    assert body["error"]["message"] == "服务内部错误"
    # 关键断言：500 响应携带 CORS 头——否则浏览器层 TypeError，前端误报"无法连接"
    assert resp.headers.get("access-control-allow-origin") == "app://scandetection"


def test_normal_responses_unaffected() -> None:
    app = _app()

    @app.get("/api/v1/ok")
    def ok() -> dict[str, bool]:
        return {"ok": True}

    with TestClient(app) as client:
        resp = client.get("/api/v1/ok", headers={"Origin": "app://scandetection"})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
