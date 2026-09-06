"""安全响应头 + 基础限流 + IPC 令牌校验中间件。

- 安全头：CSP / X-Frame-Options / X-Content-Type-Options / Referrer-Policy /
  Permissions-Policy（桌面本地 WebView 场景，收紧外部访问面）。
- 限流：每客户端 IP 滑动窗口计数，防单来源打爆 API（本地桌面低风险，
  但设计文档  要求防护；阈值宽松，不干扰正常使用）。
- IPC 令牌（C-17）：业务请求须携带启动期一次性令牌（X-IPC-Token 头）或
  已带会话凭据——防其他本机进程误调/网页 CSRF 式调用。诚实边界：本机回环
  明文传输，令牌不解决传输加密（需 TLS 后续挂本机证书）。
"""

from __future__ import annotations

import hmac
import json
import threading
import time
from collections import defaultdict
from typing import ClassVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """为所有响应附加安全头。"""

    _HEADERS: ClassVar[dict[str, str]] = {
        "Content-Security-Policy": "default-src 'self'; img-src 'self' data: blob:; "
        "style-src 'self' 'unsafe-inline'; script-src 'self'",
        "X-Frame-Options": "DENY",
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "no-referrer",
        "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    }

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        for key, value in self._HEADERS.items():
            response.headers.setdefault(key, value)
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """每客户端 IP 滑动窗口限流。

    默认 240 req/min 每 IP：桌面单机下壳启动探测（500ms 间隔）+ 前端启动
    轮询 + 批量评片状态轮询 + 用户操作全部从 127.0.0.1 出去共享同一 IP 桶，
    旧的 60/min 实测误伤（/health 429 → 前端误判"后端离线"）；240 仍远超
    正常用量上限语义（仅拦截异常打爆）。可用环境变量 SCAN_RATE_LIMIT 覆盖：
    0 = 禁用限流（测试环境用，避免 TestClient 共享计数误伤）。
    /health、/metrics 探针豁免（见 _EXEMPT_SUFFIXES）——liveness 被限流会把
    "活着"误报成故障。内存态计数，进程重启清零——对桌面单机足够。
    """

    _EXEMPT_PATHS = ("/api/v1/health", "/api/v1/metrics", "/health", "/metrics")

    def __init__(self, app, *, limit: int | None = None, window_s: float = 60.0) -> None:
        super().__init__(app)
        import os

        if limit is None:
            raw = os.environ.get("SCAN_RATE_LIMIT", "")
            limit = int(raw) if raw.isdigit() else 240
        self._limit = max(0, limit)
        self._window = window_s
        self._hits: dict[str, list[float]] = defaultdict(list)
        # 实例级锁：原模块级单锁会让多个 app 实例（测试）互相串行
        self._hits_lock = threading.Lock()

    async def dispatch(self, request: Request, call_next) -> Response:
        if self._limit <= 0:  # 禁用限流
            return await call_next(request)
        path = request.url.path
        if path in self._EXEMPT_PATHS:
            return await call_next(request)
        client = request.client.host if request.client else "unknown"
        now = time.monotonic()
        with self._hits_lock:
            bucket = self._hits[client]
            cutoff = now - self._window
            self._hits[client] = [t for t in bucket if t > cutoff]
            if len(self._hits[client]) >= self._limit:
                body = {
                    "error": {"code": "RATE_LIMITED", "message": "请求过于频繁", "detail": None}
                }
                return Response(
                    content=json.dumps(body, ensure_ascii=False),
                    status_code=429,
                    media_type="application/json",
                )
            self._hits[client].append(now)
        return await call_next(request)


class IpcTokenMiddleware(BaseHTTPMiddleware):
    """IPC 一次性令牌校验（C-17）。

    enforce=true 时：除豁免路径外，请求须满足其一——
      1. ``X-IPC-Token`` 头 = 启动期一次性令牌（Tauri 注入 WebView 后前端统一携带）；
      2. 已携带会话凭据（Authorization: Bearer ... 或 ?access_token=，登录引导
         与直链下载场景）——凭据有效性由下游 get_principal 校验，本中间件
         只判"有无"，不重复验会话。

    豁免：/health、/metrics（存活/可观测探针）、/auth/*（登录引导需先于
    令牌分发）、静态资源与非 /api 路径（SPA 托管）。

    令牌经 ensure_token 懒签发（与 lifespan 共用同一进程内令牌槽），
    保证中间件与落盘文件始终一致。
    """

    def __init__(self, app, *, enforce: bool = True, data_dir: str = "data") -> None:
        super().__init__(app)
        self._enforce = enforce
        self._data_dir = data_dir

    async def dispatch(self, request: Request, call_next) -> Response:
        if not self._enforce:
            return await call_next(request)
        path = request.url.path
        # 豁免：非 API（根/SPA 静态资源）+ 存活/指标/认证端点
        if not path.startswith("/api/"):
            return await call_next(request)
        if path in ("/api/v1/health", "/api/v1/metrics") or path.startswith("/api/v1/auth"):
            return await call_next(request)
        from backend.infra.ipc_token import ensure_token

        token = ensure_token(self._data_dir)
        supplied = request.headers.get("X-IPC-Token", "")
        try:
            token_ok = hmac.compare_digest(supplied.encode(), token.encode())
        except UnicodeEncodeError:  # 非法头值按不匹配处理
            token_ok = False
        if token_ok:
            return await call_next(request)
        # 会话凭据在场则放行给下游真实鉴权（只判有无，不重复验）
        auth_header = request.headers.get("Authorization", "")
        if auth_header.lower().startswith("bearer ") or request.query_params.get("access_token"):
            return await call_next(request)
        from fastapi.responses import JSONResponse

        return JSONResponse(
            {
                "error": {
                    "code": "IPC_TOKEN_REQUIRED",
                    "message": "缺少 IPC 一次性令牌（本机进程间调用须携带 X-IPC-Token 或有效会话）",
                    "detail": None,
                }
            },
            status_code=401,
        )
