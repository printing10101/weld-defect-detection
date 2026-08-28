"""安全响应头 + 基础限流中间件。

- 安全头：CSP / X-Frame-Options / X-Content-Type-Options / Referrer-Policy /
  Permissions-Policy（桌面本地 WebView 场景，收紧外部访问面）。
- 限流：每客户端 IP 滑动窗口计数，防单来源打爆 API（本地桌面低风险，
  但设计文档  要求防护；阈值宽松，不干扰正常使用）。
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import ClassVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

_HITS_LOCK = threading.Lock()


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

    默认 60 req/min 每 IP（远超桌面本地正常用量，仅拦截异常打爆）。
    可通过环境变量 SCAN_RATE_LIMIT 覆盖：0 = 禁用限流（测试环境用，
    避免 TestClient 共享计数误伤；生产保持默认或按需调高）。
    内存态计数，进程重启清零——对桌面单机足够。
    """

    def __init__(self, app, *, limit: int | None = None, window_s: float = 60.0) -> None:
        super().__init__(app)
        import os

        if limit is None:
            raw = os.environ.get("SCAN_RATE_LIMIT", "")
            limit = int(raw) if raw.isdigit() else 60
        self._limit = max(0, limit)
        self._window = window_s
        self._hits: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next) -> Response:
        if self._limit <= 0:  # 禁用限流
            return await call_next(request)
        client = request.client.host if request.client else "unknown"
        now = time.monotonic()
        with _HITS_LOCK:
            bucket = self._hits[client]
            cutoff = now - self._window
            self._hits[client] = [t for t in bucket if t > cutoff]
            if len(self._hits[client]) >= self._limit:
                return Response(
                    content='{"error":{"code":"RATE_LIMITED","message":"请求过于频繁","detail":null}}',
                    status_code=429,
                    media_type="application/json",
                )
            self._hits[client].append(now)
        return await call_next(request)
