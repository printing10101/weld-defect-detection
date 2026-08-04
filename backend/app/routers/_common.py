"""路由公共工具（§T5）。"""
from __future__ import annotations

from fastapi.responses import JSONResponse


def not_implemented(stage: str) -> JSONResponse:
    """M1 骨架期占位：返回 501 并标注计划实现里程碑（§19.5）。"""
    return JSONResponse(
        {
            "error": {
                "code": "NOT_IMPLEMENTED",
                "message": f"planned in {stage}",
                "detail": None,
            }
        },
        status_code=501,
    )
