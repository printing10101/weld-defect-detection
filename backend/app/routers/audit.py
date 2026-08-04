"""不可变审计日志（§12.5）。"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.app.routers._common import not_implemented

router = APIRouter(tags=["audit"])


@router.get("/audit")
def audit() -> JSONResponse:
    return not_implemented("M6: 审计日志")
