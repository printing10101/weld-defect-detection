"""历史检索与统计（§7.3）。"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.app.routers._common import not_implemented

router = APIRouter(tags=["records"])


@router.get("/records")
def records() -> JSONResponse:
    return not_implemented("M6: 检索/统计")
