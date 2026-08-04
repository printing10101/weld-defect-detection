"""报告生成（§7.2）。"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.app.routers._common import not_implemented

router = APIRouter(tags=["report"])


@router.post("/report")
def report() -> JSONResponse:
    return not_implemented("M6: PDF/A 报告")
