"""缺陷检测 + 量化（§5）。"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.app.routers._common import not_implemented

router = APIRouter(tags=["detect"])


@router.post("/detect")
def detect() -> JSONResponse:
    return not_implemented("M4: 检测 + 量化")
