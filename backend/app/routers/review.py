"""双人复核/仲裁工作流（§12.2）。"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.app.routers._common import not_implemented

router = APIRouter(tags=["review"])


@router.post("/review")
def review() -> JSONResponse:
    return not_implemented("M5: 双人复核/仲裁")
