"""Grad-CAM 可解释热力图（§12.3，默认仅人工复核视图开启）。"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.app.routers._common import not_implemented

router = APIRouter(tags=["explain"])


@router.post("/explain")
def explain() -> JSONResponse:
    return not_implemented("M6: Grad-CAM")
