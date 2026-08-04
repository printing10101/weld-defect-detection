"""图像预处理（§4）。"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.app.routers._common import not_implemented

router = APIRouter(tags=["preprocess"])


@router.post("/preprocess")
def preprocess() -> JSONResponse:
    return not_implemented("M2: 预处理算法")
