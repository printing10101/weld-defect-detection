"""IQI + 黑度校验（§4.2）。"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.app.routers._common import not_implemented

router = APIRouter(tags=["verify"])


@router.post("/verify")
def verify() -> JSONResponse:
    return not_implemented("M2: IQI + 黑度校验")
