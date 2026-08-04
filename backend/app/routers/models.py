"""模型列表与热切换（§7.4）。"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.app.routers._common import not_implemented

router = APIRouter(tags=["models"])


@router.get("/models")
def list_models() -> JSONResponse:
    return not_implemented("M4: 模型注册")


@router.post("/models/{model_id}/activate")
def activate_model(model_id: str) -> JSONResponse:
    return not_implemented("M4: 模型热切换")
