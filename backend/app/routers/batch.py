"""批量处理与任务队列（§12.1）。"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.app.routers._common import not_implemented

router = APIRouter(tags=["batch"])


@router.post("/batch")
def submit_batch() -> JSONResponse:
    return not_implemented("M6: 批量队列")


@router.get("/batch/{batch_id}")
def batch_status(batch_id: str) -> JSONResponse:
    return not_implemented("M6: 批量队列")


@router.post("/batch/{batch_id}/cancel")
def cancel_batch(batch_id: str) -> JSONResponse:
    return not_implemented("M6: 批量队列")
