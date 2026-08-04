"""标准判定（§6 / §T8 熔断）。

注意：标准数值未授权（authorized=false）时不得输出级别，
判定结果必须 need_review=true（防静默错判）。
"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.app.routers._common import not_implemented

router = APIRouter(tags=["judge"])


@router.post("/judge")
def judge() -> JSONResponse:
    return not_implemented("M5: 标准判定 + 熔断规则")
