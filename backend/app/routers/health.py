"""健康检查（§14 GET /api/v1/health）。"""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from backend.app.dependencies import Registry, get_registry

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check(reg: Annotated[Registry, Depends(get_registry)]) -> dict:
    return reg.health
