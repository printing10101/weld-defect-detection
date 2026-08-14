"""历史检索与统计（§7.3，M6 真实实现）。

GET /api/v1/records?level=&class=&from=&to=&workpiece=&page=&size=
→ {items[], total, stats}；多条件过滤 + 分页 + 缺陷统计。
"""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from backend.app.auth import get_current_user
from backend.app.dependencies import Registry, get_registry

router = APIRouter(tags=["records"], dependencies=[Depends(get_current_user)])

_DATE_PATTERN = r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?)?$"


class RecordsResponse(BaseModel):
    items: list[dict]
    total: int
    stats: dict


@router.get("/records", response_model=RecordsResponse)
def records(
    reg: Annotated[Registry, Depends(get_registry)],
    # level 走 Literal：非法值在入口 422，而非到仓储层抛 ValueError → 500
    level: Annotated[Literal["I", "II", "III", "IV"] | None, Query()] = None,
    class_id: Annotated[int | None, Query(alias="class", ge=0)] = None,
    date_from: Annotated[str | None, Query(alias="from", pattern=_DATE_PATTERN)] = None,
    date_to: Annotated[str | None, Query(alias="to", pattern=_DATE_PATTERN)] = None,
    workpiece: Annotated[str | None, Query(max_length=128)] = None,
    page: Annotated[int, Query(ge=1)] = 1,
    size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> RecordsResponse:
    try:
        items, total = reg.repository.list_records(
            level=level,
            class_id=class_id,
            date_from=date_from,
            date_to=date_to,
            workpiece=workpiece,
            page=page,
            size=size,
        )
    except ValueError as exc:  # 日期越界等仓储层校验 → 422 而非 500
        raise HTTPException(
            status_code=422, detail={"code": "INVALID_QUERY", "message": str(exc)}
        ) from None
    return RecordsResponse(items=items, total=total, stats=reg.repository.stats())
