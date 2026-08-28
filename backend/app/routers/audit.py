"""不可变审计日志查询。

GET /api/v1/audit：按 actor/action/object_type/object_id/limit 过滤返回审计链
（谁/何时/对何对象/前后值/哈希链）。写入由业务动作（复核、报告重生成等）
经 repository.append_audit 完成，本端点只提供只读检索。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from backend.app.dependencies import Registry, get_registry

router = APIRouter(tags=["audit"])


class AuditEntry(BaseModel):
    seq: int
    actor: str
    action: str
    object_type: str
    object_id: str
    before: object | None = None
    after: object | None = None
    note: str | None = None
    prev_hash: str
    hash: str
    created_at: str | None


class AuditResponse(BaseModel):
    entries: list[AuditEntry]
    total: int
    chain_valid: bool


@router.get("/audit", response_model=AuditResponse)
def audit(
    reg: Annotated[Registry, Depends(get_registry)],
    actor: Annotated[str | None, Query()] = None,
    action: Annotated[str | None, Query()] = None,
    object_type: Annotated[str | None, Query()] = None,
    object_id: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> AuditResponse:
    # total 取匹配总数（原实现用 len(entries)，超过 limit 时低报，审计场景不可接受）
    entries, total = reg.repository.list_audit(
        actor=actor,
        action=action,
        object_type=object_type,
        object_id=object_id,
        limit=limit,
        offset=offset,
    )
    return AuditResponse(
        entries=[AuditEntry(**e) for e in entries],
        total=total,
        chain_valid=reg.repository.verify_chain(),
    )
