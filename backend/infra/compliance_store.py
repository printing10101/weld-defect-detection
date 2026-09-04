"""涉密载体台账与导出审批存储（C-12/C-14）。

- CarrierStore：载体（底片/报告/备份）登记、借用、归还、销毁（双确认）台账；
- ExportStore：导出审批流（申请 → 批准/拒绝 → 一次性令牌 → 凭令核销）。

与 InspectionRepository 同库但职责分离；返回值均为 dict。载体借还/销毁动作
由调用方（路由层）入主审计链 + 安全审计链，本层只管台账本身。
"""

from __future__ import annotations

import uuid
from datetime import timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.infra.db import Base, CarrierRecord, ExportRequestRecord, create_db_engine
from backend.infra.security_store import _fmt, _now

CARRIER_KINDS = ("film", "report", "backup")
CARRIER_STATUSES = ("in_stock", "borrowed", "returned", "pending_destroy", "destroyed")


def _new_id() -> str:
    return uuid.uuid4().hex


class CarrierStore:
    """载体台账（C-12）。销毁为两段式：request_destroy（保密员）→ confirm_destroy（系统管理员）。"""

    def __init__(self, db_path: str) -> None:
        self._engine = create_db_engine(db_path)
        from backend.infra.migrate import DDL_LOCK

        with DDL_LOCK:  # 与迁移线程串行化，避免并发建表撞表
            Base.metadata.create_all(self._engine)

    def register(
        self,
        *,
        carrier_id: str,
        kind: str,
        object_id: str | None = None,
        secret_level: int = 0,
        owner: str | None = None,
    ) -> dict[str, Any]:
        """登记载体。编号全局唯一，登记即入台账（编号建议 CN-年份-序号）。"""
        if kind not in CARRIER_KINDS:
            raise ValueError(f"invalid kind: {kind!r}, expected one of {CARRIER_KINDS}")
        if not 0 <= int(secret_level) <= 3:
            raise ValueError("secret_level must be 0~3")
        try:
            with Session(self._engine) as session, session.begin():
                session.add(
                    CarrierRecord(
                        id=carrier_id,
                        kind=kind,
                        object_id=object_id,
                        secret_level=int(secret_level),
                        owner=owner,
                        borrow_history=[
                            {"action": "register", "operator": owner, "at": _fmt(_now())}
                        ],
                    )
                )
                session.flush()
        except Exception as exc:
            if "UNIQUE" in str(exc).upper():
                raise ValueError(f"载体编号已存在: {carrier_id}") from exc
            raise
        return self.get(carrier_id)  # type: ignore[return-value]

    def get(self, carrier_id: str) -> dict[str, Any] | None:
        with Session(self._engine) as session:
            rec = session.get(CarrierRecord, carrier_id)
            return self._to_dict(rec) if rec else None

    def list(self, *, status: str | None = None, kind: str | None = None) -> list[dict[str, Any]]:
        with Session(self._engine) as session:
            stmt = select(CarrierRecord).order_by(CarrierRecord.created_at.desc())
            if status:
                stmt = stmt.where(CarrierRecord.status == status)
            if kind:
                stmt = stmt.where(CarrierRecord.kind == kind)
            return [self._to_dict(r) for r in session.scalars(stmt)]

    def _transition(
        self,
        carrier_id: str,
        *,
        expect: tuple[str, ...],
        status: str,
        action: str,
        operator: str,
        note: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """状态机推进 + 借还历史追加（非法状态迁移 → ValueError）。"""
        with Session(self._engine) as session, session.begin():
            rec = session.get(CarrierRecord, carrier_id)
            if rec is None:
                raise KeyError(f"carrier not found: {carrier_id}")
            if rec.status not in expect:
                raise ValueError(
                    f"carrier status is {rec.status!r}, expected one of {list(expect)}"
                )
            rec.status = status
            history = list(rec.borrow_history or [])
            history.append(
                {
                    "action": action,
                    "operator": operator,
                    "at": _fmt(_now()),
                    "note": note,
                }
            )
            rec.borrow_history = history
            for k, v in extra.items():
                setattr(rec, k, v)
            session.flush()
            return self._to_dict(rec)

    def borrow(self, carrier_id: str, *, operator: str, note: str | None = None) -> dict[str, Any]:
        """借用（在库/已归还 → 借出）。"""
        return self._transition(
            carrier_id,
            expect=("in_stock", "returned"),
            status="borrowed",
            action="borrow",
            operator=operator,
            note=note,
        )

    def give_back(
        self, carrier_id: str, *, operator: str, note: str | None = None
    ) -> dict[str, Any]:
        """归还（借出 → 已归还）。"""
        return self._transition(
            carrier_id,
            expect=("borrowed",),
            status="returned",
            action="return",
            operator=operator,
            note=note,
        )

    def request_destroy(
        self, carrier_id: str, *, operator: str, destroy_method: str, note: str | None = None
    ) -> dict[str, Any]:
        """发起销毁（保密员）：记录销毁方式，状态 → 待销毁。"""
        return self._transition(
            carrier_id,
            expect=("in_stock", "returned"),
            status="pending_destroy",
            action="destroy_request",
            operator=operator,
            note=note,
            destroy_method=destroy_method,
            destroy_requested_by=operator,
        )

    def confirm_destroy(
        self, carrier_id: str, *, operator: str, note: str | None = None
    ) -> dict[str, Any]:
        """确认销毁（系统管理员，与发起人分属不同角色双确认）→ 已销毁。"""
        return self._transition(
            carrier_id,
            expect=("pending_destroy",),
            status="destroyed",
            action="destroy_confirm",
            operator=operator,
            note=note,
            destroy_confirmed_by=operator,
            destroyed_at=_now(),
        )

    @staticmethod
    def _to_dict(r: CarrierRecord) -> dict[str, Any]:
        return {
            "carrier_id": r.id,
            "kind": r.kind,
            "object_id": r.object_id,
            "secret_level": int(r.secret_level or 0),
            "owner": r.owner,
            "status": r.status,
            "borrow_history": list(r.borrow_history or []),
            "destroy_method": r.destroy_method,
            "destroy_note": r.destroy_note,
            "destroy_requested_by": r.destroy_requested_by,
            "destroy_confirmed_by": r.destroy_confirmed_by,
            "destroyed_at": _fmt(r.destroyed_at),
            "created_at": _fmt(r.created_at),
        }


class ExportStore:
    """导出审批（C-14）：一次性令牌在库中只存 SM3 哈希，明文仅签发时返回一次。"""

    def __init__(self, db_path: str) -> None:
        self._engine = create_db_engine(db_path)
        from backend.infra.migrate import DDL_LOCK

        with DDL_LOCK:  # 与迁移线程串行化，避免并发建表撞表
            Base.metadata.create_all(self._engine)

    def create_request(
        self, *, subject: str, requested_by: str, reason: str | None = None
    ) -> dict[str, Any]:
        if not subject.strip():
            raise ValueError("subject is required")
        rid = _new_id()
        rec = ExportRequestRecord(id=rid, subject=subject, requested_by=requested_by, reason=reason)
        with Session(self._engine) as session, session.begin():
            session.add(rec)
            session.flush()
        return self.get(rid)  # type: ignore[return-value]

    def get(self, request_id: str) -> dict[str, Any] | None:
        with Session(self._engine) as session:
            rec = session.get(ExportRequestRecord, request_id)
            return self._to_dict(rec) if rec else None

    def list(self, *, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        with Session(self._engine) as session:
            stmt = (
                select(ExportRequestRecord)
                .order_by(ExportRequestRecord.created_at.desc())
                .limit(limit)
            )
            if status:
                stmt = stmt.where(ExportRequestRecord.status == status)
            return [self._to_dict(r) for r in session.scalars(stmt)]

    def decide(self, request_id: str, *, decided_by: str, approved: bool) -> dict[str, Any]:
        """保密员批准/拒绝（仅 pending 态可决策）。"""
        status = "approved" if approved else "rejected"
        with Session(self._engine) as session, session.begin():
            rec = session.get(ExportRequestRecord, request_id)
            if rec is None:
                raise KeyError(f"export request not found: {request_id}")
            if rec.status != "pending":
                raise ValueError(f"export request status is {rec.status!r}, expected 'pending'")
            rec.status = status
            rec.decided_by = decided_by
            rec.decided_at = _now()
            session.flush()
            return self._to_dict(rec)

    def issue_token(self, request_id: str, *, token_hash: str, ttl_sec: int) -> dict[str, Any]:
        """为已批准的申请签发一次性令牌（存哈希 + 有效期）。"""
        with Session(self._engine) as session, session.begin():
            rec = session.get(ExportRequestRecord, request_id)
            if rec is None:
                raise KeyError(f"export request not found: {request_id}")
            if rec.status != "approved":
                raise ValueError(f"export request status is {rec.status!r}, expected 'approved'")
            rec.token_hash = token_hash
            rec.token_expires_at = _now() + timedelta(seconds=ttl_sec)
            session.flush()
            return self._to_dict(rec)

    def consume_token(self, token_hash: str) -> dict[str, Any] | None:
        """核销一次性令牌：命中且未过期未使用 → 标记 consumed 并返回申请行；否则 None。"""
        with Session(self._engine) as session, session.begin():
            rec = session.scalars(
                select(ExportRequestRecord).where(ExportRequestRecord.token_hash == token_hash)
            ).first()
            if rec is None or rec.used_at is not None:
                return None
            if rec.token_expires_at is None or rec.token_expires_at < _now():
                return None
            rec.used_at = _now()
            rec.status = "consumed"
            session.flush()
            return self._to_dict(rec)

    @staticmethod
    def _to_dict(r: ExportRequestRecord) -> dict[str, Any]:
        return {
            "request_id": r.id,
            "subject": r.subject,
            "reason": r.reason,
            "requested_by": r.requested_by,
            "status": r.status,
            "decided_by": r.decided_by,
            "decided_at": _fmt(r.decided_at),
            "token_expires_at": _fmt(r.token_expires_at),
            "used_at": _fmt(r.used_at),
            "created_at": _fmt(r.created_at),
        }
