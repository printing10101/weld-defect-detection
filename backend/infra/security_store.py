"""安全存储：三员账号、会话、告警与独立安全审计链（C-06/C-07/C-09/C-19）。

与 InspectionRepository 同库（SQLite）但职责分离：本仓储只管"身份与安全治理"
数据，检查业务数据仍走 repository.py。返回值均为 dict（JSON 可序列化），
禁止 ORM 对象跨层传递。

安全审计链（security_audit）：结构同主审计链（audit_log），独立 SM3 哈希链，
防分叉策略与主链一致（进程内锁串行化读-改-写）。
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.infra.crypto import sm3_hex
from backend.infra.db import (
    AccountRecord,
    AlertRecord,
    Base,
    SecurityAuditRecord,
    SessionRecord,
    create_db_engine,
)

_LOG = logging.getLogger("scandetection.security_store")

ROLES = ("sysadmin", "secadmin", "auditor")


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _fmt(dt: datetime | None) -> str | None:
    return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else None


class SecurityStore:
    """账号/会话/告警/安全审计链存储（线程安全；单进程桌面部署）。"""

    def __init__(self, db_path: str) -> None:
        self._engine = create_db_engine(db_path)
        self._chain_lock = threading.Lock()  # 安全审计链写串行化（防分叉）
        from backend.infra.migrate import DDL_LOCK

        with DDL_LOCK:  # 与迁移线程串行化，避免并发建表撞表
            Base.metadata.create_all(self._engine)

    # ---- 账号 ----

    def create_account(
        self,
        *,
        username: str,
        role: str,
        sm2_public_key: str | None = None,
        auth_mode: str = "soft",
        created_by: str | None = None,
    ) -> dict[str, Any]:
        """创建账号。username 唯一（一人一岗：一个账号一个角色）。"""
        if role not in ROLES:
            raise ValueError(f"invalid role: {role!r}, expected one of {ROLES}")
        username = (username or "").strip()
        if not username or len(username) > 64:
            raise ValueError("username 必须为 1~64 字符")
        rec = AccountRecord(
            id=uuid.uuid4().hex,
            username=username,
            role=role,
            sm2_public_key=(sm2_public_key or None),
            auth_mode=auth_mode,
            created_by=created_by,
        )
        try:
            with Session(self._engine) as session, session.begin():
                session.add(rec)
                session.flush()
                out = self._account_to_dict(rec)  # 会话关闭前捕获，避免脱管刷新
        except Exception as exc:  # UNIQUE 冲突 → 友好报错（不含 SQL 细节）
            if "UNIQUE" in str(exc).upper():
                raise ValueError(f"账号已存在: {username}") from exc
            raise
        return out

    def list_accounts(self) -> list[dict[str, Any]]:
        with Session(self._engine) as session:
            rows = list(
                session.scalars(select(AccountRecord).order_by(AccountRecord.created_at.asc()))
            )
            return [self._account_to_dict(r) for r in rows]

    def get_account(self, account_id: str) -> dict[str, Any] | None:
        with Session(self._engine) as session:
            rec = session.get(AccountRecord, account_id)
            return self._account_to_dict(rec) if rec else None

    def get_account_by_username(self, username: str) -> dict[str, Any] | None:
        with Session(self._engine) as session:
            rec = session.scalars(
                select(AccountRecord).where(AccountRecord.username == username)
            ).first()
            return self._account_to_dict(rec) if rec else None

    def count_accounts(self) -> int:
        with Session(self._engine) as session:
            return int(
                session.scalar(select(func.count()).select_from(AccountRecord)) or 0
            )

    def set_account_key(self, account_id: str, sm2_public_key: str) -> None:
        """登记/更换账号 SM2 公钥（软证书登记或 UKey 公钥导出）。"""
        with Session(self._engine) as session, session.begin():
            rec = session.get(AccountRecord, account_id)
            if rec is None:
                raise KeyError(f"account not found: {account_id}")
            rec.sm2_public_key = sm2_public_key

    def set_account_status(self, account_id: str, status: str) -> dict[str, Any]:
        """启用/停用账号（停用后已有会话由调用方一并吊销）。"""
        if status not in ("active", "disabled"):
            raise ValueError(f"invalid status: {status!r}")
        with Session(self._engine) as session, session.begin():
            rec = session.get(AccountRecord, account_id)
            if rec is None:
                raise KeyError(f"account not found: {account_id}")
            rec.status = status
            out = self._account_to_dict(rec)
        return out

    def register_failed_attempt(
        self, account_id: str, *, max_attempts: int, lock_minutes: int
    ) -> tuple[dict[str, Any], bool]:
        """记录一次挑战失败；连续失败达阈值锁定账号，返回 (账号快照, 是否本次触发锁定)。"""
        with Session(self._engine) as session, session.begin():
            rec = session.get(AccountRecord, account_id)
            if rec is None:
                raise KeyError(f"account not found: {account_id}")
            rec.failed_attempts = int(rec.failed_attempts or 0) + 1
            locked_now = False
            if rec.failed_attempts >= max_attempts and rec.status == "active":
                rec.locked_until = _now() + timedelta(minutes=lock_minutes)
                rec.failed_attempts = 0  # 锁定期从头计数
                locked_now = True
            out = self._account_to_dict(rec)
        return out, locked_now

    def reset_failed_attempts(self, account_id: str) -> None:
        with Session(self._engine) as session, session.begin():
            rec = session.get(AccountRecord, account_id)
            if rec is None:
                raise KeyError(f"account not found: {account_id}")
            rec.failed_attempts = 0

    def unlock_account(self, account_id: str) -> dict[str, Any]:
        """人工解锁（锁定到期后登录自然恢复；保密/审计要求支持提前处置）。"""
        with Session(self._engine) as session, session.begin():
            rec = session.get(AccountRecord, account_id)
            if rec is None:
                raise KeyError(f"account not found: {account_id}")
            rec.locked_until = None
            rec.failed_attempts = 0
            out = self._account_to_dict(rec)
        return out

    def account_locked(self, account: dict[str, Any]) -> bool:
        """账号是否处于锁定窗口（按快照的 locked_until 判定）。"""
        lu = account.get("locked_until")
        if not lu:
            return False
        if isinstance(lu, str):
            try:
                lu = datetime.fromisoformat(lu)
            except ValueError:
                return False
        return bool(lu and lu > _now())

    # ---- 会话 ----

    def create_session(
        self,
        *,
        token_hash: str,
        account_id: str,
        ttl_sec: int,
        max_sessions: int,
    ) -> None:
        """创建会话；超过单账号并发上限时吊销最旧会话（单点登录语义）。"""
        with Session(self._engine) as session, session.begin():
            if max_sessions >= 1:
                active = list(
                    session.scalars(
                        select(SessionRecord)
                        .where(
                            SessionRecord.account_id == account_id,
                            SessionRecord.revoked.is_(False),
                        )
                        .order_by(SessionRecord.last_seen_at.asc())
                    )
                )
                # 吊销超出上限的最旧会话（含本次新建在内保留 max_sessions 条）
                while len(active) >= max_sessions:
                    oldest = active.pop(0)
                    oldest.revoked = True
            session.add(
                SessionRecord(
                    token_hash=token_hash,
                    account_id=account_id,
                    created_at=_now(),
                    last_seen_at=_now(),
                    expires_at=_now() + timedelta(seconds=ttl_sec),
                )
            )

    def get_session(self, token_hash: str) -> dict[str, Any] | None:
        with Session(self._engine) as session:
            rec = session.get(SessionRecord, token_hash)
            if rec is None:
                return None
            return {
                "token_hash": rec.token_hash,
                "account_id": rec.account_id,
                "created_at": _fmt(rec.created_at),
                "last_seen_at": _fmt(rec.last_seen_at),
                "expires_at": _fmt(rec.expires_at),
                "revoked": bool(rec.revoked),
                "_created_dt": rec.created_at,
                "_last_seen_dt": rec.last_seen_at,
                "_expires_dt": rec.expires_at,
            }

    def touch_session(self, token_hash: str, ttl_sec: int) -> None:
        """刷新 last_seen_at（滑动过期：空闲超时从最后一次活动起算）。"""
        with Session(self._engine) as session, session.begin():
            rec = session.get(SessionRecord, token_hash)
            if rec is None or rec.revoked:
                return
            rec.last_seen_at = _now()
            rec.expires_at = _now() + timedelta(seconds=ttl_sec)

    def revoke_session(self, token_hash: str) -> None:
        with Session(self._engine) as session, session.begin():
            rec = session.get(SessionRecord, token_hash)
            if rec is not None:
                rec.revoked = True

    def revoke_account_sessions(self, account_id: str) -> int:
        """吊销某账号全部会话（停用账号/密码学凭据变更时调用）。返回吊销数。"""
        with Session(self._engine) as session, session.begin():
            rows = list(
                session.scalars(
                    select(SessionRecord).where(
                        SessionRecord.account_id == account_id,
                        SessionRecord.revoked.is_(False),
                    )
                )
            )
            for r in rows:
                r.revoked = True
            return len(rows)

    # ---- 告警 ----

    def raise_alert(
        self,
        *,
        kind: str,
        message: str,
        level: str = "warn",
        detail: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        rec = AlertRecord(id=uuid.uuid4().hex, kind=kind, level=level, message=message, detail=detail)
        with Session(self._engine) as session, session.begin():
            session.add(rec)
            session.flush()
            return self._alert_to_dict(rec)

    def list_alerts(self, *, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        with Session(self._engine) as session:
            stmt = select(AlertRecord).order_by(AlertRecord.created_at.desc()).limit(limit)
            if status:
                stmt = stmt.where(AlertRecord.status == status)
            return [self._alert_to_dict(r) for r in session.scalars(stmt)]

    def count_alerts(self, *, kind: str | None = None, status: str | None = None) -> int:
        """按 kind/status 统计告警数（kind=egress_blocked 持久计数 C-15/C-16；
        status="open" 即未读告警计数，C-22 拉取式通知用）。"""
        with Session(self._engine) as session:
            conds = []
            if kind:
                conds.append(AlertRecord.kind == kind)
            if status:
                conds.append(AlertRecord.status == status)
            return int(
                session.scalar(
                    select(func.count()).select_from(AlertRecord).where(*conds)
                )
                or 0
            )

    def resolve_alert(self, alert_id: str, *, resolved_by: str, note: str | None = None) -> dict[str, Any]:
        with Session(self._engine) as session, session.begin():
            rec = session.get(AlertRecord, alert_id)
            if rec is None:
                raise KeyError(f"alert not found: {alert_id}")
            rec.status = "resolved"
            rec.resolved_by = resolved_by
            rec.resolved_at = _now()
            rec.note = note
            out = self._alert_to_dict(rec)
        return out

    def ack_alert(self, alert_id: str, *, acked_by: str, note: str | None = None) -> dict[str, Any]:
        """确认（已读）告警（C-22）：status → acknowledged，不等于处置（resolved）。

        诚实说明：alerts 表结构未区分"确认人/处置人"两列（避免为读确认引入
        迁移），这里复用 resolved_by/resolved_at 记录确认人与时间——语义为
        "首个确认该告警的人"，后续 resolve 会覆盖为处置人。"""
        with Session(self._engine) as session, session.begin():
            rec = session.get(AlertRecord, alert_id)
            if rec is None:
                raise KeyError(f"alert not found: {alert_id}")
            rec.status = "acknowledged"
            rec.resolved_by = acked_by
            rec.resolved_at = _now()
            rec.note = note if note is not None else rec.note
            out = self._alert_to_dict(rec)
        return out

    # ---- 独立安全审计链（C-19）----

    def append_security_audit(
        self,
        *,
        actor: str,
        action: str,
        object_type: str,
        object_id: str,
        before: dict[str, Any] | None,
        after: dict[str, Any] | None,
        note: str | None = None,
    ) -> dict[str, Any]:
        """追加一条安全审计（独立 SM3 哈希链，防分叉同主链）。"""
        now = _now()
        payload = json.dumps(
            {
                "actor": actor,
                "action": action,
                "object_type": object_type,
                "object_id": object_id,
                "before": before,
                "after": after,
                "note": note,
                "created_at": now.isoformat(timespec="seconds"),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        with self._chain_lock, Session(self._engine) as session, session.begin():
            last = session.scalars(
                select(SecurityAuditRecord)
                .order_by(SecurityAuditRecord.seq.desc())
                .limit(1)
            ).first()
            prev_hash = last.hash if last is not None else "0" * 64
            rec = SecurityAuditRecord(
                actor=actor,
                action=action,
                object_type=object_type,
                object_id=object_id,
                before=before,
                after=after,
                note=note,
                created_at=now,
                prev_hash=prev_hash,
                hash=sm3_hex(f"{prev_hash}|{payload}".encode()),
            )
            session.add(rec)
            session.flush()
            return self._audit_to_dict(rec)

    def list_security_audit(
        self,
        *,
        actor: str | None = None,
        action: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        limit = max(1, min(limit, 500))
        offset = max(0, offset)
        conds = []
        if actor:
            conds.append(SecurityAuditRecord.actor == actor)
        if action:
            conds.append(SecurityAuditRecord.action == action)
        with Session(self._engine) as session:
            total = int(
                session.scalar(
                    select(func.count()).select_from(SecurityAuditRecord).where(*conds)
                )
                or 0
            )
            rows = list(
                session.scalars(
                    select(SecurityAuditRecord)
                    .where(*conds)
                    .order_by(SecurityAuditRecord.seq.desc())
                    .offset(offset)
                    .limit(limit)
                )
            )
            return [self._audit_to_dict(r) for r in rows], total

    def verify_security_chain(self) -> bool:
        """校验安全审计哈希链连续性（全 SM3，同主链防分叉策略）。"""
        with Session(self._engine) as session:
            prev = "0" * 64
            stmt = (
                select(SecurityAuditRecord)
                .order_by(SecurityAuditRecord.seq.asc())
                .execution_options(yield_per=1000)
            )
            for r in session.scalars(stmt):
                payload = json.dumps(
                    {
                        "actor": r.actor,
                        "action": r.action,
                        "object_type": r.object_type,
                        "object_id": r.object_id,
                        "before": r.before,
                        "after": r.after,
                        "note": r.note,
                        "created_at": r.created_at.isoformat(timespec="seconds"),
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                )
                expected = sm3_hex(f"{prev}|{payload}".encode())
                if r.prev_hash != prev or r.hash != expected:
                    return False
                prev = r.hash
            return True

    # ---- 序列化 ----

    @staticmethod
    def _account_to_dict(r: AccountRecord) -> dict[str, Any]:
        return {
            "account_id": r.id,
            "username": r.username,
            "role": r.role,
            "sm2_public_key": r.sm2_public_key,
            "auth_mode": r.auth_mode,
            "status": r.status,
            "failed_attempts": int(r.failed_attempts or 0),
            "locked_until": _fmt(r.locked_until),
            "created_by": r.created_by,
            "created_at": _fmt(r.created_at),
        }

    @staticmethod
    def _alert_to_dict(r: AlertRecord) -> dict[str, Any]:
        return {
            "alert_id": r.id,
            "kind": r.kind,
            "level": r.level,
            "message": r.message,
            "detail": r.detail,
            "status": r.status,
            "resolved_by": r.resolved_by,
            "resolved_at": _fmt(r.resolved_at),
            "note": r.note,
            "created_at": _fmt(r.created_at),
        }

    @staticmethod
    def _audit_to_dict(r: SecurityAuditRecord) -> dict[str, Any]:
        return {
            "seq": r.seq,
            "actor": r.actor,
            "action": r.action,
            "object_type": r.object_type,
            "object_id": r.object_id,
            "before": r.before,
            "after": r.after,
            "note": r.note,
            "prev_hash": r.prev_hash,
            "hash": r.hash,
            "created_at": _fmt(r.created_at),
        }
