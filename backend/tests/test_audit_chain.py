"""审计哈希链防篡改测试。

覆盖：
- 洁净链 verify_chain 返回 True；
- 中间条目被篡改（改 after）→ 检测到 False；
- 末尾条目被篡改 → 因 verify_chain 重算每条哈希，同样检测到 False
  （旧实现仅校验 prev_hash 连续性，末尾篡改不可见，已修复）。
"""

from __future__ import annotations

import tempfile

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.infra.db import AuditRecord
from backend.infra.repository import InspectionRepository


def _tmp_db() -> str:
    return tempfile.mktemp(suffix=".db")


def _append_three(repo: InspectionRepository) -> None:
    for i in range(3):
        repo.append_audit(
            actor=f"u{i}",
            action="inspect",
            object_type="image",
            object_id=f"img-{i}",
            before=None,
            after={"lvl": i},
        )


def _tamper(repo: InspectionRepository, seq: int) -> None:
    """直接改库内某条审计记录的 after 字段，模拟篡改已落库日志。"""
    with Session(repo._engine) as session, session.begin():
        r = session.scalars(select(AuditRecord).where(AuditRecord.seq == seq)).first()
        assert r is not None, f"audit seq={seq} 不存在"
        r.after = {"lvl": 999, "tampered": True}


def test_audit_chain_clean_is_valid() -> None:
    repo = InspectionRepository(_tmp_db())
    _append_three(repo)
    assert repo.verify_chain() is True


def test_audit_chain_middle_tamper_detected() -> None:
    repo = InspectionRepository(_tmp_db())
    _append_three(repo)
    _tamper(repo, seq=2)  # 中间条目
    assert repo.verify_chain() is False


def test_audit_chain_trailing_tamper_detected() -> None:
    repo = InspectionRepository(_tmp_db())
    _append_three(repo)
    _tamper(repo, seq=3)  # 末尾条目：强化后的 verify_chain 须仍能检出
    assert repo.verify_chain() is False
