"""审计哈希链防篡改测试。

覆盖：
- 洁净链 verify_chain 返回 True；
- 中间条目被篡改（改 after）→ 检测到 False；
- 末尾条目被篡改 → 因 verify_chain 重算每条哈希，同样检测到 False
  （旧实现仅校验 prev_hash 连续性，末尾篡改不可见，已修复）；
- 国密化（C-02）：新记录哈希为 SM3；
- 混合链兼容：历史 SHA-256 段 + 新增 SM3 段可整链校验，两段内篡改均可检出。
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.infra.crypto import sm3_hex
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


def _audit_payload(r: AuditRecord) -> str:
    """与 repository.verify_chain 同构的载荷序列化（测试自证一致性）。"""
    return json.dumps(
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


def _insert_legacy_sha256(repo: InspectionRepository, actor: str, after: dict) -> None:
    """按国密化前的旧格式（SHA-256）手工写入一条审计记录，模拟存量数据。"""
    with Session(repo._engine) as session, session.begin():
        last = session.scalars(
            select(AuditRecord).order_by(AuditRecord.seq.desc()).limit(1)
        ).first()
        prev_hash = last.hash if last is not None else "0" * 64
        # 时间戳取整秒并与载荷共用同一值（verify_chain 按 isoformat 秒级重算）
        now = datetime.now(UTC).replace(tzinfo=None, microsecond=0)
        payload = json.dumps(
            {
                "actor": actor,
                "action": "inspect",
                "object_type": "image",
                "object_id": f"img-{actor}",
                "before": None,
                "after": after,
                "note": None,
                "created_at": now.isoformat(timespec="seconds"),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        rec = AuditRecord(
            actor=actor,
            action="inspect",
            object_type="image",
            object_id=f"img-{actor}",
            before=None,
            after=after,
            note=None,
            created_at=now,
            prev_hash=prev_hash,
            hash=hashlib.sha256(f"{prev_hash}|{payload}".encode()).hexdigest(),
        )
        session.add(rec)


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


# ---------------------------------------------------------------------------
# 国密化（C-02）：SM3 新记录 + 新旧混合链
# ---------------------------------------------------------------------------


def test_audit_new_records_use_sm3() -> None:
    """新写入的审计记录哈希为 SM3（与逐条重算一致）。"""
    repo = InspectionRepository(_tmp_db())
    repo.append_audit(
        actor="u0",
        action="inspect",
        object_type="image",
        object_id="img-0",
        before=None,
        after={"lvl": 1},
    )
    with Session(repo._engine) as session:
        r = session.scalars(select(AuditRecord).order_by(AuditRecord.seq.desc()).limit(1)).first()
        assert r is not None
        expected = sm3_hex(f"{'0' * 64}|{_audit_payload(r)}".encode())
        assert r.hash == expected, "新记录哈希应为 SM3"
        assert r.prev_hash == "0" * 64


def test_audit_chain_mixed_sha256_and_sm3_is_valid() -> None:
    """历史 SHA-256 段 + 新增 SM3 段：混合链整链校验通过。"""
    repo = InspectionRepository(_tmp_db())
    _insert_legacy_sha256(repo, "legacy-1", {"lvl": 0})
    _insert_legacy_sha256(repo, "legacy-2", {"lvl": 1})
    repo.append_audit(  # 新增段（SM3）
        actor="u-new",
        action="inspect",
        object_type="image",
        object_id="img-new",
        before=None,
        after={"lvl": 2},
    )
    repo.append_audit(
        actor="u-new2",
        action="inspect",
        object_type="image",
        object_id="img-new2",
        before=None,
        after={"lvl": 3},
    )
    assert repo.verify_chain() is True


def test_audit_chain_mixed_legacy_segment_tamper_detected() -> None:
    """混合链：历史 SHA-256 段内篡改仍被检出。"""
    repo = InspectionRepository(_tmp_db())
    _insert_legacy_sha256(repo, "legacy-1", {"lvl": 0})
    repo.append_audit(
        actor="u-new",
        action="inspect",
        object_type="image",
        object_id="img-new",
        before=None,
        after={"lvl": 2},
    )
    _tamper(repo, seq=1)  # 历史段条目
    assert repo.verify_chain() is False


def test_audit_chain_mixed_sm3_segment_tamper_detected() -> None:
    """混合链：新增 SM3 段内篡改仍被检出。"""
    repo = InspectionRepository(_tmp_db())
    _insert_legacy_sha256(repo, "legacy-1", {"lvl": 0})
    repo.append_audit(
        actor="u-new",
        action="inspect",
        object_type="image",
        object_id="img-new",
        before=None,
        after={"lvl": 2},
    )
    _tamper(repo, seq=2)  # SM3 段条目
    assert repo.verify_chain() is False


def test_audit_chain_pure_legacy_sha256_is_valid() -> None:
    """未升级的存量库（纯 SHA-256 链）校验仍通过（向后兼容）。"""
    repo = InspectionRepository(_tmp_db())
    _insert_legacy_sha256(repo, "legacy-1", {"lvl": 0})
    _insert_legacy_sha256(repo, "legacy-2", {"lvl": 1})
    assert repo.verify_chain() is True
