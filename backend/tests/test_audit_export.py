"""审计归档导出测试（C-20）+ 安全审计链防篡改补测。

覆盖：
- GET /audit/export（审计员）：JSONL 结构（header/记录行/footer）、每行含
  记录与链校验状态、双链全量导出；
- 导出动作本身入主审计链 + 安全审计链（审计员自身操作留痕）；
- 角色管控：非审计员 403；
- C-20(b) 链校验坏链检测：主链篡改已有 test_audit_chain 覆盖，这里补
  安全链（security_audit）中间/末尾篡改均被 verify_security_chain 检出。
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager

from fastapi import Request
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app import auth as auth_mod
from backend.app.main import app
from backend.infra.db import SecurityAuditRecord
from backend.infra.security_store import SecurityStore


@contextmanager
def principal_role(role: str, username: str = "审计员") -> Iterator[None]:
    """以指定角色的 principal 覆盖鉴权依赖（退出恢复 conftest 注入）。"""
    saved = app.dependency_overrides.get(auth_mod.get_principal)

    def fake(request: Request):
        p = auth_mod.Principal("account-" + role, username, role)
        request.state.principal = p
        return p

    app.dependency_overrides[auth_mod.get_principal] = fake
    try:
        yield
    finally:
        app.dependency_overrides.pop(auth_mod.get_principal, None)
        if saved is not None:
            app.dependency_overrides[auth_mod.get_principal] = saved


def _parse_jsonl(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_audit_export_jsonl_structure_and_audit_trail():
    """归档导出：header/记录行/footer 结构齐全，导出动作入双链审计。"""
    from backend.app.dependencies import get_registry as _gr

    reg = _gr()
    # 保证两条链上至少各有一条记录（单文件运行时全新测试库，链可能为空）
    reg.repository.append_audit(
        actor="审计员",
        action="inspect",
        object_type="image",
        object_id="audit-export-fixture",
        before=None,
        after={"fixture": True},
    )
    reg.security_store.append_security_audit(
        actor="审计员",
        action="account_create",
        object_type="account",
        object_id="audit-export-fixture",
        before=None,
        after={"fixture": True},
    )
    before_main, _ = reg.repository.list_audit(action="audit_export", limit=1)
    n_before = len(
        [e for e in reg.security_store.list_security_audit(action="audit_export", limit=500)[0]]
    )
    with principal_role("auditor"), TestClient(app) as c:
        resp = c.get("/api/v1/audit/export")
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("application/x-ndjson")

    lines = _parse_jsonl(resp.text)
    assert lines[0]["type"] == "export_header"
    assert lines[0]["format"] == "scandetection-audit-export/1"
    footer = lines[-1]
    assert footer["type"] == "export_footer"

    records = [ln for ln in lines if ln["type"] == "record"]
    # 记录行数 == 主链 + 安全链总数；每行含记录与整链校验状态
    assert len(records) == footer["main_chain_total"] + footer["security_chain_total"]
    chains = {r["chain"] for r in records}
    assert chains == {"main", "security"}
    for r in records:
        assert set(r) == {"type", "chain", "seq", "record", "chain_valid"}
        assert r["chain_valid"] is True  # 测试环境双链均应完好
        assert r["record"]["seq"] >= 1
    # 链序（seq 升序，只追加语义）
    for chain in ("main", "security"):
        seqs = [r["seq"] for r in records if r["chain"] == chain]
        assert seqs == sorted(seqs)

    # 导出动作入主链 + 安全链（C-19 审计员自身操作留痕）
    _after, total = reg.repository.list_audit(action="audit_export", limit=10)
    assert total > len(before_main)
    entries, _ = reg.security_store.list_security_audit(action="audit_export", limit=500)
    assert len(entries) > n_before


def test_audit_export_auditor_only():
    """角色管控：审计员专属（sysadmin 403；403 同时落 unauthorized_access 告警）。"""
    reg = _get_reg()
    before = reg.security_store.count_alerts(kind="unauthorized_access")
    with principal_role("sysadmin", username="系统管理员"), TestClient(app) as c:
        resp = c.get("/api/v1/audit/export")
    assert resp.status_code == 403
    assert reg.security_store.count_alerts(kind="unauthorized_access") > before


def _get_reg():
    from backend.app.dependencies import get_registry

    return get_registry()


# ---------------------------------------------------------------------------
# C-20(b)：安全审计链坏链检测（主链篡改场景由 test_audit_chain 覆盖）
# ---------------------------------------------------------------------------


def _tmp_security_store() -> SecurityStore:
    return SecurityStore(tempfile.mktemp(suffix=".db"))


def _tamper_security_audit(store: SecurityStore, seq: int, *, trailing: bool = False) -> None:
    """直接改库内安全链记录，模拟篡改已落库日志。"""
    with Session(store._engine) as session, session.begin():
        r = session.scalars(
            select(SecurityAuditRecord).where(SecurityAuditRecord.seq == seq)
        ).first()
        assert r is not None, f"security audit seq={seq} 不存在"
        if trailing:
            r.note = "tampered-trailing"
        else:
            r.after = {"tampered": True}


def test_security_chain_clean_is_valid():
    store = _tmp_security_store()
    for i in range(3):
        store.append_security_audit(
            actor=f"a{i}",
            action="account_create",
            object_type="account",
            object_id=str(i),
            before=None,
            after={"i": i},
        )
    assert store.verify_security_chain() is True


def test_security_chain_middle_tamper_detected():
    store = _tmp_security_store()
    for i in range(3):
        store.append_security_audit(
            actor=f"a{i}",
            action="account_create",
            object_type="account",
            object_id=str(i),
            before=None,
            after={"i": i},
        )
    _tamper_security_audit(store, 2)
    assert store.verify_security_chain() is False


def test_security_chain_trailing_tamper_detected():
    """末尾篡改同样可检出（verify 逐条重算哈希，不依赖 prev_hash 连续性）。"""
    store = _tmp_security_store()
    for i in range(2):
        store.append_security_audit(
            actor=f"a{i}",
            action="account_create",
            object_type="account",
            object_id=str(i),
            before=None,
            after={"i": i},
        )
    _tamper_security_audit(store, 2, trailing=True)
    assert store.verify_security_chain() is False
