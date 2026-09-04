"""交付物生成器测试（V-02~V-05，要求文档 §6 表 6-1）。

覆盖：四个端点的报告结构与"活生成"证据（矩阵解析计数 / 长跑与演练聚合 /
评价产物聚合 / 边界文本）、产物 JSON+PDF/A 落盘、缺口如实列示（无记录
不表述为通过）、动作入双链审计、角色矩阵（仅审计员/保密员）、未知编号 404。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from fastapi import Request
from fastapi.testclient import TestClient

from backend.app import auth as auth_mod
from backend.app.dependencies import get_registry
from backend.app.main import app


@contextmanager
def principal_role(role: str, username: str = "测试用户") -> Iterator[None]:
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


def _client() -> TestClient:
    return TestClient(app)


def _compliance_dir() -> Path:
    reg = get_registry()
    out = Path(reg.config.paths.data_dir) / "compliance"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _files_ok(files: dict[str, str]) -> bool:
    return all(Path(v).is_file() and Path(v).stat().st_size > 0 for v in files.values())


def _pdfa_marker(path: str) -> bool:
    """PDF/A-1b 标识：XMP 元数据含 pdfaid:part / pdfaid:conformance。"""
    raw = Path(path).read_bytes()
    return raw[:5] == b"%PDF-" and b"pdfaid:part" in raw and b"pdfaid:conformance" in raw


# ---------------------------------------------------------------------------
# V-02 信创兼容性验证报告
# ---------------------------------------------------------------------------


def test_v02_matrix_aggregation_and_files():
    with principal_role("auditor"), _client() as c:
        resp = c.post("/api/v1/compliance/deliverables/V-02")
    assert resp.status_code == 200
    body = resp.json()
    assert body["spec"] == "V-02"
    assert body["report"]["matrix_found"] is True
    rows = body["report"]["rows"]
    counts = body["report"]["counts"]
    assert rows, "矩阵解析结果为空"
    assert sum(counts.values()) == len(rows)
    # 矩阵存在未验证项（诚实标注）→ warning，缺口非空
    assert body["overall"] in {"pass", "warning"}
    if body["overall"] == "warning":
        assert body["report"]["gaps"]
    assert _files_ok(body["files"])
    assert _pdfa_marker(body["files"]["pdf"])
    # 活检查：明细行维度来自矩阵小节（CPU/OS/数据库/推理/密码）
    sections = {r["section"] for r in rows}
    assert any("CPU" in s or "算力" in s for s in sections)


# ---------------------------------------------------------------------------
# V-03 稳定性测试报告（有/无实测记录两条路径）
# ---------------------------------------------------------------------------


def test_v03_without_records_lists_gaps():
    reg = get_registry()
    out_dir = Path(reg.config.paths.data_dir) / "compliance"
    # 本测试不预置任何 soak/drill 记录 → 缺口如实列出，overall=fail
    existing_soaks = sorted(out_dir.glob("soak_*.json"))
    existing_drills = sorted(out_dir.glob("recovery_drill_*.json"))
    stash = [(p, p.with_suffix(p.suffix + ".bak")) for p in existing_soaks + existing_drills]
    for src, dst in stash:
        src.rename(dst)
    try:
        with principal_role("auditor"), _client() as c:
            resp = c.post("/api/v1/compliance/deliverables/V-03")
        assert resp.status_code == 200
        body = resp.json()
        assert body["overall"] == "fail"
        assert any("长跑" in m for m in body["report"]["missing"])
        assert any("恢复演练" in m for m in body["report"]["missing"])
    finally:
        for src, dst in stash:
            dst.rename(src)


def test_v03_with_records_aggregates_and_passes():
    out_dir = _compliance_dir()
    # 预置一份 PASS 长跑 + 一份 PASS 演练（字段对齐 scripts/soak_72h.py 与 recovery_drill.py 产出）
    soak = {
        "drill": "soak",
        "rounds": 3,
        "success_rate": 1.0,
        "failures": 0,
        "rss_slope_mb_per_round": 0.01,
        "leak_suspected": False,
        "planned_hours": 0.01,
        "conclusion": "PASS",
    }
    drill = {
        "drill": "recovery_drill",
        "steps": [
            {"step": "create_backup(sm3)", "ok": True},
            {"step": "inject_corruption", "ok": True},
            {"step": "detect_corruption", "ok": True},
            {"step": "restore_and_verify", "ok": True},
        ],
        "rto_sec": 0.5,
        "total_elapsed_sec": 1.2,
        "conclusion": "PASS",
    }
    (out_dir / "soak_test_v03.json").write_text(json.dumps(soak), encoding="utf-8")
    (out_dir / "recovery_drill_test_v03.json").write_text(json.dumps(drill), encoding="utf-8")
    try:
        with principal_role("secadmin"), _client() as c:
            resp = c.post("/api/v1/compliance/deliverables/V-03")
        assert resp.status_code == 200
        body = resp.json()
        # 取最新记录聚合（文件名倒序，test_v03 排最前）
        soaks = body["report"]["soak_reports"]
        drills = body["report"]["recovery_drills"]
        assert soaks and soaks[0]["conclusion"] == "PASS"
        assert drills and drills[0]["steps_ok"] == 4 and drills[0]["steps_total"] == 4
        assert drills[0]["rto_sec"] == 0.5
        assert body["overall"] == "pass"
        assert body["report"]["missing"] == []
        assert _files_ok(body["files"])
        assert _pdfa_marker(body["files"]["pdf"])
    finally:
        (out_dir / "soak_test_v03.json").unlink(missing_ok=True)
        (out_dir / "recovery_drill_test_v03.json").unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# V-04 评价体系合规报告
# ---------------------------------------------------------------------------


def test_v04_reads_std_eval_artifact():
    reg = get_registry()
    eval_dir = Path(reg.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": "2026-08-30T00:00:00",
        "result": {
            "standard": {"tdr": 0.9, "wdr": 0.9, "kdr": 0.95, "frr": 0.05, "level": "L2"},
            "strict": {},
        },
    }
    target = eval_dir / "std_eval.json"
    backup = target.with_suffix(".json.bak") if target.exists() else None
    if backup:
        target.rename(backup)
    target.write_text(json.dumps(payload), encoding="utf-8")
    try:
        with principal_role("auditor"), _client() as c:
            resp = c.post("/api/v1/compliance/deliverables/V-04")
        assert resp.status_code == 200
        body = resp.json()
        assert body["report"]["std_eval_found"] is True
        m = body["report"]["metrics"]
        assert m["tdr"] == 0.9 and m["kdr"] == 0.95
        assert body["report"]["level"] == "L2"
        assert body["overall"] in {"pass", "warning"}
        assert _files_ok(body["files"])
        assert _pdfa_marker(body["files"]["pdf"])
    finally:
        target.unlink(missing_ok=True)
        if backup and backup.exists():
            backup.rename(target)


# ---------------------------------------------------------------------------
# V-05 边界声明
# ---------------------------------------------------------------------------


def test_v05_boundary_statement():
    with principal_role("secadmin"), _client() as c:
        resp = c.post("/api/v1/compliance/deliverables/V-05")
    assert resp.status_code == 200
    body = resp.json()
    st = body["report"]["statement"]
    assert "第三方" in st and "不构成" in st and "密评" in st
    assert body["overall"] == "pass"
    assert _files_ok(body["files"])
    assert _pdfa_marker(body["files"]["pdf"])


# ---------------------------------------------------------------------------
# 角色矩阵 / 未知编号 / 审计留痕
# ---------------------------------------------------------------------------


def test_deliverable_role_matrix_and_unknown_spec():
    with principal_role("sysadmin"), _client() as c:
        assert c.post("/api/v1/compliance/deliverables/V-05").status_code == 403
    with principal_role("auditor"), _client() as c:
        assert c.post("/api/v1/compliance/deliverables/V-99").status_code == 404


def test_deliverable_export_audited_to_both_chains():
    reg = get_registry()
    _, main_total = reg.repository.list_audit(action="deliverable_export", limit=1)
    _, sec_total = reg.security_store.list_security_audit(limit=1)
    with principal_role("auditor", "审计员甲"), _client() as c:
        assert c.post("/api/v1/compliance/deliverables/V-05").status_code == 200
    entries, new_main_total = reg.repository.list_audit(action="deliverable_export", limit=1)
    assert new_main_total > main_total
    after = entries[0]["after"] or {}
    assert after.get("spec") == "V-05"
    _, new_sec_total = reg.security_store.list_security_audit(limit=1)
    assert new_sec_total > sec_total
    sec_entries, _ = reg.security_store.list_security_audit(limit=1)
    assert sec_entries[0]["action"] == "deliverable_export"
