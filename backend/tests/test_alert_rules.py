"""异常行为告警测试（C-22）：告警规则配置化、批量导出告警、越权 403 告警、
未读告警计数与确认（ack）端点、锁定告警配置开关。

模式：与 test_export_control 同构——principal_role 覆盖鉴权依赖；
告警计数经 security_store.count_alerts 持久统计。
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager

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


def _make_report_with_pdf() -> str:
    """直接落库一个报告 + 伪造 PDF 文件（不经检测管线，聚焦下载告警）。"""
    reg = get_registry()
    image_id = uuid.uuid4().hex
    report_id = f"rep_{uuid.uuid4().hex[:12]}"
    reg.repository.create_inspection(
        image={
            "id": image_id,
            "path": f"{image_id}.png",
            "source_type": "image",
            "modality": "GENERIC",
        },
        defects=[],
        report={"id": report_id, "image_id": image_id, "pdf_path": f"{report_id}.pdf"},
    )
    from backend.infra.config import resolve_config_path

    pdf = resolve_config_path(reg.config.paths.reports_dir) / f"{report_id}.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.write_bytes(b"%PDF-1.4\n% fake alert-rule fixture\n")
    return report_id


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


def test_alerts_config_defaults_and_schema_sync():
    """alerts 节默认值可加载，且 schema.yaml 与 default.yaml 无漂移。"""
    cfg = get_registry().config
    assert cfg.alerts.batch_export.enabled is True
    assert cfg.alerts.batch_export.window_min == 10
    assert cfg.alerts.batch_export.threshold == 5
    assert cfg.alerts.unauthorized_access.enabled is True
    assert cfg.alerts.account_lockout.enabled is True


# ---------------------------------------------------------------------------
# 批量导出告警
# ---------------------------------------------------------------------------


def test_batch_export_alert_triggers_once_at_threshold():
    """窗口期内第 threshold 次下载触发一次 high 告警，之后不重复。"""
    reg = get_registry()
    actor = f"sec_{uuid.uuid4().hex[:8]}"
    report_id = _make_report_with_pdf()
    original = reg.config.alerts.batch_export
    reg.config.alerts.batch_export = type(original)(enabled=True, window_min=600, threshold=2)
    try:
        before = reg.security_store.count_alerts(kind="batch_export")
        with principal_role("secadmin", username=actor), TestClient(app) as c:
            for _ in range(3):
                resp = c.get(f"/api/v1/report/{report_id}/pdf")
                assert resp.status_code == 200, resp.text
        after = reg.security_store.count_alerts(kind="batch_export")
        # 第 2 次跨越阈值恰好告警一次；第 3 次不重复
        assert after == before + 1
        alerts = reg.security_store.list_alerts(status=None, limit=20)
        hit = next(a for a in alerts if a["kind"] == "batch_export")
        assert hit["level"] == "high"
        assert hit["detail"]["actor"] == actor
        # 告警同时入安全审计链（system 落 alert_raised）
        entries, _ = reg.security_store.list_security_audit(action="alert_raised", limit=10)
        assert any(e["object_id"] == "batch_export" for e in entries)
    finally:
        reg.config.alerts.batch_export = original


def test_batch_export_alert_disabled():
    """enabled=false 时同窗口批量下载不产生告警（规则可配置化）。"""
    reg = get_registry()
    actor = f"sec_{uuid.uuid4().hex[:8]}"
    report_id = _make_report_with_pdf()
    original = reg.config.alerts.batch_export
    reg.config.alerts.batch_export = type(original)(enabled=False, window_min=600, threshold=1)
    try:
        before = reg.security_store.count_alerts(kind="batch_export")
        with principal_role("secadmin", username=actor), TestClient(app) as c:
            assert c.get(f"/api/v1/report/{report_id}/pdf").status_code == 200
            assert c.get(f"/api/v1/report/{report_id}/pdf").status_code == 200
        assert reg.security_store.count_alerts(kind="batch_export") == before
    finally:
        reg.config.alerts.batch_export = original


# ---------------------------------------------------------------------------
# 越权 403 告警
# ---------------------------------------------------------------------------


def test_unauthorized_access_alert_on_403():
    """require_role 拒绝（403）→ main.py AuthError 处理路径落 unauthorized_access 告警。"""
    reg = get_registry()
    before = reg.security_store.count_alerts(kind="unauthorized_access")
    # secadmin 调用 sysadmin 专属端点 → 403
    with principal_role("secadmin", username="保密员"), TestClient(app) as c:
        resp = c.post("/api/v1/auth/accounts", json={"username": "x", "role": "sysadmin"})
    assert resp.status_code == 403
    assert reg.security_store.count_alerts(kind="unauthorized_access") > before
    alerts = reg.security_store.list_alerts(status=None, limit=20)
    hit = next(a for a in alerts if a["kind"] == "unauthorized_access")
    assert hit["detail"]["path"] == "/api/v1/auth/accounts"
    assert hit["detail"]["role"] == "secadmin"


# ---------------------------------------------------------------------------
# 未读计数 / 确认（ack）
# ---------------------------------------------------------------------------


def test_unread_count_and_ack_flow():
    """未读计数（拉取式通知）+ ack 确认（secadmin/auditor）+ 角色管控。"""
    reg = get_registry()
    alert = reg.security_store.raise_alert(
        kind="unauthorized_access", level="warn", message="ack 流程测试告警"
    )
    alert_id = alert["alert_id"]
    with principal_role("sysadmin", username="系统管理员"), TestClient(app) as c:
        # 任意已登录角色可拉取计数（登录时拉取语义）
        unread = c.get("/api/v1/alerts/unread-count")
        assert unread.status_code == 200
        assert unread.json()["unread"] >= 1
        # sysadmin 无权 ack（仅 secadmin/auditor）
        assert c.post(f"/api/v1/alerts/{alert_id}/ack").status_code == 403
    with principal_role("secadmin", username="保密员"), TestClient(app) as c:
        acked = c.post(f"/api/v1/alerts/{alert_id}/ack", json={"note": "已阅"})
        assert acked.status_code == 200, acked.text
        assert acked.json()["status"] == "acknowledged"
        assert acked.json()["resolved_by"] == "保密员"
        assert c.get("/api/v1/alerts/unread-count").json()["unread"] >= 0
    # ack 动作入安全审计链
    entries, _ = reg.security_store.list_security_audit(action="alert_ack", limit=10)
    assert any(e["object_id"] == alert_id for e in entries)
    # auditor 亦可 ack（另一条新告警）
    alert2 = reg.security_store.raise_alert(
        kind="unauthorized_access", level="warn", message="auditor ack"
    )
    with principal_role("auditor", username="审计员"), TestClient(app) as c:
        assert c.post(f"/api/v1/alerts/{alert2['alert_id']}/ack").status_code == 200
    # 不存在的告警 404
    with principal_role("secadmin", username="保密员"), TestClient(app) as c:
        assert c.post("/api/v1/alerts/nonexistent/ack").status_code == 404


# ---------------------------------------------------------------------------
# 锁定告警配置开关
# ---------------------------------------------------------------------------


def test_account_lockout_alert_config_gate():
    """account_lockout.enabled=false：锁定仍生效，但不再落告警（审计链不受影响）。"""
    reg = get_registry()
    original = reg.config.alerts.account_lockout
    reg.config.alerts.account_lockout = type(original)(enabled=False)
    try:
        from backend.app.auth import get_auth_service

        account = reg.security_store.create_account(
            username=f"lock_{uuid.uuid4().hex[:8]}", role="sysadmin"
        )
        reg.security_store.set_account_key(account["account_id"], "ab" * 64)
        svc = get_auth_service(reg)
        before_alerts = reg.security_store.count_alerts(kind="account_locked")
        svc._on_account_locked(account)
        assert reg.security_store.count_alerts(kind="account_locked") == before_alerts
    finally:
        reg.config.alerts.account_lockout = original
    # 默认开启时锁定仍落告警
    account2 = reg.security_store.create_account(
        username=f"lock_{uuid.uuid4().hex[:8]}", role="sysadmin"
    )
    from backend.app.auth import get_auth_service

    before = reg.security_store.count_alerts(kind="account_locked")
    get_auth_service(reg)._on_account_locked(account2)
    assert reg.security_store.count_alerts(kind="account_locked") == before + 1
