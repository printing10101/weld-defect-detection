"""导出管控测试（C-14）：审批流、一次性令牌、越权拒绝、高密级拒导。"""

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


def _make_report_with_pdf() -> str:
    """直接落库一个报告 + 伪造 PDF 文件（不经检测管线，聚焦导出管控）。"""
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
    reports_dir = get_registry().config.paths.reports_dir
    from backend.infra.config import resolve_config_path

    pdf = resolve_config_path(reports_dir) / f"{report_id}.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.write_bytes(b"%PDF-1.4\n% fake export-control fixture\n")
    return report_id


def test_approval_flow_end_to_end():
    """申请 → 保密员批准 → 一次性令牌 → 凭令下载 → 二次下载拒绝。"""
    reg = get_registry()
    report_id = _make_report_with_pdf()
    original = reg.config.export.require_approval
    reg.config.export.require_approval = True
    try:
        # 系统管理员直接下载 → 401（需审批/令牌）
        with principal_role("sysadmin", username="系统管理员"), TestClient(app) as c:
            denied = c.get(f"/api/v1/report/{report_id}/pdf")
            assert denied.status_code == 401
            assert denied.json()["detail"]["code"] == "EXPORT_TOKEN_REQUIRED"
        # 保密员（审批人）本人可直接导出（预授权语义）
        with principal_role("secadmin", username="保密员"), TestClient(app) as c:
            ok = c.get(f"/api/v1/report/{report_id}/pdf")
            assert ok.status_code == 200
            assert ok.headers["content-type"].startswith("application/pdf")
        # 普通系统管理员走完整审批流
        with principal_role("sysadmin", username="系统管理员"), TestClient(app) as c:
            req = c.post(
                "/api/v1/export/requests",
                json={"subject": f"report:{report_id}", "reason": "归档调阅"},
            )
            assert req.status_code == 200
            request_id = req.json()["request_id"]
        with principal_role("secadmin", username="保密员"), TestClient(app) as c:
            approve = c.post(f"/api/v1/export/requests/{request_id}/approve")
            assert approve.status_code == 200
            assert approve.json()["status"] == "approved"
        with principal_role("sysadmin", username="系统管理员"), TestClient(app) as c:
            tok = c.post(f"/api/v1/export/requests/{request_id}/token")
            assert tok.status_code == 200
            token = tok.json()["token"]
            # 凭令下载 → 200
            dl = c.get(
                f"/api/v1/report/{report_id}/pdf",
                headers={"X-Export-Token": token},
            )
            assert dl.status_code == 200
            # 一次性：二次下载 → 401
            again = c.get(
                f"/api/v1/report/{report_id}/pdf",
                headers={"X-Export-Token": token},
            )
            assert again.status_code == 401
            assert again.json()["detail"]["code"] == "EXPORT_TOKEN_INVALID"
        # 审批动作入安全审计链（C-19）
        entries, _ = reg.security_store.list_security_audit(limit=50)
        assert any(
            e["action"] == "export_request_approve" and e["object_id"] == request_id
            for e in entries
        )
    finally:
        reg.config.export.require_approval = original


def test_reject_flow():
    reg = get_registry()
    report_id = _make_report_with_pdf()
    original = reg.config.export.require_approval
    reg.config.export.require_approval = True
    try:
        with principal_role("sysadmin", username="系统管理员"), TestClient(app) as c:
            req = c.post("/api/v1/export/requests", json={"subject": f"report:{report_id}"})
            request_id = req.json()["request_id"]
        with principal_role("secadmin", username="保密员"), TestClient(app) as c:
            rej = c.post(f"/api/v1/export/requests/{request_id}/reject")
            assert rej.status_code == 200
            assert rej.json()["status"] == "rejected"
            # 已拒绝不能再签发令牌
            tok = c.post(f"/api/v1/export/requests/{request_id}/token")
            assert tok.status_code == 409
    finally:
        reg.config.export.require_approval = original


def test_wrong_subject_token_rejected():
    """令牌与导出对象（subject）绑定：错配 401。"""
    reg = get_registry()
    report_id = _make_report_with_pdf()
    other_report = _make_report_with_pdf()
    original = reg.config.export.require_approval
    reg.config.export.require_approval = True
    try:
        with principal_role("sysadmin", username="系统管理员"), TestClient(app) as c:
            req = c.post("/api/v1/export/requests", json={"subject": f"report:{report_id}"})
            request_id = req.json()["request_id"]
        with principal_role("secadmin", username="保密员"), TestClient(app) as c:
            c.post(f"/api/v1/export/requests/{request_id}/approve")
        with principal_role("sysadmin", username="系统管理员"), TestClient(app) as c:
            token = c.post(f"/api/v1/export/requests/{request_id}/token").json()["token"]
            wrong = c.get(
                f"/api/v1/report/{other_report}/pdf",
                headers={"X-Export-Token": token},
            )
            assert wrong.status_code == 401
    finally:
        reg.config.export.require_approval = original


def test_no_approval_mode_allows_download():
    """require_approval=false（单机调试）：登录即可下载。"""
    reg = get_registry()
    report_id = _make_report_with_pdf()
    original = reg.config.export.require_approval
    reg.config.export.require_approval = False
    try:
        with principal_role("sysadmin", username="系统管理员"), TestClient(app) as c:
            resp = c.get(f"/api/v1/report/{report_id}/pdf")
            assert resp.status_code == 200
    finally:
        reg.config.export.require_approval = original


def test_false_reports_rejects_high_secret_level():
    """C-10×C-14：误报清单含秘密/机密底片 → 拒绝导出（JSON/CSV 同门禁）。"""
    reg = get_registry()
    image_id = uuid.uuid4().hex
    reg.repository.create_inspection(
        image={
            "id": image_id,
            "path": f"{image_id}.png",
            "source_type": "image",
            "modality": "GENERIC",
        },
        defects=[],
    )
    reg.repository.set_secret_level(image_id, secret_level=2, classification_basis="测试定密")
    original = reg.config.export.require_approval
    reg.config.export.require_approval = False
    try:
        with principal_role("sysadmin", username="系统管理员"), TestClient(app) as c:
            resp = c.get(
                "/api/v1/std-eval/false-reports",
                params={"eval_result_path": str(_write_eval_json_with_film(image_id))},
            )
            assert resp.status_code == 403
            assert resp.json()["detail"]["code"] == "SECRET_LEVEL_DENIED"
    finally:
        reg.config.export.require_approval = original


def _write_eval_json_with_film(image_id: str):
    import json
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp(prefix="export_ctl_"))
    payload = {
        "n_defect_images": 1,
        "n_no_defect_images": 0,
        "false_report_films": [{"id": image_id, "path": f"{image_id}.png", "n_false_reports": 1}],
        "result": {},
    }
    p = tmp / "std_eval.json"
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return str(p)


def test_secadmin_cannot_approve_own_request():
    """职责分离（与载体销毁跨账号确认同口径）：申请人=审批人 → 409 SELF_APPROVAL。"""
    reg = get_registry()
    report_id = _make_report_with_pdf()
    original = reg.config.export.require_approval
    reg.config.export.require_approval = True
    try:
        with principal_role("secadmin", username="保密员"), TestClient(app) as c:
            req = c.post(
                "/api/v1/export/requests",
                json={"subject": f"report:{report_id}", "reason": "自批约束测试"},
            )
            assert req.status_code == 200
            request_id = req.json()["request_id"]
            approve = c.post(f"/api/v1/export/requests/{request_id}/approve")
            assert approve.status_code == 409
            assert approve.json()["detail"]["code"] == "SELF_APPROVAL"
            # 状态必须仍是 pending（未产生审批事实）
            assert reg.export_store.get(request_id)["status"] == "pending"
    finally:
        reg.config.export.require_approval = original
