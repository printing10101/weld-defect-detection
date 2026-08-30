"""载体台账测试（C-12）：登记/借还/销毁双确认/销毁证明 PDF/审计留痕。"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import Iterator

from fastapi import Request
from fastapi.testclient import TestClient

from backend.app import auth as auth_mod
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


def _new_carrier_id() -> str:
    return f"CN-TEST-{uuid.uuid4().hex[:10]}"


def _register(client: TestClient, carrier_id: str) -> None:
    resp = client.post(
        "/api/v1/carriers",
        json={
            "carrier_id": carrier_id,
            "kind": "film",
            "secret_level": 1,
            "owner": "张三",
        },
    )
    assert resp.status_code == 200, resp.text


def test_register_requires_secadmin():
    carrier_id = _new_carrier_id()
    with principal_role("sysadmin"):
        with TestClient(app) as c:
            resp = c.post(
                "/api/v1/carriers",
                json={"carrier_id": carrier_id, "kind": "film"},
            )
            assert resp.status_code == 403  # 登记仅保密员
    with principal_role("secadmin"):
        with TestClient(app) as c:
            _register(c, carrier_id)
            dup = c.post(
                "/api/v1/carriers",
                json={"carrier_id": carrier_id, "kind": "film"},
            )
            assert dup.status_code == 422  # 编号唯一
            assert c.get("/api/v1/carriers").status_code == 200


def test_borrow_return_flow():
    carrier_id = _new_carrier_id()
    with principal_role("secadmin"):
        with TestClient(app) as c:
            _register(c, carrier_id)
    with principal_role("sysadmin", username="操作员A"):
        with TestClient(app) as c:
            resp = c.post(f"/api/v1/carriers/{carrier_id}/borrow", json={"note": "外借评片"})
            assert resp.status_code == 200
            assert resp.json()["status"] == "borrowed"
            # 重复借用 → 409
            assert (
                c.post(f"/api/v1/carriers/{carrier_id}/borrow").status_code == 409
            )
            back = c.post(f"/api/v1/carriers/{carrier_id}/return", json={"note": "归还"})
            assert back.status_code == 200
            assert back.json()["status"] == "returned"
            detail = c.get(f"/api/v1/carriers/{carrier_id}").json()
            actions = [h["action"] for h in detail["borrow_history"]]
            assert actions == ["register", "borrow", "return"]


def test_destroy_dual_confirmation_and_certificate():
    carrier_id = _new_carrier_id()
    with principal_role("secadmin"):
        with TestClient(app) as c:
            _register(c, carrier_id)
            req = c.post(
                f"/api/v1/carriers/{carrier_id}/destroy-request",
                json={"destroy_method": "碎纸/消磁", "note": "到期销毁"},
            )
            assert req.status_code == 200
            assert req.json()["status"] == "pending_destroy"
    # 系统管理员确认（与保密员不同账号）→ 已销毁
    with principal_role("sysadmin", username="系统管理员"):
        with TestClient(app) as c:
            conf = c.post(f"/api/v1/carriers/{carrier_id}/destroy-confirm")
            assert conf.status_code == 200, conf.text
            assert conf.json()["status"] == "destroyed"
            cert = c.get(f"/api/v1/carriers/{carrier_id}/destroy-certificate.pdf")
            assert cert.status_code == 200
            assert cert.headers["content-type"].startswith("application/pdf")
    # 登记与销毁入安全审计链（C-19）
    from backend.app.dependencies import get_registry

    reg = get_registry()
    sec_entries, _ = reg.security_store.list_security_audit(limit=50)
    actions = {e["action"] for e in sec_entries if e["object_id"] == carrier_id}
    assert {"carrier_register", "carrier_destroy"} <= actions


def test_destroy_self_confirm_rejected():
    """销毁双确认：同账号自确认 409 拒绝。"""
    carrier_id = _new_carrier_id()
    with principal_role("secadmin", username="保密员甲"):
        with TestClient(app) as c:
            _register(c, carrier_id)
            req = c.post(
                f"/api/v1/carriers/{carrier_id}/destroy-request",
                json={"destroy_method": "焚烧"},
            )
            assert req.status_code == 200
            # 同一保密员账号自确认：角色不符 403（确认仅限系统管理员）
            resp = c.post(f"/api/v1/carriers/{carrier_id}/destroy-confirm")
            assert resp.status_code == 403
    # 即便把自己角色临时给系统管理员，同账号自确认仍 409
    with principal_role("sysadmin", username="保密员甲"):
        with TestClient(app) as c:
            resp = c.post(f"/api/v1/carriers/{carrier_id}/destroy-confirm")
            assert resp.status_code == 409
            assert resp.json()["detail"]["code"] == "SELF_CONFIRM"


def test_destroy_requires_pending_state():
    carrier_id = _new_carrier_id()
    with principal_role("secadmin"):
        with TestClient(app) as c:
            _register(c, carrier_id)
    with principal_role("sysadmin"):
        with TestClient(app) as c:
            # 未发起销毁直接确认 → 409
            resp = c.post(f"/api/v1/carriers/{carrier_id}/destroy-confirm")
            assert resp.status_code == 409


def test_destroy_certificate_blocked_before_destroy():
    carrier_id = _new_carrier_id()
    with principal_role("secadmin"):
        with TestClient(app) as c:
            _register(c, carrier_id)
            resp = c.get(f"/api/v1/carriers/{carrier_id}/destroy-certificate.pdf")
            assert resp.status_code == 409  # 未销毁不能出证明
