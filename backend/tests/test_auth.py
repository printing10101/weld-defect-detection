"""三员身份认证端到端测试（C-06/C-07/C-09/C-19）。

覆盖：引导窗口、SM2 挑战-响应登录（后端代签 + 客户端自签两种凭据）、
无 token 401、越权 403、三员互斥（权限矩阵抽查）、挑战一次一用、
连续失败锁定 + 告警入库、会话并发上限、注销。

模式：账号管理等准备动作以 conftest 注入的 sysadmin principal 完成；
随后经 real_auth_client 上下文摘除覆盖，验证**真实鉴权链路**。
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager

from fastapi.testclient import TestClient

from backend.app import auth as auth_mod
from backend.app.dependencies import get_registry
from backend.app.main import app


@contextmanager
def real_auth_client():
    """摘除 conftest 的 principal 覆盖，走真实鉴权链路（退出恢复）。"""
    saved = app.dependency_overrides.pop(auth_mod.get_principal, None)
    try:
        with TestClient(app) as c:
            yield c
    finally:
        if saved is not None:
            app.dependency_overrides[auth_mod.get_principal] = saved


def _admin_client() -> TestClient:
    """conftest 注入的 sysadmin principal 上下文（账号准备用）。"""
    return TestClient(app)


def _err_code(resp) -> str:
    """统一取错误码（AppError 信封 error.code / HTTPException detail.code 兼容）。"""
    body = resp.json()
    err = body.get("error") or body.get("detail") or {}
    return err.get("code", "")


def _login(client: TestClient, username: str, private_key: str):
    ch = client.get("/api/v1/auth/challenge").json()
    return client.post(
        "/api/v1/auth/login",
        json={
            "username": username,
            "challenge_id": ch["challenge_id"],
            "private_key": private_key,
        },
    )


def _create_account(client: TestClient, role: str) -> tuple[str, str]:
    """以 sysadmin 覆盖身份创建账号并签发软证书，返回 (username, private_key)。"""
    username = f"u_{uuid.uuid4().hex[:10]}"
    resp = client.post("/api/v1/auth/accounts", json={"username": username, "role": role})
    assert resp.status_code == 200, resp.text
    kp = client.post(f"/api/v1/auth/accounts/{resp.json()['account_id']}/keypair")
    assert kp.status_code == 200, kp.text
    return username, kp.json()["private_key"]


# ---------------------------------------------------------------------------
# 引导窗口 / 账号管理
# ---------------------------------------------------------------------------


def test_bootstrap_only_when_no_accounts():
    """引导窗口语义：有账号后 bootstrap 永久关闭（409）。"""
    reg = get_registry()
    if reg.security_store.count_accounts() > 0:
        with _admin_client() as c:
            resp = c.post(
                "/api/v1/auth/bootstrap",
                json={"username": f"boot_{uuid.uuid4().hex[:8]}", "role": "sysadmin"},
            )
            assert resp.status_code == 409
            assert _err_code(resp) == "BOOTSTRAP_CLOSED"
    else:
        with _admin_client() as c:
            resp = c.post(
                "/api/v1/auth/bootstrap",
                json={"username": f"boot_{uuid.uuid4().hex[:8]}", "role": "sysadmin"},
            )
            assert resp.status_code == 200, resp.text
            assert resp.json()["private_key"]  # 未带公钥 → 现场签发软证书


def test_account_role_whitelist():
    """一人一岗：角色只能是三员之一；非法角色 422。"""
    with _admin_client() as c:
        resp = c.post(
            "/api/v1/auth/accounts",
            json={"username": f"x_{uuid.uuid4().hex[:8]}", "role": "operator"},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 登录链路
# ---------------------------------------------------------------------------


def test_login_success_and_me():
    with _admin_client() as c:
        username, priv = _create_account(c, "sysadmin")
    with real_auth_client() as c:
        login = _login(c, username, priv)
        assert login.status_code == 200, login.text
        body = login.json()
        assert body["role"] == "sysadmin"
        token = body["token"]
        me = c.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me.status_code == 200
        assert me.json()["username"] == username
        # 无 token → 401
        assert c.get("/api/v1/auth/me").status_code == 401


def test_login_with_client_signature():
    """客户端自签模式：signature 与后端代签等价（同一私钥）。"""
    from backend.infra.crypto import sm2_sign_with_private

    with _admin_client() as c:
        username, priv = _create_account(c, "auditor")
    with real_auth_client() as c:
        ch = c.get("/api/v1/auth/challenge").json()
        sig = sm2_sign_with_private(priv, ch["nonce"].encode())
        resp = c.post(
            "/api/v1/auth/login",
            json={
                "username": username,
                "challenge_id": ch["challenge_id"],
                "signature": sig,
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["role"] == "auditor"


def test_challenge_one_time_use():
    """挑战一次一用：同一 challenge_id 二次登录失败。"""
    with _admin_client() as c:
        username, priv = _create_account(c, "sysadmin")
    with real_auth_client() as c:
        ch = c.get("/api/v1/auth/challenge").json()
        first = c.post(
            "/api/v1/auth/login",
            json={"username": username, "challenge_id": ch["challenge_id"], "private_key": priv},
        )
        assert first.status_code == 200
        second = c.post(
            "/api/v1/auth/login",
            json={"username": username, "challenge_id": ch["challenge_id"], "private_key": priv},
        )
        assert second.status_code == 401


def test_login_with_wrong_key_fails():
    with _admin_client() as c:
        username, _priv = _create_account(c, "sysadmin")
    with real_auth_client() as c:
        ch = c.get("/api/v1/auth/challenge").json()
        resp = c.post(
            "/api/v1/auth/login",
            json={
                "username": username,
                "challenge_id": ch["challenge_id"],
                "private_key": "0" * 64,  # 错误私钥
            },
        )
        assert resp.status_code == 401


def test_unknown_account_uniform_error():
    """防用户枚举：不存在账号与凭据错误同文案。"""
    with real_auth_client() as c:
        ch = c.get("/api/v1/auth/challenge").json()
        resp = c.post(
            "/api/v1/auth/login",
            json={
                "username": f"ghost_{uuid.uuid4().hex[:8]}",
                "challenge_id": ch["challenge_id"],
                "private_key": "0" * 64,
            },
        )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# 锁定与告警（C-19）
# ---------------------------------------------------------------------------


def test_lockout_after_failed_attempts_raises_alert():
    reg = get_registry()
    original = reg.config.auth.max_failed_attempts
    reg.config.auth.max_failed_attempts = 2
    try:
        with _admin_client() as c:
            username, priv = _create_account(c, "sysadmin")
        with real_auth_client() as c:
            for _ in range(2):  # 连续失败 2 次（阈值=2）→ 锁定
                ch = c.get("/api/v1/auth/challenge").json()
                resp = c.post(
                    "/api/v1/auth/login",
                    json={
                        "username": username,
                        "challenge_id": ch["challenge_id"],
                        "private_key": "0" * 64,
                    },
                )
                assert resp.status_code == 401
            # 第三次：账号已锁定 → 423（即使凭据正确也拒绝）
            ch = c.get("/api/v1/auth/challenge").json()
            resp = c.post(
                "/api/v1/auth/login",
                json={
                    "username": username,
                    "challenge_id": ch["challenge_id"],
                    "private_key": priv,
                },
            )
            assert resp.status_code == 423
            assert _err_code(resp) == "ACCOUNT_LOCKED"
    finally:
        reg.config.auth.max_failed_attempts = original
    # 告警入库 + 保密员可见 + 解锁入安全审计链
    assert any(
        a["kind"] == "account_locked" and username in a["message"]
        for a in get_registry().security_store.list_alerts()
    ), "锁定必须落告警"
    with _admin_client() as c:
        sec_user, sec_key = _create_account(c, "secadmin")
        account = next(
            a for a in c.get("/api/v1/auth/accounts").json() if a["username"] == username
        )
    with real_auth_client() as c:
        sec_token = _login(c, sec_user, sec_key).json()["token"]
        alerts = c.get(
            "/api/v1/auth/alerts", headers={"Authorization": f"Bearer {sec_token}"}
        ).json()
        assert any(username in a["message"] for a in alerts), "保密员可见锁定告警"
        unlock = c.post(
            f"/api/v1/auth/accounts/{account['account_id']}/unlock",
            headers={"Authorization": f"Bearer {sec_token}"},
        )
        assert unlock.status_code == 200


def test_security_chain_readable_by_auditor_only():
    """C-19：安全审计链仅审计员只读；sysadmin/保密员越权 403。"""
    reg = get_registry()
    original = reg.config.auth.max_failed_attempts
    reg.config.auth.max_failed_attempts = 5
    try:
        with _admin_client() as c:
            aud_user, aud_key = _create_account(c, "auditor")
            sec_user, sec_key = _create_account(c, "secadmin")
        with real_auth_client() as c:
            aud_token = _login(c, aud_user, aud_key).json()["token"]
            sec_token = _login(c, sec_user, sec_key).json()["token"]
            ok = c.get("/api/v1/audit/security", headers={"Authorization": f"Bearer {aud_token}"})
            assert ok.status_code == 200
            assert ok.json()["chain_valid"] is True
            denied = c.get(
                "/api/v1/audit/security", headers={"Authorization": f"Bearer {sec_token}"}
            )
            assert denied.status_code == 403
    finally:
        reg.config.auth.max_failed_attempts = original


# ---------------------------------------------------------------------------
# 会话治理
# ---------------------------------------------------------------------------


def test_session_limit_revokes_oldest():
    reg = get_registry()
    original = reg.config.auth.max_sessions
    reg.config.auth.max_sessions = 1
    try:
        with _admin_client() as c:
            username, priv = _create_account(c, "sysadmin")
        with real_auth_client() as c:
            t1 = _login(c, username, priv).json()["token"]
            t2 = _login(c, username, priv).json()["token"]  # 顶掉 t1（单点登录语义）
            assert (
                c.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {t1}"}).status_code
                == 401
            )
            assert (
                c.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {t2}"}).status_code
                == 200
            )
    finally:
        reg.config.auth.max_sessions = original


def test_logout_revokes_session():
    with _admin_client() as c:
        username, priv = _create_account(c, "sysadmin")
    with real_auth_client() as c:
        token = _login(c, username, priv).json()["token"]
        headers = {"Authorization": f"Bearer {token}"}
        assert c.post("/api/v1/auth/logout", headers=headers).status_code == 200
        assert c.get("/api/v1/auth/me", headers=headers).status_code == 401


def test_disabled_account_sessions_revoked():
    with _admin_client() as c:
        username, priv = _create_account(c, "sysadmin")
        accounts = c.get("/api/v1/auth/accounts").json()
        account_id = next(a["account_id"] for a in accounts if a["username"] == username)
    with real_auth_client() as c:
        token = _login(c, username, priv).json()["token"]
    with _admin_client() as c:
        resp = c.post(f"/api/v1/auth/accounts/{account_id}/status", json={"status": "disabled"})
        assert resp.status_code == 200
    with real_auth_client() as c:
        assert (
            c.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code
            == 401
        )


def test_ukey_mode_reserved():
    """UKey 硬件模式接口预留：登录返回 501（未真机验证，诚实声明）。"""
    reg = get_registry()
    username = f"uk_{uuid.uuid4().hex[:8]}"
    account = reg.security_store.create_account(
        username=username,
        role="sysadmin",
        auth_mode="ukey",
        sm2_public_key="ab" * 64,
        created_by="test",
    )
    with real_auth_client() as c:
        ch = c.get("/api/v1/auth/challenge").json()
        resp = c.post(
            "/api/v1/auth/login",
            json={
                "username": username,
                "challenge_id": ch["challenge_id"],
                "signature": "0" * 128,
            },
        )
        assert resp.status_code == 501
        assert _err_code(resp) == "UKEY_NOT_AVAILABLE"
    reg.security_store.set_account_status(account["account_id"], "disabled")


# ---------------------------------------------------------------------------
# 业务端点鉴权抽查（无 token 401 / 越权 403 / 三员互斥）
# ---------------------------------------------------------------------------


def test_business_endpoints_require_token():
    """敏感业务端点未登录一律 401；认证/存活端点保持开放。"""
    with real_auth_client() as c:
        assert c.get("/api/v1/audit").status_code == 401
        assert c.get("/api/v1/records").status_code == 401
        assert c.get("/api/v1/carriers").status_code == 401
        assert c.get("/api/v1/export/requests").status_code == 401
        assert c.post("/api/v1/system/backup").status_code == 401
        assert c.get("/api/v1/auth/challenge").status_code == 200


def test_role_matrix_on_sensitive_ops():
    """三员互斥抽查（C-06 权限矩阵）：账号管理仅系统管理员；
    密级变更仅保密员；审计员越权写操作 403。"""
    reg = get_registry()
    original = reg.config.auth.max_failed_attempts
    reg.config.auth.max_failed_attempts = 5
    try:
        with _admin_client() as c:
            sec_user, sec_key = _create_account(c, "secadmin")
            aud_user, aud_key = _create_account(c, "auditor")
        with real_auth_client() as c:
            sec_token = _login(c, sec_user, sec_key).json()["token"]
            aud_token = _login(c, aud_user, aud_key).json()["token"]
            # 账号管理：保密员/审计员 → 403
            for tok in (sec_token, aud_token):
                resp = c.get("/api/v1/auth/accounts", headers={"Authorization": f"Bearer {tok}"})
                assert resp.status_code == 403
            # 密级变更：审计员 → 403（角色判定先于资源存在性 404）
            resp = c.post(
                "/api/v1/classification/image/no-such-image",
                json={"secret_level": 2, "classification_basis": "测试依据"},
                headers={"Authorization": f"Bearer {aud_token}"},
            )
            assert resp.status_code == 403
    finally:
        reg.config.auth.max_failed_attempts = original


def test_operator_name_prefers_account_identity():
    """登录态下审计 actor 以账号为准（X-Operator-Name 不覆盖账号身份）；
    未登录时头部仅作审计 actor 记录（缺省 local）。"""
    from starlette.requests import Request as StarletteRequest

    from backend.app.dependencies import get_operator_name

    scope = {"type": "http", "headers": [], "method": "GET", "path": "/", "query_string": b""}
    req = StarletteRequest(scope)
    req.state.principal = auth_mod.Principal("a", "账号甲", "sysadmin")
    assert get_operator_name(req, "someone") == "账号甲"  # 账号优先
    req2 = StarletteRequest({**scope, "state": {}})  # 独立 state（scope 中 state 为共享 dict）
    assert get_operator_name(req2, "someone") == "someone"  # 未登录：头部仅作 actor 记录
    assert get_operator_name(req2, None) == "local"
