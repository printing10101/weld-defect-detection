"""T3 鉴权与 RBAC 集成测试（§T3，P0 用户权限与登录）。

通过 create_app() 取得无 conftest 覆盖的全新实例，验证真实鉴权链路：
- 引导管理员播种 + 登录；
- 无令牌 → 401、篡改令牌 → 401、错误密码 → 401；
- 合法令牌可访问自身信息 / 受保护端点；
- RBAC：注册/用户列表/改密/审计 的角色边界（reviewer/auditor/admin）；
- 禁用用户禁止登录；
- 审计/签字 actor 闭环（见 test_audit_actor_is_logged_in_user，经真实评片链路）。

本文件不依赖 conftest 的 admin 覆盖（仅复用其确定性环境变量 SCAN_AUTH_SECRET /
SCAN_ADMIN_USERNAME / SCAN_ADMIN_PASSWORD），故能测到真正的 401/403 行为。
"""

from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from backend.app.main import create_app

# 与 conftest 环境变量一致：bootstrap 管理员凭据可复现。
ADMIN_USER = "admin"
ADMIN_PW = "TestPassw0rd!"


def _client() -> TestClient:
    return TestClient(create_app())


def _login(client: TestClient, username: str, password: str):
    return client.post("/api/v1/auth/login", json={"username": username, "password": password})


def _token(client: TestClient, username: str, password: str) -> str:
    r = _login(client, username, password)
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def _auth(token: str) -> dict:
    return {"X-Scan-Token": token}


def _make_user(client: TestClient, admin_token: str, role: str) -> tuple[str, str]:
    """以 admin 身份注册一个唯一用户，返回 (username, password)。"""
    username = f"{role[:3]}_{uuid.uuid4().hex[:8]}"
    pw = "startpw12"
    r = client.post(
        "/api/v1/auth/register",
        headers=_auth(admin_token),
        json={"username": username, "password": pw, "role": role},
    )
    assert r.status_code == 200, r.text
    return username, pw


# ---------------------------------------------------------------------------
# 引导管理员 + 登录
# ---------------------------------------------------------------------------
def test_bootstrap_admin_can_login() -> None:
    with _client() as c:
        r = _login(c, ADMIN_USER, ADMIN_PW)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["token_type"] == "bearer"
        assert body["access_token"]
        assert body["user"]["username"] == ADMIN_USER
        assert body["user"]["role"] == "admin"


def test_login_wrong_password_rejected() -> None:
    with _client() as c:
        r = _login(c, ADMIN_USER, "definitely-wrong")
        assert r.status_code == 401
        assert r.json()["detail"]["code"] == "INVALID_CREDENTIALS"


# ---------------------------------------------------------------------------
# 令牌要求与校验
# ---------------------------------------------------------------------------
def test_me_requires_token() -> None:
    with _client() as c:
        r = c.get("/api/v1/auth/me")
        assert r.status_code == 401
        assert r.json()["detail"]["code"] == "UNAUTHORIZED"


def test_protected_endpoint_requires_token() -> None:
    with _client() as c:
        r = c.get("/api/v1/models")
        assert r.status_code == 401
        assert r.json()["detail"]["code"] == "UNAUTHORIZED"


def test_tampered_token_rejected() -> None:
    with _client() as c:
        tok = _token(c, ADMIN_USER, ADMIN_PW)
        r = c.get("/api/v1/auth/me", headers=_auth(tok + "x"))
        assert r.status_code == 401
        assert r.json()["detail"]["code"] == "INVALID_TOKEN"


def test_me_with_valid_token() -> None:
    with _client() as c:
        tok = _token(c, ADMIN_USER, ADMIN_PW)
        r = c.get("/api/v1/auth/me", headers=_auth(tok))
        assert r.status_code == 200, r.text
        assert r.json()["username"] == ADMIN_USER
        assert r.json()["role"] == "admin"


# ---------------------------------------------------------------------------
# RBAC：注册 / 用户列表
# ---------------------------------------------------------------------------
def test_register_requires_admin_role() -> None:
    with _client() as c:
        tok = _token(c, ADMIN_USER, ADMIN_PW)
        uname, pw = _make_user(c, tok, "reviewer")
        rtok = _token(c, uname, pw)
        # reviewer 试图注册他人 → 403
        r = c.post(
            "/api/v1/auth/register",
            headers=_auth(rtok),
            json={
                "username": f"x_{uuid.uuid4().hex[:8]}",
                "password": "anotherpw1",
                "role": "reviewer",
            },
        )
        assert r.status_code == 403
        assert r.json()["detail"]["code"] == "FORBIDDEN"


def test_list_users_rbac() -> None:
    with _client() as c:
        tok = _token(c, ADMIN_USER, ADMIN_PW)
        uname, pw = _make_user(c, tok, "reviewer")
        rtok = _token(c, uname, pw)
        # reviewer 访问用户列表 → 403
        assert c.get("/api/v1/auth/users", headers=_auth(rtok)).status_code == 403
        # admin 访问 → 200，列表含 admin 与新建 reviewer
        r = c.get("/api/v1/auth/users", headers=_auth(tok))
        assert r.status_code == 200
        usernames = {u["username"] for u in r.json()}
        assert ADMIN_USER in usernames and uname in usernames


# ---------------------------------------------------------------------------
# RBAC：改密边界
# ---------------------------------------------------------------------------
def test_change_own_password() -> None:
    with _client() as c:
        tok = _token(c, ADMIN_USER, ADMIN_PW)
        uname, pw = _make_user(c, tok, "reviewer")
        rtok = _token(c, uname, pw)
        # 旧密码错误 → 401
        r = c.post(
            "/api/v1/auth/change-password",
            headers=_auth(rtok),
            json={"new_password": "newpass12", "old_password": "wrongold"},
        )
        assert r.status_code == 401
        # 正确旧密码 → ok
        r2 = c.post(
            "/api/v1/auth/change-password",
            headers=_auth(rtok),
            json={"new_password": "newpass12", "old_password": pw},
        )
        assert r2.status_code == 200 and r2.json()["ok"] is True
        # 新密码可登录
        assert _login(c, uname, "newpass12").status_code == 200


def test_admin_change_other_password() -> None:
    with _client() as c:
        tok = _token(c, ADMIN_USER, ADMIN_PW)
        uname, _ = _make_user(c, tok, "reviewer")
        # admin 代改，免 old_password
        r = c.post(
            "/api/v1/auth/change-password",
            headers=_auth(tok),
            json={"username": uname, "new_password": "adminnew1"},
        )
        assert r.status_code == 200, r.text
        assert _login(c, uname, "adminnew1").status_code == 200


def test_reviewer_cannot_change_other_password() -> None:
    with _client() as c:
        tok = _token(c, ADMIN_USER, ADMIN_PW)
        u1, pw1 = _make_user(c, tok, "reviewer")
        u2, _ = _make_user(c, tok, "reviewer")
        r1tok = _token(c, u1, pw1)
        r = c.post(
            "/api/v1/auth/change-password",
            headers=_auth(r1tok),
            json={"username": u2, "new_password": "hacked12"},
        )
        assert r.status_code == 403


# ---------------------------------------------------------------------------
# RBAC：审计端点仅限 auditor / admin
# ---------------------------------------------------------------------------
def test_audit_endpoint_rbac() -> None:
    with _client() as c:
        tok = _token(c, ADMIN_USER, ADMIN_PW)
        rev, rev_pw = _make_user(c, tok, "reviewer")
        aud, aud_pw = _make_user(c, tok, "auditor")
        rtok = _token(c, rev, rev_pw)
        atok = _token(c, aud, aud_pw)
        # reviewer → 403
        assert c.get("/api/v1/audit", headers=_auth(rtok)).status_code == 403
        # auditor → 200（可为空日志；响应含 entries/total/chain_valid）
        r = c.get("/api/v1/audit", headers=_auth(atok))
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body.get("entries"), list)
        assert "chain_valid" in body


# ---------------------------------------------------------------------------
# 禁用用户
# ---------------------------------------------------------------------------
def test_disabled_user_cannot_login() -> None:
    with _client() as c:
        tok = _token(c, ADMIN_USER, ADMIN_PW)
        uname, pw = _make_user(c, tok, "reviewer")
        r = c.post(
            f"/api/v1/auth/users/{uname}/disable",
            headers=_auth(tok),
            json={"disabled": True},
        )
        assert r.status_code == 200, r.text
        assert _login(c, uname, pw).status_code == 401


# ---------------------------------------------------------------------------
# F5：登录防爆破锁定
# ---------------------------------------------------------------------------
def test_login_lockout_after_repeated_failures() -> None:
    with _client() as c:
        victim = "lockout_candidate"
        # 连续错误密码（同一用户名）达到阈值后触发锁定
        for _ in range(5):
            r = _login(c, victim, "wrong-pw")
            assert r.status_code == 401, r.text
        # 第 6 次（含后续）应被锁定，返回 429 且带 Retry-After
        r = _login(c, victim, "wrong-pw")
        assert r.status_code == 429
        assert r.json()["detail"]["code"] == "ACCOUNT_LOCKED"
        assert "Retry-After" in r.headers


def test_login_lockout_reset_on_success() -> None:
    with _client() as c:
        # 少量失败不应锁定，且成功登录后计数清零
        for _ in range(3):
            assert _login(c, ADMIN_USER, "wrong").status_code == 401
        assert _login(c, ADMIN_USER, ADMIN_PW).status_code == 200


# ---------------------------------------------------------------------------
# F6：令牌吊销 + 轮换
# ---------------------------------------------------------------------------
def test_logout_revokes_token() -> None:
    with _client() as c:
        tok = _token(c, ADMIN_USER, ADMIN_PW)
        # 注销前可访问
        assert c.get("/api/v1/auth/me", headers=_auth(tok)).status_code == 200
        # 注销
        r = c.post("/api/v1/auth/logout", headers=_auth(tok))
        assert r.status_code == 200 and r.json()["ok"] is True
        # 注销后原令牌立即失效
        r2 = c.get("/api/v1/auth/me", headers=_auth(tok))
        assert r2.status_code == 401
        assert r2.json()["detail"]["code"] == "TOKEN_REVOKED"


def test_refresh_rotates_and_revokes_old() -> None:
    with _client() as c:
        tok1 = _token(c, ADMIN_USER, ADMIN_PW)
        r = c.post("/api/v1/auth/refresh", headers=_auth(tok1))
        assert r.status_code == 200, r.text
        tok2 = r.json()["access_token"]
        assert tok2 and tok2 != tok1
        # 旧令牌已吊销
        assert c.get("/api/v1/auth/me", headers=_auth(tok1)).status_code == 401
        # 新令牌可用
        assert c.get("/api/v1/auth/me", headers=_auth(tok2)).status_code == 200
