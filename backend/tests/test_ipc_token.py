"""C-17 IPC 一次性令牌：真实链路专项测试。

conftest 全局置 SCAN_IPC__ENFORCE=false（对既有测试零侵入）；本模块临时
置 true 构造独立 app 实例走真实 enforce 链路：

- 令牌缺失 → 401（IPC_TOKEN_REQUIRED）；
- 正确 X-IPC-Token 头 → 放行；
- /health /metrics /auth 豁免；静态资源（非 /api 路径）豁免；
- 已带会话凭据（Authorization Bearer / ?access_token）→ 中间件放行给下游鉴权；
- 错误令牌 → 401；
- enforce=false → 全放行；
- 令牌落盘 data/ipc_token（进程生命周期有效）。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.infra.ipc_token import token_file_path


@pytest.fixture()
def enforced_app(tmp_path: Path, monkeypatch):
    """enforce=true 的独立 app 实例：数据目录指向 tmp（令牌文件不落真实 data/）。

    create_app() 产出全新实例——conftest 的 principal 覆盖挂在模块级单例上，
    这里同步拷贝到新实例，使业务端点在测试下走与既有套件相同的鉴权替身
    （本模块只关心 IPC 中间件行为，不重复覆盖真实三员鉴权——那是 test_auth 的职责）。
    """
    monkeypatch.setenv("SCAN_IPC__ENFORCE", "true")
    monkeypatch.setenv("SCAN_PATHS__DATA_DIR", str(tmp_path / "data"))
    from backend.app import auth as _auth
    from backend.app.main import create_app
    from backend.app.main import app as _module_app
    from backend.infra.config import resolve_config_path

    app = create_app()
    saved_override = _module_app.dependency_overrides.get(_auth.get_principal)
    if saved_override is not None:
        app.dependency_overrides[_auth.get_principal] = saved_override
    try:
        yield app, resolve_config_path(str(tmp_path / "data"))
    finally:
        app.dependency_overrides.pop(_auth.get_principal, None)


def test_missing_token_401_and_correct_token_passes(enforced_app):
    """令牌缺失 401；正确 X-IPC-Token 放行；令牌已落盘。"""
    app, data_dir = enforced_app
    with TestClient(app) as client:
        resp = client.get("/api/v1/records")
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "IPC_TOKEN_REQUIRED"

        # lifespan 已签发令牌并落盘（进程生命周期有效）
        token_file = token_file_path(data_dir)
        assert token_file.is_file()
        token = token_file.read_text(encoding="utf-8").strip()
        assert len(token) >= 32  # token_urlsafe(32B) ≥ 43 字符

        ok = client.get("/api/v1/records", headers={"X-IPC-Token": token})
        assert ok.status_code == 200


def test_wrong_token_401(enforced_app):
    app, data_dir = enforced_app
    with TestClient(app) as client:
        token = token_file_path(data_dir).read_text(encoding="utf-8").strip()
        resp = client.get("/api/v1/records", headers={"X-IPC-Token": "wrong-" + token})
        assert resp.status_code == 401


def test_health_metrics_auth_and_static_exempt(enforced_app):
    """豁免面：存活/指标/认证端点与静态资源不要求 IPC 令牌。"""
    app, _data_dir = enforced_app
    with TestClient(app) as client:
        assert client.get("/api/v1/health").status_code == 200
        assert client.get("/api/v1/metrics").status_code == 200
        # 登录引导链路（挑战签发）必须先于令牌分发可用
        assert client.get("/api/v1/auth/challenge").status_code == 200
        # 非 /api 路径 = SPA 静态托管豁免（dist 缺失时 404，但不是 IPC 401）
        resp = client.get("/")
        assert resp.status_code != 401


def test_session_credential_passes_middleware(enforced_app):
    """已带会话凭据（Bearer / access_token）→ 中间件放行，由下游鉴权校验。"""
    app, _data_dir = enforced_app
    with TestClient(app) as client:
        # conftest 的 principal 覆盖在位：凭据在场即 200（真实会话校验由
        # test_auth 覆盖；本测试只断言中间件不拦"有凭据"的请求）
        authed = client.get("/api/v1/records", headers={"Authorization": "Bearer dummy"})
        assert authed.status_code == 200
        via_query = client.get("/api/v1/records?access_token=dummy")
        assert via_query.status_code == 200


def test_enforce_off_allows_all(monkeypatch, tmp_path: Path):
    """enforce=false（conftest 兼容态）：无令牌全放行。"""
    monkeypatch.setenv("SCAN_IPC__ENFORCE", "false")
    monkeypatch.setenv("SCAN_PATHS__DATA_DIR", str(tmp_path / "data"))
    from backend.app import auth as _auth
    from backend.app.main import create_app
    from backend.app.main import app as _module_app

    app = create_app()
    saved_override = _module_app.dependency_overrides.get(_auth.get_principal)
    if saved_override is not None:
        app.dependency_overrides[_auth.get_principal] = saved_override
    with TestClient(app) as client:
        assert client.get("/api/v1/records").status_code == 200
        # 未签发令牌（enforce 关不落盘）
        assert not token_file_path(tmp_path / "data").exists()
