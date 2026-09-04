"""C-18 远程运维受控：GET /audit/operations 运维操作清单 + 回放专项测试。

- 时间线只含系统管理类操作（白名单动作），业务动作（inspect 等）不混入；
- 结构化字段齐全：时间 / actor / 动作 / 参数摘要 / 结果；
- 仅审计员（auditor）可读：sysadmin 越权 403；
- 仅审计员可读的既有语义（/audit/security）不受影响。
"""

from __future__ import annotations

from contextlib import contextmanager

from fastapi import Request
from fastapi.testclient import TestClient

from backend.app import auth as auth_mod
from backend.app.dependencies import get_registry
from backend.app.main import app
from backend.app.routers.audit import OPERATION_ACTIONS


@contextmanager
def _role_override(role: str):
    """以指定角色临时替换 conftest 注入的 principal（退出恢复 sysadmin 覆盖）。"""
    saved = app.dependency_overrides.get(auth_mod.get_principal)

    def _fake(request: Request):
        principal = auth_mod.Principal(account_id="role-test", username=f"测试{role}", role=role)
        request.state.principal = principal
        return principal

    app.dependency_overrides[auth_mod.get_principal] = _fake
    try:
        yield TestClient(app)
    finally:
        if saved is None:
            app.dependency_overrides.pop(auth_mod.get_principal, None)
        else:
            app.dependency_overrides[auth_mod.get_principal] = saved


def _seed_operations() -> None:
    """写入两类操作：运维白名单内（备份/模型激活）+ 业务动作（对照，应被排除）。"""
    reg = get_registry().repository
    reg.append_audit(
        actor="sysadmin-01",
        action="backup_create",
        object_type="backup",
        object_id="ops_test_backup.zip",
        before=None,
        after={"sha256": "a" * 64, "entries": 3},
        note="C-18 回放测试备份",
    )
    reg.append_audit(
        actor="sysadmin-01",
        action="model_activate",
        object_type="model",
        object_id="ops_test_model",
        before=None,
        after={"model_id": "ops_test_model"},
        note=None,
    )
    reg.append_audit(
        actor="operator-01",
        action="inspect",
        object_type="image",
        object_id="ops_test_image",
        before=None,
        after={"defects": 2},
        note="业务动作不应出现在运维时间线",
    )


def test_operations_timeline_returns_only_operation_actions():
    _seed_operations()
    with _role_override("auditor") as client:
        resp = client.get("/api/v1/audit/operations?limit=200")
    assert resp.status_code == 200
    body = resp.json()
    assert body["actions"] == list(OPERATION_ACTIONS)
    ops = [o for o in body["operations"] if o["object_id"].startswith("ops_test")]
    actions = {o["action"] for o in ops}
    assert actions == {"backup_create", "model_activate"}
    assert all(o["action"] != "inspect" for o in body["operations"]), "业务动作不得混入运维时间线"

    backup = next(o for o in ops if o["action"] == "backup_create")
    assert backup["actor"] == "sysadmin-01"
    assert backup["created_at"]  # 时间要素
    assert backup["params"]["sha256"] == "a" * 64  # 参数摘要（after）
    assert backup["result"] == "C-18 回放测试备份"  # 结果（note）
    # note 为空时结果回退 None（字段仍存在，结构稳定可回放）
    model = next(o for o in ops if o["action"] == "model_activate")
    assert model["result"] is None
    assert model["params"]["model_id"] == "ops_test_model"
    # total 覆盖全库白名单动作（≥ 本测试写入的 2 条）
    assert body["total"] >= 2


def test_operations_auditor_only():
    """权限矩阵：运维操作回放仅审计员可读，sysadmin 403。"""
    _seed_operations()
    with TestClient(app) as client:  # conftest 默认 sysadmin 覆盖
        resp = client.get("/api/v1/audit/operations")
        assert resp.status_code == 403
    with _role_override("auditor") as client:
        assert client.get("/api/v1/audit/operations").status_code == 200
