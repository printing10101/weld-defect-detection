"""密级标识测试（C-10）：设定/变更密级 API、权限、双链审计、PDF 嵌入。"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import Iterator

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from backend.app import auth as auth_mod
from backend.app.dependencies import get_registry
from backend.app.main import app
from backend.infra.reporting.pdf_reporter import classification_label


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


def _make_image() -> str:
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
    return image_id


def test_classification_label_mapping():
    """C-10 密级枚举语义：0=非密（不绘制横标）1=内部 2=秘密 3=机密。"""
    assert classification_label(0) == ""
    assert classification_label(1) == "内部"
    assert classification_label(2) == "秘密"
    assert classification_label(3) == "机密"


def test_set_and_get_secret_level_secadmin():
    image_id = _make_image()
    with principal_role("secadmin"):
        with TestClient(app) as c:
            resp = c.post(
                f"/api/v1/classification/image/{image_id}",
                json={"secret_level": 2, "classification_basis": "某某密级目录第3条"},
            )
            assert resp.status_code == 200, resp.text
            assert resp.json()["secret_level_name"] == "秘密"
            got = c.get(f"/api/v1/classification/image/{image_id}")
            assert got.status_code == 200
            assert got.json()["secret_level"] == 2
            assert got.json()["classification_basis"] == "某某密级目录第3条"
    # 密级变更入主审计链 + 独立安全审计链（C-19 双链）
    reg = get_registry()
    entries, _ = reg.repository.list_audit(action="secret_level_change", limit=10)
    assert any(e["object_id"] == image_id for e in entries)
    sec_entries, _ = reg.security_store.list_security_audit(
        action="secret_level_change", limit=10
    )
    assert any(e["object_id"] == image_id for e in sec_entries)


def test_set_secret_level_requires_secadmin():
    """C-06 权限矩阵：密级变更仅安全保密管理员（系统管理员/审计员 403）。"""
    image_id = _make_image()
    for role in ("sysadmin", "auditor"):
        with principal_role(role):
            with TestClient(app) as c:
                resp = c.post(
                    f"/api/v1/classification/image/{image_id}",
                    json={"secret_level": 1, "classification_basis": "测试"},
                )
                assert resp.status_code == 403, role


def test_set_secret_level_requires_basis():
    """变更密级必须登记定密依据（缺依据 422）。"""
    image_id = _make_image()
    with principal_role("secadmin"):
        with TestClient(app) as c:
            resp = c.post(
                f"/api/v1/classification/image/{image_id}",
                json={"secret_level": 1, "classification_basis": "  "},
            )
            assert resp.status_code == 422


def test_set_secret_level_invalid_value():
    image_id = _make_image()
    with principal_role("secadmin"):
        with TestClient(app) as c:
            resp = c.post(
                f"/api/v1/classification/image/{image_id}",
                json={"secret_level": 5, "classification_basis": "测试"},
            )
            assert resp.status_code == 422  # pydantic ge/le 拦截


def test_report_pdf_embeds_classification():
    """C-10：报告行同步密级；密级随影像快照进入报告内容（PDF 嵌入数据源）。"""
    from backend.domain.report.content import build_report_content

    image_id = _make_image()
    reg = get_registry()
    with principal_role("secadmin"):
        with TestClient(app) as c:
            resp = c.post(
                f"/api/v1/classification/image/{image_id}",
                json={"secret_level": 3, "classification_basis": "机密依据条款"},
            )
            assert resp.status_code == 200
    image = reg.repository.get_image(image_id)
    content = build_report_content(image, [], image.get("report"))
    assert content.secret_level == 3
    assert content.classification_basis == "机密依据条款"
    # 密级纳入报告防篡改指纹
    from backend.infra.reporting.pdf_reporter import report_fingerprint

    fp = report_fingerprint(image, [], image.get("report"))
    image2 = {**image, "secret_level": 0}
    assert report_fingerprint(image2, [], image.get("report")) != fp


def test_levels_endpoint_readonly():
    with TestClient(app) as c:
        resp = c.get("/api/v1/classification/levels")
        assert resp.status_code == 200
        names = {lv["level"]: lv["name"] for lv in resp.json()["levels"]}
        assert names == {0: "非密", 1: "内部", 2: "秘密", 3: "机密"}
