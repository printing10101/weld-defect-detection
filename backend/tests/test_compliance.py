"""合规端点测试（C-23 分级保护自查 / C-24 密评材料 / C-25 安全加固自检）。

覆盖：三类端点的角色矩阵、报告结构（五类各≥3 项、分级/结论合法性）、
活检查证据非空、产物文件（JSON/Markdown/PDF）落盘、动作入审计、
未授权接口扫描器（正常 app 为空集 + 注入无鉴权路由可检出）。
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Iterator
from contextlib import contextmanager

from fastapi import APIRouter, Request
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


def _files_exist(files: dict[str, str]) -> bool:
    from pathlib import Path

    return all(Path(v).is_file() and Path(v).stat().st_size > 0 for v in files.values())


# ---------------------------------------------------------------------------
# C-23 分级保护自查
# ---------------------------------------------------------------------------


def test_self_check_structure_and_files():
    """五类各≥3 项；每项含名称/依据/结论/证据；产物 JSON+PDF 落盘；动作入审计。"""
    reg = get_registry()
    before, _ = reg.repository.list_audit(action="compliance_selfcheck", limit=1)
    with principal_role("auditor", username="审计员"), TestClient(app) as c:
        resp = c.post("/api/v1/compliance/self-check")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body["categories"]) == {
        "身份鉴别",
        "访问控制",
        "安全审计",
        "边界防护",
        "信息流转控制",
    }
    results = set()
    for items in body["categories"].values():
        assert len(items) >= 3, "每类至少 3 项真实检查"
        for it in items:
            assert it["name"] and it["basis"] and it["evidence"]
            assert it["result"] in {"pass", "fail", "warning"}
            results.add(it["result"])
    assert body["overall"] in {"pass", "fail", "warning"}
    assert body["summary"]["total"] == sum(len(v) for v in body["categories"].values())
    assert _files_exist(body["files"])
    assert body["files"]["json"].endswith(".json") and body["files"]["pdf"].endswith(".pdf")
    # 落盘 JSON 与响应一致（摘要级断言）
    saved = json.loads(pathlib.Path(body["files"]["json"]).read_text(encoding="utf-8"))
    assert saved["overall"] == body["overall"]
    # 动作入主审计链
    _after, total = reg.repository.list_audit(action="compliance_selfcheck", limit=10)
    assert total > len(before)
    assert results, "检查项非空"


def test_self_check_role_matrix():
    """self-check 仅审计员/保密员：sysadmin 403；secadmin 200。"""
    with principal_role("sysadmin", username="系统管理员"), TestClient(app) as c:
        assert c.post("/api/v1/compliance/self-check").status_code == 403
    with principal_role("secadmin", username="保密员"), TestClient(app) as c:
        assert c.post("/api/v1/compliance/self-check").status_code == 200


def test_self_check_audit_category_reports_chain_state():
    """活检查：安全审计类结论与 verify_chain 真实状态一致（不硬编码 pass）。"""
    reg = get_registry()
    with principal_role("auditor", username="审计员"), TestClient(app) as c:
        body = c.post("/api/v1/compliance/self-check").json()
    audit_items = {i["name"]: i for i in body["categories"]["安全审计"]}
    main_item = next(v for k, v in audit_items.items() if "主审计链" in k)
    assert str(reg.repository.verify_chain()) in main_item["evidence"]
    assert main_item["result"] == ("pass" if reg.repository.verify_chain() else "fail")
    # 归档导出行数核对（活检查：构造的 JSONL 行数与链内条数一致）
    archive_item = next(v for k, v in audit_items.items() if "归档导出" in k)
    assert archive_item["result"] == "pass"


# ---------------------------------------------------------------------------
# C-24 密评材料
# ---------------------------------------------------------------------------


def test_crypto_materials_content_and_files():
    """算法清单含 SM4/SM3/SM2/AES；差距声明存在；MD+PDF 落盘；动作入审计。"""
    reg = get_registry()
    before, _ = reg.repository.list_audit(action="crypto_materials_export", limit=1)
    with principal_role("secadmin", username="保密员"), TestClient(app) as c:
        resp = c.post("/api/v1/compliance/crypto-materials")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    md = body["markdown"]
    for token in (
        "SM4-CTR",
        "HMAC-SM3",
        "SM2",
        "SM3",
        "AES-256-GCM",
        "合规差距声明",
        "密钥管理",
        "密码调用链",
    ):
        assert token in md, f"缺少章节/算法: {token}"
    assert "SCAN_CRYPTO_KEY" in md and "Pkcs11Provider" in md
    assert _files_exist({k: v for k, v in body["files"].items() if k in ("markdown", "pdf")})
    assert body["files"]["markdown"].endswith(".md") and body["files"]["pdf"].endswith(".pdf")
    _after, total = reg.repository.list_audit(action="crypto_materials_export", limit=10)
    assert total > len(before)


def test_crypto_materials_secadmin_only():
    """密评材料仅保密员：auditor 403。"""
    with principal_role("auditor", username="审计员"), TestClient(app) as c:
        assert c.post("/api/v1/compliance/crypto-materials").status_code == 403


# ---------------------------------------------------------------------------
# C-25 安全加固自检
# ---------------------------------------------------------------------------


def test_hardening_check_structure_and_files():
    """检查项分级合法、未授权接口扫描为 pass（空集）、高危项列表、产物落盘。"""
    reg = get_registry()
    before, _ = reg.repository.list_audit(action="hardening_check", limit=1)
    with principal_role("auditor", username="审计员"), TestClient(app) as c:
        resp = c.post("/api/v1/compliance/hardening-check")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["findings"], "检查项非空"
    for f in body["findings"]:
        assert f["severity"] in {"high", "medium", "low"}
        assert f["result"] in {"pass", "fail", "warning"}
        assert f["evidence"]
    names = {f["name"] for f in body["findings"]}
    for expected in ("默认口令/引导窗口", "监听地址仅本机", "未授权接口扫描", "传输加密（TLS）"):
        assert expected in names
    scan = next(f for f in body["findings"] if f["name"] == "未授权接口扫描")
    assert scan["result"] == "pass"  # 全部路由已挂鉴权依赖
    assert isinstance(body["high_findings"], list)
    assert body["overall"] in {"pass", "fail", "warning"}
    assert _files_exist(body["files"])
    _after, total = reg.repository.list_audit(action="hardening_check", limit=10)
    assert total > len(before)


def test_hardening_check_role_matrix():
    """hardening 仅审计员/系统管理员：secadmin 403；sysadmin 200。"""
    with principal_role("secadmin", username="保密员"), TestClient(app) as c:
        assert c.post("/api/v1/compliance/hardening-check").status_code == 403
    with principal_role("sysadmin", username="系统管理员"), TestClient(app) as c:
        assert c.post("/api/v1/compliance/hardening-check").status_code == 200


def test_unauth_route_scanner_detects_injected_route():
    """扫描器自证：正常 app 空集；注入无鉴权路由可被检出（活检查不造假）。"""
    from backend.app.auth import get_principal
    from backend.infra.compliance.hardening import unauthenticated_api_routes

    assert unauthenticated_api_routes(app, get_principal) == []

    rogue = APIRouter()

    @rogue.post("/rogue-open")
    def rogue_open():
        return {"ok": True}

    from fastapi import FastAPI

    test_app = FastAPI()
    test_app.include_router(rogue, prefix="/api/v1")
    hits = unauthenticated_api_routes(test_app, get_principal)
    assert hits == ["POST /api/v1/rogue-open"]
