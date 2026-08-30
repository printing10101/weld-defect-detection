"""分级保护合规端点（C-23~C-25）。

POST /compliance/self-check        分级保护五类自查（审计员/保密员）
POST /compliance/crypto-materials  密码应用自评估说明导出（保密员）
POST /compliance/hardening-check   安全加固自检（审计员/系统管理员）

全部为"活检查/活导出"端点：检查逻辑真实查询运行时状态（表、配置、守卫、
路由表），产物（JSON + PDF，密评材料为 Markdown + PDF）落 data/compliance/，
动作入主审计链 + 安全审计链。
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.app.auth import Principal, require_role
from backend.app.dependencies import Registry, get_registry

router = APIRouter(prefix="/compliance", tags=["compliance"])


def _compliance_dir(reg: Registry) -> Path:
    """合规产物目录（data/compliance/，锚定安装根目录）。"""
    out = Path(reg.config.paths.data_dir) / "compliance"
    if not out.is_absolute():
        from backend.infra.config import resolve_config_path

        out = resolve_config_path(str(out))
    out.mkdir(parents=True, exist_ok=True)
    return out


def _audit_action(
    reg: Registry,
    principal: Principal,
    action: str,
    after: dict[str, Any],
    note: str | None = None,
) -> None:
    """合规动作入主审计链 + 安全审计链（审计员/保密员自身操作留痕）。"""
    reg.repository.append_audit(
        actor=principal.username,
        action=action,
        object_type="compliance",
        object_id=action,
        before=None,
        after=after,
        note=note,
    )
    reg.security_store.append_security_audit(
        actor=principal.username,
        action=action,
        object_type="compliance",
        object_id=action,
        before=None,
        after=after,
        note=note,
    )


# ---------------------------------------------------------------------------
# C-23 分级保护自查
# ---------------------------------------------------------------------------


class SelfCheckOut(BaseModel):
    generated_at: str
    standard: str
    overall: str  # pass | fail | warning
    summary: dict[str, Any]
    categories: dict[str, list[dict[str, Any]]]
    files: dict[str, str]


@router.post("/self-check", response_model=SelfCheckOut)
def self_check(
    principal: Annotated[Principal, Depends(require_role("auditor", "secadmin"))],
    reg: Annotated[Registry, Depends(get_registry)],
) -> SelfCheckOut:
    """分级保护安全自查（C-23，安全审计员/安全保密管理员）。

    按五类（身份鉴别/访问控制/安全审计/边界防护/信息流转控制）逐项**活检查**：
    真实查询账号表、会话表、审计链、配置、守卫对象、影像目录，不硬编码结论。
    产物 JSON + PDF 落 data/compliance/，动作入双链审计。
    """
    from backend.infra.compliance.selfcheck import run_selfcheck, write_selfcheck

    report = run_selfcheck(reg)
    files = write_selfcheck(report, _compliance_dir(reg))
    _audit_action(
        reg,
        principal,
        "compliance_selfcheck",
        after={"overall": report["overall"], "files": files},
        note="C-23 分级保护自查",
    )
    return SelfCheckOut(files=files, **{k: v for k, v in report.items() if k in SelfCheckOut.model_fields})


# ---------------------------------------------------------------------------
# C-24 密评材料导出
# ---------------------------------------------------------------------------


class CryptoMaterialsOut(BaseModel):
    generated_at: str
    markdown: str  # Markdown 全文（便于前端预览/复制）
    files: dict[str, str]


@router.post("/crypto-materials", response_model=CryptoMaterialsOut)
def crypto_materials(
    principal: Annotated[Principal, Depends(require_role("secadmin"))],
    reg: Annotated[Registry, Depends(get_registry)],
) -> CryptoMaterialsOut:
    """导出《密码应用自评估说明》（C-24，安全保密管理员）。

    算法清单来自 crypto provider 运行时内省；密钥管理/密码调用链/合规差距
    按代码与配置如实陈述（差距供密评机构认定，不视为符合结论）。
    产物 Markdown + JSON + PDF 落 data/compliance/，动作入双链审计。
    """
    from backend.infra.compliance.crypto_materials import (
        build_crypto_materials,
        write_crypto_materials,
    )

    built = build_crypto_materials()
    files = write_crypto_materials(_compliance_dir(reg))
    _audit_action(
        reg,
        principal,
        "crypto_materials_export",
        after={"files": files},
        note="C-24 密评材料导出",
    )
    return CryptoMaterialsOut(
        generated_at=built["data"]["generated_at"], markdown=built["markdown"], files=files
    )


# ---------------------------------------------------------------------------
# C-25 安全加固自检
# ---------------------------------------------------------------------------


class HardeningOut(BaseModel):
    generated_at: str
    overall: str
    summary: dict[str, Any]
    findings: list[dict[str, Any]]
    high_findings: list[dict[str, Any]]
    files: dict[str, str]


@router.post("/hardening-check", response_model=HardeningOut)
def hardening_check(
    principal: Annotated[Principal, Depends(require_role("auditor", "sysadmin"))],
    reg: Annotated[Registry, Depends(get_registry)],
) -> HardeningOut:
    """安全加固自检（C-25，安全审计员/系统管理员）。

    检查：默认口令（引导窗口/未绑定公钥账号）、开放端口（配置 + psutil 实测
    本进程监听，未装 psutil 如实降级）、未授权接口（内省 app.routes，应为
    空集）、TLS/传输、文件权限。分级 high/medium/low，高危项附修复建议。
    """
    from backend.app.main import app as current_app
    from backend.infra.compliance.hardening import run_hardening_check, write_hardening_report

    report = run_hardening_check(reg, current_app)
    files = write_hardening_report(report, _compliance_dir(reg))
    _audit_action(
        reg,
        principal,
        "hardening_check",
        after={"overall": report["overall"], "high": len(report["high_findings"]), "files": files},
        note="C-25 安全加固自检",
    )
    return HardeningOut(files=files, **{k: v for k, v in report.items() if k in HardeningOut.model_fields})


# ---------------------------------------------------------------------------
# V-02~V-05 交付物生成（要求文档 §6 表 6-1）
# ---------------------------------------------------------------------------


class DeliverableOut(BaseModel):
    spec: str  # V-02 | V-03 | V-04 | V-05
    title: str
    generated_at: str
    overall: str
    files: dict[str, str]
    report: dict[str, Any]  # 全量报告（前端可直接展示汇总与缺口清单）


_BUILDERS = {
    "V-02": ("信创兼容性验证报告", "build_v02"),
    "V-03": ("稳定性测试报告", "build_v03"),
    "V-04": ("评价体系合规报告", "build_v04"),
    "V-05": ("边界声明", "build_v05"),
}


@router.post("/deliverables/{spec}", response_model=DeliverableOut)
def build_deliverable(
    spec: str,
    principal: Annotated[Principal, Depends(require_role("auditor", "secadmin"))],
    reg: Annotated[Registry, Depends(get_registry)],
) -> DeliverableOut:
    """交付物一键生成（V-02~V-05，安全审计员/安全保密管理员）。

    V-02 解析 docs/国产化适配矩阵.md；V-03 聚合 data/compliance 长跑与演练
    实测记录；V-04 聚合 data/eval 评价产物；V-05 固定边界文本。全部"活生成"，
    缺失证据如实列缺口（不预填通过）。产物 JSON + PDF/A 落 data/compliance/，
    动作入双链审计。
    """
    from backend.infra.compliance.deliverables import (
        build_v02,
        build_v03,
        build_v04,
        build_v05,
        write_deliverable,
    )

    key = spec.upper()
    if key not in _BUILDERS:
        raise HTTPException(status_code=404, detail=f"未知交付物编号：{spec}（V-02~V-05）")
    report = {"V-02": build_v02, "V-03": build_v03, "V-04": build_v04, "V-05": build_v05}[key](
        *(() if key in ("V-02", "V-05") else (reg,))
    )
    files = write_deliverable(report, _compliance_dir(reg))
    _audit_action(
        reg,
        principal,
        "deliverable_export",
        after={"spec": key, "overall": report.get("overall"), "files": files},
        note=f"{key} {report.get('title', '')}",
    )
    return DeliverableOut(
        spec=key,
        title=report.get("title", _BUILDERS[key][0]),
        generated_at=report.get("generated_at", ""),
        overall=report.get("overall", "—"),
        files=files,
        report=report,
    )
