"""不可变审计日志查询。

GET /api/v1/audit：按 actor/action/object_type/object_id/limit 过滤返回审计链
（谁/何时/对何对象/前后值/哈希链）。写入由业务动作（复核、报告重生成等）
经 repository.append_audit 完成，本端点只提供只读检索。

GET /api/v1/audit/export（C-20）：把主链 + 安全链全量导出为只追加 JSONL
（每行 = 记录 + 链校验状态），导出动作本身入双链审计。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Response
from pydantic import BaseModel

from backend.app.auth import Principal, require_role
from backend.app.dependencies import Registry, get_registry

router = APIRouter(tags=["audit"])


class AuditEntry(BaseModel):
    seq: int
    actor: str
    action: str
    object_type: str
    object_id: str
    before: object | None = None
    after: object | None = None
    note: str | None = None
    prev_hash: str
    hash: str
    created_at: str | None


class AuditResponse(BaseModel):
    entries: list[AuditEntry]
    total: int
    chain_valid: bool


@router.get("/audit", response_model=AuditResponse)
def audit(
    reg: Annotated[Registry, Depends(get_registry)],
    actor: Annotated[str | None, Query()] = None,
    action: Annotated[str | None, Query()] = None,
    object_type: Annotated[str | None, Query()] = None,
    object_id: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> AuditResponse:
    # total 取匹配总数（原实现用 len(entries)，超过 limit 时低报，审计场景不可接受）
    entries, total = reg.repository.list_audit(
        actor=actor,
        action=action,
        object_type=object_type,
        object_id=object_id,
        limit=limit,
        offset=offset,
    )
    return AuditResponse(
        entries=[AuditEntry(**e) for e in entries],
        total=total,
        chain_valid=reg.repository.verify_chain(),
    )


# ---------------------------------------------------------------------------
# 独立安全审计链（C-19 双链）：安全审计员只读主链 + 安全链。
# ---------------------------------------------------------------------------


class SecurityAuditResponse(BaseModel):
    entries: list[AuditEntry]
    total: int
    chain_valid: bool


@router.get("/audit/security", response_model=SecurityAuditResponse)
def security_audit(
    _: Annotated[object, Depends(require_role("auditor"))],
    reg: Annotated[Registry, Depends(get_registry)],
    actor: Annotated[str | None, Query()] = None,
    action: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> SecurityAuditResponse:
    """安全审计链检索（仅安全审计员，只读）。"""
    entries, total = reg.security_store.list_security_audit(
        actor=actor, action=action, limit=limit, offset=offset
    )
    return SecurityAuditResponse(
        entries=[AuditEntry(**e) for e in entries],
        total=total,
        chain_valid=reg.security_store.verify_security_chain(),
    )


# ---------------------------------------------------------------------------
# 运维操作清单 + 回放（C-18）：主审计链中"系统管理类"操作的结构化时间线。
# 当前无远程运维通道；本端点满足"可审计可回放"的软件侧能力——
# 远程维护须经堡垒机且全程录屏（物理/流程基线见 docs/deployment-baseline.md）。
# ---------------------------------------------------------------------------

# 系统管理类操作白名单（运维时间线只呈现这些动作，业务评片/复核不混入）
OPERATION_ACTIONS: tuple[str, ...] = (
    "backup_create",  # 备份
    "backup_restore",  # 恢复
    "model_activate",  # 模型激活（检测器热切换）
    "gate_reject",  # 质量门禁拦截留档
    "secret_level_change",  # 密级变更
    "account_lock",  # 账号锁定
    "account_unlock",  # 账号解锁
    "alert_resolve",  # 告警处置
    "egress_blocked",  # 外联拦截（C-16）
    "export_request_create",  # 导出申请
    "export_token_issue",  # 导出令牌签发
    "export_download",  # 导出下载
    "export_denied",  # 导出拒绝
    "carrier_certificate_export",  # 载体证书导出
    "borrow",  # 载体借用
    "return",  # 载体归还
    "destroy_request",  # 载体销毁发起
    "destroy_confirm",  # 载体销毁确认
    "audit_export",  # 审计归档导出（C-20）
    "model_activate_failed",  # 模型热切换失败（C-21）
    "device_register",  # 设备登记（C-21）
    "device_calibrate",  # 设备标定（C-21）
    "compliance_selfcheck",  # 分级保护自查（C-23）
    "crypto_materials_export",  # 密评材料导出（C-24）
    "hardening_check",  # 安全加固自检（C-25）
    "deliverable_export",  # 交付物生成（V-02~V-05）
)


class OperationEntry(BaseModel):
    """运维操作时间线条目（时间/actor/动作/参数摘要/结果，可回放）。"""

    seq: int
    created_at: str | None  # 时间
    actor: str  # 操作者
    action: str  # 动作
    object_type: str
    object_id: str
    params: object | None  # 参数摘要（after 优先，回退 before）
    result: str | None  # 结果（审计 note）


class OperationsResponse(BaseModel):
    operations: list[OperationEntry]
    total: int
    actions: list[str]  # 时间线覆盖的动作清单（调用方可据此过滤）


@router.get("/audit/operations", response_model=OperationsResponse)
def audit_operations(
    _: Annotated[object, Depends(require_role("auditor"))],
    reg: Annotated[Registry, Depends(get_registry)],
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> OperationsResponse:
    """系统管理类操作时间线（仅安全审计员，只读）。

    从主审计链（SM3 哈希链）按运维动作白名单过滤，返回结构化时间线：
    谁（actor）/ 何时（created_at）/ 做了什么（action + params 摘要）/
    结果如何（note）——满足"运维操作可审计、可回放"的软件侧能力。
    """
    entries, total = reg.repository.list_audit(
        actions=list(OPERATION_ACTIONS), limit=limit, offset=offset
    )
    operations = [
        OperationEntry(
            seq=e["seq"],
            created_at=e["created_at"],
            actor=e["actor"],
            action=e["action"],
            object_type=e["object_type"],
            object_id=e["object_id"],
            params=e["after"] if e["after"] is not None else e["before"],
            result=e["note"],
        )
        for e in entries
    ]
    return OperationsResponse(
        operations=operations,
        total=total,
        actions=list(OPERATION_ACTIONS),
    )


# ---------------------------------------------------------------------------
# 审计归档导出（C-20）：主链 + 安全链全量只追加 JSONL 归档。
# ---------------------------------------------------------------------------

# list_audit/list_security_audit 单页上限 500；全量导出按 offset 翻页取整链。
_EXPORT_PAGE = 500


def _iter_full_chain(list_fn, total: int) -> Iterator[dict[str, Any]]:
    """按页取完整审计链（列表接口按 seq 降序，这里还原为 seq 升序输出）。"""
    fetched: list[dict[str, Any]] = []
    offset = 0
    while offset < total:
        page, _ = list_fn(limit=_EXPORT_PAGE, offset=offset)
        if not page:
            break
        fetched.extend(page)
        offset += len(page)
    fetched.reverse()  # 降序 → 升序（归档件按链序只追加）
    yield from fetched


def build_audit_export(reg: Registry, actor: str) -> tuple[str, dict[str, Any]]:
    """构造审计归档 JSONL（C-20）。

    行格式（每行一个 JSON 对象，只追加语义、按链序排列）：
    - 首行 header：导出元数据（时间/操作者/两条链的整链校验结论/总条数）；
    - 记录行：{"type":"record","chain":"main"|"security","seq":n,
      "record":{...},"chain_valid":<该链整链校验结论>}；
    - 末行 footer：两条链的条数与校验结论汇总（供归档校验程序核对）。
    返回 (jsonl 文本, footer dict)。
    """
    main_valid = reg.repository.verify_chain()
    sec_valid = reg.security_store.verify_security_chain()
    main_entries, main_total = reg.repository.list_audit(limit=_EXPORT_PAGE, offset=0)
    del main_entries
    sec_entries, sec_total = reg.security_store.list_security_audit(limit=_EXPORT_PAGE, offset=0)
    del sec_entries

    now = datetime.now(UTC).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = [
        json.dumps(
            {
                "type": "export_header",
                "format": "scandetection-audit-export/1",
                "exported_at": now,
                "exported_by": actor,
                "main_chain_total": main_total,
                "security_chain_total": sec_total,
                "main_chain_valid": main_valid,
                "security_chain_valid": sec_valid,
            },
            ensure_ascii=False,
        )
    ]
    for chain, entries, valid in (
        ("main", _iter_full_chain(reg.repository.list_audit, main_total), main_valid),
        (
            "security",
            _iter_full_chain(reg.security_store.list_security_audit, sec_total),
            sec_valid,
        ),
    ):
        for e in entries:
            lines.append(
                json.dumps(
                    {
                        "type": "record",
                        "chain": chain,
                        "seq": e["seq"],
                        "record": e,
                        "chain_valid": valid,
                    },
                    ensure_ascii=False,
                )
            )
    footer = {
        "type": "export_footer",
        "main_chain_total": main_total,
        "security_chain_total": sec_total,
        "main_chain_valid": main_valid,
        "security_chain_valid": sec_valid,
        "exported_at": now,
    }
    lines.append(json.dumps(footer, ensure_ascii=False))
    return "\n".join(lines) + "\n", footer


@router.get("/audit/export")
def audit_export(
    principal: Annotated[Principal, Depends(require_role("auditor"))],
    reg: Annotated[Registry, Depends(get_registry)],
) -> Response:
    """审计归档导出（C-20，仅安全审计员）：主链 + 安全链全量 JSONL。

    每行含完整记录与所在链的整链校验状态（chain_valid），归档件可离线
    逐行核对。导出动作本身入主审计链 + 安全审计链（审计员自身操作留痕，
    C-19/C-21）。链校验状态是导出时刻的真实快照——导出后链上新增的记录
    不在本归档件中（只追加语义）。
    """
    body, footer = build_audit_export(reg, principal.username)
    reg.repository.append_audit(
        actor=principal.username,
        action="audit_export",
        object_type="audit",
        object_id=f"main:{footer['main_chain_total']}+security:{footer['security_chain_total']}",
        before=None,
        after={
            "main_chain_total": footer["main_chain_total"],
            "security_chain_total": footer["security_chain_total"],
            "main_chain_valid": footer["main_chain_valid"],
            "security_chain_valid": footer["security_chain_valid"],
        },
        note="C-20 审计归档导出",
    )
    reg.security_store.append_security_audit(
        actor=principal.username,
        action="audit_export",
        object_type="audit",
        object_id="archive",
        before=None,
        after={"records": footer["main_chain_total"] + footer["security_chain_total"]},
        note="C-20 审计归档导出",
    )
    ts = datetime.now(UTC).replace(tzinfo=None).strftime("%Y%m%d_%H%M%S")
    return Response(
        content=body,
        media_type="application/x-ndjson",
        headers={"Content-Disposition": f'attachment; filename="audit_export_{ts}.jsonl"'},
    )
