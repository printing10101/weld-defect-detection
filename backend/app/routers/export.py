"""导出管控（C-14）：导出审批流与下载门禁。

流程：POST /export/requests（申请，任意已登录角色）→ POST .../approve|reject
（安全保密员审批）→ POST .../token（签发一次性导出令牌，明文仅返回一次）→
凭 ``X-Export-Token`` 头访问受控导出端点（如 GET /report/{id}/pdf）。

受控端点统一经 ensure_export_allowed 门禁：
- export.require_approval=false（单机调试）：登录即可导出；
- true（默认）：安全保密管理员（审批人）本人可直接导出（预授权语义），
  其余角色必须持有效一次性令牌（subject 必须匹配）；
- 全部申请/批准/拒绝/下载/拒绝导出动作入主审计链；批准/拒绝入独立安全
  审计链（C-19）。

USB/外设管控属 OS 层基线，见 docs/deployment-hardening.md 的部署要求。
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from backend.app.auth import Principal, get_principal, require_role
from backend.app.dependencies import Registry, get_registry
from backend.infra.crypto import sm3_hex

router = APIRouter(prefix="/export", tags=["export"])


def _err(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


class ExportRequestIn(BaseModel):
    subject: str = Field(min_length=1, max_length=128)  # 如 report:<report_id>
    reason: str | None = None


class ExportRequestOut(BaseModel):
    request_id: str
    subject: str
    reason: str | None
    requested_by: str
    status: str
    decided_by: str | None
    decided_at: str | None
    token_expires_at: str | None
    used_at: str | None
    created_at: str | None


class DecisionIn(BaseModel):
    note: str | None = None


def _row_out(row: dict[str, Any]) -> ExportRequestOut:
    return ExportRequestOut(**row)


@router.post("/requests", response_model=ExportRequestOut)
def create_export_request(
    body: ExportRequestIn,
    principal: Annotated[Principal, Depends(get_principal)],
    reg: Annotated[Registry, Depends(get_registry)],
) -> ExportRequestOut:
    """申请导出（任意已登录角色；留痕）。"""
    try:
        row = reg.export_store.create_request(
            subject=body.subject, requested_by=principal.username, reason=body.reason
        )
    except ValueError as exc:
        raise _err(422, "INVALID_REQUEST", str(exc)) from None
    reg.repository.append_audit(
        actor=principal.username,
        action="export_request_create",
        object_type="export_request",
        object_id=row["request_id"],
        before=None,
        after={"subject": body.subject},
        note=body.reason,
    )
    return _row_out(row)


@router.get("/requests", response_model=list[ExportRequestOut])
def list_export_requests(
    _: Annotated[Principal, Depends(require_role("secadmin"))],
    reg: Annotated[Registry, Depends(get_registry)],
    status: str | None = None,
) -> list[ExportRequestOut]:
    """审批队列（安全保密管理员）。"""
    return [_row_out(r) for r in reg.export_store.list(status=status)]


@router.get("/requests/{request_id}", response_model=ExportRequestOut)
def get_export_request(
    request_id: str,
    principal: Annotated[Principal, Depends(get_principal)],
    reg: Annotated[Registry, Depends(get_registry)],
) -> ExportRequestOut:
    """查询申请状态（申请人本人或保密员）。"""
    row = reg.export_store.get(request_id)
    if row is None:
        raise _err(404, "NOT_FOUND", f"export request not found: {request_id}")
    if principal.role != "secadmin" and row["requested_by"] != principal.username:
        raise _err(403, "FORBIDDEN", "仅申请人本人或安全保密管理员可查询")
    return _row_out(row)


@router.post("/requests/{request_id}/approve", response_model=ExportRequestOut)
def approve_request(
    request_id: str,
    principal: Annotated[Principal, Depends(require_role("secadmin"))],
    reg: Annotated[Registry, Depends(get_registry)],
    body: DecisionIn | None = None,
) -> ExportRequestOut:
    """批准导出（仅安全保密管理员；入安全审计链）。"""
    try:
        row = reg.export_store.decide(request_id, decided_by=principal.username, approved=True)
    except KeyError as exc:
        raise _err(404, "NOT_FOUND", str(exc)) from None
    except ValueError as exc:
        raise _err(409, "INVALID_STATE", str(exc)) from None
    _decision_audit(reg, principal.username, "export_request_approve", row, body)
    return _row_out(row)


@router.post("/requests/{request_id}/reject", response_model=ExportRequestOut)
def reject_request(
    request_id: str,
    principal: Annotated[Principal, Depends(require_role("secadmin"))],
    reg: Annotated[Registry, Depends(get_registry)],
    body: DecisionIn | None = None,
) -> ExportRequestOut:
    """拒绝导出（仅安全保密管理员；入安全审计链）。"""
    try:
        row = reg.export_store.decide(request_id, decided_by=principal.username, approved=False)
    except KeyError as exc:
        raise _err(404, "NOT_FOUND", str(exc)) from None
    except ValueError as exc:
        raise _err(409, "INVALID_STATE", str(exc)) from None
    _decision_audit(reg, principal.username, "export_request_reject", row, body)
    return _row_out(row)


def _decision_audit(
    reg: Registry, actor: str, action: str, row: dict[str, Any], body: DecisionIn | None
) -> None:
    note = body.note if body else None
    reg.repository.append_audit(
        actor=actor,
        action=action,
        object_type="export_request",
        object_id=row["request_id"],
        before={"status": "pending"},
        after={"status": row["status"]},
        note=note,
    )
    reg.security_store.append_security_audit(
        actor=actor,
        action=action,
        object_type="export_request",
        object_id=row["request_id"],
        before={"subject": row["subject"]},
        after={"status": row["status"]},
        note=note,
    )


@router.post("/requests/{request_id}/token")
def issue_export_token(
    request_id: str,
    principal: Annotated[Principal, Depends(get_principal)],
    reg: Annotated[Registry, Depends(get_registry)],
) -> dict[str, Any]:
    """为已批准的申请签发一次性导出令牌（明文仅本次返回，库存 SM3 哈希）。"""
    row = reg.export_store.get(request_id)
    if row is None:
        raise _err(404, "NOT_FOUND", f"export request not found: {request_id}")
    if row["requested_by"] != principal.username and principal.role != "secadmin":
        raise _err(403, "FORBIDDEN", "仅申请人本人或安全保密管理员可领取令牌")
    token = secrets.token_urlsafe(32)
    try:
        reg.export_store.issue_token(
            request_id,
            token_hash=sm3_hex(token.encode()),
            ttl_sec=reg.config.export.token_ttl_sec,
        )
    except ValueError as exc:
        raise _err(409, "INVALID_STATE", str(exc)) from None
    reg.repository.append_audit(
        actor=principal.username,
        action="export_token_issue",
        object_type="export_request",
        object_id=request_id,
        before=None,
        after={"subject": row["subject"]},
        note=None,
    )
    return {"token": token, "expires_in_sec": reg.config.export.token_ttl_sec}


def _fmt_window(dt: datetime) -> str:
    """与审计链 created_at 序列化格式（%Y-%m-%d %H:%M:%S）对齐，便于字符串比较。"""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _batch_export_alert(reg: Registry, principal: Principal) -> None:
    """异常行为告警（C-22）：窗口期内同一操作者导出下载次数达阈值 → high 告警。

    计数来源为主审计链 action=export_download 的持久记录（含历史进程）；
    仅在"恰好跨越阈值"的那一刻告警一次，后续导出不再重复告警（防刷屏）。
    告警失败不掩盖已成功的下载语义。
    """
    cfg = reg.config.alerts.batch_export
    if not cfg.enabled:
        return
    window_start = _fmt_window(datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=cfg.window_min))
    entries, _total = reg.repository.list_audit(action="export_download", limit=500)
    recent = [
        e
        for e in entries
        if e["actor"] == principal.username and (e["created_at"] or "") >= window_start
    ]
    if len(recent) != cfg.threshold:  # 未达阈值，或已越过（只告警跨越那一次）
        return
    try:
        reg.security_store.raise_alert(
            kind="batch_export",
            level="high",
            message=(
                f"批量导出告警：{principal.username} 在 {cfg.window_min} 分钟内"
                f"导出下载 {len(recent)} 次（阈值 {cfg.threshold}）"
            ),
            detail={
                "actor": principal.username,
                "window_min": cfg.window_min,
                "count": len(recent),
                "threshold": cfg.threshold,
            },
        )
        reg.security_store.append_security_audit(
            actor="system",
            action="alert_raised",
            object_type="alert",
            object_id="batch_export",
            before=None,
            after={"actor": principal.username, "count": len(recent)},
            note="C-22 批量导出异常行为告警",
        )
    except Exception as exc:  # noqa: BLE001 - 告警失败不影响导出本身
        import logging

        logging.getLogger("scandetection.export").warning("批量导出告警落库失败: %s", exc)


def ensure_export_allowed(
    subject: str,
    request: Request,
    principal: Principal,
    reg: Registry,
) -> None:
    """受控导出端点统一门禁（C-14）。不通过抛 HTTPException；通过则留痕下载。

    - require_approval=false：登录即可（调试模式，仅留痕）；
    - 安全保密管理员：审批人预授权，直接放行；
    - 其他角色：须持有效一次性令牌（X-Export-Token 头，subject 匹配、一次一用）。
    """
    if not reg.config.export.require_approval:
        reg.repository.append_audit(
            actor=principal.username,
            action="export_download",
            object_type="export",
            object_id=subject,
            before=None,
            after={"mode": "no_approval"},
            note=None,
        )
        _batch_export_alert(reg, principal)
        return
    if principal.role == "secadmin":
        reg.repository.append_audit(
            actor=principal.username,
            action="export_download",
            object_type="export",
            object_id=subject,
            before=None,
            after={"mode": "secadmin"},
            note=None,
        )
        _batch_export_alert(reg, principal)
        return
    token = (request.headers.get("X-Export-Token") or "").strip()
    if not token:
        reg.repository.append_audit(
            actor=principal.username,
            action="export_denied",
            object_type="export",
            object_id=subject,
            before=None,
            after={"reason": "token_missing"},
            note=None,
        )
        raise _err(
            401,
            "EXPORT_TOKEN_REQUIRED",
            "导出需审批：先 POST /export/requests 申请并由安全保密管理员批准",
        )
    row = reg.export_store.consume_token(sm3_hex(token.encode()))
    if row is None or row["subject"] != subject:
        reg.repository.append_audit(
            actor=principal.username,
            action="export_denied",
            object_type="export",
            object_id=subject,
            before=None,
            after={"reason": "token_invalid"},
            note=None,
        )
        raise _err(401, "EXPORT_TOKEN_INVALID", "导出令牌无效、已使用或与导出对象不符")
    reg.repository.append_audit(
        actor=principal.username,
        action="export_download",
        object_type="export",
        object_id=subject,
        before=None,
        after={"mode": "token", "request_id": row["request_id"]},
        note=None,
    )
    _batch_export_alert(reg, principal)
