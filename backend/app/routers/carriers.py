"""涉密载体台账（C-12）：登记/借用/归还/销毁（双确认）+ 销毁证明导出。

权限语义（C-06 权限矩阵）：
- 登记（register）：安全保密管理员；
- 借用/归还：任意已登录角色（责任人字段记录操作人）；
- 销毁：两段式双确认——安全保密管理员发起（记录销毁方式）→ 系统管理员确认；
  与发起人必须分属两个账号（同账号自确认 409 拒绝）。
全部动作入主审计链；登记与销毁另入独立安全审计链（C-19）。
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.app.auth import Principal, get_principal, require_role
from backend.app.dependencies import Registry, get_registry
from backend.infra.reporting.certificates import build_destroy_certificate

router = APIRouter(prefix="/carriers", tags=["carriers"])


def _err(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


class CarrierIn(BaseModel):
    carrier_id: str = Field(min_length=1, max_length=64)  # 载体编号（如 CN-2026-0001）
    kind: str  # film | report | backup
    object_id: str | None = None
    secret_level: int = Field(default=0, ge=0, le=3)
    owner: str | None = None


class CarrierOut(BaseModel):
    carrier_id: str
    kind: str
    object_id: str | None
    secret_level: int
    owner: str | None
    status: str
    borrow_history: list[dict[str, Any]]
    destroy_method: str | None
    destroy_note: str | None
    destroy_requested_by: str | None
    destroy_confirmed_by: str | None
    destroyed_at: str | None
    created_at: str | None


class NoteIn(BaseModel):
    note: str | None = None


class DestroyRequestIn(NoteIn):
    destroy_method: str = Field(min_length=1, max_length=64)  # 碎纸/消磁/焚烧...


def _audit(
    reg: Registry, principal: Principal, action: str, carrier: dict, note: str | None
) -> None:
    """载体动作留痕：主链全记；登记/销毁类关键动作另入安全审计链（C-19）。"""
    reg.repository.append_audit(
        actor=principal.username,
        action=action,
        object_type="carrier",
        object_id=str(carrier["carrier_id"]),
        before=None,
        after={"status": carrier["status"], "kind": carrier["kind"]},
        note=note,
    )
    if action in ("carrier_register", "carrier_destroy"):
        reg.security_store.append_security_audit(
            actor=principal.username,
            action=action,
            object_type="carrier",
            object_id=str(carrier["carrier_id"]),
            before=None,
            after={"status": carrier["status"]},
            note=note,
        )


def _carrier_or_404(reg: Registry, carrier_id: str) -> dict[str, Any]:
    c = reg.carrier_store.get(carrier_id)
    if c is None:
        raise _err(404, "NOT_FOUND", f"carrier not found: {carrier_id}")
    return c


@router.post("", response_model=CarrierOut)
def register_carrier(
    body: CarrierIn,
    principal: Annotated[Principal, Depends(require_role("secadmin"))],
    reg: Annotated[Registry, Depends(get_registry)],
) -> CarrierOut:
    """登记载体（仅安全保密管理员；入双链）。"""
    try:
        c = reg.carrier_store.register(
            carrier_id=body.carrier_id,
            kind=body.kind,
            object_id=body.object_id,
            secret_level=body.secret_level,
            owner=body.owner or principal.username,
        )
    except ValueError as exc:
        raise _err(422, "INVALID_CARRIER", str(exc)) from None
    _audit(reg, principal, "carrier_register", c, None)
    return CarrierOut(**c)


@router.get("", response_model=list[CarrierOut])
def list_carriers(
    reg: Annotated[Registry, Depends(get_registry)],
    status: str | None = None,
    kind: str | None = None,
) -> list[CarrierOut]:
    """载体台账查询（任意已登录角色）。"""
    return [CarrierOut(**c) for c in reg.carrier_store.list(status=status, kind=kind)]


@router.get("/{carrier_id}", response_model=CarrierOut)
def get_carrier(
    carrier_id: str,
    reg: Annotated[Registry, Depends(get_registry)],
) -> CarrierOut:
    return CarrierOut(**_carrier_or_404(reg, carrier_id))


@router.post("/{carrier_id}/borrow", response_model=CarrierOut)
def borrow_carrier(
    carrier_id: str,
    principal: Annotated[Principal, Depends(get_principal)],
    reg: Annotated[Registry, Depends(get_registry)],
    body: NoteIn | None = None,
) -> CarrierOut:
    """借用（在库/已归还 → 借出）。"""
    try:
        c = reg.carrier_store.borrow(
            carrier_id, operator=principal.username, note=body.note if body else None
        )
    except KeyError as exc:
        raise _err(404, "NOT_FOUND", str(exc)) from None
    except ValueError as exc:
        raise _err(409, "INVALID_STATE", str(exc)) from None
    _audit(reg, principal, "carrier_borrow", c, None)
    return CarrierOut(**c)


@router.post("/{carrier_id}/return", response_model=CarrierOut)
def return_carrier(
    carrier_id: str,
    principal: Annotated[Principal, Depends(get_principal)],
    reg: Annotated[Registry, Depends(get_registry)],
    body: NoteIn | None = None,
) -> CarrierOut:
    """归还（借出 → 已归还）。"""
    try:
        c = reg.carrier_store.give_back(
            carrier_id, operator=principal.username, note=body.note if body else None
        )
    except KeyError as exc:
        raise _err(404, "NOT_FOUND", str(exc)) from None
    except ValueError as exc:
        raise _err(409, "INVALID_STATE", str(exc)) from None
    _audit(reg, principal, "carrier_return", c, None)
    return CarrierOut(**c)


@router.post("/{carrier_id}/destroy-request", response_model=CarrierOut)
def request_destroy(
    carrier_id: str,
    body: DestroyRequestIn,
    principal: Annotated[Principal, Depends(require_role("secadmin"))],
    reg: Annotated[Registry, Depends(get_registry)],
) -> CarrierOut:
    """发起销毁（安全保密管理员，记录销毁方式；状态 → 待销毁）。"""
    try:
        c = reg.carrier_store.request_destroy(
            carrier_id,
            operator=principal.username,
            destroy_method=body.destroy_method,
            note=body.note,
        )
    except KeyError as exc:
        raise _err(404, "NOT_FOUND", str(exc)) from None
    except ValueError as exc:
        raise _err(409, "INVALID_STATE", str(exc)) from None
    _audit(reg, principal, "carrier_destroy_request", c, body.destroy_method)
    return CarrierOut(**c)


@router.post("/{carrier_id}/destroy-confirm", response_model=CarrierOut)
def confirm_destroy(
    carrier_id: str,
    principal: Annotated[Principal, Depends(require_role("sysadmin"))],
    reg: Annotated[Registry, Depends(get_registry)],
    body: NoteIn | None = None,
) -> CarrierOut:
    """确认销毁（系统管理员；须与发起的保密员为不同账号，双确认）。"""
    c = _carrier_or_404(reg, carrier_id)
    if c["destroy_requested_by"] and c["destroy_requested_by"] == principal.username:
        raise _err(409, "SELF_CONFIRM", "销毁须保密员+系统管理员双确认，禁止同一账号自确认")
    try:
        c = reg.carrier_store.confirm_destroy(
            carrier_id, operator=principal.username, note=body.note if body else None
        )
    except KeyError as exc:
        raise _err(404, "NOT_FOUND", str(exc)) from None
    except ValueError as exc:
        raise _err(409, "INVALID_STATE", str(exc)) from None
    _audit(reg, principal, "carrier_destroy", c, c.get("destroy_method"))
    return CarrierOut(**c)


@router.get("/{carrier_id}/destroy-certificate.pdf")
def destroy_certificate(
    carrier_id: str,
    principal: Annotated[Principal, Depends(require_role("secadmin", "sysadmin", "auditor"))],
    reg: Annotated[Registry, Depends(get_registry)],
) -> Any:
    """销毁证明导出（PDF，C-12；仅已销毁载体可导出）。"""
    c = _carrier_or_404(reg, carrier_id)
    if c["status"] != "destroyed":
        raise _err(409, "NOT_DESTROYED", "载体尚未完成销毁，不能导出销毁证明")
    out_dir = reg.config.paths.reports_dir
    from backend.infra.config import resolve_config_path

    out = resolve_config_path(out_dir) / f"destroy_cert_{carrier_id}.pdf"
    build_destroy_certificate(c, out)
    reg.repository.append_audit(
        actor=principal.username,
        action="carrier_certificate_export",
        object_type="carrier",
        object_id=carrier_id,
        before=None,
        after={"pdf": out.name},
        note=None,
    )
    from fastapi.responses import FileResponse

    return FileResponse(str(out), media_type="application/pdf", filename=out.name)
