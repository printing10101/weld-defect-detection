"""三员身份认证路由（C-06/C-07/C-09/C-19）。

端点：
- GET  /auth/challenge                     签发登录挑战（一次一用，60s 有效）
- POST /auth/bootstrap                     引导窗口：无账号时创建第一个账号
- POST /auth/login                         SM2 挑战-响应登录（signature 或 private_key 二选一）
- POST /auth/logout                        注销当前会话
- GET  /auth/me                            当前身份
- GET/POST /auth/accounts                  账号管理（系统管理员）
- POST /auth/accounts/{id}/keypair         签发 SM2 软证书（私钥一次性返回）
- POST /auth/accounts/{id}/status          启用/停用
- POST /auth/accounts/{id}/unlock          解锁（保密员/系统管理员）
- GET  /auth/alerts                        安全告警列表（保密员）
- POST /auth/alerts/{id}/resolve           告警处置（保密员，C-19 波次3扩展前留位）

**引导窗口说明（README 同步）**：系统首次启动时 accounts 表为空，允许调用
POST /auth/bootstrap 创建第一个账号（建议角色 sysadmin），该端点在存在任意
账号后即永久关闭（409）。公钥登记两种方式：bootstrap/创建时直接带
public_key（128 hex），或创建后由 keypair 端点签发软证书（私钥一次性下发，
交本人保存，系统不留存）。
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from backend.app.auth import (
    AuthError,
    Principal,
    get_auth_service,
    get_principal,
    require_role,
)
from backend.app.dependencies import Registry, get_registry
from backend.infra.security_store import ROLES

router = APIRouter(prefix="/auth", tags=["auth"])


def _http(exc: AuthError) -> HTTPException:
    return HTTPException(status_code=exc.status, detail={"code": exc.code, "message": exc.message})


class ChallengeOut(BaseModel):
    challenge_id: str
    nonce: str  # 需以账号 SM2 私钥对该串签名（SM3withSM2）


class BootstrapIn(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    role: str = "sysadmin"
    public_key: str | None = None  # 128 hex（x||y）；缺省则现场签发软证书


class LoginIn(BaseModel):
    username: str
    challenge_id: str
    signature: str | None = None  # 客户端自签（前端集成 SM2 库时）
    private_key: str | None = None  # 软件模式·后端代签：私钥文件内容（64 hex）


class LoginOut(BaseModel):
    token: str
    account_id: str
    username: str
    role: str
    idle_timeout_min: int


class AccountOut(BaseModel):
    account_id: str
    username: str
    role: str
    sm2_public_key: str | None
    auth_mode: str
    status: str
    failed_attempts: int
    locked_until: str | None
    created_by: str | None
    created_at: str | None


class AccountCreateIn(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    role: str
    public_key: str | None = None


class StatusIn(BaseModel):
    status: str  # active | disabled


class BootstrapOut(AccountOut):
    private_key: str | None = None  # 仅引导窗口未带公钥时一次性返回


@router.get("/challenge", response_model=ChallengeOut)
def challenge(
    svc: Annotated[Any, Depends(get_auth_service)],
) -> ChallengeOut:
    """签发登录挑战：随机数一次一用，60s 有效（防重放）。"""
    challenge_id, nonce = svc.challenge()
    return ChallengeOut(challenge_id=challenge_id, nonce=nonce)


@router.post("/bootstrap", response_model=BootstrapOut)
def bootstrap(
    body: BootstrapIn,
    svc: Annotated[Any, Depends(get_auth_service)],
) -> BootstrapOut:
    """引导窗口：仅当系统尚无任何账号时可用（创建后即关闭，409）。"""
    try:
        account = svc.bootstrap(username=body.username, role=body.role)
        priv = None
        if body.public_key:
            svc._store.set_account_key(account["account_id"], body.public_key)
            account["sm2_public_key"] = body.public_key
        else:
            kp = svc.issue_keypair(account["account_id"], actor="bootstrap")
            priv = kp["private_key"]
            account["sm2_public_key"] = kp["public_key"]
        return BootstrapOut(private_key=priv, **_account_fields(account))
    except AuthError as exc:
        raise _http(exc) from None


@router.post("/login", response_model=LoginOut)
def login(
    body: LoginIn,
    svc: Annotated[Any, Depends(get_auth_service)],
) -> LoginOut:
    """SM2 挑战-响应登录。

    软件模式：signature（客户端自签）或 private_key（后端代签，前端不碰密码学）
    二选一；UKey 硬件模式走 Pkcs11Provider（本环境未真机验证，501 预留）。
    失败计数达阈值锁定账号并落告警（C-19）。
    """
    try:
        out = svc.login(
            username=body.username,
            challenge_id=body.challenge_id,
            signature=body.signature,
            private_key=body.private_key,
        )
    except AuthError as exc:
        raise _http(exc) from None
    return LoginOut(**out)


class MeOut(BaseModel):
    account_id: str
    username: str
    role: str


@router.get("/me", response_model=MeOut)
def me(principal: Annotated[Principal, Depends(get_principal)]) -> MeOut:
    return MeOut(account_id=principal.account_id, username=principal.username, role=principal.role)


@router.post("/logout")
def logout(
    request: Request,
    principal: Annotated[Principal, Depends(get_principal)],
    reg: Annotated[Registry, Depends(get_registry)],
) -> dict[str, bool]:
    """注销当前会话（token 从请求头解析后吊销其哈希；入安全审计链留痕）。"""
    from backend.app.auth import _extract_token, _query_token

    token = _extract_token(request.headers.get("authorization")) or _query_token(request)
    if token:
        from backend.infra.crypto import sm3_hex

        reg.security_store.revoke_session(sm3_hex(token.encode()))
    # 会话终结入独立安全审计链（C-21 审计完整性盘点：注销属写操作需留痕）
    reg.security_store.append_security_audit(
        actor=principal.username,
        action="session_logout",
        object_type="session",
        object_id=principal.account_id,
        before=None,
        after={"revoked": True},
        note=None,
    )
    del principal
    return {"ok": True}


def _account_fields(a: dict[str, Any]) -> dict[str, Any]:
    return {k: a.get(k) for k in AccountOut.model_fields}


@router.get("/accounts", response_model=list[AccountOut])
def list_accounts(
    _: Annotated[Principal, Depends(require_role("sysadmin"))],
    reg: Annotated[Registry, Depends(get_registry)],
) -> list[AccountOut]:
    return [AccountOut(**_account_fields(a)) for a in reg.security_store.list_accounts()]


@router.post("/accounts", response_model=AccountOut)
def create_account(
    body: AccountCreateIn,
    principal: Annotated[Principal, Depends(require_role("sysadmin"))],
    svc: Annotated[Any, Depends(get_auth_service)],
) -> AccountOut:
    """创建三员账号（一人一岗：一个账号一个角色；关键操作入安全审计链）。"""
    if body.role not in ROLES:
        raise HTTPException(
            422, detail={"code": "INVALID_ROLE", "message": f"role 须为 {'/'.join(ROLES)}"}
        )
    try:
        account = svc.create_account(
            username=body.username,
            role=body.role,
            actor=principal.username,
            public_key=body.public_key,
        )
    except AuthError as exc:
        raise _http(exc) from None
    except ValueError as exc:
        raise HTTPException(422, detail={"code": "INVALID_ACCOUNT", "message": str(exc)}) from None
    return AccountOut(**_account_fields(account))


@router.post("/accounts/{account_id}/keypair")
def issue_keypair(
    account_id: str,
    principal: Annotated[Principal, Depends(require_role("sysadmin"))],
    svc: Annotated[Any, Depends(get_auth_service)],
) -> dict[str, str]:
    """为账号签发 SM2 软证书：公钥入库，私钥一次性返回（交本人保存）。"""
    try:
        return svc.issue_keypair(account_id, actor=principal.username)
    except AuthError as exc:
        raise _http(exc) from None
    except KeyError as exc:
        raise HTTPException(404, detail={"code": "NOT_FOUND", "message": str(exc)}) from None


@router.post("/accounts/{account_id}/status", response_model=AccountOut)
def set_account_status(
    account_id: str,
    body: StatusIn,
    principal: Annotated[Principal, Depends(require_role("sysadmin"))],
    svc: Annotated[Any, Depends(get_auth_service)],
) -> AccountOut:
    """启用/停用账号（停用同时吊销全部会话；入安全审计链）。"""
    try:
        return AccountOut(
            **_account_fields(svc.set_status(account_id, body.status, actor=principal.username))
        )
    except AuthError as exc:
        raise _http(exc) from None
    except KeyError as exc:
        raise HTTPException(404, detail={"code": "NOT_FOUND", "message": str(exc)}) from None


@router.post("/accounts/{account_id}/unlock", response_model=AccountOut)
def unlock_account(
    account_id: str,
    principal: Annotated[Principal, Depends(require_role("secadmin", "sysadmin"))],
    reg: Annotated[Registry, Depends(get_registry)],
) -> AccountOut:
    """解锁账号（锁定告警的处置动作之一；入安全审计链）。"""
    try:
        out = reg.security_store.unlock_account(account_id)
    except KeyError as exc:
        raise HTTPException(404, detail={"code": "NOT_FOUND", "message": str(exc)}) from None
    reg.security_store.append_security_audit(
        actor=principal.username,
        action="account_unlock",
        object_type="account",
        object_id=account_id,
        before=None,
        after=out,
        note=None,
    )
    # 主审计链同步留痕（C-21：运维操作时间线 /audit/operations 读主链，
    # account_unlock 在其白名单内，只入安全链会导致运维时间线缺项）
    reg.repository.append_audit(
        actor=principal.username,
        action="account_unlock",
        object_type="account",
        object_id=account_id,
        before={"locked": True},
        after={"locked": False, "username": out["username"]},
        note=None,
    )
    return AccountOut(**_account_fields(out))


class AlertOut(BaseModel):
    alert_id: str
    kind: str
    level: str
    message: str
    detail: object | None
    status: str
    resolved_by: str | None
    resolved_at: str | None
    note: str | None
    created_at: str | None


class ResolveIn(BaseModel):
    note: str | None = None


@router.get("/alerts", response_model=list[AlertOut])
def list_alerts(
    _: Annotated[Principal, Depends(require_role("secadmin"))],
    reg: Annotated[Registry, Depends(get_registry)],
    status: str | None = None,
) -> list[AlertOut]:
    """安全告警列表（安全保密管理员，C-19 波次3扩展处置流转）。"""
    return [AlertOut(**a) for a in reg.security_store.list_alerts(status=status)]


@router.post("/alerts/{alert_id}/resolve", response_model=AlertOut)
def resolve_alert(
    alert_id: str,
    body: ResolveIn,
    principal: Annotated[Principal, Depends(require_role("secadmin"))],
    reg: Annotated[Registry, Depends(get_registry)],
) -> AlertOut:
    """处置告警（仅安全保密管理员；处置动作入安全审计链）。"""
    try:
        out = reg.security_store.resolve_alert(
            alert_id, resolved_by=principal.username, note=body.note
        )
    except KeyError as exc:
        raise HTTPException(404, detail={"code": "NOT_FOUND", "message": str(exc)}) from None
    reg.security_store.append_security_audit(
        actor=principal.username,
        action="alert_resolve",
        object_type="alert",
        object_id=alert_id,
        before={"status": "open"},
        after={"status": "resolved"},
        note=body.note,
    )
    return AlertOut(**out)


# ---------------------------------------------------------------------------
# 告警通知（C-22）：拉取式通知 + 确认（ack）。
#
# **诚实边界**：单机桌面部署无 WebSocket/SSE 等服务端推送通道，做不到"告警
# 产生即推送"。本组端点提供拉取式替代：登录后/任意时刻前端调用
# GET /alerts/unread-count 拉取未读计数（可轮询），确认动作走 ack 端点。
# ---------------------------------------------------------------------------

alerts_router = APIRouter(prefix="/alerts", tags=["alerts"])


class UnreadCountOut(BaseModel):
    """未读告警计数（status=open 的告警条数；acknowledged/resolved 不计）。"""

    unread: int


@alerts_router.get("/unread-count", response_model=UnreadCountOut)
def unread_count(
    _: Annotated[Principal, Depends(get_principal)],
    reg: Annotated[Registry, Depends(get_registry)],
) -> UnreadCountOut:
    """未读告警计数（任意已登录角色可拉取；保密管理员登录后据此提示）。

    拉取式通知（无推送通道）：仅暴露计数，不泄露告警内容。
    """
    return UnreadCountOut(unread=reg.security_store.count_alerts(status="open"))


class AckIn(BaseModel):
    note: str | None = None


@alerts_router.post("/{alert_id}/ack", response_model=AlertOut)
def ack_alert(
    alert_id: str,
    principal: Annotated[Principal, Depends(require_role("secadmin", "auditor"))],
    reg: Annotated[Registry, Depends(get_registry)],
    body: AckIn | None = None,
) -> AlertOut:
    """确认（已读）告警（仅安全保密管理员/安全审计员）。

    ack ≠ 处置：acknowledged 表示"已知悉"，处置（resolved）仍走
    POST /auth/alerts/{id}/resolve。确认动作入安全审计链。
    """
    try:
        out = reg.security_store.ack_alert(
            alert_id, acked_by=principal.username, note=body.note if body else None
        )
    except KeyError as exc:
        raise HTTPException(404, detail={"code": "NOT_FOUND", "message": str(exc)}) from None
    reg.security_store.append_security_audit(
        actor=principal.username,
        action="alert_ack",
        object_type="alert",
        object_id=alert_id,
        before={"status": "open"},
        after={"status": "acknowledged"},
        note=body.note if body else None,
    )
    return AlertOut(**out)
