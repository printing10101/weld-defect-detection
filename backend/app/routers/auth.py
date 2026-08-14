"""用户鉴权与 RBAC 接口（§T3，P0 用户权限与登录）。

- POST /auth/login      公开：用户名+密码 → 访问令牌。
- GET  /auth/me          登录用户自身信息。
- POST /auth/register    仅管理员：创建用户。
- GET  /auth/users       仅管理员：用户列表。
- POST /auth/users/{u}/role    仅管理员：改角色。
- POST /auth/users/{u}/disable 仅管理员：启/停用。
- POST /auth/change-password   本人或管理员：改密码。

令牌经 X-Scan-Token 或 Authorization: Bearer 携带；其余 /api/v1 路由经 router-level
依赖统一要求登录（见各 router 的 dependencies=[Depends(get_current_user)]）。
"""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.app.auth import (
    ROLE_ADMIN,
    CurrentUser,
    authenticate_user,
    create_access_token,
    get_current_user,
    hash_password,
    require_roles,
)
from backend.app.dependencies import Registry, get_registry

router = APIRouter(tags=["auth"])

_ROLE_LIT = Literal["reviewer", "auditor", "admin"]
_MIN_PW = 8


# ---------------------------------------------------------------------------
# 请求/响应模型
# ---------------------------------------------------------------------------
class LoginIn(BaseModel):
    username: str = Field(min_length=1)
    password: str = Field(min_length=1)


class RegisterIn(BaseModel):
    username: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    display_name: str | None = Field(default=None, max_length=128)
    role: _ROLE_LIT = "reviewer"
    password: str = Field(min_length=_MIN_PW, description="至少 8 位")


class UserOut(BaseModel):
    id: str
    username: str
    display_name: str | None
    role: str
    disabled: bool
    created_at: str | None
    created_by: str | None
    last_login_at: str | None


class RoleIn(BaseModel):
    role: _ROLE_LIT


class DisableIn(BaseModel):
    disabled: bool


class ChangePasswordIn(BaseModel):
    new_password: str = Field(min_length=_MIN_PW)
    old_password: str | None = None
    username: str | None = None  # 管理员代改时指定；本人修改留空


class LoginOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


# ---------------------------------------------------------------------------
# 接口
# ---------------------------------------------------------------------------
@router.post("/auth/login", response_model=LoginOut)
def login(body: LoginIn, reg: Annotated[Registry, Depends(get_registry)]) -> LoginOut:
    """用户名+密码登录；成功返回令牌与用户信息。失败 → 401（不暴露具体原因）。"""
    user = authenticate_user(reg.repository, body.username, body.password)
    if user is None:
        raise HTTPException(
            status_code=401, detail={"code": "INVALID_CREDENTIALS", "message": "用户名或密码错误"}
        )
    reg.repository.update_last_login(user["username"])
    token = create_access_token(
        subject=user["username"], role=user["role"], display_name=user.get("display_name") or ""
    )
    return LoginOut(access_token=token, user=UserOut(**user))


@router.get("/auth/me", response_model=UserOut)
def me(user: Annotated[CurrentUser, Depends(get_current_user)]) -> UserOut:
    """当前登录用户信息。"""
    return UserOut(
        id=user.username,  # 占位（CurrentUser 不携带 id；下行用 repository 补全）
        username=user.username,
        display_name=user.display_name,
        role=user.role,
        disabled=False,
        created_at=None,
        created_by=None,
        last_login_at=None,
    )


@router.post("/auth/register", response_model=UserOut)
def register(
    body: RegisterIn,
    reg: Annotated[Registry, Depends(get_registry)],
    admin: Annotated[CurrentUser, Depends(require_roles(ROLE_ADMIN))],
) -> UserOut:
    """创建用户（仅管理员）。"""
    try:
        user = reg.repository.create_user(
            username=body.username,
            display_name=body.display_name,
            role=body.role,
            password_hash=hash_password(body.password),
            created_by=admin.username,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail={"code": "REGISTER_FAILED", "message": str(exc)}
        ) from exc
    return UserOut(**user)


@router.get("/auth/users", response_model=list[UserOut])
def list_users(
    reg: Annotated[Registry, Depends(get_registry)],
    _: Annotated[CurrentUser, Depends(require_roles(ROLE_ADMIN))],
) -> list[UserOut]:
    """用户列表（仅管理员）。"""
    return [UserOut(**u) for u in reg.repository.list_users()]


@router.post("/auth/users/{username}/role", response_model=UserOut)
def set_role(
    username: str,
    body: RoleIn,
    reg: Annotated[Registry, Depends(get_registry)],
    _: Annotated[CurrentUser, Depends(require_roles(ROLE_ADMIN))],
) -> UserOut:
    """修改用户角色（仅管理员）。"""
    try:
        reg.repository.set_role(username, body.role)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail={"code": "NOT_FOUND", "message": str(exc)}
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail={"code": "BAD_ROLE", "message": str(exc)}
        ) from exc
    return UserOut(**reg.repository.get_user_by_username(username))


@router.post("/auth/users/{username}/disable", response_model=UserOut)
def set_disabled(
    username: str,
    body: DisableIn,
    reg: Annotated[Registry, Depends(get_registry)],
    _: Annotated[CurrentUser, Depends(require_roles(ROLE_ADMIN))],
) -> UserOut:
    """启/停用用户（仅管理员）。停用的用户禁止登录，但历史留痕保留。"""
    try:
        reg.repository.set_disabled(username, body.disabled)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail={"code": "NOT_FOUND", "message": str(exc)}
        ) from exc
    return UserOut(**reg.repository.get_user_by_username(username))


@router.post("/auth/change-password", response_model=dict)
def change_password(
    body: ChangePasswordIn,
    reg: Annotated[Registry, Depends(get_registry)],
    current: Annotated[CurrentUser, Depends(get_current_user)],
) -> dict:
    """修改密码：本人（需 old_password）或管理员代改（指定 username，免 old_password）。"""
    # 代改：管理员指定 username 且当前为管理员
    if body.username is not None:
        if not current.is_admin:
            raise HTTPException(
                status_code=403,
                detail={"code": "FORBIDDEN", "message": "仅管理员可代改他人密码"},
            )
        target = body.username
    else:
        target = current.username

    # 本人修改必须校验旧密码
    if target == current.username:
        if not body.old_password:
            raise HTTPException(
                status_code=422,
                detail={"code": "MISSING_OLD_PASSWORD", "message": "修改自身密码需提供旧密码"},
            )
        existing = reg.repository.get_user_by_username(target)
        if existing is None or not authenticate_user(reg.repository, target, body.old_password):
            raise HTTPException(
                status_code=401, detail={"code": "INVALID_CREDENTIALS", "message": "旧密码错误"}
            )

    try:
        reg.repository.set_password(target, hash_password(body.new_password))
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail={"code": "NOT_FOUND", "message": str(exc)}
        ) from exc
    return {"ok": True, "username": target}
