"""用户鉴权与 RBAC（§T3，P0 用户权限与登录）。

设计取舍（在环境约束下取最稳方案）：
- 密码哈希：PBKDF2-HMAC-SHA256（标准库 hashlib，零新增编译依赖；NIST SP 800-63B /
  OWASP 推荐，迭代 260000）。本机离线桌面 NDT 场景足够，且规避引入 argon2/bcrypt
  等第三方编译包在 safe-delete shim 下的安装风险。
- 访问令牌：HMAC-SHA256 签名的无状态令牌（payload = sub/role/exp/jti），不落库、
  到期即失效；密钥经 env SCAN_AUTH_SECRET → 持久化随机密钥文件 data/.auth_secret
  → 临时随机（重启失效，仅开发）三级解析。
- RBAC 三角色：reviewer 评片员 / auditor 审核员 / admin 管理员；权限矩阵见各路由依赖。

所有密码明文永不入库；审计 actor 一律取自登录用户（闭合 T1/T2 的占位缺口）。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, HTTPException, Request

_LOG = logging.getLogger("scandetection.auth")

# ---- 角色（§T3 RBAC）----
ROLE_REVIEWER = "reviewer"  # 评片员：评片/出报告/初评
ROLE_AUDITOR = "auditor"  # 审核员：复核/仲裁/查看审计
ROLE_ADMIN = "admin"  # 管理员：用户管理 + 全部权限
ROLES = (ROLE_REVIEWER, ROLE_AUDITOR, ROLE_ADMIN)
ROLE_LABELS = {
    ROLE_REVIEWER: "评片员",
    ROLE_AUDITOR: "审核员",
    ROLE_ADMIN: "管理员",
}

_PBKDF2_ITERS = 260_000
_TOKEN_TTL_DEFAULT = 60 * 24  # 分钟（24h）


# ---------------------------------------------------------------------------
# 密码哈希（PBKDF2-HMAC-SHA256）
# ---------------------------------------------------------------------------
def hash_password(password: str) -> str:
    """派生密码哈希，返回 `pbkdf2_sha256$iters$salt_hex$hash_hex`。"""
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERS)
    return f"pbkdf2_sha256${_PBKDF2_ITERS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """恒定时间校验密码与存储哈希是否匹配；格式异常一律 False。"""
    try:
        algo, iters_s, salt_hex, hash_hex = stored.split("$")
    except ValueError:
        return False
    if algo != "pbkdf2_sha256":
        return False
    try:
        iters = int(iters_s)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except ValueError:
        return False
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iters)
    return hmac.compare_digest(dk, expected)


# ---------------------------------------------------------------------------
# 令牌密钥解析（三级回退）
# ---------------------------------------------------------------------------
def harden_secret_file(path: str | Path) -> None:
    """收紧密钥类文件权限（F7）。

    - POSIX：chmod 0600（同机其它用户不可读，防伪造令牌）。
    - Windows：os.chmod 对 ACL 无效，改用 icacls 去掉继承并仅授权当前用户
      （R,W），避免密钥文件被同机其它账户读取。
    """
    path = Path(path)
    if os.name == "nt":
        try:
            import subprocess

            username = os.environ.get("USERNAME", "")
            if username:
                subprocess.run(
                    [
                        "icacls",
                        str(path),
                        "/inheritance:r",
                        "/grant:r",
                        f"{username}:(R,W)",
                    ],
                    check=False,
                    capture_output=True,
                )
        except Exception:  # noqa: BLE001 - 收紧密钥失败非致命（仅降低防护）
            _LOG.warning("icacls 收紧密钥文件权限失败（非致命）：%s", path)
    else:
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass


def resolve_auth_secret(data_dir: str) -> str:
    """解析 HMAC 签名密钥：env SCAN_AUTH_SECRET → 持久化文件 data/.auth_secret → 临时随机。"""
    env = os.environ.get("SCAN_AUTH_SECRET")
    if env:
        return env
    p = Path(data_dir) / ".auth_secret"
    try:
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
        key = secrets.token_hex(32)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(key, encoding="utf-8")
        # 密钥文件强制收敛权限（F7：POSIX 0600 / Windows icacls 仅授权当前用户）
        harden_secret_file(p)
        _LOG.warning("生成持久化鉴权密钥 %s（首次启动；生产请用 SCAN_AUTH_SECRET 注入）", p)
        return key
    except Exception:  # noqa: BLE001 - 密钥文件不可用则降为临时密钥（重启失效，仅开发）
        _LOG.warning("鉴权密钥持久化失败，使用临时随机密钥（重启失效）")
        return secrets.token_hex(32)


def get_auth_secret() -> str:
    """惰性获取密钥（避免顶层 import 循环；首次调用触发 registry 初始化）。"""
    from backend.app.dependencies import get_registry

    return resolve_auth_secret(get_registry().config.paths.data_dir)


# ---------------------------------------------------------------------------
# 访问令牌（HMAC 签名，无状态）
# ---------------------------------------------------------------------------
def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def create_access_token(
    *,
    subject: str,
    role: str,
    display_name: str = "",
    ttl_minutes: int = _TOKEN_TTL_DEFAULT,
    secret: str | None = None,
) -> str:
    """签发访问令牌 `body.signature`（body 为 base64url(JSON)）。"""
    secret = secret or get_auth_secret()
    now = int(time.time())
    payload = {
        "sub": subject,
        "role": role,
        "name": display_name or subject,
        "iat": now,
        "exp": now + ttl_minutes * 60,
        "jti": uuid.uuid4().hex,
    }
    body = _b64url_encode(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )
    sig = hmac.new(secret.encode("utf-8"), body.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{body}.{sig}"


def decode_token(token: str, secret: str | None = None) -> dict | None:
    """校验并解码令牌；无效/过期/篡改返回 None。"""
    secret = secret or get_auth_secret()
    try:
        body, sig = token.rsplit(".", 1)
    except ValueError:
        return None
    expected = hmac.new(secret.encode("utf-8"), body.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig):
        return None
    try:
        payload = json.loads(_b64url_decode(body))
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload.get("exp"), int) or payload["exp"] < int(time.time()):
        return None
    return payload


# ---------------------------------------------------------------------------
# 当前用户依赖 + RBAC
# ---------------------------------------------------------------------------
@dataclass
class CurrentUser:
    username: str
    role: str
    display_name: str
    jti: str = ""

    @property
    def is_admin(self) -> bool:
        return self.role == ROLE_ADMIN

    @property
    def is_auditor(self) -> bool:
        return self.role in (ROLE_AUDITOR, ROLE_ADMIN)


def get_current_user(request: Request) -> CurrentUser:
    """解析 X-Scan-Token 或 Authorization: Bearer；缺/错/过期 → 401。"""
    token = request.headers.get("X-Scan-Token")
    if not token:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:].strip()
    if not token:
        raise HTTPException(
            status_code=401,
            detail={
                "code": "UNAUTHORIZED",
                "message": "缺少认证令牌（X-Scan-Token 或 Authorization）",
            },
        )
    payload = decode_token(token)
    if payload is None:
        raise HTTPException(
            status_code=401, detail={"code": "INVALID_TOKEN", "message": "令牌无效或已过期"}
        )
    role = payload.get("role")
    if role not in ROLES:
        raise HTTPException(
            status_code=401, detail={"code": "INVALID_TOKEN", "message": "角色非法"}
        )
    jti = payload.get("jti")
    if is_token_revoked(jti):
        raise HTTPException(
            status_code=401, detail={"code": "TOKEN_REVOKED", "message": "令牌已注销"}
        )
    return CurrentUser(
        username=payload["sub"],
        role=role,
        display_name=payload.get("name", payload["sub"]),
        jti=jti or "",
    )


def require_roles(*roles: str):
    """RBAC 依赖工厂：当前用户角色不在集合内 → 403。"""

    def _dep(user: Annotated[CurrentUser, Depends(get_current_user)]) -> CurrentUser:
        if user.role not in roles:
            need = " / ".join(ROLE_LABELS.get(r, r) for r in roles)
            raise HTTPException(
                status_code=403, detail={"code": "FORBIDDEN", "message": f"需要角色：{need}"}
            )
        return user

    return _dep


# ---------------------------------------------------------------------------
# 鉴权辅助（app 层）
# ---------------------------------------------------------------------------
def authenticate_user(repository, username: str, password: str) -> dict[str, Any] | None:
    """校验用户名+密码；成功返回用户 dict，失败/禁用返回 None（恒定时间对外不暴露原因）。"""
    user = repository.get_user_by_username(username)
    if user is None or user.get("disabled"):
        # 仍跑一次哈希以恒定时间，避免用户名枚举计时侧信道
        verify_password(password, hash_password("dummy"))
        return None
    # 密码哈希由专用 getter 取回（_user_to_dict 已剥离，绝不进对外 dict）。
    stored = repository.get_user_password_hash(username)
    if stored is None or not verify_password(password, stored):
        return None
    return user


# ---------------------------------------------------------------------------
# 登录防爆破（F5）：内存级失败计数 + 锁定（单进程桌面场景足够；重启即清空）。
# ---------------------------------------------------------------------------
class LoginGuard:
    """按用户名的登录失败计数与锁定；成功登录即清零。"""

    def __init__(self, max_attempts: int = 5, lock_seconds: int = 900) -> None:
        self.max_attempts = max_attempts
        self.lock_seconds = lock_seconds
        self._fails: dict[str, int] = {}
        self._locked_until: dict[str, float] = {}

    @staticmethod
    def _key(username: str) -> str:
        return (username or "").strip().lower()

    def locked(self, username: str) -> tuple[bool, int]:
        k = self._key(username)
        until = self._locked_until.get(k, 0.0)
        if until > time.time():
            return True, int(until - time.time())
        return False, 0

    def failure(self, username: str) -> tuple[bool, int]:
        locked, remaining = self.locked(username)
        if locked:
            return True, remaining
        k = self._key(username)
        self._fails[k] = self._fails.get(k, 0) + 1
        if self._fails[k] >= self.max_attempts:
            self._locked_until[k] = time.time() + self.lock_seconds
            return True, self.lock_seconds
        return False, 0

    def success(self, username: str) -> None:
        k = self._key(username)
        self._fails.pop(k, None)
        self._locked_until.pop(k, None)


_login_guard = LoginGuard()


def check_login_locked(username: str) -> tuple[bool, int]:
    return _login_guard.locked(username)


def register_login_failure(username: str) -> tuple[bool, int]:
    return _login_guard.failure(username)


def register_login_success(username: str) -> None:
    _login_guard.success(username)


# ---------------------------------------------------------------------------
# 令牌吊销（F6）：jti 已在 create_access_token 写入；吊销集落盘
# data/.revoked_tokens.json（同机持久，注销跨重启仍生效）。
# ---------------------------------------------------------------------------
_REVOKED: set[str] = set()
_REVOKED_LOADED = False


def _revoked_store_path() -> Path:
    from backend.app.dependencies import get_registry

    return Path(get_registry().config.paths.data_dir) / ".revoked_tokens.json"


def _ensure_revoked_loaded() -> None:
    global _REVOKED_LOADED
    if _REVOKED_LOADED:
        return
    try:
        p = _revoked_store_path()
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            _REVOKED.update(data.get("jtis", []))
    except Exception:  # noqa: BLE001
        _LOG.warning("读取吊销令牌列表失败（忽略）")
    _REVOKED_LOADED = True


def _persist_revoked() -> None:
    try:
        p = _revoked_store_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"jtis": sorted(_REVOKED)}), encoding="utf-8")
        harden_secret_file(p)
    except Exception:  # noqa: BLE001
        _LOG.warning("持久化吊销令牌列表失败（忽略）")


def revoke_token(jti: str) -> None:
    if not jti:
        return
    _ensure_revoked_loaded()
    _REVOKED.add(jti)
    _persist_revoked()


def is_token_revoked(jti: str) -> bool:
    if not jti:
        return False
    _ensure_revoked_loaded()
    return jti in _REVOKED
