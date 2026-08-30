"""三员身份认证核心（C-06/C-07/C-09/C-19）。

职责：
- Principal / get_principal：Bearer 会话鉴权依赖（token 只存 SM3 哈希）；
- require_role：三员权限矩阵（C-06）的路由级强制依赖；
- AuthService：SM2 挑战-响应登录（挑战一次一用、60s 有效）、失败锁定+告警、
  会话签发（空闲超时 + 并发上限）、账号引导。

**引导窗口（首次启动无账号）**：accounts 表为空时允许 POST /auth/bootstrap
创建第一个账号（建议为系统管理员），并立即为其登记 SM2 公钥；之后再调
bootstrap 返回 409。详见 routers/auth.py 与 README 部署说明。

**软件模式 vs UKey 模式（诚实声明）**：软件模式由管理员为账号签发 SM2 软
证书（/auth/accounts/{id}/keypair 生成，私钥一次性交本人保存）或登记既有
公钥；登录支持两种凭据方式——(a) 客户端自行签名后提交 signature（前端集成
SM2 库时），(b) 提交私钥文件内容 private_key 由后端代签后验签（单机本地
软件可接受的简化，私钥仅在本机进程内存中出现、不落日志/审计）。UKey 硬件
模式（auth_mode=ukey）走 Pkcs11Provider 骨架签名，本环境无硬件、未真机
验证，配置即抛带指引的错误（见 infra.crypto.Pkcs11Provider）。

X-Operator-Name 兼容语义：登录态下身份以账号为准（get_operator_name 优先
取账号名）；未登录请求该头仅作审计 actor 记录，不构成身份。
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import Depends, Header, Request
from fastapi.security.utils import get_authorization_scheme_param

from backend.app.dependencies import Registry, get_registry
from backend.infra.crypto import (
    sm2_generate_keypair,
    sm2_sign_with_private,
    sm2_verify_with_public,
)
from backend.infra.security_store import SecurityStore

# 挑战存储（进程内；单机单进程部署，重启即失效属可接受的挑战语义）
_CHALLENGE_TTL_DEFAULT = 60.0


@dataclass(frozen=True)
class Principal:
    """已认证身份（三员之一）。"""

    account_id: str
    username: str
    role: str  # sysadmin | secadmin | auditor

    @property
    def is_admin(self) -> bool:
        return self.role in ("sysadmin", "secadmin")


class AuthError(Exception):
    """认证/授权失败（由路由层转 HTTP；code 与统一错误包对齐）。"""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


# ---------------------------------------------------------------------------
# 挑战存储（一次一用、60s 有效，防重放）
# ---------------------------------------------------------------------------


class ChallengeStore:
    """进程内挑战表：{challenge_id: (nonce_hex, expires_at, used)}。"""

    def __init__(self, ttl_sec: float = _CHALLENGE_TTL_DEFAULT) -> None:
        self._ttl = ttl_sec
        self._lock = threading.Lock()
        self._items: dict[str, tuple[str, float, bool]] = {}

    def issue(self) -> tuple[str, str]:
        """签发挑战，返回 (challenge_id, nonce_hex)。"""
        import secrets

        challenge_id = uuid.uuid4().hex
        nonce = secrets.token_hex(16)
        with self._lock:
            self._gc()
            self._items[challenge_id] = (nonce, time.monotonic() + self._ttl, False)
        return challenge_id, nonce

    def consume(self, challenge_id: str) -> str:
        """取走挑战明文（一次一用）：无效/过期/已用抛 AuthError。"""
        with self._lock:
            self._gc()
            item = self._items.pop(challenge_id, None)
        if item is None:
            raise AuthError(401, "CHALLENGE_INVALID", "挑战不存在或已使用（一次一用）")
        _nonce, expires, used = item
        if used or expires < time.monotonic():
            raise AuthError(401, "CHALLENGE_EXPIRED", "挑战已过期（60s 有效）")
        return _nonce

    def _gc(self) -> None:
        now = time.monotonic()
        for k in [k for k, (_, exp, _) in self._items.items() if exp < now]:
            self._items.pop(k, None)


# ---------------------------------------------------------------------------
# 登录服务
# ---------------------------------------------------------------------------


class AuthService:
    """挑战-响应登录与会话治理（依赖 SecurityStore 持久化 + 配置阈值）。"""

    def __init__(self, store: SecurityStore, config) -> None:
        self._store = store
        self._cfg = config
        self._challenges = ChallengeStore(ttl_sec=config.challenge_ttl_sec)

    # ---- 挑战 ----

    def challenge(self) -> tuple[str, str]:
        return self._challenges.issue()

    # ---- 登录 ----

    def login(
        self,
        *,
        username: str,
        challenge_id: str,
        signature: str | None = None,
        private_key: str | None = None,
    ) -> dict[str, Any]:
        """SM2 挑战-响应登录，成功返回一次性明文 token 与账号快照。

        失败路径：账号不存在/停用（不区分提示，防用户枚举）、挑战非法、
        验签失败（计数，达阈值锁定并落告警 + 安全审计）、UKey 未对接。
        """
        account = self._store.get_account_by_username(username)
        if account is None or account["status"] != "active":
            raise AuthError(401, "BAD_CREDENTIALS", "账号不存在、已停用或凭据不匹配")
        if self._store.account_locked(account):
            raise AuthError(423, "ACCOUNT_LOCKED", "账号已锁定，请联系安全保密管理员解锁")

        nonce = self._challenges.consume(challenge_id)  # 一次一用（先取后验，防重放）

        if account["auth_mode"] == "ukey":
            # UKey 硬件模式：签名在 UKey 内完成，服务端仅持公钥。走
            # Pkcs11Provider 骨架（未配置/未真机验证 → 显式报错，接口预留）。
            raise AuthError(
                501,
                "UKEY_NOT_AVAILABLE",
                "UKey 硬件登录未在本环境启用（需对接商密硬件并完成真机联调，"
                "见 infra.crypto.Pkcs11Provider）",
            )

        if not account["sm2_public_key"]:
            raise AuthError(422, "NO_PUBLIC_KEY", "账号未登记 SM2 公钥，无法登录")

        if private_key:
            # 软件模式·后端代签（前端不碰密码学）：私钥仅在本进程内存中出现
            sig = self._sign_with_client_key(private_key, nonce)
        elif signature:
            sig = signature.strip()
        else:
            raise AuthError(422, "MISSING_CREDENTIAL", "需提供 signature 或 private_key")

        if not sm2_verify_with_public(account["sm2_public_key"], nonce.encode(), sig):
            snap, locked = self._store.register_failed_attempt(
                account["account_id"],
                max_attempts=self._cfg.max_failed_attempts,
                lock_minutes=self._cfg.lockout_min,
            )
            if locked:
                self._on_account_locked(snap)
            raise AuthError(401, "BAD_CREDENTIALS", "账号不存在、已停用或凭据不匹配")

        self._store.reset_failed_attempts(account["account_id"])
        token = uuid.uuid4().hex + uuid.uuid4().hex  # 32B 随机（双 uuid4 拼接）
        from backend.infra.crypto import sm3_hex

        self._store.create_session(
            token_hash=sm3_hex(token.encode()),
            account_id=account["account_id"],
            ttl_sec=self._cfg.idle_timeout_min * 60,
            max_sessions=self._cfg.max_sessions,
        )
        return {
            "token": token,
            "account_id": account["account_id"],
            "username": account["username"],
            "role": account["role"],
            "idle_timeout_min": self._cfg.idle_timeout_min,
        }

    @staticmethod
    def _sign_with_client_key(private_key: str, nonce: str) -> str:
        """用客户端提交的私钥代签挑战；格式非法 → 401（不泄露格式细节）。"""
        key = private_key.strip().lower().replace(" ", "")
        if key.startswith("0x"):
            key = key[2:]
        if len(key) != 64:
            raise AuthError(
                422, "INVALID_KEY_FORMAT", "私钥须为 64 位 hex（可由管理员签发的证书文件获得）"
            )
        try:
            return sm2_sign_with_private(key, nonce.encode())
        except Exception as exc:  # noqa: BLE001 - 任何签名异常都归一为凭据错误
            raise AuthError(401, "BAD_CREDENTIALS", "凭据验签失败") from exc

    def _on_account_locked(self, account: dict[str, Any]) -> None:
        """锁定处置：告警入库 + 安全审计链留痕（C-19；主链亦记一条便于统一查询）。"""
        from backend.app.dependencies import get_registry

        msg = f"账号连续登录失败已锁定: {account['username']}"
        try:
            # C-22：告警规则可配置化——account_lockout.enabled=false 时跳过告警
            # （安全审计链/主链留痕不受影响，锁定本身也不受影响）。
            from backend.app.dependencies import get_registry as _gr

            try:
                alerts_enabled = _gr().config.alerts.account_lockout.enabled
            except Exception:  # noqa: BLE001 - registry 未就绪时按默认开启
                alerts_enabled = True
            if alerts_enabled:
                self._store.raise_alert(
                    kind="account_locked",
                    level="critical",
                    message=msg,
                    detail={"account_id": account["account_id"], "role": account["role"]},
                )
            self._store.append_security_audit(
                actor="system",
                action="account_lock",
                object_type="account",
                object_id=account["account_id"],
                before={"status": "active"},
                after={"locked": True, "lockout_min": self._cfg.lockout_min},
                note=msg,
            )
            get_registry().repository.append_audit(
                actor="system",
                action="account_lock",
                object_type="account",
                object_id=account["account_id"],
                before=None,
                after={"username": account["username"]},
                note=msg,
            )
        except Exception as exc:  # noqa: BLE001 - 告警失败不掩盖 401 响应
            import logging

            logging.getLogger("scandetection.auth").warning("锁定告警落库失败: %s", exc)

    # ---- 账号管理 ----

    def bootstrap(self, *, username: str, role: str) -> dict[str, Any]:
        """引导：仅当系统尚无任何账号时创建第一个账号（引导窗口语义）。"""
        if self._store.count_accounts() > 0:
            raise AuthError(409, "BOOTSTRAP_CLOSED", "系统已存在账号，引导窗口已关闭")
        return self._store.create_account(username=username, role=role, created_by="bootstrap")

    def create_account(
        self, *, username: str, role: str, actor: str, public_key: str | None = None
    ) -> dict[str, Any]:
        out = self._store.create_account(
            username=username, role=role, sm2_public_key=public_key, created_by=actor
        )
        self._sec_audit(actor, "account_create", "account", out["account_id"], None, out)
        return out

    def issue_keypair(self, account_id: str, *, actor: str) -> dict[str, Any]:
        """为账号签发 SM2 软证书：公钥登记入库，私钥一次性返回（交本人保存）。"""
        private_key, public_key = sm2_generate_keypair()
        before = self._store.get_account(account_id)
        if before is None:
            raise KeyError(f"account not found: {account_id}")
        self._store.set_account_key(account_id, public_key)
        self._sec_audit(
            actor,
            "account_key_register",
            "account",
            account_id,
            {"had_key": bool(before["sm2_public_key"])},
            {"had_key": True},
        )
        return {"account_id": account_id, "public_key": public_key, "private_key": private_key}

    def set_status(self, account_id: str, status: str, *, actor: str) -> dict[str, Any]:
        before = self._store.get_account(account_id)
        if before is None:
            raise KeyError(f"account not found: {account_id}")
        out = self._store.set_account_status(account_id, status)
        if status == "disabled":
            self._store.revoke_account_sessions(account_id)
        self._sec_audit(actor, "account_status", "account", account_id, before, out)
        return out

    def _sec_audit(
        self,
        actor: str,
        action: str,
        object_type: str,
        object_id: str,
        before: Any,
        after: Any,
    ) -> None:
        """管理员关键操作入独立安全审计链（C-19 双链之安全链）。"""
        self._store.append_security_audit(
            actor=actor,
            action=action,
            object_type=object_type,
            object_id=object_id,
            before=before,
            after=after,
            note=None,
        )


# ---------------------------------------------------------------------------
# FastAPI 依赖
# ---------------------------------------------------------------------------


def get_auth_service(reg: Annotated[Registry, Depends(get_registry)]) -> AuthService:
    """从 registry 取认证服务（随 registry 单例缓存——挑战表必须跨请求存活）。"""
    svc = getattr(reg, "_auth_service", None)
    if svc is None:
        with reg._lock:
            svc = getattr(reg, "_auth_service", None)
            if svc is None:
                svc = AuthService(reg.security_store, reg.config.auth)
                reg._auth_service = svc
    return svc


def get_principal(
    request: Request,
    reg: Annotated[Registry, Depends(get_registry)],
    authorization: Annotated[str | None, Header()] = None,
) -> Principal:
    """Bearer 会话鉴权依赖（集中入口；测试经 app.dependency_overrides 注入）。

    - token 明文不落库：库中只比对 SM3(token)；
    - 空闲超时（滑动）：now - last_seen_at > idle_timeout → 吊销并 401；
    - 支持 ?access_token= 查询参数回退（<img>/下载直链无法带 Authorization 头，
      如影像预览、报告 PDF；与 Bearer 头等价校验）。
    """
    token = _extract_token(authorization) or _query_token(request)
    if not token:
        raise AuthError(401, "UNAUTHORIZED", "未提供会话令牌（请先登录）")
    from backend.infra.crypto import sm3_hex

    sess = reg.security_store.get_session(sm3_hex(token.encode()))
    if sess is None or sess["revoked"]:
        raise AuthError(401, "SESSION_INVALID", "会话无效或已注销")
    idle_sec = reg.config.auth.idle_timeout_min * 60
    now = datetime.now(UTC).replace(tzinfo=None)
    # 绝对有效期（自签发起算的硬上限，防"永不断活"的滑动会话）
    created = sess.get("_created_dt")
    if created is not None:
        if (now - created).total_seconds() > reg.config.auth.session_ttl_min * 60:
            reg.security_store.revoke_session(sess["token_hash"])
            raise AuthError(401, "SESSION_EXPIRED", "会话已过期，请重新登录")
    last_seen = sess.get("_last_seen_dt")
    if last_seen is not None:
        if (now - last_seen).total_seconds() > idle_sec:
            reg.security_store.revoke_session(sess["token_hash"])
            raise AuthError(401, "SESSION_EXPIRED", "会话已空闲超时，请重新登录")
    account = reg.security_store.get_account(sess["account_id"])
    if account is None or account["status"] != "active":
        raise AuthError(401, "SESSION_INVALID", "账号不可用，会话失效")
    reg.security_store.touch_session(sess["token_hash"], idle_sec)
    principal = Principal(
        account_id=account["account_id"], username=account["username"], role=account["role"]
    )
    request.state.principal = principal
    return principal


def _extract_token(authorization: str | None) -> str | None:
    scheme, param = get_authorization_scheme_param(authorization or "")
    if scheme.lower() == "bearer" and param.strip():
        return param.strip()
    return None


def _query_token(request: Request) -> str | None:
    tok = request.query_params.get("access_token")
    return tok.strip() if tok else None


def require_role(*roles: str):
    """三员权限矩阵依赖工厂（C-06）：角色不符 → 403。

    用法：``principal: Annotated[Principal, Depends(require_role("secadmin"))]``。
    """

    def _dep(principal: Annotated[Principal, Depends(get_principal)]) -> Principal:
        if principal.role not in roles:
            raise AuthError(
                403,
                "FORBIDDEN",
                f"当前角色 {principal.role!r} 无权执行该操作（需 {'/'.join(roles)}）",
            )
        return principal

    return _dep
