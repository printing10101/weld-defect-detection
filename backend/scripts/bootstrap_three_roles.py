#!/usr/bin/env python3
"""三员账号引导（C-06/C-07，GB/T 28452 身份鉴别）——一次性部署脚本。

流程（对应 docs/deployment-baseline.md 第一节）：
1. 引导窗口创建第一个账号 sysadmin-01（窗口在存在任意账号后永久关闭）；
2. 以 sysadmin 身份创建 secadmin-01 / auditor-01（一人一岗）；
3. 为全部账号签发 SM2 软证书——私钥**仅此一次返回**，系统不留存；
4. 私钥写入 data/bootstrap_keys/<username>.key（0600 尽力而为），
   并生成 README 提示保管要求；
5. 对三个账号各做一次真实挑战-应答登录自证，确认密钥可用。

用法（仓库根目录，accounts 表为空时才可执行）：
    python -m backend.scripts.bootstrap_three_roles
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

# Request 须模块级可解析：脚本启用 PEP 563（延迟注解），FastAPI 解析
# override 函数注解时在模块全局查找名字，函数内导入会解析失败（实测踩中）。
from fastapi import Request

_INSTALL_ROOT = Path(__file__).resolve().parents[2]
_KEY_DIR = _INSTALL_ROOT / "data" / "bootstrap_keys"

_ROLES = [
    ("sysadmin-01", "sysadmin", "系统管理员"),
    ("secadmin-01", "secadmin", "安全保密管理员"),
    ("auditor-01", "auditor", "安全审计员"),
]

_README = """# 三员私钥保管须知（生成于 {ts}）

本目录存放三员账号的 SM2 私钥（64 hex 文本，<username>.key）：

- 私钥即身份：任何人持有该文件即可冒用对应账号，**按涉密载体管理**；
- 建议立即转移至加密 U 盘/密码机等受控介质，并从工作机删除本目录；
- 系统侧只登记公钥，私钥丢失不可补发，只能停用账号重新签发；
- 交付/验收材料中不得出现本目录内容。
"""


def _save_key(username: str, key: str, ts: str) -> Path:
    _KEY_DIR.mkdir(parents=True, exist_ok=True)
    path = _KEY_DIR / f"{username}.key"
    path.write_text(key + "\n", encoding="utf-8")
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)  # 0600（Windows 尽力而为）
    except OSError:
        pass
    readme = _KEY_DIR / "README.md"
    if not readme.exists():
        readme.write_text(_README.format(ts=ts), encoding="utf-8")
    return path


def main() -> int:
    from fastapi.testclient import TestClient

    from backend.app import auth as auth_mod
    from backend.app.main import app

    def _fake(request: Request):
        # 仅用于挂载鉴权依赖；引导/登录端点本身在豁免清单
        p = auth_mod.Principal(account_id="bootstrap-cli", username="引导脚本", role="sysadmin")
        request.state.principal = p
        return p

    app.dependency_overrides[auth_mod.get_principal] = _fake

    saved: list[tuple[str, str]] = []
    ts = ""
    with TestClient(app) as client:
        token_hdr: dict[str, str] = {}

        def _login(username: str, private_key: str) -> str:
            ch = client.get("/api/v1/auth/challenge").json()
            r = client.post(
                "/api/v1/auth/login",
                json={
                    "username": username,
                    "challenge_id": ch["challenge_id"],
                    "private_key": private_key,
                },
            )
            if r.status_code != 200:
                raise SystemExit(f"登录自证失败 {username}: {r.status_code} {r.text[:200]}")
            return r.json()["token"]

        # 1. 引导窗口创建第一个账号
        r = client.post(
            "/api/v1/auth/bootstrap", json={"username": _ROLES[0][0], "role": _ROLES[0][1]}
        )
        if r.status_code == 409:
            print("引导窗口已关闭（accounts 非空）——如需重置请走账号停用/新签流程")
            return 1
        if r.status_code != 200:
            print(f"bootstrap 失败: {r.status_code} {r.text[:200]}")
            return 1
        first = r.json()
        keys = {_ROLES[0][0]: first["private_key"]}
        from datetime import UTC, datetime

        ts = datetime.now(UTC).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")

        # 2. sysadmin 登录 → 创建另外两员
        token = _login(_ROLES[0][0], keys[_ROLES[0][0]])
        token_hdr = {"Authorization": f"Bearer {token}"}
        for username, role, _label in _ROLES[1:]:
            r = client.post(
                "/api/v1/auth/accounts",
                json={"username": username, "role": role},
                headers=token_hdr,
            )
            if r.status_code != 200:
                print(f"创建 {username} 失败: {r.status_code} {r.text[:200]}")
                return 1
            r2 = client.post(
                f"/api/v1/auth/accounts/{r.json()['account_id']}/keypair", headers=token_hdr
            )
            if r2.status_code != 200:
                print(f"签发 {username} 密钥失败: {r2.status_code} {r2.text[:200]}")
                return 1
            keys[username] = r2.json()["private_key"]

        # 3. 私钥落盘（不回显）
        for username, role, _label in _ROLES:
            saved.append((username, str(_save_key(username, keys[username], ts))))

        # 4. 三账号登录自证
        for username, role, _label in _ROLES:
            _login(username, keys[username])
            print(f"[ok] {username} ({role}) 登录自证通过")

    print("\n三员账号就绪；私钥已写入（系统不留存，务必转移受控介质后删除）：")
    for username, path in saved:
        print(f"  {username}: {path}")
    print("详见 data/bootstrap_keys/README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
