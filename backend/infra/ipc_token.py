"""IPC 一次性启动令牌（C-17 前后端通信加固）。

后端每次启动生成一次性令牌（secrets.token_urlsafe，32 字节熵），写入数据
目录 ``data/ipc_token`` 文件；有效期 = 进程生命周期（重启即换）。Tauri 外壳
在端口就绪后读取该文件并注入 WebView（window.__IPC_TOKEN__），前端统一携带
``X-IPC-Token`` 头。

威胁模型（诚实声明，不夸大）：
- 本机前后端为回环明文 HTTP，令牌**不解决传输加密**——它防的是"其他本机
  进程误调 / 浏览器网页 CSRF 式调用本机 API"（无令牌的跨源/异进程请求被
  401 拒绝）；需要传输加密时应挂本机证书启用 TLS，不在本次范围；
- 文件权限仅尽力而为：POSIX 上 chmod 600；Windows 下文件落在用户数据目录
  继承用户级 ACL（其他普通用户默认不可读），不针对 ACL 做强承诺（文档已
  注明）。
"""

from __future__ import annotations

import logging
import secrets
import threading
from pathlib import Path

_LOG = logging.getLogger("scandetection.ipc")

# 进程内令牌槽：(数据目录绝对路径, 令牌)；ensure 时按目录缓存，重复调用同目录复用
_holder: tuple[str, str] | None = None
_holder_lock = threading.Lock()


def token_file_path(data_dir: str | Path) -> Path:
    """令牌文件路径（数据目录下固定文件名）。"""
    return Path(data_dir) / "ipc_token"


def _write_token_file(path: Path, token: str) -> None:
    """令牌落盘（仅本机用户可读为尽力而为，见模块 docstring）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = path.open("w", encoding="utf-8")
    try:
        fd.write(token)
    finally:
        fd.close()
    try:
        # POSIX：仅属主可读写；Windows 上此调用被忽略，依赖数据目录继承的
        # 用户级 ACL（其他普通用户默认不可读）——尽力而为，不做强承诺。
        path.chmod(0o600)
    except OSError as exc:
        _LOG.warning("IPC 令牌文件权限收紧失败（尽力而为）: %s", exc)


def issue_token(data_dir: str | Path) -> str:
    """强制生成新令牌并落盘（启动期调用；重启即换新）。"""
    global _holder
    token = secrets.token_urlsafe(32)
    _write_token_file(token_file_path(data_dir), token)
    with _holder_lock:
        _holder = (str(Path(data_dir).resolve()), token)
    _LOG.info("IPC 一次性令牌已签发: %s", token_file_path(data_dir))
    return token


def ensure_token(data_dir: str | Path) -> str:
    """取当前令牌；同目录已签发则复用，否则签发（中间件/lifespan 共用入口）。"""
    key = str(Path(data_dir).resolve())
    with _holder_lock:
        if _holder is not None and _holder[0] == key:
            return _holder[1]
    return issue_token(data_dir)
