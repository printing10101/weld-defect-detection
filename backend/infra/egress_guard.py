"""进程级外联防护（C-16 零外联检测）。

单机纯离线部署的核心软件侧防线：在进程内 monkeypatch
``socket.socket.connect`` 与 ``urllib.request.OpenerDirector.open``，
对每次外联目的地址做白名单校验——

- 本机回环（127.0.0.0/8、::1/128）恒放行（前后端 / 标注器 / TestClient
  通信必需，代码级保证、不随配置丢失）；
- 其余目的地址须落在 ``egress.allow_cidrs`` 登记的网段内（默认空）；
- 非白名单目的 → 阻断（抛 :class:`EgressBlockedError`，原 connect/open
  不会执行）+ 安全告警入库（alerts，level=high）+ 主审计链留痕
  （action=egress_blocked）。

装配点在 app 启动（main.lifespan，任何请求处理前）；进程级、幂等——
重复装配只更新白名单/开关，不重复打补丁。``egress.enabled=false`` 时
守卫置空（check 全放行），且补丁保留但永不拦截（避免热卸载补丁引入的
线程竞态）。

诚实边界（不夸大）：
- 主机名目的地址需先 getaddrinfo 解析才能判定，解析本身可能产生 DNS
  查询外发；离线部署应配置 IP 字面量端点（文档已注明），解析后仍会
  对每个结果地址做白名单校验，TCP 连接必被校验；
- 本守卫只覆盖 Python 进程内的 socket/urllib 外联，不能替代 OS 层
  （主机防火墙出站默认拒绝）边界管控，两者互补。
"""

from __future__ import annotations

import ipaddress
import logging
import socket
import threading
import urllib.request
from collections.abc import Callable
from urllib.parse import urlsplit

_LOG = logging.getLogger("scandetection.egress")

# 代码级恒放行的回环网段（本机 IPC 通信必需，不提供配置关闭）
_LOOPBACK_V4 = ipaddress.ip_network("127.0.0.0/8")
_LOOPBACK_V6 = ipaddress.ip_network("::1/128")


class EgressBlockedError(ConnectionError):
    """外联被拦截（继承 ConnectionError：调用方按网络失败语义处理即可）。"""

    def __init__(self, host: str, port: int | None, context: str) -> None:
        self.host = host
        self.port = port
        self.context = context
        super().__init__(f"外联已被拦截（C-16 零外联策略）: {host}:{port} via {context}")


class EgressGuard:
    """外联白名单校验器（线程安全；阻断判定 + 告警/审计落库）。"""

    def __init__(self, allow_cidrs: list[str]) -> None:
        self._lock = threading.Lock()
        self._allow_networks = self._parse_cidrs(allow_cidrs)
        self.blocked_total = 0  # 进程内计数（可观测；持久化以 alerts 表为准）

    @staticmethod
    def _parse_cidrs(cidrs: list[str]) -> list[ipaddress.IPv4Network | ipaddress.IPv6Network]:
        nets = []
        for c in cidrs or []:
            try:
                nets.append(ipaddress.ip_network(c, strict=False))
            except ValueError:
                # 配置错误不静默吞掉：启动期显式告警日志（放行规则宁缺毋滥）
                _LOG.error("egress.allow_cidrs 配置非法，已忽略该条: %r", c)
        return nets

    # ---- 判定 ----

    def is_allowed(self, host: str) -> bool:
        """目的地址是否在白名单（回环恒放行；主机名逐个解析结果校验）。"""
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            # 主机名：解析后校验全部结果地址（任一不在白名单即不放行）。
            # 诚实注：getaddrinfo 本身可能产生 DNS 外联，见模块 docstring。
            try:
                infos = socket.getaddrinfo(host, None)
            except OSError:
                return False  # 无法确定目的地址 → 从严拦截
            return all(self._ip_allowed(info[4][0]) for info in infos)
        return self._ip_allowed(str(ip))

    def _ip_allowed(self, ip: str) -> bool:
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            return False
        if addr in _LOOPBACK_V4 or addr in _LOOPBACK_V6:
            return True
        return any(addr in net for net in self._allow_networks)

    def check(self, host: str, port: int | None = None, *, context: str = "socket") -> None:
        """白名单校验入口：不在白名单 → 阻断 + 告警 + 审计。放行则静默返回。"""
        if self.is_allowed(host):
            return
        with self._lock:
            self.blocked_total += 1
        _LOG.warning("外联拦截（C-16）: %s:%s via %s", host, port, context)
        self._record_block(host, port, context)
        raise EgressBlockedError(host, port, context)

    def _record_block(self, host: str, port: int | None, context: str) -> None:
        """阻断事件持久化：安全告警（high）+ 主审计链。失败不掩盖阻断本身。

        记录器由 app 层在 registry 装配完成后注入（set_block_recorder，
        watchdog raise_alert/append_audit 注入同范式）；未注入前的极早期
        阻断仅日志留痕。
        """
        if _recorder is None:
            return  # registry 装配中的极早期阻断：日志已留痕
        try:
            _recorder(host, port, context)
        except Exception as exc:  # noqa: BLE001 - 告警落库失败不影响阻断语义
            _LOG.warning("外联拦截告警落库失败: %s", exc)


# ---------------------------------------------------------------------------
# 进程级装配（幂等）
# ---------------------------------------------------------------------------

_guard: EgressGuard | None = None
_patched = False
_guard_lock = threading.Lock()
_recorder: Callable[[str, int | None, str], None] | None = None


def set_block_recorder(recorder: Callable[[str, int | None, str], None]) -> None:
    """注入阻断事件记录器（告警 + 审计），由 app 层在 registry 就绪后调用。

    记录器签名 (host, port, context)；注入动作幂等（后者覆盖前者）。
    """
    global _recorder
    _recorder = recorder
    _LOG.info("egress guard block recorder wired")


_ORIG_SOCKET_CONNECT = socket.socket.connect
_ORIG_OPENER_OPEN = urllib.request.OpenerDirector.open


def _address_host(address: object) -> tuple[str, int | None] | None:
    """从 connect 参数提取 (host, port)；非 INET 族（如 AF_UNIX 路径）返回 None 不校验。"""
    if isinstance(address, (tuple, list)) and address:
        host = address[0]
        port = address[1] if len(address) > 1 else None
        if isinstance(host, str):
            return host, port if isinstance(port, int) else None
        return None
    return None  # 字节路径（AF_UNIX）等本机 IPC，不属外联


def _guarded_socket_connect(sock_self, address):
    parsed = _address_host(address)
    if parsed is not None and _guard is not None:
        _guard.check(parsed[0], parsed[1], context="socket")
    return _ORIG_SOCKET_CONNECT(sock_self, address)


def _guarded_opener_open(opener_self, fullurl, data=None, timeout=socket._GLOBAL_DEFAULT_TIMEOUT):
    try:
        host = urlsplit(str(getattr(fullurl, "full_url", fullurl))).hostname
    except ValueError:
        host = None
    if host and _guard is not None:
        _guard.check(host, None, context="urllib")
    return _ORIG_OPENER_OPEN(opener_self, fullurl, data, timeout)


def configure_egress_guard(enabled: bool, allow_cidrs: list[str]) -> EgressGuard | None:
    """按配置装配/更新守卫（幂等；进程级补丁只打一次）。

    - enabled=True：安装补丁（若未装）并启用白名单校验；
    - enabled=False：守卫置 None（check 全放行）。补丁保留但形同虚设，
      避免运行中卸载补丁与并发 connect 的竞态（诚实取舍：空转开销可忽略）。
    返回当前生效守卫（None=防护关闭）。
    """
    global _guard, _patched
    with _guard_lock:
        if not _patched:
            socket.socket.connect = _guarded_socket_connect  # type: ignore[method-assign]
            urllib.request.OpenerDirector.open = _guarded_opener_open  # type: ignore[method-assign]
            _patched = True
            _LOG.info("egress guard patched (socket.connect / OpenerDirector.open)")
        _guard = EgressGuard(allow_cidrs) if enabled else None
        if _guard is None:
            _LOG.warning("egress guard 已按配置关闭（egress.enabled=false）——仅建议单机调试使用")
        else:
            _LOG.info("egress guard enabled (allow_cidrs=%s)", allow_cidrs)
        return _guard


def get_guard() -> EgressGuard | None:
    """当前生效守卫（未装配/已关闭 → None）。"""
    return _guard


def check_url(endpoint: str) -> None:
    """同步通道等显式外联的前置校验：URL 目的不在白名单 → EgressBlockedError。"""
    guard = get_guard()
    if guard is None:
        return
    try:
        host = urlsplit(endpoint).hostname
    except ValueError:
        host = None
    if host:
        guard.check(host, None, context="sync_http")
