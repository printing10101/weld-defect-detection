"""C-16 零外联检测：进程级 egress guard 专项测试。

覆盖：
- 非回环目的地址连接被阻断（原 connect 不执行）；
- 阻断落安全告警（alerts，level=high）+ 主审计链（action=egress_blocked）；
- 白名单网段放行、回环恒放行；
- guard 关闭（egress.enabled=false）时不拦截；
- urllib OpenerDirector.open 层同样拦截；
- 同步通道 UrllibJsonPoster 接入 guard（被阻断则推送失败且留本地队列）；
- TestClient 回环通信不受影响（127.0.0.1 在白名单）。
"""

from __future__ import annotations

import socket
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from backend.app.dependencies import get_registry
from backend.app.main import app
from backend.infra import egress_guard
from backend.infra.egress_guard import (
    EgressBlockedError,
    check_url,
    configure_egress_guard,
    get_guard,
)
from backend.infra.sync_io import UrllibJsonPoster

# 非回环测试用地址（RFC 5737 文档段；guard 阻断后真实 connect 不会发生，无外发流量）
EXTERNAL_IP = "192.0.2.10"  # TEST-NET-1（白名单用例放行它所在 /24）
EXTERNAL_IP_OTHER = "198.51.100.7"  # TEST-NET-2（始终不在白名单）


@contextmanager
def _guard_enabled(allow_cidrs: list[str] | None = None):
    """临时启用 guard（回环外默认全拦截），退出恢复关闭态（不污染其它测试）。"""
    configure_egress_guard(True, allow_cidrs or [])
    try:
        yield get_guard()
    finally:
        configure_egress_guard(False, [])


def test_connect_to_non_loopback_is_blocked(monkeypatch):
    """非白名单目的连接：阻断抛 EgressBlockedError，原 connect 不执行。"""
    with _guard_enabled():
        executed: list[object] = []
        monkeypatch.setattr(
            egress_guard, "_ORIG_SOCKET_CONNECT", lambda self, addr: executed.append(addr)
        )
        s = socket.socket()
        try:
            with pytest.raises(EgressBlockedError):
                s.connect((EXTERNAL_IP, 80))
        finally:
            s.close()
        assert executed == [], "被拦截的连接绝不应触达真实 connect"


def test_block_writes_alert_and_audit():
    """阻断事件落库：安全告警（high）+ 主审计链 egress_blocked。"""
    reg = get_registry()
    before_alerts = reg.security_store.count_alerts(kind="egress_blocked")
    with _guard_enabled(), pytest.raises(EgressBlockedError):
        guard = get_guard()
        assert guard is not None, "守卫应已装配"
        guard.check(EXTERNAL_IP, 8080, context="unit-test")
    after_alerts = reg.security_store.count_alerts(kind="egress_blocked")
    assert after_alerts == before_alerts + 1
    entries, _total = reg.repository.list_audit(action="egress_blocked", limit=5)
    assert entries, "egress_blocked 必须入主审计链"
    latest = entries[0]
    assert latest["after"]["host"] == EXTERNAL_IP
    assert latest["after"]["blocked"] is True


def test_allowlisted_cidr_and_loopback_pass(monkeypatch):
    """白名单网段放行（原 connect 执行）；回环恒放行（代码级保证）。"""
    with _guard_enabled(allow_cidrs=["192.0.2.0/24"]):
        executed: list[object] = []
        monkeypatch.setattr(
            egress_guard, "_ORIG_SOCKET_CONNECT", lambda self, addr: executed.append(addr)
        )
        s = socket.socket()
        try:
            s.connect((EXTERNAL_IP, 80))  # 落在 192.0.2.0/24 → 放行
        finally:
            s.close()
        assert executed == [(EXTERNAL_IP, 80)]
        guard = get_guard()
        assert guard is not None, "守卫应已装配"
        assert guard.is_allowed("127.0.0.1")
        assert guard.is_allowed("127.8.8.8")  # 整个 127.0.0.0/8
        assert guard.is_allowed("::1")
        assert not guard.is_allowed(EXTERNAL_IP_OTHER)  # 未列入白名单的外部地址
        assert not guard.is_allowed("nonexistent.invalid.example")  # 解析失败从严拦截


def test_guard_disabled_does_not_block(monkeypatch):
    """egress.enabled=false：check 全放行（原 connect 直达）。"""
    configure_egress_guard(False, [])
    assert get_guard() is None
    executed: list[object] = []
    monkeypatch.setattr(
        egress_guard, "_ORIG_SOCKET_CONNECT", lambda self, addr: executed.append(addr)
    )
    s = socket.socket()
    try:
        s.connect((EXTERNAL_IP, 80))
    finally:
        s.close()
    assert executed == [(EXTERNAL_IP, 80)]


def test_urllib_opener_blocked(monkeypatch):
    """urllib 层：OpenerDirector.open 非白名单目的同样拦截。"""
    with _guard_enabled():
        executed: list[object] = []
        monkeypatch.setattr(
            egress_guard,
            "_ORIG_OPENER_OPEN",
            lambda self, url, data=None, timeout=None: executed.append(url),
        )
        import urllib.request

        opener = urllib.request.build_opener()
        with pytest.raises(EgressBlockedError):
            opener.open(f"http://{EXTERNAL_IP}/push")
        assert executed == []
        # 白名单内 URL 不拦截
        monkeypatch.setattr(
            egress_guard, "_ORIG_OPENER_OPEN", lambda self, url, data=None, timeout=None: "ok"
        )
        configure_egress_guard(True, ["192.0.2.0/24"])
        assert opener.open(f"http://{EXTERNAL_IP}/push") == "ok"


def test_sync_poster_blocked_by_guard():
    """同步通道接入：目的被拦截 → post 尽力而为返回（推送失败、本地留档语义由上层保证）。"""
    reg = get_registry()
    before = reg.security_store.count_alerts(kind="egress_blocked")
    with _guard_enabled():
        with pytest.raises(EgressBlockedError):
            check_url(f"http://{EXTERNAL_IP}:9000/api/sync")
        # post 内部消化 EgressBlockedError（与既有"失败不阻断主流程"语义一致）
        UrllibJsonPoster(timeout=1).post(f"http://{EXTERNAL_IP}:9000/api/sync", None, {"k": 1})
    assert reg.security_store.count_alerts(kind="egress_blocked") == before + 2


def test_testclient_loopback_unaffected():
    """关键兼容：guard 启用时 TestClient（回环）通信不受影响。"""
    with _guard_enabled():
        with TestClient(app) as client:
            resp = client.get("/api/v1/health")
        assert resp.status_code == 200
