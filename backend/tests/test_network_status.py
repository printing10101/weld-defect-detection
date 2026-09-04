"""C-15 纯离线证明：启动自检 + GET /system/network-status 专项测试。

- 离线结论判定（sync.kind=local → offline_mode=True；http/cloud → False）；
- 状态端点返回 offline_mode / sync_kind / egress_guard_enabled / 外联事件计数；
- 外联拦截事件（C-16 告警）→ 端点计数递增（联动闭环）。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.dependencies import get_registry
from backend.app.main import app
from backend.app.routers.system import offline_conclusion
from backend.infra.config import AppConfig, EgressCfg, SyncCfg
from backend.infra.egress_guard import (
    EgressBlockedError,
    configure_egress_guard,
    get_guard,
)


def test_offline_conclusion_defaults_to_local():
    """默认配置（sync.kind=local）→ 离线模式成立。"""
    conclusion = offline_conclusion(AppConfig())
    assert conclusion["offline_mode"] is True
    assert conclusion["sync_kind"] == "local"
    assert conclusion["egress_guard_enabled"] is True


def test_offline_conclusion_flags_http_and_cloud():
    """sync.kind=http/cloud = 显式选择数据出本机 → offline_mode=False。"""
    for kind in ("http", "cloud"):
        cfg = AppConfig(sync=SyncCfg(kind=kind))
        conclusion = offline_conclusion(cfg)
        assert conclusion["offline_mode"] is False
        assert conclusion["sync_kind"] == kind


def test_offline_conclusion_reflects_egress_switch():
    cfg = AppConfig(egress=EgressCfg(enabled=False))
    assert offline_conclusion(cfg)["egress_guard_enabled"] is False


def test_network_status_endpoint():
    """状态端点：字段齐全，拦截计数为 alerts 表持久统计（int ≥ 0）。"""
    reg = get_registry()
    configure_egress_guard(False, [])  # 端点本身与 guard 运行态无关
    with TestClient(app) as client:
        resp = client.get("/api/v1/system/network-status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["offline_mode"] is True  # 测试环境 sync=local
    assert body["sync_kind"] == "local"
    assert isinstance(body["egress_guard_enabled"], bool)
    assert body["egress_blocked_events"] == reg.security_store.count_alerts(kind="egress_blocked")


def test_network_status_counts_grow_on_block():
    """外联拦截事件（kind=egress_blocked 告警）→ 端点计数递增。"""
    configure_egress_guard(False, [])
    with TestClient(app) as client:
        before = client.get("/api/v1/system/network-status").json()["egress_blocked_events"]
        get_registry().security_store.raise_alert(
            kind="egress_blocked", level="high", message="计数联动测试", detail={}
        )
        after = client.get("/api/v1/system/network-status").json()["egress_blocked_events"]
    assert after == before + 1


def test_block_event_feeds_network_status_count():
    """真实拦截 → 告警入库 → 端点计数递增（C-15/C-16 联动）。"""
    configure_egress_guard(True, [])
    try:
        guard = get_guard()
        assert guard is not None
        with pytest.raises(EgressBlockedError):
            guard.check("192.0.2.9", 80, context="network-status-test")
        with TestClient(app) as client:
            body = client.get("/api/v1/system/network-status").json()
        assert body["egress_blocked_events"] >= 1
    finally:
        configure_egress_guard(False, [])  # 恢复关闭态，不残留启用态
