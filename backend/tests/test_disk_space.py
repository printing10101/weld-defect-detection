"""S-20 磁盘水位看门狗：unit 测试（注入确定性 sampler / 回调）。

覆盖：
- 正常水位 → 不告警；
- 低比例（free_ratio_pct 低于阈值）→ 告警（disk_space_low + 审计 action=disk_space）；
- 低绝对字节（free_bytes 低于 warn_min_bytes）→ 告警；
- 连续低水位只告警一次（去重防刷屏）；回落后再超 → 可再次告警；
- 采样失败（OSError）→ 不告警、不崩溃；
- snapshot 反映 enabled / 最近一次采样 / breach_count；
- 配置默认 enable=true、可经 SCAN_ 环境变量覆盖。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from backend.infra.config import DiskSpaceCfg, load_config
from backend.infra.disk_space import DiskWatchdog, sample_disk_usage

_GB = 1024 * 1024 * 1024
_TOTAL = 512 * _GB  # 512 GiB 模拟分区别太大


def _fake_sink() -> tuple[list[dict], list[dict], Callable[..., Any], Callable[..., Any]]:
    alerts: list[dict] = []
    audits: list[dict] = []

    def raise_alert(*, kind, message, level="warn", detail=None):  # type: ignore[no-untyped-def]
        alerts.append({"kind": kind, "level": level, "detail": detail})

    def append_audit(*, actor, action, object_type, object_id, before=None, after=None, note=None):  # type: ignore[no-untyped-def]
        audits.append({"action": action, "detail": after})

    return alerts, audits, raise_alert, append_audit


def _make_wd(free: int, total: int = _TOTAL, **overrides) -> tuple[DiskWatchdog, list, list]:
    alerts, audits, ra, aa = _fake_sink()
    wd = DiskWatchdog(
        interval_sec=1.0,
        warn_ratio_pct=10.0,
        warn_min_bytes=1 * _GB,
        data_dir="data",
        raise_alert=ra,
        append_audit=aa,
        sampler=lambda: (free, total),
        **overrides,
    )
    return wd, alerts, audits


def test_healthy_disk_no_alert():
    wd, alerts, audits = _make_wd(free=200 * _GB)  # ~39% 剩余
    snap = wd.check_once()
    assert alerts == []
    assert audits == []
    assert snap["breach_count"] == 0
    assert snap["free_ratio_pct"] == pytest.approx(200 / 512 * 100.0)


def test_low_ratio_triggers_alert_and_audit():
    wd, alerts, audits = _make_wd(free=30 * _GB)  # ~5.9% < 10%
    snap = wd.check_once()
    assert len(alerts) == 1
    assert alerts[0]["kind"] == "disk_space_low"
    assert alerts[0]["level"] == "warn"
    assert len(audits) == 1
    assert audits[0]["action"] == "disk_space"
    assert snap["breach_count"] == 1


def test_low_absolute_bytes_triggers_alert():
    # ratio 正常（~0.04% ）且 ratio 阈值关掉，仅"绝对字节"路径触发（1 GB 阈值下 0.2 GB 剩余）。
    alerts, _, _, _ = _fake_sink()
    audits: list = []
    wd = DiskWatchdog(
        interval_sec=1.0,
        warn_ratio_pct=0.0,
        warn_min_bytes=1 * _GB,
        data_dir="data",
        sampler=lambda: (int(0.2 * _GB), _TOTAL),
    )
    wd._raise_alert = lambda **kw: alerts.append(kw)
    wd._append_audit = lambda **kw: audits.append(kw)
    wd.check_once()
    assert len(alerts) == 1 and alerts[0]["kind"] == "disk_space_low"


def test_continuous_low_alert_once_then_recoverable():
    wd, alerts, _ = _make_wd(free=30 * _GB)
    wd.check_once()
    wd.check_once()
    wd.check_once()
    assert len(alerts) == 1  # 连续低水位去重
    assert wd.snapshot()["breach_count"] == 3

    # 模拟回升：换回高水位采样器后再次低水位应可重新告警。
    wd._sampler = lambda: (200 * _GB, _TOTAL)
    wd.check_once()  # 回落
    alerts.clear()
    wd._sampler = lambda: (30 * _GB, _TOTAL)
    wd.check_once()  # 再次低水位
    assert len(alerts) == 1


def test_sample_failure_no_alert_no_crash():
    def boom():
        raise OSError("simulated disk_usage failure")

    wd = DiskWatchdog(
        interval_sec=1.0,
        warn_ratio_pct=10.0,
        warn_min_bytes=1 * _GB,
        data_dir="data",
        sampler=boom,
    )
    snap = wd.check_once()  # 不应抛异常
    assert snap["breach_count"] == 0
    assert snap["free_bytes"] is None


def test_start_stop_lifecycle_and_snapshot_enabled(tmp_path):
    wd, _alerts, _ = _make_wd(free=30 * _GB)
    assert wd.snapshot()["enabled"] is False
    wd.start()
    assert wd.snapshot()["enabled"] is True
    wd.stop()
    assert wd.snapshot()["enabled"] is False


def test_real_sampler_returns_bytes():
    free, total = sample_disk_usage("data")
    assert total > 0
    assert 0 <= free <= total


def test_config_defaults_and_env_override(monkeypatch):
    cfg = DiskSpaceCfg()
    assert cfg.enabled is True
    assert cfg.interval_sec == 300.0
    assert cfg.warn_ratio_pct == 10.0
    assert cfg.warn_min_bytes == 1 * _GB

    monkeypatch.setenv("SCAN_DISK_SPACE__ENABLED", "false")
    monkeypatch.setenv("SCAN_DISK_SPACE__WARN_RATIO_PCT", "5")
    loaded = load_config()
    assert loaded.disk_space.enabled is False
    assert loaded.disk_space.warn_ratio_pct == 5.0


def test_schema_and_default_yaml_drift_safe():
    """新增 disk_space 配置节须 register 进 schema，避免配置漂移告警。"""
    import yaml

    from backend.infra import config as C

    raw = yaml.safe_load((C._BASE / "default.yaml").read_text(encoding="utf-8")) or {}
    assert "disk_space" in raw
    # validate_config_against_schema 对 default.yaml 全量校验，disk_space 缺 schema 会触发告警
    assert C.validate_config_against_schema(raw) == []
