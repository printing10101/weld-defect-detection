"""A2 配置漂移防护回归测试（§部署硬化 配置漂移）。

锁定两类问题：
- DetectCfg.baseline_enabled 安全默认必须为 False（缺键时不应静默落 blob 基线）；
- validate_config_against_schema 能捕获 schema/default.yaml 不一致（缺键→高危；多键→未补 schema）。
"""

from __future__ import annotations

import copy

import yaml

from backend.infra import config as C
from backend.infra.config import DetectCfg


def test_detect_baseline_default_is_false() -> None:
    # 安全默认 = 训练模型路径；缺键时不应静默发 blob 检测器（曾默认 True）
    assert DetectCfg().baseline_enabled is False


def test_schema_drift_detects_missing_key() -> None:
    raw = yaml.safe_load((C._BASE / "default.yaml").read_text(encoding="utf-8")) or {}
    raw2 = copy.deepcopy(raw)
    del raw2["detect"]["baseline_enabled"]
    issues = C.validate_config_against_schema(raw2)
    assert any("detect.baseline_enabled" in i for i in issues)


def test_schema_drift_detects_extra_key() -> None:
    raw = yaml.safe_load((C._BASE / "default.yaml").read_text(encoding="utf-8")) or {}
    raw3 = copy.deepcopy(raw)
    raw3["detect"]["new_unregistered"] = 1
    issues = C.validate_config_against_schema(raw3)
    assert any("new_unregistered" in i for i in issues)


def test_schema_drift_clean_on_current_yaml() -> None:
    raw = yaml.safe_load((C._BASE / "default.yaml").read_text(encoding="utf-8")) or {}
    assert C.validate_config_against_schema(raw) == []
