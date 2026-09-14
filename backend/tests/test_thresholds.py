"""单元测试：检测工作模式阈值解析（高检出/高准确双档）。"""

from __future__ import annotations

import pytest

from backend.domain.detect.thresholds import (
    BALANCED,
    PRECISION_FIRST,
    RECALL_FIRST,
    resolve_class_conf,
)

_CLASS_CONF = {0: 0.30, 1: 0.12, 4: 0.05}


def test_balanced_is_identity() -> None:
    """balanced 模式阈值原样透传（历史行为不变）。"""
    conf, cc = resolve_class_conf(
        0.3, dict(_CLASS_CONF), BALANCED, recall_scale=0.5, precision_scale=1.8
    )
    assert conf == 0.3
    assert cc == _CLASS_CONF


def test_recall_first_relaxes_and_clips() -> None:
    """recall_first 整体放宽；缩放到下限以下裁剪到 0.01。"""
    conf, cc = resolve_class_conf(
        0.3, dict(_CLASS_CONF), RECALL_FIRST, recall_scale=0.1, precision_scale=1.8
    )
    assert conf == pytest.approx(0.03)
    assert cc[0] == pytest.approx(0.03)
    assert cc[1] == pytest.approx(0.012)
    assert cc[4] == 0.01  # 0.05 * 0.1 = 0.005 → 裁到下限


def test_precision_first_tightens() -> None:
    """precision_first 整体收紧；类间相对次序不变（裂纹仍最低）。"""
    conf, cc = resolve_class_conf(
        0.3, dict(_CLASS_CONF), PRECISION_FIRST, recall_scale=0.5, precision_scale=1.8
    )
    assert conf == pytest.approx(0.54)
    assert cc[0] == pytest.approx(0.54)
    assert cc[1] == pytest.approx(0.216)
    assert cc[4] < cc[1] < cc[0]


def test_unknown_mode_raises() -> None:
    """未知模式显式抛错——静默回落 balanced 会让人误以为换了档。"""
    with pytest.raises(ValueError, match="未知检测模式"):
        resolve_class_conf(0.3, {}, "aggressive", recall_scale=0.5, precision_scale=1.8)


def test_input_mapping_not_mutated() -> None:
    """输入映射防御拷贝：解析不得污染共享配置对象。"""
    cc_in = dict(_CLASS_CONF)
    resolve_class_conf(0.3, cc_in, PRECISION_FIRST, recall_scale=0.5, precision_scale=1.8)
    assert cc_in == _CLASS_CONF


def test_empty_class_conf_falls_back() -> None:
    """空逐类表回落 infer_conf（缩放后），不炸不伪造键。"""
    conf, cc = resolve_class_conf(0.3, {}, RECALL_FIRST, recall_scale=0.5, precision_scale=1.8)
    assert conf == pytest.approx(0.15)
    assert cc == {}
