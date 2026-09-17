"""鲁棒性扰动评估（backend.evaluation.robustness）单测。

用可控的假检测器验证评估逻辑本身：恒等检测器应稳定通过，对亮度敏感的
检测器应被正确判 fail；空 GT 走判不通过口径。
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.evaluation.robustness import (
    DEFAULT_CONDITIONS,
    RobustnessCfg,
    perturb,
    robustness_metrics,
)


def _make_sample(seed: int = 0) -> tuple[np.ndarray, list[dict]]:
    """合成 uint16 底片：亮背景 + 一处暗缺陷（GT 框 20x10 @ (50,40)）。"""
    rng = np.random.default_rng(seed)
    img = rng.integers(40000, 50000, size=(100, 200), dtype=np.uint16)
    img[40:50, 50:70] = 8000
    targets = [{"bbox": [50.0, 40.0, 20.0, 10.0], "class_id": 0}]
    return img, targets


def test_perturb_preserves_dtype_shape_and_changes_pixels():
    img, _ = _make_sample()
    for kind, amount in DEFAULT_CONDITIONS:
        out = perturb(img, kind, amount)
        assert out.dtype == img.dtype
        assert out.shape == img.shape
        assert not np.array_equal(out, img), f"{kind}@{amount} 应改变像素"


def test_perturb_identity_amount_is_noop():
    img, _ = _make_sample()
    assert np.array_equal(perturb(img, "brightness_gain", 1.0), img)
    assert np.array_equal(perturb(img, "gamma", 1.0), img)
    assert np.array_equal(perturb(img, "contrast", 1.0), img)
    assert np.array_equal(perturb(img, "brightness_offset", 0.0), img)


def test_perturb_unknown_kind_raises():
    with pytest.raises(ValueError, match="未知扰动类型"):
        perturb(np.zeros((4, 4), dtype=np.uint8), "median_filter", 1.0)


def test_stable_detector_passes():
    """恒等检测器（任何条件下都返回 GT 框）应全条件稳定通过。"""

    def stable(_img):
        return [{"bbox": [50.0, 40.0, 20.0, 10.0], "class_id": 0, "score": 0.9}]

    report = robustness_metrics([_make_sample()], stable)
    assert report.passed, report.failures
    assert report.retention_worst == 1.0
    assert report.quant_dev_worst == 0.0


def test_brightness_sensitive_detector_fails():
    """低亮度增益下漏检的检测器应被判不通过，并给出具体条件名。"""

    def sensitive(img):
        # 欠曝（gain=0.7 → 均值 ~31.5k）时"检不出"：模拟对成像条件敏感的缺陷模型
        if float(img.mean()) < 35000:
            return []
        return [{"bbox": [50.0, 40.0, 20.0, 10.0], "class_id": 0, "score": 0.9}]

    report = robustness_metrics([_make_sample()], sensitive)
    assert not report.passed
    assert any("brightness_gain@0.7" in f for f in report.failures)
    assert report.retention_worst < 1.0


def test_quantization_drift_fails():
    """检出但框尺寸漂移超阈值时，量化偏差判据应触发。"""
    long_box = [50.0, 40.0, 30.0, 10.0]  # 长边 30 vs GT 20 → 偏差 0.5

    def drifted(_img):
        return [{"bbox": long_box, "class_id": 0, "score": 0.9}]

    report = robustness_metrics(
        [_make_sample()], drifted, conditions=(("brightness_gain", 1.2),), cfg=RobustnessCfg()
    )
    assert not report.passed
    assert any("量化平均偏差" in f for f in report.failures)


def test_wrong_class_is_not_matched():
    """类别不一致的检出不计入命中（防止以误检充数）。"""

    def wrong_class(_img):
        return [{"bbox": [50.0, 40.0, 20.0, 10.0], "class_id": 4, "score": 0.9}]

    report = robustness_metrics(
        [_make_sample()], wrong_class, conditions=(("brightness_gain", 1.2),)
    )
    assert report.conditions["brightness_gain@1.2"]["retention"] == 0.0
    assert not report.passed


def test_empty_targets_fails_honestly():
    """无 GT 时不得静默通过（保守口径）。"""

    def any_detector(_img):
        return []

    img = np.zeros((10, 10), dtype=np.uint8)
    report = robustness_metrics([(img, [])], any_detector)
    assert not report.passed
    assert report.failures
