"""逐类温度校准单测（纯函数 + 检测器阈值协同）。

- temperature_transform：T=1 恒等、T>1 软化、T<1 锐化、数值稳定；
- apply_class_temperature：逐类独立、未列类不变、类数不一致安全跳过；
- grid_search_temperature：欠自信数据应找到 T<1；样本不足诚实跳过；
- parse_calibration_payload：指纹/结构/数值三重校验；
- YoloDetector._eff_thr：分数与阈值同变换（保持检出集合不变的关键）。
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.domain.detect.calibration import (
    MIN_PAIRS_PER_CLASS,
    apply_class_temperature,
    grid_search_temperature,
    parse_calibration_payload,
    temperature_transform,
)
from backend.domain.detect.yolo_detector import YoloDetector


class TestTemperatureTransform:
    def test_identity_at_t1(self):
        p = np.array([0.1, 0.5, 0.9])
        assert temperature_transform(p, 1.0) == pytest.approx(p)

    def test_softening_above_t1(self):
        # T>1 软化：高置信下调、低置信上调（向 0.5 收缩）
        out = temperature_transform(np.array([0.9, 0.3]), 3.0)
        assert out[0] < 0.9 and out[1] > 0.3

    def test_sharpening_below_t1(self):
        out = temperature_transform(np.array([0.9, 0.3]), 0.25)
        assert out[0] > 0.9 and out[1] < 0.3

    def test_monotonic_order_preserved(self):
        p = np.array([0.31, 0.55, 0.82, 0.97])
        for t in (0.25, 0.5, 2.0, 8.0):
            out = temperature_transform(p, t)
            assert list(out) == sorted(out)

    def test_extreme_values_stable(self):
        out = temperature_transform(np.array([0.0, 1.0, 0.5]), 4.0)
        assert np.all(np.isfinite(out)) and np.all((out >= 0) & (out <= 1))

    def test_nonpositive_temp_rejected(self):
        with pytest.raises(ValueError):
            temperature_transform(np.array([0.5]), 0.0)


class TestApplyClassTemperature:
    def test_per_class_independence(self):
        # [anchors, nc]：仅类 0 被锐化，类 1 原样
        scores = np.array([[0.9, 0.9], [0.3, 0.3]])
        out = apply_class_temperature(scores, {0: 0.25}, class_axis=-1)
        assert out[0, 0] > 0.9 and out[1, 0] < 0.3
        assert out[:, 1] == pytest.approx(scores[:, 1])

    def test_class_axis_zero(self):
        # [nc=2, N=2] 布局：类 0 被锐化，类 1 原样
        scores = np.array([[0.9, 0.9], [0.4, 0.4]])
        out = apply_class_temperature(scores, {0: 0.25}, class_axis=0)
        assert out[0, 0] > 0.9
        assert out[1] == pytest.approx(scores[1])

    def test_unknown_class_skipped_safely(self):
        scores = np.array([[0.9, 0.9]])
        out = apply_class_temperature(scores, {5: 0.5}, class_axis=-1)  # 模型只有 2 类
        assert out == pytest.approx(scores)

    def test_empty_temps_identity(self):
        scores = np.array([[0.7, 0.2]])
        assert apply_class_temperature(scores, {}, class_axis=-1) == pytest.approx(scores)


def _synthetic_underconfident(n: int = 80, seed: int = 3) -> tuple[list[float], list[bool]]:
    """欠自信样本：置信度 ~U(0.6,0.85)，正确率 90%。"""
    rng = np.random.default_rng(seed)
    confs = rng.uniform(0.6, 0.85, n)
    oks = rng.random(n) < 0.9
    return confs.tolist(), oks.tolist()


class TestGridSearchTemperature:
    def test_underconfident_data_finds_sharpening(self):
        confs, oks = _synthetic_underconfident()
        r = grid_search_temperature(confs, oks)
        assert not r["skipped"]
        assert r["best_temp"] < 1.0
        assert r["ece_after"] < r["ece_before"]

    def test_insufficient_samples_skips_honestly(self):
        confs, oks = _synthetic_underconfident(MIN_PAIRS_PER_CLASS - 1)
        r = grid_search_temperature(confs, oks)
        assert r["skipped"] and r["best_temp"] == 1.0

    def test_empty_input_skips(self):
        r = grid_search_temperature([], [])
        assert r["skipped"] and r["n"] == 0


class TestParseCalibrationPayload:
    def test_valid_payload(self):
        temps = parse_calibration_payload(
            {"model_id": "best::abc123", "temperatures": {"0": 0.25, "3": 2.0}},
            "best::abc123",
        )
        assert temps == {0: 0.25, 3: 2.0}

    def test_model_id_mismatch_returns_none(self):
        assert (
            parse_calibration_payload(
                {"model_id": "best::old", "temperatures": {"0": 0.5}}, "best::new"
            )
            is None
        )

    def test_bad_structure_returns_none(self):
        assert parse_calibration_payload({"temperatures": {"0": 1.0}}, "m") is None
        assert parse_calibration_payload({"model_id": "m", "temperatures": {}}, "m") is None
        assert parse_calibration_payload({"model_id": "m", "temperatures": {"x": 1.0}}, "m") is None
        assert (
            parse_calibration_payload({"model_id": "m", "temperatures": {"0": -1.0}}, "m") is None
        )
        assert parse_calibration_payload("not-a-dict", "m") is None


class TestDetectorEffThr:
    def test_threshold_transformed_with_scores(self):
        det = YoloDetector()
        det.class_temperature = {0: 0.25}
        # 逐类阈值 0.30，锐化后同向变换 → 与校准分数比较时检出集合不变
        raw = det._thr_for(0, 0.5, {0: 0.30})
        cal = det._eff_thr(0, 0.5, {0: 0.30})
        assert raw == pytest.approx(0.30)
        assert cal == pytest.approx(float(temperature_transform(np.array([raw]), 0.25)[0]))
        assert cal < raw  # T<1 锐化把 0.30 拉向两端 → 阈值下降

    def test_unlisted_class_identity(self):
        det = YoloDetector()
        det.class_temperature = {0: 0.25}
        assert det._eff_thr(2, 0.5, None) == det._thr_for(2, 0.5, None)

    def test_no_calibration_identity(self):
        det = YoloDetector()
        assert det._eff_thr(0, 0.3, None) == 0.3
