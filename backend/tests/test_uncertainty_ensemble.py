"""增广集成不确定性（TTA 路径）单测。

estimate_ensemble_uncertainty / ensemble_uncertainty_stats 的纯函数语义：
- 全视角一致检出 → 低不确定性；缺视角检出 → 高不确定性；
- 得分跨视角剧烈波动 → 高不确定性；
- 匹配按"同类 + IoU 达标"，跨类不算检出。
"""

from __future__ import annotations

import pytest

from backend.domain.detect.uncertainty import (
    ensemble_uncertainty_stats,
    estimate_ensemble_uncertainty,
)
from backend.domain.dto import BBox


def _box(x: float, y: float, w: float = 20.0, h: float = 10.0) -> BBox:
    return BBox(x=x, y=y, w=w, h=h)


class TestEstimateEnsembleUncertainty:
    def test_full_agreement_zero(self):
        assert estimate_ensemble_uncertainty(1.0, 0.0) == 0.0

    def test_missing_votes_raise(self):
        # 一半视角未检出 → 0.5；三分之二未检出 → ~0.667
        assert estimate_ensemble_uncertainty(0.5, 0.0) == 0.5
        assert estimate_ensemble_uncertainty(1 / 3, 0.0) == pytest.approx(0.6667, abs=1e-3)

    def test_score_spread_dominates(self):
        # std 达到归一常数（默认 0.15）即满分
        assert estimate_ensemble_uncertainty(1.0, 0.15) == 1.0
        assert estimate_ensemble_uncertainty(1.0, 0.075) == 0.5

    def test_clamped_to_unit_interval(self):
        assert estimate_ensemble_uncertainty(-0.5, 99.0) == 1.0
        assert estimate_ensemble_uncertainty(2.0, -1.0) == 0.0

    def test_custom_spread_norm(self):
        assert estimate_ensemble_uncertainty(1.0, 0.3, spread_norm=0.3) == 1.0


class TestEnsembleUncertaintyStats:
    def _views(self):
        """3 视角：缺陷 A 三视角全检出（0.9/0.9/0.9），缺陷 B 仅视角 1 检出。"""
        det_a1 = (_box(10, 10), 0, 0.9)
        det_a2 = (_box(11, 10), 0, 0.9)  # 微小漂移，IoU 仍高
        det_a3 = (_box(10, 11), 0, 0.9)
        det_b1 = (_box(100, 100), 1, 0.7)
        return [
            [det_a1, det_b1],
            [det_a2],
            [det_a3],
        ]

    def test_stable_defect_full_votes_no_spread(self):
        stats = ensemble_uncertainty_stats([(_box(10, 10), 0)], self._views(), match_iou=0.5)
        ((vote, std),) = stats
        assert vote == 1.0
        assert std == 0.0

    def test_single_view_defect_low_votes(self):
        stats = ensemble_uncertainty_stats([(_box(100, 100), 1)], self._views(), match_iou=0.5)
        ((vote, std),) = stats
        assert vote == pytest.approx(1 / 3)
        assert std == 0.0

    def test_cross_class_not_matched(self):
        # kept 为类 1 的框，视角内只有类 0 检出在相同位置 → 不算检出
        views = [[(_box(10, 10), 0, 0.9)], [], []]
        stats = ensemble_uncertainty_stats([(_box(10, 10), 1)], views, match_iou=0.5)
        ((vote, _std),) = stats
        assert vote == 0.0

    def test_iou_threshold_filters_distant_match(self):
        views = [[(_box(200, 200), 0, 0.9)], [], []]
        stats = ensemble_uncertainty_stats([(_box(10, 10), 0)], views, match_iou=0.5)
        ((vote, _std),) = stats
        assert vote == 0.0

    def test_empty_views_never_crash(self):
        stats = ensemble_uncertainty_stats([(_box(0, 0), 0)], [], match_iou=0.5)
        ((vote, _std),) = stats
        assert vote == 0.0  # n_views 下限 1，不除零
