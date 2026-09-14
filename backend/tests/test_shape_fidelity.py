"""形状保真指标单测（backend/evaluation/harness.shape_fidelity_metrics）。

验证：完美匹配→0；长宽比/尺度/中心三类误差各自可分离；IoU 或类别不达阈值的
不参与匹配；贪心匹配按置信度降序、每个真值只被占用一次。
"""

from __future__ import annotations

from pytest import approx

from backend.evaluation.harness import shape_fidelity_metrics


def _gt(x: float, y: float, w: float, h: float, cid: int = 0) -> dict:
    return {"bbox": [x, y, w, h], "class_id": cid}


def _pred(x: float, y: float, w: float, h: float, cid: int = 0, score: float = 0.9) -> dict:
    return {"bbox": [x, y, w, h], "class_id": cid, "score": score}


def test_perfect_match_is_zero() -> None:
    m = shape_fidelity_metrics([_pred(0, 0, 40, 20)], [_gt(0, 0, 40, 20)])
    assert m["n_matched"] == 1
    assert m["center_err"] == approx(0.0, abs=1e-6)
    assert m["scale_err"] == approx(0.0, abs=1e-6)
    assert m["aspect_err"] == approx(0.0, abs=1e-6)


def test_aspect_error_isolated() -> None:
    """AR 2 → 4（IoU=0.5 恰好匹配），形状误差 log(2)，尺度误差 0.25。"""
    m = shape_fidelity_metrics([_pred(0, 0, 40, 10)], [_gt(0, 0, 40, 20)])
    assert m["n_matched"] == 1
    assert m["aspect_err"] == approx(0.6931, abs=1e-3)
    assert m["scale_err"] == approx(0.25, abs=1e-3)
    assert m["center_err"] == approx(0.1118, abs=1e-3)


def test_center_error_isolated() -> None:
    """仅平移 10px（IoU=0.6），仅定位误差非零。"""
    m = shape_fidelity_metrics([_pred(10, 0, 40, 20)], [_gt(0, 0, 40, 20)])
    assert m["n_matched"] == 1
    assert m["center_err"] == approx(0.2236, abs=1e-3)
    assert m["scale_err"] == approx(0.0, abs=1e-6)
    assert m["aspect_err"] == approx(0.0, abs=1e-6)


def test_scale_error_without_center_shift() -> None:
    """绕中心放大（中心不变，IoU=0.667）：中心误差 0，尺度/形状非零。"""
    m = shape_fidelity_metrics([_pred(-10, 0, 60, 20)], [_gt(0, 0, 40, 20)])
    assert m["n_matched"] == 1
    assert m["center_err"] == approx(0.0, abs=1e-6)
    assert m["scale_err"] == approx(0.25, abs=1e-3)
    assert m["aspect_err"] == approx(0.4055, abs=1e-3)


def test_below_iou_threshold_not_matched() -> None:
    m = shape_fidelity_metrics([_pred(30, 0, 40, 20)], [_gt(0, 0, 40, 20)])
    assert m["n_matched"] == 0
    assert m["center_err"] is None
    assert m["scale_err"] is None
    assert m["aspect_err"] is None


def test_class_mismatch_not_matched() -> None:
    m = shape_fidelity_metrics([_pred(0, 0, 40, 20, cid=1)], [_gt(0, 0, 40, 20, cid=0)])
    assert m["n_matched"] == 0


def test_no_targets_returns_empty() -> None:
    m = shape_fidelity_metrics([_pred(0, 0, 40, 20)], [])
    assert m["n_matched"] == 0
    assert m["aspect_err"] is None


def test_each_target_consumed_once() -> None:
    """两个高置信预测抢同一真值：只能匹配一次（一对一）。"""
    preds = [_pred(0, 0, 40, 20, score=0.9), _pred(5, 0, 40, 20, score=0.8)]
    m = shape_fidelity_metrics(preds, [_gt(0, 0, 40, 20)])
    assert m["n_matched"] == 1


def test_multi_defect_averages() -> None:
    """两处缺陷：完美 + 长宽比错，均值取平均。"""
    preds = [_pred(0, 0, 40, 20, score=0.9), _pred(200, 0, 40, 10, score=0.8)]
    targets = [_gt(0, 0, 40, 20), _gt(200, 0, 40, 20)]
    m = shape_fidelity_metrics(preds, targets)
    assert m["n_matched"] == 2
    assert m["aspect_err"] == approx(0.6931 / 2, abs=1e-3)
