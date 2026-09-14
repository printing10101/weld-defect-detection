"""单元测试：POD（按缺陷尺寸的检出概率）曲线。"""

from __future__ import annotations

from backend.evaluation.harness import pod_curve


def _t(cid: int, x: float, y: float, s: float) -> dict:
    return {"class_id": cid, "bbox": [x, y, s, s]}


def _p(cid: int, x: float, y: float, s: float, score: float = 0.9) -> dict:
    return {"class_id": cid, "bbox": [x, y, s, s], "score": score}


def test_empty_targets() -> None:
    out = pod_curve([], [])
    assert out["overall"]["n"] == 0
    assert out["bins"] == []


def test_perfect_detection() -> None:
    targets = [_t(0, 0, 0, 10), _t(0, 50, 50, 20), _t(1, 100, 100, 40)]
    preds = [_p(0, 0, 0, 10), _p(0, 50, 50, 20), _p(1, 100, 100, 40)]
    out = pod_curve(preds, targets)
    assert out["overall"]["pod"] == 1.0
    assert all(b["pod"] == 1.0 for b in out["bins"])


def test_small_defects_missed() -> None:
    """2 大（检出）+ 2 小（漏检）：总体 POD=0.5，小尺寸箱 POD=0。"""
    targets = [_t(0, 0, 0, 8), _t(0, 20, 20, 8), _t(0, 100, 100, 40), _t(0, 200, 200, 60)]
    preds = [_p(0, 100, 100, 40), _p(0, 200, 200, 60)]
    out = pod_curve(preds, targets, iou_threshold=0.5)
    assert out["overall"]["detected"] == 2
    assert out["overall"]["pod"] == 0.5
    smallest = min(out["bins"], key=lambda b: b["size_min_px"])
    assert smallest["pod"] == 0.0
    biggest = max(out["bins"], key=lambda b: b["size_max_px"])
    assert biggest["pod"] == 1.0


def test_class_mismatch_not_detected() -> None:
    """同位置但类别不同的预测不算检出（裂纹检成气孔=漏检）。"""
    targets = [_t(4, 0, 0, 20)]
    preds = [_p(0, 0, 0, 20)]
    out = pod_curve(preds, targets)
    assert out["overall"]["detected"] == 0


def test_low_iou_not_detected() -> None:
    """位置偏移过大的预测不构成检出。"""
    targets = [_t(0, 0, 0, 10)]
    preds = [_p(0, 100, 100, 10)]
    out = pod_curve(preds, targets, iou_threshold=0.5)
    assert out["overall"]["detected"] == 0


def test_identical_sizes_single_bin() -> None:
    """全部同尺寸退化为单箱，样本不丢。"""
    targets = [_t(0, i * 100.0, 0, 12) for i in range(4)]
    preds = [_p(0, i * 100.0, 0, 12) for i in range(4)]
    out = pod_curve(preds, targets)
    assert len(out["bins"]) == 1
    assert out["bins"][0]["n"] == 4
    assert out["bins"][0]["pod"] == 1.0


def test_wilson_ci_bounds() -> None:
    """置信区间有序且落在 [0,1]；POD=0 时上界随样本量收缩。"""
    targets = [_t(0, i * 50.0, 0, 10) for i in range(10)]
    out = pod_curve([], targets)
    assert out["overall"]["pod"] == 0.0
    lo, hi = out["overall"]["ci95"]
    assert 0.0 <= lo <= hi <= 1.0
    assert hi < 0.4  # 10 连败的 Wilson 上界约 0.31
