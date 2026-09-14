"""形状保真接入 Golden 评估闭环的测试。

两部分：
1. ``_aggregate_shape_fidelity`` 纯函数：按匹配对数加权（不是简单取图均值）、
   全无匹配时返回 None（不用 0 冒充"完美"）；
2. 闭环实证——**mAP@0.5 满分而形状失真**：假检测器输出一个"中心略偏、长宽比压扁
   但仍满足 IoU≥0.5"的框，mAP 报 1.0，而 shape_fidelity 的 aspect_err 把盲区量出来。
   这正是引入该指标的动机（§5.4 形状归类 / §6.2 评级不看 mAP）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.domain.dto import BBox, DefectClass, Detection
from backend.evaluation.run_eval import _aggregate_shape_fidelity

# 合成真值（64×64 图，YOLO 归一化 "0 0.5 0.5 0.3 0.3"）换算后：
# 中心 (32.0, 32.0)，尺寸 19.2×19.2 → bbox (22.4, 22.4, 19.2, 19.2)，长宽比 1.0。
_GT_LINE = "0 0.5 0.5 0.3 0.3\n"

# 失真预测：19×12 居中（长宽比 1.583，被压扁），中心右移 1px → (33.0, 32.0)。
# IoU = 217.2 / 379.44 = 0.5724 ≥ 0.5 → 仍算 TP，但形状已失真。
_PRED_DISTORTED = {"x": 23.5, "y": 26.0, "w": 19.0, "h": 12.0}
_EXPECT_ASPECT = 0.4595  # |log((19/12) / (19.2/19.2))| = log(1.5833)
_EXPECT_SCALE = 0.1927  # (|19-19.2|/19.2 + |12-19.2|/19.2) / 2
_EXPECT_CENTER = 0.0368  # hypot(1, 0) / hypot(19.2, 19.2) = 1 / 27.1529


def _make_golden(golden_dir: Path, n_images: int = 1) -> None:
    import cv2

    (golden_dir / "images").mkdir(parents=True)
    (golden_dir / "labels").mkdir(parents=True)
    for i in range(n_images):
        img = np.full((64, 64), 200, dtype=np.uint8)
        cv2.imwrite(str(golden_dir / "images" / f"img{i}.png"), img)
        (golden_dir / "labels" / f"img{i}.txt").write_text(_GT_LINE, encoding="utf-8")


class _DistortedDetector:
    """每张图回一个"对位但长宽比错"的框（mAP 仍满分）。"""

    def infer(self, image, conf=None, iou=None, class_conf=None):
        return [
            Detection(
                id="d0",
                bbox=BBox(**_PRED_DISTORTED),
                class_id=DefectClass.POROSITY,
                score=0.9,
                uncertainty=0.1,
            )
        ]


# ---------------------------------------------------------------------------
# 纯函数：加权聚合
# ---------------------------------------------------------------------------


def test_aggregate_weights_by_matched_pairs() -> None:
    """10 对匹配的图应主导结果，不能被只有 1 对的图等权稀释。"""
    per_image = [
        {"n_matched": 10, "center_err": 0.1, "scale_err": 0.2, "aspect_err": 0.3},
        {"n_matched": 1, "center_err": 10.0, "scale_err": 10.0, "aspect_err": 10.0},
    ]
    agg = _aggregate_shape_fidelity(per_image)
    assert agg["n_matched"] == 11
    # 逐指标加权：(0.1*10 + 10*1)/11 = 1.0；(0.2*10 + 10*1)/11 = 1.0909；
    # (0.3*10 + 10*1)/11 = 1.1818。若误用"图均值"则三者都会是 5.15 → 能被这条测出。
    assert agg["center_err"] == pytest.approx(1.0, abs=1e-4)
    assert agg["scale_err"] == pytest.approx(12 / 11, abs=1e-4)
    assert agg["aspect_err"] == pytest.approx(13 / 11, abs=1e-4)


def test_aggregate_all_unmatched_returns_none() -> None:
    agg = _aggregate_shape_fidelity([{"n_matched": 0, "center_err": None}])
    assert agg["n_matched"] == 0
    assert agg["aspect_err"] is None and agg["scale_err"] is None and agg["center_err"] is None


def test_aggregate_empty_returns_none() -> None:
    agg = _aggregate_shape_fidelity([])
    assert agg["n_matched"] == 0
    assert agg["aspect_err"] is None


def test_aggregate_skips_none_within_image() -> None:
    """某图有匹配对但某指标为 None（真值退化 0 宽高）→ 该指标跳过而非拉低。"""
    per_image = [
        {"n_matched": 2, "center_err": 0.2, "scale_err": 0.4, "aspect_err": None},
        {"n_matched": 3, "center_err": 0.0, "scale_err": 0.0, "aspect_err": None},
    ]
    agg = _aggregate_shape_fidelity(per_image)
    assert agg["n_matched"] == 5
    assert agg["center_err"] == pytest.approx(0.08, abs=1e-4)
    assert agg["aspect_err"] is None


# ---------------------------------------------------------------------------
# 闭环：mAP 满分 ≠ 形状可信
# ---------------------------------------------------------------------------


def test_closed_loop_reports_shape_blind_spot(tmp_path: Path) -> None:
    golden_dir = tmp_path / "golden"
    _make_golden(golden_dir)
    eval_dir = tmp_path / "eval"
    exp_dir = tmp_path / "experiments"

    from backend.evaluation.run_eval import run_golden_evaluation

    summary = run_golden_evaluation(
        "fake::shape01",
        _DistortedDetector(),
        golden_dir=golden_dir,
        eval_dir=eval_dir,
        experiments_dir=exp_dir,
        drift_baseline_path=tmp_path / "drift.json",
        spacing_mm=1.0,
    )

    # 盲区本身：mAP 认为"完全正确"
    assert summary["metrics"]["mAP50"] == 1.0

    sf = summary["shape_fidelity"]
    assert sf["n_matched"] == 1
    assert sf["aspect_err"] == pytest.approx(_EXPECT_ASPECT, abs=1e-3)
    assert sf["scale_err"] == pytest.approx(_EXPECT_SCALE, abs=1e-3)
    assert sf["center_err"] == pytest.approx(_EXPECT_CENTER, abs=1e-3)


def test_closed_loop_persists_report_and_card(tmp_path: Path) -> None:
    """形状保真必须落到评估报告与模型卡（否则前端/下游看不到）。"""
    import json

    golden_dir = tmp_path / "golden"
    _make_golden(golden_dir, n_images=2)
    eval_dir = tmp_path / "eval"
    exp_dir = tmp_path / "experiments"

    from backend.evaluation.run_eval import run_golden_evaluation

    summary = run_golden_evaluation(
        "fake::shape02",
        _DistortedDetector(),
        golden_dir=golden_dir,
        eval_dir=eval_dir,
        experiments_dir=exp_dir,
        drift_baseline_path=tmp_path / "drift.json",
        spacing_mm=1.0,
    )

    # 评估报告（models API 的 metric_map 数据源）
    report = json.loads((eval_dir / "fake__shape02.json").read_text(encoding="utf-8"))
    assert report["shape_fidelity"]["n_matched"] == 2
    assert report["shape_fidelity"]["aspect_err"] == pytest.approx(_EXPECT_ASPECT, abs=1e-3)

    # 模型卡：mAP 与 shape_fidelity 并存，且新增了一条"形状不可信"的局限说明
    card = summary["model_card"]
    assert card["metrics"]["mAP50"] == 1.0
    assert card["metrics"]["shape_fidelity"]["n_matched"] == 2
    assert any("shape_fidelity" in s for s in card["limitations"])

    # 实验追踪：shape_* 标量入 Run
    runs = [json.loads(x) for x in (exp_dir / "experiments.jsonl").read_text().splitlines() if x]
    metrics = runs[-1]["metrics"]
    assert metrics["mAP50"] == 1.0
    assert "shape_aspect_err" in metrics
    assert "shape_center_err" in metrics


def test_gate_untouched_by_shape_fidelity(tmp_path: Path) -> None:
    """回归门禁不受影响：metrics 结构不变，check_regression 仍只看 mAP/recall/precision。"""
    from backend.evaluation.harness import check_regression

    golden_dir = tmp_path / "golden"
    _make_golden(golden_dir)

    from backend.evaluation.run_eval import run_golden_evaluation

    summary = run_golden_evaluation(
        "fake::shape03",
        _DistortedDetector(),
        golden_dir=golden_dir,
        eval_dir=tmp_path / "eval",
        experiments_dir=tmp_path / "experiments",
        drift_baseline_path=tmp_path / "drift.json",
        spacing_mm=1.0,
    )
    # 形状保真缺席 metrics 顶层（它是独立键），故历史 baseline.json 不受影响
    assert "shape_fidelity" not in summary["metrics"]

    baseline = {k: v for k, v in summary["metrics"].items() if isinstance(v, (int, float))}
    gate = check_regression(summary["metrics"], baseline)
    assert gate.passed is True
