"""单元测试：评估 Harness + 回归门禁 + 漂移监控 + 实验追踪 + 模型卡。"""

from __future__ import annotations

from pathlib import Path

from backend.evaluation import drift, harness, tracking


def _pred(bbox, class_id: int, score: float) -> dict:
    return {"bbox": bbox, "class_id": class_id, "score": score}


def _gt(bbox, class_id: int) -> dict:
    return {"bbox": bbox, "class_id": class_id}


# ---------------------------------------------------------------------------
# 评估 Harness
# ---------------------------------------------------------------------------


def test_iou_basic() -> None:
    a = [0.0, 0.0, 10.0, 10.0]
    assert harness.iou(a, a) == 1.0
    assert harness.iou(a, [10, 0, 10, 10]) == 0.0  # 无交叠
    assert harness.iou(a, [5, 0, 10, 10]) > 0.0  # 半重叠


def test_detection_metrics_perfect() -> None:
    preds = [_pred([0, 0, 10, 10], 0, 0.9), _pred([100, 100, 10, 10], 1, 0.8)]
    targets = [_gt([0, 0, 10, 10], 0), _gt([100, 100, 10, 10], 1)]
    m = harness.detection_metrics(preds, targets)
    assert m["mAP50"] == 1.0
    assert m["recall"] == 1.0
    assert m["gt_total"] == 2
    assert m["by_class"]["0"]["ap50"] == 1.0


def test_detection_metrics_miss() -> None:
    """漏检一个类 → mAP/recall 下降；精确保留（无假阳）。"""
    preds = [_pred([0, 0, 10, 10], 0, 0.9)]
    targets = [_gt([0, 0, 10, 10], 0), _gt([100, 100, 10, 10], 1)]  # 类 1 全漏
    m = harness.detection_metrics(preds, targets)
    assert m["mAP50"] < 1.0
    assert m["recall"] < 1.0
    assert m["by_class"]["1"]["recall"] == 0.0


def test_check_regression_blocks_on_map_drop() -> None:
    """：mAP 下降 >1.0 点（0.01）→ 阻断。"""
    baseline = {"mAP50": 0.90, "recall": 0.90, "precision": 0.90}
    current = {"mAP50": 0.88, "recall": 0.90, "precision": 0.90}  # -0.02
    gate = harness.check_regression(current, baseline)
    assert gate.passed is False
    assert any("mAP" in v for v in gate.violations)
    assert gate.deltas["mAP50"] == -0.02


def test_check_regression_passes_within_tolerance() -> None:
    baseline = {"mAP50": 0.90, "recall": 0.90, "precision": 0.90}
    current = {"mAP50": 0.895, "recall": 0.91, "precision": 0.90}  # -0.005 容差内
    gate = harness.check_regression(current, baseline)
    assert gate.passed is True
    assert gate.violations == []


def test_golden_set_fingerprint_stable_and_sensitive(tmp_path: Path) -> None:
    """Golden Set 指纹：同内容稳定；改文件即变（版本化语义）。"""
    d = tmp_path / "golden"
    d.mkdir()
    (d / "a.txt").write_text("x", encoding="utf-8")
    (d / "b.txt").write_text("y", encoding="utf-8")
    fp1 = harness.golden_set_fingerprint(d)
    fp2 = harness.golden_set_fingerprint(d)
    assert fp1 == fp2
    (d / "b.txt").write_text("z", encoding="utf-8")
    fp3 = harness.golden_set_fingerprint(d)
    assert fp1 != fp3


def test_eval_report_roundtrip(tmp_path: Path) -> None:
    """评估报告落盘 → 可读回（models API metric_map 数据源）。"""
    path = harness.save_eval_report(
        "best::abc123",
        {"mAP50": 0.85, "recall": 0.80, "precision": 0.9},
        eval_dir=tmp_path,
        golden_fingerprint="fp123",
    )
    assert path.exists()
    report = harness.load_eval_report("best::abc123", tmp_path)
    assert report is not None
    assert report["metrics"]["mAP50"] == 0.85
    assert report["golden_set"] == "fp123"
    assert harness.load_eval_report("missing", tmp_path) is None


# ---------------------------------------------------------------------------
# 漂移监控
# ---------------------------------------------------------------------------


def test_drift_detects_size_shift() -> None:
    baseline = {"size_median_mm": 3.0, "conf_mean": 0.8, "class_ratio": {0: 1.0}}
    # 新样本尺寸普遍 6mm（中位数偏移 100% > 25%）
    new = [{"class_id": 0, "score": 0.8, "L_mm": 6.0} for _ in range(10)]
    res = drift.estimate_drift(new, baseline)
    assert res.drift is True
    assert any("尺寸" in a for a in res.alerts)


def test_drift_no_alert_within_tolerance() -> None:
    baseline = {"size_median_mm": 3.0, "conf_mean": 0.8, "class_ratio": {0: 1.0}}
    new = [{"class_id": 0, "score": 0.8, "L_mm": 3.1} for _ in range(10)]
    res = drift.estimate_drift(new, baseline)
    assert res.drift is False
    assert res.alerts == []


def test_drift_empty_samples_no_alert() -> None:
    res = drift.estimate_drift([], {"size_median_mm": 3.0, "conf_mean": 0.8, "class_ratio": {}})
    assert res.drift is False
    assert res.metrics["n"] == 0


# ---------------------------------------------------------------------------
# 实验追踪 + 模型卡 + 数据版本
# ---------------------------------------------------------------------------


def test_experiment_tracker(tmp_path: Path) -> None:
    tracker = tracking.ExperimentTracker(tmp_path / "experiments")
    run_id = tracker.start_run("real_synth2", params={"epochs": 80, "lr": 0.001})
    tracker.log_metrics(run_id, {"mAP50": 0.86, "recall": 0.9})
    tracker.log_artifact(run_id, "runs/x/best.pt")
    runs = tracker.list_runs()
    assert len(runs) == 1
    assert runs[0]["params"]["epochs"] == 80
    assert runs[0]["metrics"]["mAP50"] == 0.86
    assert any("best.pt" in a for a in runs[0]["artifacts"])
    got = tracker.get_run(run_id)
    assert got is not None
    assert got["run_id"] == run_id


def test_build_model_card() -> None:
    card = tracking.build_model_card(
        model_id="best::abc",
        version="v1",
        metrics={"mAP50": 0.85},
        data_summary={"classes": 6, "images": 1000},
        limitations=["小样本稀有类泛化有限"],
        ethics=["仅用于持证评片员辅助，不替代最终判定"],
    )
    assert card["metrics"]["mAP50"] == 0.85
    assert card["data_distribution"]["classes"] == 6
    assert card["limitations"]
    assert card["ethics"]
    assert "created_at" in card


def test_dataset_fingerprint(tmp_path: Path) -> None:
    """数据版本（DVC 轻量替代）：同数据集稳定指纹，改文件即变。"""
    d = tmp_path / "dataset"
    d.mkdir()
    (d / "img1.png").write_bytes(b"\x89PNG\x0d\x0a" + b"x" * 100)
    fp1 = tracking.dataset_fingerprint(d)
    assert fp1 == tracking.dataset_fingerprint(d)
    (d / "img2.png").write_bytes(b"\x89PNG\x0d\x0a" + b"y" * 100)
    fp2 = tracking.dataset_fingerprint(d)
    assert fp1 != fp2
