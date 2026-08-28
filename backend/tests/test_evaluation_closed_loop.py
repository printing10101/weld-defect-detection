"""P1 落地：MLOps 闭环接线测试。

- run_golden_evaluation：合成 Golden Set + 假检测器 → 写报告/漂移基线/模型卡/实验；
- POST /api/v1/models/{id}/evaluate：Golden Set 缺失→409，未知模型→404；
- POST /api/v1/evaluation/drift：基线存在→200，基线缺失→409；
- activate 后自动触发 Golden Set 评估（auto_on_activate 接线）。
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.dependencies import get_registry
from backend.app.main import app
from backend.domain.dto import BBox, DefectClass, Detection
from backend.infra.model_registry import ModelRegistry

_CANDIDATES = [
    "_pkg/ScanDetection/models/weights/best.onnx",
    "data/real_label/runs/real_synth2/weights/best.onnx",
    "data/real_label/runs/yolo11n_real_rare/train/weights/best.onnx",
]


def _two_distinct_onnx(tmp_path) -> str:
    found = [p for p in _CANDIDATES if os.path.exists(p)]
    if len(found) < 2:
        pytest.skip("需要两个可加载的 onnx 权重")
    wd = tmp_path / "weights"
    wd.mkdir()
    shutil.copy(found[0], wd / "bestA.onnx")
    shutil.copy(found[1], wd / "bestB.onnx")
    return str(wd)


def _make_golden(golden_dir: Path) -> None:
    """合成 Golden Set：两张小图 + 居中真值标签（与假检测器输出对齐 → mAP≈1）。"""
    import cv2

    (golden_dir / "images").mkdir(parents=True)
    (golden_dir / "labels").mkdir(parents=True)
    for name in ("img1.png", "img2.png"):
        img = np.full((64, 64), 200, dtype=np.uint8)
        cv2.imwrite(str(golden_dir / "images" / name), img)
        (golden_dir / "labels" / f"{Path(name).stem}.txt").write_text(
            "0 0.5 0.5 0.3 0.3\n", encoding="utf-8"
        )


class _FakeDetector:
    """假检测器：每张图回一个居中气孔检测（与合成真值对齐）。"""

    def infer(self, image, conf=None, iou=None, class_conf=None):
        h, w = image.shape[:2]
        bw, bh = int(w * 0.3), int(h * 0.3)
        x, y = (w - bw) // 2, (h - bh) // 2
        return [
            Detection(
                id="d0",
                bbox=BBox(x, y, bw, bh),
                class_id=DefectClass.POROSITY,
                score=0.9,
                uncertainty=0.1,
            )
        ]


def test_run_golden_evaluation_closed_loop(tmp_path: Path) -> None:
    golden_dir = tmp_path / "golden"
    _make_golden(golden_dir)
    eval_dir = tmp_path / "eval"
    exp_dir = tmp_path / "experiments"
    baseline = tmp_path / "drift_baseline.json"

    from backend.evaluation.run_eval import run_golden_evaluation

    summary = run_golden_evaluation(
        "fake::abc123",
        _FakeDetector(),
        golden_dir=golden_dir,
        eval_dir=eval_dir,
        experiments_dir=exp_dir,
        drift_baseline_path=baseline,
        spacing_mm=1.0,
    )
    # 度量
    assert summary["metrics"]["mAP50"] == 1.0
    assert summary["metrics"]["gt_total"] == 2
    # 评估报告落盘（供 models API metric_map）
    assert (eval_dir / "fake__abc123.json").exists()
    # 漂移基线首跑建立（不报漂移）
    assert baseline.exists()
    assert summary["drift"]["drift"] is False
    # 实验追踪 run 落盘
    assert (exp_dir / "experiments.jsonl").exists()
    # 第二次运行应比对基线并报（非基线变更）漂移判定
    summary2 = run_golden_evaluation(
        "fake::abc123",
        _FakeDetector(),
        golden_dir=golden_dir,
        eval_dir=eval_dir,
        experiments_dir=exp_dir,
        drift_baseline_path=baseline,
        spacing_mm=1.0,
    )
    assert summary2["drift"]["drift"] is False  # 同分布 → 无漂移


def test_evaluate_endpoint_409_and_404(tmp_path: Path) -> None:
    wd = _two_distinct_onnx(tmp_path)
    state = str(tmp_path / "state.json")
    reg = get_registry()
    original = reg.model_registry
    reg.model_registry = ModelRegistry(wd, state)
    missing_golden = tmp_path / "no_golden"
    reg.config.eval.golden_dir = str(missing_golden)
    app.dependency_overrides[get_registry] = lambda: reg
    try:
        with TestClient(app) as client:
            ids = [m["id"] for m in client.get("/api/v1/models").json()["models"]]
            # Golden Set 缺失 → 409
            r = client.post(f"/api/v1/models/{ids[0]}/evaluate")
            assert r.status_code == 409
            assert r.json()["detail"]["code"] == "NO_GOLDEN_SET"
            # 未知模型 → 404
            bad = client.post("/api/v1/models/nope::000000000000/evaluate")
            assert bad.status_code == 404
    finally:
        app.dependency_overrides.pop(get_registry, None)
        reg.model_registry = original
        reg.config.eval.golden_dir = "data/eval/golden"


def test_drift_endpoint_baseline_flow(tmp_path: Path) -> None:
    reg = get_registry()
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        '{"size_median_mm": 3.0, "conf_mean": 0.8, "class_ratio": {"0": 1.0}}',
        encoding="utf-8",
    )
    reg.config.eval.drift_baseline_path = str(baseline)
    app.dependency_overrides[get_registry] = lambda: reg
    try:
        with TestClient(app) as client:
            # 基线存在 → 200 + 漂移结果
            r = client.post(
                "/api/v1/evaluation/drift",
                json={"samples": [{"class_id": 0, "score": 0.8, "L_mm": 3.1}] * 10},
            )
            assert r.status_code == 200
            body = r.json()
            assert body["drift"] is False
            # 读取基线
            g = client.get("/api/v1/evaluation/drift/baseline")
            assert g.status_code == 200
    finally:
        app.dependency_overrides.pop(get_registry, None)
        reg.config.eval.drift_baseline_path = "data/eval/drift_baseline.json"


def test_drift_endpoint_no_baseline_409(tmp_path: Path) -> None:
    reg = get_registry()
    reg.config.eval.drift_baseline_path = str(tmp_path / "nonexistent_baseline.json")
    app.dependency_overrides[get_registry] = lambda: reg
    try:
        with TestClient(app) as client:
            r = client.post(
                "/api/v1/evaluation/drift",
                json={"samples": [{"class_id": 0, "score": 0.8, "L_mm": 3.1}]},
            )
            assert r.status_code == 409
            assert r.json()["detail"]["code"] == "NO_BASELINE"
    finally:
        app.dependency_overrides.pop(get_registry, None)
        reg.config.eval.drift_baseline_path = "data/eval/drift_baseline.json"


def test_activate_triggers_auto_eval(monkeypatch, tmp_path: Path) -> None:
    """激活后自动跑 Golden Set 评估（auto_on_activate 接线），不阻塞、fail-soft。"""
    import backend.evaluation.run_eval as re

    wd = _two_distinct_onnx(tmp_path)
    state = str(tmp_path / "state.json")
    reg = get_registry()
    original = reg.model_registry
    reg.model_registry = ModelRegistry(wd, state)
    reg.config.eval.auto_on_activate = True
    called: dict = {}

    def fake_eval(*a, **k):
        called["model_id"] = a[0]
        return {
            "model_id": a[0],
            "metrics": {"mAP50": 0.9},
            "golden_fingerprint": "x",
            "drift": {"drift": False, "alerts": [], "metrics": {}},
            "experiment_run_id": "r",
        }

    monkeypatch.setattr(re, "run_golden_evaluation", fake_eval)
    try:
        ids = reg.model_registry.scan()
        target = ids[1].id
        reg.activate_model(target)
        # 后台守护线程执行；轮询确认被触发
        for _ in range(50):
            if called:
                break
            time.sleep(0.1)
        assert called.get("model_id") == target
    finally:
        reg.model_registry = original
        reg.config.eval.auto_on_activate = True
