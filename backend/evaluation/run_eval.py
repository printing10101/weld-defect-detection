"""Golden Set 评估闭环编排。

把已实现的评估工具（harness / drift / tracking）真正接通到生产路径：
- 加载固定 Golden Set（images + YOLO 真值标签）→ 跑检测器 → 计算 mAP/召回/精确；
- 落盘评估报告（供 models API 的 metric_map）→ 建立/比对漂移基线 → 触发再训练告警；
- 生成模型卡 + 写实验追踪 Run（MLflow 可演进）。

设计要点：
- `run_golden_evaluation` 接受已加载的 `detector`（避免重复加载 ONNX），也可传入
  preprocess_fn 复现生产增强链路，保证 Golden Set mAP 与生产一致；
- Golden Set 缺失 → 抛 FileNotFoundError（端点据此返回 409，提示先准备评估集）；
- 漂移基线首跑自动建立（不报漂移），后续运行与基线比较触发告警；
- 纯 numpy + 标准库，无外部服务依赖，可离线 CI 调用。
"""

from __future__ import annotations

import dataclasses
import json
import logging
import threading
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from backend.evaluation.drift import estimate_drift
from backend.evaluation.harness import (
    detection_metrics,
    golden_set_fingerprint,
    save_eval_report,
)
from backend.evaluation.tracking import ExperimentTracker, build_model_card

_LOG = logging.getLogger("scandetection.eval")

# Golden Set 布局：<golden_dir>/images/{stem}.<ext>  +  <golden_dir>/labels/{stem}.txt
# 标签为 YOLO 格式：每行 "<class_id> <cx> <cy> <w> <h>"（归一化，中心 xy + wh）。

_DEDUP_LOCK = threading.Lock()


def _load_golden(golden_dir: Path) -> list[tuple[Path, list[dict[str, Any]]]]:
    """读取 Golden Set：返回 [(image_path, targets)]。

    targets 为 harness 协议：[{bbox:[x,y,w,h] 绝对像素, class_id:int}]。
    """
    from backend.infra.image_loader import load_image

    img_dir = golden_dir / "images"
    lbl_dir = golden_dir / "labels"
    pairs: list[tuple[Path, list[dict[str, Any]]]] = []
    if not img_dir.is_dir():
        # 兼容平铺：golden_dir 下直接放图 + labels 子目录
        img_dir = golden_dir
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    for img in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in exts):
        stem = img.stem
        targets: list[dict[str, Any]] = []
        lbl = lbl_dir / f"{stem}.txt"
        if lbl.exists():
            gray, _ = load_image(img)
            h_img, w_img = gray.shape[:2]
            for line in lbl.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                cid = int(float(parts[0]))
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
                x = (cx - bw / 2) * w_img
                y = (cy - bh / 2) * h_img
                targets.append(
                    {
                        "bbox": [x, y, bw * w_img, bh * h_img],
                        "class_id": cid,
                    }
                )
        pairs.append((img, targets))
    return pairs


def _aggregate(per_image: list[dict[str, Any]]) -> dict[str, Any]:
    """逐图 metrics 加权聚合（按 gt 数），还原 harness 语义。"""
    if not per_image:
        return {"mAP50": 0.0, "recall": 0.0, "precision": 0.0, "gt_total": 0, "by_class": {}}
    tot_gt = sum(m["gt_total"] for m in per_image) or 1
    mAP = sum(m["mAP50"] * m["gt_total"] for m in per_image) / tot_gt
    rec = sum(m["recall"] * m["gt_total"] for m in per_image) / tot_gt
    prec = sum(m["precision"] * m["gt_total"] for m in per_image) / tot_gt
    by_class: dict[str, Any] = {}
    for m in per_image:
        for cid, cm in m.get("by_class", {}).items():
            agg = by_class.setdefault(
                cid, {"ap50": 0.0, "recall": 0.0, "precision": 0.0, "gt_count": 0}
            )
            g = cm.get("gt_count", 0)
            agg["ap50"] += cm.get("ap50", 0.0) * g
            agg["recall"] += cm.get("recall", 0.0) * g
            agg["precision"] += cm.get("precision", 0.0) * g
            agg["gt_count"] += g
    for agg in by_class.values():
        g = agg["gt_count"] or 1
        agg["ap50"] = round(agg["ap50"] / g, 4)
        agg["recall"] = round(agg["recall"] / g, 4)
        agg["precision"] = round(agg["precision"] / g, 4)
    return {
        "mAP50": round(mAP, 4),
        "recall": round(rec, 4),
        "precision": round(prec, 4),
        "gt_total": sum(m["gt_total"] for m in per_image),
        "by_class": by_class,
    }


def _build_baseline(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """从一批样本聚合漂移基线（尺寸中位数/置信度均值/类别占比）。"""
    sizes = [float(s.get("L_mm", 0.0)) for s in samples]
    confs = [float(s.get("score", 0.0)) for s in samples]
    n = len(samples)
    class_ratio = {}
    if n:
        for cid, cnt in Counter(int(s["class_id"]) for s in samples).items():
            class_ratio[cid] = cnt / n
    return {
        "size_median_mm": float(np.median(sizes)) if sizes else 0.0,
        "conf_mean": float(np.mean(confs)) if confs else 0.0,
        "class_ratio": class_ratio,
    }


def run_golden_evaluation(
    model_id: str,
    detector,  # DefectDetector：含 .infer(image, conf, iou, class_conf=None)
    *,
    golden_dir: str | Path,
    eval_dir: str | Path,
    experiments_dir: str | Path,
    drift_baseline_path: str | Path,
    conf: float | None = None,
    iou: float | None = None,
    class_conf: dict[int, float] | None = None,
    preprocess_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    spacing_mm: float = 1.0,  # Golden Set 多无像素标定，用 1.0（漂移为相对量，单位抵消）
) -> dict[str, Any]:
    """在固定 Golden Set 上评估模型并闭环写报告/漂移/模型卡/实验。

    返回摘要 dict（供 API 与日志）。任何一步失败都上抛，调用方负责降级处理。
    """
    from backend.infra.image_loader import load_image

    golden_dir = Path(golden_dir)
    if not golden_dir.is_dir():
        raise FileNotFoundError(f"golden set dir not found: {golden_dir}")

    pairs = _load_golden(golden_dir)
    if not pairs:
        raise FileNotFoundError(f"golden set empty: {golden_dir}")

    per_image: list[dict[str, Any]] = []
    all_samples: list[dict[str, Any]] = []
    for img_path, targets in pairs:
        gray, _ = load_image(img_path)
        frame = preprocess_fn(gray) if preprocess_fn is not None else gray
        detections = detector.infer(frame, conf=conf, iou=iou, class_conf=class_conf)
        preds = [
            {
                "bbox": [d.bbox.x, d.bbox.y, d.bbox.w, d.bbox.h],
                "class_id": d.class_id.value,
                "score": d.score,
            }
            for d in detections
        ]
        per_image.append(detection_metrics(preds, targets))
        for d in detections:
            all_samples.append(
                {
                    "class_id": d.class_id.value,
                    "score": d.score,
                    "L_mm": float(max(d.bbox.w, d.bbox.h) * spacing_mm),
                }
            )

    metrics = _aggregate(per_image)
    golden_fingerprint = golden_set_fingerprint(golden_dir)
    report_path = save_eval_report(
        model_id,
        metrics,
        eval_dir=eval_dir,
        golden_fingerprint=golden_fingerprint,
    )

    # ---- 漂移基线：首跑建立，后续比较 ----
    drift_baseline_path = Path(drift_baseline_path)
    baseline: dict[str, Any] | None = None
    if drift_baseline_path.exists():
        try:
            baseline = json.loads(drift_baseline_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            baseline = None
    if baseline is None:
        baseline = _build_baseline(all_samples)
        drift_baseline_path.parent.mkdir(parents=True, exist_ok=True)
        drift_baseline_path.write_text(
            json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        drift = {
            "drift": False,
            "alerts": [],
            "metrics": {"n": len(all_samples), "baseline_established": True},
        }
    else:
        drift = dataclasses.asdict(estimate_drift(all_samples, baseline))

    # ---- 模型卡 + 实验追踪 ----
    class_ratio_now = {
        cid: round(r, 4) for cid, r in Counter(int(s["class_id"]) for s in all_samples).items()
    }
    data_summary = {
        "golden_fingerprint": golden_fingerprint,
        "gt_total": metrics["gt_total"],
        "image_count": len(pairs),
        "class_ratio": class_ratio_now,
    }
    model_card = build_model_card(
        model_id=model_id,
        version=model_id.split("::")[-1] if "::" in model_id else model_id,
        metrics=metrics,
        data_summary=data_summary,
        limitations=[
            "仅基于固定 Golden Set 评估，不代表全部工况泛化；",
            "小样本/少标注时 mAP 不稳定，须结合人工复核；",
            "安全关键缺陷（裂纹/未熔合/未焊透）零容忍，模型未检出即漏判风险。",
        ],
        ethics=[
            "本系统为辅助判定，最终级别须由持证评片员确认；",
            "禁止将模型输出直接作为不合格处置依据。",
        ],
    )
    tracker = ExperimentTracker(experiments_dir)
    run_id = tracker.start_run(
        "golden_eval",
        params={"model_id": model_id, "golden_fingerprint": golden_fingerprint},
    )
    tracker.log_metrics(run_id, {k: v for k, v in metrics.items() if isinstance(v, (int, float))})
    tracker.log_artifact(run_id, str(report_path))

    _LOG.info(
        "golden eval model=%s mAP50=%.4f gt=%d drift=%s run=%s",
        model_id,
        metrics["mAP50"],
        metrics["gt_total"],
        drift["drift"],
        run_id,
    )
    return {
        "model_id": model_id,
        "metrics": metrics,
        "golden_fingerprint": golden_fingerprint,
        "drift": drift,
        "model_card": model_card,
        "experiment_run_id": run_id,
        "eval_report_path": str(report_path),
    }
