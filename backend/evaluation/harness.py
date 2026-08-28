"""评估 Harness 与回归门禁。

设计文档要求：
- Golden Set：固定、版本化、禁止用于训练；每次提交自动评估；
- 回归门禁：PR 不得使 mAP@0.5 下降 >1.0 点，否则阻断合并；
- 评估 Harness：固定测试集 + 持续评估脚本。

本模块提供**纯 numpy 实现**的检测评估（IoU 匹配 → AP/mAP@0.5/召回/精确），
不依赖 sklearn/onnxruntime，可离线单测；Golden Set 以"目录 + 内容哈希指纹"
轻量实现（DVC 的数据版本语义的本地替代，见  数据版本）。

输入协议（preds/targets）：
- 预测: list[dict]  {bbox:[x,y,w,h], class_id:int, score:float}
- 真值: list[dict]  {bbox:[x,y,w,h], class_id:int}
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GateResult:
    """回归门禁结果：任一指标超差即阻断。"""

    passed: bool
    deltas: dict[str, float]  # 当前 - 基线（负=退化）
    violations: list[str]  # 触发的违约描述


def _bbox_area(bbox: list[float]) -> float:
    return max(0.0, bbox[2]) * max(0.0, bbox[3])


def iou(a: list[float], b: list[float]) -> float:
    """两个 [x,y,w,h] 框的 IoU（纯 numpy）。"""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[0] + a[2], b[0] + b[2])
    y2 = min(a[1] + a[3], b[1] + b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = _bbox_area(a) + _bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def _ap(precisions: np.ndarray, recalls: np.ndarray) -> float:
    """AP 的 11 点插值近似（PASCAL VOC 风格，够作回归门禁基线）。"""
    if len(precisions) == 0:
        return 0.0
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 11):
        p = precisions[recalls >= t]
        ap += float(p.max()) if len(p) > 0 else 0.0
    return ap / 11.0


def _class_ap(
    preds: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    iou_threshold: float,
) -> tuple[float, float, float, int]:
    """单类 AP/召回/精确（按置信度降序贪心匹配 GT）。"""
    if not preds:
        n_gt = len(targets)
        return 0.0, 0.0, 0.0, n_gt
    sorted_preds = sorted(preds, key=lambda p: -p["score"])
    used: list[bool] = [False] * len(targets)
    tp: list[bool] = []
    for p in sorted_preds:
        matched = False
        for gi, g in enumerate(targets):
            if used[gi]:
                continue
            if iou(p["bbox"], g["bbox"]) >= iou_threshold:
                used[gi] = True
                matched = True
                break
        tp.append(matched)
    tp_arr = np.array(tp, dtype=float)
    fp_arr = (~np.array(tp)).astype(float)
    cum_tp = np.cumsum(tp_arr)
    cum_fp = np.cumsum(fp_arr)
    n_gt = len(targets)
    recalls = cum_tp / n_gt if n_gt > 0 else np.zeros_like(cum_tp)
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
    ap = _ap(precisions, recalls)
    recall = float(cum_tp[-1] / n_gt) if n_gt > 0 else 0.0
    precision = float(cum_tp[-1] / max(cum_tp[-1] + cum_fp[-1], 1e-9)) if tp else 0.0
    return ap, recall, precision, n_gt


def detection_metrics(
    preds: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """计算 mAP@0.5 / 召回 / 精确（逐类加权， 门禁依据）。"""
    by_class: dict[int, dict[str, list]] = {}
    for p in preds:
        by_class.setdefault(p["class_id"], {"preds": [], "targets": []})["preds"].append(p)
    for t in targets:
        by_class.setdefault(t["class_id"], {"preds": [], "targets": []})["targets"].append(t)

    class_metrics: dict[str, Any] = {}
    total_ap, total_recall, total_precision, total_gt = 0.0, 0.0, 0.0, 0
    for cid, group in by_class.items():
        ap, recall, precision, n_gt = _class_ap(group["preds"], group["targets"], iou_threshold)
        class_metrics[str(cid)] = {
            "ap50": round(ap, 4),
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "gt_count": n_gt,
        }
        total_ap += ap * n_gt
        total_recall += recall * n_gt
        total_precision += precision * n_gt
        total_gt += n_gt

    if total_gt == 0:
        return {
            "mAP50": 0.0,
            "recall": 0.0,
            "precision": 0.0,
            "gt_total": 0,
            "by_class": class_metrics,
        }
    return {
        "mAP50": round(total_ap / total_gt, 4),
        "recall": round(total_recall / total_gt, 4),
        "precision": round(total_precision / total_gt, 4),
        "gt_total": total_gt,
        "by_class": class_metrics,
    }


def check_regression(
    current: dict[str, Any],
    baseline: dict[str, Any],
    *,
    mAP_tolerance: float = 0.01,  # §15.6：mAP@0.5 下降 >1.0 点阻断
    recall_tolerance: float = 0.02,
    precision_tolerance: float = 0.02,
) -> GateResult:
    """回归门禁：当前 vs 基线，任一指标退化超容差即阻断。"""
    deltas = {
        "mAP50": round(current["mAP50"] - baseline["mAP50"], 4),
        "recall": round(current["recall"] - baseline["recall"], 4),
        "precision": round(current["precision"] - baseline["precision"], 4),
    }
    violations: list[str] = []
    if deltas["mAP50"] < -mAP_tolerance:
        violations.append(f"mAP@0.5 下降 {abs(deltas['mAP50']):.4f} > {mAP_tolerance}")
    if deltas["recall"] < -recall_tolerance:
        violations.append(f"recall 下降 {abs(deltas['recall']):.4f} > {recall_tolerance}")
    if deltas["precision"] < -precision_tolerance:
        violations.append(f"precision 下降 {abs(deltas['precision']):.4f} > {precision_tolerance}")
    return GateResult(passed=not violations, deltas=deltas, violations=violations)


# ---------------------------------------------------------------------------
# Golden Set 轻量版本化
# ---------------------------------------------------------------------------


def golden_set_fingerprint(directory: str | Path) -> str:
    """Golden Set 内容指纹：对目录内文件（排序后）逐文件 sha256 再整体哈希。

    固定测试集必须版本化：任何文件增删改都会改变指纹，从而在评估记录中
    留下 Golden Set 版本痕迹（禁止用于训练，版本变更需显式记录）。
    """
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"golden set dir not found: {directory}")
    files = sorted(p for p in root.rglob("*") if p.is_file())
    h = hashlib.sha256()
    for p in files:
        rel = str(p.relative_to(root))
        h.update(rel.encode("utf-8"))
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# 评估报告落盘（供 models API 展示 metric_map， 模型卡）
# ---------------------------------------------------------------------------


def save_eval_report(
    model_id: str,
    metrics: dict[str, Any],
    *,
    eval_dir: str | Path,
    golden_fingerprint: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """把一次评估指标落盘为 data/eval/{model_id}.json（含 Golden Set 指纹）。

    models API 读取该文件填充 ModelInfo.metric_map，形成"评估→模型卡"闭环。
    model_id 含 '::'（registry 指纹格式），文件名转义为 '__'（Windows 安全）。
    """
    root = Path(eval_dir)
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": model_id,
        "metrics": metrics,
        "golden_set": golden_fingerprint,
        **(extra or {}),
    }
    path = root / f"{model_id.replace('::', '__')}.json"
    import json

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_eval_report(model_id: str, eval_dir: str | Path) -> dict[str, Any] | None:
    """读取某模型的最近评估报告；无则返回 None。"""
    path = Path(eval_dir) / f"{model_id.replace('::', '__')}.json"
    if not path.exists():
        return None
    import json

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
