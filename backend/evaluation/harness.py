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
from itertools import pairwise
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


# ---------------------------------------------------------------------------
# 形状保真（补 mAP@0.5 的边界/形状盲区）
# ---------------------------------------------------------------------------


def shape_fidelity_metrics(
    preds: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """形状保真指标：量化检测框的定位/尺度/长宽比误差。

    动机（受 Marigold V2 的 SEE 指标启发——**聚合误差会掩盖边界与形状质量**）：
    mAP@0.5 只要求 IoU≥阈值，"框盖住缺陷但长宽比错"仍计为 TP；而本系统 §5.4
    按 ``aspect_ratio = L/W``（≤3 圆形 / >3 条形）归类、§6.2 据此评级——形状
    误差会直接改变评级结论（细长裂纹尤甚），却完全不进现有指标。

    对同类别、IoU≥阈值的一对一匹配（按置信度降序，取 IoU 最高的真值），逐对
    量化三类误差：
    - ``center_err``：中心欧氏距离 / 真值对角线（定位误差，纯平移）；
    - ``scale_err`` ：宽、高相对误差的均值（尺度误差，纯缩放）；
    - ``aspect_err``：``|log(预测宽高比 / 真值宽高比)|``（形状误差，对称，0=一致）。
    三者 0 为完美、越大越差。无匹配时 ``n_matched=0`` 且各值为 ``None``。
    """
    used = [False] * len(targets)
    matched: list[tuple[list[float], list[float]]] = []
    for p in sorted(preds, key=lambda d: -d["score"]):
        best_i = -1
        best_iou = iou_threshold  # 需 ≥ 阈值
        for gi, g in enumerate(targets):
            if used[gi] or g["class_id"] != p["class_id"]:
                continue
            v = iou(p["bbox"], g["bbox"])
            if v >= best_iou:
                best_iou = v
                best_i = gi
        if best_i < 0:
            continue
        used[best_i] = True
        matched.append((p["bbox"], targets[best_i]["bbox"]))

    if not matched:
        return {"n_matched": 0, "center_err": None, "scale_err": None, "aspect_err": None}

    centers: list[float] = []
    scales: list[float] = []
    aspects: list[float] = []
    for pb, gb in matched:
        gw, gh = float(gb[2]), float(gb[3])
        pw, ph = float(pb[2]), float(pb[3])
        diag = float(np.hypot(gw, gh))
        if diag > 0:
            dcx = (pb[0] + pw / 2) - (gb[0] + gw / 2)
            dcy = (pb[1] + ph / 2) - (gb[1] + gh / 2)
            centers.append(float(np.hypot(dcx, dcy)) / diag)
        if gw > 0 and gh > 0:
            scales.append((abs(pw - gw) / gw + abs(ph - gh) / gh) / 2.0)
        if gw > 0 and gh > 0 and pw > 0 and ph > 0:
            aspects.append(abs(float(np.log((pw / ph) / (gw / gh)))))

    return {
        "n_matched": len(matched),
        "center_err": round(float(np.mean(centers)), 4) if centers else None,
        "scale_err": round(float(np.mean(scales)), 4) if scales else None,
        "aspect_err": round(float(np.mean(aspects)), 4) if aspects else None,
    }


# ---------------------------------------------------------------------------
# POD（Probability of Detection，按缺陷尺寸的检出概率）
# ---------------------------------------------------------------------------


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 得分区间（小样本下比正态近似诚实，NDT 可靠性口径通用）。"""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / denom
    return max(0.0, center - half), min(1.0, center + half)


def pod_curve(
    preds: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    iou_threshold: float = 0.5,
    *,
    n_bins: int = 4,
) -> dict[str, Any]:
    """按真值缺陷尺寸分箱的操作点检出概率（POD）曲线。

    POD 是 NDT 行业的可靠性通用语言（缺陷尺寸 → 检出概率，MIL-HDBK-1823A
    口径）；本实现给出**工作点 POD**：某真值在操作阈值下存在同类别、
    IoU≥阈值的预测即记检出。尺寸取特征长度 sqrt(面积)（px），按分位数
    等频分箱（各箱样本量均衡，小数据集不因固定宽度分箱出现空箱）。

    返回 {overall: {n, detected, pod, ci95}, bins: [{size_min_px, size_max_px,
    n, detected, pod, ci95}]}；无真值时 overall.n=0、bins=[]。
    """
    sizes = [float(np.sqrt(_bbox_area(t["bbox"]))) for t in targets]
    if not sizes:
        return {"overall": {"n": 0, "detected": 0, "pod": 0.0, "ci95": [0.0, 1.0]}, "bins": []}

    # 逐真值判定检出：同类别且与任一预测 IoU≥阈值（预测已按操作阈值过滤）
    detected_flags: list[bool] = []
    for t in targets:
        detected_flags.append(
            any(
                p["class_id"] == t["class_id"] and iou(p["bbox"], t["bbox"]) >= iou_threshold
                for p in preds
            )
        )

    # 分位数分箱：重复边界去重（尺寸高度集中时退化为单箱）
    edges = np.quantile(np.asarray(sizes), np.linspace(0.0, 1.0, n_bins + 1))
    uniq_edges = sorted({round(float(e), 6) for e in edges})
    if len(uniq_edges) == 1:  # 全部同尺寸：单箱覆盖，避免"有 GT 无分箱"
        uniq_edges = [uniq_edges[0], uniq_edges[0]]

    bins: list[dict[str, Any]] = []
    last_edge = uniq_edges[-1]
    for lo, hi in pairwise(uniq_edges):
        # 半开区间 [lo, hi)；末箱闭合（s ≥ 最大边界的样本全归末箱）
        idx = [i for i, s in enumerate(sizes) if lo <= s < hi or (hi == last_edge and s >= hi)]
        k = sum(detected_flags[i] for i in idx)
        n = len(idx)
        if n == 0:
            continue  # 分位数边界重合会挤出空箱（如 n=3 请求 4 箱），跳过
        ci_lo, ci_hi = _wilson_ci(k, n)
        bins.append(
            {
                "size_min_px": round(lo, 2),
                "size_max_px": round(hi, 2),
                "n": n,
                "detected": int(k),
                "pod": round(k / n, 4) if n else 0.0,
                "ci95": [round(ci_lo, 4), round(ci_hi, 4)],
            }
        )

    k_all = sum(detected_flags)
    n_all = len(targets)
    ci_lo, ci_hi = _wilson_ci(k_all, n_all)
    overall = {
        "n": n_all,
        "detected": int(k_all),
        "pod": round(k_all / n_all, 4) if n_all else 0.0,
        "ci95": [round(ci_lo, 4), round(ci_hi, 4)],
    }
    return {"overall": overall, "bins": bins}


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
