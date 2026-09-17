"""亮度/对比度扰动鲁棒性评估（任务1"质量与鲁棒性验证"）。

技术路线要求：同一缺陷在不同图像条件下保持稳定检出与量化。本模块用
亮度增益/偏移、Gamma、对比度四族灰度扰动模拟扫描仪/曝光/工艺差异，
在带 GT 的评估集上量化检出保持率与量化偏差，作为可回归的验收指标。

与 harness.py 同一输入协议（preds/targets: {bbox:[x,y,w,h], class_id, score}），
纯 numpy/cv2 实现、可离线单测；推理经 ``infer_fn`` 回调注入（依赖倒置，
不绑定具体检测器）。空 GT 判不通过的保守口径与 agreement.py 一致。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from backend.evaluation.harness import iou

# 推理回调：image → preds（与 harness 输入协议一致）
InferFn = Callable[[np.ndarray], list[dict[str, Any]]]


# ---------------------------------------------------------------------------
# 扰动族（模拟不同图像条件）
# ---------------------------------------------------------------------------
def _max_value(dtype: np.dtype) -> float:
    return float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else 1.0


def perturb(image: np.ndarray, kind: str, amount: float) -> np.ndarray:
    """对灰度图施加单一灰度扰动，返回与输入同 dtype/形状的新图。

    amount 语义：gain=乘性系数（1.0 为不变）；offset=相对位深的偏移比例
    （0.2 = 加 20% 满量程）；gamma=Gamma 值（1.0 为不变）；contrast=对比度
    系数（1.0 为不变，绕均值缩放）。
    """
    arr = image.astype(np.float64)
    hi = _max_value(image.dtype)
    if kind == "brightness_gain":
        out = arr * amount
    elif kind == "brightness_offset":
        out = arr + amount * hi
    elif kind == "gamma":
        out = hi * np.power(np.clip(arr, 0.0, hi) / hi, amount)
    elif kind == "contrast":
        mean = float(arr.mean())
        out = mean + (arr - mean) * amount
    else:
        raise ValueError(f"未知扰动类型: {kind!r}")
    return np.clip(out, 0.0, hi).astype(image.dtype)


# 默认扰动条件（覆盖欠曝/过曝/软硬胶片响应/低对比四类常见差异）
DEFAULT_CONDITIONS: tuple[tuple[str, float], ...] = (
    ("brightness_gain", 0.7),
    ("brightness_gain", 1.3),
    ("brightness_offset", -0.15),
    ("brightness_offset", 0.15),
    ("gamma", 0.6),
    ("gamma", 1.6),
    ("contrast", 0.7),
)


@dataclass(frozen=True)
class RobustnessCfg:
    """稳定性判定阈值。

    min_retention：任一扰动条件下检出保持率的下限；quant_rel_tol：检出框
    长边相对 GT 长边的允许偏差；max_fp_increase：扰动引入的每图新增误检上限。
    """

    iou_threshold: float = 0.5
    quant_rel_tol: float = 0.10
    min_retention: float = 0.90
    max_fp_increase: float = 0.5


@dataclass
class RobustnessReport:
    """扰动稳定性评估结果（逐条件 + 汇总判定）。"""

    conditions: dict[str, dict[str, float]] = field(default_factory=dict)
    retention_worst: float = 1.0
    quant_dev_worst: float = 0.0
    fp_increase_worst: float = 0.0
    passed: bool = False
    failures: list[str] = field(default_factory=list)


def _match_targets(
    preds: list[dict[str, Any]], targets: list[dict[str, Any]], iou_threshold: float
) -> tuple[int, list[float]]:
    """贪心匹配（置信度降序、类别一致、IoU 达标），返回命中数与命中框长边相对偏差。

    长边偏差以 GT 长边为基准——GT 恒定，跨扰动条件可比。
    """
    sorted_preds = sorted(preds, key=lambda p: -p["score"])
    used = [False] * len(targets)
    hits = 0
    devs: list[float] = []
    for p in sorted_preds:
        for gi, g in enumerate(targets):
            if used[gi] or p["class_id"] != g["class_id"]:
                continue
            if iou(p["bbox"], g["bbox"]) < iou_threshold:
                continue
            used[gi] = True
            hits += 1
            gt_long = max(g["bbox"][2], g["bbox"][3])
            pred_long = max(p["bbox"][2], p["bbox"][3])
            if gt_long > 0:
                devs.append(abs(pred_long - gt_long) / gt_long)
            break
    return hits, devs


def robustness_metrics(
    samples: list[tuple[np.ndarray, list[dict[str, Any]]]],
    infer_fn: InferFn,
    conditions: tuple[tuple[str, float], ...] = DEFAULT_CONDITIONS,
    cfg: RobustnessCfg | None = None,
) -> RobustnessReport:
    """评估同一批缺陷在各灰度扰动下的检出/量化稳定性。

    samples: (图像, GT 列表) 序列；infer_fn 在每个扰动版本上被调用。
    baseline（无扰动）作为量化偏差与误检的参照系。
    """
    cfg = cfg or RobustnessCfg()
    report = RobustnessReport()
    total_gt = sum(len(t) for _, t in samples)
    if total_gt == 0:
        # 空 GT 判不通过：稳定率无从谈起，不得静默给满分（保守口径）。
        report.failures.append("评估集无真值标注，无法验证稳定性")
        return report

    # 误检增量以各条件中最接近恒等水平者为基线（理想情况下应提供无扰动对照）。
    for kind, amount in conditions:
        hits = 0
        devs: list[float] = []
        fp_total = 0
        for image, targets in samples:
            preds = infer_fn(perturb(image, kind, amount))
            n_hits, d = _match_targets(preds, targets, cfg.iou_threshold)
            hits += n_hits
            devs.extend(d)
            fp_total += len(preds) - n_hits
        retention = hits / total_gt
        quant_dev = float(np.mean(devs)) if devs else 1.0
        fp_per_image = fp_total / len(samples)
        report.conditions[f"{kind}@{amount}"] = {
            "retention": round(retention, 4),
            "quant_dev_mean": round(quant_dev, 4),
            "fp_per_image": round(fp_per_image, 4),
        }
        report.retention_worst = min(report.retention_worst, retention)
        report.quant_dev_worst = max(report.quant_dev_worst, quant_dev)

    fp_values = [c["fp_per_image"] for c in report.conditions.values()]
    baseline_fp = min(fp_values) if fp_values else 0.0
    report.fp_increase_worst = round(max(fp_values) - baseline_fp, 4) if fp_values else 0.0

    for name, c in report.conditions.items():
        if c["retention"] < cfg.min_retention:
            report.failures.append(f"{name} 检出保持率 {c['retention']:.3f} < {cfg.min_retention}")
        if c["quant_dev_mean"] > cfg.quant_rel_tol:
            report.failures.append(
                f"{name} 量化平均偏差 {c['quant_dev_mean']:.3f} > {cfg.quant_rel_tol}"
            )
    if report.fp_increase_worst > cfg.max_fp_increase:
        report.failures.append(
            f"扰动新增误检 {report.fp_increase_worst:.3f}/图 > {cfg.max_fp_increase}"
        )
    report.passed = not report.failures
    return report
