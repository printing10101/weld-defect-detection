"""漂移监控。

设计文档：定期用新数据评估，触发再训练告警。

实现：对一批新样本的检出统计（尺寸分布 / 置信度分布 / 类别分布）与
参考基线（训练/验收时的分布）比较：
- 尺寸漂移：新样本缺陷长径中位数相对基线的相对偏移超阈值；
- 置信度漂移：新样本平均置信度相对基线的绝对偏移超阈值；
- 类别分布漂移：新样本各缺陷类别占比与基线的最大绝对差超阈值。

任一维度超阈值即返回 drift=True（触发再训练/人工复核告警）。
纯 numpy 实现，可离线单测；分布统计为启发式代理（非 KS 检验等严苛检验，
满足"定期评估触发告警"的运营目的即可，docstring 注明替代关系）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DriftResult:
    """漂移评估结果。"""

    drift: bool
    alerts: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


def _median(vals: list[float]) -> float:
    return float(np.median(vals)) if vals else 0.0


def _mean(vals: list[float]) -> float:
    return float(np.mean(vals)) if vals else 0.0


def estimate_drift(
    new_samples: list[dict[str, Any]],
    baseline: dict[str, Any],
    *,
    size_tol: float = 0.25,  # 长径中位数相对偏移 >25%
    conf_tol: float = 0.10,  # 平均置信度绝对偏移 >0.10
    class_tol: float = 0.20,  # 类别占比最大绝对差 >0.20
) -> DriftResult:
    """评估新样本相对基线的漂移。

    new_samples : 新数据检出列表 [{class_id, score, L_mm, ...}]（或空=无检出）
    baseline    : 参考基线 {size_median_mm, conf_mean, class_ratio:{cid:ratio}}
    """
    if not new_samples:
        return DriftResult(drift=False, alerts=[], metrics={"n": 0})

    sizes = [float(s.get("L_mm", 0.0)) for s in new_samples if s.get("L_mm") is not None]
    confs = [float(s.get("score", 0.0)) for s in new_samples if s.get("score") is not None]
    n = len(new_samples)

    metrics: dict[str, float] = {"n": n}
    alerts: list[str] = []

    # 1) 尺寸漂移
    cur_size = _median(sizes)
    base_size = float(baseline.get("size_median_mm", 0.0) or 0.0)
    metrics["size_median_mm"] = cur_size
    if base_size > 0 and sizes:
        rel = abs(cur_size - base_size) / base_size
        metrics["size_rel_drift"] = round(rel, 4)
        if rel > size_tol:
            alerts.append(
                f"缺陷尺寸漂移：中位 {cur_size:.2f}mm vs 基线 {base_size:.2f}mm（偏移 {rel:.1%}）"
            )

    # 2) 置信度漂移
    cur_conf = _mean(confs)
    base_conf = float(baseline.get("conf_mean", 0.0) or 0.0)
    metrics["conf_mean"] = cur_conf
    if confs:
        conf_diff = abs(cur_conf - base_conf)
        metrics["conf_diff"] = round(conf_diff, 4)
        if conf_diff > conf_tol:
            alerts.append(
                f"置信度漂移：均值 {cur_conf:.3f} vs 基线 {base_conf:.3f}（差 {conf_diff:.3f}）"
            )

    # 3) 类别分布漂移
    # 基线经 JSON 落盘后 int 键会变成 str，这里统一强转 int 避免 lookup 失败
    # （否则重新加载的基线 class_ratio 键为 "0" 而非 0，导致类别漂移误报）。
    base_ratios: dict[int, float] = {
        int(k): float(v) for k, v in (baseline.get("class_ratio", {}) or {}).items()
    }
    if new_samples:
        counts: dict[int, int] = {}
        for s in new_samples:
            counts[int(s["class_id"])] = counts.get(int(s["class_id"]), 0) + 1
        max_diff = 0.0
        for cid, cnt in counts.items():
            cur_ratio = cnt / n
            base_ratio = float(base_ratios.get(cid, 0.0))
            max_diff = max(max_diff, abs(cur_ratio - base_ratio))
        metrics["class_ratio_max_diff"] = round(max_diff, 4)
        if max_diff > class_tol:
            alerts.append(f"类别分布漂移：最大占比差 {max_diff:.2%} 超阈值")

    return DriftResult(drift=bool(alerts), alerts=alerts, metrics=metrics)
