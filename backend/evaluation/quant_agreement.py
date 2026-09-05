"""量化一致性 harness（技术规格 §15.2：量化相对误差 ≤5% + Bland–Altman 分析）。

口径：
- 参考值（ref）：真值框（GT bbox）经包围盒量化 × 像素标定 → 长宽/面积；
- 自动值（auto）：**同一真值框定位**下掩膜精修量化（MaskQuantifier）。
  定位两侧同源，把"量化误差"从"检出定位误差"中剥离——规格考核的是
  几何量化环节本身。
- Bland–Altman：差值（auto−ref）均值偏倚 ±1.96×标准差（95% 一致性界限，
  LoA），同时给出平均绝对相对误差与 ≤5% 占比。

掩膜精修退化（无法取轮廓/被禁用）时 auto=ref（包围盒路径），配对仍有效
——结论会呈现"量化无增益"，不呈现虚假误差。
"""

from __future__ import annotations

import math
from typing import Any

from backend.domain.dto import BBox, DefectClass, Detection
from backend.domain.quantify import BBoxQuantifier, MaskQuantifier

DEFAULT_REL_THRESHOLD = 0.05  # 规格 §15.2：量化相对误差 ≤5%


def bland_altman(
    auto_mm: list[float],
    ref_mm: list[float],
    *,
    rel_threshold: float = DEFAULT_REL_THRESHOLD,
) -> dict[str, Any]:
    """配对测量的 Bland–Altman 统计 + 相对误差判定。

    任一输入为空返回 n=0 的空壳（verdict 不通过——无数据不构成达标证据）。
    """
    if len(auto_mm) != len(ref_mm):
        raise ValueError("auto_mm 与 ref_mm 长度必须一致")
    n = len(auto_mm)
    if n == 0:
        return {
            "n": 0,
            "mean_bias": 0.0,
            "sd_diff": 0.0,
            "loa_upper": 0.0,
            "loa_lower": 0.0,
            "mean_abs_rel_err": 0.0,
            "p95_abs_rel_err": 0.0,
            "pct_within_threshold": 0.0,
            "thresholds": {"rel_err_max": rel_threshold},
            "verdict": {"passed": False},
        }
    diffs = [a - r for a, r in zip(auto_mm, ref_mm)]
    mean_bias = sum(diffs) / n
    var = sum((d - mean_bias) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    sd = math.sqrt(var)
    # 相对误差：ref=0 且 auto=0 是完美一致（差值为 0），记 0.0 而非 inf——
    # 记 inf 会把它错排成"最差样本"污染均值/分位数与达标占比。
    rel = [
        abs(a - r) / abs(r) if r != 0 else (0.0 if a == 0 else math.inf)
        for a, r in zip(auto_mm, ref_mm)
    ]
    finite_rel = sorted(r for r in rel if math.isfinite(r))
    mean_rel = sum(finite_rel) / len(finite_rel) if finite_rel else math.inf
    p95_rel = (
        finite_rel[min(len(finite_rel) - 1, int(0.95 * len(finite_rel)))]
        if finite_rel
        else math.inf
    )
    within = sum(1 for r in rel if r <= rel_threshold)

    return {
        "n": n,
        "mean_bias": round(mean_bias, 4),
        "sd_diff": round(sd, 4),
        "loa_upper": round(mean_bias + 1.96 * sd, 4),
        "loa_lower": round(mean_bias - 1.96 * sd, 4),
        "mean_abs_rel_err": round(mean_rel, 4),
        "p95_abs_rel_err": round(p95_rel, 4),
        "pct_within_threshold": round(within / n, 4),
        "thresholds": {"rel_err_max": rel_threshold},
        "verdict": {"passed": mean_rel <= rel_threshold},
    }


def geometry_pairs_for_image(
    image,
    gt_boxes: list[list[float]],
    spacing: float,
    cfg=None,
) -> list[tuple[dict[str, float], dict[str, float]]]:
    """单图配对：GT 框 → (掩膜精修几何, 包围盒参考几何)。

    gt_boxes: [[x,y,w,h], ...] 原图像素坐标；spacing: mm/px。
    cfg: 掩膜精修参数（MaskRefineCfg）——评估须与生产同参（生产经
    configs 的 mask_refine 构造），缺省用数据类默认值（仅测试用）。
    """
    auto_q = MaskQuantifier()
    ref_q = BBoxQuantifier()
    pairs: list[tuple[dict[str, float], dict[str, float]]] = []
    for i, box in enumerate(gt_boxes):
        det = Detection(
            id=f"gt-{i}",
            bbox=BBox(x=float(box[0]), y=float(box[1]), w=float(box[2]), h=float(box[3])),
            class_id=DefectClass.POROSITY,  # 量化与类别无关，占位满足契约
            score=1.0,
            uncertainty=0.0,
        )
        ref = ref_q.quantify(det, spacing, image=None, cfg=None)
        auto = auto_q.quantify(det, spacing, image=image, cfg=cfg)
        pairs.append(
            (
                {
                    "length_mm": auto.length_mm,
                    "width_mm": auto.width_mm,
                    "area_mm2": auto.area_mm2,
                },
                {
                    "length_mm": ref.length_mm,
                    "width_mm": ref.width_mm,
                    "area_mm2": ref.area_mm2,
                },
            )
        )
    return pairs


def quantification_summary(
    pairs: list[tuple[dict[str, float], dict[str, float]]],
    *,
    rel_threshold: float = DEFAULT_REL_THRESHOLD,
) -> dict[str, Any]:
    """全部配对按维度（长/宽/面积）聚合相对误差与 Bland–Altman。"""
    dims = ("length_mm", "width_mm", "area_mm2")
    out: dict[str, Any] = {"n_defects": len(pairs)}
    for d in dims:
        out[d] = bland_altman(
            [a[d] for a, _ in pairs], [r[d] for _, r in pairs], rel_threshold=rel_threshold
        )
    # 总体判定：三个维度全部达标才算通过（规格对量化误差未分维度豁免）
    out["verdict"] = {"passed": all(out[d]["verdict"]["passed"] for d in dims)}
    return out
