"""线型像质计（wire IQI）自动识别（§4.2，M2 基线实现）。纯算法。

基线假设（标准线型 IQI 通用几何）：
- 丝在 ROI 内沿水平方向平行排列、垂直方向等距分布；
- 金属丝吸收射线 → 底片黑度低 → 透射数字化影像上呈"亮线"。
算法：对每根丝取水平行带 → 行均值剖面 → 峰对比度 vs 局部噪声 → 判定可见；
achieved = 最细可见丝号（丝号越大越细）；passed = achieved ≥ required。
全自动 ROI 搜索（模板匹配）留作后续增强；当前 ROI 由前端/人工提供。
丝号-直径表为公开参考值，正式使用前须按 IQI 标准复核（§T8 熔断精神）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.domain.dto import IQIResult


@dataclass(frozen=True)
class IqiConfig:
    wire_diameters_mm: tuple[float, ...]  # 丝号 1..N 对应直径（mm），递增
    required_wire_no: int  # 工艺要求的最细丝号（越大越难）
    min_contrast_ratio: float = 3.0  # 峰对比度 / 噪声 阈值
    band_radius_px: int = 2  # 每根丝行带半宽


def verify_wire_iqi(
    image: np.ndarray,
    cfg: IqiConfig,
    roi: tuple[int, int, int, int] | None = None,
) -> IQIResult:
    """在 ROI（默认全图）内逐丝测量可见性，返回可达最细丝号。"""
    h_img, w_img = image.shape[:2]
    x, y, w, h = roi or (0, 0, w_img, h_img)
    patch = image[y : y + h, x : x + w]

    achieved: int | None = None
    if patch.size > 0:
        n = len(cfg.wire_diameters_mm)
        for i in range(n):
            y_center = round((i + 0.5) / n * h)
            contrast, noise = _row_profile_contrast(patch, y_center, cfg.band_radius_px)
            if contrast > cfg.min_contrast_ratio * noise:
                achieved = i + 1  # 从粗到细遍历，最终值即最细可见丝号

    passed = achieved is not None and achieved >= cfg.required_wire_no
    return IQIResult(
        iqi_type="wire",
        achieved=str(achieved) if achieved is not None else None,
        required=str(cfg.required_wire_no),
        passed=bool(passed),
    )


def _row_profile_contrast(
    patch: np.ndarray, y_center: int, radius: int
) -> tuple[float, float]:
    """返回 (峰对比度, 噪声σ)。

    丝为水平亮线：对比度体现在**逐行均值剖面**（垂直方向）上——
    丝带行均值显著高于背景中位数。噪声仅在**背景行（≤中位数）**上以
    MAD 稳健估计，使信号占比较高（甚至 ≥50% 行）时噪声估计不被丝峰污染。
    """
    row_mean = patch.mean(axis=1).astype(np.float64)
    if row_mean.size < 2 * radius + 1:
        return 0.0, 0.0
    lo, hi = max(0, y_center - radius), min(row_mean.size, y_center + radius + 1)
    center = float(row_mean[lo:hi].mean())
    # 背景簇：30 分位以下的行（信号占比 <70% 时均落在背景区）。
    q30 = float(np.percentile(row_mean, 30))
    bg_mask = row_mean <= q30
    if int(bg_mask.sum()) >= 2:
        bg_level = float(np.median(row_mean[bg_mask]))
        # 可见性对比对象是**像素级噪声**（丝对比度 vs σ_pix），
        # 而非行均值噪声（列平均会淹没像素噪声，导致过度敏感）。
        noise = float(np.std(patch[bg_mask, :].astype(np.float64)))
    else:
        bg_level = float(np.median(row_mean))
        noise = float(patch.std())
    return center - bg_level, max(noise, 1e-6)
