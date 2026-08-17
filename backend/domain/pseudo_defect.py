"""伪缺陷筛查（§4.2 底片质量校验）。

伪缺陷 = 非真实焊接缺陷的胶片/成像瑕疵：划痕、尘点、显影不均、大面积污渍、
指纹等。其形态与真实缺陷相似，若混入评片会误导检测模型与评片员。

策略（无训练数据前提下的可解释启发式，非 ML）：
- 长直划痕：Canny 边缘 + HoughLinesP 检测跨图像的长直特征；
- 显影不均/大面积污渍：低频能量（强高斯模糊后仍残留的大尺度灰度漂移）；
- 尘点密集：形态学 top-hat 提取小亮/暗斑，连通域计数。

默认仅"长直划痕"这类**无歧义严重**伪缺陷阻断评片（passed=False），其余
作为告警 notes 返回，避免对真实焊缝的自然灰度梯度（焊缝本身亮于母材）
过度阻断。阈值集中在 PseudoDefectCfg，便于按片种调参。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass(frozen=True)
class PseudoDefectCfg:
    hough_threshold: int = 60  # HoughLinesP 累加阈值（越低越灵敏）
    scratch_min_ratio: float = 0.5  # 划痕最小长度 / 图像对角线（低于忽略）
    scratch_grating_min_lines: int = 5  # 长直线段数 ≥ 此值视为周期性光栅（像质计），非孤立划痕
    canny_lo: int = 40
    canny_hi: int = 120
    uniformity_low_freq: float = 0.012  # 低频核占比（相对图像短边）上限判定不均匀
    uniformity_max_ratio: float = 6.0  # 低频漂移 σ / 局部纹理 σ 上限（超则"显影不均"）
    dust_tophat_k: int = 15  # 尘点 top-hat 核（奇数）
    dust_min_area: int = 10  # 尘点最小面积（px，过滤底片本底噪声小斑）
    dust_max_count: int = 400  # 尘点连通域上限（超过判"尘点密集/污渍"）
    block_on_scratch: bool = True  # 长直划痕是否阻断评片
    block_on_uniformity: bool = False  # 显影不均默认仅告警（焊件本身灰度梯度大）
    block_on_dust: bool = False  # 尘点密集默认仅告警


@dataclass(frozen=True)
class PseudoDefectReport:
    """伪缺陷筛查结果。passed=False 表示存在应阻断评片的严重伪缺陷。"""

    passed: bool
    notes: tuple[str, ...] = field(default_factory=tuple)
    metrics: dict = field(default_factory=dict)  # 透明指标，供前端/报告展示


def screen_pseudo_defects(image: np.ndarray, cfg: PseudoDefectCfg) -> PseudoDefectReport:
    """对整张底片做伪缺陷筛查，返回阻断判定 + 告警 notes + 透明指标。"""
    notes: list[str] = []
    metrics: dict = {}
    block = False

    gray = image.astype(np.float32)
    if gray.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape[:2]
    diag = float(np.hypot(h, w))

    # 1) 长直划痕
    scratch_long = _long_scratch_len_ratio(gray, cfg, diag)
    metrics["scratch_max_ratio"] = round(float(scratch_long), 4)
    if scratch_long >= cfg.scratch_min_ratio:
        notes.append(f"疑似长直划痕（最长 {scratch_long:.0%} 图像对角线）")
        if cfg.block_on_scratch:
            block = True

    # 2) 显影不均 / 大面积污渍（低频漂移）
    uni_ratio = _low_freq_ratio(gray, cfg)
    metrics["uniformity_ratio"] = round(float(uni_ratio), 4)
    if uni_ratio >= cfg.uniformity_max_ratio:
        notes.append(f"疑似显影不均/大面积污渍（低频漂移比 {uni_ratio:.1f}×）")
        if cfg.block_on_uniformity:
            block = True

    # 3) 尘点密集
    dust_count = _dust_count(gray, cfg)  # 过滤本底噪声小斑后的尘点连通域数
    metrics["dust_count"] = int(dust_count)
    if dust_count > cfg.dust_max_count:
        notes.append(f"疑似尘点密集/污渍（小斑 {dust_count} 处）")
        if cfg.block_on_dust:
            block = True

    return PseudoDefectReport(
        passed=not block,
        notes=tuple(notes),
        metrics=metrics,
    )


def _long_scratch_len_ratio(gray: np.ndarray, cfg: PseudoDefectCfg, diag: float) -> float:
    """返回最长"长直"特征长度 / 图像对角线。

    关键区分：孤立长划痕通常为 1~数条长直线段；而像质计（IQI）丝是成排、平行、
    周期排列的长直线 → 形成"光栅"。若检测到的长直线段数 ≥ scratch_grating_min_lines，
    判定为周期性光栅（像质计）而非伪缺陷划痕，返回 0（不阻断），避免把合法 IQI
    误判为划痕而错误阻断评片。仅当长直线段较少（孤立）时才按最长长度计。
    """
    edges = cv2.Canny(gray.astype(np.uint8), cfg.canny_lo, cfg.canny_hi)
    if edges.sum() == 0:
        return 0.0
    min_len = max(10, int(cfg.scratch_min_ratio * diag * 0.5))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=cfg.hough_threshold,
        minLineLength=min_len,
        maxLineGap=max(4, int(diag * 0.02)),
    )
    if lines is None:
        return 0.0
    # 周期性光栅（像质计）判定：长直线段数过多 → 视为 IQI，不阻断。
    if lines.shape[0] >= cfg.scratch_grating_min_lines:
        return 0.0
    best = 0.0
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        length = np.hypot(x2 - x1, y2 - y1)
        best = max(best, length)
    return float(best / diag)


def _low_freq_ratio(gray: np.ndarray, cfg: PseudoDefectCfg) -> float:
    """低频漂移 σ / 局部纹理 σ。越大表示大尺度灰度不均匀越严重。"""
    local_std = float(np.std(gray)) + 1e-6
    k = max(3, int(min(gray.shape[:2]) * cfg.uniformity_low_freq) | 1)
    blurred = cv2.GaussianBlur(gray, (k, k), 0)
    low_freq_std = float(np.std(blurred)) + 1e-6
    return low_freq_std / local_std


def _dust_count(gray: np.ndarray, cfg: PseudoDefectCfg) -> int:
    """形态学 top-hat 提取亮/暗小斑，返回较大者连通域数（尘点指标）。"""
    k = cfg.dust_tophat_k | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    tophat = cv2.morphologyEx(gray.astype(np.uint8), cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(gray.astype(np.uint8), cv2.MORPH_BLACKHAT, kernel)
    count = 0
    for surf in (tophat, blackhat):
        _, b = cv2.threshold(surf, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, _, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
        # 跳过极小噪点（<dust_min_area px）与背景（label 0）
        for s in stats[1:]:
            if s[cv2.CC_STAT_AREA] >= cfg.dust_min_area:
                count += 1
    return count
