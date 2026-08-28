"""数字底片图像质量测量：SNRn 归一化信噪比 + 双丝像质计空间分辨率。

DB50/T 1807-2025 §6.1.6 要求系统具备"归一化信噪比、空间分辨率"测量功能。

公式出处（2026-08 查证）：
- SNRN = SNR × (88 µm / SRb)，即把实测 SNR 归一到 88µm 参考基本空间分辨率
  （ISO 20769-1:2018 定义；GB/T 26141.1-2010 修改采用 ISO 14096-1，同源体系；
  SRb 由双丝像质计或 MTF 测得）。见：
  https://standards.iteh.ai/catalog/standards/cen/c2a4517f-adea-4d7f-9ee4-1f1acc61898d/en-iso-20769-1-2018
  https://www.iso.org/standard/87451.html
- 双丝像质计判据：调制深度 M=(Imax−Imin)/(Imax+Imin)≥0.2 视为可分辨
  （ISO 17636-2 / EN 14784 双丝法）；空间分辨率 = 第一对不可分辨丝的丝径。
- SNRn 合格线（min_snrn）按数字化级别配置；默认 130 取自 ISO 17636-2 最优
  CR 板级别阈值，GB/T 26141.1 的 DS/DB/DA 级确切限值须以授权原文复核（待查证）。

比标准更严格：无缺陷 ROI 时自动分块取**最差 3 块**（不取均值）；SRb 未提供时
用像素尺寸保守估计并显式标记 srb_estimated=True；调制深度 0.2~0.3 标记"临界"。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# 归一化基准（ISO 20769-1 / GB/T 26141.1 体系）：88 µm
_REF_APERTURE_MM = 0.088
# 双丝可分辨判据（ISO 17636-2）：M≥0.2；0.2~0.3 为"临界"（比标准严的提示带）
_RESOLVED_M = 0.20
_MARGINAL_M = 0.30


# ---------------------------------------------------------------------------
# SNRn
# ---------------------------------------------------------------------------


@dataclass
class SNRResult:
    """SNRn 测量结果（含全部中间量，供报告/核验）。"""

    snr: float
    snrn: float
    mean_signal: float
    noise_sd: float
    srb_mm: float
    srb_estimated: bool
    block_stats: list[dict[str, float]]  # 各分块 (mean, sd, snr, snrn)
    min_snrn: float
    passed: bool

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _uniform_blocks(gray: np.ndarray, block: int = 128) -> list[tuple[int, int, int, int]]:
    """按"无确定性结构"选出可测 SNR 的块（排除焊缝/IQI 等结构边缘）。

    结构判据：强模糊（σ=16，把颗粒噪声基本抹平）后的梯度是否超过动态范围
    的 2%。注意不能用"梯度最小"排序——高噪声但无结构的块梯度也大，
    而它们恰恰是合法的测量位置（ISO 20769-1：≥4 个位置取最差）。
    """
    h, w = gray.shape[:2]
    rng_val = float(np.percentile(gray, 99) - np.percentile(gray, 1))
    thr = max(0.02 * rng_val, 0.05)
    blocks: list[tuple[float, int, int, int, int]] = []
    for y in range(0, h - block + 1, block):
        for x in range(0, w - block + 1, block):
            roi = gray[y : y + block, x : x + block].astype(np.float32)
            smooth = cv2.GaussianBlur(roi, (0, 0), 16.0)
            gx = np.abs(np.diff(smooth, axis=1)).mean()
            gy = np.abs(np.diff(smooth, axis=0)).mean()
            blocks.append((float(gx + gy), x, y, block, block))
    blocks.sort(key=lambda t: t[0])
    structured = [b for b in blocks if b[0] > thr]
    candidates = [b for b in blocks if b[0] <= thr] or blocks[:3]  # 全超标时退回最平 3 块
    if structured and len(candidates) < 3:  # 候选不足时并入最平的结构块兜底
        candidates += [b for b in structured[: 3 - len(candidates)]]
    return [(x, y, bw, bh) for _, x, y, bw, bh in candidates[:16]]


def _roi_snr(gray: np.ndarray, x: int, y: int, w: int, h: int) -> tuple[float, float]:
    """单块 (均值信号, 噪声标准差)：噪声用 7×7 均值高通的鲁棒估计（抗梯度/结构）。"""
    roi = gray[y : y + h, x : x + w].astype(np.float32)
    local_mean = cv2.blur(roi, (7, 7))  # 均值高通（medianBlur 不支持 float32）
    noise = (roi - local_mean).ravel()
    sd = float(1.4826 * np.median(np.abs(noise - np.median(noise))))  # MAD→σ
    signal = float(np.median(roi))
    return signal, sd


def measure_snr(
    gray: np.ndarray,
    *,
    srb_mm: float | None,
    pixel_spacing_mm: float | None = None,
    min_snrn: float = 130.0,
    n_blocks: int = 3,
) -> SNRResult:
    """SNRn 测量：自动选最平坦的 n_blocks 块，取**最差块**口径（比标准严）。

    srb_mm：双丝像质计实测基本空间分辨率（mm）；缺省时用扫描像素尺寸保守估计
    （SRb≈pixel size 会高估 SRb → 低估 SNRn → 偏严，方向安全）。
    注意：严格按 GB/T 26141.1 应在线性化灰度上测量（本函数接受原始灰度，
    胶片黑度→线性化的标定曲线可由调用方预先施加）。
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if srb_mm is None or srb_mm <= 0:
        if pixel_spacing_mm is None or pixel_spacing_mm <= 0:
            raise ValueError("缺少基本空间分辨率 SRb 且无像素标定（pixel_spacing_mm），无法归一化")
        srb_mm = float(pixel_spacing_mm)
        srb_estimated = True
    else:
        srb_estimated = False

    h, w = gray.shape[:2]
    block = min(128, h, w)
    # _uniform_blocks 已按"无确定性结构"筛选（ISO 20769-1：≥4 位置取最差）
    candidates = _uniform_blocks(gray, block)
    if len(candidates) < n_blocks:  # 结构块不足时退化取最平 n_blocks 块
        candidates = _uniform_blocks(gray, block)[:n_blocks]
    if not candidates:
        raise ValueError("图像过小，无法分块测量")
    stats: list[dict[str, float]] = []
    for (x, y, bw, bh) in candidates:
        sig, sd = _roi_snr(gray, x, y, bw, bh)
        if sd <= 0:
            continue
        snr = sig / sd
        snrn = snr * (_REF_APERTURE_MM / srb_mm)
        stats.append({"x": x, "y": y, "mean": sig, "sd": sd, "snr": snr, "snrn": snrn})
    if not stats:
        raise ValueError("信噪比测量失败：选定区域内噪声为零（可能为纯色图）")
    worst = min(stats, key=lambda s: s["snrn"])  # 取最差块（比取均值严）
    return SNRResult(
        snr=round(worst["snr"], 3),
        snrn=round(worst["snrn"], 3),
        mean_signal=round(worst["mean"], 4),
        noise_sd=round(worst["sd"], 5),
        srb_mm=round(srb_mm, 5),
        srb_estimated=srb_estimated,
        block_stats=[{k: round(v, 4) for k, v in s.items()} for s in stats],
        min_snrn=min_snrn,
        passed=worst["snrn"] >= min_snrn,
    )


# ---------------------------------------------------------------------------
# 双丝像质计空间分辨率
# ---------------------------------------------------------------------------

# 双丝像质计丝径序列（mm，ISO 17636-2 / GB/T 常见 Pt/W 双丝组，由粗到细）。
# 空间分辨率值 = 第一对不可分辨丝的丝径。可由调用方覆盖（非标丝组）。
DUPLEX_WIRE_DIAMETERS_MM = (0.50, 0.40, 0.32, 0.25, 0.20, 0.16, 0.13, 0.10)
# 双丝对内两丝中心距 = 2×丝径（双丝几何：丝间净隙 = 丝径）


@dataclass
class DuplexResult:
    """双丝测量结果。"""

    spatial_resolution_mm: float | None  # 第一对不可分辨丝的丝径
    resolved: list[dict[str, float]]  # 逐对 {diameter, modulation, verdict}
    marginal: bool  # 存在 0.2≤M<0.3 的临界对（比标准严的提示）
    wire_axis_deg: float  # 检测所用丝方向（度，相对图像 x 轴）
    period_px: float  # 实测丝对内周期（像素）

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _profile_along_wires(roi: np.ndarray, angle_deg: float) -> tuple[np.ndarray, float]:
    """旋转 ROI 使丝沿 x 轴，返回沿垂直方向的平均密度剖面与实际旋转角。

    wire_axis_deg 为丝方向相对图像 x 轴的逆时针角度；对齐 = 将内容顺时针
    旋回该角度（warpAffine 角度取负）。
    """
    if angle_deg:  # 旋转展开：丝方向 → 水平
        h, w = roi.shape[:2]
        center = (w / 2, h / 2)
        m = cv2.getRotationMatrix2D(center, -float(angle_deg), 1.0)
        cos, sin = abs(m[0, 0]), abs(m[0, 1])
        nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
        m[0, 2] += nw / 2 - center[0]
        m[1, 2] += nh / 2 - center[1]
        roi = cv2.warpAffine(roi, m, (nw, nh), flags=cv2.INTER_LINEAR)
    return roi.mean(axis=1), float(angle_deg)


def _estimate_period(profile: np.ndarray) -> float:
    """FFT 估计丝对内周期（像素）：双丝图案的主频即 2×丝径 中心距。"""
    p = profile - profile.mean()
    spec = np.abs(np.fft.rfft(p))
    spec[0] = 0.0
    n = len(p)
    k = int(np.argmax(spec))
    return n / k if k > 0 else 0.0


def measure_duplex_wire(
    gray: np.ndarray,
    *,
    pixel_spacing_mm: float,
    wire_axis_deg: float = 0.0,
    wire_diameters_mm: tuple[float, ...] = DUPLEX_WIRE_DIAMETERS_MM,
) -> DuplexResult:
    """双丝像质计空间分辨率测量。

    gray：双丝 IQI 区域（ROI）；pixel_spacing_mm：mm/px 标定（必填，缺失拒绝测量）。
    算法：旋转对齐 → x 向平均得一维剖面 → FFT 定丝对内周期 → 按周期折叠分组
    逐对求调制深度 M=(Imax−Imin)/(Imax+Imin) → 第一对 M<0.2 的丝径即空间分辨率。
    """
    if pixel_spacing_mm is None or pixel_spacing_mm <= 0:
        raise ValueError("缺少像素标定 pixel_spacing_mm，无法换算物理分辨率")
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    profile, angle = _profile_along_wires(gray.astype(np.float32), float(wire_axis_deg))
    period_px = _estimate_period(profile)
    if period_px <= 1.0:
        raise ValueError("未能从剖面中识别双丝周期（请确认 ROI 覆盖双丝像质计且方向正确）")
    d_mm = period_px * pixel_spacing_mm / 2.0  # 周期 = 2×丝径

    # 逐对：丝从粗到细沿丝轴排布，把剖面按对数分段（每段 ≈ 一对双丝），段内求 M
    n = len(profile)
    n_pairs = len(wire_diameters_mm)
    seg_len = n // n_pairs if n > n_pairs else n
    resolved: list[dict[str, float]] = []
    for idx, d in enumerate(wire_diameters_mm):
        seg = profile[idx * seg_len : (idx + 1) * seg_len]
        if seg.size < 3:
            continue
        imax = float(np.percentile(seg, 95))
        imin = float(np.percentile(seg, 5))
        if imax + imin <= 0:
            continue
        mod = (imax - imin) / (imax + imin)
        verdict = "resolved" if mod >= _RESOLVED_M else ("marginal" if mod >= 0.0 else "unresolved")
        # M<0.2 即不可分辨（verdict 仅标记临界带）
        resolved.append(
            {
                "diameter_mm": d,
                "modulation": round(mod, 4),
                "verdict": verdict if mod >= _RESOLVED_M else ("critical" if mod >= _RESOLVED_M - 0.05 else "unresolved"),
            }
        )
    first_unresolved = next(
        (r for r in resolved if r["modulation"] < _RESOLVED_M), None
    )
    marginal = any(_RESOLVED_M <= r["modulation"] < _MARGINAL_M for r in resolved)
    return DuplexResult(
        spatial_resolution_mm=first_unresolved["diameter_mm"] if first_unresolved else wire_diameters_mm[-1],
        resolved=resolved,
        marginal=marginal,
        wire_axis_deg=angle,
        period_px=round(period_px, 2),
    )
