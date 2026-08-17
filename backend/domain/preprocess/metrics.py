"""预处理质量度量（§4.4）。纯算法，无 I/O。

- 有参考：PSNR / SSIM（skimage）
- 无参考：拉普拉斯法噪声估计（σ ≈ std(Laplacian)/√6，适用于高斯噪声）
- 无参考盲评：BRISQUE 风格特征提取（MSCN + 4 乘积，2 尺度，GGD/AGGD 参数估计）
  + 复合射线底片质量指数（RQI）作为实际门禁。

说明：官方 BRISQUE 的"分数"依赖训练的 SVR（libsvm）模型文件，离线部署未捆绑
（且本沙箱无 sklearn）。因此 `brisque_features()` 提供 faithful 的 36 维无参考特征
向量（可作为训练/回归输入或失真描述），门禁判定改用可解释的 RQI 复合分（噪声 /
锐度 / 对比度 / 动态范围 / 均匀性 / 伪缺陷），两者在同一 `assess_quality()` 中产出。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
from scipy.optimize import brentq
from scipy.signal import convolve2d
from scipy.special import gamma as _gamma


def _data_range(a: np.ndarray) -> float:
    """按 dtype 推断 PSNR/SSIM 的动态范围（避免 uint16/float 误用 255）。"""
    if a.dtype == np.uint8:
        return 255.0
    if a.dtype == np.uint16:
        return 65535.0
    return 1.0


def psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """峰值信噪比（dB），data_range 按数组 dtype 推断；完全一致时返回有限值。"""
    from skimage.metrics import peak_signal_noise_ratio

    val = float(peak_signal_noise_ratio(original, processed, data_range=_data_range(original)))
    return val if np.isfinite(val) else 99.0


def ssim(original: np.ndarray, processed: np.ndarray) -> float:
    """结构相似度 [0,1]，data_range 按数组 dtype 推断；多通道自动指定 channel_axis。"""
    from skimage.metrics import structural_similarity

    kwargs: dict = {}
    if processed.ndim == 3:
        kwargs["channel_axis"] = -1
    val = structural_similarity(original, processed, data_range=_data_range(original), **kwargs)
    return float(np.clip(val, 0.0, 1.0))


_LAPLACIAN_KERNEL_L2 = np.sqrt(
    20.0
)  # cv2.Laplacian(ksize=1) 核 [[0,1,0],[1,-4,1],[0,1,0]] 的 L2 范数


def estimate_noise(image: np.ndarray) -> float:
    """无参考噪声估计（拉普拉斯法）：σ ≈ std(Laplacian)/‖K‖₂，K 为 cv2 拉普拉斯核。"""
    if image.size == 0:
        return 0.0
    lap = np.asarray(cv2.Laplacian(image, cv2.CV_64F), dtype=np.float64)
    return float(np.std(lap) / _LAPLACIAN_KERNEL_L2)


# ---------------------------------------------------------------------------
# §4.4 BRISQUE 风格无参考特征提取
# ---------------------------------------------------------------------------
_BRISQUE_C = 1.0 / 255.0  # MSCN 归一化常数（避免分母为零）


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """归一化 2D 高斯核（MSCN 局部均值/方差估计用）。"""
    ax = np.arange(-(size // 2), size // 2 + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return k / k.sum()


def _mscn_coefficients(gray: np.ndarray) -> np.ndarray:
    """Mean-Subtracted Contrast-Normalized 系数（§4.4，BRISQUE 核心）。

    MSCN = (I − μ) / (σ + C)，μ/σ 由 7×7 高斯（σ=7/6）局部估计，输出近似零均值、
    单位方差，集中刻画局部对比度结构（自然影像服从特定广义高斯分布）。
    """
    img = gray.astype(np.float64)
    if img.size and img.max() > 1.0:
        img = img / 255.0
    ker = _gaussian_kernel(7, 7.0 / 6.0)
    mu = convolve2d(img, ker, mode="same", boundary="symm")
    mu_sq = convolve2d(img * img, ker, mode="same", boundary="symm")
    sigma = np.sqrt(np.maximum(mu_sq - mu * mu, 1e-12))
    return (img - mu) / (sigma + _BRISQUE_C)


def _mscn_products(mscn: np.ndarray) -> list[np.ndarray]:
    """4 个相邻乘积极坐标（水平/垂直/两对角），用于 AGGD 拟合。"""
    h = mscn[:, :-1] * mscn[:, 1:]
    v = mscn[:-1, :] * mscn[1:, :]
    d1 = mscn[:-1, :-1] * mscn[1:, 1:]
    d2 = mscn[:-1, 1:] * mscn[1:, :-1]
    return [h.ravel(), v.ravel(), d1.ravel(), d2.ravel()]


def _ggd_params(x: np.ndarray) -> tuple[float, float]:
    """零均值广义高斯分布（GGD）参数估计 → (形状 α, 尺度 σ)。

    α 由矩比 R = E[|x|]² / E[x²] = Γ(2/α)² / (Γ(1/α)Γ(3/α)) 反解（R 随 α 单调递增），
    单变量 brentq 求根；σ = √E[x²]。数学已用理论矩一致性验证。
    """
    x = x.astype(np.float64).ravel()
    if x.size == 0:
        return 1.0, 1.0
    m2 = float(np.mean(x * x))
    m1 = float(np.mean(np.abs(x)))
    if m2 <= 0:
        return 1.0, 1.0
    r = (m1 * m1) / m2

    def _f(a: float) -> float:
        return (_gamma(2.0 / a) ** 2) / (_gamma(1.0 / a) * _gamma(3.0 / a)) - r

    try:
        alpha = cast(float, brentq(_f, 0.2, 10.0))
    except ValueError:
        alpha = 1.0
    return float(alpha), float(np.sqrt(m2))


def _aggd_params(x: np.ndarray) -> tuple[float, float, float, float]:
    """非对称广义高斯分布（AGGD）参数估计 → (形状 α, 均值 μ, 左尺度 β_l, 右尺度 β_r)。

    由三矩方程联立反解（自推导，scipy 验证）：设 u=β_l+β_r、v=β_l−β_r、G2=Γ(2/α)/Γ(1/α)、
    G3=Γ(3/α)/Γ(1/α)，则 E[|x|]=S、E[x]=μ、E[x²]=σ² 给出
    u=(S+√(S²−μ²))/G2、v=μ/G2，且 α 满足 (u²+3v²)/4·G3 = σ²。
    单变量 brentq 对 α 求根后回代 β_l=(u+v)/2、β_r=(u−v)/2。
    """
    x = x.astype(np.float64).ravel()
    if x.size == 0:
        return 1.0, 0.0, 1.0, 1.0
    mu = float(np.mean(x))
    s_abs = float(np.mean(np.abs(x)))
    sigma2 = float(np.mean(x * x))
    if sigma2 <= 0:
        return 1.0, mu, 1.0, 1.0

    def _solve_alpha(a: float) -> float:
        g2 = _gamma(2.0 / a) / _gamma(1.0 / a)
        g3 = _gamma(3.0 / a) / _gamma(1.0 / a)
        v = mu / g2
        disc = max(s_abs * s_abs - mu * mu, 0.0)
        u = (s_abs + np.sqrt(disc)) / g2
        return (u * u + 3.0 * v * v) / 4.0 * g3 - sigma2

    try:
        alpha = cast(float, brentq(_solve_alpha, 0.2, 10.0))
    except ValueError:
        alpha = 1.0
    g2 = _gamma(2.0 / alpha) / _gamma(1.0 / alpha)
    v = mu / g2
    disc = max(s_abs * s_abs - mu * mu, 0.0)
    u = (s_abs + np.sqrt(disc)) / g2
    # 由 E[x]=G2·(β_r−β_l) 推得：β_l 为较小侧（x<0），β_r 为较大侧（x>0）。
    beta_l = (u - v) / 2.0
    beta_r = (u + v) / 2.0
    return float(alpha), float(mu), float(max(beta_l, 1e-6)), float(max(beta_r, 1e-6))


def brisque_features(gray: np.ndarray) -> np.ndarray:
    """BRISQUE 风格无参考特征向量（36 维，§4.4）。

    2 个尺度（原图 + 0.5× 下采样）×（MSCN 的 GGD 2 参数 + 4 乘积的 AGGD 各 4 参数）
    = 2 × (2 + 16) = 36 维。该向量描述底片局部对比度结构失真，可作质量回归/SVR 输入。
    注：官方 BRISQUE *分数* 需训练好的 SVR 模型（libsvm），离线未捆绑；门禁判定见
    `assess_quality()` 的 RQI 复合分。
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float64)
    if gray.size == 0:
        return np.zeros(36, dtype=np.float64)

    feats: list[float] = []
    scales = [gray]
    h, w = gray.shape
    if min(h, w) >= 4:
        scales.append(cv2.resize(gray, (max(1, w // 2), max(1, h // 2))))

    for sc in scales:
        sc = sc.astype(np.float64)
        mscn = _mscn_coefficients(sc)
        a, s = _ggd_params(mscn.ravel())
        feats.extend([a, s])
        for prod in _mscn_products(mscn):
            pa, pmu, pl, pr = _aggd_params(prod)
            feats.extend([pa, pmu, pl, pr])
    vec = np.array(feats, dtype=np.float64)
    if vec.size < 36:  # 极小图仅一个尺度 → 补足 36 维（重复末尺度尾部）
        vec = np.concatenate([vec, vec[-(36 - vec.size) :]])
    return vec[:36]


# ---------------------------------------------------------------------------
# §4.4 复合射线底片质量指数（RQI）——实际门禁判定
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QualityCfg:
    """射线底片质量门禁配置（§4.4，§T8 三处同步）。

    门禁采用可解释 RQI（0–100，越高越好）= Σ wᵢ·sᵢ，sᵢ∈[0,1] 为各子指标分。
    min_score 以下判不合格；block_on_quality=True 时不合格阻断评片（默认 True，
    反证测试确认真实好底片 0 误杀；不合格即阻断评片并提示重拍/换片）。

    另含三类硬门禁（反证测试后补强）：模糊(Laplacian 方差)、曝光(直方图熵)、污渍
    (平滑异常斑块)。任一触发即 passed=False，不扰动已标定的 RQI 复合分权重。
    """

    w_noise: float = 0.25
    w_sharp: float = 0.20
    w_contrast: float = 0.20
    w_dynamic: float = 0.15
    w_uniform: float = 0.10
    w_artifact: float = 0.10
    min_score: float = 70.0  # RQI 门限（0–100），已按真实底片标定
    block_on_quality: bool = True  # True=不合格阻断评片；False=仅告警+need_review
    noise_good: float = 4.0  # 噪声σ ≤ 此值满分
    noise_bad: float = 14.0  # 噪声σ ≥ 此值 0 分
    sharp_good: float = 1.5  # 平均梯度幅值满分阈值（真实底片实测 ~0.7–8.7，原 18 严重偏高）
    contrast_good: float = 25.0  # 信号对比度（中值滤波后 std）满分阈值
    dr_good: float = 0.6  # 动态范围利用率（(p99−p1)/255）满分阈值
    uniformity_low_freq: float = 0.012  # 低频核占比（相对短边）
    uniformity_max_ratio: float = 6.0  # 低频漂移 σ / 局部 σ 上限
    dust_tophat_k: int = 15  # 尘点 top-hat 核（奇数）
    dust_min_area: int = 10  # 尘点最小面积（px）
    dust_max_count: int = 400  # 尘点连通域上限
    # —— 三类硬门禁（反证测试后补强；任一触发即判不合格，不扰动已标定 RQI 复合分）——
    blur_lap_bad: float = 30.0  # Laplacian 方差低于此值判失焦/模糊（真实底片实测 ≥45）
    exposure_entropy_bad: float = 0.62  # 直方图熵(归一化)低于此值判过/欠曝（真实底片实测 ≥0.73）
    stain_smooth_bad: float = (
        0.15  # 平滑异常斑块占比高于此值判污渍（真实底片实测 ≤0.14；此阈值为保守安全网）
    )


@dataclass(frozen=True)
class QualityReport:
    """质量度量结果（§4.4）。"""

    score: float  # RQI 复合质量分（0–100，越高越好）
    passed: bool  # 是否达到 min_score
    metrics: dict  # 子指标分 + 原始量 + brisque_features
    brisque_features: np.ndarray | None = None  # 36 维无参考特征


def _uniformity_score(gray: np.ndarray, cfg: QualityCfg) -> float:
    """均匀性分：低频漂移 σ / 局部 σ 之比越小越好。"""
    g = gray.astype(np.float64)
    h, w = g.shape
    k = max(3, int(min(h, w) * cfg.uniformity_low_freq))
    k = k if k % 2 == 1 else k + 1
    low = cv2.GaussianBlur(g, (k, k), 0)
    drift = float(np.std(low))
    local = float(np.std(g))
    ratio = drift / max(local, 1e-6)
    return 1.0 - _clip01(ratio / cfg.uniformity_max_ratio)


def _artifact_score(gray: np.ndarray, cfg: QualityCfg) -> float:
    """伪缺陷分：尘点/污渍 top-hat 连通域计数（严重污渍大幅降分）。"""
    g = gray.astype(np.uint8) if gray.dtype != np.uint8 else gray
    tophat = cv2.morphologyEx(
        g, cv2.MORPH_TOPHAT, np.ones((cfg.dust_tophat_k, cfg.dust_tophat_k), np.uint8)
    )
    _, b = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    count = sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= cfg.dust_min_area)
    if count <= cfg.dust_max_count:
        return 1.0 - 0.5 * _clip01(count / cfg.dust_max_count)
    return 0.3


def _laplacian_variance(gray: np.ndarray) -> float:
    """模糊/失焦判据：Laplacian 方差（焦点测度）。值越低越模糊。
    真实底片实测 ≥45；严重失焦(9×9~21×21 高斯模糊)降至 1~23。"""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _exposure_entropy(gray: np.ndarray) -> float:
    """曝光判据：归一化直方图熵(0~1)。过曝/欠曝压窄直方图→熵骤降。
    真实底片实测 ≥0.73；过曝(×2.2)降至 0.45~0.48；严重欠曝(×0.3)降至 0.64~0.76。"""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    p = hist / hist.sum()
    p = p[p > 0]
    ent = -float(np.sum(p * np.log2(p)))
    return ent / 8.0  # log2(256)


def _stain_smooth_frac(
    gray: np.ndarray, res_tol: float = 30.0, std_tol: float = 8.0, min_area: int = 4000
) -> float:
    """污渍判据：大块、色调异常(偏离 151px 低频背景)且内部纹理平滑的区域占比。
    焊接区/边缘纹理强(loc_std 高)→被 std_tol 排除；均匀背景残差≈0→被 res_tol 排除。
    注意：射线底片本身含大块平滑结构(背景/母材/焊缝带)，真实底片此值可达 0.14，
    故 stain_smooth_bad 阈值须设于真实样本上限之上仅作极端安全网。"""
    g = gray.astype(np.float32)
    bg = cv2.GaussianBlur(g, (151, 151), 0)
    residual = np.abs(g - bg)
    ks = 11
    m1 = cv2.blur(g, (ks, ks))
    m2 = cv2.blur(g * g, (ks, ks))
    loc_var = np.clip(m2 - m1 * m1, 0, None)
    loc_std = np.sqrt(loc_var)
    mask = ((residual > res_tol) & (loc_std < std_tol)).astype(np.uint8)
    num, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    big = 0
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            big += int(stats[i, cv2.CC_STAT_AREA])
    return big / g.size


def assess_quality(gray: np.ndarray, cfg: QualityCfg) -> QualityReport:
    """对底片做无参考质量评估（§4.4），返回 RQI 复合分 + 36 维 BRISQUE 特征。

    子指标：噪声(拉普拉斯法)、锐度(平均梯度幅值)、对比度(中值滤波后 std)、
    动态范围利用率(p1–p99)、均匀性(低频漂移)、伪缺陷(尘点)。加权得 RQI。

    另含三类硬门禁（反证测试后补强）：模糊(Laplacian 方差)、曝光(直方图熵)、
    污渍(平滑异常斑块)。任一触发即 passed=False，不扰动已标定的 RQI 复合分。
    """
    g = gray.astype(np.float64)
    if g.size == 0:
        return QualityReport(score=0.0, passed=False, metrics={"score": 0.0}, brisque_features=None)

    noise = estimate_noise(gray)
    s_noise = _clip01((cfg.noise_bad - noise) / (cfg.noise_bad - cfg.noise_good))

    gx = np.gradient(g, axis=1)
    gy = np.gradient(g, axis=0)
    grad = float(np.mean(np.sqrt(gx * gx + gy * gy)))
    s_sharp = _clip01(grad / cfg.sharp_good)

    med = (
        cv2.medianBlur(gray.astype(np.uint8), 5)
        if gray.dtype != np.uint8
        else cv2.medianBlur(gray, 5)
    )
    contrast = float(np.std(med.astype(np.float64)))
    s_contrast = _clip01(contrast / cfg.contrast_good)

    p1, p99 = np.percentile(gray, [1, 99])
    dr = float(p99 - p1)
    s_dr = _clip01(dr / (255.0 * cfg.dr_good))

    s_uniform = _uniformity_score(gray, cfg)
    s_artifact = _artifact_score(gray, cfg)

    scores = [s_noise, s_sharp, s_contrast, s_dr, s_uniform, s_artifact]
    weights = [
        cfg.w_noise,
        cfg.w_sharp,
        cfg.w_contrast,
        cfg.w_dynamic,
        cfg.w_uniform,
        cfg.w_artifact,
    ]
    wsum = sum(weights) or 1.0
    rqi = 100.0 * sum(s * w for s, w in zip(scores, weights)) / wsum

    # —— 三类硬门禁（反证测试后补强）——
    lap_var = _laplacian_variance(gray)
    entropy = _exposure_entropy(gray)
    stain_smooth = _stain_smooth_frac(gray)
    blur_severe = bool(lap_var < cfg.blur_lap_bad)
    exposure_severe = bool(entropy < cfg.exposure_entropy_bad)
    stain_severe = bool(stain_smooth > cfg.stain_smooth_bad)
    hard_fail = blur_severe or exposure_severe or stain_severe

    brisque = brisque_features(gray)
    passed = bool(rqi >= cfg.min_score) and not hard_fail
    metrics = {
        "score": round(rqi, 2),
        "noise": round(noise, 3),
        "noise_score": round(s_noise, 3),
        "sharpness": round(grad, 3),
        "sharpness_score": round(s_sharp, 3),
        "contrast": round(contrast, 3),
        "contrast_score": round(s_contrast, 3),
        "dynamic_range": round(dr, 3),
        "dynamic_range_score": round(s_dr, 3),
        "uniformity_score": round(s_uniform, 3),
        "artifact_score": round(s_artifact, 3),
        "laplacian_var": round(lap_var, 3),
        "blur_severe": blur_severe,
        "exposure_entropy": round(entropy, 4),
        "exposure_severe": exposure_severe,
        "stain_smooth": round(stain_smooth, 6),
        "stain_severe": stain_severe,
        "hard_fail": hard_fail,
        "brisque_features": brisque.tolist(),
    }
    return QualityReport(score=rqi, passed=passed, metrics=metrics, brisque_features=brisque)
