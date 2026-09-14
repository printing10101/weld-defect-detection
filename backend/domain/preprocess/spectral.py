"""频域分解与频带提示（移植自 WRT-SAM 的 FPG 频域 Prompt 思想）。纯算法，无 I/O。

背景与动机
----------
射线底片是单通道衰减图，天然缺乏色彩与语义线索，检测器只能从灰度梯度里找缺陷。
WRT-SAM（arXiv 2502.11338，中国特种设备检测研究院）针对焊缝 RT 提出频率提示生成
器（FPG）：对灰度图做 2D DCT，按径向频率切成若干环带，再逐带重建回空间域得到
"频带图"，作为额外提示通道补足单通道灰度图的信息缺口。本模块移植其**配方**
（DCT 分带 + 逐带重建），不依赖任何预训练基座，因而可即插即用。

为什么这对焊缝 RT 有物理意义
--------------------------
- 低频带：厚度/密度的大范围起伏（母材、焊缝带、黑度漂移）；
- 中频带：缺陷本体（气孔/夹渣的圆形暗斑，典型尺寸数 px 至数十 px）；
- 高频带：裂纹细边 + 量子噪声 / 胶片颗粒。

现有 :func:`backend.domain.preprocess.metrics.estimate_noise`（拉普拉斯法）无法区分
"高频噪声"与"高频细节（裂纹边缘）"——两者都会抬高 σ。频带能量占比能把它们分开：
裂纹边缘是**各向异性**高频，胶片颗粒近似**各向同性**高频。

归一化约定
----------
全程使用正交归一化 DCT（``norm="ortho"``），因此满足 Parseval 能量守恒：
``sum(gray²) == sum(dct2(gray)²)``。于是"频带能量占比"可直接解释为该频带承载了
多少图像能量，无需额外标定。

频带划分约定
----------
第 ``k`` 个 DCT 系数对应的归一化频率为 ``k / (2N)``（DCT-II 的周期延拓语义），
频点 ``(u, v)`` 的径向频率取 ``sqrt((u/2H)² + (v/2W)²)``，再除以 ``sqrt(0.5)``
（DC 对角方向的近似最大值）映射到 ``[0, 1]`` 便于跨分辨率比较。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "band_edges",
    "band_energy_fractions",
    "band_pass_images",
    "dct2",
    "idct2",
    "normalized_radius",
    "radial_band_masks",
    "spectral_prompt",
]

# DC 对角方向的径向频率上界（sqrt((1/2)² + (1/2)²) 的 1/√2 归一化前身）；
# 用 sqrt(0.5) 作为除数把 r 映射到约 [0,1]。
_RADIUS_NORM = float(np.sqrt(0.5))


def _require_2d(image: np.ndarray, name: str = "image") -> np.ndarray:
    """频域算法契约：非空单通道二维图。返回 float64 数组。"""
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise ValueError(f"{name} 必须为单通道二维灰度图")
    if image.size == 0:
        raise ValueError(f"{name} 为空")
    return image.astype(np.float64, copy=False)


def _check_bands(n_bands: int) -> int:
    if n_bands < 1:
        raise ValueError(f"n_bands 必须为正整数，实得 {n_bands!r}")
    return int(n_bands)


def dct2(image: np.ndarray) -> np.ndarray:
    """2D DCT-II（正交归一化）。

    返回与输入同形的 float64 系数矩阵，低频在左上角，``[0, 0]`` 为 DC。
    scipy 懒加载：后端进程导入期不被 scipy 拖慢（与 ``metrics`` 同策略）。
    """
    img = _require_2d(image)
    from scipy.fft import dctn

    return np.asarray(dctn(img, type=2, norm="ortho"), dtype=np.float64)


def idct2(spectrum: np.ndarray) -> np.ndarray:
    """2D 逆 DCT（正交归一化），:func:`dct2` 的精确逆。"""
    spec = _require_2d(spectrum, "spectrum")
    from scipy.fft import idctn

    return np.asarray(idctn(spec, type=2, norm="ortho"), dtype=np.float64)


def normalized_radius(shape: tuple[int, int]) -> np.ndarray:
    """每个频点的归一化径向频率 ``r ∈ [0, ~1]``（形状同 ``shape``）。

    ``r[u, v] = sqrt((u/2H)² + (v/2W)²) / sqrt(0.5)``；DC 频点 ``r = 0``。
    """
    h, w = int(shape[0]), int(shape[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"shape 必须为正，实得 {shape!r}")
    fu = np.arange(h, dtype=np.float64) / (2.0 * h)
    fv = np.arange(w, dtype=np.float64) / (2.0 * w)
    return np.sqrt(fu[:, None] ** 2 + fv[None, :] ** 2) / _RADIUS_NORM


def band_edges(n_bands: int, *, max_radius: float = 1.0, scheme: str = "linear") -> np.ndarray:
    """频带边界（长度 ``n_bands + 1``，位于 ``[0, max_radius]``）。

    区间为左闭右开，最后一个区间右端**包含**边界值，使所有频点必被且仅被
    一个频带覆盖（完备且互斥）。

    ``scheme`` 决定边界分布：

    - ``"linear"``：径向等分。数学最直观，但真实射线底片能量高度集中于低频
      （实测 2448×2048 底片低频占比 0.9993），等分会让绝大多频带近乎全空、
      特征失去动态范围；
    - ``"log"``：几何级数（低频窄、高频宽），把频率分辨率还给低频。真实底片上
      推荐用这一档。
    """
    n = _check_bands(n_bands)
    if not 0.0 < max_radius <= 1.0:
        raise ValueError(f"max_radius 须在 (0, 1] 内，实得 {max_radius!r}")
    if scheme == "linear":
        return np.linspace(0.0, float(max_radius), n + 1, dtype=np.float64)
    if scheme == "log":
        top = float(max_radius)
        # n == 1 时 geomspace(x, top, 1) 只返回起点（不含终点），会让边界停在 x
        # 而非 top，导致 (x, top] 的频点无频带覆盖、能量占比之和不再为 1。
        inner = np.geomspace(max(0.02 * top, 1e-6), top, n) if n > 1 else np.array([top])
        return np.concatenate([[0.0], inner]).astype(np.float64)
    raise ValueError(f"未知 scheme：{scheme!r}（可选 linear / log）")


def radial_band_masks(
    shape: tuple[int, int],
    n_bands: int = 4,
    *,
    edges: np.ndarray | None = None,
    max_radius: float = 1.0,
    scheme: str = "linear",
) -> list[np.ndarray]:
    """径向频带掩膜列表（布尔数组，长度 = 频带数）。

    掩膜互斥且并集覆盖全部频点。``edges`` 给定时忽略 ``n_bands`` / ``max_radius``
    / ``scheme``（长度须为频带数 + 1 且严格递增）。
    """
    r = normalized_radius(shape)
    if edges is None:
        e = band_edges(n_bands, max_radius=max_radius, scheme=scheme)
    else:
        e = np.asarray(edges, dtype=np.float64)
        if e.ndim != 1 or e.size < 2:
            raise ValueError("edges 必须为一维且至少含 2 个边界")
        if not np.all(np.diff(e) > 0):
            raise ValueError("edges 必须严格递增")
    masks: list[np.ndarray] = []
    for i in range(e.size - 1):
        lo, hi = float(e[i]), float(e[i + 1])
        is_last = i == e.size - 2
        m = (r >= lo) & (r <= hi if is_last else r < hi)
        masks.append(m)
    return masks


def band_pass_images(
    gray: np.ndarray,
    n_bands: int = 4,
    *,
    edges: np.ndarray | None = None,
    max_radius: float = 1.0,
    scheme: str = "linear",
) -> np.ndarray:
    """逐频带重建的空间域频带图，形状 ``(n_bands, H, W)``。

    做法：整图做 DCT → 每带只保留本带系数、其余置零 → 逆 DCT 回空间域。
    由于 DCT 是线性正交变换且频带划分为完备划分，各带图**求和恰为原图**
    （数值误差 <1e-9，单测覆盖该可加性）。

    返回值带符号（零均值化后的振荡量），保留相位信息；如需 DL 输入用的
    ``[0, 1]`` 提示张量请用 :func:`spectral_prompt`。
    """
    img = _require_2d(gray)
    spec = dct2(img)
    masks = radial_band_masks(img.shape, n_bands, edges=edges, max_radius=max_radius, scheme=scheme)
    out = np.empty((len(masks), *img.shape), dtype=np.float64)
    for i, mask in enumerate(masks):
        out[i] = idct2(np.where(mask, spec, 0.0))
    return out


def band_energy_fractions(
    gray: np.ndarray,
    n_bands: int = 4,
    *,
    edges: np.ndarray | None = None,
    max_radius: float = 1.0,
    scheme: str = "linear",
) -> np.ndarray:
    """各频带能量占比，形状 ``(n_bands,)``，和为 1（Parseval 保证）。

    这是对频域分布最凝练的描述量：低频占比高 = 图像以大范围灰度起伏为主；
    高频占比异常高 = 噪声/颗粒主导（可用于判别"伪缺陷"）。全零图返回等分占比。

    真实射线底片能量高度集中于低频，**建议配 ``scheme="log"``**——否则高频带
    占比会小到接近 1e-5、跨图方差近乎为零，失去判别力（实测见 ADR-012）。
    """
    img = _require_2d(gray)
    spec = dct2(img)
    energy = spec * spec
    total = float(energy.sum())
    masks = radial_band_masks(img.shape, n_bands, edges=edges, max_radius=max_radius, scheme=scheme)
    if total <= 0.0:
        return np.full(len(masks), 1.0 / len(masks), dtype=np.float64)
    return np.array([float(energy[m].sum()) / total for m in masks], dtype=np.float64)


def spectral_prompt(
    gray: np.ndarray,
    n_bands: int = 4,
    *,
    edges: np.ndarray | None = None,
    max_radius: float = 1.0,
    scheme: str = "linear",
    eps: float = 1e-12,
) -> np.ndarray:
    """频域提示张量，形状 ``(n_bands, H, W)``，值域 ``[0, 1]``，以 0.5 表示"该频带无响应"。

    两步预处理缺一不可（两者都由单测锁定，去掉任一步该通道即退化）：

    1. **逐带去均值**——低频带含 DC（常数分量），不去掉时该通道会整体被抬到
       1.0 附近（实测值域曾塌成 ``[0.84, 1.0]``），动态范围被压扁、丧失提示作用；
    2. **逐带（而非全局）归一化**——各带能量可差数个数量级，全局缩放会让高频带
       彻底塌成常数。

    于是 ``prompt = 0.5 + 0.5 * (bp - mean(bp)) / (max|bp - mean(bp)| + eps)``，
    每个通道都以 0.5 居中、至少一端触达 0 或 1。常数图退化为全 0.5（无信息，符合语义）。

    用途：拼到检测器输入上（``(1 + n_bands, H, W)`` 多通道），或作为分割网络的
    提示特征。不依赖任何预训练基座，因此可直接用于现有 YOLO 链路的数据准备。
    """
    if eps <= 0.0:
        raise ValueError(f"eps 必须为正，实得 {eps!r}")
    bps = band_pass_images(gray, n_bands, edges=edges, max_radius=max_radius, scheme=scheme)
    centered = bps - bps.mean(axis=(1, 2), keepdims=True)
    peak = np.max(np.abs(centered), axis=(1, 2), keepdims=True)
    return np.asarray(0.5 + 0.5 * centered / (peak + eps), dtype=np.float64)
