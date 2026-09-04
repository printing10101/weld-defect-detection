"""预处理算法。纯算法，无 I/O。

实现冻结的 Preprocessor 契约（backend/domain/interfaces.py）。
策略与硬约束：以"不损害缺陷边缘"为第一约束；
可调参数经构造器注入（来自 configs，禁硬编码）。
- 降噪：bilateral（保边去噪）+ 轻度 median（胶片颗粒）
- 增强：Gamma（低黑度补偿）+ CLAHE（局部灰度不均）
- 边缘：Canny 自适应阈值（Otsu 推导高低阈值）+ ROI 限定
- 形态学：开运算去孤立杂点、闭运算修复断裂边缘
"""

from __future__ import annotations

import cv2
import numpy as np


class OpencvPreprocessor:
    """OpenCV 实现的预处理管线。"""

    def __init__(
        self,
        bilateral_d: int = 9,
        bilateral_sigma_color: float = 75.0,
        bilateral_sigma_space: float = 75.0,
        median_k: int = 3,
        clahe_clip: float = 2.0,
        clahe_grid: int = 8,
        canny_kernel: int = 5,
        morph_k_open: int = 3,
        morph_k_close: int = 3,
    ) -> None:
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.median_k = median_k
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid
        self.canny_kernel = canny_kernel
        self.morph_k_open = morph_k_open
        self.morph_k_close = morph_k_close

    def denoise(self, image: np.ndarray) -> np.ndarray:
        """保边去噪：双边滤波 + 轻度中值（颗粒噪声）。

        cv2.bilateralFilter 只接受 CV_8U / CV_32F，直接喂 16bit 底片会抛
        cv2.error；这里对非 8bit 输入走 float32 中转再还原量纲。
        """
        _require_2d(image)
        ksize = self.median_k if self.median_k % 2 == 1 else self.median_k + 1
        ksize = max(3, ksize)
        if image.dtype == np.uint8:
            b = cv2.bilateralFilter(
                image,
                self.bilateral_d,
                self.bilateral_sigma_color,
                self.bilateral_sigma_space,
            )
            return cv2.medianBlur(b, min(ksize, 5))
        work = image.astype(np.float32)
        b = cv2.bilateralFilter(
            work,
            self.bilateral_d,
            self.bilateral_sigma_color,
            self.bilateral_sigma_space,
        )
        # medianBlur 对 CV_32F 仅支持 ksize 3/5
        b = cv2.medianBlur(b, min(ksize, 5))
        if np.issubdtype(image.dtype, np.integer):
            info = np.iinfo(image.dtype)
            return np.clip(np.rint(b), info.min, info.max).astype(image.dtype)
        return b.astype(image.dtype)

    def enhance(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """对比度/灰度校正：Gamma（低黑度补偿）+ CLAHE（局部不均）。

        实现要点：
        1. gamma≈1.0 直接跳过幂运算（对 4k 底片省去约 128 MB 的 float64
           临时内存且毫无效果）；
        2. 8bit 走 256 项 LUT（cv2.LUT），复杂度与像素位深解耦；
        3. 量纲按 dtype 取（固定除以 255 时 16bit 输入会 >1 再溢出回绕，
           产出雪花噪声），四舍五入而非截断，消除 -0.5 LSB 系统偏差。
        """
        _require_2d(image)
        if not np.isfinite(gamma) or gamma <= 0.0:
            raise ValueError(f"gamma 必须为正有限值，实得 {gamma!r}")
        look = image if abs(gamma - 1.0) < 1e-9 else _apply_gamma(image, gamma)
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip,
            tileGridSize=(max(1, self.clahe_grid), max(1, self.clahe_grid)),
        )
        # CLAHE 仅支持 CV_8UC1 / CV_16UC1
        if look.dtype not in (np.uint8, np.uint16):
            look = cv2.normalize(look, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # type: ignore[arg-type]
        return clahe.apply(look)

    def edges(
        self,
        image: np.ndarray,
        roi: tuple[int, int, int, int] | None = None,
    ) -> np.ndarray:
        """Canny 边缘：自适应阈值（梯度分位数，）+ ROI 限定。

        注：Otsu 在稀疏缺陷边缘场景阈值过高导致漏检，故采用
        Sobel 梯度幅值的 90 分位作为高阈值（有下限 20），低阈值取半。
        """
        _require_2d(image)
        out = np.zeros(image.shape[:2], np.uint8)  # 输出恒为 8bit 掩膜
        patch = _slice_roi(image, roi)
        if patch.size == 0:
            return out  # 空 ROI：直接送 GaussianBlur 会抛 cv2.error
        # Canny 只接受 CV_8U；16bit 底片须先做量纲映射
        patch8 = (
            patch
            if patch.dtype == np.uint8
            else cv2.normalize(patch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # type: ignore[arg-type]
        )
        k = self.canny_kernel if self.canny_kernel % 2 == 1 else self.canny_kernel + 1
        k = max(3, min(k, 31))
        blur = cv2.GaussianBlur(patch8, (k, k), 0)
        gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0)
        gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1)
        mag = np.sqrt(gx * gx + gy * gy)
        high = float(max(np.percentile(mag, 90), 20.0))
        low = 0.5 * high
        # L2gradient=True：用真实欧氏梯度幅值而非 |gx|+|gy| 近似，与上面按
        # sqrt(gx²+gy²) 分位数推导的阈值同量纲，否则阈值被系统性高估。
        edges = cv2.Canny(blur, low, high, L2gradient=True)
        if roi is None:
            return edges
        x, y, w, h = _clamp_roi(roi, image.shape)
        out[y : y + h, x : x + w] = edges
        return out

    def morph(self, edges: np.ndarray, k_open: int, k_close: int) -> np.ndarray:
        """开运算去孤立杂点、闭运算修复断裂边缘。"""
        k1 = np.ones((max(k_open, 1), max(k_open, 1)), np.uint8)
        k2 = np.ones((max(k_close, 1), max(k_close, 1)), np.uint8)
        opened = cv2.morphologyEx(edges, cv2.MORPH_OPEN, k1)
        return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k2)


def _require_2d(image: np.ndarray) -> None:
    """预处理契约：单通道二维灰度图且非空。"""
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise ValueError("预处理仅接受单通道二维灰度图")
    if image.size == 0:
        raise ValueError("输入影像为空")


def _apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """按 dtype 满量程做 Gamma 校正（8bit 走 LUT，其余走浮点）。"""
    if image.dtype == np.uint8:
        lut = (
            np.rint(np.power(np.arange(256, dtype=np.float32) / 255.0, gamma) * 255.0)
            .clip(0, 255)
            .astype(np.uint8)
        )
        return cv2.LUT(image, lut)
    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        full = float(info.max)
        out = np.power(image.astype(np.float32) / full, gamma) * full
        return np.clip(np.rint(out), info.min, info.max).astype(image.dtype)
    # 浮点影像按 [0,1] 归一化语义处理
    return np.power(np.clip(image.astype(np.float32), 0.0, 1.0), gamma)


def _slice_roi(image: np.ndarray, roi: tuple[int, int, int, int] | None) -> np.ndarray:
    if roi is None:
        return image
    x, y, w, h = _clamp_roi(roi, image.shape)
    return image[y : y + h, x : x + w]


def _clamp_roi(roi: tuple[int, int, int, int], shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    """把 ROI 夹到影像范围内并强制整型（浮点索引会让 numpy 切片抛 TypeError）。"""
    h_img, w_img = shape[:2]
    x, y, w, h = (int(v) for v in roi)
    x = min(max(x, 0), w_img)
    y = min(max(y, 0), h_img)
    w = min(max(w, 0), w_img - x)
    h = min(max(h, 0), h_img - y)
    return x, y, w, h
