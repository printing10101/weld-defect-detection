"""预处理算法（§4.3，M3 实现）。纯算法，无 I/O。

实现冻结的 Preprocessor 契约（backend/domain/interfaces.py）。
策略与硬约束（§4.3/§11）：以"不损害缺陷边缘"为第一约束；
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
        """保边去噪：双边滤波 + 轻度中值（颗粒噪声）。"""
        b = cv2.bilateralFilter(
            image,
            self.bilateral_d,
            self.bilateral_sigma_color,
            self.bilateral_sigma_space,
        )
        return cv2.medianBlur(b, self.median_k)

    def enhance(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """对比度/灰度校正：Gamma（低黑度补偿）+ CLAHE（局部不均）。"""
        look = (np.power(image.astype(np.float64) / 255.0, gamma) * 255.0).astype(np.uint8)
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip,
            tileGridSize=(self.clahe_grid, self.clahe_grid),
        )
        return clahe.apply(look)

    def edges(
        self,
        image: np.ndarray,
        roi: tuple[int, int, int, int] | None = None,
    ) -> np.ndarray:
        """Canny 边缘：自适应阈值（梯度分位数，§4.3）+ ROI 限定。

        注：Otsu 在稀疏缺陷边缘场景阈值过高导致漏检，故采用
        Sobel 梯度幅值的 90 分位作为高阈值（有下限 20），低阈值取半。
        """
        patch = _slice_roi(image, roi)
        blur = cv2.GaussianBlur(patch, (self.canny_kernel, self.canny_kernel), 0)
        gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0)
        gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1)
        mag = np.sqrt(gx * gx + gy * gy)
        high = float(max(np.percentile(mag, 90), 20.0))
        low = 0.5 * high
        edges = cv2.Canny(blur, low, high)
        if roi is None:
            return edges
        full = np.zeros_like(image)
        x, y, w, h = _clamp_roi(roi, image.shape)
        full[y : y + h, x : x + w] = edges
        return full

    def morph(self, edges: np.ndarray, k_open: int, k_close: int) -> np.ndarray:
        """开运算去孤立杂点、闭运算修复断裂边缘。"""
        k1 = np.ones((max(k_open, 1), max(k_open, 1)), np.uint8)
        k2 = np.ones((max(k_close, 1), max(k_close, 1)), np.uint8)
        opened = cv2.morphologyEx(edges, cv2.MORPH_OPEN, k1)
        return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k2)


def _slice_roi(
    image: np.ndarray, roi: tuple[int, int, int, int] | None
) -> np.ndarray:
    if roi is None:
        return image
    x, y, w, h = _clamp_roi(roi, image.shape)
    return image[y : y + h, x : x + w]


def _clamp_roi(
    roi: tuple[int, int, int, int], shape: tuple[int, ...]
) -> tuple[int, int, int, int]:
    h_img, w_img = shape[:2]
    x, y, w, h = roi
    x = min(max(x, 0), w_img)
    y = min(max(y, 0), h_img)
    w = min(max(w, 0), w_img - x)
    h = min(max(h, 0), h_img - y)
    return x, y, w, h
