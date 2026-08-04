"""预处理质量度量（§4.4）。纯算法，无 I/O。

- 有参考：PSNR / SSIM（skimage）
- 无参考：拉普拉斯法噪声估计（σ ≈ std(Laplacian)/√6，适用于高斯噪声）
BRISQUE（无参考盲评）需独立模型文件，留作可扩展项（§4.4 备注）。
"""
from __future__ import annotations

import cv2
import numpy as np


def psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """峰值信噪比（dB），data_range 按 uint8=255。"""
    from skimage.metrics import peak_signal_noise_ratio

    return float(peak_signal_noise_ratio(original, processed, data_range=255))


def ssim(original: np.ndarray, processed: np.ndarray) -> float:
    """结构相似度 [0,1]，data_range 按 uint8=255。"""
    from skimage.metrics import structural_similarity

    val = np.asarray(
        structural_similarity(original, processed, data_range=255),
        dtype=np.float64,
    )
    return float(np.mean(val))


def estimate_noise(image: np.ndarray) -> float:
    """无参考噪声估计（拉普拉斯法）：σ ≈ std(Laplacian)/√6。"""
    lap = np.asarray(cv2.Laplacian(image, cv2.CV_64F), dtype=np.float64)
    return float(np.std(lap) / np.sqrt(6.0))
