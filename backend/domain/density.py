"""黑度估计与门限。纯算法，无 I/O。

数字化影像中黑度 D 与灰阶 G 的关系（透射式近似）：
    D = log10( 2^bits / (G + 1) )
G=峰值灰阶(全白) → D≈0；G=0(全黑) → D≈bits*log10(2)（8bit≈2.41 / 16bit≈4.82）。
AB 级工艺要求黑度 2.0–4.5（NB/T47013.2 公开解读，门限可配置）。
"""

from __future__ import annotations

import numpy as np


def estimate_density(gray: np.ndarray, bit_depth: int | None = None) -> float:
    """估计整图平均黑度。支持 uint8/uint16/float32(0-1 归一化)。

    bit_depth 用于 uint16 容器（如 12bit 装于 uint16）中正确还原光学黑度；
    缺省按 16bit 处理。空数组返回 0.0。
    """
    if gray is None or gray.size == 0:
        return 0.0
    if gray.dtype == np.uint8:
        bits = 8
    elif gray.dtype == np.uint16:
        bits = bit_depth if bit_depth in (12, 16) else 16
    else:
        # float32：[0,1] 归一化，按 16bit 当量计算
        a = np.clip(gray.astype(np.float64), 0.0, 1.0)
        return float(np.mean(np.log10(65536.0 / (a * 65535.0 + 1.0))))
    a = gray.astype(np.float64)
    return float(np.mean(np.log10((2.0**bits) / (a + 1.0))))


def check_density(density: float, low: float = 2.0, high: float = 4.5) -> bool:
    """AB 级黑度范围判定（门限可配置， 禁硬编码）。"""
    return low <= density <= high
