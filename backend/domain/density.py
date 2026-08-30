"""黑度估计与门限。纯算法，无 I/O。

数字化影像中黑度 D 与灰阶 G 的关系（透射式近似）：
    D = log10( 2^bits / (G + 1) )
G=峰值灰阶(全白) → D≈0；G=0(全黑) → D≈bits*log10(2)（8bit≈2.41 / 16bit≈4.82）。
AB 级工艺要求黑度 2.0–4.5（NB/T47013.2 公开解读，门限可配置）。
"""

from __future__ import annotations

import numpy as np


def estimate_density(
    gray: np.ndarray, bit_depth: int | None = None, mask: np.ndarray | None = None
) -> float:
    """估计平均黑度（可限定胶片掩膜内）。支持 uint8/uint16/float32(0-1 归一化)。

    bit_depth 用于 uint16 容器（如 12bit 装于 uint16）中正确还原光学黑度；
    缺省按 16bit 处理。空数组返回 0.0。

    mask（bool，形状须与 gray 一致）给定时常在胶片掩膜上计算：翻拍影像的
    灯箱亮背景会把整图平均灰阶拉高、黑度被严重低估，只统计胶片区才有意义。
    形状不匹配或掩膜为空时忽略掩膜（回退整图）——黑度是门禁输入，须稳健。
    """
    if gray is None or gray.size == 0:
        return 0.0
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape == gray.shape and m.any():
            gray = gray[m]
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
