"""共享预处理变换契约（ADR-007，§T3）。

训练（backend/models/train.py）与推理（detect 管线）**必须共用**本函数，
保证同图两条路径输出逐元素一致，杜绝"训练/服务预处理漂移"。

输入：单通道灰度 (H,W) 或 (H,W,1)，uint8(0-255) 或 float32(0-1)。
输出：CHW 布局 float32 张量 (3, 640, 640)。
参数（TARGET_SIZE / MEAN / STD / RGB_ORDER）为冻结占位，正式训练调参时
**两条路径同时更新**并经 ADR-007 记录，禁止单边修改。
"""

from __future__ import annotations

import numpy as np

TARGET_SIZE: int = 640
# 归一化参数（冻结占位；训练标定后经 ADR-007 更新，禁止单边修改）
MEAN: tuple[float, float, float] = (0.0, 0.0, 0.0)
STD: tuple[float, float, float] = (1.0, 1.0, 1.0)
RGB_ORDER: bool = False  # False = BGR（OpenCV 习惯）


def to_model_input(raw: np.ndarray) -> np.ndarray:
    """将原始灰度图转为模型输入张量（letterbox 640 + 归一化 + CHW）。"""
    import cv2  # 局部导入：避免无 opencv 环境时模块导入失败

    if raw.ndim == 2:
        raw = raw[..., None]
    img = raw.astype(np.float32)
    if raw.dtype == np.uint8:
        img = img / 255.0

    h, w = img.shape[:2]
    scale = min(TARGET_SIZE / h, TARGET_SIZE / w)
    nw, nh = round(w * scale), round(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((TARGET_SIZE, TARGET_SIZE, 1), np.float32)
    top = (TARGET_SIZE - nh) // 2
    left = (TARGET_SIZE - nw) // 2
    canvas[top : top + nh, left : left + nw, 0] = resized

    canvas = np.repeat(canvas, 3, axis=2)
    if not RGB_ORDER:
        canvas = canvas[..., ::-1]
    canvas = (canvas - np.asarray(MEAN, np.float32)) / np.asarray(STD, np.float32)
    return np.transpose(canvas, (2, 0, 1))  # CHW
