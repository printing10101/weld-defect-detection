"""train/serve 预处理一致性测试（ADR-007 / §T3）。"""
from __future__ import annotations

import numpy as np

from backend.domain.preprocess.transform import to_model_input


def test_train_serve_parity() -> None:
    """同一输入两次调用必须逐元素一致（train 与 serve 共用的保证）。"""
    raw = np.zeros((800, 600), dtype=np.uint8)
    a = to_model_input(raw)
    b = to_model_input(raw)
    assert (a == b).all()
    assert a.shape == (3, 640, 640)
    assert a.dtype == np.float32


def test_uint8_and_float32_same_scale() -> None:
    raw_u8 = np.full((64, 64), 255, np.uint8)
    raw_f32 = np.ones((64, 64), np.float32)  # 1.0 == 255/255
    assert (to_model_input(raw_u8) == to_model_input(raw_f32)).all()
