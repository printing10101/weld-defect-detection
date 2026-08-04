"""线型 IQI 识别测试（§4.2，M2 基线）。"""
from __future__ import annotations

import cv2
import numpy as np

from backend.domain.iqi import IqiConfig, verify_wire_iqi

_N_WIRES = 19


def _synthetic_iqi(amps: list[float], n: int = _N_WIRES, w: int = 640, h: int = 190) -> np.ndarray:
    rng = np.random.default_rng(0)
    img = rng.normal(128.0, 2.0, (h, w)).astype(np.uint8)
    for i, amp in enumerate(amps):
        y = round((i + 0.5) / n * h)
        cv2.line(img, (0, y), (w - 1, y), int(128 + amp), 3)
    return img


def _cfg(required: int = 10) -> IqiConfig:
    return IqiConfig(
        wire_diameters_mm=tuple(float(i) for i in range(1, _N_WIRES + 1)),
        required_wire_no=required,
    )


def test_all_wires_visible() -> None:
    img = _synthetic_iqi([40.0] * _N_WIRES)
    res = verify_wire_iqi(img, _cfg())
    assert res.achieved == str(_N_WIRES)
    assert res.passed is True


def test_thin_wires_invisible_fails() -> None:
    amps = [50.0] * 10 + [4.0] * 9  # 丝号 1..10 可见，11..19 不可见
    img = _synthetic_iqi(amps)
    res = verify_wire_iqi(img, _cfg(required=12))
    assert res.achieved == "10"
    assert res.passed is False


def test_required_met() -> None:
    img = _synthetic_iqi([50.0] * 12 + [4.0] * 7)
    res = verify_wire_iqi(img, _cfg(required=10))
    assert res.achieved == "12"
    assert res.passed is True


def test_roi_out_of_range_safe() -> None:
    """越界 ROI 必须优雅降级：返回不通过而非崩溃。"""
    img = _synthetic_iqi([40.0] * _N_WIRES)
    roi = (0, 500, 640, 100)  # y 起点超出图像高度
    res = verify_wire_iqi(img, _cfg(required=19), roi=roi)
    assert res.achieved is None
    assert res.passed is False
