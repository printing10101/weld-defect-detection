"""线型 IQI 识别测试（§4.2，M2 基线）。"""

from __future__ import annotations

import cv2
import numpy as np

from backend.domain.iqi import (
    IqiConfig,
    enrich_grade,
    locate_iqi,
    map_sensitivity_grade,
    verify_hole_iqi,
    verify_iqi,
    verify_wire_iqi,
)

_N_WIRES = 19


def _synthetic_iqi(
    amps: list[float], n: int = _N_WIRES, w: int = 640, h: int = 190, bright: bool = True
) -> np.ndarray:
    """生成合成像质计图。bright=True 画亮线（线型丝），bright=False 画暗线（孔型孔）。"""
    rng = np.random.default_rng(0)
    img = rng.normal(128.0, 2.0, (h, w)).astype(np.uint8)
    for i, amp in enumerate(amps):
        y = round((i + 0.5) / n * h)
        val = int(128 + amp) if bright else int(max(0, 128 - amp))
        cv2.line(img, (0, y), (w - 1, y), val, 3)
    return img


def _cfg(required: int = 10) -> IqiConfig:
    return IqiConfig(
        wire_diameters_mm=tuple(float(i) for i in range(1, _N_WIRES + 1)),
        required_wire_no=required,
    )


def _cfg_hole(required: int = 10) -> IqiConfig:
    return IqiConfig(
        type="hole",
        hole_diameters_mm=tuple(float(i) for i in range(1, _N_WIRES + 1)),
        required_hole_no=required,
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


def test_hole_all_visible() -> None:
    img = _synthetic_iqi([40.0] * _N_WIRES, bright=False)
    res = verify_hole_iqi(img, _cfg_hole())
    assert res.iqi_type == "hole"
    assert res.achieved == str(_N_WIRES)
    assert res.passed is True


def test_hole_thin_invisible_fails() -> None:
    amps = [50.0] * 10 + [4.0] * 9  # 孔号 1..10 可见，11..19 不可见
    img = _synthetic_iqi(amps, bright=False)
    res = verify_hole_iqi(img, _cfg_hole(required=12))
    assert res.achieved == "10"
    assert res.passed is False


def test_verify_iqi_dispatches_hole() -> None:
    img = _synthetic_iqi([40.0] * _N_WIRES, bright=False)
    res = verify_iqi(img, _cfg_hole(), iqi_type="hole")
    assert res.iqi_type == "hole"
    assert res.achieved == str(_N_WIRES)


def test_verify_iqi_default_uses_cfg_type() -> None:
    img = _synthetic_iqi([40.0] * _N_WIRES)
    # 未传 iqi_type → 回落 cfg.type（默认 wire）
    res = verify_iqi(img, _cfg())
    assert res.iqi_type == "wire"


# --------------------------------------------------------------------------
# M2 增强：IQI 自动定位（§4.2 模板匹配/小目标检测）+ A/AB/B 等级映射
# --------------------------------------------------------------------------


def _locate_cfg() -> IqiConfig:
    return IqiConfig(
        wire_diameters_mm=tuple(float(i) for i in range(1, _N_WIRES + 1)),
        required_wire_no=10,
        hole_diameters_mm=tuple(float(i) for i in range(1, _N_WIRES + 1)),
        required_hole_no=10,
    )


def test_locate_wire_full_frame() -> None:
    """全幅线型 IQI（丝跨全宽、有间隙）→ 自动定位成功且验证 ≥2 可见。"""
    img = _synthetic_iqi([40.0] * _N_WIRES)
    band = locate_iqi(img, _locate_cfg(), iqi_type="wire", threshold=0.3)
    assert band is not None
    sub = img[band[1] : band[1] + band[3], band[0] : band[0] + band[2]]
    res = verify_wire_iqi(sub, _locate_cfg())
    assert res.achieved is not None and int(res.achieved) >= 2


def test_locate_hole() -> None:
    """孔型 IQI（暗线）→ 自动定位成功且验证 ≥2 可见。"""
    img = _synthetic_iqi([40.0] * _N_WIRES, bright=False)
    band = locate_iqi(img, _locate_cfg(), iqi_type="hole", threshold=0.3)
    assert band is not None
    sub = img[band[1] : band[1] + band[3], band[0] : band[0] + band[2]]
    res = verify_hole_iqi(sub, _locate_cfg())
    assert res.achieved is not None and int(res.achieved) >= 2


def test_locate_none_on_pure_noise() -> None:
    """纯噪声（无结构）→ 返回 None（不误报）。"""
    rng = np.random.default_rng(7)
    img = rng.normal(128.0, 4.0, (200, 200)).astype(np.uint8)
    assert locate_iqi(img, _locate_cfg(), iqi_type="wire") is None


def test_locate_none_on_single_scratch() -> None:
    """单条长划痕 → 边缘能量带被定位但验证单元数 <2 → 回退 None。"""
    rng = np.random.default_rng(3)
    img = rng.normal(128.0, 2.0, (400, 400)).astype(np.uint8)
    cv2.line(img, (0, 200), (399, 200), 200, 3)
    assert locate_iqi(img, _locate_cfg(), iqi_type="wire") is None


def test_locate_none_on_degenerate_block() -> None:
    """退化情形（丝间距 ≤ 丝宽，无间隙 → 合并为无内部边缘的实心块）。

    此类配置在物理上不可分辨（真实 IQI 丝间必有可分辨间隙），边缘能量法
    正确返回 None——它与「平滑亮区」不可区分，强制定位会引入误报。
    属可接受边界，记为已知限制而非缺陷。
    """
    rng = np.random.default_rng(0)
    img = rng.normal(128.0, 2.0, (400, 600)).astype(np.uint8)
    for i in range(_N_WIRES):
        y = round((i + 0.5) / _N_WIRES * 40)  # 19 根 2px 丝挤在顶部 40px
        cv2.line(img, (0, y), (599, y), 170, 2)
    assert locate_iqi(img, _locate_cfg(), iqi_type="wire") is None


def test_map_sensitivity_grade_tiers() -> None:
    """A/AB/B 等级随厚度与可达丝号变化；缺失厚度/丝号返回 None。"""
    table = IqiConfig().sensitivity
    # 薄壁 (≤2mm) 要求丝号 14：achieved 19 → A；厚壁 (100mm) 要求 A=6。
    assert map_sensitivity_grade(19, 1.0, table) == "A"
    assert map_sensitivity_grade(19, 100.0, table) == "A"
    # achieved=5 @100mm: 要求 A6/AB5/B4 → AB
    assert map_sensitivity_grade(5, 100.0, table) == "AB"
    # achieved=4 @100mm → B
    assert map_sensitivity_grade(4, 100.0, table) == "B"
    # achieved=3 @100mm → 连 B 都不及 → None
    assert map_sensitivity_grade(3, 100.0, table) is None
    # 缺失厚度或丝号 → 不臆造
    assert map_sensitivity_grade(19, None, table) is None
    assert map_sensitivity_grade(None, 1.0, table) is None


def test_enrich_grade() -> None:
    """enrich_grade 用厚度表补全 IQIResult.grade 且不破坏原字段。"""
    from backend.domain.dto import IQIResult

    res = IQIResult(iqi_type="wire", achieved="19", required="10", passed=True)
    out = enrich_grade(res, 1.0, IqiConfig().sensitivity)
    assert out.grade == "A"
    assert out.achieved == "19" and out.passed is True
    # 厚度缺失 → grade 为 None
    out2 = enrich_grade(res, None, IqiConfig().sensitivity)
    assert out2.grade is None
