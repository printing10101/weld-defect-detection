"""量化扩展（G13 中心线 / G14 钟点位轴向 / G17 最近邻间距）单测。"""

from __future__ import annotations

import cv2
import numpy as np

from backend.domain.dto import BBox, DefectClass, Detection
from backend.domain.quantify import (
    MaskQuantifier,
    clock_position,
    nearest_neighbor_gaps,
    weld_axis,
)


def _det(id_: str, x: float, y: float, w: float, h: float) -> Detection:
    return Detection(
        id=id_,
        bbox=BBox(x=x, y=y, w=w, h=h),
        class_id=DefectClass.SLAG,
        score=0.9,
        uncertainty=0.1,
    )


# ---------------------------------------------------------------------------
# G13 中心线
# ---------------------------------------------------------------------------
def _linear_mask_bent() -> tuple[np.ndarray, Detection]:
    """合成弯曲条形缺陷（S 形暗带）+ 包住它的检测框。"""
    img = np.full((120, 240), 40000, dtype=np.uint16)
    xs = np.arange(40, 200)
    for x in xs:
        cy = int(60 + 18 * np.sin((x - 40) / 160 * 2 * np.pi))
        img[cy - 2 : cy + 3, x] = 8000  # 宽 ~5px 的弯带
    det = _det("d1", 35, 35, 170, 55)
    return img, det


def test_centerline_used_for_linear_defect():
    img, det = _linear_mask_bent()
    g = MaskQuantifier().quantify_from_image(img, det, pixel_spacing_mm=0.1)
    # 弯带弦长（矩形长边）≈170px，弧长必然更长；中心线字段应产出且 ≥ 0.9×length
    assert g.centerline_mm is not None
    assert g.centerline_mm >= g.length_mm * 0.9
    # 条形缺陷 length 采用中心线口径
    assert g.length_mm == g.centerline_mm


def test_round_defect_has_no_centerline():
    """圆形缺陷长度保持矩形口径，centerline 为 None。"""
    img = np.full((80, 80), 40000, dtype=np.uint16)
    cv2.circle(img, (40, 40), 8, 8000, -1)
    det = _det("d1", 30, 30, 20, 20)
    g = MaskQuantifier().quantify_from_image(img, det, pixel_spacing_mm=0.1)
    assert g.centerline_mm is None


# ---------------------------------------------------------------------------
# G14 钟点位 / 轴向
# ---------------------------------------------------------------------------
def test_clock_position_quadrants():
    """以焊缝圆心为参考的四象限钟点位（y 向下坐标系）。"""
    cx, cy = 100.0, 100.0
    assert clock_position(cx, cy, 100.0, 0.0) == "12:00"  # 正上
    assert clock_position(cx, cy, 200.0, 100.0) == "3:00"  # 正右
    assert clock_position(cx, cy, 100.0, 200.0) == "6:00"  # 正下
    assert clock_position(cx, cy, 0.0, 100.0) == "9:00"  # 正左


def test_clock_position_half_hour_quantization():
    # 右上 45° → 1:30
    assert clock_position(0.0, 0.0, 1.0, -1.0) == "1:30"


def test_weld_axis_from_film_box():
    assert weld_axis((0, 0, 800, 100)) == "h"
    assert weld_axis((0, 0, 100, 800)) == "v"
    assert weld_axis((0, 0, 100, 100)) == "h"  # 近方形保守默认


# ---------------------------------------------------------------------------
# G17 最近邻间距
# ---------------------------------------------------------------------------
def test_nearest_neighbor_gaps():
    a = _det("a", 10, 10, 10, 10)
    b = _det("b", 40, 10, 10, 10)  # 距 a 边缘 20px
    c = _det("c", 80, 80, 10, 10)
    gaps = nearest_neighbor_gaps([a, b, c])
    assert gaps["a"] == ("b", 20.0)
    assert gaps["b"] == ("a", 20.0)
    assert gaps["c"][0] in {"a", "b"}


def test_nearest_neighbor_single_defect_empty():
    assert nearest_neighbor_gaps([_det("a", 0, 0, 5, 5)]) == {}


def test_nearest_neighbor_overlapping_gap_zero():
    a = _det("a", 10, 10, 10, 10)
    b = _det("b", 15, 10, 10, 10)  # x 方向重叠 → gap 0
    assert nearest_neighbor_gaps([a, b])["a"] == ("b", 0.0)
