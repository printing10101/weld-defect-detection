"""NB/T47013 规则引擎测试（§6，M5）。"""
from __future__ import annotations

import pytest

from backend.domain.dto import BBox, DefectClass, Detection, ImageMeta, JointLevel, Modality
from backend.domain.errors import GradingAmbiguousError
from backend.domain.grade.nb47013 import Nb47013Grader
from backend.domain.standards.tables.loader import StandardTables

_AUTHORIZED = StandardTables(
    standard_id="NB/T47013.2-2015",
    version="2015",
    authorized=True,
    data={
        "round_rating_zone_mm": [
            {"max_t": 25, "width": 10},
            {"max_t": 100, "width": 20},
            {"max_t": 999999, "width": 30},
        ],
        "round_points": [
            {"max_d_mm": 1.0, "points": 1},
            {"max_d_mm": 2.0, "points": 2},
            {"max_d_mm": 3.0, "points": 3},
            {"max_d_mm": 4.0, "points": 6},
            {"max_d_mm": 6.0, "points": 12},
            {"max_d_mm": 10.0, "points": 28},
        ],
        "round_ignore_size_mm": [{"max_t": 25, "max_d": 0.5}, {"max_t": 999999, "max_d": 1.5}],
        "round_grade_limits": [{"max_t": 10, "I": 3, "II": 6, "III": 12}, {"max_t": 999999, "I": 6, "II": 12, "III": 18}],
        "linear_limits": {
            "level2": {"t_factor": 0.333, "min_mm": 4, "max_mm": 20},
            "level3": {"t_factor": 0.667, "min_mm": 6, "max_mm": 30},
        },
    },
)

_UNAUTHORIZED = StandardTables(
    standard_id="NB/T47013.2-2015",
    version="2015",
    authorized=False,
    data={},
)


def _ctx(t: float, spacing: float = 0.1) -> ImageMeta:
    return ImageMeta(
        modality=Modality.GENERIC,
        pixel_spacing_mm=spacing,
        base_metal_thickness_mm=t,
    )


def _det(class_id: DefectClass, w_px: float, h_px: float) -> Detection:
    return Detection(
        id="d",
        bbox=BBox(0, 0, w_px, h_px),
        class_id=class_id,
        score=0.5,
        uncertainty=0.5,
    )


def test_unauthorized_fuses() -> None:
    grader = Nb47013Grader(_UNAUTHORIZED)
    with pytest.raises(GradingAmbiguousError):
        grader.grade([_det(DefectClass.POROSITY, 10, 10)], _ctx(10))


def test_zero_tolerance_crack_iv() -> None:
    grader = Nb47013Grader(_AUTHORIZED)
    res = grader.grade([_det(DefectClass.CRACK, 5, 50)], _ctx(20))
    assert res.joint_level is JointLevel.IV
    assert res.need_review is True


def test_round_points_accumulate_to_iv() -> None:
    grader = Nb47013Grader(_AUTHORIZED)
    # 方形缺陷（ratio=1 → 圆形）；T=40（T/2=20，10mm 不直判）；3×10mm → 84 点 > 18 → IV
    res = grader.grade([_det(DefectClass.POROSITY, 100, 100)] * 3, _ctx(40))
    assert res.joint_level is JointLevel.IV


def test_round_diameter_over_half_t_direct_iv() -> None:
    grader = Nb47013Grader(_AUTHORIZED)
    # T=8：方形长径 5mm > T/2=4 → 直判 IV（与点数无关）
    res = grader.grade([_det(DefectClass.POROSITY, 50, 50)], _ctx(8))
    assert res.joint_level is JointLevel.IV


def test_round_grade_ii_by_points() -> None:
    grader = Nb47013Grader(_AUTHORIZED)
    # T=20：方形长径 5mm（<T/2=10）→ 12 点 → 上限 I:6 II:12 III:18 → II
    res = grader.grade([_det(DefectClass.POROSITY, 50, 50)], _ctx(20))
    assert res.joint_level is JointLevel.II


def test_linear_single_length_grade() -> None:
    grader = Nb47013Grader(_AUTHORIZED)
    # T=30：II 限值 min(max(30/3,4),20)=min(10,20)=10mm；单个 5mm → II
    res = grader.grade([_det(DefectClass.SLAG, 50, 5)], _ctx(30))
    assert res.joint_level is JointLevel.II


def test_combined_rating() -> None:
    grader = Nb47013Grader(_AUTHORIZED)
    # 圆形（2mm→2 点→I 级） + 条形（5mm→II 级） → 1+2-1=2 → II
    round_d = _det(DefectClass.POROSITY, 20, 20)  # 方形 2mm → 圆形
    linear_d = _det(DefectClass.SLAG, 50, 5)  # ratio 10 → 条形
    res = grader.grade([round_d, linear_d], _ctx(30))
    assert res.joint_level is JointLevel.II
