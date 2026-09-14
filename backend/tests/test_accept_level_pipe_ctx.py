"""G20 验收级别合格性判定 + G18 管径上下文备查 单测。"""

from __future__ import annotations

from backend.domain.dto import BBox, DefectClass, Detection, ImageMeta, JointLevel, Modality
from backend.domain.grade.nb47013 import Nb47013Grader
from backend.domain.recommend import DISPOSITION_ACCEPT, DISPOSITION_REWORK, recommend
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
        "round_grade_limits": [
            {"max_t": 10, "I": 3, "II": 6, "III": 12},
            {"max_t": 999999, "I": 6, "II": 12, "III": 18},
        ],
        "linear_limits": {
            "level2": {"t_factor": 0.333, "min_mm": 4, "max_mm": 20},
            "level3": {"t_factor": 0.667, "min_mm": 6, "max_mm": 30},
            "group": {
                "zone_t_factor": 12,
                "level2": {"t_factor": 0.667, "min_mm": 6, "max_mm": 30},
                "level3": {"t_factor": 1.0, "min_mm": 12, "max_mm": 40},
            },
        },
    },
)


def _porosity(id_: str, length_px: float) -> Detection:
    return Detection(
        id=id_,
        bbox=BBox(x=10, y=10, w=length_px, h=length_px),
        class_id=DefectClass.POROSITY,
        score=0.95,
        uncertainty=0.0,
    )


def _meta(**kw) -> ImageMeta:
    return ImageMeta(
        modality=Modality.GENERIC, pixel_spacing_mm=0.1, base_metal_thickness_mm=12.0, **kw
    )


# ---------------------------------------------------------------------------
# G20 验收级别
# ---------------------------------------------------------------------------
def test_accept_level_stricter_rejects_level_iii():
    """评级 III + 验收 II 级合格 → 不合格返修（默认口径会给出"有条件"）。"""
    rec = recommend(JointLevel.III, [], accept_level="II")
    assert rec.disposition == DISPOSITION_REWORK
    assert any("高于验收合格级别 II" in b for b in rec.basis)


def test_accept_level_satisfied_accepts_level_iii():
    """评级 III + 验收 III 级合格 → 合格。"""
    rec = recommend(JointLevel.III, [], accept_level="III")
    assert rec.disposition == DISPOSITION_ACCEPT


def test_accept_level_loose_accepts_level_ii():
    rec = recommend(JointLevel.II, [], accept_level="III")
    assert rec.disposition == DISPOSITION_ACCEPT


def test_accept_level_does_not_relax_zero_tolerance():
    """零容忍缺陷不因验收等级放宽：级别 I + 裂纹 → 仍不合格。"""
    crack = Detection(
        id="c1",
        bbox=BBox(0, 0, 5, 5),
        class_id=DefectClass.CRACK,
        score=0.99,
        uncertainty=0.0,
    )
    rec = recommend(JointLevel.I, [crack], accept_level="IV")
    assert rec.disposition == DISPOSITION_REWORK


def test_accept_level_does_not_bypass_need_review():
    rec = recommend(JointLevel.I, [], need_review=True, accept_level="IV")
    assert rec.disposition == "recheck"


def test_accept_level_unrecognized_falls_back_to_default():
    """自由文本（"II级合格"）不被识别 → 忽略，走默认口径（III 条件验收）。"""
    rec = recommend(JointLevel.III, [], accept_level="II级合格")
    assert rec.disposition == "conditional"


def test_accept_level_none_keeps_default():
    """不传验收级别保持原行为：II 级直接合格。"""
    rec = recommend(JointLevel.II, [])
    assert rec.disposition == DISPOSITION_ACCEPT


# ---------------------------------------------------------------------------
# G18 管径上下文
# ---------------------------------------------------------------------------
def test_pipe_diameter_recorded_in_basis():
    grader = Nb47013Grader(_AUTHORIZED)
    result = grader.grade([_porosity("p1", 8)], _meta(pipe_outer_diameter_mm=159.0))
    assert any("Φ159" in b and "记录备查" in b for b in result.basis)


def test_pipe_diameter_absent_no_record():
    grader = Nb47013Grader(_AUTHORIZED)
    result = grader.grade([_porosity("p1", 8)], _meta())
    assert not any("记录备查" in b for b in result.basis)
