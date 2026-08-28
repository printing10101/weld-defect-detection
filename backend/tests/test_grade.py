"""NB/T47013 规则引擎测试。"""

from __future__ import annotations

import pytest

from backend.domain.dto import BBox, DefectClass, Detection, ImageMeta, JointLevel, Modality
from backend.domain.errors import GradingAmbiguousError
from backend.domain.grade.nb47013 import Nb47013Grader
from backend.domain.standards.tables.loader import StandardTables, disclaimer_for

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


def _det_at(class_id: DefectClass, x_px: float, w_px: float, h_px: float) -> Detection:
    return Detection(
        id=f"d{x_px}",
        bbox=BBox(x_px, 0, w_px, h_px),
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


def test_combined_rating_spec_formula() -> None:
    """ 综合评级 = round + linear − 1（≤IV）：圆 II + 条 III → 2+3-1=4 → IV。

    旧实现（同级+1/取最差）在此给 III，设计文档公式给 IV——本测试锁定设计文档。
    """
    grader = Nb47013Grader(_AUTHORIZED)
    # 圆：T=20 下 2 个 4mm 圆（各 6 点）→ 12 点 → II 级（上限 I:6 II:12 III:18）
    round_ds = [_det(DefectClass.POROSITY, 40, 40), _det(DefectClass.POROSITY, 40, 40)]
    # 条：T=20 下 10mm（lim2=6.67, lim3=13.33）→ III 级
    linear_d = _det(DefectClass.SLAG, 100, 10)
    res = grader.grade([*round_ds, linear_d], _ctx(20))
    assert res.joint_level is JointLevel.IV


def test_ignored_small_defects_downgrade_when_many() -> None:
    """：I 级评定区内不计点缺陷>10 个 → 降一级。

    T=20（>5），11 个 0.4mm 圆（≤不计点数阈值 0.5）→ 点数 0 → I 级，
    但不计点缺陷 11>10 → 降为 II。
    """
    grader = Nb47013Grader(_AUTHORIZED)
    small = [_det(DefectClass.POROSITY, 4, 4) for _ in range(11)]  # 0.4mm 方形
    res = grader.grade(small, _ctx(20))
    assert res.joint_level is JointLevel.II
    # 仅 10 个（≤10）不降级
    res10 = grader.grade(small[:10], _ctx(20))
    assert res10.joint_level is JointLevel.I


def test_linear_group_cumulative_downgrades() -> None:
    """ 组内(12T区)累计：单条均 ≤II 限值，但 12T 区内累计超 II 限值 → III。

    T=30：单条 8mm（lim2=10 → 单条 II）；3 条间距 100mm（12T=360 区覆盖全部）
    累计 24mm > glim2=20 → 组级别 III → 综合 III。
    """
    grader = Nb47013Grader(_AUTHORIZED)
    # 3 条 8mm 条形，x 位置 0/1000/2000px（0/100/200mm）
    ds = [
        _det_at(DefectClass.SLAG, 0, 80, 8),  # 8mm
        _det_at(DefectClass.SLAG, 1000, 80, 8),
        _det_at(DefectClass.SLAG, 2000, 80, 8),
    ]
    res = grader.grade(ds, _ctx(30))
    assert res.joint_level is JointLevel.III


def test_linear_collinear_merge() -> None:
    """ 同线间距≤小缺陷长度 → 合并：2 条 5mm 间距 2mm → 合并为 12mm。

    T=30：合并后单条 12mm（lim2=10, lim3=20）→ III；不合并单条 5mm → II。
    """
    grader = Nb47013Grader(_AUTHORIZED)
    a = _det(DefectClass.SLAG, 50, 5)  # 5mm，x 0..50px
    b = _det_at(DefectClass.SLAG, 70, 50, 5)  # 5mm，x 70..120px（间距 20px=2mm）
    merged = grader._merge_collinear([a, b], 0.1)
    assert merged == [(0.0, 12.0)]  # 合并为 12mm 区间
    res = grader.grade([a, b], _ctx(30))
    assert res.joint_level is JointLevel.III


def test_size_near_critical_triggers_review() -> None:
    """ 尺寸临界：长径≈T/2 → need_review=True（即便级别正常也复核）。

    长径 9.5mm（≈T/2=10 的 0.95 倍）：28 点 → IV 级，但因临界仍强制人工复核。
    """
    grader = Nb47013Grader(_AUTHORIZED)
    d = _det(DefectClass.POROSITY, 95, 95)  # 9.5mm 方形
    res = grader.grade([d], _ctx(20))
    assert res.joint_level is JointLevel.IV
    assert res.need_review is True

    # 点数压线：5mm 圆 → 12 点 == II 上限 → 级别 II 但压线临界
    d2 = _det(DefectClass.POROSITY, 50, 50)
    res2 = grader.grade([d2], _ctx(20))
    assert res2.joint_level is JointLevel.II
    assert res2.need_review is True


def test_unsupported_standard_fuses() -> None:
    """ 未知标准（未注册）→ 装配即熔断。"""
    from backend.domain.errors import GradingAmbiguousError
    from backend.domain.grade.registry import get_grader

    with pytest.raises(GradingAmbiguousError):
        get_grader("ISO-UNKNOWN-XXXX")  # 不在注册表


def test_registry_supports_skeletons() -> None:
    """ 骨架标准已注册（可装配），但 grade 熔断。"""
    from backend.domain.errors import GradingAmbiguousError
    from backend.domain.grade.registry import get_grader, supported_standard_ids

    assert "GB/T3323-2019" in supported_standard_ids()
    grader = get_grader("GB/T3323-2019", None)
    with pytest.raises(GradingAmbiguousError):
        grader.grade([_det(DefectClass.POROSITY, 10, 10)], _ctx(10))


def test_table_loader_rejects_bad_group_structure() -> None:
    """ 启动即失败：linear_limits.group 结构不完整必须报错。"""
    from backend.domain.standards.tables import loader as table_loader

    bad = {
        "standard_id": "NB/T47013.2-2015",
        "version": "2015",
        "authorized": True,
        "round_rating_zone_mm": [{"max_t": 999999, "width": 30}],
        "round_points": [{"max_d_mm": 10.0, "points": 28}],
        "round_ignore_size_mm": [{"max_t": 999999, "max_d": 1.5}],
        "round_grade_limits": [{"max_t": 999999, "I": 6, "II": 12, "III": 18}],
        "linear_limits": {
            "level2": {"t_factor": 0.333, "min_mm": 4, "max_mm": 20},
            "level3": {"t_factor": 0.667, "min_mm": 6, "max_mm": 30},
            "group": {"zone_t_factor": 12, "level2": {"t_factor": 0.5}},  # 缺 min_mm/max_mm
        },
    }
    with pytest.raises(ValueError, match="linear_limits.group.level2"):
        table_loader._validate(bad)


def test_deep_hole_direct_iv() -> None:
    """ 深孔（黑度>母材）直判 IV：即便尺寸小、点数低也直判，且强制人工复核。"""
    grader = Nb47013Grader(_AUTHORIZED)
    # 小尺寸圆形缺陷 + deep_hole=True（检测器标注）→ 直判 IV（与尺寸/点数无关）
    hole = Detection(
        id="deep1",
        bbox=BBox(0, 0, 10, 10),  # 1mm 小缺陷（正常会判 I 级）
        class_id=DefectClass.POROSITY,
        score=0.8,
        uncertainty=0.3,
        deep_hole=True,
    )
    res = grader.grade([hole], _ctx(20))
    assert res.joint_level is JointLevel.IV
    assert res.per_defect_grade == (JointLevel.IV,)
    assert res.need_review is True
    assert any("深孔" in b for b in res.basis)

    # 对照：同尺寸非深孔 → 正常评级（I 级）
    normal = Detection(
        id="d2",
        bbox=BBox(0, 0, 10, 10),
        class_id=DefectClass.POROSITY,
        score=0.8,
        uncertainty=0.3,
        deep_hole=False,
    )
    res2 = grader.grade([normal], _ctx(20))
    assert res2.joint_level is JointLevel.I


def test_grade_disclaimer_present_when_no_authorized_copy() -> None:
    """：未持有授权正本（authorized_copy 默认 False）→ GradeResult 带强声明。

    _AUTHORIZED 仅置 authorized=True（数值完整可运算），未置 authorized_copy，
    故免责声明须明确"非标准授权正本 / 不替代责任工程师法定评定"。
    """
    grader = Nb47013Grader(_AUTHORIZED)
    res = grader.grade([_det(DefectClass.POROSITY, 50, 50)], _ctx(20))
    assert res.disclaimer is not None
    assert "非标准授权正本" in res.disclaimer
    assert "不替代" in res.disclaimer or "法定判定" in res.disclaimer


def test_disclaimer_for_authorized_copy_true_is_light() -> None:
    """：authorized_copy=True（已持授权正本并签核）→ 轻声明，不含"非授权正本"措辞。"""
    tables = StandardTables(
        standard_id="NB/T47013.2-2015",
        version="2015",
        authorized=True,
        authorized_copy=True,
        data={},
    )
    text = disclaimer_for(tables)
    assert "已依授权标准正本" in text
    assert "非标准授权正本" not in text


def test_disclaimer_for_no_copy_includes_source_note() -> None:
    """：authorized_copy=False 时免责声明包含 source_note（来源说明）。"""
    tables = StandardTables(
        standard_id="NB/T47013.2-2015",
        version="2015",
        authorized=True,
        authorized_copy=False,
        source_note="数值转录自 2026 年公开解读。",
        data={},
    )
    text = disclaimer_for(tables)
    assert "非标准授权正本" in text
    assert "数值转录自 2026 年公开解读" in text
