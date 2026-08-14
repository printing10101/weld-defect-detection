"""发运版 NB/T47013 数值表端到端自洽测试（§6 / M5 启用验证）。

直接加载**发运版** nb47013.yaml（非测试内联 fixture），断言其在代表性
案例下产出符合标准预期的级别，锁定发运数据自洽；并验证：
- authorized=true 时不再熔断；
- 引擎在表字段缺失/超限时仍强制熔断（防腐护栏不变量）。

注意：本测试锁定"发运表 + 引擎"组合的行为；表数值源自公开标准解读，
正式使用前须以授权原文复核（见 nb47013.yaml 顶部 PROVENANCE 注释）。
"""

from __future__ import annotations

import pytest

from backend.domain.dto import BBox, DefectClass, Detection, ImageMeta, JointLevel, Modality
from backend.domain.errors import GradingAmbiguousError
from backend.domain.grade.nb47013 import Nb47013Grader
from backend.domain.standards.tables.loader import load_standard_tables


def _tables() -> object:
    t = load_standard_tables("NB/T47013.2-2015", filename="nb47013.yaml")
    assert t.authorized is True, "发运表应已授权（authorized=true）"
    return t


def _ctx(t: float, spacing: float = 0.1) -> ImageMeta:
    return ImageMeta(
        modality=Modality.GENERIC,
        pixel_spacing_mm=spacing,
        base_metal_thickness_mm=t,
    )


def _det(class_id: DefectClass, w_px: float, h_px: float, **kw: object) -> Detection:
    return Detection(
        id="d",
        bbox=BBox(0, 0, w_px, h_px),
        class_id=class_id,
        score=0.8,
        uncertainty=0.3,
        **kw,
    )


def test_e2e_crack_is_zero_tolerance_iv() -> None:
    g = Nb47013Grader(_tables())
    res = g.grade([_det(DefectClass.CRACK, 5, 50)], _ctx(20))
    assert res.joint_level is JointLevel.IV
    assert res.need_review is True


def test_e2e_deep_hole_direct_iv() -> None:
    g = Nb47013Grader(_tables())
    hole = _det(DefectClass.POROSITY, 10, 10, deep_hole=True)  # 1mm 小孔
    res = g.grade([hole], _ctx(20))
    assert res.joint_level is JointLevel.IV
    assert res.need_review is True
    assert any("深孔" in b for b in res.basis)


def test_e2e_small_round_porosity_is_i() -> None:
    g = Nb47013Grader(_tables())
    # 2mm 方形气孔（长宽比 1 → 圆形）→ 2 点 → T=20 评定区 III 上限 9 → I 级
    res = g.grade([_det(DefectClass.POROSITY, 20, 20)], _ctx(20))
    assert res.joint_level is JointLevel.I


def test_e2e_no_defect_is_i() -> None:
    g = Nb47013Grader(_tables())
    res = g.grade([], _ctx(20))
    assert res.joint_level is JointLevel.I


def test_e2e_large_round_porosity_is_iv() -> None:
    g = Nb47013Grader(_tables())
    # 10mm 方形气孔（圆形）→ 28 点 → T=20 评定区 III 上限 9 → 远超 → IV
    res = g.grade([_det(DefectClass.POROSITY, 100, 100)], _ctx(20))
    assert res.joint_level is JointLevel.IV


def test_e2e_fuse_when_thickness_invalid() -> None:
    """防腐护栏不变量：即便 authorized=true，缺有效母材厚度仍熔断（不臆造级别）。"""
    g = Nb47013Grader(_tables())
    with pytest.raises(GradingAmbiguousError):
        g.grade([_det(DefectClass.POROSITY, 20, 20)], _ctx(0.0))  # T=0 无效
