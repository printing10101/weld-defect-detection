""" 多标准注册表与能力目录（registry + GET /api/v1/standards）。

覆盖：4 标准注册齐全；能力目录状态（NB=enabled / GB=tables_missing /
ASME=method_standard / ISO=method_standard）；未知标准熔断；语义化 grade
熔断消息（方法标准/缺表不输出级别，禁止静默错判）；API 契约。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.domain.dto import BBox, DefectClass, Detection, ImageMeta, Modality
from backend.domain.errors import GradingAmbiguousError
from backend.domain.grade.registry import (
    all_standard_capabilities,
    get_grader,
    standard_capabilities,
    supported_standard_ids,
)


def _det(cls: DefectClass, w: float = 10, h: float = 10) -> Detection:
    return Detection(
        id="d1",
        bbox=BBox(0, 0, w, h),
        class_id=cls,
        score=0.9,
        uncertainty=0.1,
    )


def _ctx(t: float = 20) -> ImageMeta:
    return ImageMeta(
        modality=Modality.GENERIC,
        pixel_spacing_mm=0.1,
        base_metal_thickness_mm=t,
    )


def test_registry_lists_all_four_standards() -> None:
    ids = supported_standard_ids()
    assert set(ids) == {"NB/T47013.2-2015", "GB/T3323-2019", "ASME-V", "ISO17636"}


def test_capabilities_nb_enabled() -> None:
    cap = standard_capabilities("NB/T47013.2-2015")
    assert cap["grades_defects"] is True
    assert cap["levels"] == ["I", "II", "III", "IV"]
    assert cap["status"] == "enabled"  # 表存在且 authorized


def test_capabilities_gb_tables_missing() -> None:
    cap = standard_capabilities("GB/T3323-2019")
    assert cap["grades_defects"] is True
    assert cap["levels"] == ["I", "II", "III", "IV"]
    assert cap["status"] == "tables_missing"  # 评级数值表未转录


def test_capabilities_method_standards() -> None:
    asme = standard_capabilities("ASME-V")
    assert asme["grades_defects"] is False
    assert asme["levels"] is None
    assert asme["status"] == "method_standard"
    iso = standard_capabilities("ISO17636")
    assert iso["grades_defects"] is False
    assert iso["status"] == "method_standard"


def test_capabilities_unknown_fuses() -> None:
    with pytest.raises(GradingAmbiguousError):
        standard_capabilities("ISO-UNKNOWN")


def test_all_capabilities_ordered() -> None:
    caps = all_standard_capabilities()
    assert [c["standard_id"] for c in caps] == supported_standard_ids()
    assert all("status" in c and "note" in c for c in caps)


def test_semantic_fuse_messages() -> None:
    """方法标准/缺表适配器 grade 熔断，消息是标准语义而非"未实现"。"""
    gb = get_grader("GB/T3323-2019", None)
    with pytest.raises(GradingAmbiguousError) as ei_gb:
        gb.grade([_det(DefectClass.POROSITY)], _ctx())
    assert "数值表未转录" in str(ei_gb.value)

    asme = get_grader("ASME-V", None)
    with pytest.raises(GradingAmbiguousError) as ei_as:
        asme.grade([_det(DefectClass.POROSITY)], _ctx())
    assert "方法标准" in str(ei_as.value)
    assert "不定义缺陷验收级别" in str(ei_as.value)

    iso = get_grader("ISO17636", None)
    with pytest.raises(GradingAmbiguousError) as ei_iso:
        iso.grade([_det(DefectClass.POROSITY)], _ctx())
    assert "成像质量等级" in str(ei_iso.value)


def test_standards_api_contract() -> None:
    with TestClient(app) as client:
        resp = client.get("/api/v1/standards")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 4
    first = rows[0]
    for key in (
        "standard_id",
        "name",
        "grades_defects",
        "levels",
        "table_required",
        "status",
        "note",
    ):
        assert key in first
    assert rows[0]["standard_id"] == "NB/T47013.2-2015"
    assert rows[0]["status"] == "enabled"
    assert all(
        r["status"] in ("enabled", "unauthorized", "tables_missing", "method_standard")
        for r in rows
    )
