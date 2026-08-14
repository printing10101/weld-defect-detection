"""仓储测试（§7.1/§7.3）：CRUD / 多条件检索 / 统计。"""

from __future__ import annotations

from datetime import UTC, datetime

from backend.infra.repository import InspectionRepository


def _naive_utc(dt: datetime) -> datetime:
    """转 naive UTC（与 db.utcnow 语义一致，SQLite 存储友好）。"""
    return dt.replace(tzinfo=None) if dt.tzinfo else dt


def _image_row(
    image_id: str = "img-1",
    joint_level: str = "II",
    workpiece_no: str | None = "WP-001",
    created_at: datetime | None = None,
) -> dict:
    return {
        "id": image_id,
        "path": f"data/images/{image_id}.png",
        "source_type": "image",
        "modality": "GENERIC",
        "workpiece_no": workpiece_no,
        "weld_no": "W-1",
        "pixel_spacing_mm": 0.1,
        "base_metal_thickness_mm": 20.0,
        "iqi_pass": True,
        "iqi_detail": {"type": "wire", "achieved": "12", "required": "10"},
        "density": 3.1,
        "density_ok": True,
        "evaluable": True,
        "joint_level": joint_level,
        "need_review": False,
        "standard_id": "NB/T47013.2-2015",
        "standard_version": "2015",
        "created_at": _naive_utc(created_at or datetime(2026, 8, 4, 9, 0, 0, tzinfo=UTC)),
    }


def _defect_rows(image_id: str, class_id: int = 0, n: int = 1) -> list[dict]:
    return [
        {
            "id": f"d-{image_id}-{i}",
            "image_id": image_id,
            "class_id": class_id,
            "bbox_px": [10, 10, 20, 20],
            "shape": "round",
            "length_mm": 2.0,
            "width_mm": 2.0,
            "area_mm2": 3.14,
            "perimeter_mm": 6.28,
            "position_x": 1.0,
            "position_y": 1.0,
            "confidence": 0.8,
            "uncertainty": 0.2,
            "joint_level": "II",
            "need_review": False,
            "standard_id": "NB/T47013.2-2015",
            "standard_version": "2015",
        }
        for i in range(n)
    ]


def _report_row(image_id: str, report_id: str = "rep-1") -> dict:
    return {
        "id": report_id,
        "image_id": image_id,
        "joint_level": "II",
        "pdf_path": f"data/reports/{image_id}.pdf",
        "standard_ref": "NB/T47013.2-2015 2015",
        "signer": "tester",
        "basis": ["NB/T47013.2-2015 表 5：圆形缺陷点数上限 II 级 ≤ 12"],
    }


def test_create_and_get(tmp_path) -> None:
    repo = InspectionRepository(str(tmp_path / "t.db"))
    repo.create_inspection(_image_row(), _defect_rows("img-1"), _report_row("img-1"))
    got = repo.get_image("img-1")
    assert got is not None
    assert got["image_id"] == "img-1"
    assert got["joint_level"] == "II"
    assert got["density"] == 3.1
    assert len(got["defects"]) == 1
    assert got["defects"][0]["class_name"] == "POROSITY"
    assert got["defects"][0]["bbox_px"] == [10, 10, 20, 20]
    assert got["report"]["report_id"] == "rep-1"
    assert got["report"]["pdf_path"].endswith(".pdf")


def test_get_missing_returns_none(tmp_path) -> None:
    repo = InspectionRepository(str(tmp_path / "t.db"))
    assert repo.get_image("nope") is None


def test_list_filter_by_level(tmp_path) -> None:
    repo = InspectionRepository(str(tmp_path / "t.db"))
    repo.create_inspection(_image_row("a", joint_level="I"), _defect_rows("a"))
    repo.create_inspection(_image_row("b", joint_level="II"), _defect_rows("b"))
    repo.create_inspection(_image_row("c", joint_level="IV"), _defect_rows("c"))
    items, total = repo.list_records(level="II")
    assert total == 1
    assert [i["image_id"] for i in items] == ["b"]


def test_list_filter_by_class(tmp_path) -> None:
    repo = InspectionRepository(str(tmp_path / "t.db"))
    repo.create_inspection(_image_row("a"), _defect_rows("a", class_id=0))  # POROSITY
    repo.create_inspection(_image_row("b"), _defect_rows("b", class_id=4))  # CRACK
    items, total = repo.list_records(class_id=4)
    assert total == 1
    assert items[0]["image_id"] == "b"


def test_list_filter_by_workpiece(tmp_path) -> None:
    repo = InspectionRepository(str(tmp_path / "t.db"))
    repo.create_inspection(_image_row("a", workpiece_no="WP-001"), _defect_rows("a"))
    repo.create_inspection(_image_row("b", workpiece_no="WP-002"), _defect_rows("b"))
    items, total = repo.list_records(workpiece="002")
    assert total == 1
    assert items[0]["image_id"] == "b"


def test_list_pagination_and_defect_count(tmp_path) -> None:
    repo = InspectionRepository(str(tmp_path / "t.db"))
    for i in range(3):
        repo.create_inspection(_image_row(f"img{i}"), _defect_rows(f"img{i}", n=2))
    items, total = repo.list_records(page=2, size=2)
    assert total == 3
    assert len(items) == 1
    assert items[0]["defect_count"] == 2


def test_stats(tmp_path) -> None:
    repo = InspectionRepository(str(tmp_path / "t.db"))
    repo.create_inspection(_image_row("a", joint_level="I"), _defect_rows("a", class_id=0, n=2))
    repo.create_inspection(_image_row("b", joint_level="II"), _defect_rows("b", class_id=4))
    st = repo.stats()
    assert st["total"] == 2
    assert st["by_level"] == {"I": 1, "II": 1}
    assert st["by_class"] == {"POROSITY": 2, "CRACK": 1}


def test_update_report_path(tmp_path) -> None:
    repo = InspectionRepository(str(tmp_path / "t.db"))
    repo.create_inspection(_image_row(), _defect_rows("img-1"), _report_row("img-1", "rep-9"))
    repo.update_report("rep-9", pdf_path="data/reports/img-1_v2.pdf")
    rep = repo.get_report("rep-9")
    assert rep is not None
    assert rep["pdf_path"] == "data/reports/img-1_v2.pdf"
