"""M7 主动学习闭环：GET /api/v1/report/{report_id}/detections 端点测试。

不依赖检测器：直接经 repository 写入 report+image+defects，再以
TestClient 校验明细返回与 404 路径。覆盖原图可读（真实尺寸）与不可读
（缺陷框并集兜底）两种归一化基准。
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app import dependencies as deps
from backend.app.main import app


def _seed(reg, *, report_id: str, image_id: str, path: str | None, defects: list[dict]) -> None:
    image = {
        "id": image_id,
        "path": path or "",
        "source_type": "image",
        "modality": "GENERIC",
        "need_review": True,
    }
    report = {"id": report_id, "image_id": image_id, "pdf_path": f"data/reports/{report_id}.pdf"}
    reg.repository.create_inspection(image, defects, report)


def test_report_detections_returns_stored_defects(tmp_path) -> None:
    reg = deps.get_registry()
    # 真实可读原图：宽 640 高 120
    img_p = tmp_path / "film.png"
    cv2.imwrite(str(img_p), np.zeros((120, 640), dtype=np.uint8))

    defects = [
        {
            "id": "d1",
            "image_id": "IMG1",
            "class_id": 4,  # CRACK
            "bbox_px": [10.0, 20.0, 30.0, 40.0],
            "confidence": 0.82,
            "uncertainty": 0.6,
            "need_review": True,
            "reviewed_by": "alice",
        },
        {
            "id": "d2",
            "image_id": "IMG1",
            "class_id": 0,  # POROSITY
            "bbox_px": [100.0, 50.0, 12.0, 12.0],
            "confidence": 0.55,
            "uncertainty": 0.2,
            "need_review": False,
            "reviewed_by": None,
            "disposition": None,
        },
    ]
    _seed(reg, report_id="R1", image_id="IMG1", path=str(img_p), defects=defects)

    with TestClient(app) as client:
        resp = client.get("/api/v1/report/R1/detections")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["report_id"] == "R1"
    assert body["image_id"] == "IMG1"
    assert body["image_stem"] == "film"
    assert body["image_w"] == 640
    assert body["image_h"] == 120
    assert len(body["defects"]) == 2
    d1 = next(d for d in body["defects"] if d["id"] == "d1")
    assert d1["class_id"] == 4
    assert d1["bbox"] == [10.0, 20.0, 30.0, 40.0]
    assert d1["confidence"] == 0.82
    assert d1["uncertainty"] == 0.6
    assert d1["reviewed"] is True
    assert d1["need_review"] is True
    d2 = next(d for d in body["defects"] if d["id"] == "d2")
    assert d2["reviewed"] is False


def test_report_detections_falls_back_to_bbox_union_when_image_missing() -> None:
    reg = deps.get_registry()
    # path 指向不存在文件 → 退回缺陷框并集：max_x=130, max_y=90 → (131, 91)
    defects = [
        {"id": "x1", "image_id": "IMG2", "class_id": 2, "bbox_px": [10.0, 20.0, 30.0, 40.0]},
        {"id": "x2", "image_id": "IMG2", "class_id": 3, "bbox_px": [100.0, 50.0, 30.0, 40.0]},
    ]
    _seed(reg, report_id="R2", image_id="IMG2", path="data/images/missing.png", defects=defects)

    with TestClient(app) as client:
        resp = client.get("/api/v1/report/R2/detections")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["image_stem"] == "missing"
    assert body["image_w"] == 131
    assert body["image_h"] == 91


def test_report_detections_404_unknown_report() -> None:
    with TestClient(app) as client:
        resp = client.get("/api/v1/report/NO_SUCH/detections")
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "NOT_FOUND"
