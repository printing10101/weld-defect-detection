"""M4a 检测与量化测试（§5）：基线检测器 + 几何换算 + API。"""
from __future__ import annotations

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.domain.detect.blob_detector import BlobConfig, BlobDetector
from backend.domain.dto import BBox, DefectClass, Detection
from backend.domain.quantify import BBoxQuantifier


def _synthetic_defect_image(size: int = 256, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.clip(rng.normal(128.0, 3.0, (size, size)), 0, 255).astype(np.uint8)
    cv2.circle(img, (80, 80), 12, 60, -1)  # 暗斑（气孔/夹渣类）
    cv2.circle(img, (180, 160), 8, 70, -1)
    return img


def test_blob_detector_finds_dark_blobs() -> None:
    det = BlobDetector(BlobConfig(min_area_px=30, dark_only=True))
    det.load("baseline://none")
    res = det.infer(_synthetic_defect_image())
    assert len(res) >= 2  # 两个暗斑
    assert all(d.class_id is DefectClass.POROSITY for d in res)  # 基线占位类别
    assert all(0.3 <= d.score <= 0.7 for d in res)  # 保守置信度


def test_quantify_pixel_spacing() -> None:
    q = BBoxQuantifier()
    det = Detection(
        id="d1",
        bbox=BBox(10, 20, 100, 50),  # 100×50 px
        class_id=DefectClass.POROSITY,
        score=0.5,
        uncertainty=0.5,
    )
    g = q.measure(det, pixel_spacing_mm=0.1)
    assert g.length_mm == 10.0  # 100px × 0.1
    assert g.width_mm == 5.0
    assert g.area_mm2 == 50.0
    assert g.aspect_ratio == 2.0
    assert g.position_x_mm == 1.0


def test_detect_api() -> None:
    img = _synthetic_defect_image()
    ok, buf = cv2.imencode(".png", img)
    assert ok
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/detect",
            files={"image": ("t.png", buf.tobytes(), "image/png")},
            data={"pixel_spacing_mm": "0.1"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["defects"]) >= 2
    first = body["defects"][0]
    for key in ("L_mm", "W_mm", "area_mm2", "perimeter_mm", "aspect_ratio", "confidence"):
        assert key in first
    assert body["annotated_image"]  # 标注图 base64 非空
