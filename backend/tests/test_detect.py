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


# ---------------------------------------------------------------------------
# M4：不确定性估计（§5.5，模型无关代理）
# ---------------------------------------------------------------------------
from backend.domain.detect.uncertainty import estimate_uncertainty
from backend.domain.detect.yolo_detector import YoloDetector


def test_uncertainty_near_threshold_is_high() -> None:
    # score 刚好压线 → 最不可信
    assert estimate_uncertainty(0.30, 0.30, 0, 5000) > 0.9


def test_uncertainty_confident_large_is_low() -> None:
    # 高置信 + 大目标 → 低不确定
    assert estimate_uncertainty(0.99, 0.30, 0, 5000) < 0.1


def test_uncertainty_safety_critical_class_higher() -> None:
    # 同置信度下，安全关键类别（裂纹）不确定性基线高于气孔
    crack = estimate_uncertainty(0.99, 0.05, 4, 2000)
    porosity = estimate_uncertainty(0.99, 0.30, 0, 2000)
    assert crack > porosity


def test_uncertainty_clamped() -> None:
    assert 0.0 <= estimate_uncertainty(0.5, 0.3, 0, 1000) <= 1.0
    assert estimate_uncertainty(2.0, 0.3, 0, 1000) <= 1.0  # score 越界被 clip
    assert estimate_uncertainty(-1.0, 0.3, 0, 1000) >= 0.0


def test_yolo_to_detections_uses_estimator() -> None:
    # 大且高置信气孔 → 低不确定（不再恒等于 1-score）
    dets = YoloDetector._to_detections([(10, 10, 40, 40, 0, 0.95)], conf=0.3, class_conf=None)
    assert len(dets) == 1
    assert dets[0].uncertainty < 0.1
    assert dets[0].uncertainty != round(1.0 - 0.95, 4)  # 与旧 1-score 行为不同
