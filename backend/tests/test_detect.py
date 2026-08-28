"""检测与量化测试：基线检测器 + 几何换算 + API。"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.domain.detect.blob_detector import BlobConfig, BlobDetector
from backend.domain.dto import BBox, DefectClass, Detection
from backend.domain.errors import ModelUnavailableError
from backend.domain.quantify import (
    BBoxQuantifier,
    MaskQuantifier,
    get_quantifier,
    quantifier_capabilities,
    supported_quantifier_kinds,
)


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


def test_quantifier_registry_lists_and_resolves() -> None:
    """量化器注册表：种类清单、装配解析、未知种类复用  MODEL_UNAVAILABLE。"""
    assert set(supported_quantifier_kinds()) == {"bbox", "mask"}
    assert isinstance(get_quantifier("bbox"), BBoxQuantifier)
    assert isinstance(get_quantifier("mask"), MaskQuantifier)
    # 默认种类 = bbox（/report 历史行为，经注册表装配）
    assert isinstance(get_quantifier(), BBoxQuantifier)
    with pytest.raises(ModelUnavailableError):
        get_quantifier("nope")
    with pytest.raises(ModelUnavailableError):
        quantifier_capabilities("nope")
    assert quantifier_capabilities("mask")["needs_image"] is True
    assert quantifier_capabilities("bbox")["needs_image"] is False


def test_quantifier_unified_quantify_call() -> None:
    """统一量化入口：两量化器同签名 quantify(...)，调用点一致、可互换。"""
    det = Detection(
        id="d1",
        bbox=BBox(10, 20, 100, 50),
        class_id=DefectClass.POROSITY,
        score=0.5,
        uncertainty=0.5,
    )
    # bbox：忽略 image/cfg，等价 measure
    g_bbox = get_quantifier("bbox").quantify(det, 0.1, image=None, cfg=None)
    assert g_bbox.length_mm == 10.0
    # mask 无图：回退包围盒近似
    g_mask_fallback = get_quantifier("mask").quantify(det, 0.1)
    assert g_mask_fallback.length_mm == 10.0
    # mask 有图：走掩膜精修（紧贴暗斑的框 → 掩膜面积 < 包围盒面积，证明精修路径生效）
    img = _synthetic_defect_image()
    det_blob = Detection(
        id="b1",
        bbox=BBox(68, 68, 24, 24),  # 紧贴 (80,80) r=12 暗斑
        class_id=DefectClass.POROSITY,
        score=0.6,
        uncertainty=0.2,
    )
    g_mask_img = get_quantifier("mask").quantify(det_blob, 0.1, image=img, cfg=None)
    assert g_mask_img.length_mm > 0
    assert g_mask_img.area_mm2 > 0
    assert g_mask_img.area_mm2 < g_bbox.area_mm2  # 圆形掩膜面积 < 外接框面积


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


def test_detect_api_uncalibrated_no_pseudo_mm() -> None:
    """未提供像素标定（无请求参数、合成 PNG 无 DICOM 元数据）→ 不输出伪物理尺寸。

    与 /report 链路（grader 熔断）保持单一语义：calibrated=False 且物理字段为 None，
    aspect_ratio 等无量纲量仍有效。
    """
    img = _synthetic_defect_image()
    ok, buf = cv2.imencode(".png", img)
    assert ok
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/detect",
            files={"image": ("t.png", buf.tobytes(), "image/png")},
            # 故意不传 pixel_spacing_mm
        )
    assert resp.status_code == 200
    defects = resp.json()["defects"]
    assert len(defects) >= 2
    for d in defects:
        assert d["calibrated"] is False
        assert d["L_mm"] is None
        assert d["W_mm"] is None
        assert d["area_mm2"] is None
        assert d["perimeter_mm"] is None
        assert d["position"] is None
        # 无量纲形状量仍有效
        assert isinstance(d["aspect_ratio"], (int, float))
        assert d["bbox"]  # 像素框始终有效


# ---------------------------------------------------------------------------
# 不确定性估计
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
