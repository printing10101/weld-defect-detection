"""P1-A 多尺度训练增强 + 推理 TTA 测试。

- build_multiscale_augment：输出保持目标尺寸、bbox 标签同步变换；
- YoloDetector.infer_tta：跨尺度坐标还原 + NMS 去重 + 按置信度排序。
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from backend.domain.detect.yolo_detector import YoloDetector
from backend.domain.dto import BBox, DefectClass, Detection
from backend.training.augment import build_multiscale_augment


class ScaleAwareDetector(YoloDetector):
    """测试桩：模拟单尺度推理，按输入尺寸返回固定检出。

    返回的 bbox 坐标基于**输入图**坐标系（与真实 infer 还原语义一致）。
    0.8x 尺度返回两条**落在 1.0x 检出框内部**的低置信检出（模拟同一缺陷
    在多尺度下的分裂），1.0x/1.25x 返回单条高置信——用于验证跨尺度 NMS 去重。
    """

    def __init__(self) -> None:
        super().__init__()
        self.last_shape = None

    def infer(self, image, conf=0.3, iou=0.5, class_conf=None) -> list[Detection]:
        self.last_shape = image.shape[:2]
        h, w = image.shape[:2]
        cx, cy = w / 2, h / 2
        if abs(w - 512) < 4:  # 0.8 * 640：两条低置信，还原后完全落在 1.0x 框内
            return [
                Detection(
                    id="a",
                    bbox=BBox(cx - 15, cy - 4, 30, 8),
                    class_id=DefectClass.POROSITY,
                    score=0.6,
                    uncertainty=0.3,
                ),
                Detection(
                    id="b",
                    bbox=BBox(cx - 5, cy - 4, 30, 8),
                    class_id=DefectClass.POROSITY,
                    score=0.7,
                    uncertainty=0.3,
                ),
            ]
        # 1.0x 与 1.25x：单条高置信
        return [
            Detection(
                id="c",
                bbox=BBox(cx - 20, cy - 5, 60, 10),
                class_id=DefectClass.POROSITY,
                score=0.9,
                uncertainty=0.2,
            )
        ]


class TestMultiscaleAugment:
    def test_output_size_and_labels(self):
        pytest.importorskip("albumentations", reason="albumentations 未安装（ML 训练 venv 才有）")
        aug = build_multiscale_augment(target_hw=(320, 320), p=1.0)
        rng = np.random.default_rng(0)
        img = rng.integers(50, 200, (300, 420), dtype=np.uint8)
        out = aug(image=img, bboxes=[(0.5, 0.5, 0.2, 0.2)], class_labels=[0])
        assert out["image"].shape[:2] == (320, 320)
        assert len(out["bboxes"]) == 1
        _, _, bw, bh = out["bboxes"][0]
        assert 0.0 <= bw <= 1.0 and 0.0 <= bh <= 1.0


class TestInferTta:
    def _detector(self) -> ScaleAwareDetector:
        return ScaleAwareDetector()

    def test_coord_scaling_and_nms(self):
        det = self._detector()
        img = np.zeros((480, 640), dtype=np.uint8)
        dets = det.infer_tta(img, conf=0.3, iou=0.5, scales=(0.8, 1.0))
        # 跨尺度 NMS 后应合并为 1 条
        assert len(dets) == 1
        d = dets[0]
        # 高置信检出（0.9）保留，且坐标位于原图中心区域
        assert d.score == 0.9
        assert 280 <= d.bbox.x + d.bbox.w / 2 <= 360

    def test_single_scale_passthrough(self):
        det = self._detector()
        img = np.zeros((480, 640), dtype=np.uint8)
        dets = det.infer_tta(img, conf=0.3, iou=0.5, scales=(1.0,))
        assert len(dets) == 1

    def test_empty_image_safe(self):
        det = self._detector()
        assert det.infer_tta(np.zeros((0, 0), dtype=np.uint8), 0.3, 0.5) == []
        assert det.infer_tta(cast(np.ndarray, None), 0.3, 0.5) == []

    def test_sorted_by_score_desc(self):
        det = self._detector()
        img = np.zeros((480, 640), dtype=np.uint8)
        dets = det.infer_tta(img, conf=0.3, iou=0.0, scales=(0.8, 1.0))
        scores = [d.score for d in dets]
        assert scores == sorted(scores, reverse=True)
