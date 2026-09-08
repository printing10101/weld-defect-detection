"""infer_tta 集成不确定性端到端测试（检测桩，无需权重）。

验证：跨视角检出稳定 → uncertainty 维持单视角水平；
仅单视角检出 → 集成信号抬升 uncertainty（更早触发人工复核）。
"""

from __future__ import annotations

import numpy as np

from backend.domain.detect.yolo_detector import YoloDetector
from backend.domain.dto import BBox, DefectClass, Detection


class StubDetector(YoloDetector):
    """按输入宽度区分视角：稳定缺陷全视角检出，游移缺陷仅 0.8x 视角检出。

    物理真实语义：缺陷在原图坐标 (300, 235)/(25, 36) 处、尺寸不随缩放变——
    视角 s 下检出框位于 s×原图坐标、尺寸 s×原尺寸，还原后各视角重合（与
    真实检测器一致）。两类缺陷位置远隔（IoU≈0），跨尺度 NMS 后各自独立。
    """

    def infer(self, image, conf=0.3, iou=0.5, class_conf=None) -> list[Detection]:
        w = image.shape[1]
        s = w / 640.0
        stable = Detection(
            id="stable",
            bbox=BBox(300 * s, 235 * s, 40 * s, 10 * s),
            class_id=DefectClass.POROSITY,
            score=0.9,
            uncertainty=0.1,
        )
        if abs(s - 0.8) < 0.05:  # 仅 0.8x 视角额外检出"游移缺陷"
            unstable = Detection(
                id="unstable",
                bbox=BBox(25 * s, 36 * s, 30 * s, 8 * s),
                class_id=DefectClass.SLAG,
                score=0.7,
                uncertainty=0.1,
            )
            return [stable, unstable]
        return [stable]


def _run() -> dict[str, float]:
    det = StubDetector()
    img = np.zeros((480, 640), dtype=np.uint8)
    out = det.infer_tta(img, conf=0.3, iou=0.5, scales=(0.8, 1.0, 1.25))
    return {d.id.split("@")[0][:8] or d.id: d.uncertainty for d in out}


class TestTtaEnsembleUncertainty:
    def test_stable_defect_keeps_low_uncertainty(self):
        det = StubDetector()
        img = np.zeros((480, 640), dtype=np.uint8)
        out = det.infer_tta(img, conf=0.3, iou=0.5, scales=(0.8, 1.0, 1.25))
        stable = next(d for d in out if d.bbox.x > 200)
        # 三视角全检出且得分一致 → u_aug=0 → 维持单视角启发式值
        assert stable.uncertainty <= 0.1 + 1e-6

    def test_single_view_defect_uncertainty_boosted(self):
        det = StubDetector()
        img = np.zeros((480, 640), dtype=np.uint8)
        out = det.infer_tta(img, conf=0.3, iou=0.5, scales=(0.8, 1.0, 1.25))
        unstable = next(d for d in out if d.bbox.x < 200)
        # 3 视角仅 1 视角检出 → vote_frac=1/3 → u_aug≈0.667 → max(0.1, 0.667)
        assert unstable.uncertainty >= 0.6

    def test_both_defects_survive_merge(self):
        det = StubDetector()
        img = np.zeros((480, 640), dtype=np.uint8)
        out = det.infer_tta(img, conf=0.3, iou=0.5, scales=(0.8, 1.0, 1.25))
        assert len(out) == 2

    def test_unstable_not_lower_than_stable(self):
        det = StubDetector()
        img = np.zeros((480, 640), dtype=np.uint8)
        out = det.infer_tta(img, conf=0.3, iou=0.5, scales=(0.8, 1.0, 1.25))
        us = {("s" if d.bbox.x > 200 else "u"): d.uncertainty for d in out}
        assert us["u"] > us["s"]
