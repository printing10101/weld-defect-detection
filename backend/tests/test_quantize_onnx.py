"""ONNX INT8 量化验证测试（backend/training/quantize_onnx）。

onnx 包不在 backend/.venv（ML 训练 venv 才有）→ 量化路径测试 importorskip；
纯 numpy/cv2 的后处理/匹配逻辑在任意 venv 均可测。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.training.quantize_onnx import (
    _CLASS_CONF,
    _match,
    _nms,
    letterbox,
    postprocess,
    read_gray,
)


class TestLetterbox:
    def test_output_shape_and_scale(self):
        img = np.zeros((240, 320), dtype=np.uint8)
        blob, r, top, left = letterbox(img)
        assert blob.shape == (1, 3, 640, 640)
        assert r == pytest.approx(2.0)  # 320→640
        assert top == 80 and left == 0  # 高 240*2=480，上下各 pad 80；宽正好 640

    def test_gray_to_rgb_normalized(self):
        img = np.full((100, 100), 200, dtype=np.uint8)
        blob, *_ = letterbox(img)
        assert blob.max() <= 1.0 and blob.min() >= 0.0


class TestPostprocess:
    def _make_output(self) -> np.ndarray:
        """构造 [1, 10, 4] 输出：1 个高置信气孔 + 1 个低置信气孔。"""
        out = np.zeros((1, 10, 4), dtype=np.float32)
        # 锚框1：中心(320,320) 尺寸(60,60)，class0 score 0.8
        out[0, 0, :4] = [320.0, 320.0, 60.0, 60.0]
        out[0, 0, 4] = 0.8  # class0
        # 锚框2：低置信 0.2（低于全局 0.3）
        out[0, 1, :4] = [100.0, 100.0, 40.0, 40.0]
        out[0, 1, 5] = 0.2
        return out

    def test_channel_first_transposed_and_threshold(self):
        raw = np.zeros((1, 4 + 6, 2), dtype=np.float32)
        raw[0, 4, 0] = 0.8  # 锚1 class0
        raw[0, 0, 0] = 320.0
        raw[0, 1, 0] = 320.0
        raw[0, 2, 0] = 60.0
        raw[0, 3, 0] = 60.0
        raw[0, 4, 1] = 0.2  # 锚2 低置信
        raw[0, 0, 1] = 100.0
        raw[0, 1, 1] = 100.0
        raw[0, 2, 1] = 40.0
        raw[0, 3, 1] = 40.0
        dets = postprocess(raw, r=1.0, top=0, left=0, conf=0.3, iou=0.5)
        assert len(dets) == 1
        x, y, w, h, cls, score = dets[0]
        assert cls == 0 and score == pytest.approx(0.8)
        assert (x, y, w, h) == pytest.approx((290.0, 290.0, 60.0, 60.0))

    def test_logits_sigmoid_heuristic(self):
        raw = np.zeros((1, 10, 1), dtype=np.float32)
        raw[0, 0, 0] = 50.0  # 中心
        raw[0, 1, 0] = 50.0
        raw[0, 2, 0] = 20.0
        raw[0, 3, 0] = 20.0
        raw[0, 4, 0] = 2.0  # class0 logit（正）
        raw[0, 5, 0] = -1.0  # class1 logit（负 → min<0 触发 sigmoid）
        dets = postprocess(raw, r=1.0, top=0, left=0, conf=0.1, iou=0.5)
        assert len(dets) == 1
        assert 0.5 < dets[0][5] < 1.0  # sigmoid(2.0)=0.88

    def test_per_class_threshold(self):
        raw = np.zeros((1, 10, 2), dtype=np.float32)
        # 裂纹 class4 低置信 0.06：逐类阈值 0.05 → 放行；全局 0.3 → 拦截
        raw[0, 0, 0] = 320.0
        raw[0, 1, 0] = 320.0
        raw[0, 2, 0] = 10.0
        raw[0, 3, 0] = 10.0
        raw[0, 4 + 4, 0] = 0.06
        # 气孔 class0 同置信 0.06：阈值 0.30 → 拦截
        raw[0, 0, 1] = 100.0
        raw[0, 1, 1] = 100.0
        raw[0, 2, 1] = 10.0
        raw[0, 3, 1] = 10.0
        raw[0, 4, 1] = 0.06
        dets = postprocess(raw, r=1.0, top=0, left=0, conf=0.3, iou=0.5)
        assert len(dets) == 1
        assert dets[0][4] == 4  # 仅裂纹放行

    def test_class_conf_constant(self):
        assert _CLASS_CONF[0] >= 0.2  # 气孔高阈值抑误检
        assert _CLASS_CONF[4] <= 0.1  # 裂纹低阈值优先召回


class TestMatchAndNms:
    def test_match_greedy(self):
        d32 = [(10, 10, 20, 20, 0, 0.9)]
        d8 = [(12, 12, 18, 18, 0, 0.85)]
        matched, deltas, ious = _match(d32, d8, iou_match=0.3)
        assert matched == 1
        assert deltas[0] == pytest.approx(0.05)
        assert ious[0] > 0.3

    def test_nms_dedup(self):
        boxes = [
            (0, 0, 10, 10, 0.9, 0),
            (1, 1, 10, 10, 0.8, 0),
            (50, 50, 10, 10, 0.7, 0),
        ]
        keep = _nms([b[:4] + (b[4], b[5]) for b in boxes], 0.5)
        assert len(keep) == 2  # 前两个重叠被抑制


class TestReadGray:
    def test_read_gray_unicode_path(self, tmp_path: Path):
        img = np.full((40, 40), 128, dtype=np.uint8)
        p = tmp_path / "中文底片.png"
        import cv2

        _, buf = cv2.imencode(".png", img)
        p.write_bytes(buf.tobytes())
        out = read_gray(p)
        assert out is not None and out.shape == (40, 40)
