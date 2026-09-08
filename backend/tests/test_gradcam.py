"""Grad-CAM 与热力图回退测试。

- attention_heatmap：无 cam_model / cam_model 失效时回退 Sobel 显著性（不抛错）；
- grad_cam_map：非 torch 对象返回 None；
- ml 标记：真实 YOLO 权重 + torch 后端的真 Grad-CAM（CI 默认 skip，本机 ml venv 运行）。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.domain.detect.gradcam import grad_cam_map, letterbox_rgb
from backend.domain.dto import BBox, DefectClass, Detection
from backend.domain.explain import attention_heatmap

_ROOT = Path(__file__).resolve().parents[2]
_PT = _ROOT / "data" / "runs" / "planB_synth640" / "weights" / "best.pt"


def _det(x: float = 100.0, y: float = 80.0, w: float = 40.0, h: float = 20.0) -> Detection:
    return Detection(
        id="d0",
        bbox=BBox(x=x, y=y, w=w, h=h),
        class_id=DefectClass.POROSITY,
        score=0.8,
        uncertainty=0.2,
    )


def _img(h: int = 240, w: int = 320) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(80, 180, (h, w), dtype=np.uint8)


class TestHeatmapFallback:
    def test_no_cam_model_returns_overlay(self):
        out = attention_heatmap(_img(), _det())
        assert out.shape == (240, 320, 3)
        assert out.dtype == np.uint8

    def test_broken_cam_model_falls_back_to_sobel(self):
        # cam_model 非 torch 模型 → grad_cam_map 内部失败 → 必须回退而非抛错
        out = attention_heatmap(_img(), _det(), cam_model=object())
        assert out.shape == (240, 320, 3)
        assert out.dtype == np.uint8

    def test_degenerate_bbox_returns_base(self):
        out = attention_heatmap(_img(), _det(w=1.0, h=1.0))
        assert out.shape == (240, 320, 3)

    def test_cam_model_positional_backward_compatible(self):
        # 旧调用签名（仅 image/detection + 关键字默认）不变
        out = attention_heatmap(_img(), _det(), sigma=2.0, alpha=0.4)
        assert out.shape == (240, 320, 3)


class TestGradCamUnit:
    def test_plain_object_returns_none(self):
        assert grad_cam_map(object(), _img(), (10, 10, 40, 20)) is None

    def test_letterbox_geometry(self):
        padded, r, top, left, nh, nw = letterbox_rgb(_img(240, 320), 640)
        assert padded.shape == (640, 640, 3)
        assert r == pytest.approx(2.0)
        assert nh == 480 and nw == 640
        assert top == (640 - 480) // 2 and left == 0


@pytest.mark.ml
class TestGradCamReal:
    def test_real_grad_cam_localizes_detection(self):
        pytest.importorskip("torch")
        ultralytics = pytest.importorskip("ultralytics")
        if not _PT.exists():
            pytest.skip(f"无 torch 权重：{_PT}")

        model = ultralytics.YOLO(str(_PT))
        img_path = sorted((_ROOT / "data" / "training" / "test" / "images").glob("*.png"))
        if not img_path:
            pytest.skip("无合成测试图")
        img = cv2.imread(str(img_path[0]), cv2.IMREAD_GRAYSCALE)
        assert img is not None
        res = model.predict(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), conf=0.25, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            pytest.skip("该图无检出，无法验证 CAM 定位")
        x1, y1, x2, y2 = res.boxes.xyxy[0].tolist()
        cls = int(res.boxes.cls[0].item())
        cam = grad_cam_map(model, img, (x1, y1, x2 - x1, y2 - y1), class_id=cls)
        assert cam is not None, "真实权重下 Grad-CAM 不应失败"
        assert cam.shape == img.shape
        assert cam.dtype == np.float32
        assert 0.0 <= float(cam.min()) and float(cam.max()) <= 1.0
        assert float(cam.max()) > 0.5
        # CAM 峰值应落在检出框附近（梯度级分辨率，允许数倍框尺寸的偏差）
        cy, cx = np.unravel_index(int(cam.argmax()), cam.shape)
        bcx, bcy = (x1 + x2) / 2, (y1 + y2) / 2
        assert ((cx - bcx) ** 2 + (cy - bcy) ** 2) ** 0.5 < max(img.shape) * 0.25
