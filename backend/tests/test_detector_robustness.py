"""检测器健壮性 & 关键推理路径测试（覆盖本轮加固项）。

- 空/退化输入守卫：避免 letterbox 缩放除零与张量形状错误；
- dto.ImageMeta.density_array 注解可被正确求值（回归 F821：np 未导入）；
- 真实 ONNX 模型加载 + 预热 + 推理（关键推理路径，ML 依赖齐备时执行）。
"""

from __future__ import annotations

import importlib.util
import typing
from pathlib import Path

import numpy as np
import pytest

from backend.domain.detect.yolo_detector import YoloDetector
from backend.domain.dto import ImageMeta, Modality

_HAS_ONNX = importlib.util.find_spec("onnxruntime") is not None
_MODEL = Path(__file__).resolve().parents[1] / "models" / "weights" / "best.onnx"


def test_empty_image_guard_does_not_crash() -> None:
    """0 尺寸输入必须安全返回空结果，而非在缩放/推理路径崩溃。"""
    det = YoloDetector()
    assert det.infer(np.empty((0, 0), dtype=np.uint8), conf=0.3, iou=0.5) == []
    assert det.infer(np.empty((64, 0), dtype=np.uint8), conf=0.3, iou=0.5) == []
    assert det.infer(np.empty((0, 64), dtype=np.uint8), conf=0.3, iou=0.5) == []


def test_imagemeta_density_array_annotation_resolves() -> None:
    """dto.py 引入 numpy 后 density_array: np.ndarray 注解必须可求值。

    回归自 F821：np 未导入时 get_type_hints(ImageMeta) 会抛 NameError，
    任何运行时类型自省（pydantic / 序列化 / 文档生成）都会崩溃。
    """
    hints = typing.get_type_hints(ImageMeta)
    ann = hints["density_array"]  # np.ndarray | None → Union[ndarray, None]
    args = typing.get_args(ann)
    assert np.ndarray in args
    assert type(None) in args
    meta = ImageMeta(modality=Modality.GENERIC, density_array=np.zeros((4, 4), np.uint16))
    assert meta.density_array is not None
    assert meta.density_array.shape == (4, 4)


@pytest.mark.skipif(
    not _HAS_ONNX or not _MODEL.exists(),
    reason="onnxruntime 或模型权重不可用",
)
def test_yolo_onnx_load_and_infer() -> None:
    """真实 ONNX 模型加载（含预热前向）+ 推理关键路径。"""
    det = YoloDetector()
    det.load(str(_MODEL), backend="onnx")
    img = np.random.default_rng(0).integers(0, 255, (640, 640), dtype=np.uint8)
    dets = det.infer(img, conf=0.3, iou=0.5)
    assert isinstance(dets, list)
    for d in dets:
        assert d.bbox.w >= 0 and d.bbox.h >= 0
        assert 0.0 <= d.score <= 1.0


def test_thr_for_helper_falls_back_to_global_conf() -> None:
    """_thr_for：未指定 class_conf 时回落全局 conf；指定时取专属阈值。"""
    # 全局 conf=0.3，未指定映射 → 0.3
    assert YoloDetector._thr_for(3, 0.3, None) == 0.3
    # 指定映射：类3取0.08，类0取0.30，其余回落0.3
    cc = {0: 0.30, 3: 0.08}
    assert YoloDetector._thr_for(3, 0.3, cc) == 0.08
    assert YoloDetector._thr_for(0, 0.3, cc) == 0.30
    assert YoloDetector._thr_for(5, 0.3, cc) == 0.3


@pytest.mark.skipif(
    not _HAS_ONNX or not _MODEL.exists(),
    reason="onnxruntime 或模型权重不可用",
)
def test_class_conf_releases_rare_detections() -> None:
    """逐类阈值应让稀有类（低阈值）比全局高阈值放出更多候选。

    在安全关键缺陷（裂纹/未熔合）设更低阈值以提升召回；气孔设更高阈值抑误检。
    该测试要求 best.onnx 为已做稀有类平衡的模型，否则稀有类候选本就为空。
    """
    real_imgs = Path(__file__).resolve().parents[2] / "data" / "real_label" / "images"
    if not real_imgs.exists():
        pytest.skip("真实底片目录不可用")
    imgs = sorted(real_imgs.glob("*.jpg"))[:60]
    if not imgs:
        pytest.skip("无真实底片")

    det = YoloDetector()
    det.load(str(_MODEL), backend="onnx")
    class_conf = {0: 0.30, 1: 0.12, 2: 0.12, 3: 0.08, 4: 0.05, 5: 0.18}

    rare_uniform = 0
    rare_class = 0
    for p in imgs:
        import cv2

        arr = np.fromfile(str(p), dtype=np.uint8)
        gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        for d in det.infer(gray, conf=0.3, iou=0.5):
            if d.class_id.value in (1, 2, 3, 4, 5):
                rare_uniform += 1
        for d in det.infer(gray, conf=0.3, iou=0.5, class_conf=class_conf):
            if d.class_id.value in (1, 2, 3, 4, 5):
                rare_class += 1
    # 逐类阈值下稀有类候选不应少于统一阈值（低阈值放行更多）
    assert rare_class >= rare_uniform
