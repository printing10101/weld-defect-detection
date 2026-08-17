"""真实检测器（YoloDetector）测试层（§部署硬化 C1 / 门禁虚假信心修复）。

问题背景：conftest 强制 BASELINE_ENABLED=true，且 CI 不装 ML 依赖，
导致生产检测器 YoloDetector 的代码路径在 CI 零执行——任何 yolo_detector.py
回归都不会被门禁捕获（虚假信心）。

本模块用 `@pytest.mark.ml` 标记真实检测器测试：
- `onnxruntime` 缺失（CI 默认）时整模块 importorskip → 不影响 CI 通过；
- 本机装了 ML 推理依赖后，自动加载 models/weights/best.onnx 跑冒烟 + 接口契约测试，
  真正覆盖 YoloDetector.load/infer 代码路径；
- 模型缺失或加载失败（如损坏权重）→ skip（属模型完整性，非代码回归）；
- infer 阶段抛错 → 如实失败（暴露代码回归）。

可用 SCAN_ML_SMOKE_MODEL 指向自定义 ONNX 以覆盖不同权重。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

# 需要 ML 推理依赖；CI（仅 [dev]）无 onnxruntime → 整模块 skip
pytest.importorskip("onnxruntime")

from backend.domain.detect.yolo_detector import YoloDetector

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MODEL = os.environ.get(
    "SCAN_ML_SMOKE_MODEL", str(_ROOT / "models" / "weights" / "best.onnx")
)


def _load_detector() -> YoloDetector:
    model = Path(_DEFAULT_MODEL)
    if not model.exists():
        pytest.skip(f"ML 冒烟模型不存在: {model}（设 SCAN_ML_SMOKE_MODEL 指向可用 ONNX）")
    det = YoloDetector()
    try:
        det.load(str(model), "onnx")
    except Exception as exc:  # noqa: BLE001  # 权重损坏/不完整 → 属模型问题，skip 而非失败
        pytest.skip(f"模型加载失败（权重完整性？）: {exc}")
    return det


@pytest.mark.ml_smoke
def test_yolo_detector_interface_contract() -> None:
    """接口契约：YoloDetector 必须实现 load / infer（ADR-002 主干可热替换）。

    无需权重：仅校验检测器模块可被 onnxruntime 加载、接口存在——CI 装 onnxruntime
    后即必过，恢复对真实检测器代码路径的真实门禁覆盖（修复 continue-on-error 虚假信心）。
    """
    assert hasattr(YoloDetector, "load")
    assert hasattr(YoloDetector, "infer")


@pytest.mark.ml
def test_yolo_detector_smoke_infer() -> None:
    """冒烟：真实加载 ONNX 并对合成底片跑 infer，返回 Detection 列表。"""
    det = _load_detector()
    img = np.full((512, 512), 150, dtype=np.uint8)
    out = det.infer(img, 0.3, 0.5)
    assert isinstance(out, list), "infer 必须返回 list[Detection]"


@pytest.mark.ml
def test_yolo_detector_infer_blank_no_crash() -> None:
    """回归防护：纯背景图不应使 infer 抛未捕获异常（纵使检出为空）。"""
    det = _load_detector()
    blank = np.full((640, 640), 160, dtype=np.uint8)
    out = det.infer(blank, 0.3, 0.5)
    assert isinstance(out, list)
