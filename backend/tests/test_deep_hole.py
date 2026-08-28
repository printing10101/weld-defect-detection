"""P1-4：deep_hole 启发式推导。

基于 density_array：缺陷内部光学黑度显著高于母材（> DEEP_HOLE_DENSITY_RATIO 倍）
→ 标 deep_hole=True。Detection 为冻结 dataclass，须返回新实例。
"""

from __future__ import annotations

import numpy as np

from backend.app.pipelines import _derive_deep_hole
from backend.domain.density import estimate_density
from backend.domain.dto import BBox, DefectClass, Detection, ImageMeta, Modality


def _meta_with(arr, bit_depth=16) -> ImageMeta:
    return ImageMeta(modality=Modality.GENERIC, density_array=arr, bit_depth=bit_depth)


# 母材(值=40000@16bit)的真实光学黑度约 0.21；缺陷内部(值=2000)约 1.52。
# 用真实值而非臆造的 2.5，才能验证 interior > base*1.2 的判定。
_BASE_DENSITY = estimate_density(np.full((4, 4), 40000, dtype=np.uint16), 16)


def _det() -> Detection:
    # 居中 20x20 缺陷框
    return Detection(
        id="d0",
        bbox=BBox(20, 20, 20, 20),
        class_id=DefectClass.POROSITY,
        score=0.9,
        uncertainty=0.1,
    )


def test_deep_hole_true_when_interior_darker_than_base() -> None:
    # 16bit：母材亮(值大→低黑度)，缺陷区很暗(值小→高黑度)
    arr = np.full((64, 64), 40000, dtype=np.uint16)
    arr[20:40, 20:40] = 2000  # 缺陷：显著更暗
    out = _derive_deep_hole([_det()], _meta_with(arr), base_density=_BASE_DENSITY, bit_depth=16)
    assert out[0].deep_hole is True


def test_deep_hole_false_when_interior_similar_to_base() -> None:
    arr = np.full((64, 64), 40000, dtype=np.uint16)
    arr[20:40, 20:40] = 38000  # 缺陷区与母材接近 → 不超阈值
    out = _derive_deep_hole([_det()], _meta_with(arr), base_density=_BASE_DENSITY, bit_depth=16)
    assert out[0].deep_hole is False


def test_deep_hole_false_when_no_density_array() -> None:
    # 8bit 底片无 density_array
    out = _derive_deep_hole([_det()], _meta_with(None), base_density=2.5, bit_depth=8)
    assert out[0].deep_hole is False


def test_deep_hole_false_when_base_density_invalid() -> None:
    arr = np.full((64, 64), 40000, dtype=np.uint16)
    out = _derive_deep_hole([_det()], _meta_with(arr), base_density=0.0, bit_depth=16)
    assert out[0].deep_hole is False
    # 原 detection 的其它字段被保留（返回新实例而非丢弃）
    assert out[0].class_id is DefectClass.POROSITY
