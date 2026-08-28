"""编排层（backend/app/pipelines.py）测试（F31）。

不依赖 ML 权重：复用 conftest 注入的 baseline(blob) 检测器 + 临时库，
覆盖：
- run_inspection 全链路（force 出片）返回结构正确且落库/报告/审计到位；
- 不合格底片（无 IQI）在 force=False 时阻断评片（IQIFailError）；
- 纯编排辅助函数 _resolve_spacing / _shape_of / _derive_deep_hole 的确定性行为。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from skimage.io import imsave

from backend.app.dependencies import get_registry
from backend.app.pipelines import (
    InspectionPipeline,
    _derive_deep_hole,
    _resolve_spacing,
    _shape_of,
)
from backend.domain.dto import BBox, DefectClass, DefectShape, Detection, ImageMeta, Modality
from backend.domain.errors import IQIFailError
from backend.domain.quantify import BBoxQuantifier


def _synthetic_image(path: Path, *, with_defect: bool = True) -> Path:
    """生成 512×512 8bit 灰度底片：浅色母材 + 深色焊缝带 + 可选圆形缺陷。"""
    arr = np.full((512, 512), 160, dtype=np.uint8)
    arr[:, 230:282] = 120  # 焊缝带
    if with_defect:
        ys, xs = np.ogrid[0:512, 0:512]
        mask = (xs - 256) ** 2 + (ys - 256) ** 2 <= 18**2
        arr[mask] = 55  # 暗缺陷
    imsave(str(path), arr)
    return path


# ---------------------------------------------------------------------------
# 全链路编排（force 出片）
# ---------------------------------------------------------------------------
def test_run_inspection_force_persists_and_reports(tmp_path: Path) -> None:
    reg = get_registry()
    pipe = InspectionPipeline(reg)
    img = _synthetic_image(tmp_path / "film.png")
    out = pipe.run_inspection(
        img,
        pixel_spacing_mm=0.1,
        base_metal_thickness_mm=20,
        force=True,
        actor="tester",
    )
    # 1. 返回结构完整
    for key in (
        "image_id",
        "report_id",
        "joint_level",
        "need_review",
        "evaluable",
        "density",
        "density_ok",
        "iqi_pass",
        "defect_count",
        "pdf_path",
    ):
        assert key in out
    assert out["image_id"] and out["report_id"]
    assert out["pdf_path"]
    assert Path(out["pdf_path"]).exists(), "报告 PDF 应已生成并落盘"
    # 2. 落库：可按 image_id 取回
    stored = reg.repository.get_image(out["image_id"])
    assert stored is not None, "评片结果应已落库"
    # 3. 不可评底片（未授权标准）→ 产出 None 级别 + 需人工复核（不臆造级别）
    assert out["joint_level"] is None
    assert out["need_review"] is True
    # 4. 审计：应有一笔 inspect 记录
    entries, total = reg.repository.list_audit(object_id=out["image_id"])
    assert total >= 1, "应写入 inspect 审计"
    assert any(e.get("action") == "inspect" for e in entries), "审计动作应为 inspect"


def test_run_inspection_nonforce_bad_film_blocks(tmp_path: Path) -> None:
    """不合格底片（无 IQI）在 force=False 时阻断评片（IQIFailError）。"""
    reg = get_registry()
    pipe = InspectionPipeline(reg)
    # 均匀灰片：无 IQI/缺陷，质量门禁判定不可评
    uniform = tmp_path / "uniform.png"
    imsave(str(uniform), np.full((256, 256), 128, dtype=np.uint8))
    with pytest.raises(IQIFailError):
        pipe.run_inspection(
            uniform,
            pixel_spacing_mm=0.1,
            base_metal_thickness_mm=20,
            force=False,
        )


# ---------------------------------------------------------------------------
# 纯编排辅助函数（确定性，无需 Registry）
# ---------------------------------------------------------------------------
def test_resolve_spacing_priority_and_fallback() -> None:
    assert _resolve_spacing(0.2, None) == (0.2, True)
    assert _resolve_spacing(None, 0.3) == (0.3, True)
    assert _resolve_spacing(0.0, None) == (1.0, False)  # 全无效 → 占位 1.0 且不可信
    assert _resolve_spacing(-1.0, -2.0) == (1.0, False)
    # requested 优先于 from_meta
    assert _resolve_spacing(0.5, 0.9) == (0.5, True)


def test_shape_of_prefers_detection_shape() -> None:
    q = BBoxQuantifier()
    det = Detection(
        id="x",
        bbox=BBox(0, 0, 40, 40),
        class_id=next(iter(DefectClass)),
        score=0.9,
        uncertainty=0.1,
        shape=DefectShape.LINEAR,
        mask_ref=None,
        deep_hole=False,
    )
    geom = q.measure(det, 1.0)
    # 检测器已给 shape → 直接采用（不依赖长宽比）
    assert _shape_of(det, geom, 3.0) is DefectShape.LINEAR

    det_no_shape = Detection(
        id="y",
        bbox=BBox(0, 0, 40, 40),
        class_id=next(iter(DefectClass)),
        score=0.9,
        uncertainty=0.1,
        shape=None,
        mask_ref=None,
        deep_hole=False,
    )
    geom2 = q.measure(det_no_shape, 1.0)
    expected = DefectShape.ROUND if geom2.aspect_ratio <= 3.0 else DefectShape.LINEAR
    assert _shape_of(det_no_shape, geom2, 3.0) is expected


def test_derive_deep_hole(monkeypatch: pytest.MonkeyPatch) -> None:
    # 固定 density 估计语义，使比较确定
    monkeypatch.setattr(
        "backend.app.pipelines.estimate_density",
        lambda arr, bit_depth: float(np.asarray(arr).mean()),
    )
    da = np.zeros((60, 60), dtype=float)
    da[10:30, 10:30] = 3.0  # 内部黑度远高于母材
    cls = next(iter(DefectClass))
    d_in = Detection(
        id="d1",
        bbox=BBox(10, 10, 20, 20),
        class_id=cls,
        score=0.9,
        uncertainty=0.1,
        shape=DefectShape.ROUND,
        mask_ref=None,
        deep_hole=False,
    )
    out = _derive_deep_hole(
        [d_in], ImageMeta(modality=Modality.CR, density_array=da), base_density=1.0, bit_depth=8
    )
    assert out[0].deep_hole is True

    # 低密度区不应标 deep_hole
    d_low = Detection(
        id="d2",
        bbox=BBox(40, 40, 10, 10),
        class_id=cls,
        score=0.9,
        uncertainty=0.1,
        shape=DefectShape.ROUND,
        mask_ref=None,
        deep_hole=False,
    )
    out2 = _derive_deep_hole(
        [d_low], ImageMeta(modality=Modality.CR, density_array=da), base_density=1.0, bit_depth=8
    )
    assert out2[0].deep_hole is False


def test_derive_deep_hole_no_density_array_passthrough() -> None:
    cls = next(iter(DefectClass))
    d_in = Detection(
        id="d1",
        bbox=BBox(0, 0, 5, 5),
        class_id=cls,
        score=0.9,
        uncertainty=0.1,
        shape=DefectShape.ROUND,
        mask_ref=None,
        deep_hole=False,
    )
    # 无 density_array → 原样返回，不臆造
    assert (
        _derive_deep_hole([d_in], ImageMeta(modality=Modality.CR, density_array=None), 1.0, 8)[
            0
        ].deep_hole
        is False
    )
