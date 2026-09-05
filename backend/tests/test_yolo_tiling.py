"""Tiling 分块推理单测（不依赖 ONNX/torch 权重，monkeypatch _infer_single）。

覆盖：网格覆盖完整性、瓦片坐标平移回全图、跨瓦片 NMS 合并、触发条件、
瓦片数上限平滑降级、合并后 id 全局唯一。
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from backend.domain.detect.yolo_detector import YoloDetector
from backend.domain.dto import BBox, DefectClass, Detection


def _det(x: float, y: float, w: float, h: float, score: float = 0.9) -> Detection:
    return Detection(
        id="X",
        bbox=BBox(x=x, y=y, w=w, h=h),
        class_id=DefectClass.POROSITY,
        score=score,
        uncertainty=0.1,
    )


def _install_fake_single(monkeypatch, det: Detection, calls: list) -> None:
    """把 _infer_single 替换为恒返回"裁切图中心处 det"的哑实现，并记录每次瓦片尺寸。

    放裁切中心：中心必然落在核心区（上下文外扩对称），不会被跨瓦片
    归属过滤丢掉；坐标断言据此计算期望值。
    """

    def fake(self, image, conf, iou, class_conf=None):
        calls.append(image.shape)
        ch, cw = image.shape[:2]
        return [
            Detection(
                id="X",
                bbox=BBox(x=cw / 2 - det.bbox.w / 2, y=ch / 2 - det.bbox.h / 2, w=det.bbox.w, h=det.bbox.h),
                class_id=det.class_id,
                score=det.score,
                uncertainty=det.uncertainty,
            )
        ]

    monkeypatch.setattr(YoloDetector, "_infer_single", fake)


def _expected_core_coord(span: int, tile: int, ov: float, origin: int, det_side: float) -> float:
    """期望的全图坐标：裁切中心减去半个检测框（与 _install_fake_single 对应）。"""
    pad = max(0, int(tile * ov / 2))
    lo, hi = max(0, origin - pad), min(span, origin + tile + pad)
    return lo + (hi - lo) / 2 - det_side / 2


def test_grid_origins_covers_span() -> None:
    """起点序列（含贴边收尾）的瓦片并集必须完整覆盖 [0, span)。"""
    for span, tile, ov in [
        (2500, 1000, 0.2),
        (8000, 1280, 0.2),
        (800, 1000, 0.2),
        (2401, 1200, 0.5),
    ]:
        origins = YoloDetector._grid_origins(span, tile, ov)
        step = tile * (1 - ov)
        covered_end = 0
        for o in origins:
            assert o <= covered_end + 1e-9, f"瓦片间出现未覆盖缝隙: {origins}"
            covered_end = o + tile
        assert covered_end >= span, f"末尾未覆盖到 {span}: {origins}"
        assert origins[0] == 0
        # 步进不超过 (1-ov)*tile 才能保证重叠
        for a, b in pairwise(origins):
            assert b - a <= step + 1e-9


def test_tiled_offsets_and_coverage(monkeypatch) -> None:
    """每个瓦片的检出坐标须平移回全图坐标系（含上下文外扩回移）；瓦片网格铺满整图。"""
    det = _det(10, 10, 50, 50)
    calls: list = []
    _install_fake_single(monkeypatch, det, calls)
    d = YoloDetector()
    d.tile_size = 1000
    d.tile_trigger_side = 0  # 强制触发
    img = np.zeros((3000, 2000), dtype=np.uint8)
    out = d.infer(img, conf=0.3, iou=0.5)
    # x span 2000：step 800 → 起点 [0,800]+贴边 1000；y span 3000：[0,800,1600]+贴边 2000
    assert len(calls) == 12
    assert len(out) == 12
    # 期望坐标 = 各瓦片（含外扩后裁切）的裁切中心，全 12 块各贡献一条
    x_origins = YoloDetector._grid_origins(2000, 1000, 0.2)
    y_origins = YoloDetector._grid_origins(3000, 1000, 0.2)
    expect_xs = {round(_expected_core_coord(2000, 1000, 0.2, o, 50)) for o in x_origins}
    expect_ys = {round(_expected_core_coord(3000, 1000, 0.2, o, 50)) for o in y_origins}
    xs = {round(dv.bbox.x) for dv in out}
    ys = {round(dv.bbox.y) for dv in out}
    assert xs == expect_xs
    assert ys == expect_ys
    # id 全局唯一
    ids = [dv.id for dv in out]
    assert len(ids) == len(set(ids))


def test_tiled_small_image_single_path(monkeypatch) -> None:
    """最长边未超触发阈值 → 不分块，单次整图推理。"""
    calls: list = []
    _install_fake_single(monkeypatch, _det(5, 5, 10, 10), calls)
    d = YoloDetector()
    d.tile_size = 1280
    d.tile_trigger_side = 2400
    img = np.zeros((1200, 2000), dtype=np.uint8)
    out = d.infer(img, conf=0.3, iou=0.5)
    assert len(calls) == 1
    assert len(out) == 1
    # 坐标不做任何平移/缩放：哑实现画在整图中心
    assert out[0].bbox.x == 2000 / 2 - 5
    assert out[0].bbox.y == 1200 / 2 - 5


def test_tile_size_zero_disables(monkeypatch) -> None:
    """tile_size=0（默认）→ 永远整图推理。"""
    calls: list = []
    _install_fake_single(monkeypatch, _det(5, 5, 10, 10), calls)
    d = YoloDetector()
    img = np.zeros((6000, 8000), dtype=np.uint8)
    d.infer(img, conf=0.3, iou=0.5)
    assert len(calls) == 1


def test_tiled_nms_merges_overlap_dups(monkeypatch) -> None:
    """重叠区同一缺陷被相邻瓦片重复检出 → 跨瓦片 NMS 合并（按类独立、宽松阈值）。"""
    # 每瓦片返回其裁切中心的同一块"缺陷"（900×900）；overlap=0.9 时相邻瓦片
    # 的检出框 IoU 很高，应被合并。
    calls: list = []
    _install_fake_single(monkeypatch, _det(0, 0, 900, 900), calls)
    d = YoloDetector()
    d.tile_size = 1000
    d.tile_overlap = 0.9
    d.tile_trigger_side = 0
    img = np.zeros((1000, 2000), dtype=np.uint8)  # x 方向 11 块瓦片，y 方向 1 块
    out = d.infer(img, conf=0.3, iou=0.5)
    assert len(calls) == 11
    assert 0 < len(out) < 11  # 合并发生
    ids = [dv.id for dv in out]
    assert len(ids) == len(set(ids))


def test_tiled_merge_keeps_different_classes(monkeypatch) -> None:
    """跨瓦片合并按类独立：同区域不同类的两检出不得互吞（类别不被高分者吞并）。"""
    calls: list = []
    _install_fake_single(monkeypatch, _det(0, 0, 900, 900), calls)

    orig_infer_single = YoloDetector._infer_single

    def fake(self, image, conf, iou, class_conf=None):
        dets = orig_infer_single(self, image, conf, iou, class_conf)
        # 同一框位置交替给两个类别（气孔 0 / 夹渣 1），坐标相同 → 类无关 NMS
        # 会互吞，类无关实现下相邻瓦片同类框仍正常合并。
        out = []
        for k, det in enumerate(dets):
            out.append(
                Detection(
                    id=det.id,
                    bbox=det.bbox,
                    class_id=DefectClass.POROSITY if len(calls) % 2 == 0 else DefectClass.SLAG,
                    score=det.score,
                    uncertainty=det.uncertainty,
                )
            )
        return out

    monkeypatch.setattr(YoloDetector, "_infer_single", fake)
    d = YoloDetector()
    d.tile_size = 1000
    d.tile_overlap = 0.9
    d.tile_trigger_side = 0
    img = np.zeros((1000, 2000), dtype=np.uint8)
    out = d.infer(img, conf=0.3, iou=0.5)
    # 同类框各自合并：结果中两类都应有存活（类无关 NMS 会只剩一类）
    classes = {dv.class_id for dv in out}
    assert DefectClass.POROSITY in classes
    assert DefectClass.SLAG in classes


def test_nms_handles_zero_score_candidates() -> None:
    """全零分候选（conf=0 合法输入）经 cv2 NMS 不得抛 IndexError。"""
    boxes = [(0.0, 0.0, 10.0, 10.0, 0.0, 0), (1.0, 1.0, 11.0, 11.0, 0.0, 0)]
    keep = YoloDetector._nms(boxes, 0.5)
    assert isinstance(keep, list)
    assert all(0 <= i < len(boxes) for i in keep)


def test_nms_class_aware() -> None:
    """class_aware=True：不同类的高 IoU 框互不抑制。"""
    a = (0.0, 0.0, 10.0, 10.0, 0.9, 0)
    b = (0.0, 0.0, 10.0, 10.0, 0.8, 1)
    assert len(YoloDetector._nms([a, b], 0.5, class_aware=True)) == 2
    assert len(YoloDetector._nms([a, b], 0.5, class_aware=False)) == 1


def test_effective_tile_downgrades_over_budget() -> None:
    """瓦片数超 max_count → 瓦片边长按 1.5× 递增直至预算内（平滑降级）。"""
    d = YoloDetector()
    d.tile_size = 1000
    d.tile_trigger_side = 0
    d.tile_max_count = 400
    img = np.zeros((20000, 20000), dtype=np.uint8)
    assert d._effective_tile(img) == 1500  # 1000→576 块超限；1500→289 块≤400


def test_effective_tile_none_when_disabled_or_small() -> None:
    d = YoloDetector()
    assert d._effective_tile(np.zeros((100, 100), dtype=np.uint8)) is None  # 关闭
    d.tile_size = 1280
    d.tile_trigger_side = 2400
    assert d._effective_tile(np.zeros((2000, 2000), dtype=np.uint8)) is None  # 未达阈值
    assert d._effective_tile(np.zeros((100, 4000), dtype=np.uint8)) is not None  # 长条触发


@pytest.mark.parametrize("shape", [(0, 100), (50, 0)])
def test_empty_image_returns_empty(shape) -> None:
    d = YoloDetector()
    d.tile_size = 1280
    assert d.infer(np.zeros(shape, dtype=np.uint8), 0.3, 0.5) == []
