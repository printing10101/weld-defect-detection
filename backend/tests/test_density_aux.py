"""缺陷密度图辅助任务单测（backend/training/density_aux.py）。

覆盖：核宽自适应（几何均值 vs 面积）、min/max 夹取、peak/integral 两种归一化、
越界裁剪不引入假信号、类别过滤、YOLO 标签解析容错、分块视图与 SinkLoss 对接、
峰值计数（skimage 懒加载 → 缺失则 skip）。
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.training.density_aux import (
    boxes_from_yolo_lines,
    count_peaks,
    density_integral,
    density_tiles,
    gaussian_density_map,
    load_yolo_boxes,
    sigma_for_box,
    to_sinkhorn_scale,
)
from backend.training.losses import tile_sinkhorn_loss

# ---------------------------------------------------------------------------
# 核宽自适应
# ---------------------------------------------------------------------------


def test_sigma_grows_with_box_size() -> None:
    assert sigma_for_box(10, 10) < sigma_for_box(20, 20) < sigma_for_box(40, 40)


def test_sigma_uses_geometric_mean_not_area() -> None:
    """细长裂纹 40×2 → 核宽按 sqrt(80)≈8.94 计，而非按面积 80。

    若误用面积，细长目标的核会在"薄"方向严重失配，密度图上糊成一片。
    """
    from math import sqrt

    got = sigma_for_box(40, 2, ratio=0.25, min_sigma=0.1)
    assert got == pytest.approx(0.25 * sqrt(80), abs=1e-6)
    assert got < 0.25 * 80  # 显著小于面积口径


def test_sigma_clamped_by_min() -> None:
    """1px 缺陷若不加下限，sigma→0.25 会退化成单像素冲激。"""
    assert sigma_for_box(1, 1, ratio=0.25, min_sigma=1.0) == pytest.approx(1.0)


def test_sigma_clamped_by_max() -> None:
    assert sigma_for_box(400, 400, ratio=0.25, min_sigma=1.0, max_sigma=20.0) == pytest.approx(20.0)


def test_sigma_degenerate_box_returns_min() -> None:
    assert sigma_for_box(0, 10, min_sigma=1.5) == pytest.approx(1.5)
    assert sigma_for_box(10, -3, min_sigma=2.0) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 密度图渲染
# ---------------------------------------------------------------------------


def test_peak_mode_max_is_one_at_center() -> None:
    d = gaussian_density_map((64, 64), [[27.0, 27.0, 10.0, 10.0]])
    assert d.dtype == np.float32
    assert float(d.max()) == pytest.approx(1.0, abs=1e-6)
    cy, cx = np.unravel_index(int(np.argmax(d)), d.shape)
    assert (cy, cx) == (32, 32)  # 中心 = 27 + 10/2


def test_peak_mode_decays_monotonically_away_from_center() -> None:
    d = gaussian_density_map((64, 64), [[27.0, 27.0, 10.0, 10.0]])
    row = d[32, 32:]
    assert np.all(np.diff(row) <= 1e-6)  # 沿行单调不增


def test_peak_mode_clips_overlap_to_one() -> None:
    """3 个完全重合的框：叠加后远超 1，必须裁剪（losses 输入契约是 [0,1]）。"""
    box = [27.0, 27.0, 10.0, 10.0]
    d = gaussian_density_map((64, 64), [box, box, box])
    assert float(d.max()) == pytest.approx(1.0, abs=1e-6)


def test_integral_mode_sums_to_defect_count() -> None:
    """integral 模式下全图积分 = 缺陷个数（密集气孔群计数的依据）。"""
    boxes = [[10.0, 10.0, 8.0, 8.0], [30.0, 30.0, 8.0, 8.0], [50.0, 50.0, 6.0, 6.0]]
    d = gaussian_density_map((64, 64), boxes, mode="integral")
    assert density_integral(d) == pytest.approx(3.0, abs=0.02)


def test_modes_differ_in_integral() -> None:
    """同样的框，peak 图积分随核宽变化，不等于个数——故计数必须用 integral 模式。"""
    boxes = [[20.0, 20.0, 16.0, 16.0], [44.0, 44.0, 16.0, 16.0]]
    peak = density_integral(gaussian_density_map((64, 64), boxes, mode="peak"))
    integ = density_integral(gaussian_density_map((64, 64), boxes, mode="integral"))
    assert integ == pytest.approx(2.0, abs=0.02)
    assert peak != pytest.approx(2.0, abs=0.05)


def test_class_filter_keeps_only_requested() -> None:
    boxes = [[10.0, 10.0, 8.0, 8.0], [40.0, 40.0, 8.0, 8.0]]
    cids = [0, 3]
    only3 = gaussian_density_map((64, 64), boxes, classes={3}, class_ids=cids, mode="integral")
    assert density_integral(only3) == pytest.approx(1.0, abs=0.02)
    assert only3[12, 12] == pytest.approx(0.0, abs=1e-6)  # class 0 的框未渲染


def test_class_ids_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="长度"):
        gaussian_density_map((32, 32), [[1.0, 1.0, 4.0, 4.0]], classes={0}, class_ids=[0, 1])


def test_invalid_mode_raises() -> None:
    with pytest.raises(ValueError, match="mode"):
        gaussian_density_map((32, 32), [], mode="max")


def test_invalid_shape_raises() -> None:
    with pytest.raises(ValueError, match="shape"):
        gaussian_density_map((0, 32), [])


def test_box_outside_canvas_is_noop() -> None:
    """完全在图外的框不得凭空造出信号（补零会造出假缺陷）。"""
    d = gaussian_density_map((32, 32), [[1000.0, 1000.0, 10.0, 10.0]])
    assert float(d.sum()) == pytest.approx(0.0)


def test_partial_outside_keeps_interior_signal() -> None:
    """贴边缺陷：核被裁，但图内部分仍有信号（峰值可低于 1，这是刻意的）。"""
    d = gaussian_density_map((32, 32), [[-4.0, 12.0, 10.0, 10.0]])
    assert float(d.max()) > 0.5
    assert float(d.max()) <= 1.0


def test_degenerate_box_skipped() -> None:
    d = gaussian_density_map((32, 32), [[10.0, 10.0, 0.0, 5.0]])
    assert float(d.sum()) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 分块视图（喂 SinkLoss）
# ---------------------------------------------------------------------------


def test_density_tiles_shape_and_content() -> None:
    arr = np.zeros((8, 4))
    arr[:4, :] = 1.0
    arr[4:, :] = 5.0
    tiles = density_tiles(arr, 4)
    assert tiles.shape == (2, 16)
    assert np.allclose(tiles[0], 1.0)
    assert np.allclose(tiles[1], 5.0)


def test_density_tiles_trims_remainder() -> None:
    """9×9 图按 tile=4 切 → 只用 8×8，第 9 行/列的哨兵值不得出现。"""
    arr = np.zeros((9, 9))
    arr[8, :] = 99.0
    arr[:, 8] = 99.0
    tiles = density_tiles(arr, 4)
    assert tiles.shape == (4, 16)
    assert float(tiles.max()) < 1.0


def test_density_tiles_too_small_returns_empty() -> None:
    tiles = density_tiles(np.zeros((3, 3)), 4)
    assert tiles.shape == (0, 16)


def test_density_tiles_bad_input() -> None:
    with pytest.raises(ValueError, match="二维"):
        density_tiles(np.zeros((4, 4, 1)), 2)
    with pytest.raises(ValueError, match="tile"):
        density_tiles(np.zeros((4, 4)), 0)


def test_density_tiles_bridge_to_sinkloss_identical_is_near_zero() -> None:
    """分块视图必须能直接喂 SinkLoss；自身与自身匹配代价≈0，空预测则代价显著。"""
    d = gaussian_density_map((64, 64), [[20.0, 20.0, 12.0, 12.0], [44.0, 44.0, 8.0, 8.0]])
    scaled = to_sinkhorn_scale(d, magnitude=10.0)
    ident = tile_sinkhorn_loss(scaled, scaled, tile=8)
    empty = tile_sinkhorn_loss(np.zeros_like(scaled), scaled, tile=8)
    assert ident < 1e-2
    assert empty > 50 * ident


# ---------------------------------------------------------------------------
# 缩放：避免 SinkLoss 在 [0,1] 值域下静默失效
# ---------------------------------------------------------------------------


def _contrast(density: np.ndarray) -> float:
    """空预测代价 / 正确预测代价 —— 损失的可分辨性（越低越接近"平坦无梯度"）。"""
    ident = tile_sinkhorn_loss(density, density, tile=8)
    empty = tile_sinkhorn_loss(np.zeros_like(density), density, tile=8)
    return empty / max(ident, 1e-12)


def test_to_sinkhorn_scale_normalizes_peak() -> None:
    d = gaussian_density_map((32, 32), [[10.0, 10.0, 8.0, 8.0]])
    s = to_sinkhorn_scale(d, magnitude=7.0)
    assert float(s.max()) == pytest.approx(7.0, abs=1e-6)


def test_to_sinkhorn_scale_zero_map_is_safe() -> None:
    """全零图不得除零（Golden Set 允许无缺陷负样本）。"""
    s = to_sinkhorn_scale(np.zeros((8, 8)))
    assert s.shape == (8, 8)
    assert float(s.sum()) == pytest.approx(0.0)


def test_to_sinkhorn_scale_restores_contrast() -> None:
    """回归护栏：不缩放时对比度塌缩到个位数（静默失效），缩放后必须恢复两个数量级。"""
    d = gaussian_density_map((64, 64), [[20.0, 20.0, 12.0, 12.0], [44.0, 44.0, 8.0, 8.0]])
    raw = _contrast(d)
    scaled = _contrast(to_sinkhorn_scale(d, magnitude=10.0))
    assert raw < 20  # [0,1] 值域：几乎不可分辨
    assert scaled > 100  # 尺度 10：可分辨
    assert scaled > 10 * raw


# ---------------------------------------------------------------------------
# 标签解析
# ---------------------------------------------------------------------------


def test_boxes_from_yolo_lines_roundtrip() -> None:
    boxes, cids = boxes_from_yolo_lines(["0 0.5 0.5 0.3 0.3"], 64, 64)
    assert cids == [0]
    assert boxes[0] == pytest.approx([22.4, 22.4, 19.2, 19.2])


def test_boxes_from_yolo_lines_tolerates_junk() -> None:
    """真实标注文件常带尾随空行/半行残迹，须容错而非抛异常。"""
    boxes, cids = boxes_from_yolo_lines(
        ["", "  ", "0 0.5 0.5 0.3 0.3", "abc def", "1 0.5 0.5", "2 0.5 0.5 0.0 0.2"],
        64,
        64,
    )
    assert cids == [0]
    assert len(boxes) == 1


def test_load_yolo_boxes_missing_file_is_empty(tmp_path) -> None:
    """Golden Set 允许"无缺陷"负样本（无标签文件），不得报错。"""
    assert load_yolo_boxes(tmp_path / "nope.txt", 64, 64) == ([], [])


def test_load_yolo_boxes_reads_file(tmp_path) -> None:
    p = tmp_path / "a.txt"
    p.write_text("3 0.25 0.75 0.1 0.2\n", encoding="utf-8")
    boxes, cids = load_yolo_boxes(p, 100, 200)
    assert cids == [3]
    assert boxes[0] == pytest.approx([40.0, 65.0, 20.0, 20.0])


# ---------------------------------------------------------------------------
# 峰值计数（密集气孔群逐个读数）
# ---------------------------------------------------------------------------


def test_count_peaks_finds_separated_defects() -> None:
    pytest.importorskip("skimage")
    boxes = [[8.0, 8.0, 8.0, 8.0], [28.0, 28.0, 8.0, 8.0], [48.0, 48.0, 8.0, 8.0]]
    d = gaussian_density_map((64, 64), boxes)
    peaks = count_peaks(d, min_distance=3, threshold_abs=0.3)
    assert len(peaks) == 3
    assert peaks == sorted(peaks)


def test_count_peaks_empty_map_returns_empty() -> None:
    pytest.importorskip("skimage")
    assert count_peaks(np.zeros((32, 32))) == []


def test_count_peaks_bad_dims_raises() -> None:
    pytest.importorskip("skimage")
    with pytest.raises(ValueError, match="二维"):
        count_peaks(np.zeros((4, 4, 2)))
