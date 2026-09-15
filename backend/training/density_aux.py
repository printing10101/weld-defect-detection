"""缺陷密度图辅助任务：把"目标检测框"转成稠密回归目标。

为什么需要它
------------
本系统当前以 YOLO 检测为主（框回归 + 分类），但检测范式在两个场景下表达力不足，
而这两个场景恰好是 §16 的硬骨头：

1. **密集气孔群（§16 场景 4）**：几十个直径 1–2mm 的气孔彼此紧邻，NMS 会把它们
   并成一个框，"逐个计数"这一评片要求直接失真。密度图的**积分等于缺陷个数**，
   天然适配计数；
2. **含噪标注**：X 光缺陷边界半透明、评片员轮廓不一致、稀有类样本极少。框回归
   对定位误差敏感，而密度图 + :mod:`backend.training.losses` 的 SinkLoss（源自
   Marigold V2 的 SinkLoss）只约束"块内值的集合"，对位置抖动不敏感。

本模块提供"框 → 密度图"的正向映射（生产可用），以及供 SinkLoss 消费的分块视图。
反向（密度图 → 框/计数）走 :func:`density_integral` 与 :func:`count_peaks`。

约定
----
- 输入框为**绝对像素** ``[x, y, w, h]``（左上角 + 宽高，与
  :mod:`backend.evaluation.harness` 的 bbox 协议一致）；
- 归一化 YOLO 标签 → 绝对框用 :func:`load_yolo_boxes`（``cx cy w h`` 归一化）；
- 密度图单通道、float32。``mode="peak"`` 时值域裁剪到 [0,1]（与 losses 模块的
  输入契约一致）；``mode="integral"`` 时每团高斯积分为 1，全图积分 = 缺陷个数。

依赖：仅 numpy。峰值提取用 scikit-image（核心依赖，懒加载，缺失时该函数不可用）。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

__all__ = [
    "boxes_from_yolo_lines",
    "count_peaks",
    "density_integral",
    "density_tiles",
    "gaussian_density_map",
    "load_yolo_boxes",
    "sigma_for_box",
    "to_sinkhorn_scale",
]

# 高斯核的截断半径（以 sigma 计）。±3σ 覆盖 99.7% 质量，再远可忽略。
_TRUNC_SIGMA = 3.0


def sigma_for_box(
    w: float,
    h: float,
    *,
    ratio: float = 0.25,
    min_sigma: float = 1.0,
    max_sigma: float | None = None,
) -> float:
    """按框尺寸自适应决定高斯核宽度。

    取 ``ratio * sqrt(w*h)``（几何平均尺寸的固定比例），并夹到
    ``[min_sigma, max_sigma]``。用几何平均而非面积，是为了让细长裂纹
    （如 40×2）与等面积的方形得到相近的核宽——若用面积，细长目标会得到
    与其"薄"方向严重失配的大核，密度图上就糊成一片。

    :param min_sigma: 下限。像素级小缺陷（1px）若不加下限，sigma 会趋近 0，
        退化成"单像素冲激"，既不可学也无法与相邻缺陷区分。
    """
    if w <= 0 or h <= 0:
        return float(min_sigma)
    sigma = float(ratio) * float(np.sqrt(w * h))
    sigma = max(float(min_sigma), sigma)
    if max_sigma is not None:
        sigma = min(sigma, float(max_sigma))
    return sigma


def _add_gaussian(
    canvas: np.ndarray,
    cx: float,
    cy: float,
    sigma: float,
    *,
    unit_integral: bool,
) -> None:
    """把一团高斯的贡献原地加到 canvas 上（带边界裁剪，越界部分自然丢弃）。"""
    r = int(np.ceil(_TRUNC_SIGMA * sigma))
    h, w = canvas.shape
    x0, x1 = max(0, int(np.floor(cx)) - r), min(w, int(np.ceil(cx)) + r + 1)
    y0, y1 = max(0, int(np.floor(cy)) - r), min(h, int(np.ceil(cy)) + r + 1)
    if x0 >= x1 or y0 >= y1:
        return
    ys = np.arange(y0, y1, dtype=np.float64)[:, None] - cy
    xs = np.arange(x0, x1, dtype=np.float64)[None, :] - cx
    g = np.exp(-(xs**2 + ys**2) / (2.0 * sigma * sigma))
    if unit_integral:
        g = g / (2.0 * np.pi * sigma * sigma)
    canvas[y0:y1, x0:x1] += g


def gaussian_density_map(
    shape: tuple[int, int],
    boxes: Iterable[Sequence[float]],
    *,
    ratio: float = 0.25,
    min_sigma: float = 1.0,
    max_sigma: float | None = None,
    mode: str = "peak",
    classes: Iterable[int] | None = None,
    class_ids: Iterable[int] | None = None,
    dtype: type = np.float32,
) -> np.ndarray:
    """把一组框渲染成稠密密度图（各框中心叠加高斯）。

    :param shape: ``(H, W)``。
    :param boxes: 框序列，每项 ``[x, y, w, h]``（绝对像素）；若同时给了
        ``class_ids``，则按 ``classes`` 过滤。
    :param mode: ``"peak"``（默认）——每团高斯峰值 1，叠加后裁剪到 [0,1]，
        值域与 :mod:`backend.training.losses` 的输入契约一致；
        ``"integral"``——每团高斯积分为 1，**全图积分 = 保留的缺陷个数**，
        供 :func:`density_integral` 做计数。
    :param classes: 只保留这些 class_id（``None`` = 全部）。
    :param class_ids: 与 ``boxes`` 等长的 class_id 序列，配合 ``classes`` 使用。
    :returns: ``(H, W)`` 的数组。

    边界处理：高斯核超出画布的部分被裁掉，因此贴着边缘的缺陷其峰值可能略低于 1
    （``mode="peak"``）。这是刻意的——补零（padding）会凭空造出"图外还有缺陷"
    的假信号，而裁剪只损失边缘信息量，不引入错误方向。
    """
    h, w = int(shape[0]), int(shape[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"shape 须为正的 (H, W)，实际 {shape}")
    if mode not in ("peak", "integral"):
        raise ValueError(f"mode 须为 'peak' 或 'integral'，实际 {mode!r}")

    box_list = [list(map(float, b)) for b in boxes]
    if classes is not None and class_ids is not None:
        keep = {int(c) for c in classes}
        cids = [int(c) for c in class_ids]
        if len(cids) != len(box_list):
            raise ValueError(f"class_ids 长度 {len(cids)} 与 boxes 长度 {len(box_list)} 不一致")
        box_list = [b for b, c in zip(box_list, cids, strict=True) if c in keep]

    canvas = np.zeros((h, w), dtype=np.float64)
    for bx, by, bw, bh in box_list:
        if bw <= 0 or bh <= 0:
            continue
        sigma = sigma_for_box(bw, bh, ratio=ratio, min_sigma=min_sigma, max_sigma=max_sigma)
        _add_gaussian(
            canvas,
            bx + bw / 2.0,
            by + bh / 2.0,
            sigma,
            unit_integral=(mode == "integral"),
        )
    if mode == "peak":
        np.clip(canvas, 0.0, 1.0, out=canvas)
    return canvas.astype(dtype)


def density_integral(density: np.ndarray) -> float:
    """密度图积分。

    仅在 ``mode="integral"`` 渲染的密度图上等于缺陷个数；对 ``mode="peak"`` 的图
    该值无量纲、仅可作相对比较（这也是为什么计数必须用 integral 模式）。
    """
    return float(np.sum(np.asarray(density, dtype=np.float64)))


def density_tiles(density: np.ndarray, tile: int) -> np.ndarray:
    """把 ``(H, W)`` 密度图切成 ``(T, L)`` 分块视图，喂给 SinkLoss。

    ``T = (H//tile) * (W//tile)`` 个块，每块 ``L = tile*tile`` 个值（按行优先展平），
    与 :func:`backend.training.losses.torch_sinkhorn_cost` 的输入形状 (T, L) 对齐。
    不足一块的右/下边缘像素被裁掉（与 :func:`~backend.training.losses.tile_sinkhorn_loss`
    的切块口径一致，保证 numpy 版与 torch 版损失可比）。
    """
    arr = np.asarray(density, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"density 须为二维 (H, W)，实际 shape={arr.shape}")
    tile = int(tile)
    if tile <= 0:
        raise ValueError(f"tile 须为正整数，实际 {tile}")
    h, w = arr.shape
    n_h, n_w = h // tile, w // tile
    if n_h == 0 or n_w == 0:
        return np.zeros((0, tile * tile), dtype=np.float64)
    trimmed = arr[: n_h * tile, : n_w * tile]
    # (n_h, tile, n_w, tile) -> (n_h, n_w, tile, tile) -> (T, L)
    blocks = trimmed.reshape(n_h, tile, n_w, tile).transpose(0, 2, 1, 3)
    return blocks.reshape(n_h * n_w, tile * tile)


def to_sinkhorn_scale(density: np.ndarray, *, magnitude: float = 10.0) -> np.ndarray:
    """把密度图缩放到 SinkLoss 可用的量级。

    背景见 :mod:`backend.training.losses` 的"输入量纲契约"：传输代价
    ``C=(pred_i - gt_j)^2`` 在 [0,1] 值域下被压得极平。同一张 64×64 密度图实测：
    全零预测与正确预测的代价之比在尺度 1.0 时只有 **6.4×**，尺度 10.0 时 **466×**，
    尺度 100.0 时 **7.2 万×**。不缩放时 SinkLoss 几乎不给梯度——**静默失效**
    （不报错，只是不学），所以这一步不是可选优化而是必要前置。

    按峰值归一化再乘 ``magnitude``，使不同信噪比的输入落到同一量级：
    ``out = density / max(density) * magnitude``。全零图原样返回（避免除零）。
    """
    arr = np.asarray(density, dtype=np.float64)
    if arr.size == 0:
        return arr
    peak = float(arr.max())
    if peak <= 0:
        return arr
    return arr / peak * float(magnitude)


def count_peaks(
    density: np.ndarray,
    *,
    min_distance: int = 2,
    threshold_abs: float = 0.3,
) -> list[tuple[int, int]]:
    """从密度图提取峰值坐标（密集缺陷逐个计数的读数实现）。

    用 scikit-image 的 ``peak_local_max``（核心依赖，本函数内懒加载）。
    返回按行优先排序的 ``(y, x)`` 列表。

    :param min_distance: 两峰最小间距（px）。应取略小于最小缺陷间距，否则紧邻
        气孔仍会被并成一个峰——这正是 §16 场景 4 要避免的。
    :param threshold_abs: 绝对阈值下限（``mode="peak"`` 的图建议 0.3 左右）。
    """
    arr = np.asarray(density, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"density 须为二维 (H, W)，实际 shape={arr.shape}")
    if arr.size == 0 or float(arr.max()) < threshold_abs:
        return []
    from skimage.feature import peak_local_max

    coords = peak_local_max(
        arr,
        min_distance=int(min_distance),
        threshold_abs=float(threshold_abs),
        exclude_border=False,
    )
    return [(int(y), int(x)) for y, x in coords]


def boxes_from_yolo_lines(
    lines: Iterable[str],
    height: int,
    width: int,
) -> tuple[list[list[float]], list[int]]:
    """YOLO 标签行（``cls cx cy w h``，归一化）→ 绝对框 + class_id。

    返回 ``(boxes, class_ids)``，boxes 为 ``[x, y, w, h]``（左上角 + 宽高），
    与 harness 协议一致。跳过空行与字段不足的行（容错，不抛异常——
    标注文件常带尾随空行与半行残迹）。
    """
    boxes: list[list[float]] = []
    cids: list[int] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        if bw <= 0 or bh <= 0:
            continue
        boxes.append([(cx - bw / 2.0) * width, (cy - bh / 2.0) * height, bw * width, bh * height])
        cids.append(cls)
    return boxes, cids


def load_yolo_boxes(
    label_path: str | Path,
    height: int,
    width: int,
) -> tuple[list[list[float]], list[int]]:
    """读取 YOLO 标签文件 → ``(boxes, class_ids)``（见 :func:`boxes_from_yolo_lines`）。

    文件不存在时返回空列表（Golden Set 中允许存在"无缺陷"的负样本图）。
    """
    p = Path(label_path)
    if not p.exists():
        return [], []
    return boxes_from_yolo_lines(p.read_text(encoding="utf-8").splitlines(), height, width)
