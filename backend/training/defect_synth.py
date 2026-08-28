"""GLASS 式缺陷合成生成器（P0-B）。

借鉴 GLASS（ECCV'24，gradient-ascent anomaly synthesis）的核心思路——
「程序化合成缺陷样本，低成本扩充训练池」——本项目用可解释的几何+灰度
合成方法在模拟 X 光底片上生成六类焊缝缺陷（气孔/夹渣/未焊透/未熔合/裂纹/咬边），
输出 YOLO images+labels，落到 data/training/raw/synthetic（dataset_builder 自动纳入，
无需改装配逻辑）。

与 augment.copy-paste 互补：
- copy-paste：复用真实缺陷裁剪，保真但数量受限于已有标注；
- 本模块：从零合成，**无限扩量 + 精确控制类别/尺寸分布**，尤其可定向扩充
  罕见且安全关键的裂纹/未熔合/未焊透（真实标注仅 27 框，远不够）。

合成语义（X 光底片：缺陷呈暗色，位于焊缝亮带内/边缘）：
- 气孔(0)：圆形暗斑，软边（高斯衰减）
- 夹渣(1)：不规则多边形暗块，边缘毛糙
- 未焊透(2)：焊缝根部细暗线（水平，窄长）
- 未熔合(3)：拉长条带，软边，微旋转
- 裂纹(4)：细锯齿折线（随机游走）
- 咬边(5)：焊缝边缘一排小凹口（垂直短线段）

生成器是纯 numpy/cv2 实现，可单测（box 与掩膜对齐、YOLO 格式合法、类别分布可控）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

# 类别名（与 DefectClass / YOLO_CLASSES 一致）
CLASS_NAMES = ["气孔", "夹渣", "未焊透", "未熔合", "裂纹", "咬边"]

# 默认类别权重：罕见且安全关键的缺陷加权，缓解长尾（气孔在真实集中占 94.6%）
_DEFAULT_CLASS_WEIGHTS = {
    0: 1.0,  # 气孔（保持一定比例，别全压没）
    1: 2.5,  # 夹渣
    2: 3.0,  # 未焊透
    3: 3.5,  # 未熔合
    4: 4.0,  # 裂纹（最罕见、最危险）
    5: 2.0,  # 咬边
}


def _make_background(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """模拟 X 光底片背景：胶片灰底 + 水平焊缝亮带（高斯边缘）+ 细粒度噪声。"""
    base = rng.uniform(55, 85)
    img = np.full((h, w), base, dtype=np.float32)
    img += rng.normal(0, 7, (h, w)).astype(np.float32)  # 底片颗粒
    band_h = max(8, int(h * 0.10))
    yy = np.arange(h, dtype=np.float32)[:, None]
    profile = np.exp(-0.5 * ((yy - h / 2) / (band_h * 0.45)) ** 2)
    img += (profile * rng.uniform(45, 70)) * np.ones((1, w), dtype=np.float32)
    img += rng.normal(0, 4, (h, w)).astype(np.float32)
    img = cv2.GaussianBlur(img, (3, 3), 0.8)
    return np.clip(img, 0, 255).astype(np.uint8)


def _ellipse_mask(
    rng: np.random.Generator, h: int, w: int, cx: float, cy: float, rx: float, ry: float
) -> np.ndarray:
    """带高斯软边的椭圆掩膜（气孔/条带共用）。"""
    yy, xx = np.mgrid[0:h, 0:w]
    dist2 = ((xx - cx) / max(rx, 1e-3)) ** 2 + ((yy - cy) / max(ry, 1e-3)) ** 2
    soft = np.clip(1.0 - dist2, 0.0, 1.0) ** 2
    return soft.astype(np.float32)


def _polygon_mask(
    rng: np.random.Generator, h: int, w: int, cx: float, cy: float, r: float, n: int
) -> np.ndarray:
    """不规则多边形掩膜（夹渣）。"""
    pts = []
    for _ in range(n):
        ang = rng.uniform(0, 2 * np.pi)
        rad = r * rng.uniform(0.55, 1.15)
        pts.append((int(cx + rad * np.cos(ang)), int(cy + rad * np.sin(ang))))
    poly = np.zeros((h, w), np.uint8)
    cv2.fillPoly(poly, [np.array(pts, np.int32)], 255)
    return (cv2.GaussianBlur(poly, (3, 3), 1.0) / 255.0).astype(np.float32)


def _line_mask(
    rng: np.random.Generator, h: int, w: int, x1: float, y1: float, x2: float, y2: float, thick: int
) -> np.ndarray:
    """细线掩膜（未焊透/裂纹骨架）。"""
    mask = np.zeros((h, w), np.uint8)
    cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, thick)
    blur_k = thick * 2 + 1
    return (cv2.GaussianBlur(mask, (blur_k, blur_k), 0.8) / 255.0).astype(np.float32)


def _crack_mask(
    rng: np.random.Generator, h: int, w: int, cx: float, cy: float, length: float
) -> np.ndarray:
    """锯齿折线裂纹掩膜（随机游走）。"""
    mask = np.zeros((h, w), np.uint8)
    n_seg = max(3, int(length / 12))
    pts = []
    x, y = cx - length / 2, cy
    dx = length / n_seg
    for _ in range(n_seg + 1):
        pts.append((x, y))
        x += dx
        y += rng.uniform(-3, 3)  # 锯齿偏摆
    for i in range(len(pts) - 1):
        cv2.line(
            mask, (int(pts[i][0]), int(pts[i][1])), (int(pts[i + 1][0]), int(pts[i + 1][1])), 255, 1
        )
    return (cv2.GaussianBlur(mask, (3, 3), 0.6) / 255.0).astype(np.float32)


def _undercut_mask(
    rng: np.random.Generator, h: int, w: int, cx: float, edge_y: float, length: float
) -> np.ndarray:
    """咬边：焊缝上/下边缘的一排小凹口（短竖线段）。"""
    mask = np.zeros((h, w), np.uint8)
    n = max(3, int(length / 10))
    for i in range(n):
        x = int(cx - length / 2 + i * (length / n))
        depth = int(rng.uniform(2, 5))
        side = 1 if rng.random() < 0.5 else -1
        y1 = int(edge_y)
        y2 = int(edge_y + side * depth)
        cv2.line(mask, (x, min(y1, y2)), (x, max(y1, y2)), 255, rng.choice([1, 2]))
    return (cv2.GaussianBlur(mask, (3, 3), 0.6) / 255.0).astype(np.float32)


def _bbox_of(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0.15)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _synthesize_one(
    rng: np.random.Generator, h: int, w: int, class_id: int
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """合成单个缺陷掩膜 + bbox (x1,y1,x2,y2)。缺陷统一画在焊缝亮带附近。

    弱掩膜（bbox 过小）时最多重试 20 次，避免递归深度失控。
    """
    for _ in range(20):
        cx = rng.uniform(w * 0.2, w * 0.8)
        cy = rng.uniform(h * 0.45, h * 0.55)  # 焊缝带内
        band_h = max(8, int(h * 0.10))
        edge_top = h / 2 - band_h * 0.5
        edge_bot = h / 2 + band_h * 0.5

        if class_id == 0:  # 气孔
            r = rng.uniform(2.0, 7.0)
            mask = _ellipse_mask(rng, h, w, cx, cy, r, r * rng.uniform(0.8, 1.2))
        elif class_id == 1:  # 夹渣
            mask = _polygon_mask(rng, h, w, cx, cy, rng.uniform(5.0, 12.0), int(rng.integers(5, 9)))
        elif class_id == 2:  # 未焊透：焊缝根部细长暗线
            length = rng.uniform(40.0, 160.0)
            mask = _line_mask(
                rng, h, w, cx - length / 2, cy, cx + length / 2, cy, int(rng.choice([1, 2]))
            )
        elif class_id == 3:  # 未熔合：拉长条带（沿焊缝方向，微旋转）
            length = rng.uniform(50.0, 150.0)
            ry = rng.uniform(3.0, 8.0)
            ang = rng.uniform(-0.15, 0.15)
            yy, xx = np.mgrid[0:h, 0:w]
            xr = (xx - cx) * np.cos(ang) + (yy - cy) * np.sin(ang)
            yr = -(xx - cx) * np.sin(ang) + (yy - cy) * np.cos(ang)
            dist2 = (xr / (length / 2)) ** 2 + (yr / max(ry, 1e-3)) ** 2
            mask = np.clip(1.0 - dist2, 0.0, 1.0) ** 2
        elif class_id == 4:  # 裂纹
            mask = _crack_mask(rng, h, w, cx, cy, rng.uniform(40.0, 150.0))
        else:  # 5 咬边
            edge = rng.choice([edge_top, edge_bot])
            mask = _undercut_mask(rng, h, w, cx, edge, rng.uniform(40.0, 140.0))

        x1, y1, x2, y2 = _bbox_of(mask)
        if (x2 - x1) >= 2 and (y2 - y1) >= 2:
            return mask, (x1, y1, x2, y2)
    # 兜底：全重试失败则返回单像素掩膜（保证不抛错）
    mask = np.zeros((h, w), np.float32)
    mask[h // 2, w // 2] = 1.0
    return mask, (w // 2, h // 2, w // 2 + 1, h // 2 + 1)


def _apply(img: np.ndarray, mask: np.ndarray, strength: float) -> np.ndarray:
    """暗化叠加缺陷（X 光底片缺陷呈暗色）。"""
    dark = img.astype(np.float32) * (1.0 - strength)
    blend = img.astype(np.float32) * (1.0 - mask) + dark * mask
    return np.clip(blend, 0, 255).astype(np.uint8)


def synthesize_image(
    rng_seed: int,
    h: int,
    w: int,
    n_defects: int,
    class_weights: dict[int, float] | None = None,
) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
    """合成一张带缺陷的模拟底片。

    返回 (image_gray_uint8, yolo_labels)；
    yolo_labels: [(class_id, cx, cy, w, h)]，坐标已归一化 [0,1]。
    """
    rng = np.random.default_rng(rng_seed)
    cw = class_weights or _DEFAULT_CLASS_WEIGHTS
    img = _make_background(rng, h, w)
    labels: list[tuple[int, float, float, float, float]] = []
    for _ in range(n_defects):
        class_id = int(rng.choice(list(cw.keys()), p=_normalize_weights(cw)))
        strength = rng.uniform(0.35, 0.6)  # 暗化强度（0.35~0.6，可见但非全黑）
        mask, (x1, y1, x2, y2) = _synthesize_one(rng, h, w, class_id)
        img = _apply(img, mask, strength)
        bw = max(x2 - x1, 2)
        bh = max(y2 - y1, 2)
        labels.append(
            (
                class_id,
                (x1 + bw / 2) / w,
                (y1 + bh / 2) / h,
                bw / w,
                bh / h,
            )
        )
    # 整体微模糊 + 轻噪声，让合成缺陷与底片融为一体
    img = cv2.GaussianBlur(img, (3, 3), 0.5)
    return img, labels


def _normalize_weights(cw: dict[int, float]) -> list[float]:
    """类别权重 → 归一化概率（零/负权重防除零）。"""
    vals = np.array([max(v, 1e-6) for v in cw.values()], dtype=np.float64)
    return (vals / vals.sum()).tolist()


def generate_synthetic_set(
    out_dir: str | Path,
    n_images: int = 200,
    size: tuple[int, int] = (640, 480),
    per_image: tuple[int, int] = (1, 4),
    seed: int = 42,
    class_weights: dict[int, float] | None = None,
) -> int:
    """批量合成缺陷底片 → out_dir/{images,labels}（YOLO）。

    返回生成的图像数。cv2 中文路径安全：np.fromfile/imencode 通道。
    """
    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    lbl_dir = out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    w, h = size
    made = 0
    for i in range(n_images):
        n_def = int(rng.integers(per_image[0], per_image[1] + 1))
        image, labels = synthesize_image(rng.integers(0, 2**31 - 1), h, w, n_def, class_weights)
        stem = f"gl_{seed}_{i:05d}"
        ok, buf = cv2.imencode(".png", image)
        if not ok:
            continue
        (img_dir / f"{stem}.png").write_bytes(buf.tobytes())
        lines = [f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for c, cx, cy, bw, bh in labels]
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        made += 1
    print(
        f"[defect_synth] 合成 {made} 张 → {out_dir}（类别权重 {class_weights or _DEFAULT_CLASS_WEIGHTS}）"
    )
    return made


def main() -> None:
    ap = argparse.ArgumentParser(description="GLASS 式缺陷合成（P0-B）")
    ap.add_argument("--n", type=int, default=200, help="合成图数")
    ap.add_argument("--out", default="data/training/raw/synthetic", help="输出 YOLO 目录")
    ap.add_argument("--size", default="640x480", help="图像尺寸 WxH")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    w, h = (int(v) for v in args.size.lower().split("x"))
    n = generate_synthetic_set(args.out, n_images=args.n, size=(w, h), seed=args.seed)
    print(f"generated {n}")


if __name__ == "__main__":
    main()
