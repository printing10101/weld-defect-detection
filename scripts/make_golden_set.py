# !/usr/bin/env python3
"""生成确定性合成 Golden Set。

输出目录：data/eval/golden/{images,labels}
- images/*.png：512x512 灰度合成射线底片风（暗缺陷团块）
- labels/*.txt：YOLO 格式真值（"<class_id> <cx> <cy> <w> <h>" 归一化，中心 xy + wh）
- _META.json：生成参数 + 合规声明

设计约束（对齐 #4 评估回归门禁目标）：
- 确定性：固定 seed，可完全复现；任何文件增删改都会改变 harness 内容指纹，
  从而在评估记录中留下 Golden Set 版本痕迹（禁止用于训练，版本变更需显式记录）。
- 零真实数据：纯合成暗缺陷，无真实患者/工业机密影像，无标准授权依赖，
  规避"真实授权文本缺失"约束，同时提供可复现回归基准。
- 多类覆盖（v2）：覆盖 NB/T47013 五类 + 咬边共 6 类的合成形态与真值标签，
  使"标签解析 → 逐类指标 → 量化 → 评级 → 报告 → 漂移 → 跟踪"全链路获得多类回归覆盖。
  CI 评估门禁本身用 BlobDetector（仅检暗缺陷、统一判 class 0），故逐类>0 的 mAP
  不来自门禁；多类真值用于锁定标签/量化/评级/报告的解析与聚合不被静默破坏。
- 主信号保持：≥80% 图像含 class 0（POROSITY）暗团块，使 blob 门禁召回/mAP 仍有意义。
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

SEED = 42
N_IMAGES = 36
SIZE = 512
BG = 150  # 母材背景灰度
DEFECT_VAL = 55  # 缺陷团块灰度（明显暗于背景）
POROSITY_RATIO = 0.8  # ≥此比例图像含 class 0（保证 blob 门禁主信号）

CLASS_NAMES = [
    "POROSITY",  # 0 气孔
    "SLAG",  # 1 夹渣
    "INCOMPLETE_PENETRATION",  # 2 未焊透
    "LACK_OF_FUSION",  # 3 未熔合
    "CRACK",  # 4 裂纹
    "UNDERCUT",  # 5 咬边
]

# 仓库根：scripts/make_golden_set.py -> parents[1]
ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "data" / "eval" / "golden"


def _base_image(rng: np.random.Generator) -> np.ndarray:
    """母材背景 + 轻微梯度 + 高斯噪声（确定性）。"""
    img = np.full((SIZE, SIZE), BG, dtype=np.int16)
    grad = np.linspace(0, 18, SIZE, dtype=np.int16)
    img = img + grad[None, :]
    noise = rng.normal(0, 4.0, (SIZE, SIZE)).astype(np.int16)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def _draw_porosity(
    img: np.ndarray, rng: np.random.Generator, val: int
) -> tuple[int, int, int, int]:
    """气孔：1–3 个聚集暗椭圆团块。返回绝对像素 bbox (x,y,w,h)。"""
    cx0 = int(rng.integers(40, SIZE - 40))
    cy0 = int(rng.integers(40, SIZE - 40))
    x0, y0, x1, y1 = SIZE, SIZE, 0, 0
    for _ in range(int(rng.integers(1, 4))):
        cx = cx0 + int(rng.integers(-12, 12))
        cy = cy0 + int(rng.integers(-12, 12))
        rx = int(rng.integers(5, 12))
        ry = int(rng.integers(5, 12))
        cv2.ellipse(
            img, (cx, cy), (rx, ry), int(rng.integers(0, 180)), 0, 360, color=val, thickness=-1
        )
        x0, y0 = min(x0, cx - rx), min(y0, cy - ry)
        x1, y1 = max(x1, cx + rx), max(y1, cy + ry)
    return (max(0, x0), max(0, y0), min(SIZE, x1) - max(0, x0), min(SIZE, y1) - max(0, y0))


def _draw_slag(img: np.ndarray, rng: np.random.Generator, val: int) -> tuple[int, int, int, int]:
    """夹渣：不规则暗点簇。"""
    cx = int(rng.integers(40, SIZE - 40))
    cy = int(rng.integers(40, SIZE - 40))
    x0, y0, x1, y1 = SIZE, SIZE, 0, 0
    for _ in range(int(rng.integers(5, 10))):
        ox = int(rng.integers(-15, 15))
        oy = int(rng.integers(-15, 15))
        r = int(rng.integers(2, 6))
        cv2.circle(img, (cx + ox, cy + oy), r, color=val, thickness=-1)
        x0, y0 = min(x0, cx + ox - r), min(y0, cy + oy - r)
        x1, y1 = max(x1, cx + ox + r), max(y1, cy + oy + r)
    return (max(0, x0), max(0, y0), min(SIZE, x1) - max(0, x0), min(SIZE, y1) - max(0, y0))


def _draw_incomplete_penetration(
    img: np.ndarray, rng: np.random.Generator, val: int
) -> tuple[int, int, int, int]:
    """未焊透：沿焊缝中心线的低矮暗带。"""
    cyc = SIZE // 2 + int(rng.integers(-20, 20))
    half_h = int(rng.integers(6, 14))
    x0 = int(rng.integers(40, SIZE // 2))
    x1 = x0 + int(rng.integers(80, 200))
    cv2.rectangle(img, (x0, cyc - half_h), (x1, cyc + half_h), color=val, thickness=-1)
    return (x0, cyc - half_h, x1 - x0, 2 * half_h)


def _draw_lack_of_fusion(
    img: np.ndarray, rng: np.random.Generator, val: int
) -> tuple[int, int, int, int]:
    """未熔合：线性暗条纹。"""
    x0 = int(rng.integers(40, SIZE - 80))
    y0 = int(rng.integers(60, SIZE - 60))
    dx = int(rng.integers(60, 160))
    dy = int(rng.integers(-40, 40))
    x1, y1 = x0 + dx, y0 + dy
    cv2.line(img, (x0, y0), (x1, y1), color=val, thickness=int(rng.integers(2, 5)))
    t = 3
    bx0, by0 = min(x0, x1), min(y0, y1)
    bx1, by1 = max(x0, x1), max(y0, y1)
    return (bx0 - t, by0 - t, (bx1 - bx0) + 2 * t, (by1 - by0) + 2 * t)


def _draw_crack(img: np.ndarray, rng: np.random.Generator, val: int) -> tuple[int, int, int, int]:
    """裂纹：细锯齿状暗折线。"""
    x = int(rng.integers(40, SIZE - 100))
    y = int(rng.integers(60, SIZE - 60))
    pts = [(x, y)]
    for _ in range(int(rng.integers(3, 6))):
        x += int(rng.integers(15, 40))
        y += int(rng.integers(-20, 20))
        pts.append((x, y))
    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i + 1], color=val, thickness=1)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    t = 4
    return (min(xs) - t, min(ys) - t, (max(xs) - min(xs)) + 2 * t, (max(ys) - min(ys)) + 2 * t)


def _draw_undercut(
    img: np.ndarray, rng: np.random.Generator, val: int
) -> tuple[int, int, int, int]:
    """咬边：近边缘的浅凹暗弧。"""
    cx = int(rng.integers(80, SIZE - 80))
    cy = int(rng.integers(SIZE - 60, SIZE - 30))
    r = int(rng.integers(40, 90))
    cv2.ellipse(
        img, (cx, cy), (r, r // 3), 0, 200, 340, color=val, thickness=int(rng.integers(2, 4))
    )
    return (cx - r, cy - r // 3, 2 * r, 2 * (r // 3))


RENDERERS = {
    0: _draw_porosity,
    1: _draw_slag,
    2: _draw_incomplete_penetration,
    3: _draw_lack_of_fusion,
    4: _draw_crack,
    5: _draw_undercut,
}


def _make_image(
    rng: np.random.Generator, idx: int
) -> tuple[np.ndarray, list[tuple[int, int, int, int, int]]]:
    """生成单张合成底片 + 多类缺陷真值框（class_id, x, y, w, h 绝对像素）。"""
    img = _base_image(rng)
    n_def = int(rng.integers(1, 4))
    chosen = list(rng.choice(range(6), size=n_def, replace=False))
    # 主信号：≥POROSITY_RATIO 的图像含 class 0
    if rng.random() < POROSITY_RATIO and 0 not in chosen:
        chosen[0] = 0
    targets: list[tuple[int, int, int, int, int]] = []
    for cid in chosen:
        x, y, w, h = RENDERERS[cid](img, rng, DEFECT_VAL)
        targets.append((cid, x, y, w, h))
    return img, targets


def _write_yolo(path: Path, targets: list[tuple[int, int, int, int, int]]) -> None:
    lines = []
    for cid, x, y, w, h in targets:
        cx = (x + w / 2) / SIZE
        cy = (y + h / 2) / SIZE
        nw = w / SIZE
        nh = h / SIZE
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


def main() -> None:
    rng = np.random.default_rng(SEED)
    (GOLDEN / "images").mkdir(parents=True, exist_ok=True)
    (GOLDEN / "labels").mkdir(parents=True, exist_ok=True)
    class_counts = {c: 0 for c in range(6)}
    for i in range(N_IMAGES):
        img, targets = _make_image(rng, i)
        stem = f"synthetic_{i:03d}"
        # cv2.imwrite 在 Windows 上对含非 ASCII 路径会静默失败 → imencode + 写字节
        png_path = GOLDEN / "images" / f"{stem}.png"
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            raise RuntimeError(f"imencode 失败: {png_path}")
        png_path.write_bytes(buf.tobytes())
        _write_yolo(GOLDEN / "labels" / f"{stem}.txt", targets)
        for cid, *_ in targets:
            class_counts[cid] += 1
    meta = {
        "synthetic": True,
        "version": 2,
        "seed": SEED,
        "n_images": N_IMAGES,
        "size": SIZE,
        "n_classes": 6,
        "classes": CLASS_NAMES,
        "class_box_counts": class_counts,
        "porosity_ratio_target": POROSITY_RATIO,
        "gate_detector": "BlobDetector(dark_only=True) — 仅判 class 0；逐类>0 mAP 不来自门禁",
        "provenance": (
            "合成缺陷图（暗团块/条纹/弧线），无真实患者/工业机密影像，无标准授权依赖；"
            "仅作评估回归基准，禁止用于训练。"
        ),
        "generated_by": "scripts/make_golden_set.py",
    }
    (GOLDEN / "_META.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Golden Set v2 written to {GOLDEN} ({N_IMAGES} images, classes={class_counts})")


if __name__ == "__main__":
    main()
