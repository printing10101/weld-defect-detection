"""用  基线检测器（BlobDetector）为用户 165 张 定检 图生成预标注（warm start）。

目的：加速人工标注。BlobDetector 只能定位暗斑（class 占位 0=气孔），
不能分类——产出的 YOLO txt 给出缺陷**位置**初稿，人工在 Label Studio 中
校正类别并补漏。最终校正后的 labels 落到 data/training/raw/user/labels 即被
dataset_builder 自动纳入训练集（域适应微调集）。

用法：
  python -m backend.training.prelabel_user
  python -m backend.training.prelabel_user --src 图片/定检 --dst data/training/raw/user
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from backend.domain.detect.blob_detector import BlobConfig, BlobDetector
from backend.infra.config import load_config


def prelabel(src: Path, dst: Path, conf: float = 0.1, iou: float = 0.5) -> int:
    cfg = load_config().detect
    det = BlobDetector(
        BlobConfig(
            min_area_px=cfg.min_area_px,
            max_area_px=cfg.max_area_px,
            min_size_px=cfg.min_size_px,
            noise_sigma_ratio=cfg.noise_sigma_ratio,
            abs_threshold=cfg.abs_threshold,
            dark_only=cfg.dark_only,
        )
    )
    out_img = dst / "images"
    out_lbl = dst / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    imgs = [p for p in sorted(src.rglob("*")) if p.suffix.lower() in exts]
    n = 0
    for img_p in imgs:
        gray = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        h, w = gray.shape
        dets = det.infer(gray, conf=conf, iou=iou)
        lines = []
        for d in dets:
            x, y, bw, bh = d.bbox.x, d.bbox.y, d.bbox.w, d.bbox.h
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            # 占位类别 0（气孔）；人工在 Label Studio 中校正为真实类别
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        cv2.imwrite(str(out_img / img_p.name), gray)
        (out_lbl / (img_p.stem + ".txt")).write_text("\n".join(lines))
        n += 1
        if dets:
            print(f"  {img_p.name}: {len(dets)} 个候选框（位置初稿）")
    print(f"[prelabel] 完成 {n} 张 → {dst}（类别待人工校正）")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="定检图预标注（BlobDetector warm start）")
    ap.add_argument("--src", default="图片/定检", help="原始 定检 图目录")
    ap.add_argument("--dst", default="data/training/raw/user", help="输出 YOLO 目录")
    args = ap.parse_args()
    prelabel(Path(args.src), Path(args.dst))


if __name__ == "__main__":
    main()
