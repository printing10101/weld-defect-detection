# -*- coding: utf-8 -*-
"""正式训练 Step1：稀有类 Copy-Paste 均衡增强。

对 train 集中最稀有的 4 类（夹渣1/未焊透2/裂纹4/咬边5）做 Copy-Paste，
输出到 data/training/raw/rare_aug/，随后合并回 train 集。
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.training.augment import generate_rare_copy_paste  # noqa: E402


def main() -> None:
    train = ROOT / "data" / "training" / "train"
    out = ROOT / "data" / "training" / "raw" / "rare_aug"
    n = generate_rare_copy_paste(
        source_dir=train,
        out_dir=out,
        rare_classes={1, 2, 4, 5},  # 夹渣/未焊透/裂纹/咬边（最缺）
        per_source=6,
        seed=42,
    )
    print(f"[step1] 生成 {n} 张增强图")

    # 合并回 train 集（增强图只进 train，不污染 val/test）
    dst_img = train / "images"
    dst_lbl = train / "labels"
    copied = 0
    for img in sorted((out / "images").glob("*.png")):
        shutil.copy(img, dst_img / img.name)
        lbl = out / "labels" / (img.stem + ".txt")
        if lbl.exists():
            shutil.copy(lbl, dst_lbl / (img.stem + ".txt"))
        copied += 1
    print(f"[step1] 合并 {copied} 张到 train 集")
    print(f"[step1] train 图像总数: {len(list(dst_img.glob('*')))}")


if __name__ == "__main__":
    main()
