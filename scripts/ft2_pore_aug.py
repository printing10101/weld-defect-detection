"""第二轮微调数据：给 ft_train 补气孔 Copy-Paste 增强（val 90 框里 86 是气孔）。

用 generate_copy_paste（通用，主要粘贴气孔）对 ft_train 生成 per_image=3 张增强，
合并回 ft_train，随后第二轮微调（freeze=0, lr=5e-4）。
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.training.augment import generate_copy_paste


def main() -> None:
    ft = ROOT / "data" / "real_label" / "ft_train"
    aug = ROOT / "data" / "real_label" / "ft_pore_aug"
    n = generate_copy_paste(source_dir=ft, out_dir=aug, per_image=3, seed=7)
    print(f"[ft2] 气孔/通用增强 {n} 张")

    dst_img = ft / "images"
    dst_lbl = ft / "labels"
    merged = 0
    for img in sorted((aug / "images").glob("*.png")):
        shutil.copy(img, dst_img / img.name)
        lbl = aug / "labels" / (img.stem + ".txt")
        if lbl.exists():
            shutil.copy(lbl, dst_lbl / (img.stem + ".txt"))
        merged += 1
    print(f"[ft2] 合并 {merged} 张 → ft_train 共 {len(list(dst_img.glob('*')))} 张")


if __name__ == "__main__":
    main()
