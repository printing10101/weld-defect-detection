"""现场微调 Step1：准备微调数据。

- 复制 train.txt 的 102 张现场图 + 标注到 ft_train/
- 对现场稀有类（夹渣1/未焊透2/未熔合3/裂纹4/咬边5）做 Copy-Paste ×12（方案A 策略）
- 合并增强图到 ft_train/
- 写 ft_data.yaml（train=ft_train/images, val=val.txt）
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.training.augment import generate_rare_copy_paste


def main() -> None:
    rl = ROOT / "data" / "real_label"
    ft = rl / "ft_train"
    ft_img = ft / "images"
    ft_lbl = ft / "labels"
    ft_img.mkdir(parents=True, exist_ok=True)
    ft_lbl.mkdir(parents=True, exist_ok=True)

    # 1) 复制 train.txt 的图 + 标注
    copied = 0
    for line in (rl / "train.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        img = Path(line.replace("\\", "/"))
        if not img.is_absolute():
            img = rl / "images" / img.name
        if not img.exists():
            continue
        shutil.copy(img, ft_img / img.name)
        lbl = rl / "labels" / (img.stem + ".txt")
        if lbl.exists():
            shutil.copy(lbl, ft_lbl / (img.stem + ".txt"))
        copied += 1
    print(f"[ft] 复制现场 train 图 {copied} 张 → {ft_img}")

    # 2) 现场稀有类 Copy-Paste ×12（方案A 策略）
    aug = rl / "ft_aug"
    n = generate_rare_copy_paste(
        source_dir=ft,
        out_dir=aug,
        rare_classes={1, 2, 3, 4, 5},
        per_source=12,
        seed=42,
    )
    print(f"[ft] 稀有类增强 {n} 张")

    # 3) 合并增强图到 ft_train
    merged = 0
    for img in sorted((aug / "images").glob("*.png")):
        shutil.copy(img, ft_img / img.name)
        lbl = aug / "labels" / (img.stem + ".txt")
        if lbl.exists():
            shutil.copy(lbl, ft_lbl / (img.stem + ".txt"))
        merged += 1
    print(f"[ft] 合并增强 {merged} 张 → ft_train 共 {len(list(ft_img.glob('*')))} 张")

    # 4) ft_data.yaml
    yaml_text = f"""# 现场微调数据（自动生成）：train=现场102+稀有增强，val=现场val.txt
path: {str(rl).replace(chr(92), "/")}
train: ft_train/images
val: val.txt
nc: 6
names:
  0: qikong
  1: jiazha
  2: wei_hantou
  3: wei_ronghe
  4: lie_wen
  5: yao_bian
"""
    (rl / "ft_data.yaml").write_text(yaml_text, encoding="utf-8")
    print(f"[ft] 已写 {rl / 'ft_data.yaml'}")


if __name__ == "__main__":
    main()
