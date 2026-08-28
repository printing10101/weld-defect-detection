"""现场数据微调：steel_balanced 预训练基底 → 现场标注（data/real_label）。

- 基底：runs/train_steel_v1/steel_balanced/weights/best.pt（已见 2699 张 X 光缺陷）
- 数据：ft_data.yaml（train=现场102+稀有增强216，val=现场val 25）
- 低 lr 微调：lr0=1e-4，epochs=80，patience=25，batch=8
- 目的：适配现场成像风格（域偏移），保留 steel 学到的缺陷形状先验
"""

from __future__ import annotations

from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    base = ROOT / "runs" / "train_steel_v1" / "steel_balanced" / "weights" / "best.pt"
    print(f"基底: {base} 存在={base.exists()}")
    print(f"cuda={torch.cuda.is_available()}")
    m = YOLO(str(base))
    m.train(
        data=str(ROOT / "data" / "real_label" / "ft_data.yaml"),
        epochs=80,
        patience=25,
        imgsz=640,
        batch=8,
        device=0,
        workers=2,
        project=str(ROOT / "runs" / "ft_field"),
        name="steel_field_ft",
        exist_ok=True,
        verbose=False,
        cache=False,
        lr0=1e-4,
        lrf=1e-2,
        # 现场数据小：轻增强 + 冻结前 8 层（保护 steel 特征）
        freeze=8,
        mosaic=0.5,
        mixup=0.1,
        degrees=5.0,
        translate=0.05,
        scale=0.2,
        flipud=0.0,
        fliplr=0.5,
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.2,
    )
    print("FT_DONE")


if __name__ == "__main__":
    main()
