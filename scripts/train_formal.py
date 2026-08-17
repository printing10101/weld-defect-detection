# -*- coding: utf-8 -*-
"""正式训练：steel + synthetic + 稀有类 Copy-Paste 均衡（用户要求）。

与 smoke 的区别：
- 数据：train 4641 张（含 rare copy-paste 均衡增强，稀有类×8）
- epoch：120（patience 25 早停）
- 增强参数：方案A 验证过的配置（mosaic/mixup/翻转/hsv）
- 评估：训练后自动 val + test 集专项（裂纹/稀有类）
"""
from __future__ import annotations

from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    print(f"cuda={torch.cuda.is_available()} {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
    m = YOLO(str(ROOT / "data" / "real_label" / "yolo11n.pt"))
    m.train(
        data=str(ROOT / "data" / "training" / "data.yaml"),
        epochs=120,
        patience=25,
        imgsz=640,
        batch=16,
        device=0,
        workers=2,
        project=str(ROOT / "runs" / "train_steel_v1"),
        name="steel_balanced",
        exist_ok=True,
        verbose=False,
        cache=False,
        # 方案A 验证过的增强/优化参数
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
        lr0=1e-3,
        lrf=1e-2,
    )
    print("TRAIN_DONE")


if __name__ == "__main__":
    main()
