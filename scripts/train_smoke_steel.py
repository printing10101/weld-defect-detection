"""Smoke 训练：steel + synthetic 数据可用性验证（用户要求）。

目的：验证 data/training（steel 2699 + synthetic 300）能正常训练，且裂纹（class 4）有检出。
非调优：30 epoch 够看趋势即可；不追求精度。
"""

from __future__ import annotations

from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    print(
        f"cuda={torch.cuda.is_available()} n={torch.cuda.device_count()} {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}"
    )
    m = YOLO(str(ROOT / "data" / "real_label" / "yolo11n.pt"))
    m.train(
        data=str(ROOT / "data" / "training" / "data.yaml"),
        epochs=30,
        imgsz=640,
        batch=16,
        device=0,
        workers=2,
        project=str(ROOT / "runs" / "smoke_steel"),
        name="steel_synth",
        exist_ok=True,
        verbose=False,
        cache=False,
    )
    print("TRAIN_DONE")


if __name__ == "__main__":
    main()
