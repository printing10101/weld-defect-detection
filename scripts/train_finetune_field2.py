"""第二轮现场微调：数据 1272 张（现场+稀有增强+气孔增强），freeze=0 全量适配，lr=5e-4。

从 steel_balanced 干净基底重新微调（第二轮数据更丰富，避免累积过拟合）。
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    base = ROOT / "runs" / "train_steel_v1" / "steel_balanced" / "weights" / "best.pt"
    print(f"基底: {base}")
    m = YOLO(str(base))
    m.train(
        data=str(ROOT / "data" / "real_label" / "ft_data.yaml"),
        epochs=100,
        patience=25,
        imgsz=640,
        batch=8,
        device=0,
        workers=2,
        project=str(ROOT / "runs" / "ft_field"),
        name="steel_field_ft2",
        exist_ok=True,
        verbose=False,
        cache=False,
        freeze=0,  # 全量微调（现场风格差异大）
        lr0=5e-4,  # 略高学习率
        lrf=1e-2,
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
    print("FT2_DONE")


if __name__ == "__main__":
    main()
