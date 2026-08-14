"""训练入口（§17，M4 里程碑实现 / M4b）。

用法：
  python -m backend.models.train --epochs 80 --imgsz 640 --model yolov8m.pt
  python -m backend.models.train --smoke        # 合成数据 2 epoch 自检

流程：
1. 数据集就绪：data/training/data.yaml 由 dataset_builder 生成；若缺失则尝试从
   data/external/swrd（download_swrd --ingest）与 data/training/raw/user 重建。
2. Ultralytics YOLO 训练：预训练主干（yolov8m.pt）+ 强增强 + 冻结/微调策略
   （详见 DATA_LICENSE.md 与 ADR-002/010）。
3. 导出 best.pt → models/weights/best.onnx，供 ONNX Runtime 部署推理。

合规：训练数据须为 CC BY 4.0（SWRD）等可训练+比赛的数据集；用户 165 张 定检
图作域适应微调集；合成数据作增强。比赛/论文须按 DATA_LICENSE.md 署名。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from backend.training import dataset_builder


def _resolve_pretrained(model: str) -> str:
    # 允许传入架构名（如 yolov8m.yaml）从零训练，或权重名（yolov8m.pt）迁移学习
    return model


def _unique_run_name(project: str, name: str) -> str:
    """避免复用已存在的 run 目录（ultralytics 会 unlink 旧 results.csv，
    在本沙箱中 safe-delete 拦截删除导致训练中断）。"""
    base = Path(project) / name
    if not base.exists():
        return name
    return f"{name}_{int(time.time())}"


def train(
    model: str = "yolov8m.pt",
    epochs: int = 80,
    imgsz: int = 640,
    batch: int = 16,
    project: str = "data/runs",
    name: str = "weld_defect",
    export_onnx: bool = True,
) -> Path:
    data_yaml = dataset_builder.ensure_dataset()
    from ultralytics import YOLO

    run_name = _unique_run_name(project, name)
    m = YOLO(_resolve_pretrained(model))
    m.train(
        data=str(data_yaml.resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=run_name,
        project=project,
        exist_ok=True,
        # 强增强（小样本抗过拟合）：mosaic/hsv/flip 由 ultralytics 默认开启；
        # 自定义 X 光增强见 training/augment.build_xray_augment（离线预增强用）。
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        verbose=True,
    )
    # ultralytics 实际落盘含 task 子目录（runs/detect/...），以 trainer.save_dir 为准
    save_dir = Path(m.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"训练未产出 {best}（save_dir={save_dir}）")
    out_onnx = Path("models/weights/best.onnx")
    if export_onnx:
        out_onnx.parent.mkdir(parents=True, exist_ok=True)
        # torch>=2.9 默认 dynamo 导出器在部分模型上会卡死；强制 legacy 导出器
        import torch as _torch

        _orig_export = _torch.onnx.export

        def _legacy_export(*a, **k):
            k["dynamo"] = False
            return _orig_export(*a, **k)

        _torch.onnx.export = _legacy_export
        try:
            # 显式从 best.pt 导出，避免从初始权重导出导致文件名错乱
            ex = YOLO(str(best))
            ex.export(format="onnx", imgsz=imgsz)
        finally:
            _torch.onnx.export = _orig_export
        exported = save_dir / "weights" / "best.onnx"
        if exported.exists():
            out_onnx.write_bytes(exported.read_bytes())
            print(f"[train] 已导出 ONNX → {out_onnx}")
    print(f"[train] 完成，最佳权重 → {best}")
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="焊缝缺陷检测训练（M4b）")
    ap.add_argument("--model", default="yolov8m.pt", help="预训练权重或架构 yaml")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="weld_defect")
    ap.add_argument("--no-export", action="store_true", help="不导出 ONNX")
    ap.add_argument("--smoke", action="store_true", help="合成数据快速自检（≈2 epoch）")
    args = ap.parse_args()

    if args.smoke:
        from backend.training.smoke_test import main as smoke_main

        smoke_main()
        return
    train(
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        export_onnx=not args.no_export,
    )


if __name__ == "__main__":
    main()
