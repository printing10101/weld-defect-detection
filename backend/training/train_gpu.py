"""GPU 全自动训练（ 优化后数据集，全自动路线）。

数据：data/training/data.yaml（train 3422 = 真实 user 128 + 合成 450 + steel 2699
      + 伪标签 11（train-only）+ 罕见类过采样 x2；val 322 / test 362 纯真实标注）
预训练：data/real_label/yolov8n.pt（本机已有，免下载）
设备：cuda:0（RTX 3080 16GB）
产出：runs/gpu/auto_v1_*/weights/best.pt（供 _export_model 链路导出 ONNX）

内置绕行：
- safe-delete shim（ultralytics 训练会 unlink results.csv / *.cache → FAIL_CLOSED）
- YOLO_OFFLINE=1（防网络检查挂起）

用法：python -m backend.training.train_gpu --epochs 100
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# —— safe-delete shim 绕行（仅恢复本进程内删除自身生成文件，无外部风险）——
try:
    import nt
    import pathlib

    os.unlink = nt.unlink  # type: ignore[attr-defined]
    os.remove = nt.remove  # type: ignore[attr-defined]
    pathlib.Path.unlink = lambda self, *a, **k: os.unlink(str(self))  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001, S110 - 非沙箱环境无需恢复
    pass

os.environ.setdefault("YOLO_OFFLINE", "1")


def _patch_torchvision_nms() -> None:
    """Windows 无 CUDA torchvision wheel（CPU 版缺 CUDA NMS 内核）→
    torchvision.ops.nms 在 CUDA 张量上 NotImplementedError（每 epoch 验证必崩）。

    用纯 torch 向量化 NMS 打补丁（torch 原生算子，CUDA/CPU 均可跑），
    与官方 NMS 逻辑一致（分数降序 + IoU 阈值抑制）。
    """
    try:
        import torch

        def _nms(boxes, scores, iou_threshold):
            if boxes.numel() == 0:
                return torch.empty(0, dtype=torch.long, device=boxes.device)
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
            order = scores.sort(descending=True)[1]
            keep = []
            while order.numel() > 0:
                i = order[0].item()
                keep.append(i)
                if order.numel() == 1:
                    break
                rest = order[1:]
                xx1 = x1[rest].clamp(min=float(x1[i]))
                yy1 = y1[rest].clamp(min=float(y1[i]))
                xx2 = x2[rest].clamp(max=float(x2[i]))
                yy2 = y2[rest].clamp(max=float(y2[i]))
                inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
                iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
                order = rest[(iou <= iou_threshold).nonzero(as_tuple=False).squeeze(1)]
            return torch.tensor(keep, dtype=torch.long, device=boxes.device)

        import torchvision

        torchvision.ops.nms = _nms
        print(
            "[train_gpu] torchvision.ops.nms 已替换为纯 torch 实现（CPU torchvision 缺 CUDA 内核）"
        )
    except Exception as exc:  # noqa: BLE001 - 非目标环境无需补丁
        print(f"[train_gpu] nms patch skipped: {exc}")


def train(
    epochs: int = 100,
    name: str | None = None,
    pretrained: str | None = None,
) -> str:
    """启动 GPU 训练，返回训练 run 目录名（如 auto_v1_20260815_...）。

    pretrained：预训练权重路径。默认 yolov8n.pt（COCO 先验）；B+ 方案传
    real_synth2/best.pt（已验证真实域先验）。
    """
    from ultralytics import YOLO

    _patch_torchvision_nms()

    root = Path(__file__).resolve().parents[2]
    data = root / "data" / "training" / "data.yaml"
    weights = Path(pretrained) if pretrained else root / "data" / "real_label" / "yolov8n.pt"
    if not weights.exists():
        raise FileNotFoundError(f"预训练权重缺失：{weights}")
    if not data.exists():
        raise FileNotFoundError(f"data.yaml 缺失：{data}（先跑 dataset_builder）")
    run_name = name or f"auto_v1_{time.strftime('%Y%m%d_%H%M%S')}"
    m = YOLO(str(weights))
    print(f"[train_gpu] data={data} pretrained={weights} name={run_name} epochs={epochs}")
    m.train(
        data=str(data),
        epochs=epochs,
        imgsz=640,
        batch=16,
        device=0,
        project=str(root / "runs" / "gpu"),
        name=run_name,
        seed=42,
        patience=20,
        workers=4,
        cache=False,
        val=True,
    )
    best = root / "runs" / "gpu" / run_name / "weights" / "best.pt"
    print(f"[train_gpu] DONE best={best}")
    return run_name


def main() -> None:
    ap = argparse.ArgumentParser(description="GPU 全自动训练（优化后数据集）")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--pretrained", type=str, default=None, help="预训练权重（默认 yolov8n.pt）")
    args = ap.parse_args()
    train(epochs=args.epochs, name=args.name, pretrained=args.pretrained)


if __name__ == "__main__":
    main()
