"""伪标签自动晋升（替代人工标注，全自动扩数据）。

背景：用户不采用人工核对流程。本脚本把 active_prelabel 产出的预标注中
**高置信**候选自动晋升为正式训练标注（pseudo-labeling / self-training）：

1. 对 data/real_label 未标注图用部署模型重新推理，拿到逐框 score/uncertainty；
2. 采纳规则（安全关键系统，宁保守勿引入噪声）：
   - 框置信 score ≥ score_thr（默认 0.6，高置信）；
   - 且 uncertainty < unc_thr（默认 0.5，非临界——临界缺陷必须人工，绝不自动采纳）；
   - 图内至少 1 个采纳框才晋升该图，只写采纳框；
3. 输出到 data/training/raw/pseudo/{images,labels}（**train-only 源**，不污染
   val/test 评估）；绝不覆盖用户真实标注（labels/ 已有文件跳过）。

用法：
  python -m backend.training.auto_pseudo_label --real data/real_label \
    --out data/training/raw/pseudo --score 0.6 --unc 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from backend.domain.dto import Detection
from backend.infra.config import load_config, resolve_config_path
from backend.training.active_prelabel import _build_detector, _read_gray
from backend.training.dataset_builder import build_dataset

_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _to_yolo_line(d: Detection, w: int, h: int) -> str:
    """Detection → YOLO normalized 标注行（坐标裁剪到 [0,1]）。"""
    bw = max(0.0, min(d.bbox.w / max(w, 1e-9), 1.0))
    bh = max(0.0, min(d.bbox.h / max(h, 1e-9), 1.0))
    cx = max(0.0, min((d.bbox.x + d.bbox.w / 2) / max(w, 1e-9), 1.0))
    cy = max(0.0, min((d.bbox.y + d.bbox.h / 2) / max(h, 1e-9), 1.0))
    return f"{int(d.class_id.value)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def promote_high_conf(
    real_root: Path,
    out_dir: Path,
    detector,
    *,
    conf: float = 0.3,
    iou: float = 0.5,
    class_conf: dict[int, float] | None = None,
    score_thr: float = 0.6,
    unc_thr: float = 0.5,
) -> dict:
    """把未标注图中高置信检出晋升为训练标注（pseudo 源）。

    返回统计 {promoted_images, promoted_boxes, skipped_existing, filtered_images}。
    """
    real_root = Path(real_root)
    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    lbl_dir = out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    lbl_real = real_root / "labels"

    stats = {"promoted_images": 0, "promoted_boxes": 0, "skipped_existing": 0, "filtered_images": 0}
    # 遍历全部图（含已标注），显式跳过已有真实标注——绝不覆盖
    for img_p in sorted((real_root / "images").iterdir()):
        if img_p.suffix.lower() not in _EXTS:
            continue
        if (lbl_real / f"{img_p.stem}.txt").exists():
            stats["skipped_existing"] += 1
            continue  # 已有真实标注，绝不覆盖
        gray = _read_gray(img_p)
        if gray is None:
            continue
        h, w = gray.shape
        dets = detector.infer(gray, conf=conf, iou=iou, class_conf=class_conf)
        accepted = [d for d in dets if d.score >= score_thr and d.uncertainty < unc_thr]
        if not accepted:
            stats["filtered_images"] += 1
            continue
        # 复制图像（中文路径安全：np.fromfile 读 + imencode 写）
        arr = np.fromfile(str(img_p), dtype=np.uint8)
        (img_dir / img_p.name).write_bytes(arr.tobytes())
        lines = [_to_yolo_line(d, w, h) for d in accepted]
        (lbl_dir / f"{img_p.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        stats["promoted_images"] += 1
        stats["promoted_boxes"] += len(accepted)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="伪标签自动晋升（替代人工标注）")
    ap.add_argument("--real", default="data/real_label")
    ap.add_argument("--out", default="data/training/raw/pseudo")
    ap.add_argument("--score", type=float, default=0.6, help="采纳置信度下限")
    ap.add_argument("--unc", type=float, default=0.5, help="采纳不确定性上限（超此不自动采纳）")
    ap.add_argument(
        "--rebuild", action="store_true", help="晋升后重建训练集（pseudo 为 train-only 源）"
    )
    args = ap.parse_args()

    real_root = resolve_config_path(args.real)
    out_dir = resolve_config_path(args.out)
    cfg = load_config()
    dcfg = cfg.detect
    det = _build_detector()
    stats = promote_high_conf(
        real_root,
        out_dir,
        det,
        conf=dcfg.infer_conf,
        iou=dcfg.infer_iou,
        class_conf=dcfg.class_conf,
        score_thr=args.score,
        unc_thr=args.unc,
    )
    print(f"[auto_pseudo] {stats}")
    if args.rebuild:
        # pseudo 源只进 train，评估（val/test）保持纯真实标注
        yaml = build_dataset(train_only_sources={"pseudo"})
        print(f"[auto_pseudo] 训练集已重建：{yaml}")


if __name__ == "__main__":
    main()
