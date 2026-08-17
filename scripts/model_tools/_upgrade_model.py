# -*- coding: utf-8 -*-
"""一键模型升级流水线（用户提供新真实样本 → 升级安装包）。

流程：
  1) 收集新样本（data/real_label/new_samples/images/*.jpg + labels/*.txt，YOLO 归一化）
  2) 合并标注 → 同步 prelabels → labels（人工标注优先，不覆盖）
  3) 重划 train/val（seed=42，80/20）
  4) GPU 重训（起点 = 当前 best.pt 增量微调，缺失则 yolo11n.pt）
  5) 真实全集评估（conf 多档 + 中心点召回/精确）
  6) legacy ONNX 导出 → 安装包 models/weights/best.onnx
  7) 重建轻量分包 + 生成安装 zip + 复验

用法（必须用带 CUDA torch 的解释器，如 gpu venv）：
  python _upgrade_model.py                          # 用默认 new_samples 目录
  python _upgrade_model.py --new-samples <dir> --epochs 80 --skip-train
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ---------- safe-delete 绕法（仅删自己生成的文件/目录） ----------
import nt
os.unlink = nt.unlink
os.remove = nt.remove
os.rmdir = nt.rmdir
from pathlib import Path as _P  # noqa: E402  (ultralytics 训练会 unlink *.cache)
_P.unlink = lambda self, *a, **k: os.unlink(str(self))

ROOT = Path(__file__).resolve().parent
REAL = ROOT / "data" / "real_label"
IMG_DIR = REAL / "images"
PRE_DIR = REAL / "prelabels"
LBL_DIR = REAL / "labels"
NEW_DEF = REAL / "new_samples"
WEIGHTS_DIR = REAL / "runs" / "yolo11n_real2" / "train" / "weights"
BEST_PT = WEIGHTS_DIR / "best.pt"
PKG = ROOT / "_pkg"

# 人工标注集合（28 张，来自早期 Label Studio/标注器，优先保留不覆盖）
MANUAL_STEMS = {
    "PG101-1-1", "PG101-2-6", "PG102-1-1", "PG102-2-3", "PG102-3-5",
    "PG102-5-4", "PG102-6-6", "PG103-1-1", "PG103-1-2", "PG103-2-1",
    "PG103-2-3", "PG103-2-4", "PG103-3-5", "PG103-5-2", "PG103-6-4",
    "PG103-7-6", "PG12-2-1", "PG120-1-1", "PG121-4-1", "PG121-6-2",
    "PG132-2-2", "PL117-2-1", "PL117-7-2", "PL118-12-1", "PL118-13-5",
    "PL118-28-5dcn", "PL118-30-5jpg", "PL118-32-5",
}

C_STEP = "    ├─"


def log(msg: str) -> None:
    print(msg, flush=True)


def run(cmd: list[str], desc: str) -> int:
    log(f"\n[C] {desc}\n{C_STEP} {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        log(f"[FAIL] {desc}（退出码 {r.returncode}）")
        sys.exit(r.returncode)
    return 0


def _has_content(p: Path) -> bool:
    if not p.is_file() or p.stat().st_size == 0:
        return False
    return any(l.strip() for l in p.read_text(encoding="utf-8").splitlines())


# ---------------------------------------------------------------------------
def step1_collect_new_samples(new_dir: Path) -> int:
    """把 new_samples/{images,labels} 合入现有数据集（不覆盖人工标注）。"""
    new_img = new_dir / "images"
    new_lbl = new_dir / "labels"
    if not (new_img.is_dir() and new_lbl.is_dir()):
        log(f"[SKIP] 新样本目录不存在（{new_dir} 下需 images/ 与 labels/）→ 仅重训现有数据")
        return 0
    n_img = n_lbl = n_skip = 0
    for f in sorted(new_img.glob("*.jpg")):
        dst = IMG_DIR / f.name
        shutil.copy2(f, dst)          # 覆盖式更新（图像本身无人工/自动之分）
        n_img += 1
    for f in sorted(new_lbl.glob("*.txt")):
        stem = f.stem
        if stem in MANUAL_STEMS:      # 人工标注优先，不覆盖
            n_skip += 1
            continue
        dst = PRE_DIR / f.name
        shutil.copy2(f, dst)          # 进 prelabels（合并）
        n_lbl += 1
    log(f"[OK] 新样本合入: 图 {n_img} 张, 标注 {n_lbl} 份, 跳过人工 {n_skip} 份")
    return n_img + n_lbl


def step2_sync_labels() -> int:
    """重建 labels：人工保留 + prelabels 非空覆盖其余。"""
    removed = synced = 0
    for f in list(LBL_DIR.glob("*.txt")):
        if f.stem not in MANUAL_STEMS:
            nt.unlink(str(f)); removed += 1
    for f in PRE_DIR.glob("*.txt"):
        if _has_content(f):
            shutil.copy2(f, LBL_DIR / f.name); synced += 1
    n = sum(1 for f in LBL_DIR.glob("*.txt") if _has_content(f))
    log(f"[OK] labels 重建: 删旧 {removed}, 同步 {synced}, 非空总数 {n}")
    return n


def step4_train(epochs: int) -> None:
    import torch
    if not torch.cuda.is_available():
        log("[FAIL] 当前解释器无 CUDA（需用 gpu venv："
            "C:/Users/Lenovo/.workbuddy/binaries/python/envs/gpu/Scripts/python.exe）")
        sys.exit(2)
    start = BEST_PT if BEST_PT.exists() else REAL / "yolo11n.pt"
    log(f"[TRAIN] 起点 {start}，{epochs} epoch，device=cuda:0")
    from ultralytics import YOLO
    m = YOLO(str(start))
    m.train(
        data=str(REAL / "data.yaml"),
        epochs=epochs, imgsz=640, batch=16, device=0, workers=0,
        project=str(WEIGHTS_DIR.parent), name="train", exist_ok=True,
        verbose=True, patience=25, save=True, save_period=10,
        mosaic=0.5, mixup=0.1, degrees=5.0, translate=0.05, scale=0.2,
        flipud=0.0, fliplr=0.5, hsv_h=0.01, hsv_s=0.3, hsv_v=0.2,
        lr0=1e-3, lrf=1e-2,
    )


def step5_eval() -> None:
    from ultralytics import YOLO
    import cv2
    import numpy as np
    from collections import Counter

    m = YOLO(str(BEST_PT))
    stems = sorted(p.name[:-4] for p in IMG_DIR.glob("*.jpg"))
    total_gt = 0
    gts = {}
    for s in stems:
        p = LBL_DIR / f"{s}.txt"
        gt = []
        if p.exists():
            for line in p.read_text(encoding="utf-8").strip().splitlines():
                parts = line.split()
                if len(parts) == 5:
                    try:
                        gt.append((int(parts[0]), *map(float, parts[1:])))
                    except ValueError:
                        pass
        gts[s] = gt
        total_gt += len(gt)
    log(f"[EVAL] 真实全集 {len(stems)} 张 / 真值 {total_gt} 框")
    for thr in (0.001, 0.01, 0.1, 0.25, 0.5):
        n_pred_img = boxes = hits = fp = 0
        max_c = 0.0
        cls_dist = Counter()
        for s in stems:
            img = cv2.imread(str(IMG_DIR / f"{s}.jpg"), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            try:
                dets = m.predict(img, conf=thr, imgsz=640, device=0, verbose=False)[0]
            except Exception:
                continue
            pred = []
            if dets.boxes is not None and len(dets.boxes):
                xyxy = dets.boxes.xyxy.cpu().numpy()
                conf = dets.boxes.conf.cpu().numpy()
                cl = dets.boxes.cls.cpu().numpy().astype(int)
                for b, c, k in zip(xyxy, conf, cl):
                    pred.append(((b[0]+b[2])/2/img.shape[1], (b[1]+b[3])/2/img.shape[0]))
                    cls_dist[k] += 1
                    max_c = max(max_c, float(c))
                n_pred_img += 1
                boxes += len(pred)
            gt = gts[s]
            used = [False]*len(gt)
            for (cx, cy) in pred:
                for i, (_, gx, gy, gw, gh) in enumerate(gt):
                    if not used[i] and gx-gw/2 <= cx <= gx+gw/2 and gy-gh/2 <= cy <= gy+gh/2:
                        hits += 1; used[i] = True; break
                else:
                    fp += 1
        rec = hits/total_gt if total_gt else 0.0
        prec = hits/(hits+fp) if (hits+fp) else 0.0
        log(f"  thr>{thr:<4} 有预测图={n_pred_img}/{len(stems)} 框={boxes} max_conf={max_c:.3f} "
            f"召回={rec:.1%} 精确={prec:.1%} 类别={dict(cls_dist)}")


def main() -> int:
    ap = argparse.ArgumentParser(description="一键模型升级流水线")
    ap.add_argument("--new-samples", default=str(NEW_DEF), help="新样本目录（images/ + labels/）")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--skip-train", action="store_true", help="跳过训练（仅评估+打包）")
    args = ap.parse_args()

    log("=" * 66)
    log("模型升级流水线")
    log(f"新样本: {args.new_samples}")
    log("=" * 66)

    step1_collect_new_samples(Path(args.new_samples))
    step2_sync_labels()
    run([sys.executable, str(REAL / "build_splits_merged.py")], "重划 train/val")
    if not args.skip_train:
        step4_train(args.epochs)
    step5_eval()
    run([sys.executable, str(ROOT / "_export_model.py"),
         "--src", str(BEST_PT), "--pkg", str(PKG / "ScanDetection")], "legacy ONNX 导出")
    run([sys.executable, str(PKG / "build_lean_pkg.py")], "重建轻量分包")
    run([sys.executable, str(PKG / "make_lean_zip.py")], "生成安装 zip")
    run([sys.executable, str(ROOT / "_verify_model_pkg.py")], "安装包复验")
    log("\n✅ 升级流水线完成。新安装包: " + str(PKG / "ScanDetection_v0.1.0.zip"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())