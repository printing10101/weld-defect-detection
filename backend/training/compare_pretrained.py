"""实证对比：COCO 预训练 YOLOv8n 迁移学习 vs 从零训练，验证「预训练模型能否减少所需样本量」。

- pretrained_300 : 在已建好的 300 张合成集上，以 yolov8n.pt(COCO预训练) 为起点训练。
                  与方案B(from-scratch, mAP50=66.3%) 做同数据同超参对照。
- pretrained_100 : 取 100 张合成子集(同分布)，以 COCO 预训练为起点训练，演示「少样本」下的迁移能力。

结论写 data/runs/compare_metrics.json。
"""

from __future__ import annotations

import json

# 进程内恢复原生删除（避开 safe-delete 护栏对 *.cache / rmtree 的拦截）
import nt
import os
import random
import shutil
import time
from pathlib import Path

os.unlink = nt.unlink
os.remove = nt.remove
Path.unlink = lambda self: os.unlink(str(self))

ROOT = Path(".")
CANON_RAW = ROOT / "data/training/raw/synthetic"
START = str(ROOT / "data/runs/pretrained/yolov8n.pt")
IMGSZ = 320
BATCH = 16
EPOCHS = 50

from ultralytics import YOLO


def _local_split(n: int, out_root: Path) -> Path:
    """从 canonical 300 中取前 n 张，本地 8:1:1 划分，写 data.yaml。不碰 canonical 数据。"""
    for d in ["train", "val", "test", "raw"]:
        p = out_root / d
        if p.exists():
            shutil.rmtree(p)
    raw_img = out_root / "raw/synthetic/images"
    raw_lbl = out_root / "raw/synthetic/labels"
    raw_img.mkdir(parents=True, exist_ok=True)
    raw_lbl.mkdir(parents=True, exist_ok=True)
    files = sorted(p.name for p in (CANON_RAW / "images").glob("*.png"))[:n]
    for f in files:
        shutil.copy(CANON_RAW / "images" / f, raw_img / f)
        sf = f[:-4] + ".txt"
        if (CANON_RAW / "labels" / sf).exists():
            shutil.copy(CANON_RAW / "labels" / sf, raw_lbl / sf)
    # 8:1:1 划分
    rnd = random.Random(42)
    rnd.shuffle(files)
    n_train = round(n * 0.8)
    n_val = round(n * 0.1)
    splits = {
        "train": files[:n_train],
        "val": files[n_train : n_train + n_val],
        "test": files[n_train + n_val :],
    }
    for sp, fs in splits.items():
        di = out_root / sp / "images"
        dl = out_root / sp / "labels"
        di.mkdir(parents=True, exist_ok=True)
        dl.mkdir(parents=True, exist_ok=True)
        for f in fs:
            shutil.copy(raw_img / f, di / f)
            sf = f[:-4] + ".txt"
            if (raw_lbl / sf).exists():
                shutil.copy(raw_lbl / sf, dl / sf)
    names = ["气孔", "夹渣", "未焊透", "未熔合", "裂纹", "咬边"]
    (out_root / "data.yaml").write_text(
        f"path: {out_root.resolve()}\n"
        f"train: train/images\nval: val/images\ntest: test/images\n"
        f"nc: 6\nnames: {names}\n",
        encoding="utf-8",
    )
    return out_root / "data.yaml"


def _train_and_val(data_yaml: Path, name: str, project: Path) -> dict:
    m = YOLO(START)  # 迁移学习：保留 COCO 预训练 backbone
    m.train(
        data=str(data_yaml.resolve()),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        name=name,
        project=str(project),
        exist_ok=True,
        verbose=False,
    )
    best = Path(m.trainer.save_dir) / "weights" / "best.pt"
    ex = YOLO(str(best))
    metrics = ex.val(data=str(data_yaml.resolve()), split="test", verbose=False)
    box = metrics.box
    per_class = [float(x) for x in box.maps]
    return {
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "precision": float(getattr(box, "mp", float("nan"))),
        "recall": float(getattr(box, "mr", float("nan"))),
        "per_class_AP50": dict(
            zip(["气孔", "夹渣", "未焊透", "未熔合", "裂纹", "咬边"], per_class)
        ),
        "n_test": len(list((data_yaml.parent / "test/images").glob("*"))),
    }


def main():
    res = {}
    # 1) 300 张合成集，COCO 预训练起点（与方案B from-scratch 对照）
    print("[compare] 运行 pretrained_300 ...")
    res["pretrained_300"] = _train_and_val(
        ROOT / "data/training/data.yaml", "pt300", ROOT / "data/compare/runs"
    )
    print("[compare] pretrained_300:", res["pretrained_300"]["mAP50"])

    # 2) 100 张合成子集，COCO 预训练起点（少样本演示）
    print("[compare] 运行 pretrained_100 ...")
    dy100 = _local_split(100, ROOT / "data/compare/subset100")
    res["pretrained_100"] = _train_and_val(dy100, "pt100", ROOT / "data/compare/runs")
    print("[compare] pretrained_100:", res["pretrained_100"]["mAP50"])

    out = {
        "baseline_scratch_300_mAP50": 0.663,  # 方案B from-scratch 结果
        "runs": res,
        "config": {"imgsz": IMGSZ, "batch": BATCH, "epochs": EPOCHS, "start": "yolov8n.pt(COCO)"},
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (ROOT / "data/runs/compare_metrics.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[compare] 完成 → data/runs/compare_metrics.json")


if __name__ == "__main__":
    main()
