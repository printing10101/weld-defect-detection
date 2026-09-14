# -*- coding: utf-8 -*-
"""Golden Set v3 生成：与当前部署模型训练数据同源（backend/training/planB_run）。

背景（2026-09 能力评估）：现有 data/eval/golden（v2，make_golden_set.py 画法）
与训练语料分布不一致，部署模型在 v2 上 mAP50≈0.03（定位对但类别错乱），
在训练分布 test 划分上为 0.693——异分布集对训练模型没有回归监测意义。

进一步核查（本脚本定稿前）：当前权重（best::8370aeeef0b1）的训练语料
data/training/{train,val,test} 由 **planB_run.generate(seed=12345)** 生成
（亮底片背景 150–180、随机焊缝带/水渍/暗角/仿射，7 类，stem syn_%04d），
而非 defect_synth 的暗底片路线（灰度 55–85）——后者是另一批次原始数据，
不是本版权重的拟合分布。

本脚本用 **同一 planB 生成器 + 全新种子** 产出 v3：
- 通过 monkeypatch 重定向输出目录（不改训练脚本一行；绝不写入
  data/training/raw，防评估样本回流训练池造成评估泄漏）；
- 生成后对全部训练 split 做 dHash 感知汉明距离校验，量化"零重叠"；
- 固定种子 + 指纹 + 诚实口径写入 _META.json。

用途边界：v3 服务于**权重版本回归门禁**（同分布上比较新旧权重的可复现相对
指标），**不是**真实底片精度的证书。真实泛化仍须真实标注数据（当前缺失）。

用法：
  python scripts/make_golden_set_v3.py                      # 生成 data/eval/golden_v3
  python scripts/make_golden_set_v3.py --n 200 --seed 1     # 自定义规模/种子
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from backend.training import planB_run
from backend.training.planB_run import _dhash_u8, _hamming, generate

DEFAULT_OUT = _ROOT / "data" / "eval" / "golden_v3"
TRAIN_ROOT = _ROOT / "data" / "training"


def main() -> None:
    ap = argparse.ArgumentParser(description="Golden Set v3（训练同源 planB 分布）生成")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--n", type=int, default=120, help="生成图像数")
    ap.add_argument("--seed", type=int, default=20260913, help="起始种子（独立于训练种子 12345）")
    ap.add_argument("--max-seed-tries", type=int, default=50,
                    help="零重叠校验不过时自动换种子的最大尝试次数")
    args = ap.parse_args()

    out = Path(args.out)
    if (out / "images").is_dir() and any((out / "images").iterdir()):
        raise SystemExit(
            f"refuse: {out} 已存在且非空——Golden 集必须一次性固化为一个版本，"
            f"如需重建请先删除整个目录（防评估集复用/泄漏）"
        )

    # 训练集 dHash 一次性预计算（零重叠校验用）。
    train_hashes: list[int] = []
    for split in ("train", "val", "test"):
        d = TRAIN_ROOT / split / "images"
        if d.is_dir():
            train_hashes.extend(
                _dhash_u8(cv2.imdecode(np.fromfile(str(p), dtype=np.uint8),
                                       cv2.IMREAD_GRAYSCALE))
                for p in sorted(d.glob("*.png"))
            )

    # 重定向 planB 生成器的输出目录（模块级常量），不触碰 data/training/raw。
    img_dir = out / "images"
    lbl_dir = out / "labels"
    planB_run.RAW = out
    planB_run.IMG = img_dir
    planB_run.LBL = lbl_dir

    # 种子自动搜索：不同种子的"灰底+焊缝带"背景在 dHash 下可能撞训练集
    # （planB 生成期内去重只管集合内部），跨集合零重叠必须显式校验；
    # 不过阈值即换种子重生成（先清空上次产物）。
    chosen_seed = None
    min_h = -1
    for attempt in range(args.max_seed_tries):
        seed = args.seed + attempt
        for stale in (img_dir, lbl_dir):
            if stale.is_dir():
                for f in stale.iterdir():
                    f.unlink()
        generate(args.n, seed=seed)
        golden_hashes = [_dhash_u8(cv2.imdecode(np.fromfile(str(p), dtype=np.uint8),
                                                cv2.IMREAD_GRAYSCALE))
                         for p in sorted(img_dir.glob("*.png"))]
        min_h = min((_hamming(g, t) for g in golden_hashes for t in train_hashes),
                    default=-1)
        if min_h > 4:
            chosen_seed = seed
            break
        print(f"[golden_v3] 种子 {seed} 与训练集最小汉明距离 {min_h} ≤ 4，换种子重试")
    if chosen_seed is None:
        raise SystemExit(
            f"refuse: {args.max_seed_tries} 个种子均未通过零重叠校验（阈值 >4）——"
            f"请人工检查训练集构成或调整生成参数"
        )

    # ---- 元信息（与 golden v2 _META 同构 + 诚实口径） ----
    counts: Counter[int] = Counter()
    for lbl in lbl_dir.glob("*.txt"):
        for line in lbl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                counts[int(float(line.split()[0]))] += 1
    meta = {
        "synthetic": True,
        "version": 3,
        "seed": chosen_seed,
        "seed_search": f"起始 {args.seed}，第 {chosen_seed - args.seed + 1} 次尝试通过（跨集合零重叠）",
        "train_seed": 12345,
        "n_images": len(list(img_dir.glob("*.png"))),
        "size": [640, 640],
        "n_classes": 7,
        "classes": planB_run.CLASS_NAMES_ZH,
        "class_box_counts": {str(k): v for k, v in sorted(counts.items())},
        "generator": "backend/training/planB_run.py::generate（与当前部署权重的训练语料同源，种子不同）",
        "purpose": "权重版本回归门禁（同分布相对比较）；非真实底片精度证书",
        "provenance": (
            "合成缺陷图，无真实影像；禁止用于训练；"
            f"零重叠校验：与训练集 {len(train_hashes)} 张最小 dHash 汉明距离 = {min_h}"
            "（阈值 >4，与 planB 生成期内去重阈值一致）"
        ),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "generated_by": "scripts/make_golden_set_v3.py",
    }
    (out / "_META.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[golden_v3] {meta['n_images']} 张 → {out}（种子 {chosen_seed}）")
    print(f"[golden_v3] 每类框数: {dict(meta['class_box_counts'])}")
    print(f"[golden_v3] 零重叠校验: 最小汉明距离 {min_h}（阈值 >4）")


if __name__ == "__main__":
    main()
