# -*- coding: utf-8 -*-
"""新模型能力探针（训练完成后运行）。

目的：
  1) 坍塌判定：对 zeros / ones / 随机噪声做输入无关性测试。若输出与输入无关
     （box 与 logit 完全一致），则权重坍塌，方案 A 无意义。
  2) 稀有类候选探查：在真实底片上以极低的全局 conf（0.001）取全部候选，统计
     各类检出数与置信度分位数。若稀有类在 0.001 阈值下仍为 0，说明模型根本不
     产生稀有类候选框 —— 方案 A（逐类降阈值）无从释放，瓶颈在模型能力/数据量。

运行：GPU venv（含 ultralytics + torch + CUDA）
  python _probe_newmodel.py [--src PATH]
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from PIL import Image

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DEFAULT_SRC = ROOT / "data" / "real_label" / "runs" / "yolo8n_real_rare" / "train" / "weights" / "best.pt"
IMG_DIR = ROOT / "data" / "real_label" / "images"
NAMES = ["气孔", "夹渣", "未焊透", "未熔合", "裂纹", "咬边"]


def load_gray(path: Path) -> np.ndarray | None:
    try:
        return np.asarray(Image.open(str(path)).convert("L"))
    except Exception:
        return None


def to_rgb_t(p: Path) -> np.ndarray | None:
    g = load_gray(p)
    if g is None or g.size == 0:
        return None
    if g.ndim == 2 or (g.ndim == 3 and g.shape[2] == 1):
        g = np.stack([g, g, g], axis=-1)
    return g


def input_response_test(m: YOLO) -> None:
    """输入无关性测试：坍塌模型对所有输入输出恒定。"""
    print("\n=== [1] 输入无关性测试（坍塌判定）===")
    variants = {
        "zeros": np.zeros((640, 640, 3), np.uint8),
        "ones": np.ones((640, 640, 3), np.uint8) * 255,
        "random": np.random.randint(0, 256, (640, 640, 3), np.uint8),
    }
    sigs: dict[str, tuple] = {}
    for name, img in variants.items():
        res = m(img, conf=0.001, iou=0.1, verbose=False)[0]
        n = 0
        box_sig = ""
        score_sig = 0.0
        if res.boxes is not None and len(res.boxes) > 0:
            n = len(res.boxes)
            b = res.boxes.xyxy.cpu().numpy()
            s = res.boxes.conf.cpu().numpy()
            # 用首框坐标与最大分数构造签名
            box_sig = ",".join(f"{v:.1f}" for v in b[0].tolist())
            score_sig = float(s.max()) if len(s) else 0.0
        sigs[name] = (n, box_sig, round(score_sig, 4))
        print(f"  {name:<8} boxes={n} 首框={box_sig} 最大score={score_sig}")
    boxes_set = {v[0] for v in sigs.values()}
    # 坍塌判据：三种输入检出数完全一致且 box 签名相同
    if len(boxes_set) == 1 and len({v[1] for v in sigs.values()}) == 1:
        print("  >>> 判定：坍塌（输出与输入无关）。方案 A 无意义，需重训。")
    else:
        print("  >>> 判定：非坍塌（模型对输入有响应）。可继续。")
    return sigs


def rare_candidate_probe(m: YOLO) -> None:
    """极低 conf 探查真实底片上的候选框分布。"""
    print("\n=== [2] 真实底片稀有类候选探查（conf=0.001）===")
    stems = sorted(p.stem for p in IMG_DIR.glob("*.jpg") if not p.name.startswith("."))
    print(f"  真实底片 {len(stems)} 张")
    cnt = Counter()
    scores_by_cls = defaultdict(list)
    img_with_rare = set()
    t0 = time.time()
    for stem in stems:
        rgb = to_rgb_t(IMG_DIR / (stem + ".jpg"))
        if rgb is None:
            continue
        res = m(rgb, conf=0.001, iou=0.1, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        clss = res.boxes.cls.cpu().numpy().astype(int)
        scs = res.boxes.conf.cpu().numpy()
        for c, s in zip(clss, scs):
            cnt[c] += 1
            scores_by_cls[c].append(float(s))
            if c != 0:
                img_with_rare.add(stem)
    dt = time.time() - t0
    print(f"\n  {'类别':<10}{'候选数':>10}{'占比%':>10}{'min':>9}{'p50':>9}{'p90':>9}{'max':>9}")
    print("  " + "-" * 66)
    total = sum(cnt.values())
    for c in range(6):
        sc = sorted(scores_by_cls[c])
        n = len(sc)
        qs = (f"{sc[0]:.3f}" if n else "  -")
        p50 = (f"{sc[n // 2]:.3f}" if n else "  -")
        p90 = (f"{sc[int(n * 0.9)]:.3f}" if n else "  -")
        mx = (f"{sc[-1]:.3f}" if n else "  -")
        pct = (f"{100 * n / total:.1f}" if total else "  -")
        print(f"  {NAMES[c]:<10}{n:>10}{pct:>10}{qs:>9}{p50:>9}{p90:>9}{mx:>9}")
    print("  " + "-" * 66)
    print(f"  全部候选={total}  含稀有类底片={len(img_with_rare)}/{len(stems)}")
    rare_total = sum(cnt[c] for c in range(1, 6))
    print(f"  稀有类候选合计={rare_total}")
    if rare_total == 0:
        print("  >>> 判定：稀有类在 0.001 仍 0 候选 —— 方案 A 无法释放，瓶颈在模型能力/数据。")
    else:
        print("  >>> 判定：稀有类有候选 —— 方案 A 可经逐类降阈值释放，按 p50 调参。")
    return cnt, scores_by_cls


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    args = ap.parse_args()
    src = Path(args.src)
    if not src.exists():
        print(f"[probe] ERROR: 权重不存在 {src}", file=sys.stderr)
        raise SystemExit(1)
    print(f"[probe] loading {src}")
    m = YOLO(str(src))
    print("[probe] classes:", m.names)
    input_response_test(m)
    rare_candidate_probe(m)


if __name__ == "__main__":
    main()
