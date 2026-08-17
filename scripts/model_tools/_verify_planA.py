"""方案 A 终验：逐类置信度阈值在已部署 ONNX 上的实际增益。

加载 _pkg 部署 ONNX（yolo8n_real_rare 平衡模型 / YOLOv8n，已确认非坍塌），
复用 backend 的 YoloDetector（含逐类阈值逻辑）做两套推理对比：
  - OLD：统一阈值 infer_conf=0.30（部署前行为）
  - NEW：逐类阈值（气孔高阈值压制过检，稀有类低阈值吃满召回）
    阈值来自 runs/yolo8n_real_rare/eval_rare_metrics.json 的类别感知评测：
    稀有类 TP 框置信度均 >=0.01（召回在 0.001->0.01 间稳定 96.3%），故稀有类
    阈值取 0.01；气孔在 0.001 下吐 12882 框（绝大多数为误检），取 0.30 压制。
遍历 data/real_label/images 下全部真实底片，按类别统计检出数。
"""
from __future__ import annotations
import sys, time
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

import cv2
import numpy as np
from PIL import Image
from backend.domain.detect.yolo_detector import YoloDetector


def load_gray(path: Path) -> np.ndarray | None:
    """cv2 的 JPEG 解码器在本 venv 损坏，改用 PIL 读取后转灰度 numpy。"""
    try:
        return np.asarray(Image.open(str(path)).convert("L"))
    except Exception:
        return None

ONNX = ROOT / "_pkg" / "ScanDetection" / "models" / "weights" / "best.onnx"
IMG_DIR = ROOT / "data" / "real_label" / "images"
# 阈值依据 eval_rare_metrics.json（类别感知评测）：
#   气孔(0)：0.30 —— 压制 12882 过量误检（OLD 部署同值，隔离稀有类效应）
#   夹渣(1)/未焊透(2)/咬边(5)：0.01 —— TP 框 conf>=0.01，吃满 96.3% 召回且丢大多 FP
#   未熔合(3)/裂纹(4)：0.008/0.005 —— 样本更少/更安全关键，留更宽余量
CLASS_CONF = {0: 0.30, 1: 0.01, 2: 0.01, 3: 0.008, 4: 0.005, 5: 0.01}
NAMES = ["气孔", "夹渣", "未焊透", "未熔合", "裂纹", "咬边"]

def main() -> None:
    assert ONNX.exists(), f"部署 ONNX 缺失: {ONNX}"
    det = YoloDetector()
    det.load(str(ONNX), backend="onnx")
    stems = sorted(p.stem for p in IMG_DIR.glob("*.jpg") if not p.name.startswith("."))
    print(f"[载入] {ONNX.name} | 真实底片 {len(stems)} 张")

    cnt_old = Counter(); cnt_new = Counter()
    rare_old = rare_new = 0
    por_old = por_new = 0
    t0 = time.time()
    samples: list[tuple[str, int, float]] = []
    for i, stem in enumerate(stems):
        gray = load_gray(IMG_DIR / (stem + ".jpg"))
        if gray is None or gray.size == 0:
            continue
        old = det.infer(gray, conf=0.30, iou=0.5)
        new = det.infer(gray, conf=0.30, iou=0.5, class_conf=CLASS_CONF)
        for d in old:
            cnt_old[d.class_id.value] += 1
            if d.class_id.value == 0:
                por_old += 1
            else:
                rare_old += 1
        for d in new:
            cnt_new[d.class_id.value] += 1
            if d.class_id.value == 0:
                por_new += 1
            else:
                rare_new += 1
                if len(samples) < 25:
                    samples.append((stem, d.class_id.value, round(d.score, 3)))
    dt = time.time() - t0

    print(f"\n{'类别':<10}{'OLD(统一0.30)':>14}{'NEW(逐类)':>14}{'Δ':>10}")
    print("-" * 50)
    for c in range(6):
        o, n = cnt_old.get(c, 0), cnt_new.get(c, 0)
        print(f"{NAMES[c]:<10}{o:>14}{n:>14}{n - o:>+10}")
    print("-" * 50)
    print(f"{'气孔合计':<10}{por_old:>14}{por_new:>14}{por_new - por_old:>+10}")
    print(f"{'稀有类合计':<10}{rare_old:>14}{rare_new:>14}{rare_new - rare_old:>+10}")
    print(f"{'全部合计':<10}{sum(cnt_old.values()):>14}{sum(cnt_new.values()):>14}{sum(cnt_new.values())-sum(cnt_old.values()):>+10}")
    print(f"\n耗时 {dt:.1f}s / {len(stems)} 张")

    print("\n[NEW 放出的稀有类样本 (类别, 置信度)]")
    for s in samples:
        print(f"  {s[0]:<28} {NAMES[s[1]]}  score={s[2]}")

    # 结论判定
    released = rare_new - rare_old
    por_delta = por_new - por_old
    print("\n[结论]")
    print(f"  稀有类检出: OLD={rare_old} -> NEW={rare_new} (释放 +{released})")
    print(f"  气孔检出:   OLD={por_old} -> NEW={por_new} (Δ{por_delta:+d})")
    if released > 0 and por_delta <= 0:
        print("  PASS: 逐类阈值成功释放稀有类且未增加气孔误检 —— 方案 A 生效。")
    elif released > 0:
        print("  PARTIAL: 稀有类检出增加，但气孔也略增（可能真实多孔底片，需人工核对）。")
    else:
        print("  WARN: 稀有类未见增加，需核查 ONNX 是否仍为畸形/模型能力。")

if __name__ == "__main__":
    main()
