"""正式训练评估：test 集整体 + 全部稀有类召回（对比 smoke 基线）。

用 best.pt 在 test 集上 val，输出整体指标；再对 5 个稀有类
（夹渣1/未焊透2/未熔合3/裂纹4/咬边5）做 conf 阈值扫描召回。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["气孔", "夹渣", "未焊透", "未熔合", "裂纹", "咬边"]
RARE = [1, 2, 3, 4, 5]


def main() -> None:
    m = YOLO(str(ROOT / "runs" / "train_steel_v1" / "steel_balanced" / "weights" / "best.pt"))
    data = str(ROOT / "data" / "training" / "data.yaml")
    test_img = ROOT / "data" / "training" / "test" / "images"
    test_lbl = ROOT / "data" / "training" / "test" / "labels"

    # 1) test 集整体指标
    res = m.val(data=data, split="test", imgsz=640, device=0, verbose=False)
    print(
        f"\n=== test 集整体 ===  mAP50={res.box.map50:.4f} mAP50-95={res.box.map:.4f} P={res.box.mp:.4f} R={res.box.mr:.4f}"
    )

    # 2) 读 test GT（按类）
    gts: dict[int, dict[str, list[tuple[float, float, float, float]]]] = {c: {} for c in RARE}
    for lbl in sorted(test_lbl.glob("*.txt")):
        for line in lbl.read_text(encoding="utf-8").splitlines():
            p = line.strip().split()
            if len(p) != 5:
                continue
            try:
                c = int(p[0])
                if c in RARE:
                    gts[c].setdefault(lbl.stem, []).append(tuple(map(float, p[1:])))
            except ValueError:
                continue

    # 3) conf 扫描：逐稀有类召回
    print("\n=== 稀有类召回（test 集 conf 扫描）===")
    for conf in (0.05, 0.10, 0.25, 0.5):
        line = f"  conf={conf:<4}"
        for c in RARE:
            if not gts[c]:
                continue
            hits = 0
            total = 0
            for stem, boxes in gts[c].items():
                img_path = None
                for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                    cand = test_img / f"{stem}{ext}"
                    if cand.exists():
                        img_path = cand
                        break
                if img_path is None:
                    continue
                arr = np.fromfile(str(img_path), dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                dets = m.predict(img, conf=conf, imgsz=640, device=0, verbose=False)[0]
                det_c = []
                if dets.boxes is not None and len(dets.boxes):
                    xyxy = dets.boxes.xyxy.cpu().numpy()
                    cl = dets.boxes.cls.cpu().numpy().astype(int)
                    for b, cc in zip(xyxy, cl):
                        if cc == c:
                            det_c.append(b)
                total += len(boxes)
                for cx, cy, w, h in boxes:
                    x1, y1 = (cx - w / 2) * img.shape[1], (cy - h / 2) * img.shape[0]
                    x2, y2 = (cx + w / 2) * img.shape[1], (cy + h / 2) * img.shape[0]
                    gw, gh = w * img.shape[1], h * img.shape[0]
                    for b in det_c:
                        ix1, iy1 = max(x1, b[0]), max(y1, b[1])
                        ix2, iy2 = min(x2, b[2]), min(y2, b[3])
                        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
                        if iw * ih > 0.3 * gw * gh:
                            hits += 1
                            break
            line += f" {NAMES[c]}={hits}/{total}({hits / total if total else 0:.0%})"
        print(line)
    print("EVAL_DONE")


if __name__ == "__main__":
    main()
