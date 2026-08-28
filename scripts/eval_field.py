"""现场 val 评估（微调前后对比用）。

用法：python eval_field.py <model.pt>
输出：现场 val 25 张的整体指标 + 稀有类召回。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["气孔", "夹渣", "未焊透", "未熔合", "裂纹", "咬边"]
RARE = [1, 2, 3, 4, 5]


def main() -> None:
    model_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else ROOT / "runs" / "ft_field" / "steel_field_ft" / "weights" / "best.pt"
    )
    print(f"模型: {model_path}")
    m = YOLO(str(model_path))
    data = str(ROOT / "data" / "real_label" / "data.yaml")
    rl_img = ROOT / "data" / "real_label" / "images"
    rl_lbl = ROOT / "data" / "real_label" / "labels"

    # 现场 val 列表（val.txt）
    val_stems: list[str] = []
    for line in (ROOT / "data" / "real_label" / "val.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            val_stems.append(Path(line).stem)

    res = m.val(data=data, split="val", imgsz=640, device=0, verbose=False)
    print(
        f"=== 现场 val（{len(val_stems)} 张）===  mAP50={res.box.map50:.4f} mAP50-95={res.box.map:.4f} P={res.box.mp:.4f} R={res.box.mr:.4f}"
    )

    # 稀有类 GT（val 图）
    gts: dict[int, dict[str, list]] = {c: {} for c in RARE}
    for stem in val_stems:
        lbl = rl_lbl / f"{stem}.txt"
        if not lbl.exists():
            continue
        for line in lbl.read_text(encoding="utf-8").splitlines():
            p = line.strip().split()
            if len(p) != 5:
                continue
            try:
                c = int(p[0])
                if c in RARE:
                    gts[c].setdefault(stem, []).append(tuple(map(float, p[1:])))
            except ValueError:
                continue

    print("=== 稀有类召回（现场 val，conf 扫描）===")
    for conf in (0.05, 0.25):
        line = f"  conf={conf:<4}"
        for c in RARE:
            if not gts[c]:
                continue
            hits = 0
            total = 0
            for stem, boxes in gts[c].items():
                img_path = None
                for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                    cand = rl_img / f"{stem}{ext}"
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
