"""Smoke 训练评估：重点验证裂纹（class 4）检出效果。

用 best.pt 在 test 集上 val，输出 per-class P/R/mAP50；
并做 conf 阈值扫描看裂纹召回随阈值的变化（低阈值放行更多裂纹候选）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["气孔", "夹渣", "未焊透", "未熔合", "裂纹", "咬边"]


def main() -> None:
    m = YOLO(str(ROOT / "runs" / "smoke_steel" / "steel_synth" / "weights" / "best.pt"))
    # 1) test 集 val（默认 conf=0.001 全量评估）
    res = m.val(
        data=str(ROOT / "data" / "training" / "data.yaml"),
        split="test",
        imgsz=640,
        device=0,
        verbose=False,
    )
    print("\n=== overall (test set) ===")
    print(
        f"mAP50={res.box.map50:.4f} mAP50-95={res.box.map:.4f} P={res.box.mp:.4f} R={res.box.mr:.4f}"
    )

    # 2) 裂纹（class 4）conf 阈值扫描：统计 test 集 GT 裂纹 vs 检出
    import cv2

    test_img = ROOT / "data" / "training" / "test" / "images"
    test_lbl = ROOT / "data" / "training" / "test" / "labels"
    gts: dict[str, list] = {}
    for lbl in sorted(test_lbl.glob("*.txt")):
        lines = []
        for line in lbl.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 5 and parts[0] == "4":  # 裂纹 GT
                lines.append(tuple(map(float, parts[1:])))
        if lines:
            gts[lbl.stem] = lines

    print(f"\n=== 裂纹(class4) test GT：{sum(len(v) for v in gts.values())} 框 / {len(gts)} 图 ===")
    for conf in (0.05, 0.10, 0.25, 0.5):
        hits = 0
        total = 0
        n_skipped_img = 0
        for stem, gt_lines in gts.items():
            img_path = None
            for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                cand = test_img / f"{stem}{ext}"
                if cand.exists():
                    img_path = cand
                    break
            if img_path is None:
                n_skipped_img += 1
                continue
            arr = np.fromfile(str(img_path), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                n_skipped_img += 1
                continue
            dets = m.predict(img, conf=conf, imgsz=640, device=0, verbose=False)[0]
            det_crack = []
            if dets.boxes is not None and len(dets.boxes):
                xyxy = dets.boxes.xyxy.cpu().numpy()
                cl = dets.boxes.cls.cpu().numpy().astype(int)
                for b, c in zip(xyxy, cl):
                    if c == 4:
                        det_crack.append(b)
            total += len(gt_lines)
            for cx, cy, w, h in gt_lines:
                x1, y1 = (cx - w / 2) * img.shape[1], (cy - h / 2) * img.shape[0]
                x2, y2 = (cx + w / 2) * img.shape[1], (cy + h / 2) * img.shape[0]
                for b in det_crack:
                    ix1, iy1 = max(x1, b[0]), max(y1, b[1])
                    ix2, iy2 = min(x2, b[2]), min(y2, b[3])
                    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                    if iw * ih > 0.3 * w * img.shape[1] * h * img.shape[0]:
                        hits += 1
                        break
        print(f"  conf={conf:<4} 裂纹召回 {hits}/{total} = {hits / total if total else 0:.1%}")

    print("EVAL_DONE")


if __name__ == "__main__":
    main()
