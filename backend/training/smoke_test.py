"""M4b 端到端烟雾测试（合成数据，无许可风险）。

流程：合成 X 光风格图像 + YOLO 标签 → dataset_builder.build_dataset
→ 训练 2 个 epoch（tiny，从零架构，免下载权重）→ YoloDetector(torch) 推理回一条
→ 校验 Detection 字段与 need_review 链路可读。

不依赖 SWRD / 用户数据；仅用于验证 M4b 代码链路可跑通。
用法：python -m backend.training.smoke_test
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from backend.domain.detect.yolo_detector import YoloDetector
from backend.training import dataset_builder

RAW = Path("data/training/raw/synthetic")
N_TOTAL = 160
IMGSZ = 320
SEED = 7


def _make_image(h: int = 320, w: int = 320, n_defects: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    base = np.full((h, w), 180, np.uint8)
    base = np.clip(base.astype(int) + rng.integers(-10, 10, (h, w)), 0, 255).astype(np.uint8)
    band = slice(w // 2 - 30, w // 2 + 30)
    base[:, band] = np.clip(base[:, band].astype(int) + 40, 0, 255).astype(np.uint8)
    labels: list[str] = []
    for _ in range(n_defects):
        cls = int(rng.integers(0, 6))
        bw = int(rng.integers(8, 30))
        bh = int(rng.integers(8, 30))
        cx = int(rng.integers(bw, w - bh))
        cy = int(rng.integers(bh, h - bh))
        cv2.ellipse(base, (cx, cy), (bw // 2, bh // 2), 0, 0, 360, int(rng.integers(40, 90)), -1)
        labels.append(f"{cls} {cx / w:.6f} {cy / h:.6f} {bw / w:.6f} {bh / h:.6f}")
    return base, labels


def synthesize() -> None:
    img_dir = RAW / "images"
    lbl_dir = RAW / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for i in range(N_TOTAL):
        nd = int(np.random.default_rng(SEED + i).integers(1, 4))
        img, labels = _make_image(seed=SEED + i, n_defects=nd)
        cv2.imwrite(str(img_dir / f"syn_{i:04d}.png"), img)
        (lbl_dir / f"syn_{i:04d}.txt").write_text("\n".join(labels))
    print(f"[smoke] 合成 {N_TOTAL} 张 → {RAW}")


def main() -> None:
    synthesize()
    data_yaml = dataset_builder.build_dataset()
    from ultralytics import YOLO

    # 从零架构训练（免下载预训练权重），仅验证链路；真实训练请用 yolov8m.pt 预训练。
    model = YOLO("yolov8n.yaml")
    model.train(
        data=str(data_yaml.resolve()),
        epochs=2,
        imgsz=IMGSZ,
        batch=8,
        name=f"smoke_{int(time.time())}",
        project="data/runs",
        exist_ok=True,
        verbose=False,
    )
    # ultralytics 实际落盘路径含 task 子目录（runs/detect/...），以 trainer.save_dir 为准
    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    assert best.exists(), f"训练未产出 {best}（实际 save_dir={save_dir}）"

    det = YoloDetector()
    det.load(str(best), backend="torch")
    test_img = max((RAW / "images").iterdir())
    arr = cv2.imread(str(test_img), cv2.IMREAD_GRAYSCALE)
    dets = det.infer(arr, conf=0.25, iou=0.5)
    print(f"[smoke] 测试图 {test_img.name} → {len(dets)} 个检测")
    for d in dets[:5]:
        print(
            f"   {d.class_id.name} score={d.score:.3f} unc={d.uncertainty:.3f} "
            f"bbox=({d.bbox.x:.0f},{d.bbox.y:.0f},{d.bbox.w:.0f},{d.bbox.h:.0f}) "
            f"shape={d.shape.value if d.shape else None}"
        )
    print("[smoke] ✅ M4b 链路跑通（合成数据）")


if __name__ == "__main__":
    main()
