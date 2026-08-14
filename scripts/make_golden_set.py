#!/usr/bin/env python3
"""生成确定性合成 Golden Set（§15.6 评估回归门禁用）。

输出目录：data/eval/golden/{images,labels}
- images/*.png：512x512 灰度合成射线底片风（暗缺陷团块）
- labels/*.txt：YOLO 格式真值（class 0 = POROSITY，暗缺陷）
- _META.json：生成参数 + 合规声明

设计约束（对齐 #4 评估回归门禁目标）：
- 确定性：固定 seed，可完全复现；任何文件增删改都会改变 harness 内容指纹，
  从而在评估记录中留下 Golden Set 版本痕迹（禁止用于训练，版本变更需显式记录）。
- 零真实数据：纯合成暗团块，无真实患者/工业机密影像，无标准授权依赖，
  规避"真实授权文本缺失"约束，同时提供可复现回归基准。
- 与 BlobDetector 对齐：检测器基线仅预测 POROSITY（class 0）暗缺陷，
  故真值类别统一为 0，使召回/mAP 对检测→指标→报告→漂移→跟踪全链路有意义。

真实工业 YOLO 模型的评估走同一流水线（python -m backend.evaluation --backend yolo），
但因需权重与 ML 依赖，不阻塞 CI；其基线单独管理。
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

SEED = 42
N_IMAGES = 12
SIZE = 512
BG = 150            # 母材背景灰度
DEFECT_VAL = 55     # 缺陷团块灰度（明显暗于背景）
CLSID = 0           # POROSITY（DefectClass.POROSITY = 0）

# 仓库根：scripts/make_golden_set.py -> parents[1]
ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "data" / "eval" / "golden"


def _make_image(rng: np.random.Generator, idx: int) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """生成单张合成底片 + 暗缺陷真值框（绝对像素，左上角 + 宽高）。"""
    img = np.full((SIZE, SIZE), BG, dtype=np.int16)
    # 轻微水平梯度（模拟母材明暗起伏），幅度小不产生均匀暗区
    grad = np.linspace(0, 18, SIZE, dtype=np.int16)
    img = img + grad[None, :]
    # 合成高斯噪声
    noise = rng.normal(0, 4.0, (SIZE, SIZE)).astype(np.int16)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    targets: list[tuple[int, int, int, int]] = []
    n_def = int(rng.integers(1, 5))
    for _ in range(n_def):
        cx = int(rng.integers(40, SIZE - 40))
        cy = int(rng.integers(40, SIZE - 40))
        rx = int(rng.integers(7, 20))
        ry = int(rng.integers(7, 20))
        # 不规则暗团块（气孔/夹渣类形态）
        cv2.ellipse(
            img,
            (cx, cy),
            (rx, ry),
            angle=int(rng.integers(0, 180)),
            startAngle=0,
            endAngle=360,
            color=int(DEFECT_VAL),
            thickness=-1,
        )
        # 真值框（绝对像素，左上角 + 宽高），与检测器归一化输出同一语义
        x = max(0, cx - rx)
        y = max(0, cy - ry)
        w = min(SIZE, cx + rx) - x
        h = min(SIZE, cy + ry) - y
        targets.append((x, y, w, h))
    return img, targets


def _write_yolo(path: Path, targets: list[tuple[int, int, int, int]]) -> None:
    lines = []
    for x, y, w, h in targets:
        cx = (x + w / 2) / SIZE
        cy = (y + h / 2) / SIZE
        nw = w / SIZE
        nh = h / SIZE
        lines.append(f"{CLSID} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


def main() -> None:
    rng = np.random.default_rng(SEED)
    (GOLDEN / "images").mkdir(parents=True, exist_ok=True)
    (GOLDEN / "labels").mkdir(parents=True, exist_ok=True)
    for i in range(N_IMAGES):
        img, targets = _make_image(rng, i)
        stem = f"synthetic_{i:03d}"
        # 注意：cv2.imwrite 在 Windows 上对含非 ASCII（中文）路径会静默失败，
        # 故走 imencode + 写字节的 unicode 安全路径（与 image_loader._imread_unicode 对应）。
        png_path = GOLDEN / "images" / f"{stem}.png"
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            raise RuntimeError(f"imencode 失败: {png_path}")
        png_path.write_bytes(buf.tobytes())
        _write_yolo(GOLDEN / "labels" / f"{stem}.txt", targets)
    meta = {
        "synthetic": True,
        "seed": SEED,
        "n_images": N_IMAGES,
        "size": SIZE,
        "class_id": CLSID,
        "classes": ["POROSITY"],
        "provenance": (
            "合成缺陷图（暗团块），无真实患者/工业机密影像，无标准授权依赖；"
            "仅作评估回归基准，禁止用于训练。"
        ),
        "generated_by": "scripts/make_golden_set.py",
    }
    (GOLDEN / "_META.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Golden Set written to {GOLDEN} ({N_IMAGES} images)")


if __name__ == "__main__":
    main()
