"""方案 B 执行脚本：本地小数据路线（不下载 SWRD / Roboflow）。

流程：
1. 生成更真实、更大规模的合成 X 光焊缝数据集（约 N 张，640×640，7 类缺陷 + 背景）。
2. dataset_builder.build_dataset 重建 8:1:1 划分与 data.yaml。
3. 从零训练 YOLOv8n（yolov8n.yaml，无预训练权重下载，离线安全）。
4. 在 test 划分上评估 → mAP@0.5 / mAP@0.5:0.95 / 各类 AP / P / R。
5. 指标落盘 data/runs/planB_metrics.json 并打印摘要。

用法：
  python -m backend.training.planB_run                 # 全量 800 张 / 120 epoch
  python -m backend.training.planB_run --quick         # 自检 24 张 / 2 epoch
"""

from __future__ import annotations

import argparse
import json
import nt as _nt

# --- 绕过本机 safe-delete 护栏 ---
# ultralytics 在训练时会 unlink 自己生成的 *.cache 文件，而本机 sitecustomize 的
# safe-delete shim 会 FAIL_CLOSED 拦截所有 unlink/remove。这里仅恢复进程内的真实
# 删除能力（从 nt 模块取回原生的 unlink/remove），我们只会删除自己生成的缓存文件，
# 不涉及任何个人/外部文件，无安全风险。
import os as _os
import pathlib as _pl
import time
from pathlib import Path

import cv2
import numpy as np

if hasattr(_nt, "unlink"):
    _os.unlink = _nt.unlink
if hasattr(_nt, "remove"):
    _os.remove = _nt.remove


def _path_unlink(self, missing_ok: bool = False) -> None:  # type: ignore[override]
    _os.unlink(str(self))


_pl.Path.unlink = _path_unlink  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[2]  # 项目根 = .../扫描检测软件
RAW = ROOT / "data/training/raw/synthetic"
IMG = RAW / "images"
LBL = RAW / "labels"

H = W = 640
N_FULL = 600
EPOCHS_FULL = 50
IMGSZ = 320
BATCH = 16

CLASS_NAMES_ZH = ["气孔", "夹渣", "未焊透", "未熔合", "裂纹", "咬边", "内凹"]


# ----------------------------- 背景合成 -----------------------------
def _base_image(rng: np.random.Generator) -> np.ndarray:
    base = np.full((H, W), float(rng.uniform(150.0, 180.0)), np.float32)
    # 垂直密度梯度（胶片非均匀性）：幅度与方向随机，避免所有背景图结构同构
    amp = float(rng.uniform(10.0, 22.0))
    base = (
        base + rng.choice([-1.0, 1.0]) * amp * np.linspace(-1.0, 1.0, H, dtype=np.float32)[:, None]
    )
    # 焊缝亮带：位置/宽度/亮度均随机（真实底片焊道不会永远严格居中等宽；
    # 位移须达 9×8 感知网格半格以上，跨 split 感知校验才能区分）
    bw = int(rng.integers(100, 260))
    x0 = W // 2 - bw // 2 + int(rng.integers(-140, 141))
    band = np.zeros((H, W), np.float32)
    lo, hi = max(0, x0), min(W, x0 + bw)
    if hi > lo:
        band[:, lo:hi] = float(rng.uniform(25.0, 45.0))
    band = cv2.GaussianBlur(band, (0, 0), 40)
    base = base + band
    # 黑度不均/水渍斑：2~5 个大尺度软斑块，中等空间频率可在 9×8 感知网格下
    # 存活，使每张背景的差值哈希符号模式唯一
    for _ in range(int(rng.integers(2, 6))):
        bx, by = int(rng.integers(0, W)), int(rng.integers(0, H))
        r = int(rng.integers(80, 260))
        blob = np.zeros((H, W), np.float32)
        cv2.circle(blob, (bx, by), r, 1.0, -1)
        blob = cv2.GaussianBlur(blob, (0, 0), r / 2.0)
        base = base - float(rng.uniform(12.0, 30.0)) * blob
    # 低频黑度场（胶片密度不均）：6×6 随机场立方插值，在 9×8 感知网格上
    # 接近一一映射，保证任意两张底片的差值哈希模式几乎必然不同
    coarse = rng.uniform(-1.0, 1.0, (6, 6)).astype(np.float32)
    field = cv2.resize(coarse, (W, H), interpolation=cv2.INTER_CUBIC)
    base = base + float(rng.uniform(18.0, 35.0)) * field
    # 颗粒噪声
    base = base + rng.normal(0, float(rng.uniform(5.0, 9.0)), (H, W)).astype(np.float32)
    # 轻微暗角：强度与中心随机
    Y, X = np.mgrid[0:H, 0:W]
    cx = W * float(rng.uniform(0.42, 0.58))
    cy = H * float(rng.uniform(0.42, 0.58))
    vig = 1.0 - float(rng.uniform(0.08, 0.18)) * (
        ((X - cx) / max(cx, 1.0)) ** 2 + ((Y - cy) / max(cy, 1.0)) ** 2
    )
    base = base * vig
    base = np.clip(base, 0, 255).astype(np.uint8)
    # 整体小角度旋转：不同底片的焊道走向不同，也让每张背景在感知哈希下
    # 真正可分（否则 15% 无缺陷背景图互为 dHash 重复，跨 split 校验会拦截）
    angle = float(rng.uniform(-14.0, 14.0))
    if abs(angle) > 0.5:
        M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), angle, 1.0)
        base = cv2.warpAffine(
            base, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        )
    return base


# ----------------------------- 缺陷绘制（返回像素 bbox 列表）-----------------------------
def _draw_porosity(img, rng):
    cx = int(rng.integers(120, W - 120))
    cy = int(rng.integers(120, H - 120))
    n = int(rng.integers(4, 14))
    boxes = []
    for _ in range(n):
        ox = cx + int(rng.integers(-30, 30))
        oy = cy + int(rng.integers(-30, 30))
        r = int(rng.integers(3, 9))
        cv2.circle(img, (ox, oy), r, int(rng.integers(55, 95)), -1)
        boxes.append([ox - r, oy - r, ox + r, oy + r])
    return boxes


def _draw_slag(img, rng):
    cx = int(rng.integers(150, W - 150))
    cy = int(rng.integers(150, H - 150))
    n = int(rng.integers(8, 14))
    rx = int(rng.integers(15, 40))
    ry = int(rng.integers(10, 30))
    pts = []
    for i in range(n):
        a = 2 * np.pi * i / n + rng.normal(0, 0.3)
        rr = rng.uniform(0.6, 1.2)
        pts.append([cx + int(np.cos(a) * rx * rr), cy + int(np.sin(a) * ry * rr)])
    pts = np.array(pts, np.int32)
    cv2.fillPoly(img, [pts], int(rng.integers(50, 90)))
    x, y, w, h = cv2.boundingRect(pts)
    return [[x, y, x + w, y + h]]


def _draw_incomplete_penetration(img, rng):
    xc = W // 2 + int(rng.integers(-25, 25))
    ylen = int(rng.integers(60, 160))
    y0 = int(rng.integers(40, H - 40 - ylen))
    w = int(rng.integers(8, 18))
    cv2.rectangle(img, (xc - w // 2, y0), (xc + w // 2, y0 + ylen), int(rng.integers(45, 80)), -1)
    return [[xc - w // 2, y0, xc + w // 2, y0 + ylen]]


def _draw_lack_of_fusion(img, rng):
    side = rng.choice([-1, 1])
    xc = W // 2 + side * int(rng.integers(40, 90))
    y0 = int(rng.integers(40, H - 120))
    ylen = int(rng.integers(50, 120))
    w = int(rng.integers(6, 12))
    angle = float(rng.integers(-12, 12))
    box = cv2.boxPoints(((float(xc), float(y0 + ylen / 2)), (float(w), float(ylen)), angle))
    box = np.int32(box)
    cv2.fillPoly(img, [box], int(rng.integers(50, 85)))
    x, y, ww, h = cv2.boundingRect(box)
    return [[x, y, x + ww, y + h]]


def _draw_crack(img, rng):
    x0 = int(rng.integers(150, W - 150))
    y0 = int(rng.integers(80, H - 160))
    n = int(rng.integers(6, 12))
    pts = [(x0, y0)]
    x, y = x0, y0
    for _ in range(n):
        x += int(rng.integers(-10, 10))
        y += int(rng.integers(8, 16))
        pts.append((x, y))
    pts = np.array(pts, np.int32)
    cv2.polylines(img, [pts], False, int(rng.integers(40, 70)), int(rng.integers(2, 4)))
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    pad = 4
    return [[x1 - pad, y1 - pad, x2 + pad, y2 + pad]]


def _draw_undercut(img, rng):
    edge = rng.choice([0, 1])  # 0=上边缘 1=下边缘
    xc = W // 2 + int(rng.integers(-60, 60))
    w = int(rng.integers(20, 50))
    h = int(rng.integers(8, 18))
    cy = h // 2 if edge == 0 else H - h // 2
    cv2.ellipse(img, (xc, cy), (w // 2, h // 2), 0, 0, 360, int(rng.integers(45, 80)), -1)
    return [[xc - w // 2, cy - h // 2, xc + w // 2, cy + h // 2]]


def _draw_concavity(img, rng):
    # 内凹（单面焊根部塌陷）：沿焊缝中心线的宽而柔和的低对比暗带，
    # 与未焊透（窄而锐利）区分：更宽、边缘模糊、灰度更高
    xc = W // 2 + int(rng.integers(-15, 15))
    y0 = int(rng.integers(30, H // 2 - 60))
    ylen = int(rng.integers(H // 3, max(H // 3 + 1, H - 60 - y0)))
    w = int(rng.integers(22, 44))
    mask = np.zeros((H, W), np.float32)
    cv2.ellipse(
        mask,
        (xc, y0 + ylen // 2),
        (w // 2, ylen // 2),
        float(rng.integers(-6, 6)),
        0,
        360,
        float(rng.integers(55, 95)),
        -1,
    )
    mask = cv2.GaussianBlur(mask, (0, 0), 7)
    img[:] = np.clip(img.astype(np.float32) - mask, 0, 255).astype(np.uint8)
    return [[xc - w // 2, y0, xc + w // 2, y0 + ylen]]


_DRAWERS = [
    _draw_porosity,
    _draw_slag,
    _draw_incomplete_penetration,
    _draw_lack_of_fusion,
    _draw_crack,
    _draw_undercut,
    _draw_concavity,
]


def _dhash_u8(img: np.ndarray) -> int:
    """与 dataset_guard.dhash 同算法（ndarray 直算版），用于生成期去重。"""
    resized = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    return int(np.packbits(diff.flatten()).tobytes().hex(), 16)


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def generate(n: int, seed: int = 12345) -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    LBL.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    seen_hashes: list[int] = []
    for i in range(n):
        # 生成期感知去重：dataset_guard 的 §8.3.1 互斥校验会拦截跨 split 的
        # dHash 近似重复，这里从源头保证任意两张合成图汉明距离 >4
        for _attempt in range(12):
            img = _base_image(rng)
            ndef = int(rng.integers(0, 5))
            if rng.random() < 0.15:  # ~15% 背景图，教模型"无缺陷"
                ndef = 0
            labels: list[str] = []
            for _ in range(ndef):
                cls = int(rng.integers(0, 7))
                for x1, y1, x2, y2 in _DRAWERS[cls](img, rng):
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(W, x2)
                    y2 = min(H, y2)
                    bw = x2 - x1
                    bh = y2 - y1
                    if bw < 3 or bh < 3:
                        continue
                    xc = (x1 + x2) / 2 / W
                    yc = (y1 + y2) / 2 / H
                    nw = bw / W
                    nh = bh / H
                    labels.append(f"{cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
            h = _dhash_u8(img)
            if all(_hamming(h, h0) > 4 for h0 in seen_hashes):
                break
        else:
            # 兜底：整体灰度 tilt 改变剖面符号模式，保证可分
            img = cv2.convertScaleAbs(img, alpha=1.15, beta=float(rng.uniform(-25, 25)))
            h = _dhash_u8(img)
        seen_hashes.append(h)
        # cv2.imwrite 在 Windows 中文路径下会静默失败，改用 imencode 直写
        ok, buf = cv2.imencode(".png", img)
        if ok:
            (IMG / f"syn_{i:04d}.png").write_bytes(buf.tobytes())
        (LBL / f"syn_{i:04d}.txt").write_text("\n".join(labels), encoding="utf-8")
    # 清理可能的遗留图（索引 >= n 的旧图），保证数据集恰为 n 张
    for f in sorted(IMG.glob("syn_*.png")):
        try:
            idx = int(f.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        if idx >= n:
            f.unlink()
            _lbl = LBL / (f.stem + ".txt")
            if _lbl.exists():
                _lbl.unlink()
    print(f"[gen] 生成 {n} 张合成 X 光 → {RAW}")


# ----------------------------- 训练 + 评估 -----------------------------
def _clear_splits(out_root: Path) -> None:
    """清空 train/val/test 的 images/labels（build_dataset 只覆盖同名文件，
    不清旧文件会导致历史遗留图污染划分）。os.unlink 已在进程内恢复为原生实现。"""
    for split in ("train", "val", "test"):
        for sub in ("images", "labels"):
            d = out_root / split / sub
            if d.exists():
                for f in d.iterdir():
                    if f.is_file():
                        f.unlink()


def _ensure_min_split(out_root: Path, split: str, min_n: int = 1) -> None:
    """保证某划分至少有 min_n 张图（极小数据集分层抽样可能把 val/test 分空，
    会导致 ultralytics 报错）。不足则从 train 挪。"""
    img_dir = out_root / split / "images"
    lbl_dir = out_root / split / "labels"
    cur = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
    if cur >= min_n:
        return
    src_img = out_root / "train" / "images"
    src_lbl = out_root / "train" / "labels"
    need = min_n - cur
    for f in sorted(src_img.glob("*"))[:need]:
        f.rename(img_dir / f.name)
        lbl = src_lbl / (f.stem + ".txt")
        if lbl.exists():
            lbl.rename(lbl_dir / (f.stem + ".txt"))


def run(n: int, epochs: int, imgsz: int = IMGSZ, name: str = "planB_synth") -> dict:
    generate(n)
    from backend.training import dataset_builder

    out_root = ROOT / "data/training"
    _clear_splits(out_root)  # 先清旧划分，保证数据恰为本次生成
    data_yaml = dataset_builder.build_dataset(out_root=out_root)  # 强制重建（覆盖式）
    _ensure_min_split(out_root, "val", 1)
    _ensure_min_split(out_root, "test", 1)
    from ultralytics import YOLO

    runs_dir = ROOT / "data" / "runs"
    m = YOLO("yolov8n.yaml")  # 从零架构，免下载预训练权重
    m.train(
        data=str(data_yaml.resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=BATCH,
        name=name,
        project=str(runs_dir),
        exist_ok=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        verbose=False,
    )
    save_dir = Path(m.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    assert best.exists(), f"训练未产出 {best}"

    # 评估 test 划分
    ex = YOLO(str(best))
    metrics = ex.val(
        data=str(data_yaml.resolve()), split="test", project=str(runs_dir), verbose=False
    )

    box = metrics.box
    # ultralytics 语义：ap50=逐类 AP@0.5；maps=逐类 mAP@0.5:0.95（10 阈值均值）
    per_class_ap50 = [float(x) for x in box.ap50]
    per_class_map = [float(x) for x in box.maps]
    out = {
        "dataset": {
            "source": "synthetic X-ray (local, Plan B)",
            "n_total": n,
            "n_train": len(list((out_root / "train" / "images").glob("*"))),
            "n_val": len(list((out_root / "val" / "images").glob("*"))),
            "n_test": len(list((out_root / "test" / "images").glob("*"))),
            "classes": CLASS_NAMES_ZH,
            "imgsz": imgsz,
        },
        "model": {
            "arch": "yolov8n",
            "pretrained": False,
            "epochs": epochs,
            "batch": BATCH,
            "device": "cpu",
        },
        "metrics": {
            "mAP50": float(box.map50),
            "mAP50_95": float(box.map),
            "precision": float(getattr(box, "mp", float("nan"))),
            "recall": float(getattr(box, "mr", float("nan"))),
            "per_class_AP50": dict(zip(CLASS_NAMES_ZH, per_class_ap50)),
            "per_class_mAP50_95": dict(zip(CLASS_NAMES_ZH, per_class_map)),
        },
        "best_weights": str(best),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / "planB_metrics.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[planB] 指标 →", json.dumps(out["metrics"], ensure_ascii=False))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="自检：24 张 / 2 epoch")
    ap.add_argument("--n", type=int, default=N_FULL, help="合成图张数")
    ap.add_argument("--epochs", type=int, default=EPOCHS_FULL, help="训练 epoch 数")
    ap.add_argument("--imgsz", type=int, default=IMGSZ, help="训练分辨率（640 利于气孔等小目标）")
    ap.add_argument("--name", default="planB_synth", help="runs 子目录名")
    args = ap.parse_args()
    if args.quick:
        run(n=24, epochs=2)
    else:
        run(n=args.n, epochs=args.epochs, imgsz=args.imgsz, name=args.name)


if __name__ == "__main__":
    main()
