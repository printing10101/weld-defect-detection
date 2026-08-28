"""M4b 数据增强（§17）。

两类增强：
1) Albumentations 在线增强（X 光焊缝专用：CLAHE / 模糊 / 噪声 / 亮度对比 /
   翻转 / 旋转），供离线预增强或诊断使用。
2) Copy-Paste（离线）：把源图缺陷裁剪后粘贴到随机目标图，生成困难样本，
   直接缓解小样本下的过拟合（见 DATA_LICENSE.md 训练策略）。

依赖 albumentations（延迟导入，未安装时给出明确报错，不阻断模块导入）。
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def build_xray_augment() -> Any:
    """返回 Albumentations Compose（X 光焊缝专用，YOLO bbox 格式）。"""
    try:
        import albumentations as A
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("请先安装 albumentations: pip install albumentations") from e
    return A.Compose(
        [
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.GaussNoise(var_limit=(5.0, 30.0), p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.05,
                rotate_limit=5,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.3,
            ),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )


def build_multiscale_augment(
    min_scale: float = 0.6,
    max_scale: float = 1.4,
    target_hw: tuple[int, int] = (640, 640),
    p: float = 0.5,
) -> Any:
    """多尺度训练增强（P1-A，借鉴 LF-YOLO 多尺度策略）。

    随机缩放（0.6~1.4x）+ 中心裁剪/填充回固定训练尺寸，迫使网络学习
    尺度不变的缺陷表征（尤其小目标气孔与细长裂纹），再叠加常规 X 光增强。
    供离线预增强或诊断使用；YOLO bbox 标签随变换自动更新。
    """
    try:
        import albumentations as A
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("请先安装 albumentations: pip install albumentations") from e
    h, w = target_hw
    return A.Compose(
        [
            A.RandomScale(scale_limit=(min_scale - 1.0, max_scale - 1.0), p=p),
            A.PadIfNeeded(
                min_height=h,
                min_width=w,
                border_mode=cv2.BORDER_CONSTANT,
                value=114,
            ),
            A.RandomCrop(height=h, width=w, p=1.0),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.GaussNoise(var_limit=(5.0, 30.0), p=0.4),
            A.HorizontalFlip(p=0.5),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )


def _read_gray(path: Path | str):
    """稳健读取为 2D 灰度。

    两点原因不用 cv2.imread：①cv2 对**含中文的绝对路径**在 Windows 上会静默失败
    （ANSI 编码，本项目已记录此坑），用 np.fromfile + cv2.imdecode 走 unicode 安全通道；
    ②IMREAD_GRAYSCALE 对带 alpha 的 PNG / 多通道 TIFF 会返回 3D，显式 ndim 容错转灰度，
    避免 `h, w = img.shape` 解包失败。
    """
    arr = np.fromfile(str(path), dtype=np.uint8)
    if arr.size == 0:
        return None
    im = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    elif im.ndim == 4:
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2GRAY)
    return im


def _read_yolo_pairs(root: Path) -> list[tuple[Path, Path]]:
    img_dir = root / "images"
    lbl_dir = root / "labels"
    pairs: list[tuple[Path, Path]] = []
    if not img_dir.exists():
        return pairs
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
            continue
        lbl = lbl_dir / (img.stem + ".txt")
        if lbl.exists():
            pairs.append((img, lbl))
    return pairs


def generate_copy_paste(
    source_dir: str | Path,
    out_dir: str | Path,
    per_image: int = 1,
    seed: int = 42,
) -> int:
    """将源图缺陷 copy-paste 到随机目标图，生成增强样本到 out_dir。

    返回生成的图像数。每张源图向其自身缺陷之外的随机目标图粘贴 per_image 次，
    新标签追加到目标图原有标签之后；以 alpha 混合模拟真实灰度过渡。
    """
    source_dir = Path(source_dir)
    out_dir = Path(out_dir)
    out_img = out_dir / "images"
    out_lbl = out_dir / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(seed)
    pairs = _read_yolo_pairs(source_dir)
    if not pairs:
        return 0
    imgs = [p[0] for p in pairs]
    made = 0
    for src_img, src_lbl in pairs:
        bboxes = [
            ln.split() for ln in src_lbl.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
        if not bboxes:
            continue
        base = _read_gray(src_img)
        if base is None:
            continue
        h, w = base.shape
        for _ in range(per_image):
            tgt_path = rnd.choice(imgs)
            tgt = _read_gray(tgt_path)
            if tgt is None:
                continue
            th, tw = tgt.shape
            out = tgt.copy()
            new_labels: list[str] = []
            for parts in bboxes:
                try:
                    cls = int(parts[0])
                    cx, cy, bw, bh = (float(x) for x in parts[1:5])
                except (ValueError, IndexError):
                    continue
                sx = int((cx - bw / 2) * w)
                sy = int((cy - bh / 2) * h)
                sw_ = max(2, int(bw * w))
                sh_ = max(2, int(bh * h))
                sx = max(0, min(sx, w - sw_))
                sy = max(0, min(sy, h - sh_))
                crop = base[sy : sy + sh_, sx : sx + sw_]
                if crop.size == 0:
                    continue
                scale = min(1.0, (tw / max(sw_, 1)) ** 0.5, (th / max(sh_, 1)) ** 0.5)
                cw = max(2, int(sw_ * scale))
                ch = max(2, int(sh_ * scale))
                crop = cv2.resize(crop, (cw, ch))
                tx = rnd.randint(0, max(0, tw - cw))
                ty = rnd.randint(0, max(0, th - ch))
                roi = out[ty : ty + ch, tx : tx + cw].astype(np.float32)
                blended = cv2.addWeighted(roi, 0.35, crop.astype(np.float32), 0.65, 0)
                out[ty : ty + ch, tx : tx + cw] = blended.astype(np.uint8)
                ncx = (tx + cw / 2) / tw
                ncy = (ty + ch / 2) / th
                new_labels.append(f"{cls} {ncx:.6f} {ncy:.6f} {cw / tw:.6f} {ch / th:.6f}")
            tgt_lbl = source_dir / "labels" / (tgt_path.stem + ".txt")
            existing = tgt_lbl.read_text(encoding="utf-8").splitlines() if tgt_lbl.exists() else []
            all_labels = [ln for ln in existing if ln.strip()] + new_labels
            if not all_labels:
                continue
            out_name = f"cp_{made:06d}_{src_img.stem}_x_{tgt_path.stem}.png"
            # cv2.imwrite 在含中文的绝对路径下会静默失败（与 _read_gray 对称的中文路径安全写入）
            ok, buf = cv2.imencode(".png", out)
            if not ok:
                continue
            (out_img / out_name).write_bytes(buf.tobytes())
            (out_lbl / (Path(out_name).stem + ".txt")).write_text("\n".join(all_labels))
            made += 1
    print(f"[augment] copy-paste 生成 {made} 张 → {out_dir}")
    return made


def _rare_boxes(
    lbl_path: Path, rare_classes: set[int]
) -> list[tuple[int, float, float, float, float]]:
    """读 YOLO txt，仅返回属于 rare_classes 的框 (cls, cx, cy, bw, bh)。"""
    out: list[tuple[int, float, float, float, float]] = []
    try:
        lines = Path(lbl_path).read_text(encoding="utf-8").splitlines()
    except (OSError, ValueError):
        return out
    for ln in lines:
        p = ln.strip().split()
        if len(p) < 5:
            continue
        try:
            c = int(p[0])
            if c in rare_classes:
                out.append((c, float(p[1]), float(p[2]), float(p[3]), float(p[4])))
        except (ValueError, IndexError):
            continue
    return out


def generate_rare_copy_paste(
    source_dir: str | Path,
    out_dir: str | Path,
    rare_classes: set[int] | None = None,
    per_source: int = 8,
    seed: int = 42,
    scale_jitter: float = 0.2,
    flip_p: float = 0.5,
) -> int:
    """稀有类专用 Copy-Paste：只从含稀有类(默认 cls 1-5)的图裁剪稀有框，随机粘贴到其它底片背景，
    把稀有类实例从极少(27)提升到数百，直接缓解小样本过拟合。

    与 generate_copy_paste 的区别：①只采样稀有类框，不复制气孔；②每张源图产 per_source 张合成图；
    ③随机尺度抖动 + 随机水平翻转增加多样性；④目标图排除自身避免平凡自粘贴。
    合成图写入 out_dir/{images,labels}，标签 = 目标图原标签 + 新粘贴的稀有框。

    返回生成图像数。
    """
    if rare_classes is None:
        rare_classes = {1, 2, 3, 4, 5}
    source_dir = Path(source_dir)
    out_dir = Path(out_dir)
    out_img = out_dir / "images"
    out_lbl = out_dir / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(seed)

    pairs = _read_yolo_pairs(source_dir)
    if not pairs:
        return 0
    sources = [(img, lbl) for img, lbl in pairs if _rare_boxes(lbl, rare_classes)]
    all_imgs = [p[0] for p in pairs]
    if not sources:
        return 0

    made = 0
    for src_img, src_lbl in sources:
        base = _read_gray(src_img)
        if base is None:
            continue
        h, w = base.shape
        rare = _rare_boxes(src_lbl, rare_classes)
        if not rare:
            continue
        for _ in range(per_source):
            tgt_path = rnd.choice(all_imgs)
            if Path(tgt_path) == src_img:
                tgt_path = rnd.choice(all_imgs)
            tgt = _read_gray(tgt_path)
            if tgt is None:
                continue
            th, tw = tgt.shape
            out = tgt.copy()
            new_labels: list[str] = []
            for cls, cx, cy, bw, bh in rare:
                sj = 1.0 + rnd.uniform(-scale_jitter, scale_jitter)
                sx = int((cx - bw / 2) * w)
                sy = int((cy - bh / 2) * h)
                sw_ = max(2, int(bw * w))
                sh_ = max(2, int(bh * h))
                sx = max(0, min(sx, w - sw_))
                sy = max(0, min(sy, h - sh_))
                crop = base[sy : sy + sh_, sx : sx + sw_]
                if crop.size == 0:
                    continue
                cw = min(max(2, int(sw_ * sj)), tw)
                ch = min(max(2, int(sh_ * sj)), th)
                crop = cv2.resize(crop, (cw, ch))
                if rnd.random() < flip_p:
                    crop = cv2.flip(crop, 1)
                tx = rnd.randint(0, max(0, tw - cw))
                ty = rnd.randint(0, max(0, th - ch))
                roi = out[ty : ty + ch, tx : tx + cw].astype(np.float32)
                blended = cv2.addWeighted(roi, 0.35, crop.astype(np.float32), 0.65, 0)
                out[ty : ty + ch, tx : tx + cw] = blended.astype(np.uint8)
                ncx = (tx + cw / 2) / tw
                ncy = (ty + ch / 2) / th
                new_labels.append(f"{cls} {ncx:.6f} {ncy:.6f} {cw / tw:.6f} {ch / th:.6f}")
            tgt_lbl = source_dir / "labels" / (tgt_path.stem + ".txt")
            existing = (
                [ln for ln in tgt_lbl.read_text(encoding="utf-8").splitlines() if ln.strip()]
                if tgt_lbl.exists()
                else []
            )
            all_labels = existing + new_labels
            if not all_labels:
                continue
            out_name = f"rcp_{made:06d}_{src_img.stem}_x_{tgt_path.stem}.png"
            # cv2.imwrite 在含中文的绝对路径下会静默失败（与 _read_gray 对称的中文路径安全写入）
            ok, buf = cv2.imencode(".png", out)
            if not ok:
                continue
            (out_img / out_name).write_bytes(buf.tobytes())
            (out_lbl / (Path(out_name).stem + ".txt")).write_text("\n".join(all_labels))
            made += 1
    print(f"[augment] rare copy-paste 生成 {made} 张 → {out_dir}")
    return made


if __name__ == "__main__":
    import sys

    sd = sys.argv[1] if len(sys.argv) > 1 else "data/training/raw/swrd"
    od = sys.argv[2] if len(sys.argv) > 2 else "data/training/raw/synthetic"
    n = generate_copy_paste(sd, od)
    print("generated", n)
