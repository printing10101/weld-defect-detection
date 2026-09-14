"""逐环节诊断：评片检测链路到底在哪一步丢掉了缺陷。

复现 /api/v1/detect 的 _detect_sync 全流程，逐环节打印检出数：
  加载 → 胶片区分割/背景填充 → 预处理增强 → 检测器推理 → 印字区过滤 → 掩膜精修

对同一张图分别喂「原图」和「增强图」给检测器，即可判定丢检出发生在
预处理之前还是之后。

用法（仓库根目录，后端 venv）：
  python scripts/diag_detect_pipeline.py
  python scripts/diag_detect_pipeline.py --real-dir 定检/定检 --n-real 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def synth_pores(seed: int = 42) -> np.ndarray:
    base = np.full((480, 640), 165.0, np.float32)
    base[:, 220:420] += 35.0
    base = cv2.GaussianBlur(base, (0, 0), 9)
    rng = np.random.default_rng(seed)
    img = base.copy()
    for _ in range(12):
        x, y = int(rng.integers(240, 400)), int(rng.integers(60, 420))
        cv2.circle(img, (x, y), int(rng.integers(3, 7)), -30.0, -1)
    return np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype("uint8")


def read_gray(path: Path) -> np.ndarray:
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise ValueError(f"无法解码 {path}")
    return img


def counts(dets) -> str:
    if not dets:
        return "0 个"
    from collections import Counter

    c = Counter(d.class_id.name for d in dets)
    return f"{len(dets)} 个  ({', '.join(f'{k}×{v}' for k, v in c.most_common())})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-dir", default="定检/定检")
    ap.add_argument("--n-real", type=int, default=3)
    args = ap.parse_args()

    from backend.app.dependencies import Registry
    from backend.domain.film_region import (
        FilmRegionCfg,
        detect_film_region_trusted,
        film_background_fill,
    )
    from backend.domain.stamp import StampCfg, filter_stamp_zone, read_stamp_aligned

    reg = Registry()
    dc = reg.config.detect
    print(f"检测器 kind={reg.detector_kind}  degraded={reg.detector_degraded}")
    print(f"权重 uri={reg.config.model.default_uri}")
    print(f"infer_conf={dc.infer_conf}  class_conf={dc.class_conf}")
    print(f"preprocess.enabled={reg.config.preprocess.enabled}  film_region.enabled={reg.config.film_region.enabled}")
    print(f"mask_stamp_zone={getattr(dc, 'mask_stamp_zone', None)}")

    imgs: list[tuple[str, np.ndarray]] = [("合成·密集气孔", synth_pores())]
    real_dir = ROOT / args.real_dir
    if real_dir.is_dir():
        files = sorted(
            p for p in real_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".bmp"}
        )
        print(f"\n真实底片目录 {real_dir}：{len(files)} 张，取前 {args.n_real} 张")
        for p in files[: args.n_real]:
            try:
                imgs.append((f"真实·{p.name}", read_gray(p)))
            except Exception as exc:  # noqa: BLE001
                print(f"  [skip] {p.name}: {exc}")

    fr = reg.config.film_region
    fr_cfg = FilmRegionCfg(
        min_area_frac=fr.min_area_frac,
        max_photo_area_frac=fr.max_photo_area_frac,
        surround_bright_gray=fr.surround_bright_gray,
        surround_min_frac=fr.surround_min_frac,
    )
    pp_cfg = reg.config.preprocess
    class_conf = dict(dc.class_conf)

    for label, gray in imgs:
        print(f"\n{'=' * 72}\n### {label}   shape={gray.shape} dtype={gray.dtype}  "
              f"min/med/max={gray.min()}/{int(np.median(gray))}/{gray.max()}")
        # 1) 原图直接推理
        d0 = reg.detector.infer(gray, conf=dc.infer_conf, iou=dc.infer_iou, class_conf=class_conf)
        print(f"  [1] 原图 → 检测器(conf={dc.infer_conf})           : {counts(d0)}")
        # 2) 胶片区背景填充
        film = detect_film_region_trusted(gray, fr_cfg) if fr.enabled else None
        filled = film_background_fill(gray, film)
        diff = float(np.abs(filled.astype(np.int16) - gray.astype(np.int16)).mean())
        print(f"  [2] 胶片区分割: film={'None' if film is None else getattr(film, 'bbox', film)}"
              f"  背景填充后与原图平均差={diff:.2f}")
        d1 = reg.detector.infer(filled, conf=dc.infer_conf, iou=dc.infer_iou, class_conf=class_conf)
        print(f"  [3] 填充图 → 检测器                            : {counts(d1)}")
        # 3) 预处理增强
        enhanced = filled
        if pp_cfg.enabled:
            pp = reg.preprocessor
            enhanced = pp.enhance(pp.denoise(filled), pp_cfg.gamma)
        ediff = float(np.abs(enhanced.astype(np.int16) - filled.astype(np.int16)).mean())
        print(f"  [4] 预处理增强后与原图平均差={ediff:.2f}  "
              f"min/med/max={enhanced.min()}/{int(np.median(enhanced))}/{enhanced.max()}")
        d2 = reg.detector.infer(enhanced, conf=dc.infer_conf, iou=dc.infer_iou, class_conf=class_conf)
        print(f"  [5] 增强图 → 检测器（=生产路径）                : {counts(d2)}")
        # 4) 印字区过滤
        if getattr(dc, "mask_stamp_zone", False):
            stamp = read_stamp_aligned(
                gray,
                StampCfg(
                    enabled=reg.config.stamp.enabled,
                    min_conf=reg.config.stamp.min_conf,
                    max_side=reg.config.stamp.max_side,
                ),
                film=film,
            )
            zones = stamp.text_boxes or stamp.boxes
            kept, removed = filter_stamp_zone(d2, zones) if zones else (d2, [])
            print(f"  [6] 印字区: status={stamp.status} zones={len(zones)} → "
                  f"屏蔽 {len(removed)} 个，剩余 {counts(kept)}")


if __name__ == "__main__":
    main()
