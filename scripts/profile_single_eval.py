#!/usr/bin/env python3
"""单幅评定分段耗时画像（生产口径：YOLO ONNX + 印字 OCR，含分块推理）。

perf_baseline.py 走 512px 小图 + 基线检测器，代表不了大底片分块推理的真实
耗时；本脚本按 run_inspection 的真实阶段顺序逐段计时，用于定位 15s 目标的
瓶颈。数据目录隔离到临时目录，不污染 data/。

用法（仓库根目录）：
    python scripts/profile_single_eval.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_TMP = Path(tempfile.mkdtemp(prefix="profile_eval_"))

# 隔离须先于 backend 导入生效（create_app/registry 在导入期读配置）
os.environ["SCAN_PATHS__DB_PATH"] = str(_TMP / "profile.db")
os.environ["SCAN_PATHS__IMAGES_DIR"] = str(_TMP / "images")
os.environ["SCAN_PATHS__REPORTS_DIR"] = str(_TMP / "reports")
os.environ["SCAN_PATHS__DATA_DIR"] = str(_TMP / "data")
os.environ.setdefault("SCAN_IPC__ENFORCE", "false")
os.environ.setdefault("SCAN_RATE_LIMIT", "0")
os.environ.setdefault("SCAN_LLM__ENABLED", "false")
os.environ.setdefault("SCAN_GATE__ALLOW_8BIT", "true")

sys.path.insert(0, str(_REPO))


def make_film(w: int, h: int, seed: int = 7) -> np.ndarray:
    """合成大底片：噪声母材 + 焊缝带 + 暗斑缺陷（确定性）。"""
    rng = np.random.default_rng(seed)
    img = rng.normal(150.0, 6.0, size=(h, w)).clip(0, 255).astype(np.uint8)
    band_x0, band_x1 = int(w * 0.42), int(w * 0.58)
    img[:, band_x0:band_x1] = (img[:, band_x0:band_x1] * 0.72).astype(np.uint8)
    for _ in range(max(6, (w * h) // 400_000)):
        cy, cx = int(rng.integers(40, h - 40)), int(rng.integers(40, w - 40))
        r = int(rng.integers(4, 14))
        yy, xx = np.ogrid[:h, :w]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        img[mask] = np.clip(img[mask].astype(int) - 90, 0, 255).astype(np.uint8)
    return img


def main() -> int:
    from backend.app.dependencies import get_registry
    from backend.app.pipelines import InspectionPipeline
    from backend.domain.density import estimate_density
    from backend.domain.detect.thresholds import resolve_class_conf
    from backend.domain.film_region import FilmRegionCfg as DomainFilmRegionCfg
    from backend.domain.film_region import detect_film_region_trusted as detect_film
    from backend.domain.film_region import film_background_fill
    from backend.domain.gate_adapters import iqi_cfg_from_settings, pseudo_cfg_from_settings
    from backend.domain.iqi import verify_iqi
    from backend.domain.preprocess.metrics import QualityCfg as DomainQualityCfg
    from backend.domain.preprocess.metrics import assess_quality
    from backend.domain.pseudo_defect import screen_pseudo_defects
    from backend.domain.quantify import MaskRefineCfg, get_quantifier
    from backend.domain.stamp import StampCfg, read_stamp_aligned
    from backend.infra.image_loader import load_image

    reg = get_registry()
    pp = reg.preprocessor
    dc = reg.config.detect

    sizes = [
        ("S 1024x1400（不分块）", 1400, 1024),
        ("M 2048x2600（分块 ~4）", 2600, 2048),
        ("L 3000x8000（分块 ~24）", 8000, 3000),
        ("照片 3024x4032（分块 ~12）", 4032, 3024),
    ]
    print(
        f"detector={type(reg.detector).__name__} tile={dc.tile_size} "
        f"trigger={dc.tile_trigger_side} stamp={reg.config.stamp.enabled} "
        f"preprocess={reg.config.preprocess.enabled}"
    )

    # 分段计时收口在循环外定义（stages/timings 每轮清空），闭包不捕循环变量
    stages: list[tuple[str, float]] = []
    timings: dict[str, float] = {}

    def stage(name: str, t0: float) -> float:
        dt = time.perf_counter() - t0
        stages.append((name, dt))
        return time.perf_counter()

    real_reporter_build = reg.reporter.build

    def timed_reporter_build(*a, **kw):
        t0 = time.perf_counter()
        out = real_reporter_build(*a, **kw)
        timings["reporter.build(PDF)"] = time.perf_counter() - t0
        return out

    real_persist = InspectionPipeline._persist_image

    def timed_persist(self, src, image_id, suffix):
        t0 = time.perf_counter()
        out = real_persist(self, src, image_id, suffix)
        timings["persist_image(+SM4)"] = time.perf_counter() - t0
        return out

    real_insert = reg.repository.create_inspection

    def timed_insert(*a, **kw):
        t0 = time.perf_counter()
        out = real_insert(*a, **kw)
        timings["db.create_inspection"] = time.perf_counter() - t0
        return out

    for label, w, h in sizes:
        path = _TMP / f"film_{w}x{h}.png"
        cv2.imwrite(str(path), make_film(w, h))
        stages.clear()
        timings.clear()

        t = time.perf_counter()
        gray, meta = load_image(path)
        t = stage("1 load_image", t)

        fr = reg.config.film_region
        film = (
            detect_film(
                gray,
                DomainFilmRegionCfg(
                    min_area_frac=fr.min_area_frac,
                    max_photo_area_frac=fr.max_photo_area_frac,
                    surround_bright_gray=fr.surround_bright_gray,
                    surround_min_frac=fr.surround_min_frac,
                ),
            )
            if fr.enabled
            else None
        )
        t = stage("2 film_region", t)

        density = float(estimate_density(gray, bit_depth=meta.bit_depth))
        t = stage(f"3a density (D={density:.2f})", t)
        iqi = verify_iqi(
            gray, iqi_cfg_from_settings(reg.config.iqi), iqi_type=reg.config.iqi.type
        )
        t = stage(f"3b iqi (pass={iqi.passed})", t)
        pd = screen_pseudo_defects(gray, pseudo_cfg_from_settings(reg.config.pseudo_defect))
        t = stage(f"3c pseudo (pass={pd.passed})", t)
        quality = assess_quality(gray, DomainQualityCfg(**reg.config.quality.model_dump()))
        t = stage(f"3d quality (q={quality.score:.0f})", t)

        detect_src = film_background_fill(gray, film)
        t = stage("4 background_fill", t)

        denoised = pp.denoise(detect_src)
        enhanced = pp.enhance(denoised, reg.config.preprocess.gamma)
        t = stage("5 preprocess(denoise+enhance)", t)

        stamp = read_stamp_aligned(
            gray,
            StampCfg(
                enabled=reg.config.stamp.enabled,
                min_conf=reg.config.stamp.min_conf,
                max_side=reg.config.stamp.max_side,
            ),
            film=film,
        )
        t = stage(f"6 stamp OCR ({stamp.status})", t)

        infer_conf, class_conf = resolve_class_conf(
            dc.infer_conf,
            dc.class_conf,
            dc.mode,
            recall_scale=dc.recall_conf_scale,
            precision_scale=dc.precision_conf_scale,
        )
        dets = reg.detector.infer(enhanced, conf=infer_conf, iou=dc.infer_iou, class_conf=class_conf)
        t = stage(f"7 detector.infer ({len(dets)} dets)", t)

        quant = get_quantifier(reg.config.detect.quantifier_kind)
        mrc = MaskRefineCfg(**reg.config.mask_refine.model_dump())
        for d in dets:
            quant.quantify(d, 0.1, image=enhanced, cfg=mrc)
        t = stage(f"8 quantify x{len(dets)}", t)

        # 端到端：与前端同一路径（含归档落库 + SM4 副本 + PDF 报告）；
        # 计时壳已装（见循环外），定位分段之外的开销分布。
        e2e_path = _TMP / f"e2e_{w}x{h}.png"
        cv2.imwrite(str(e2e_path), make_film(w, h, seed=8))
        reg.reporter.build = timed_reporter_build  # type: ignore[method-assign]
        InspectionPipeline._persist_image = timed_persist  # type: ignore[method-assign]
        reg.repository.create_inspection = timed_insert  # type: ignore[method-assign]
        t0 = time.perf_counter()
        InspectionPipeline(reg).run_inspection(
            e2e_path,
            pixel_spacing_mm=0.1,
            base_metal_thickness_mm=20.0,
            force=True,
            actor="profiler",
        )
        e2e = time.perf_counter() - t0
        reg.reporter.build = real_reporter_build  # type: ignore[method-assign]
        InspectionPipeline._persist_image = real_persist  # type: ignore[method-assign]
        reg.repository.create_inspection = real_insert  # type: ignore[method-assign]

        total = sum(dt for _, dt in stages)
        print(f"\n== {label}  分段合计 {total:.2f}s ｜ 端到端 {e2e:.2f}s ==")
        for name, dt in sorted(stages, key=lambda s: -s[1]):
            print(f"   {dt:7.2f}s  {100 * dt / total:5.1f}%  {name}")
        for name, dt in sorted(timings.items(), key=lambda s: -s[1]):
            print(f"   [e2e] {dt:7.2f}s  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
