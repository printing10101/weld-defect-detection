"""频域提示（FPG 移植）的可分性与性能验证。

回答两个决定"这个模块值不值得接进链路"的问题：

A. **频域特征对缺陷区域是否有判别力？**（前提条件）
   用 golden_v3 的标注框取"缺陷 ROI"与"同图背景 ROI"，比较两者的频带能量占比，
   并以现有 :func:`estimate_noise`（拉普拉斯法）作为基线。若频域特征的效应量/AUC
   不低于基线，说明它承载了现有方法之外的信息，作为额外输入通道才有意义。
   同时对比 ``linear`` 与 ``log`` 两种频带划分——真实底片能量高度集中于低频，
   划分方式会显著影响特征的动态范围。

B. **真实底片上能不能用？**（工程可行性）
   在真实 X 光底片上实测耗时与频带能量分布，并考察"高频能量占比"与拉普拉斯
   噪声估计的相关性——高度相关意味着无增量，弱相关意味着有增量。

输出精度说明：真实底片的频带能量占比可低至 1e-5~1e-6 量级，一律用科学计数法
打印——用 ``round(x, 6)`` 会把它们抹成 0.0000，得出"什么都没有"的错误印象。

边界说明：golden_v3 是**合成图**，其缺陷为程序化绘制，频域可分性天然偏高，
结论只能作为"前提条件成立"的证据，不能外推为真实底片上的检测增益。

用法：
    python scripts/bench_spectral_separability.py --out data/reports/spectral_separability.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.domain.preprocess.metrics import estimate_noise
from backend.domain.preprocess.spectral import band_energy_fractions

GOLDEN_DIR = _ROOT / "data" / "eval" / "golden_v3"
REAL_DIR = _ROOT / "定检" / "定检"
N_BANDS = 4
MIN_ROI_SIDE = 10  # ROI 过小时频带统计不稳，且拉普拉斯噪声估计不可靠
SCHEMES = ("linear", "log")


def _imread(path: Path) -> np.ndarray | None:
    """unicode 安全读取灰度图（cv2.imread 在中文路径上返回 None）。"""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def _boxes_px(label_path: Path, w: int, h: int) -> list[tuple[int, int, int, int]]:
    """YOLO 归一化标签 → 像素 (x, y, bw, bh) 列表。"""
    out: list[tuple[int, int, int, int]] = []
    try:
        text = label_path.read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, cx, cy, bw, bh = (float(v) for v in parts[:5])
        x = round((cx - bw / 2) * w)
        y = round((cy - bh / 2) * h)
        out.append((max(0, x), max(0, y), round(bw * w), round(bh * h)))
    return out


def _crop(gray: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, bw, bh = box
    h, w = gray.shape
    return gray[y : min(y + bh, h), x : min(x + bw, w)]


def _overlaps(box: tuple[int, int, int, int], others: list[tuple[int, int, int, int]]) -> bool:
    ax, ay, aw, ah = box
    for bx, by, bw, bh in others:
        if ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah:
            return True
    return False


def _sample_background(
    gray: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    size: tuple[int, int],
    rng: np.random.Generator,
    tries: int = 40,
) -> np.ndarray | None:
    """取一块与所有缺陷框均不重叠的同尺寸背景 ROI。"""
    h, w = gray.shape
    bw, bh = size
    if bw <= 0 or bh <= 0 or bw >= w or bh >= h:
        return None
    for _ in range(tries):
        x = int(rng.integers(0, w - bw + 1))
        y = int(rng.integers(0, h - bh + 1))
        if not _overlaps((x, y, bw, bh), boxes):
            return gray[y : y + bh, x : x + bw]
    return None


def cohens_d(pos: np.ndarray, neg: np.ndarray) -> float:
    """效应量 d = (mean_pos − mean_neg) / 合并标准差。"""
    n1, n2 = pos.size, neg.size
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = float(pos.var(ddof=1)), float(neg.var(ddof=1))
    pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled <= 0.0:
        # 方差全零：均值不同即完美可分，用符号表示方向
        return 0.0 if pos.mean() == neg.mean() else float(np.sign(pos.mean() - neg.mean()) * 99.0)
    return float((pos.mean() - neg.mean()) / pooled)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney U 形式的两类 AUC（无需 sklearn）。"""
    if pos.size == 0 or neg.size == 0:
        return 0.5
    from scipy.stats import rankdata

    allv = np.concatenate([pos, neg])
    ranks = rankdata(allv)
    r_pos = float(ranks[: pos.size].sum())
    u = r_pos - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def _summarize_feature(name: str, pos: np.ndarray, neg: np.ndarray) -> dict:
    a = auc(pos, neg)
    return {
        "feature": name,
        "defect_mean": float(pos.mean()),
        "background_mean": float(neg.mean()),
        "cohens_d": round(cohens_d(pos, neg), 4),
        "auc": round(a, 4),
        "separability": round(abs(a - 0.5) + 0.5, 4),  # 距离 0.5 越远越可分
    }


def part_a_golden(*, max_images: int, scheme: str) -> dict:
    """缺陷 ROI vs 背景 ROI 的频域可分性（合成图，有标注）。"""
    rng = np.random.default_rng(20260914)
    img_dir, lbl_dir = GOLDEN_DIR / "images", GOLDEN_DIR / "labels"
    images = sorted(img_dir.glob("*"))
    if max_images:
        images = images[:max_images]

    feats_defect: dict[str, list[float]] = {f"band{k}": [] for k in range(N_BANDS)}
    feats_bg: dict[str, list[float]] = {f"band{k}": [] for k in range(N_BANDS)}
    noise_defect: list[float] = []
    noise_bg: list[float] = []

    n_pairs = 0
    for img_path in images:
        gray = _imread(img_path)
        if gray is None:
            continue
        h, w = gray.shape
        boxes = _boxes_px(lbl_dir / f"{img_path.stem}.txt", w, h)
        boxes = [b for b in boxes if min(b[2], b[3]) >= MIN_ROI_SIDE]
        if not boxes:
            continue
        for box in boxes:
            roi = _crop(gray, box)
            bg = _sample_background(gray, boxes, (box[2], box[3]), rng)
            if bg is None or bg.shape != roi.shape or roi.size == 0:
                continue
            fd = band_energy_fractions(roi, n_bands=N_BANDS, scheme=scheme)
            fb = band_energy_fractions(bg, n_bands=N_BANDS, scheme=scheme)
            for k in range(N_BANDS):
                feats_defect[f"band{k}"].append(float(fd[k]))
                feats_bg[f"band{k}"].append(float(fb[k]))
            noise_defect.append(estimate_noise(roi))
            noise_bg.append(estimate_noise(bg))
            n_pairs += 1

    rows = [
        _summarize_feature(
            f"band{k} energy frac",
            np.array(feats_defect[f"band{k}"], dtype=np.float64),
            np.array(feats_bg[f"band{k}"], dtype=np.float64),
        )
        for k in range(N_BANDS)
    ]
    rows.append(
        _summarize_feature(
            "laplacian noise (baseline)",
            np.array(noise_defect, dtype=np.float64),
            np.array(noise_bg, dtype=np.float64),
        )
    )
    return {
        "scheme": scheme,
        "n_image_pairs": n_pairs,
        "n_bands": N_BANDS,
        "roi_min_side": MIN_ROI_SIDE,
        "rows": rows,
    }


def _real_stats(files: list[Path], scheme: str) -> dict:
    fr_hi: list[float] = []
    fr_mid: list[float] = []
    fr_lo: list[float] = []
    noises: list[float] = []
    times: list[float] = []
    shapes: list[tuple[int, int]] = []

    for path in files:
        gray = _imread(path)
        if gray is None:
            continue
        t0 = time.perf_counter()
        fr = band_energy_fractions(gray, n_bands=N_BANDS, scheme=scheme)
        times.append(time.perf_counter() - t0)
        fr_lo.append(float(fr[0]))
        fr_mid.append(float(fr[1]))
        fr_hi.append(float(fr[-1]))
        noises.append(estimate_noise(gray))
        shapes.append((int(gray.shape[1]), int(gray.shape[0])))

    if not fr_hi:
        return {"available": False}

    hi = np.array(fr_hi, dtype=np.float64)
    nz = np.array(noises, dtype=np.float64)
    r = float(np.corrcoef(hi, nz)[0, 1]) if hi.size > 1 and hi.std() > 0 and nz.std() > 0 else 0.0
    return {
        "available": True,
        "scheme": scheme,
        "n_images": len(fr_hi),
        "resolution_min": [min(s[0] for s in shapes), min(s[1] for s in shapes)],
        "resolution_max": [max(s[0] for s in shapes), max(s[1] for s in shapes)],
        "low_band_frac_mean": float(np.mean(fr_lo)),
        "mid_band_frac_mean": float(np.mean(fr_mid)),
        "high_band_frac_mean": float(hi.mean()),
        "high_band_frac_std": float(hi.std()),
        "high_band_frac_cv": float(hi.std() / hi.mean()) if hi.mean() > 0 else 0.0,
        "laplacian_noise_mean": float(nz.mean()),
        "pearson_highband_vs_noise": round(r, 4),
        "seconds_per_image_mean": round(float(np.mean(times)), 4),
        "seconds_per_image_max": round(float(np.max(times)), 4),
        "megapixels_per_second": round(
            float(np.mean([s[0] * s[1] for s in shapes]) / max(np.mean(times), 1e-9) / 1e6), 3
        ),
    }


def part_b_real(*, max_images: int) -> dict:
    """真实底片：耗时、频带分布、与拉普拉斯噪声估计的相关性（两种划分对比）。"""
    if not REAL_DIR.is_dir():
        return {"available": False}
    files = sorted(p for p in REAL_DIR.glob("*") if p.suffix.lower() in {".jpg", ".png"})
    if max_images:
        files = files[:max_images]
    if not files:
        return {"available": False}
    return {s: _real_stats(files, s) for s in SCHEMES}


def _print_part_a(res: dict) -> None:
    print(f"\n--- scheme = {res['scheme']} ---")
    print(f"有效配对 ROI 数: {res['n_image_pairs']}    频带数: {res['n_bands']}")
    print(f"{'特征':<30}{'缺陷区':>12}{'背景区':>12}{'效应量d':>10}{'AUC':>8}{'可分性':>9}")
    print("-" * 82)
    for row in res["rows"]:
        print(
            f"{row['feature']:<30}{row['defect_mean']:>12.3e}{row['background_mean']:>12.3e}"
            f"{row['cohens_d']:>10.3f}{row['auc']:>8.3f}{row['separability']:>9.3f}"
        )


def _print_part_b(tag: str, b: dict) -> None:
    if not b.get("available"):
        print(f"  [{tag}] 不可用，跳过")
        return
    print(f"  [{tag}] 图像数 {b['n_images']}  分辨率 {b['resolution_min']} ~ {b['resolution_max']}")
    print(
        f"      低频 {b['low_band_frac_mean']:.5f}  中频 {b['mid_band_frac_mean']:.3e}"
        f"  高频 {b['high_band_frac_mean']:.6e} ± {b['high_band_frac_std']:.3e}"
    )
    print(
        f"      高频占比变异系数 CV = {b['high_band_frac_cv']:.4f}"
        f"  （越小越说明该特征跨图近乎常数、无判别力）"
    )
    print(f"      高频占比 vs 拉普拉斯噪声 相关系数 = {b['pearson_highband_vs_noise']}")
    print(
        f"      单图耗时 {b['seconds_per_image_mean']}s（最慢 {b['seconds_per_image_max']}s）"
        f"  ≈ {b['megapixels_per_second']} MP/s"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=str, default="data/reports/spectral_separability.json")
    ap.add_argument("--max-images", type=int, default=0, help="0 = 全部")
    args = ap.parse_args()

    print("=" * 82)
    print("[A] 缺陷 ROI vs 背景 ROI 的频域可分性（golden_v3，合成图，有标注）")
    print("=" * 82)
    part_a = {s: part_a_golden(max_images=args.max_images, scheme=s) for s in SCHEMES}
    for res in part_a.values():
        _print_part_a(res)

    print()
    print("=" * 82)
    print("[B] 真实底片实测（定检/）")
    print("=" * 82)
    part_b = part_b_real(max_images=args.max_images)
    for scheme, res in part_b.items():
        if isinstance(res, dict):
            _print_part_b(scheme, res)

    out = _ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"part_a_golden": part_a, "part_b_real": part_b}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n结果已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
