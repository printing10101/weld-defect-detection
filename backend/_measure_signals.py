"""测量三类新信号在真实好片/退化样本上的分布，用于定阈值。临时脚本。"""

import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.getcwd())
RNG = np.random.default_rng(20260810)


def imread_gray(path: str):
    arr = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def degrade(orig, kind):
    g = orig.astype(np.float32)
    if kind == "original":
        return orig
    if kind == "underexposed":
        return np.clip(g * 0.30, 0, 255).astype(np.uint8)
    if kind == "overexposed":
        return np.clip(g * 2.20, 0, 255).astype(np.uint8)
    if kind == "noisy":
        return np.clip(g + RNG.normal(0, 25.0, g.shape).astype(np.float32), 0, 255).astype(np.uint8)
    if kind == "stained":
        out = orig.copy()
        h, w = out.shape
        for _ in range(8):
            cy = int(RNG.integers(0, h))
            cx = int(RNG.integers(0, w))
            r = int(RNG.integers(25, 80))
            cv2.ellipse(out, (cx, cy), (r, r), 0, 0, 360, int(RNG.integers(15, 55)), -1)
        return out
    if kind == "blurred":
        return cv2.GaussianBlur(orig, (21, 21), 0)
    if kind == "badscan":
        g2 = np.clip(g * 0.5, 0, 255)
        out = np.clip(g2 + RNG.normal(0, 15.0, g2.shape).astype(np.float32), 0, 255).astype(
            np.uint8
        )
        h, w = out.shape
        for _ in range(5):
            cy = int(RNG.integers(0, h))
            cx = int(RNG.integers(0, w))
            r = int(RNG.integers(20, 60))
            cv2.ellipse(out, (cx, cy), (r, r), 0, 0, 360, int(RNG.integers(20, 50)), -1)
        return out
    raise ValueError(kind)


def signals(gray):
    g = gray.astype(np.float64)
    lap_var = float(np.var(cv2.Laplacian(gray, cv2.CV_64F)))
    sat = float(np.mean((gray == 0) | (gray == 255)))
    base = cv2.GaussianBlur(gray, (0, 0), sigmaX=25).astype(np.float64)
    resid = np.abs(g - base)
    stain_frac25 = float(np.mean(resid > 25.0))
    stain_frac40 = float(np.mean(resid > 40.0))
    stain_mean = float(np.mean(resid))
    return lap_var, sat, stain_frac25, stain_frac40, stain_mean


VARIANTS = ["original", "underexposed", "overexposed", "noisy", "stained", "blurred", "badscan"]
bases = (
    sorted(glob.glob(os.path.join("图片", "定检", "*.jpg")))[:2]
    + sorted(glob.glob(os.path.join("图片", "*.jpg")))[:2]
)

agg = {v: {"lap": [], "sat": [], "sf25": [], "sf40": [], "sm": []} for v in VARIANTS}
for b in bases:
    base_gray = imread_gray(b)
    if base_gray is None:
        continue
    for kind in VARIANTS:
        try:
            gray = degrade(base_gray, kind)
            lap, sat, sf25, sf40, sm = signals(gray)
            agg[kind]["lap"].append(lap)
            agg[kind]["sat"].append(sat)
            agg[kind]["sf25"].append(sf25)
            agg[kind]["sf40"].append(sf40)
            agg[kind]["sm"].append(sm)
        except Exception as e:  # noqa
            print("err", b, kind, e)

print(
    f"{'variant':<14}{'lap_var':>12}{'sat%':>8}{'stainF25%':>11}{'stainF40%':>11}{'stainMean':>11}"
)
for v in VARIANTS:
    d = agg[v]
    lap = np.mean(d["lap"])
    sat = 100 * np.mean(d["sat"])
    sf25 = 100 * np.mean(d["sf25"])
    sf40 = 100 * np.mean(d["sf40"])
    sm = np.mean(d["sm"])
    print(f"{v:<14}{lap:>12.1f}{sat:>8.2f}{sf25:>11.2f}{sf40:>11.2f}{sm:>11.2f}")

with open("measure_signals.json", "w", encoding="utf-8") as fh:
    json.dump(agg, fh, ensure_ascii=False)
print("\n已落盘 measure_signals.json")
