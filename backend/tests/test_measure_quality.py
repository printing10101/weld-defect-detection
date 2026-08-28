"""SNRn / 双丝像质计空间分辨率 测量单测（DB50/T 1807-2025 ）。

合成图像：已知噪声水平验证 SNRn 公式；已知丝径双丝图案验证分辨率读数。
"""

from __future__ import annotations

import io

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.routers.measure import router as measure_router
from backend.domain.measure.image_quality import (
    measure_duplex_wire,
    measure_snr,
)


# ---------------------------------------------------------------------------
# SNRn
# ---------------------------------------------------------------------------


def _uniform_noisy(value: float, sd: float, shape=(512, 512), seed=3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.normal(value, sd, shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_snrn_formula_with_known_srb():
    # 均值 200、σ=1.0 → SNR=200；SRb=0.088mm → SNRn=SNR×(0.088/0.088)=SNR
    img = _uniform_noisy(200.0, 1.0)
    res = measure_snr(img, srb_mm=0.088)
    assert abs(res.snr - 200.0) < 20.0  # MAD 估计允许小偏差
    assert abs(res.snrn - res.snr) < 0.01  # SRb=基准 88µm → 归一化系数为 1
    assert res.srb_estimated is False
    assert res.passed is True


def test_snrn_normalization_direction():
    # SRb=0.176mm（粗一倍）→ SNRn = SNR × 0.5（归一到更细的 88µm 基准）
    img = _uniform_noisy(200.0, 1.0)
    res = measure_snr(img, srb_mm=0.176)
    assert abs(res.snrn - res.snr * 0.5) < 0.01


def test_snrn_worst_block_not_mean():
    # 三块不同噪声：结果取最差块
    img = np.full((512, 512), 128, np.float32)
    rng = np.random.default_rng(5)
    img += rng.normal(0, 1.0, img.shape)
    img[0:128, 0:128] += rng.normal(0, 8.0, (128, 128))  # 最差块
    img = np.clip(img, 0, 255).astype(np.uint8)
    res = measure_snr(img, srb_mm=0.088, n_blocks=3)
    assert res.noise_sd > 3.0  # 最差块口径（若取均值会被平坦块稀释）


def test_snr_estimated_srb_conservative():
    # 无 SRb → 用像素尺寸估计并标记；SRb(=像素尺寸) 通常 ≥88µm → SNRn ≤ SNR
    img = _uniform_noisy(150.0, 1.0)
    res = measure_snr(img, srb_mm=None, pixel_spacing_mm=0.1)
    assert res.srb_estimated is True
    assert res.snrn <= res.snr + 1e-6


def test_snr_missing_calibration_rejected():
    img = _uniform_noisy(150.0, 1.0)
    with pytest.raises(ValueError, match="标定"):
        measure_snr(img, srb_mm=None, pixel_spacing_mm=None)


# ---------------------------------------------------------------------------
# 双丝像质计
# ---------------------------------------------------------------------------


def _duplex_pattern(diameters, pixel_mm=0.02, wires_visible=(True,) * 8) -> np.ndarray:
    """合成双丝图案：丝沿 x 方向（水平），从粗到细向下排布，每对一个丝对。

    对内两丝中心距 = 2×丝径；背景亮（128），丝为暗线（40），丝宽≈丝径。
    带宽 = 6×丝径，保证对内两丝及背景都在本段内。
    """
    band_px = int(6 * max(diameters) / pixel_mm)
    h = band_px * len(diameters)
    w = 200
    img = np.full((h, w), 128.0, np.float32)
    for i, d in enumerate(diameters):
        y0 = i * band_px + band_px // 2
        half_spacing = d / pixel_mm  # 对内半间距（中心距 2d）
        wire_half = max(1, int(round(d / pixel_mm / 2)))  # 丝半宽 ≈ 丝径一半
        if wires_visible[i]:
            for dy in (-half_spacing, half_spacing):
                yy = int(round(y0 + dy))
                img[yy - wire_half : yy + wire_half + 1, :] = 40.0
    img += np.random.default_rng(11).normal(0, 1.5, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_duplex_resolves_all_wires():
    ds = (0.50, 0.40, 0.32, 0.25, 0.20, 0.16, 0.13, 0.10)
    img = _duplex_pattern(ds)  # 全部丝可分辨
    res = measure_duplex_wire(img, pixel_spacing_mm=0.02)
    assert res.resolved[0]["modulation"] > 0.5
    # 全部可分辨 → 最细丝径为读数
    assert res.spatial_resolution_mm == 0.10
    assert res.marginal is False


def test_duplex_finds_first_unresolved():
    ds = (0.50, 0.40, 0.32, 0.25, 0.20, 0.16, 0.13, 0.10)
    # 后三对（0.16/0.13/0.10）不可分辨（不画丝 → M≈0）
    img = _duplex_pattern(ds, wires_visible=(True, True, True, True, True, False, False, False))
    res = measure_duplex_wire(img, pixel_spacing_mm=0.02)
    assert res.spatial_resolution_mm == 0.16
    mods = [r["modulation"] for r in res.resolved]
    assert mods[5] < 0.2 and mods[0] >= 0.2


def test_duplex_rotated_wires():
    ds = (0.50, 0.40, 0.32, 0.25, 0.20, 0.16, 0.13, 0.10)
    img = _duplex_pattern(ds)
    h, w = img.shape
    img = cv2.warpAffine(  # 丝旋转 15°
        img,
        cv2.getRotationMatrix2D((w / 2, h / 2), 15, 1.0),
        (w, h),
        flags=cv2.INTER_LINEAR,
    )
    res = measure_duplex_wire(img, pixel_spacing_mm=0.02, wire_axis_deg=15.0)
    assert res.spatial_resolution_mm in (0.20, 0.16, 0.13, 0.10)  # 旋转插值损失允许近似
    assert abs(res.wire_axis_deg - 15.0) < 0.1


def test_duplex_missing_calibration_rejected():
    img = np.full((100, 100), 128, np.uint8)
    with pytest.raises(ValueError, match="标定"):
        measure_duplex_wire(img, pixel_spacing_mm=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(measure_router, prefix="/api/v1")
    return TestClient(app)


def _png_bytes(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_api_snr(client):
    img = _uniform_noisy(200.0, 1.0)
    r = client.post(
        "/api/v1/measure/snr",
        files={"image": ("f.png", io.BytesIO(_png_bytes(img)), "image/png")},
        data={"srb_mm": "0.088"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["passed"] is True and body["snrn"] > 0


def test_api_snr_with_roi(client):
    img = _uniform_noisy(200.0, 1.0)
    r = client.post(
        "/api/v1/measure/snr",
        files={"image": ("f.png", io.BytesIO(_png_bytes(img)), "image/png")},
        data={"srb_mm": "0.088", "roi": "10,10,128,128"},
    )
    assert r.status_code == 200
    assert r.json()["snrn"] > 0


def test_api_snr_bad_roi(client):
    img = _uniform_noisy(200.0, 1.0)
    r = client.post(
        "/api/v1/measure/snr",
        files={"image": ("f.png", io.BytesIO(_png_bytes(img)), "image/png")},
        data={"srb_mm": "0.088", "roi": "900,900,10,10"},
    )
    assert r.status_code == 422


def test_api_duplex(client):
    ds = (0.50, 0.40, 0.32, 0.25, 0.20, 0.16, 0.13, 0.10)
    img = _duplex_pattern(ds, wires_visible=(True, True, True, True, True, False, False, False))
    r = client.post(
        "/api/v1/measure/duplex-wire",
        files={"image": ("f.png", io.BytesIO(_png_bytes(img)), "image/png")},
        data={"pixel_spacing_mm": "0.02"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["spatial_resolution_mm"] == 0.16


def test_api_duplex_missing_calibration(client):
    img = np.full((200, 200), 128, np.uint8)
    r = client.post(
        "/api/v1/measure/duplex-wire",
        files={"image": ("f.png", io.BytesIO(_png_bytes(img)), "image/png")},
        data={},
    )
    assert r.status_code == 422
