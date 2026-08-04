"""预处理测试（§4.3/§4.4）：降噪/增强/边缘/形态学/质量度量/API。"""
from __future__ import annotations

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.domain.preprocess.metrics import estimate_noise
from backend.domain.preprocess.pipeline import OpencvPreprocessor


def _noisy_flat(size: int = 128, noise: float = 8.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(128.0, noise, (size, size)), 0, 255).astype(np.uint8)


def test_denoise_reduces_noise() -> None:
    pp = OpencvPreprocessor()
    img = _noisy_flat()
    out = pp.denoise(img)
    assert estimate_noise(out) < estimate_noise(img) * 0.6


def test_gamma_brightens_dark() -> None:
    pp = OpencvPreprocessor()
    dark = np.full((64, 64), 60, np.uint8)
    out = pp.enhance(dark, gamma=0.7)  # gamma<1 提亮低黑度底片
    assert out.mean() > 80


def test_edges_detect_line() -> None:
    pp = OpencvPreprocessor()
    img = np.full((100, 100), 128, np.uint8)
    cv2.line(img, (20, 50), (80, 50), 200, 2)
    e = pp.edges(img)
    assert e.max() == 255
    assert e[45:55, 45:55].max() == 255  # 线邻域检出边缘
    assert e[5, 5] == 0  # 空白区无伪边缘


def test_edges_roi_restricts() -> None:
    pp = OpencvPreprocessor()
    img = np.full((100, 100), 128, np.uint8)
    cv2.line(img, (20, 50), (80, 50), 200, 2)
    e = pp.edges(img, roi=(0, 0, 100, 20))  # 不包含线
    assert e.max() == 0


def test_morph_closes_gap() -> None:
    pp = OpencvPreprocessor()
    edges = np.zeros((50, 100), np.uint8)
    edges[24:27, 10:46] = 255  # 3px 厚（≥开运算核，开运算不腐蚀光）
    edges[24:27, 54:90] = 255  # 8px 断口
    closed = pp.morph(edges, 3, 9)
    assert closed[25, 50] == 255  # 闭运算修复断裂


def test_preprocess_api() -> None:
    rng = np.random.default_rng(1)
    img = np.clip(rng.normal(128.0, 8.0, (128, 128)), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/preprocess",
            files={"image": ("t.png", buf.tobytes(), "image/png")},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert "psnr_db" in body["metrics"]
    assert "ssim" in body["metrics"]
    assert body["metrics"]["noise_out"] < body["metrics"]["noise_in"]
    assert body["thumbnail"]
