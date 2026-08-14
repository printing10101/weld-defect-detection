"""预处理测试（§4.3/§4.4）：降噪/增强/边缘/形态学/质量度量/API。"""

from __future__ import annotations

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.domain.preprocess.metrics import QualityCfg as DomainQualityCfg
from backend.domain.preprocess.metrics import (
    assess_quality,
    brisque_features,
    estimate_noise,
)
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
    # §4.4 质量门禁字段
    assert "quality" in body
    assert "score" in body["quality"] and "passed" in body["quality"]
    assert "metrics" in body["quality"]


def _structured(size: int = 256, seed: int = 2) -> np.ndarray:
    """带结构的清晰图：背景 + 矩形 + 圆，提供对比度/锐度。"""
    rng = np.random.default_rng(seed)
    img = np.clip(rng.normal(128.0, 4.0, (size, size)), 0, 255).astype(np.uint8)
    cv2.rectangle(img, (40, 40), (size - 60, size - 60), 80, 3)
    cv2.circle(img, (size // 2, size // 2), 30, 200, -1)
    return img


def test_brisque_features_shape() -> None:
    feat = brisque_features(_structured())
    assert feat.shape == (36,)
    assert np.isfinite(feat).all()


def test_brisque_features_discriminative() -> None:
    clean = _structured()
    blur = cv2.GaussianBlur(clean, (21, 21), 10)
    noisy = np.clip(np.random.default_rng(9).normal(128.0, 35.0, clean.shape), 0, 255).astype(
        np.uint8
    )
    fc, fb, fn = brisque_features(clean), brisque_features(blur), brisque_features(noisy)
    # 失真越重，与清晰图特征距离应越大（模糊 > 噪声，定性即可）
    assert np.linalg.norm(fc - fb) > np.linalg.norm(fc - fn) * 0.5


def test_assess_quality_clean_beats_degraded() -> None:
    clean = _structured()
    blur = cv2.GaussianBlur(clean, (21, 21), 12)
    noisy = np.clip(np.random.default_rng(5).normal(128.0, 40.0, clean.shape), 0, 255).astype(
        np.uint8
    )
    qc = DomainQualityCfg()
    sc, sb, sn = (
        assess_quality(clean, qc).score,
        assess_quality(blur, qc).score,
        assess_quality(noisy, qc).score,
    )
    assert sc > sb  # 清晰 > 模糊（锐度下降）
    assert sc > sn  # 清晰 > 强噪声（噪声上升）


def test_assess_quality_report_fields() -> None:
    rep = assess_quality(_structured(), DomainQualityCfg())
    assert 0.0 <= rep.score <= 100.0
    assert isinstance(rep.passed, bool)
    assert "noise_score" in rep.metrics and "sharpness_score" in rep.metrics
    assert "brisque_features" in rep.metrics and len(rep.metrics["brisque_features"]) == 36
    assert rep.brisque_features is not None and rep.brisque_features.shape == (36,)


def test_quality_cfg_hard_gates_defaults() -> None:
    """三类硬门禁阈值默认值需与标定结论一致（§T8 同步的单一真源）。"""
    qc = DomainQualityCfg()
    assert qc.blur_lap_bad == 30.0
    assert qc.exposure_entropy_bad == 0.62
    assert qc.stain_smooth_bad == 0.15


def test_hard_gate_blur_blocks() -> None:
    """失焦(强高斯模糊)应触发 blur_severe → passed=False；RQI 复合分不被硬性门禁扰动。"""
    clean = _structured()
    blur = cv2.GaussianBlur(clean, (21, 21), 12)
    rep = assess_quality(blur, DomainQualityCfg())
    assert rep.metrics["blur_severe"] is True
    assert rep.metrics["hard_fail"] is True
    assert rep.passed is False


def test_hard_gate_exposure_blocks() -> None:
    """过曝(×2.2 裁切)应触发 exposure_severe → passed=False。"""
    clean = _structured()
    over = np.clip(clean.astype(np.float32) * 2.2, 0, 255).astype(np.uint8)
    rep = assess_quality(over, DomainQualityCfg())
    assert rep.metrics["exposure_severe"] is True
    assert rep.passed is False


def _rich_film(size: int = 256, seed: int = 3) -> np.ndarray:
    """高熵合成底片：横渐变 + 结构，模拟真实底片丰富色调（熵>0.7），
    用于验证"好底片不触发任何硬门禁"（_structured 是均匀灰底，熵偏低会被曝光门禁正确拦截）。"""
    rng = np.random.default_rng(seed)
    base = np.tile(np.linspace(40.0, 220.0, size), (size, 1)).astype(np.float32)
    base += rng.normal(0.0, 3.0, (size, size))
    img = np.clip(base, 0, 255).astype(np.uint8)
    cv2.rectangle(img, (40, 40), (size - 60, size - 60), 90, 3)
    cv2.circle(img, (size // 2, size // 2), 30, 210, -1)
    return img


def test_hard_gate_clean_passes() -> None:
    """高熵清晰底片不应触发任何硬门禁，passed=True。"""
    rep = assess_quality(_rich_film(), DomainQualityCfg())
    assert rep.metrics["hard_fail"] is False
    assert rep.metrics["blur_severe"] is False
    assert rep.metrics["exposure_severe"] is False
    assert rep.metrics["stain_severe"] is False
    assert rep.passed is True
