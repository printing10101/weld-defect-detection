"""P2-7：上传大小上限强制（413）。

detect / report 经由 staged_upload 已有 413/415；batch 先前直接 f.file.read 无上限，
本次补齐分块读 + 413。这里用极小的 max_bytes 复现超限路径（避免造 200MiB 文件）。
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.dependencies import get_registry
from backend.app.main import app


def _png_bytes(w: int = 32, h: int = 32) -> bytes:
    import numpy as np

    return __import__("cv2").imencode(".png", np.full((h, w), 128, dtype=np.uint8))[1].tobytes()


def test_detect_413_when_too_large(tmp_path) -> None:
    reg = get_registry()
    original = reg.config.upload.max_bytes
    reg.config.upload.max_bytes = 1  # 任意非空 PNG 必超上限 → 触发 413
    app.dependency_overrides[get_registry] = lambda: reg
    try:
        with TestClient(app) as client:
            r = client.post(
                "/api/v1/detect",
                files={"image": ("big.png", _png_bytes(), "image/png")},
            )
            assert r.status_code == 413
            assert r.json()["detail"]["code"] == "PAYLOAD_TOO_LARGE"
    finally:
        app.dependency_overrides.pop(get_registry, None)
        reg.config.upload.max_bytes = original


def test_batch_413_when_too_large(tmp_path) -> None:
    reg = get_registry()
    original = reg.config.upload.max_bytes
    reg.config.upload.max_bytes = 1
    app.dependency_overrides[get_registry] = lambda: reg
    try:
        with TestClient(app) as client:
            r = client.post(
                "/api/v1/batch",
                files=[("images", ("big.png", _png_bytes(), "image/png"))],
            )
            assert r.status_code == 413
            assert r.json()["detail"]["code"] == "PAYLOAD_TOO_LARGE"
    finally:
        app.dependency_overrides.pop(get_registry, None)
        reg.config.upload.max_bytes = original
