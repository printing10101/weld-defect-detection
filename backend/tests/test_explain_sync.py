"""M6 集成测试：可解释热力图（§12.3）+ 同步适配器（§7.6）。"""

from __future__ import annotations

import base64
from collections.abc import Iterator

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


@pytest.fixture(scope="module", autouse=True)
def _authorized_grader(auth_table) -> Iterator[None]:
    """explain 只走检测不评级，但保持与其他模块一致的 registry 环境。"""
    from backend.app import dependencies as deps

    deps._registry = None
    try:
        yield
    finally:
        deps._registry = None


def _film_png() -> tuple[str, bytes]:
    """噪声底片 + 2 个暗斑（blob 检测器可检出）。"""
    n, h, w = 19, 190, 640
    rng = np.random.default_rng(7)
    img = rng.normal(128.0, 2.0, (h, w)).astype(np.uint8)
    for i in range(n):
        y = round((i + 0.5) / n * h)
        cv2.line(img, (0, y), (w - 1, y), int(128 + 40.0), 3)
    cv2.circle(img, (120, 30), 10, 80, -1)
    cv2.circle(img, (420, 150), 7, 85, -1)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return "film.png", buf.tobytes()


def test_explain_returns_heatmap() -> None:
    """对合成底片生成热力图：200 + base64 PNG（可解码且为有效图像）。"""
    name, data = _film_png()
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/explain",
            files={"image": (name, data, "image/png")},
        )
    assert resp.status_code == 200, resp.text
    b64 = resp.json()["heatmap"]
    raw = base64.b64decode(b64)
    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    assert arr is not None and arr.ndim == 3 and arr.shape[2] == 3


def test_explain_unknown_defect_404() -> None:
    """指定不存在的 defect_id → 404（不静默回退到别的缺陷）。"""
    name, data = _film_png()
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/explain",
            files={"image": (name, data, "image/png")},
            data={"defect_id": "no-such-defect"},
        )
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "DEFECT_NOT_FOUND"


def test_explain_blank_film_404() -> None:
    """无缺陷底片 → 404 NO_DEFECT（无热力图可生成）。"""
    h, w = 190, 640
    img = np.full((h, w), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/explain",
            files={"image": ("blank.png", buf.tobytes(), "image/png")},
        )
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] in ("NO_DEFECT", "DEFECT_NOT_FOUND")


# ---------------------------------------------------------------------------
# §7.6 端边云 v1 LocalAdapter
# ---------------------------------------------------------------------------


def test_local_adapter_contract(tmp_path) -> None:
    """LocalAdapter：push 本地记录、pull 恒空、federate 空操作、pending 计数。"""
    from backend.domain.sync import LocalAdapter

    queue = tmp_path / "sync" / "pending.jsonl"
    adapter = LocalAdapter(queue)

    # v1 本地优先：push 只落本地 JSONL，不发网络
    adapter.push({"image_id": "img_1", "level": "II"})
    adapter.push({"image_id": "img_2", "level": None, "need_review": True})
    assert adapter.pending_count == 2

    # pull：无远端恒空
    assert adapter.pull() == []
    # federate：v3 才实现，v1 空操作不抛错
    assert adapter.federate({"layer": 1}) is None

    # 无路径（纯空操作模式）
    silent = LocalAdapter(None)
    silent.push({"a": 1})
    assert silent.pull() == []
    assert silent.pending_count == 0
