"""M6 集成测试：批量任务队列（§12.1）。

覆盖：提交（多图异步入队）、进度/结果查询、完成态计数、失败隔离
（单图损坏不影响其余）、取消（未启动任务不执行）、batch 上限 422。
依赖 conftest 的测试环境隔离（DB/影像/报告目录 + authorized 表注入）。
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


@pytest.fixture(scope="module", autouse=True)
def _authorized_grader(auth_table) -> Iterator[None]:
    """批量全链路需要正常评级/出片：注入 authorized 表 + 放宽黑度 + 关质量门禁。"""
    from backend.app import dependencies as deps
    from backend.domain.grade.nb47013 import Nb47013Grader
    from backend.domain.standards.tables.loader import load_standard_tables

    deps._registry = None
    reg = deps.get_registry()
    reg.grader = Nb47013Grader(load_standard_tables("NB/T47013.2-2015", filename=str(auth_table)))
    orig_low = reg.config.density.low
    orig_block = reg.config.quality.block_on_quality
    reg.config.density.low = 0.0
    reg.config.quality.block_on_quality = False
    try:
        yield
    finally:
        reg.config.density.low = orig_low
        reg.config.quality.block_on_quality = orig_block
        deps._registry = None


def _film_png(seed: int = 0) -> tuple[str, bytes]:
    """合成底片：噪声 + 19 根 IQI 丝 + 2 个暗斑（与 report 测试同构）。"""
    n, h, w = 19, 190, 640
    rng = np.random.default_rng(seed)
    img = rng.normal(128.0, 2.0, (h, w)).astype(np.uint8)
    for i in range(n):
        y = round((i + 0.5) / n * h)
        cv2.line(img, (0, y), (w - 1, y), int(128 + 40.0), 3)
    cv2.circle(img, (120, 30), 10, 80, -1)
    cv2.circle(img, (420, 150), 7, 85, -1)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return "film.png", buf.tobytes()


def _wait_batch(client: TestClient, batch_id: str, timeout_s: float = 60.0) -> dict:
    """轮询批量直到 finished 或超时（异步任务需等待）。"""
    deadline = time.monotonic() + timeout_s
    last: dict = {}
    while time.monotonic() < deadline:
        resp = client.get(f"/api/v1/batch/{batch_id}")
        assert resp.status_code == 200
        last = resp.json()
        if last["status"] == "finished":
            return last
        time.sleep(0.2)
    raise AssertionError(f"batch {batch_id} 未在 {timeout_s}s 内完成: {last}")


def test_batch_submit_progress_and_done() -> None:
    """提交 2 张 → 立即返回 batch_id；轮询至 finished：done=2、progress=1.0、任务有结果。"""
    name, data = _film_png(seed=1)
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/batch",
            files=[("images", (name, data, "image/png")), ("images", (name, data, "image/png"))],
            data={
                "pixel_spacing_mm": "0.1",
                "base_metal_thickness_mm": "20",
                "force": "true",  # 合成底片黑度不达标：force 出片但不定级
            },
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    batch_id = body["batch_id"]
    assert body["total"] == 2
    assert body["estimated_sec"] > 0

    finished = _wait_batch(client, batch_id)
    assert finished["status"] == "finished"
    assert finished["done"] == 2
    assert finished["failed"] == 0
    assert finished["progress"] == 1.0
    assert len(finished["tasks"]) == 2
    for task in finished["tasks"]:
        assert task["status"] == "done"
        assert task["image_id"]  # 每张图已落库
        assert task["report_id"]  # 每张图已出报告


def test_batch_failure_isolation() -> None:
    """失败隔离：1 张损坏 + 1 张正常 → 损坏的 failed，正常的 done，批次整体完成。"""
    name, data = _film_png(seed=2)
    bad_name, bad_data = "broken.png", b"not-an-image-bytes"
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/batch",
            files=[
                ("images", (name, data, "image/png")),
                ("images", (bad_name, bad_data, "image/png")),
            ],
            data={
                "pixel_spacing_mm": "0.1",
                "base_metal_thickness_mm": "20",
                "force": "true",
            },
        )
    assert resp.status_code == 200, resp.text
    batch_id = resp.json()["batch_id"]

    finished = _wait_batch(client, batch_id)
    assert finished["status"] == "finished"
    assert finished["done"] == 1
    assert finished["failed"] == 1
    assert finished["progress"] == 1.0
    by_name = {t["image_name"]: t for t in finished["tasks"]}
    assert by_name["film.png"]["status"] == "done"
    assert by_name["broken.png"]["status"] == "failed"
    assert by_name["broken.png"]["error"]  # 失败原因保留


def test_batch_cancel() -> None:
    """取消已提交批次 → ok=True；未启动任务标记 cancelled。"""
    name, data = _film_png(seed=3)
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/batch",
            files=[("images", (name, data, "image/png"))],
            data={"pixel_spacing_mm": "0.1", "base_metal_thickness_mm": "20", "force": "true"},
        )
    batch_id = resp.json()["batch_id"]
    with TestClient(app) as client:
        resp = client.post(f"/api/v1/batch/{batch_id}/cancel")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    # 等任务收敛（cancelled 或 done），避免 worker 线程在测试退出后仍写日志
    _wait_batch(client, batch_id)


def test_batch_unknown_id_404() -> None:
    with TestClient(app) as client:
        resp = client.get("/api/v1/batch/nope")
    assert resp.status_code == 404


def test_batch_too_many_images_rejected() -> None:
    """超过单批上限 → 422（防一次性打爆资源）。"""
    from backend.app import dependencies as deps

    reg = deps.get_registry()
    orig = reg.config.batch.max_per_batch
    reg.config.batch.max_per_batch = 1
    try:
        name, data = _film_png(seed=4)
        with TestClient(app) as client:
            resp = client.post(
                "/api/v1/batch",
                files=[
                    ("images", (name, data, "image/png")),
                    ("images", (name, data, "image/png")),
                ],
                data={"pixel_spacing_mm": "0.1", "base_metal_thickness_mm": "20"},
            )
        assert resp.status_code == 422
        assert resp.json()["detail"]["code"] == "BATCH_TOO_LARGE"
    finally:
        reg.config.batch.max_per_batch = orig
