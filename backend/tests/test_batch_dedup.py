"""集成测试：批量上传查重（批内重复 + 历史已检影像重复 + 人工复核）。

覆盖：
- 批内同内容文件 → 批次 awaiting_review 暂缓，duplicates 标注批内重复；
- 与历史已检影像同内容 → kind=history 带首检摘要；content_hash 落库；
- 人工 resolve：skip 不跑（cancelled）、keep 照常检测；全 skip 批次直接完成；
- dedup=false 时行为与旧版一致（重复不拦截直接跑）；
- resolve 状态校验：非暂缓批 409 / 未知批次 404。
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


def _film_png(seed: int) -> tuple[str, bytes]:
    """合成底片（seed 不同 → 内容不同，互不命中查重）。"""
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


def _submit(client: TestClient, payloads: list[bytes], names: list[str] | None = None) -> dict:
    files = [
        ("images", ((names or [f"f{i}.png" for i in range(len(payloads))])[i], data, "image/png"))
        for i, data in enumerate(payloads)
    ]
    resp = client.post(
        "/api/v1/batch",
        files=files,
        data={"pixel_spacing_mm": "0.1", "base_metal_thickness_mm": "20", "force": "true"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _wait_batch(client: TestClient, batch_id: str, timeout_s: float = 60.0) -> dict:
    """轮询批量直到 finished 或超时。"""
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


def test_batch_dedup_in_batch_hold_then_skip() -> None:
    """同内容两张 → awaiting_review；skip 第二张 → done=1/cancelled=1，原版照常出结果。"""
    _, data = _film_png(seed=101)
    with TestClient(app) as client:
        body = _submit(client, [data, data], ["a.png", "b.png"])
        assert body["status"] == "awaiting_review"
        assert body["total"] == 2
        dups = body["duplicates"]
        assert len(dups) == 1
        dup = dups[0]
        assert dup["kind"] == "batch"
        assert dup["image_name"] == "b.png"
        assert dup["duplicate_of"] == "a.png"

        # 未复核前批次不推进：任务保持 pending
        st = client.get(f"/api/v1/batch/{body['batch_id']}").json()
        assert st["status"] == "awaiting_review"
        assert all(t["status"] == "pending" for t in st["tasks"])

        r = client.post(
            f"/api/v1/batch/{body['batch_id']}/dedup/resolve",
            json={"decisions": [{"task_id": dup["task_id"], "action": "skip"}]},
        )
        assert r.status_code == 200, r.text
        assert r.json() == {"ok": True, "skipped": 1, "kept": 1}

        finished = _wait_batch(client, body["batch_id"])
    assert finished["done"] == 1
    assert finished["cancelled"] == 1
    by_name = {t["image_name"]: t for t in finished["tasks"]}
    assert by_name["a.png"]["status"] == "done"
    assert by_name["b.png"]["status"] == "cancelled"
    assert "重复" in (by_name["b.png"]["error"] or "")


def test_batch_dedup_resolve_keep_runs_both() -> None:
    """人工确认 keep → 两张同内容影像都执行检测。"""
    _, data = _film_png(seed=102)
    with TestClient(app) as client:
        body = _submit(client, [data, data])
        dup = body["duplicates"][0]
        r = client.post(
            f"/api/v1/batch/{body['batch_id']}/dedup/resolve",
            json={"decisions": [{"task_id": dup["task_id"], "action": "keep"}]},
        )
        assert r.status_code == 200
        assert r.json() == {"ok": True, "skipped": 0, "kept": 2}
        finished = _wait_batch(client, body["batch_id"])
    assert finished["done"] == 2
    assert finished["failed"] == 0


def test_batch_dedup_against_history() -> None:
    """首次检测落库后，同内容再传 → kind=history 且带首检摘要；全 skip → 0 检测。"""
    _, data = _film_png(seed=103)
    with TestClient(app) as client:
        first = _submit(client, [data])
        assert first["status"] == "running"  # 首次无重复，直接执行
        finished = _wait_batch(client, first["batch_id"])
        first_image_id = finished["tasks"][0]["image_id"]
        assert first_image_id

        # content_hash 已随检查落库
        from backend.app import dependencies as deps

        reg = deps.get_registry()
        img = reg.repository.get_image(first_image_id)
        assert img and img["content_hash"]

        second = _submit(client, [data])
        assert second["status"] == "awaiting_review"
        dup = second["duplicates"][0]
        assert dup["kind"] == "history"
        assert dup["duplicate_of"] == first_image_id
        assert dup["history"]["image_id"] == first_image_id

        # 全部跳过：批次直接完成，未重复出片
        r = client.post(f"/api/v1/batch/{second['batch_id']}/dedup/resolve", json={"decisions": []})
        assert r.status_code == 200
        assert r.json() == {"ok": True, "skipped": 1, "kept": 0}
        st = client.get(f"/api/v1/batch/{second['batch_id']}").json()
    assert st["status"] == "finished"
    assert st["cancelled"] == 1
    assert st["done"] == 0


def test_batch_dedup_disabled_runs_immediately() -> None:
    """dedup=false：同内容重复不拦截，行为与旧版一致（提交即运行）。"""
    from backend.app import dependencies as deps

    reg = deps.get_registry()
    orig = reg.config.batch.dedup
    reg.config.batch.dedup = False
    try:
        _, data = _film_png(seed=104)
        with TestClient(app) as client:
            body = _submit(client, [data, data])
            assert body["status"] == "running"
            assert body["duplicates"] == []
            finished = _wait_batch(client, body["batch_id"])
        assert finished["done"] == 2
    finally:
        reg.config.batch.dedup = orig


def test_batch_dedup_resolve_state_guards() -> None:
    """resolve 校验：非暂缓批 409；未知批次 404；cancel 暂缓批 → 全任务 cancelled。"""
    _, data = _film_png(seed=105)
    with TestClient(app) as client:
        # 无重复批直接 running（可能很快 finished）→ resolve 一律 409
        body = _submit(client, [data])
        r = client.post(f"/api/v1/batch/{body['batch_id']}/dedup/resolve", json={"decisions": []})
        assert r.status_code == 409
        assert r.json()["detail"]["code"] == "NOT_AWAITING_REVIEW"

        r404 = client.post("/api/v1/batch/nope/dedup/resolve", json={"decisions": []})
        assert r404.status_code == 404

        # 暂缓批整批取消：不复核直接放弃
        held = _submit(client, [data, data])
        assert held["status"] == "awaiting_review"
        r = client.post(f"/api/v1/batch/{held['batch_id']}/cancel")
        assert r.status_code == 200
        st = client.get(f"/api/v1/batch/{held['batch_id']}").json()
    assert st["status"] == "finished"
    assert st["cancelled"] == 2
