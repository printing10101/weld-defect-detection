"""单元测试：BatchManager 任务队列。

用假 pipeline 覆盖：多 worker 并行、失败隔离、状态快照持久化与重启恢复
（断点续跑的可查部分）、取消标记。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from backend.app.batch_queue import BatchItem, BatchManager


class _FakePipeline:
    """模拟 InspectionPipeline：bad 路径抛错（失败隔离验证），其余成功。"""

    def run_inspection(self, **options):
        img = str(options["image_path"])
        if "bad" in img:
            raise ValueError("simulated failure")
        return {
            "image_id": img,
            "report_id": "r1",
            "joint_level": None,
            "need_review": True,
        }


def _factory() -> _FakePipeline:
    return _FakePipeline()


def _wait_finished(bm: BatchManager, batch_id: str, timeout_s: float = 15.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last = bm.status(batch_id)
    while time.monotonic() < deadline:
        last = bm.status(batch_id)
        assert last is not None
        if last["status"] == "finished":
            return last
        time.sleep(0.05)
    assert last is not None
    raise AssertionError(f"batch 未完成: {last}")


def test_batch_manager_failure_isolation_and_snapshot(tmp_path: Path) -> None:
    """失败隔离 + 状态快照落盘 + 重启恢复（断点续跑可查部分）。"""
    bm = BatchManager(_factory, workers=2, per_image_estimate_sec=1.0, batch_dir=tmp_path / "batch")
    good = tmp_path / "good.png"
    bad = tmp_path / "bad.png"
    good.write_bytes(b"x")
    bad.write_bytes(b"x")
    try:
        batch_id = bm.submit([BatchItem(good), BatchItem(bad)])
        finished = _wait_finished(bm, batch_id)
        assert finished["status"] == "finished"
        assert finished["done"] == 1
        assert finished["failed"] == 1
        assert len(finished["tasks"]) == 2

        # 快照已落盘，新实例可恢复（模拟进程重启后的可查性）
        snap = tmp_path / "batch" / f"{batch_id}.json"
        assert snap.exists()
        bm2 = BatchManager(
            _factory, workers=1, per_image_estimate_sec=1.0, batch_dir=tmp_path / "batch"
        )
        try:
            bm2._load_existing()
            restored = bm2.status(batch_id)
            assert restored is not None
            assert restored["status"] == "finished"
            assert restored["done"] == 1
        finally:
            bm2.shutdown()
    finally:
        bm.shutdown()


def test_batch_manager_cancel_marks_pending(tmp_path: Path) -> None:
    """取消后未启动任务不执行（pending 不进入 running/done）。"""
    bm = BatchManager(_factory, workers=1, per_image_estimate_sec=1.0, batch_dir=tmp_path / "batch")
    files = []
    for i in range(3):
        p = tmp_path / f"f{i}.png"
        p.write_bytes(b"x")
        files.append(BatchItem(p))
    try:
        batch_id = bm.submit(files)
        assert bm.cancel(batch_id) is True
        finished = _wait_finished(bm, batch_id)
        # 取消标记生效：没有任务既 running 又 done——取消后最多 1 个 done（已启动的跑完）
        done = sum(1 for t in finished["tasks"] if t["status"] == "done")
        cancelled = sum(1 for t in finished["tasks"] if t["status"] == "cancelled")
        assert done + cancelled == 3  # 全部收敛到终态
        assert cancelled >= 1  # 至少一个被取消（未启动的）
    finally:
        bm.shutdown()


def test_batch_manager_empty_rejected(tmp_path: Path) -> None:
    bm = BatchManager(_factory, workers=1, per_image_estimate_sec=1.0, batch_dir=tmp_path / "batch")
    try:
        try:
            bm.submit([])
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError("空批次必须拒绝")
    finally:
        bm.shutdown()


def test_batch_manager_cleans_staging(tmp_path: Path) -> None:
    """P1-3：批次完成/取消后清理上传暂存目录（防 data/tmp 持续增长）。"""
    bm = BatchManager(_factory, workers=1, per_image_estimate_sec=1.0, batch_dir=tmp_path / "batch")
    staging = tmp_path / "staging_batch_x"
    staging.mkdir(parents=True, exist_ok=True)
    p = staging / "upload.png"
    p.write_bytes(b"x")
    try:
        batch_id = bm.submit([BatchItem(p, image_name="upload.png", cleanup_dir=staging)])
        finished = _wait_finished(bm, batch_id)
        assert finished["status"] == "finished"
        assert not staging.exists(), "批次完成后暂存目录应被清理"
    finally:
        bm.shutdown()


def test_batch_manager_retry_reruns_failed_and_cleans(tmp_path: Path) -> None:
    """断点续跑：failed 任务重跑成功；暂存目录在重跑前保留、重跑成功后清理。"""

    class _Flaky:
        def __init__(self, counter: dict) -> None:
            self._counter = counter

        def run_inspection(self, **options):
            self._counter["n"] += 1
            if self._counter["n"] == 1:
                raise ValueError("first attempt fails")
            return {
                "image_id": str(options["image_path"]),
                "report_id": "r1",
                "joint_level": None,
                "need_review": False,
            }

    counter: dict = {"n": 0}
    bm = BatchManager(
        lambda: _Flaky(counter), workers=1, per_image_estimate_sec=1.0, batch_dir=tmp_path / "batch"
    )
    staging = tmp_path / "staging_retry"
    staging.mkdir(parents=True, exist_ok=True)
    p = staging / "a.png"
    p.write_bytes(b"x")
    try:
        batch_id = bm.submit([BatchItem(p, image_name="a.png", cleanup_dir=staging)])
        finished = _wait_finished(bm, batch_id)
        assert finished["failed"] == 1
        assert staging.exists(), "存在 failed 任务时暂存目录应保留（供 retry 重跑）"

        # 重跑失败任务 → 全部成功 → 暂存目录清理
        assert bm.retry(batch_id) == 1
        finished2 = _wait_finished(bm, batch_id)
        assert finished2["failed"] == 0
        assert finished2["done"] == 1
        assert not staging.exists(), "重跑全部成功后暂存目录应清理"
    finally:
        bm.shutdown()


def test_batch_manager_retry_unknown_raises(tmp_path: Path) -> None:
    bm = BatchManager(_factory, workers=1, per_image_estimate_sec=1.0, batch_dir=tmp_path / "batch")
    try:
        try:
            bm.retry("nope")
        except KeyError:
            pass
        else:  # pragma: no cover
            raise AssertionError("未知批次 retry 必须抛 KeyError")
    finally:
        bm.shutdown()


def test_batch_manager_retry_none_left_returns_zero(tmp_path: Path) -> None:
    """无 failed/cancelled 任务时 retry 返回 0。"""
    bm = BatchManager(_factory, workers=1, per_image_estimate_sec=1.0, batch_dir=tmp_path / "batch")
    p = tmp_path / "ok.png"
    p.write_bytes(b"x")
    try:
        batch_id = bm.submit([BatchItem(p)])
        _wait_finished(bm, batch_id)
        assert bm.retry(batch_id) == 0
    finally:
        bm.shutdown()


def test_batch_manager_list_orders_recent_first(tmp_path: Path) -> None:
    """批次列表：最近提交在前，且带进度/计数摘要。"""
    bm = BatchManager(_factory, workers=1, per_image_estimate_sec=1.0, batch_dir=tmp_path / "batch")
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    a.write_bytes(b"x")
    b.write_bytes(b"x")
    try:
        id1 = bm.submit([BatchItem(a)])
        id2 = bm.submit([BatchItem(b)])
        rows = bm.list()
        assert [r["batch_id"] for r in rows] == [id2, id1]
        assert rows[0]["total"] == 1
        assert rows[0]["status"] == "running" or rows[0]["status"] == "finished"
        assert 0.0 <= rows[0]["progress"] <= 1.0
    finally:
        bm.shutdown()


def _seed_finished_snapshots(
    bm: BatchManager, batch_dir: Path, n: int, *, retryable: bool = False
) -> list[str]:
    """写入 n 个'已终结'批次快照（内存中同步登记），供磁盘剪枝测试。

    retryable=True 时第 0 个批次含 failed 任务（模拟可断点续跑，须恒保留）。
    mtime 随写入递增，保证 sorted(st_mtime) 是稳定的创建序。
    """
    ids: list[str] = []
    for i in range(n):
        bid = f"b{i:04d}"
        batch = {
            "batch_id": bid,
            "status": "finished",
            "total": 1,
            "done": 0 if retryable and i == 0 else 1,
            "failed": 1 if retryable and i == 0 else 0,
            "cancelled": 0,
            "tasks": [
                {
                    "task_id": f"{bid}_t",
                    "status": "failed" if retryable and i == 0 else "done",
                    "image_name": f"{i}.png",
                    "image_path": str(batch_dir / f"{i}.png"),
                }
            ],
            "created_at": "2026-01-01T00:00:00",
        }
        with bm._lock:  # noqa: SLF001 - 测试写入
            bm._batches[bid] = batch
        bm._persist(batch)
        ids.append(bid)
    return ids


def test_batch_manager_prunes_old_finished_snapshots(tmp_path: Path) -> None:
    """磁盘快照保留上限：超限只剪最旧的'已完成且无 retry 价值'文件。"""
    batch_dir = tmp_path / "batch"
    bm = BatchManager(
        _factory,
        workers=1,
        per_image_estimate_sec=1.0,
        batch_dir=batch_dir,
        max_retained_batches=50,  # 大内存上限，保持全部批次在内存以便被剪
        max_retained_snapshot_files=3,
    )
    try:
        _seed_finished_snapshots(bm, batch_dir, 6)
        files = sorted(p.name for p in batch_dir.glob("*.json"))
        assert len(files) == bm._max_snapshot_files == 3  # noqa: SLF001
        # 最旧三个被剪，保留 b0003/b0004/b0005
        assert files == ["b0003.json", "b0004.json", "b0005.json"]
    finally:
        bm.shutdown()


def test_batch_manager_prune_keeps_retryable_snapshots(tmp_path: Path) -> None:
    """磁盘剪枝不得破坏断点续跑：failed 批次（含 retry 价值）快照恒保留。"""
    batch_dir = tmp_path / "batch"
    bm = BatchManager(
        _factory,
        workers=1,
        per_image_estimate_sec=1.0,
        batch_dir=batch_dir,
        max_retained_batches=50,
        max_retained_snapshot_files=2,
    )
    try:
        for i in range(4):
            bid = f"r{i}"
            batch = {
                "batch_id": bid,
                "status": "finished",
                "total": 2,
                "done": 1,
                "failed": 0 if i else 1,
                "cancelled": 0,
                "tasks": [
                    {"task_id": f"{bid}_a", "status": "done", "image_name": "a.png"},
                    {
                        "task_id": f"{bid}_b",
                        "status": "failed" if i == 0 else "done",
                        "image_name": "b.png",
                    },
                ],
                "created_at": "2026-01-01T00:00:00",
            }
            with bm._lock:  # noqa: SLF001
                bm._batches[bid] = batch
            bm._persist(batch)
        # 保留上限 = 2；r0 含 failed（可重试）必须保留 → 只保留 r0 + 最新的一个安全批次
        names = sorted(p.name for p in batch_dir.glob("*.json"))
        assert "r0.json" in names
        assert len(names) == bm._max_snapshot_files  # noqa: SLF001
    finally:
        bm.shutdown()
