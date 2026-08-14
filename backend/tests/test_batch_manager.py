"""M6 单元测试：BatchManager 任务队列（§12.1，不依赖 API）。

用假 pipeline 覆盖：多 worker 并行、失败隔离、状态快照持久化与重启恢复
（断点续跑的可查部分）、取消标记。
"""

from __future__ import annotations

import time
from pathlib import Path

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


def _wait_finished(bm: BatchManager, batch_id: str, timeout_s: float = 15.0) -> dict:
    deadline = time.monotonic() + timeout_s
    last = bm.status(batch_id)
    while time.monotonic() < deadline:
        last = bm.status(batch_id)
        if last["status"] == "finished":
            return last
        time.sleep(0.05)
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
