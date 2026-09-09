"""批量任务队列。

BatchManager 用线程池多 worker 并行跑 ``InspectionPipeline.run_inspection``
单图全链路（校验→预处理→检测→判定→落库→报告），支持以下能力：
- 任务队列：提交后逐图入队，异步执行（接口立即返回 batch_id）；
- 多 worker 并行推理：``ThreadPoolExecutor``（worker 数可配）；
- 进度可视化：``status`` 返回 done/total/failed + 每任务明细；
- 失败隔离：单图异常捕获记 failed + error，不影响其余任务；
- 断点续跑：批次状态以 JSON 快照落盘（data/batch/{batch_id}.json），
  进程重启后 ``status`` 仍可读历史与未完成标记（v1 不自动重跑，
  已完成任务结果已入库，可重新提交未完成项继续）；
- 取消：``cancel`` 标记批次取消，未启动任务不再执行（running 任务
  线程不可抢占，v1 等待其自然结束，结果保留）。

线程安全：所有状态变更在 ``_lock`` 内；worker 只读快照对象并回写结果。
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.infra.timeutil import fmt_naive_utc

_LOG = logging.getLogger("scandetection.batch")


@dataclass
class BatchItem:
    """单个批量任务：影像路径 + run_inspection 关键字参数。"""

    image_path: Path
    options: dict[str, Any] = field(default_factory=dict)
    image_name: str | None = None  # 展示用原始文件名（缺省取路径 basename）
    cleanup_dir: Path | None = None  # 批次专属暂存目录（任务完成后整体清理，P1-3）
    content_sha256: str | None = None  # 文件内容摘要（查重索引 + 落库 content_hash）
    dup_kind: str | None = (
        None  # 重复类型：history=与历史已检影像重复 | batch=批内重复 | None=不重复
    )
    dup_ref: str | None = None  # 重复对象描述：批内原文件名或历史 image_id（复核界面展示）
    dup_history: dict[str, Any] | None = None  # kind=history 时的历史影像摘要（复核界面展示）


@dataclass
class BatchTaskState:
    """单个任务运行时状态。

    image_path / options 为断点续跑（retry）保留：failed/cancelled 任务
    重跑时需要原图与原始公共参数（快照持久化后进程重启也可用）。
    """

    task_id: str
    image_name: str
    image_path: str = ""  # 原图绝对路径（重跑用）
    options: dict[str, Any] = field(default_factory=dict)  # run_inspection 公共参数（重跑用）
    status: str = "pending"  # pending | running | done | failed | cancelled
    error: str | None = None
    result: dict[str, Any] | None = None
    content_sha256: str | None = None  # 文件内容摘要（落库与查重展示）
    dup_kind: str | None = None  # history | batch | None（查重复核展示）
    dup_ref: str | None = None  # 重复对象描述
    dup_history: dict[str, Any] | None = None  # 历史影像摘要（kind=history）


class BatchManager:
    """批量任务队列管理器（Registry 装配，单例）。"""

    def __init__(
        self,
        pipeline_factory,
        *,
        workers: int,
        per_image_estimate_sec: float,
        batch_dir: str | Path,
        max_retained_batches: int = 50,
        max_retained_snapshot_files: int = 200,
        on_finished=None,
    ) -> None:
        self._pipeline_factory = pipeline_factory
        self._workers = max(1, workers)
        self._per_image_estimate_sec = max(0.1, per_image_estimate_sec)
        self._batch_dir = Path(batch_dir)
        self._batch_dir.mkdir(parents=True, exist_ok=True)
        self._max_retained = max(1, int(max_retained_batches))
        self._max_snapshot_files = max(1, int(max_retained_snapshot_files))
        # 批次收尾钩子（可选）：签名 (batch_id, done_results) -> dict | None。
        # Registry 借此实现底片印字占比裁决（缺印字复核的批量豁免），BatchManager
        # 本身不感知仓储/配置——依赖倒置，与 pipeline_factory 同一装配模式。
        self._on_finished = on_finished
        self._pool = ThreadPoolExecutor(max_workers=self._workers, thread_name_prefix="batch")
        self._closed = False
        self._lock = threading.Lock()
        self._batches: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def submit(self, items: list[BatchItem], *, hold: bool = False) -> str:
        """提交一批任务，返回 batch_id。

        hold=True（批量查重命中）：批次置 awaiting_review 暂缓态，任务全部
        pending 不入队，等人工逐项复核（resolve_dups）后再继续执行。
        """
        self._ensure_pool()
        if not items:
            raise ValueError("batch 至少需要一张影像")
        batch_id = uuid.uuid4().hex
        # 底片归属批次号（images.batch_no）：印字复核的批量豁免按批次结算，
        # 须在任务快照（复制 options）前注入，retry 重跑时同样携带。
        for item in items:
            item.options.setdefault("batch_no", batch_id)
        tasks = [
            BatchTaskState(
                task_id=uuid.uuid4().hex,
                image_name=item.image_name or item.image_path.name,
                image_path=str(item.image_path),
                options=dict(item.options),
                content_sha256=item.content_sha256,
                dup_kind=item.dup_kind,
                dup_ref=item.dup_ref,
                dup_history=item.dup_history,
            )
            for item in items
        ]
        duplicates = [
            {
                "task_id": t.task_id,
                "image_name": t.image_name,
                "content_sha256": t.content_sha256,
                "kind": t.dup_kind,
                "duplicate_of": t.dup_ref,
                "history": t.dup_history,
            }
            for t in tasks
            if t.dup_kind
        ]
        batch = {
            "batch_id": batch_id,
            # running=立即执行；awaiting_review=查重命中，等人工复核后继续
            "status": "awaiting_review" if hold else "running",
            "total": len(items),
            "done": 0,
            "failed": 0,
            "cancelled": 0,
            "cancelled_flag": False,
            "duplicates": duplicates,
            "estimated_sec": round(len(items) * self._per_image_estimate_sec, 1),
            "created_at": fmt_naive_utc(),
            "finished_at": None,
            "tasks": [t.__dict__ for t in tasks],
            "cleanup_dirs": sorted({str(i.cleanup_dir) for i in items if i.cleanup_dir}),
        }
        with self._lock:
            self._batches[batch_id] = batch
        self._maybe_evict_excess()  # S-*：新增批次后收敛内存保留上限
        self._persist(batch)
        if not hold:
            for item, task in zip(items, tasks):
                self._pool.submit(self._run_one, batch_id, task.task_id, item)
        return batch_id

    def resolve_dups(self, batch_id: str, decisions: dict[str, str]) -> dict[str, int]:
        """人工查重复核：decisions[task_id] ∈ {"keep","skip"}（缺省 skip，宁可不跑不重跑）。

        - skip：任务标 cancelled（error 注明人工跳过），不出检测报告；
        - keep：照常入队执行（重复影像由人工确认后仍需评片归档）；
        - 全部 skip 时批次直接 finished 并清理暂存目录。
        仅 awaiting_review 状态可复核；返回 {"skipped": n, "kept": m}。
        """
        self._ensure_pool()
        with self._lock:
            batch = self._batches.get(batch_id)
            if batch is None:
                raise KeyError(f"batch not found: {batch_id}")
            if batch.get("status") != "awaiting_review":
                raise ValueError(f"batch {batch_id} not awaiting dedup review")
            kept: list[str] = []
            skipped = 0
            for t in batch["tasks"]:
                # 非重复任务无条件继续；默认 skip 只作用于重复任务（宁可不跑不重跑）
                if not t.get("dup_kind"):
                    kept.append(t["task_id"])
                    continue
                action = decisions.get(t["task_id"], "skip")
                if action == "keep":
                    kept.append(t["task_id"])
                else:
                    t["status"] = "cancelled"
                    t["error"] = "人工查重复核：内容重复，确认跳过"
                    skipped += 1
            batch["done"] = sum(1 for t in batch["tasks"] if t["status"] == "done")
            batch["failed"] = sum(1 for t in batch["tasks"] if t["status"] == "failed")
            batch["cancelled"] = sum(1 for t in batch["tasks"] if t["status"] == "cancelled")
            if kept:
                batch["status"] = "running"
                batch["finished_at"] = None
            else:
                # 全部跳过：所有任务已终态，直接收敛完成并清理暂存目录
                self._maybe_finish(batch)
        self._persist(batch)
        if not kept:
            return {"skipped": skipped, "kept": 0}
        items: list[tuple[str, str, str, dict[str, Any]]] = []
        with self._lock:
            for t in batch["tasks"]:
                if t["task_id"] in kept:
                    items.append((t["task_id"], t["image_path"], t["image_name"], t["options"]))
        for task_id, image_path, image_name, options in items:
            self._pool.submit(
                self._run_one,
                batch_id,
                task_id,
                BatchItem(image_path=Path(image_path), options=options, image_name=image_name),
            )
        return {"skipped": skipped, "kept": len(items)}

    def status(self, batch_id: str) -> dict[str, Any] | None:
        """批次进度/结果快照（含每任务明细）。"""
        with self._lock:
            batch = self._batches.get(batch_id)
            if batch is None:
                return None
            return json.loads(json.dumps(batch))  # 深拷贝，防外部篡改

    def cancel(self, batch_id: str) -> bool:
        """标记批次取消：未启动任务不再执行。返回是否命中该批次。

        awaiting_review（查重暂缓）批次：任务尚未入队，pending 直接标
        cancelled，批次即时 finished 并清理暂存目录（等不到 worker 收尾）。
        """
        with self._lock:
            batch = self._batches.get(batch_id)
            if batch is None:
                return False
            if batch["status"] == "finished":
                return True  # 已完成批次无需取消
            if batch.get("status") == "awaiting_review":
                for t in batch["tasks"]:
                    if t["status"] == "pending":
                        t["status"] = "cancelled"
                        t["error"] = "batch cancelled"
                batch["done"] = sum(1 for t in batch["tasks"] if t["status"] == "done")
                batch["failed"] = sum(1 for t in batch["tasks"] if t["status"] == "failed")
                batch["cancelled"] = sum(1 for t in batch["tasks"] if t["status"] == "cancelled")
            batch["cancelled_flag"] = True
            self._maybe_finish(batch)  # 取消后若所有任务已终态则标记 finished（纯状态变更）
        self._persist(batch)
        return True

    def retry(self, batch_id: str) -> int:
        """断点续跑：将本批 failed/cancelled 任务重新入队，返回重跑任务数。

        原图依赖「有 failed/cancelled 即保留暂存目录」的清理策略（_cleanup_staging），
        故重跑时原图仍在；重跑完成后 _maybe_finish 再触发目录清理。
        """
        self._ensure_pool()
        with self._lock:
            batch = self._batches.get(batch_id)
            if batch is None:
                raise KeyError(f"batch not found: {batch_id}")
            if batch["cancelled_flag"]:
                batch["cancelled_flag"] = False  # 用户显式重试：解除取消标记
            pending: list[tuple[str, str, str, dict[str, Any]]] = []
            for t in batch["tasks"]:
                if t["status"] in ("failed", "cancelled"):
                    t["status"] = "pending"
                    t["error"] = None
                    t["result"] = None
                    pending.append((t["task_id"], t["image_path"], t["image_name"], t["options"]))
            if not pending:
                return 0
            batch["status"] = "running"
            batch["finished_at"] = None
        self._persist(batch)
        for task_id, image_path, image_name, options in pending:
            self._pool.submit(
                self._run_one,
                batch_id,
                task_id,
                BatchItem(image_path=Path(image_path), options=options, image_name=image_name),
            )
        return len(pending)

    def list(self) -> list[dict[str, Any]]:
        """批次摘要列表（最近提交在前，供历史/断点续跑入口）。

        排序用 dict 插入序（提交/恢复序）倒序：created_at 为秒级精度，
        同秒提交时时间序不可区分，插入序是确定性的稳定序。
        """
        with self._lock:
            batches = list(self._batches.values())
        batches.reverse()  # 后提交/后恢复在前
        out: list[dict[str, Any]] = []
        for b in batches:
            total = max(1, b["total"])
            progress = min(1.0, (b["done"] + b["failed"] + b["cancelled"]) / total)
            out.append(
                {
                    "batch_id": b["batch_id"],
                    "status": b["status"],
                    "total": b["total"],
                    "done": b["done"],
                    "failed": b["failed"],
                    "cancelled": b["cancelled"],
                    "progress": round(progress, 3),
                    "estimated_sec": b.get("estimated_sec", 0.0),
                    "created_at": b.get("created_at"),
                    "finished_at": b.get("finished_at"),
                }
            )
        return out

    def shutdown(self) -> None:
        """应用退出时释放线程池（等运行中任务结束）。"""
        self._pool.shutdown(wait=True)
        self._closed = True

    def _ensure_pool(self) -> None:
        """若线程池已随 shutdown 关闭（如测试生命周期中 lifespan 多次启停），
        惰性重建，使单例 BatchManager 在进程内可继续接收提交（生产真实退出后
        不再有新提交，不受影响）。"""
        if self._closed:
            self._pool = ThreadPoolExecutor(max_workers=self._workers, thread_name_prefix="batch")
            self._closed = False

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _run_one(self, batch_id: str, task_id: str, item: BatchItem) -> None:
        # 锁仅保护状态变更；落盘（序列化整批 JSON + 写盘，I/O 较重）移到锁外，
        # 避免每次状态变更阻塞全部 worker。
        with self._lock:
            batch = self._batches[batch_id]
            if batch["cancelled_flag"]:
                self._mark(batch, task_id, "cancelled", error="batch cancelled")
                self._maybe_finish(batch)
                self._persist(batch)
                return
            self._mark(batch, task_id, "running")
        pipeline = self._pipeline_factory()
        options = dict(item.options)
        options.setdefault("image_path", item.image_path)
        try:
            result = pipeline.run_inspection(**options)
        except Exception as exc:  # noqa: BLE001  # 失败隔离：单图异常不拖垮整批
            _LOG.exception("batch task failed batch=%s task=%s", batch_id, task_id)
            with self._lock:
                batch = self._batches[batch_id]
                self._mark(batch, task_id, "failed", error=str(exc))
                self._maybe_finish(batch)
            self._persist(batch)
            return
        with self._lock:
            batch = self._batches[batch_id]
            self._mark(batch, task_id, "done", result=result)
            self._maybe_finish(batch)
        self._persist(batch)

    def _mark(self, batch: dict, task_id: str, status: str, *, error=None, result=None) -> None:
        for t in batch["tasks"]:
            if t["task_id"] == task_id:
                t["status"] = status
                t["error"] = error
                t["result"] = result
                break
        # 重算计数
        batch["done"] = sum(1 for t in batch["tasks"] if t["status"] == "done")
        batch["failed"] = sum(1 for t in batch["tasks"] if t["status"] == "failed")
        batch["cancelled"] = sum(1 for t in batch["tasks"] if t["status"] == "cancelled")
        # 注意：落盘由调用方在锁外执行，此处仅变更内存状态。

    def _maybe_finish(self, batch: dict) -> None:
        if batch["status"] == "finished":
            return
        terminal = {"done", "failed", "cancelled"}
        if all(t["status"] in terminal for t in batch["tasks"]):
            batch["status"] = "finished"
            batch["finished_at"] = fmt_naive_utc()
            self._run_finish_hook(batch)
            self._cleanup_staging(batch)

    def _run_finish_hook(self, batch: dict) -> None:
        """批次终态时执行收尾钩子（印字占比裁决等），fail-soft。

        摘要写入批次快照（batch["stamp_summary"]）随 status 接口下发；钩子异常
        不掩盖批次本身的成功终态，仅日志留痕。
        """
        if self._on_finished is None:
            return
        done_results = [
            t["result"] for t in batch.get("tasks", []) if t["status"] == "done" and t.get("result")
        ]
        try:
            summary = self._on_finished(batch["batch_id"], done_results)
        except Exception as exc:  # noqa: BLE001 - 收尾钩子失败不改变批次终态
            _LOG.exception("batch finish hook failed batch=%s: %s", batch["batch_id"], exc)
            return
        if summary:
            batch["stamp_summary"] = summary

    def _cleanup_staging(self, batch: dict) -> None:
        """批次终态后清理上传暂存目录（P1-3：防 data/tmp 持续增长）。

        断点续跑守卫：批次内存在 failed/cancelled 任务时**保留**暂存目录
        （原图供 retry 重跑），仅当全部任务成功/已处理完才整体清理。
        只删本批次自身记录的 cleanup_dirs（提交时登记的批次专属目录），
        越界路径一律跳过；删除失败仅告警（快照已持久化，不阻断）。
        """
        retryable = {"failed", "cancelled"}
        if any(t.get("status") in retryable for t in batch.get("tasks", [])):
            _LOG.info(
                "batch %s has retryable tasks, keep staging for retry",
                batch["batch_id"],
            )
            return
        import shutil as _sh

        for raw in batch.get("cleanup_dirs", []):
            path = Path(raw)
            # 安全护栏：仅允许删除本批次登记的目录，且必须是目录
            if not path.is_dir():
                continue
            try:
                _sh.rmtree(str(path), ignore_errors=True)
                _LOG.info("batch staging cleaned: %s", path)
            except OSError as exc:
                _LOG.warning("batch staging cleanup failed %s: %s", path, exc)

    def _maybe_evict_excess(self) -> None:
        """内存保留上限（S-*）：驱逐最旧的"已终结且无失败/取消任务"批次。

        只驱逐**已完成且没有任何重试价值**的批次出内存（这类批次纯属历史，
        磁盘快照 data/batch/{id}.json 仍保留，不删用户数据）；running 批与含
        failed/cancelled（可 retry）批一律保留，避免破坏断点续跑。任何微量
        内存释放都有意义——该动作把 100h 长跑下 `self._batches` 的上限钉死。
        """
        if self._max_retained <= 0:
            return
        with self._lock:
            if len(self._batches) <= self._max_retained:
                return
            retryable = {"failed", "cancelled"}
            evictable = [
                bid
                for bid, b in self._batches.items()
                if b.get("status") == "finished"
                and not any(t.get("status") in retryable for t in b.get("tasks", []))
            ]
            # 按 created_at 升序，优先驱逐最旧。
            evictable.sort(key=lambda bid: self._batches[bid].get("created_at") or "")
            excess = len(self._batches) - self._max_retained
            for bid in evictable[:excess]:
                self._batches.pop(bid, None)

    def _maybe_prune_snapshots(self) -> None:
        """磁盘快照保留上限（S-21 防磁盘随批次无界增长）。

        与内存保留（max_retained_batches）同源策略但作用于 data/batch/*.json 文件：
        - 只删除"已完成且无失败/取消可重试任务"的批次快照文件（纯历史，删除安全）；
        - running / 含 failed/cancelled（可 retry）批次的快照恒保留，避免破坏断点续跑；
        - 先读目录文件倒序（最近修改在前），超限则优先剪最旧的安全文件。
        删除失败仅告警（不影响批次内存态），避免误删有 retry 价值的用户数据。
        """
        try:
            files = sorted(
                self._batch_dir.glob("*.json"),
                key=lambda p: p.stat().st_mtime,  # 最旧在前
            )
        except OSError:
            return
        if len(files) <= self._max_snapshot_files:
            return
        with self._lock:
            retryable = {"failed", "cancelled"}

            def _is_safe(path: str) -> bool:
                # 已不在内存（被 _maybe_evict_excess 驱逐）的批次 = 已完成且无 retry 价值，
                # 是纯历史文件，可安全剪枝；仍在内存的须逐条校验不能破坏断点续跑。
                batch = self._batches.get(path)
                if batch is None:
                    return True
                return batch.get("status") == "finished" and not any(
                    t.get("status") in retryable for t in batch.get("tasks", [])
                )

            safe = [p for p in files if _is_safe(p.stem)]
        excess = len(files) - self._max_snapshot_files
        for p in safe[:excess]:
            try:
                p.unlink()
            except OSError as exc:
                _LOG.warning("batch snapshot prune failed %s: %s", p.name, exc)

    def _persist(self, batch: dict) -> None:
        """状态快照落盘（断点续跑：重启后可查历史与未完成标记）。"""
        try:
            path = self._batch_dir / f"{batch['batch_id']}.json"
            path.write_text(json.dumps(batch, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:  # 持久化失败不阻断批量（内存态仍完整）
            _LOG.warning("batch snapshot persist failed: %s", batch["batch_id"])
        finally:
            self._maybe_prune_snapshots()  # S-21：每次落盘后收敛磁盘快照，防 100h 长跑无界增长

    def _load_existing(self) -> None:
        """启动时恢复历史批次快照（断点续跑的可查部分）。

        ：进程崩溃/重启时处于 running/pending 的任务实际已中断，若原样恢复会
                永久显示「运行中」且永不重跑。这里把它们标记为 failed（interrupted），
                用户可经 retry 重新入队（retry 兼容 failed/cancelled）。
        """
        for path in self._batch_dir.glob("*.json"):
            try:
                batch = json.loads(path.read_text(encoding="utf-8"))
                self._batches[batch["batch_id"]] = batch
            except (OSError, ValueError, KeyError):
                _LOG.warning("skip corrupted batch snapshot: %s", path.name)
        for batch in self._batches.values():
            # 查重暂缓批（awaiting_review）：任务从未启动，重启后仍应等待人工
            # 复核（暂存目录未清理、原图可用），不得把 pending 误标为 failed。
            if batch.get("status") == "awaiting_review":
                continue
            changed = False
            for t in batch.get("tasks", []):
                if t.get("status") in ("running", "pending"):
                    t["status"] = "failed"
                    t["error"] = "interrupted by restart; retry available"
                    changed = True
            if changed:
                batch["done"] = sum(1 for t in batch["tasks"] if t["status"] == "done")
                batch["failed"] = sum(1 for t in batch["tasks"] if t["status"] == "failed")
                batch["cancelled"] = sum(1 for t in batch["tasks"] if t["status"] == "cancelled")
                if batch["status"] != "finished":
                    batch["status"] = "finished"
                    batch["finished_at"] = batch.get("finished_at") or time.strftime(
                        "%Y-%m-%dT%H:%M:%S"
                    )
                self._persist(batch)
        # S-*：恢复历史快照后同样收敛内存，避免积压的历史批次一次性占满内存。
        self._maybe_evict_excess()
