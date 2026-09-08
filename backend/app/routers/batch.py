"""批量处理与任务队列。

设计文档能力：多底片导入、任务队列、多 worker 并行推理、进度可视化、失败隔离、取消。
复用 InspectionPipeline 单图全链路；提交立即返回 batch_id，异步执行。

批量查重：提交时对每个文件流式计算 SHA256，先批内比对、再与已入库影像
（images.content_hash 索引）比对；命中项不直接丢弃，而是整批转入
awaiting_review 暂缓态，由人工逐项复核（跳过/仍检测）后再继续执行。
"""

from __future__ import annotations

import uuid
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from backend.app.batch_queue import BatchItem
from backend.app.dependencies import Registry, get_operator_name, get_registry
from backend.infra.config import resolve_config_path

router = APIRouter(tags=["batch"])


class BatchDuplicateOut(BaseModel):
    """单个重复文件的复核信息（submit 响应与批次 status 共用）。"""

    task_id: str
    image_name: str
    content_sha256: str | None = None
    kind: str  # history=与历史已检影像重复 | batch=批内重复
    duplicate_of: str | None = None  # 批内原文件名或历史 image_id
    history: dict | None = None  # kind=history 时的历史影像摘要


class BatchSubmitOut(BaseModel):
    batch_id: str
    total: int
    estimated_sec: float
    status: str = "running"  # running=已开始执行 | awaiting_review=查重命中待人工复核
    duplicates: list[BatchDuplicateOut] = []


class BatchTaskOut(BaseModel):
    task_id: str
    image_name: str
    status: str
    error: str | None = None
    image_id: str | None = None
    report_id: str | None = None
    joint_level: str | None = None
    need_review: bool | None = None
    content_sha256: str | None = None
    dup_kind: str | None = None
    dup_ref: str | None = None


class BatchStatusOut(BaseModel):
    batch_id: str
    status: str
    total: int
    done: int
    failed: int
    cancelled: int
    estimated_sec: float
    progress: float  # 0..1
    tasks: list[BatchTaskOut]
    duplicates: list[BatchDuplicateOut] = []


class CancelOut(BaseModel):
    ok: bool


class BatchRetryOut(BaseModel):
    ok: bool
    retried: int


class BatchSummaryOut(BaseModel):
    """GET /batches 列表项（BatchManager.list 摘要）。"""

    batch_id: str
    status: str
    total: int
    done: int
    failed: int
    cancelled: int
    progress: float
    estimated_sec: float
    created_at: str | None = None
    finished_at: str | None = None


def _suffix_ok(name: str | None, allowed: tuple[str, ...]) -> bool:
    suffix = Path(name or "upload.png").suffix.lower() or ".png"
    return suffix in allowed


@router.post("/batch", response_model=BatchSubmitOut)
def submit_batch(
    request: Request,
    reg: Annotated[Registry, Depends(get_registry)],
    operator: Annotated[str, Depends(get_operator_name)],
    images: Annotated[list[UploadFile], File()],
    pixel_spacing_mm: Annotated[float | None, Form()] = None,
    base_metal_thickness_mm: Annotated[float | None, Form()] = None,
    standard_id: Annotated[str | None, Form()] = None,
    workpiece_no: Annotated[str | None, Form()] = None,
    weld_no: Annotated[str | None, Form()] = None,
    signer: Annotated[str | None, Form()] = None,
    template: Annotated[str | None, Form()] = None,
    force: Annotated[bool, Form()] = False,
) -> BatchSubmitOut:
    """提交批量评片：多图上传 → 逐图入队异步执行。

    公共参数（标定/厚度/标准/工件信息/force）应用于批内所有影像。
    """
    # 请求级总量预拒：multipart 会先整体 spool 到磁盘才进入本函数，
    # 100 张 × max_bytes ≈ 20GB 的请求足以写满数据盘——按 Content-Length
    # 在深读前拒绝（并发若干个此类请求即成 DoS 面）。
    total_cap = reg.config.upload.max_bytes * reg.config.batch.max_per_batch
    declared = int(request.headers.get("content-length") or 0)
    if declared > total_cap:
        raise HTTPException(
            status_code=413,
            detail={
                "code": "PAYLOAD_TOO_LARGE",
                "message": f"批次总大小 {declared} 字节超过上限 {total_cap} 字节",
            },
        )
    if not images:
        raise HTTPException(
            status_code=422,
            detail={"code": "MISSING_INPUT", "message": "批量提交至少需要一张影像"},
        )
    if len(images) > reg.config.batch.max_per_batch:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "BATCH_TOO_LARGE",
                "message": f"单批最多 {reg.config.batch.max_per_batch} 张，收到 {len(images)} 张",
            },
        )
    allowed = tuple(reg.config.upload.allowed_suffixes)
    for f in images:
        if not _suffix_ok(f.filename, allowed):
            raise HTTPException(
                status_code=415,
                detail={
                    "code": "UNSUPPORTED_MEDIA_TYPE",
                    "message": f"不支持的影像格式: {Path(f.filename or 'upload').suffix}",
                },
            )

    # 批次专属暂存目录：任务完成后由 BatchManager 统一清理（P1-3）
    batch_dir = resolve_config_path(reg.config.paths.tmp_dir) / f"batch_{uuid.uuid4().hex}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    # actor = 提交批次的操作员（X-Operator-Name）；signer 缺省回填为操作员。
    options: dict = {
        "pixel_spacing_mm": pixel_spacing_mm,
        "base_metal_thickness_mm": base_metal_thickness_mm,
        "standard_id": standard_id,
        "workpiece_no": workpiece_no,
        "weld_no": weld_no,
        "signer": signer or operator,
        "actor": operator,
        "template": template or "standard",
        "force": force,
    }
    items: list[BatchItem] = []
    first_by_hash: dict[str, str] = {}  # 批内同内容首现表：hash → 原版文件名
    try:
        max_bytes = reg.config.upload.max_bytes
        dedup_on = reg.config.batch.dedup
        written: list[tuple[BatchItem, str]] = []  # (item, 文件 SHA256)
        for f in images:
            suffix = Path(f.filename or "upload.png").suffix.lower() or ".png"
            original_name = Path(f.filename or "upload.png").name
            target = batch_dir / f"{uuid.uuid4().hex}{suffix}"
            # 分块写盘 + 累计计数，超过 upload.max_bytes 立即 413（与 staged_upload 行为一致）；
            # 同一遍流式更新 SHA256，查重无需再整读一遍文件。
            size = 0
            digest = sha256()
            with target.open("wb") as fh:
                while True:
                    chunk = f.file.read(1 << 20)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > max_bytes:
                        # 超限：清理已写入的部分文件。清理失败（沙箱/回收站不可用）不得掩盖 413。
                        try:
                            target.unlink(missing_ok=True)
                        except Exception:  # noqa: BLE001, S110 - 部分环境删除受限，但不应影响 413 响应
                            pass
                        raise HTTPException(
                            status_code=413,
                            detail={
                                "code": "PAYLOAD_TOO_LARGE",
                                "message": f"文件超过上限 {max_bytes} 字节: {original_name}",
                            },
                        )
                    digest.update(chunk)
                    fh.write(chunk)
            content_hash = digest.hexdigest()
            item_options = dict(options)
            item_options["content_sha256"] = content_hash  # 落库 images.content_hash
            items.append(
                BatchItem(
                    image_path=target,
                    options=item_options,
                    image_name=original_name,
                    cleanup_dir=batch_dir,  # 批次完成后整体清理（P1-3）
                    content_sha256=content_hash,
                )
            )
            written.append((items[-1], content_hash))

        # 查重：先批内（同内容首次出现视为原版，其后为批内重复），再与历史
        # 已检影像（content_hash 索引，一次 IN 批量查询）比对。历史重复优先
        # 标注（信息更全：能看到首检记录）。命中任一重复 → 整批暂缓（hold），
        # 等人工复核决定跳过/仍检测后再执行。
        if dedup_on:
            history_index = reg.repository.find_images_by_hashes([h for _, h in written])
            for item, content_hash in written:
                if content_hash in history_index:
                    first = history_index[content_hash][0]  # created_at 升序 → 首检记录
                    item.dup_kind, item.dup_ref, item.dup_history = (
                        "history",
                        first["image_id"],
                        first,
                    )
                elif content_hash in first_by_hash:
                    item.dup_kind, item.dup_ref = "batch", first_by_hash[content_hash]
                else:
                    first_by_hash[content_hash] = item.image_name or ""
        hold = dedup_on and any(it.dup_kind for it in items)
        batch_id = reg.batch_manager.submit(items, hold=hold)
    except HTTPException:
        # 413/415/422 等由本函数有意抛出，必须原样上抛，不可被下方 except 吞掉转 500。
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"code": "BATCH_SUBMIT_FAILED", "message": str(exc)},
        ) from exc

    duplicates: list[BatchDuplicateOut] = []
    if hold:
        snap = reg.batch_manager.status(batch_id) or {}
        duplicates = [BatchDuplicateOut(**d) for d in snap.get("duplicates", [])]
    return BatchSubmitOut(
        batch_id=batch_id,
        total=len(items),
        estimated_sec=round(len(items) * reg.config.batch.per_image_estimate_sec, 1),
        status="awaiting_review" if hold else "running",
        duplicates=duplicates,
    )


class DedupDecisionIn(BaseModel):
    task_id: str
    action: Literal["skip", "keep"] = "skip"  # skip=跳过不检测 | keep=人工确认后仍检测


class DedupResolveIn(BaseModel):
    """查重复核提交：只列重复任务的决定；未列出的重复任务按 skip 处理。"""

    decisions: list[DedupDecisionIn] = []


class DedupResolveOut(BaseModel):
    ok: bool
    skipped: int
    kept: int


@router.post("/batch/{batch_id}/dedup/resolve", response_model=DedupResolveOut)
def resolve_batch_duplicates(
    batch_id: str,
    body: DedupResolveIn,
    reg: Annotated[Registry, Depends(get_registry)],
    operator: Annotated[str, Depends(get_operator_name)],
) -> DedupResolveOut:
    """人工查重复核：逐项决定重复文件跳过或仍检测，确认后批次继续执行。"""
    try:
        counts = reg.batch_manager.resolve_dups(
            batch_id, {d.task_id: d.action for d in body.decisions}
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail={"code": "NOT_FOUND", "message": f"batch not found: {batch_id}"},
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "NOT_AWAITING_REVIEW", "message": str(exc)},
        ) from exc
    # 复核动作入不可变审计链（谁对哪批做出了何种处置）
    reg.repository.append_audit(
        actor=operator,
        action="batch_dedup_resolve",
        object_type="batch",
        object_id=batch_id,
        before=None,
        after=counts,
    )
    return DedupResolveOut(ok=True, **counts)


@router.get("/batch/{batch_id}", response_model=BatchStatusOut)
def batch_status(
    batch_id: str,
    reg: Annotated[Registry, Depends(get_registry)],
) -> BatchStatusOut:
    """查询批次进度与逐任务结果。"""
    batch = reg.batch_manager.status(batch_id)
    if batch is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "NOT_FOUND", "message": f"batch not found: {batch_id}"},
        )
    tasks = []
    for t in batch["tasks"]:
        result = t.get("result") or {}
        tasks.append(
            BatchTaskOut(
                task_id=t["task_id"],
                image_name=t["image_name"],
                status=t["status"],
                error=t.get("error"),
                image_id=result.get("image_id"),
                report_id=result.get("report_id"),
                joint_level=result.get("joint_level"),
                need_review=bool(result.get("need_review")) if result else None,
                content_sha256=t.get("content_sha256"),
                dup_kind=t.get("dup_kind"),
                dup_ref=t.get("dup_ref"),
            )
        )
    total = max(1, batch["total"])
    progress = min(1.0, (batch["done"] + batch["failed"] + batch["cancelled"]) / total)
    return BatchStatusOut(
        batch_id=batch["batch_id"],
        status=batch["status"],
        total=batch["total"],
        done=batch["done"],
        failed=batch["failed"],
        cancelled=batch["cancelled"],
        estimated_sec=batch["estimated_sec"],
        progress=round(progress, 3),
        tasks=tasks,
        duplicates=[BatchDuplicateOut(**d) for d in batch.get("duplicates", [])],
    )


@router.post("/batch/{batch_id}/cancel", response_model=CancelOut)
def cancel_batch(
    batch_id: str,
    reg: Annotated[Registry, Depends(get_registry)],
) -> CancelOut:
    """取消批次：未启动任务不再执行（running 任务等待自然结束）。"""
    ok = reg.batch_manager.cancel(batch_id)
    if not ok:
        raise HTTPException(
            status_code=404,
            detail={"code": "NOT_FOUND", "message": f"batch not found: {batch_id}"},
        )
    return CancelOut(ok=True)


@router.get("/batches", response_model=list[BatchSummaryOut])
def list_batches(reg: Annotated[Registry, Depends(get_registry)]) -> list[BatchSummaryOut]:
    """批次摘要列表（按创建时间倒序），供历史查看与断点续跑入口。"""
    return [BatchSummaryOut(**row) for row in reg.batch_manager.list()]


@router.post("/batch/{batch_id}/retry", response_model=BatchRetryOut)
def retry_batch(
    batch_id: str,
    reg: Annotated[Registry, Depends(get_registry)],
) -> BatchRetryOut:
    """断点续跑：重跑本批 failed/cancelled 任务（原图暂存目录仍在）。"""
    try:
        retried = reg.batch_manager.retry(batch_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail={"code": "NOT_FOUND", "message": f"batch not found: {batch_id}"},
        ) from exc
    return BatchRetryOut(ok=True, retried=retried)
