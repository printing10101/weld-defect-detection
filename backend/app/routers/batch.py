"""批量处理与任务队列。

设计文档能力：多底片导入、任务队列、多 worker 并行推理、进度可视化、失败隔离、取消。
复用 InspectionPipeline 单图全链路；提交立即返回 batch_id，异步执行。
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from backend.app.batch_queue import BatchItem
from backend.app.dependencies import Registry, get_operator_name, get_registry

router = APIRouter(tags=["batch"])


class BatchSubmitOut(BaseModel):
    batch_id: str
    total: int
    estimated_sec: float


class BatchTaskOut(BaseModel):
    task_id: str
    image_name: str
    status: str
    error: str | None = None
    image_id: str | None = None
    report_id: str | None = None
    joint_level: str | None = None
    need_review: bool | None = None


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
    batch_dir = Path(reg.config.paths.tmp_dir) / f"batch_{uuid.uuid4().hex}"
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
    try:
        max_bytes = reg.config.upload.max_bytes
        for f in images:
            suffix = Path(f.filename or "upload.png").suffix.lower() or ".png"
            original_name = Path(f.filename or "upload.png").name
            target = batch_dir / f"{uuid.uuid4().hex}{suffix}"
            # 分块写盘 + 累计计数，超过 upload.max_bytes 立即 413（与 staged_upload 行为一致）。
            size = 0
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
                    fh.write(chunk)
            items.append(
                BatchItem(
                    image_path=target,
                    options=dict(options),
                    image_name=original_name,
                    cleanup_dir=batch_dir,  # 批次完成后整体清理（P1-3）
                )
            )
        batch_id = reg.batch_manager.submit(items)
    except HTTPException:
        # 413/415/422 等由本函数有意抛出，必须原样上抛，不可被下方 except 吞掉转 500。
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"code": "BATCH_SUBMIT_FAILED", "message": str(exc)},
        ) from exc

    return BatchSubmitOut(
        batch_id=batch_id,
        total=len(items),
        estimated_sec=round(len(items) * reg.config.batch.per_image_estimate_sec, 1),
    )


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
