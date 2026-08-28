"""主动学习闭环 API（§5.5/§5.6，M7 真实实现）。

闭环编排：检测结果 → 高价值采样（优先人工标注）→ 人工确认缺陷回流
训练池（YOLO 标注 + 数据版本 manifest）→ 供训练脚本合并重训。
只做编排，标注/采样算法在 domain/active_learning.py（§19.1 分层）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.app.dependencies import Registry, get_registry
from backend.domain.active_learning import (
    export_training_labels,
    save_pool_manifest,
    select_high_value,
    training_pool_manifest,
)
from backend.domain.dto import BBox, DefectClass, Detection
from backend.infra.fs import safe_resolve
from backend.infra.pool_store import FilePoolStore

router = APIRouter(tags=["active"])


class SampleDefectIn(BaseModel):
    id: str
    class_id: int = Field(ge=0, le=5)
    bbox: list[float] = Field(min_length=4, max_length=4)  # [x,y,w,h]
    confidence: float = 0.5
    uncertainty: float = 0.5


class SampleIn(BaseModel):
    """一次评片的检测结果（供主动学习采样）。"""

    image_id: str | None = None
    defects: list[SampleDefectIn] = []


class SampleOut(BaseModel):
    candidates: list[dict]  # 高价值候选（按价值降序）
    total: int


class ExportIn(BaseModel):
    """人工确认后的缺陷回流训练池。"""

    image_stem: str = Field(min_length=1, description="影像文件名（不含扩展名）")
    image_w: int = Field(gt=0)
    image_h: int = Field(gt=0)
    defects: list[SampleDefectIn] = []
    # 人工复核改判类别：{detection_id: 新 class_id}（如误检气孔改判裂纹）
    class_overrides: dict[str, int] = {}


class ExportOut(BaseModel):
    label_file: str
    sample_count: int
    fingerprint: str | None
    total_in_pool: int


class PoolOut(BaseModel):
    sample_count: int
    fingerprint: str | None
    files: list[str]
    exported_at: str | None


def _pool_dir(reg: Registry) -> Path:
    return Path(reg.config.paths.data_dir) / "active" / "training_pool"


def _pool_store(reg: Registry) -> FilePoolStore:
    """训练池存储（IO 经 infra FilePoolStore 注入，Task #9；domain 不触碰磁盘）。"""
    return FilePoolStore(_pool_dir(reg))


def _to_detection(d: SampleDefectIn) -> Detection:
    try:
        cls = DefectClass(d.class_id)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail={"code": "INVALID_CLASS_ID", "message": f"未知缺陷类别 {d.class_id}"},
        ) from None
    return Detection(
        id=d.id,
        bbox=BBox(d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]),
        class_id=cls,
        score=d.confidence,
        uncertainty=d.uncertainty,
    )


@router.post("/active/sample", response_model=SampleOut)
def active_sample(
    req: SampleIn,
    reg: Annotated[Registry, Depends(get_registry)],
) -> SampleOut:
    """主动学习采样：从一次评片检出中挑高价值样本（优先人工标注）。

    top_k 默认全返回按价值降序；调用方可按 value_score 阈值自行过滤。
    """
    detections = [_to_detection(d) for d in req.defects]
    candidates = select_high_value(detections, top_k=len(detections) or 1)
    return SampleOut(
        candidates=[c.__dict__ for c in candidates],
        total=len(candidates),
    )


@router.post("/active/export", response_model=ExportOut)
def active_export(
    req: ExportIn,
    reg: Annotated[Registry, Depends(get_registry)],
) -> ExportOut:
    """人工确认缺陷回流训练池（§5.5：人工复核结果写入训练池）。

    写 pool_dir/{image_stem}.txt（YOLO normalized），随后重算数据版本
    manifest（指纹随标注内容变化 → 训练侧可据此判断是否需重训）。
    """
    pool = _pool_dir(reg)
    # 防路径穿越：image_stem 只取文件名成分，且必须解析到 pool 之下（越界即 422）
    safe_stem = Path(req.image_stem).name
    try:
        safe_resolve(pool, f"{safe_stem}.txt")
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail={"code": "INVALID_IMAGE_STEM", "message": "image_stem 含非法路径字符"},
        )
    detections = [_to_detection(d) for d in req.defects]
    store = _pool_store(reg)
    label = export_training_labels(
        safe_stem,
        detections,
        float(req.image_w),
        float(req.image_h),
        store=store,
        class_overrides=req.class_overrides,
    )
    manifest = training_pool_manifest(store)
    save_pool_manifest(store, manifest)
    return ExportOut(
        label_file=str(label),
        sample_count=len(detections),
        fingerprint=manifest["fingerprint"],
        total_in_pool=manifest["sample_count"],
    )


@router.get("/active/pool", response_model=PoolOut)
def active_pool(
    reg: Annotated[Registry, Depends(get_registry)],
) -> PoolOut:
    """训练池状态：样本数 / 数据版本指纹 / 最近导出。"""
    manifest = training_pool_manifest(_pool_store(reg))
    return PoolOut(
        sample_count=manifest["sample_count"],
        fingerprint=manifest["fingerprint"],
        files=manifest["files"],
        exported_at=manifest["exported_at"],
    )
