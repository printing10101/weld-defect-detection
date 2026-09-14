"""缺陷图谱样本库（培训/比对/复核参考的典型缺陷样本）。

- POST   /atlas                      发布缺陷到图谱库（显式人工动作，留审计）
- GET    /atlas                      分页检索（类/级别/工件号过滤）
- GET    /atlas/{id}                 样本详情
- GET    /atlas/{id}/crop.png        缺陷局部图（密文副本自动解密）
- DELETE /atlas/{id}                 撤销样本（留审计）

发布语义：源缺陷必须存在（defects 表且未被软删——与复核视图同一可见性，
软删缺陷不再是有效事实）；同一缺陷只可发布一次（409）；局部图从落盘影像
（支持静态加密副本）裁出，随 security.encrypt 决定是否密文落盘。
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query, Response
from pydantic import BaseModel

from backend.app.dependencies import Registry, get_operator_name, get_registry
from backend.app.routers._common import api_error
from backend.infra.atlas_store import crop_defect_png, read_crop_bytes, write_crop_bytes

router = APIRouter(tags=["atlas"])

_LOG = logging.getLogger("scandetection.atlas_router")


class AtlasPublishRequest(BaseModel):
    defect_id: str
    note: str | None = None


class AtlasListResponse(BaseModel):
    items: list[dict]
    total: int


@router.post("/atlas")
def publish_sample(
    req: AtlasPublishRequest,
    reg: Annotated[Registry, Depends(get_registry)],
    operator: Annotated[str, Depends(get_operator_name)],
) -> dict:
    defect_id = req.defect_id.strip()
    if not defect_id:
        raise api_error(422, "INVALID_DEFECT_ID", "defect_id 不能为空")

    image = _image_of_defect(reg, defect_id)
    if image is None:
        raise api_error(404, "DEFECT_NOT_FOUND", f"源缺陷不存在: {defect_id}")
    defect = next(d for d in image["defects"] if d["id"] == defect_id)

    bbox = defect.get("bbox_px") or []
    image_path = str(image.get("path") or "")
    crop = crop_defect_png(image_path, bbox)
    if crop is None:
        raise api_error(409, "CROP_UNAVAILABLE", "源影像不可读或缺陷框无效，无法生成图谱局部图")

    atlas_id = uuid.uuid4().hex
    dest = reg.atlas_dir() / f"{atlas_id}.png"
    crop_path = write_crop_bytes(crop, dest, encrypt=bool(reg.config.security.encrypt))
    row = {
        "id": atlas_id,
        "defect_id": defect_id,
        "image_id": str(image.get("id") or ""),
        "class_id": int(defect["class_id"]),
        "joint_level": defect.get("joint_level"),
        "bbox_px": list(bbox),
        "length_mm": defect.get("length_mm"),
        "width_mm": defect.get("width_mm"),
        "confidence": float(defect.get("confidence") or 0.0),
        "source": defect.get("source"),
        "reviewed_by": defect.get("reviewed_by"),
        "workpiece_no": image.get("workpiece_no"),
        "weld_no": image.get("weld_no"),
        "standard_id": defect.get("standard_id") or image.get("standard_id"),
        "crop_path": crop_path,
        "note": (req.note or "").strip() or None,
        "created_by": operator,
    }
    try:
        reg.atlas_store().publish(row)
    except ValueError as exc:
        # 唯一约束冲突（重复发布）→ 409；其余入库错误 → 422
        if "already published" in str(exc):
            raise api_error(409, "ALREADY_PUBLISHED", str(exc)) from None
        raise api_error(422, "ATLAS_INVALID", str(exc)) from None

    reg.repository.append_audit(
        actor=operator,
        action="atlas_publish",
        object_type="atlas",
        object_id=atlas_id,
        before=None,
        after={
            "defect_id": defect_id,
            "image_id": image.get("id"),
            "class_id": row["class_id"],
            "joint_level": row["joint_level"],
        },
        note=row["note"],
    )
    return {"atlas_id": atlas_id, "defect_id": defect_id, "crop_path": crop_path}


@router.get("/atlas", response_model=AtlasListResponse)
def list_samples(
    reg: Annotated[Registry, Depends(get_registry)],
    class_id: Annotated[int | None, Query(alias="class", ge=0)] = None,
    level: Annotated[Literal["I", "II", "III", "IV"] | None, Query()] = None,
    workpiece: Annotated[str | None, Query(max_length=128)] = None,
    page: Annotated[int, Query(ge=1)] = 1,
    size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> AtlasListResponse:
    items, total = reg.atlas_store().list_samples(
        class_id=class_id, level=level, workpiece=workpiece, page=page, size=size
    )
    return AtlasListResponse(items=items, total=total)


@router.get("/atlas/{atlas_id}")
def get_sample(
    atlas_id: str,
    reg: Annotated[Registry, Depends(get_registry)],
) -> dict:
    sample = reg.atlas_store().get(atlas_id)
    if sample is None:
        raise api_error(404, "NOT_FOUND", f"图谱样本不存在: {atlas_id}")
    return sample


@router.get("/atlas/{atlas_id}/crop.png")
def get_crop(
    atlas_id: str,
    reg: Annotated[Registry, Depends(get_registry)],
) -> Response:
    crop_path = reg.atlas_store().get_crop_path(atlas_id)
    if crop_path is None:
        raise api_error(404, "NOT_FOUND", f"图谱样本不存在: {atlas_id}")
    data = read_crop_bytes(crop_path)
    if data is None:
        raise api_error(404, "CROP_MISSING", "图谱局部图缺失或不可读")
    return Response(content=data, media_type="image/png")


@router.delete("/atlas/{atlas_id}")
def remove_sample(
    atlas_id: str,
    reg: Annotated[Registry, Depends(get_registry)],
    operator: Annotated[str, Depends(get_operator_name)],
    reason: Annotated[str, Query(max_length=256)],
) -> dict:
    sample = reg.atlas_store().get(atlas_id)
    if sample is None:
        raise api_error(404, "NOT_FOUND", f"图谱样本不存在: {atlas_id}")
    # 路径经 get_crop_path 单独取（样本 dict 不含服务器路径，防信息泄漏）
    crop_path = reg.atlas_store().get_crop_path(atlas_id) or ""
    reg.atlas_store().remove(atlas_id)
    # 局部图随行删除：否则 data/atlas/ 会累积只有文件系统可见的孤儿样本。
    # 删除失败不阻断撤销（DB 行已删），降级告警留痕，由目录清理兜底。
    try:
        crop_file = Path(crop_path)
        if crop_file.is_file():
            crop_file.unlink()
    except OSError as exc:
        _LOG.warning("图谱局部图删除失败 atlas_id=%s path=%s: %s", atlas_id, crop_path, exc)
    reg.repository.append_audit(
        actor=operator,
        action="atlas_remove",
        object_type="atlas",
        object_id=atlas_id,
        before={"defect_id": sample["defect_id"]},
        after=None,
        note=reason,
    )
    return {"atlas_id": atlas_id, "removed": True}


def _image_of_defect(reg: Registry, defect_id: str) -> dict | None:
    """按缺陷 id 定位其所属影像详情（defects.id = "{image_id}:{d.id}"）。

    缺陷 id 前缀即影像 id，但该约定只在首轮检测行上成立——人工新增缺陷走
    同一拼法，仍可靠；为稳妥起见查到影像后再校验缺陷确在其中。
    """
    image_id = defect_id.split(":", 1)[0]
    image = reg.repository.get_image(image_id)
    if image is None:
        return None
    if not any(d["id"] == defect_id for d in image.get("defects") or []):
        return None
    return image
