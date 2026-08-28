"""历史检索与统计。

GET /api/v1/records?level=&class=&from=&to=&workpiece=&page=&size=
→ {items[], total, stats}；多条件过滤 + 分页 + 缺陷统计。
"""

from __future__ import annotations

from typing import Annotated, Literal

import cv2
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel

from backend.app.dependencies import Registry, get_registry
from backend.infra.diconde import parse_diconde_file
from backend.infra.reporting.pdf_reporter import _read_gray

router = APIRouter(tags=["records"])

_DATE_PATTERN = r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?)?$"


class RecordsResponse(BaseModel):
    items: list[dict]
    total: int
    stats: dict


@router.get("/records", response_model=RecordsResponse)
def records(
    reg: Annotated[Registry, Depends(get_registry)],
    # level 走 Literal：非法值在入口 422，而非到仓储层抛 ValueError → 500
    level: Annotated[Literal["I", "II", "III", "IV"] | None, Query()] = None,
    class_id: Annotated[int | None, Query(alias="class", ge=0)] = None,
    date_from: Annotated[str | None, Query(alias="from", pattern=_DATE_PATTERN)] = None,
    date_to: Annotated[str | None, Query(alias="to", pattern=_DATE_PATTERN)] = None,
    workpiece: Annotated[str | None, Query(max_length=128)] = None,
    page: Annotated[int, Query(ge=1)] = 1,
    size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> RecordsResponse:
    try:
        items, total = reg.repository.list_records(
            level=level,
            class_id=class_id,
            date_from=date_from,
            date_to=date_to,
            workpiece=workpiece,
            page=page,
            size=size,
        )
    except ValueError as exc:  # 日期越界等仓储层校验 → 422 而非 500
        raise HTTPException(
            status_code=422, detail={"code": "INVALID_QUERY", "message": str(exc)}
        ) from None
    return RecordsResponse(items=items, total=total, stats=reg.repository.stats())


_PREVIEW_MAX_SIDE = 1600  # 预览长边上限（查看器/报告页展示够用，控制传输体积）


@router.get("/images/{image_id}/preview.png")
def image_preview(
    image_id: str,
    reg: Annotated[Registry, Depends(get_registry)],
) -> Response:
    """库内影像的 PNG 预览（浏览器不解码 TIFF/DICOM，统一在此转换）。

    支持静态加密副本（SDC1 魔数自动解密）；长边超限时等比降采样。
    """
    image = reg.repository.get_image(image_id)
    if image is None:
        raise HTTPException(
            status_code=404, detail={"code": "NOT_FOUND", "message": f"image not found: {image_id}"}
        )
    gray = _read_gray(str(image["path"]))
    if gray is None:
        raise HTTPException(
            status_code=422, detail={"code": "BAD_IMAGE", "message": "影像不可读（文件缺失或损坏）"}
        )
    h, w = gray.shape[:2]
    scale = min(1.0, _PREVIEW_MAX_SIDE / max(h, w))
    if scale < 1.0:
        gray = cv2.resize(gray, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", gray)
    if not ok:
        raise HTTPException(
            status_code=500, detail={"code": "ENCODE_FAIL", "message": "预览编码失败"}
        )
    return Response(
        content=buf.tobytes(),
        media_type="image/png",
        headers={"Cache-Control": "private, max-age=3600"},
    )


@router.get("/images/{image_id}/diconde")
def image_diconde(
    image_id: str,
    reg: Annotated[Registry, Depends(get_registry)],
) -> dict:
    """DICONDE 元数据（ASTM E2339）：透照工艺/设备字段 + 隐私标签状态。

    非 DICOM 影像返回 422；患者隐私字段仅作存在性提示（脱敏由
    anonymize_images 工具负责）。
    """
    image = reg.repository.get_image(image_id)
    if image is None:
        raise HTTPException(
            status_code=404, detail={"code": "NOT_FOUND", "message": f"image not found: {image_id}"}
        )
    try:
        return parse_diconde_file(image["path"])
    except ValueError as e:
        raise HTTPException(status_code=422, detail={"code": "NOT_DICOM", "message": str(e)}) from e
