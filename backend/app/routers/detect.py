"""缺陷检测 + 量化（§5，M4a 基线版）。

multipart 上传 → 加载 → BlobDetector（零训练基线）→ BBoxQuantifier → 标注图。
基线定位语义：类别占位、置信度保守（不可用于正式评级，见 §5/ADR-002）。
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Annotated

import cv2
from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import BaseModel

from backend.app.dependencies import Registry, get_registry
from backend.domain.quantify import BBoxQuantifier
from backend.infra.fs import secure_temp_dir
from backend.infra.image_loader import load_image

router = APIRouter(tags=["detect"])


class DefectOut(BaseModel):
    id: str
    class_id: int
    class_name: str
    bbox: list[float]  # x,y,w,h (px)
    confidence: float
    uncertainty: float
    L_mm: float
    W_mm: float
    area_mm2: float
    perimeter_mm: float
    aspect_ratio: float
    position: list[float]  # x_mm, y_mm


class DetectResponse(BaseModel):
    defects: list[DefectOut]
    annotated_image: str  # base64 PNG（画框标注）


@router.post("/detect", response_model=DetectResponse)
async def detect(
    image: Annotated[UploadFile, File()],
    reg: Annotated[Registry, Depends(get_registry)],
    pixel_spacing_mm: Annotated[float | None, Form()] = None,
    conf: Annotated[float | None, Form()] = None,
) -> DetectResponse:
    data = await image.read()
    tmp_dir = secure_temp_dir()
    suffix = Path(image.filename or "upload.png").suffix or ".png"
    tmp_path = Path(tmp_dir) / f"upload{suffix}"
    tmp_path.write_bytes(data)
    try:
        gray, meta = load_image(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    spacing = pixel_spacing_mm or meta.pixel_spacing_mm or 1.0
    detections = reg.detector.infer(gray, conf=conf or 0.3, iou=0.5)
    quantifier = BBoxQuantifier()

    out: list[DefectOut] = []
    for d in detections:
        g = quantifier.measure(d, spacing)
        out.append(
            DefectOut(
                id=d.id,
                class_id=d.class_id.value,
                class_name=d.class_id.name,
                bbox=[d.bbox.x, d.bbox.y, d.bbox.w, d.bbox.h],
                confidence=d.score,
                uncertainty=d.uncertainty,
                L_mm=g.length_mm,
                W_mm=g.width_mm,
                area_mm2=g.area_mm2,
                perimeter_mm=g.perimeter_mm,
                aspect_ratio=g.aspect_ratio,
                position=[g.position_x_mm, g.position_y_mm],
            )
        )
    annotated = _annotate(gray, detections)
    return DetectResponse(defects=out, annotated_image=_to_b64(annotated))


def _annotate(image, detections) -> bytes:
    """画缺陷框 + 标签，返回 PNG bytes。"""
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for d in detections:
        x, y, w, h = int(d.bbox.x), int(d.bbox.y), int(d.bbox.w), int(d.bbox.h)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 255), 2)
        label = f"{d.class_id.name} {d.score:.2f}"
        cv2.putText(canvas, label, (x, max(y - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    ok, buf = cv2.imencode(".png", canvas)
    return buf.tobytes() if ok else b""


def _to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")
