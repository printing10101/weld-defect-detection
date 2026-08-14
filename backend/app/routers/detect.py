"""缺陷检测 + 量化（§5，M4b 掩膜精修版）。

multipart 上传 → 加载 → 增强(§4.3) → 训练模型检测器(YoloDetector) → 掩膜精修
(MaskQuantifier，§5.3/§5.4) → 量化(掩膜级 L/W/面积/周长) → 标注图。
检测在增强图上进行，掩膜精修在增强图 ROI 取轮廓；标注图叠加在原始灰阶上。
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from backend.app.auth import get_current_user
from backend.app.dependencies import Registry, get_registry
from backend.app.routers._common import staged_upload
from backend.domain.quantify import MaskQuantifier, refine_detections
from backend.infra.image_loader import load_image

router = APIRouter(tags=["detect"], dependencies=[Depends(get_current_user)])


class DefectOut(BaseModel):
    id: str
    class_id: int
    class_name: str
    shape: str  # round | linear（有向长宽比判定，§5.4）
    bbox: list[float]  # x,y,w,h (px)，精修后为最小外接矩形轴对齐框
    confidence: float
    uncertainty: float
    L_mm: float
    W_mm: float
    area_mm2: float
    perimeter_mm: float
    aspect_ratio: float
    position: list[float]  # x_mm, y_mm
    mask_ref: str | None = None  # 掩膜资源 URI（当前不落盘，留 SAM 插件接口）


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
    # 置信度显式校验：原实现 `conf or 0.3` 会把合法的 conf=0.0 静默改成 0.3。
    if conf is not None and not 0.0 <= conf <= 1.0:
        raise HTTPException(
            status_code=422,
            detail={"code": "INVALID_CONF", "message": "conf 需在 [0,1] 区间"},
        )
    if pixel_spacing_mm is not None and pixel_spacing_mm <= 0:
        raise HTTPException(
            status_code=422,
            detail={"code": "INVALID_SPACING", "message": "pixel_spacing_mm 必须为正数"},
        )

    dc = reg.config.detect
    conf_v = conf if conf is not None else dc.infer_conf

    async with staged_upload(image, reg.config) as tmp_path:
        # 解码/推理/编码均为 CPU 密集同步调用，放线程池避免阻塞事件循环（§13.11）。
        return await run_in_threadpool(
            _detect_sync, reg, tmp_path, pixel_spacing_mm, conf_v, dc.infer_iou
        )


def _detect_sync(
    reg: Registry,
    tmp_path: Path,
    pixel_spacing_mm: float | None,
    conf_v: float,
    iou_v: float,
) -> DetectResponse:
    gray, meta = load_image(tmp_path)
    dc = reg.config.detect
    pp_cfg = reg.config.preprocess
    # 与全链路一致：检测在增强图上进行（§4.3）；掩膜精修也在增强图 ROI 取轮廓。
    enhanced = gray
    if pp_cfg.enabled:
        pp = reg.preprocessor
        enhanced = pp.enhance(pp.denoise(gray), pp_cfg.gamma)
    spacing = pixel_spacing_mm or meta.pixel_spacing_mm or 1.0
    detections = reg.detector.infer(enhanced, conf=conf_v, iou=iou_v, class_conf=dc.class_conf)
    mrc = reg.config.mask_refine
    refined = refine_detections(enhanced, detections, mrc)
    mq = MaskQuantifier()

    out: list[DefectOut] = []
    for d in refined:
        g = mq.quantify_from_image(enhanced, d, spacing, mrc)
        out.append(
            DefectOut(
                id=d.id,
                class_id=d.class_id.value,
                class_name=d.class_id.name,
                shape=d.shape.value if d.shape is not None else "unknown",
                bbox=[d.bbox.x, d.bbox.y, d.bbox.w, d.bbox.h],
                confidence=d.score,
                uncertainty=d.uncertainty,
                L_mm=g.length_mm,
                W_mm=g.width_mm,
                area_mm2=g.area_mm2,
                perimeter_mm=g.perimeter_mm,
                aspect_ratio=g.aspect_ratio,
                position=[g.position_x_mm, g.position_y_mm],
                mask_ref=d.mask_ref,
            )
        )
    # 标注图叠加在原始灰阶上（真实观感），使用精修后的框。
    return DetectResponse(defects=out, annotated_image=_to_b64(_annotate(gray, refined)))


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """标注前统一到 8bit：16bit 底片直接画 (0,0,255) 会淡到不可见。"""
    if image.dtype == np.uint8:
        return image
    arr = image.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return ((arr - lo) * (255.0 / (hi - lo))).astype(np.uint8)


def _annotate(image: np.ndarray, detections) -> bytes:
    """画缺陷框 + 标签，返回 PNG bytes。"""
    canvas = cv2.cvtColor(_to_uint8(image), cv2.COLOR_GRAY2BGR)
    h_img, w_img = canvas.shape[:2]
    for d in detections:
        x, y, w, h = int(d.bbox.x), int(d.bbox.y), int(d.bbox.w), int(d.bbox.h)
        # 裁剪到画布内，越界坐标会让 rectangle 静默不画或抛错
        x0, y0 = max(0, min(x, w_img - 1)), max(0, min(y, h_img - 1))
        x1, y1 = max(0, min(x + w, w_img - 1)), max(0, min(y + h, h_img - 1))
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 0, 255), 2)
        label = f"{d.class_id.name} {d.score:.2f}"
        cv2.putText(
            canvas, label, (x0, max(y0 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
        )
    ok, buf = cv2.imencode(".png", canvas)
    if not ok:
        raise HTTPException(
            status_code=500,
            detail={"code": "ENCODE_FAILED", "message": "标注图编码失败"},
        )
    return buf.tobytes()


def _to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")
