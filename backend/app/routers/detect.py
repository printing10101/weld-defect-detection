"""缺陷检测 + 量化。

multipart 上传 → 加载 → 增强 → 训练模型检测器(YoloDetector) → 掩膜精修
(MaskQuantifier，) → 量化(掩膜级 L/W/面积/周长) → 标注图。
检测在增强图上进行，掩膜精修在增强图 ROI 取轮廓；标注图叠加在原始灰阶上。
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Annotated, Literal

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from backend.app.dependencies import Registry, get_registry
from backend.app.routers._common import staged_upload
from backend.domain.detect.thresholds import resolve_class_conf
from backend.domain.film_region import (
    FilmRegionCfg,
    detect_film_region_trusted,
    film_background_fill,
)
from backend.domain.quantify import MaskRefineCfg as DomainMaskRefineCfg
from backend.domain.quantify import (
    clock_position,
    get_quantifier,
    nearest_neighbor_gaps,
    refine_detections,
    weld_axis,
)
from backend.domain.spacing import resolve_spacing  # 单一真源（§T8/§6）
from backend.domain.stamp import StampCfg, filter_stamp_zone, read_stamp_aligned
from backend.infra.image_loader import load_image

router = APIRouter(tags=["detect"])


class DefectOut(BaseModel):
    id: str
    class_id: int
    class_name: str
    shape: str  # round | linear（有向长宽比判定，§5.4）
    bbox: list[float]  # x,y,w,h (px)，精修后为最小外接矩形轴对齐框
    confidence: float
    uncertainty: float
    calibrated: bool  # 像素标定是否可信（未标定禁输出伪物理尺寸，§T8/§6）
    L_mm: float | None = None  # None = 未标定，无物理量纲；条形缺陷为中心线弧长
    W_mm: float | None = None
    area_mm2: float | None = None
    perimeter_mm: float | None = None
    aspect_ratio: float  # 无量纲，与标定无关，恒有效
    position: list[float] | None = None  # x_mm, y_mm；None = 未标定
    mask_ref: str | None = None  # 掩膜资源 URI（当前不落盘，留 SAM 插件接口）
    # —— 位置/间距语义（任务2 结构化输出扩展）——
    centerline_mm: float | None = None  # 条形缺陷中心线弧长；圆形/未标定为 None
    clock_position: str | None = None  # "H:MM" 半小时精度；请求提供焊缝圆心时输出
    axial_position_mm: float | None = None  # 沿焊缝长轴线性位置；未标定为 None
    nearest_defect_id: str | None = None  # 最近邻缺陷 id（单缺陷为 None）
    nearest_gap_mm: float | None = None  # 最近邻边缘间距；未标定为 None


class DetectResponse(BaseModel):
    defects: list[DefectOut]
    annotated_image: str  # base64 PNG（画框标注）
    mode: str = "balanced"  # 本次推理使用的检测工作模式（balanced/recall_first/precision_first）


@router.post("/detect", response_model=DetectResponse)
async def detect(
    image: Annotated[UploadFile, File()],
    reg: Annotated[Registry, Depends(get_registry)],
    pixel_spacing_mm: Annotated[float | None, Form()] = None,
    conf: Annotated[float | None, Form()] = None,
    mode: Annotated[Literal["balanced", "recall_first", "precision_first"] | None, Form()] = None,
    weld_cx: Annotated[float | None, Form()] = None,
    weld_cy: Annotated[float | None, Form()] = None,
) -> DetectResponse:
    # 置信度显式校验：conf=0.0 是合法值，`or` 默认值会把它静默改成 0.3。
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
    # 焊缝圆心（钟点位参考）：两坐标必须成对提供，半给不出钟点位。
    if (weld_cx is None) != (weld_cy is None):
        raise HTTPException(
            status_code=422,
            detail={"code": "INVALID_WELD_CENTER", "message": "weld_cx 与 weld_cy 需成对提供"},
        )

    dc = reg.config.detect
    conf_v = conf if conf is not None else dc.infer_conf
    # 工作模式（高检出/高准确双档）：显式 conf 时以调用方为准（不再二次缩放，
    # 避免双重阈值语义）；未显式给 conf 时按模式缩放标定阈值。缩放结果以参数
    # 下传——不写回 config（共享单例，变异会泄漏到后续请求与评片链路）。
    detect_mode = mode or dc.mode
    class_conf_v = dict(dc.class_conf)
    if conf is None:
        conf_v, class_conf_v = resolve_class_conf(
            dc.infer_conf,
            class_conf_v,
            detect_mode,
            recall_scale=dc.recall_conf_scale,
            precision_scale=dc.precision_conf_scale,
        )

    async with staged_upload(image, reg.config) as tmp_path:
        # 解码/推理/编码均为 CPU 密集同步调用，放线程池避免阻塞事件循环。
        return await run_in_threadpool(
            _detect_sync,
            reg,
            tmp_path,
            pixel_spacing_mm,
            conf_v,
            class_conf_v,
            dc.infer_iou,
            detect_mode,
            weld_cx,
            weld_cy,
        )


def _detect_sync(
    reg: Registry,
    tmp_path: Path,
    pixel_spacing_mm: float | None,
    conf_v: float,
    class_conf_v: dict[int, float],
    iou_v: float,
    detect_mode: str,
    weld_cx: float | None = None,
    weld_cy: float | None = None,
) -> DetectResponse:
    gray, meta = load_image(tmp_path)
    dc = reg.config.detect
    pp_cfg = reg.config.preprocess
    # 与全链路一致：先分割胶片区并把区外背景填充为胶片中位灰阶（灯箱亮背景/
    # 翻拍边框会被误检为缺陷——预检接口此前缺这步，误检多于报告链路），
    # 再在增强图上检测；掩膜精修也在增强图 ROI 取轮廓。
    fr = reg.config.film_region
    film = (
        detect_film_region_trusted(
            gray,
            FilmRegionCfg(
                min_area_frac=fr.min_area_frac,
                max_photo_area_frac=fr.max_photo_area_frac,
                surround_bright_gray=fr.surround_bright_gray,
                surround_min_frac=fr.surround_min_frac,
            ),
        )
        if fr.enabled
        else None
    )
    enhanced = film_background_fill(gray, film)
    if pp_cfg.enabled:
        pp = reg.preprocessor
        enhanced = pp.enhance(pp.denoise(enhanced), pp_cfg.gamma)
    spacing, spacing_known = resolve_spacing(pixel_spacing_mm, meta.pixel_spacing_mm)
    detections = reg.detector.infer(enhanced, conf=conf_v, iou=iou_v, class_conf=class_conf_v)
    # 印字区误检过滤（与评片链路同口径）：印字 OCR 命中时屏蔽其外扩区域内的
    # 检出，防止预览界面把编号/日期铅字当缺陷展示给评片员。框经
    # read_stamp_aligned 映射回整图坐标，与检测框同系。
    if dc.mask_stamp_zone:
        stamp = read_stamp_aligned(
            gray,
            StampCfg(
                enabled=reg.config.stamp.enabled,
                min_conf=reg.config.stamp.min_conf,
                max_side=reg.config.stamp.max_side,
            ),
            film=film,
        )
        zones = stamp.text_boxes or stamp.boxes
        if zones:
            detections, _ = filter_stamp_zone(detections, zones)
    mrc = DomainMaskRefineCfg(**reg.config.mask_refine.model_dump())
    refined = refine_detections(enhanced, detections, mrc)
    # 经量化器注册表装配（去除 app 层 new 实现；种类由 detect.quantifier_kind 配置驱动）。
    quantifier = get_quantifier(dc.quantifier_kind)

    out: list[DefectOut] = []
    # 位置语义（G14/G17）：钟点位需请求提供焊缝圆心；轴向位置由胶片区长边
    # 方向近似（RT 底片焊缝沿长边布置）；间距为最近邻边缘间隙，未标定时
    # 物理量置 None，nearest_defect_id 无量纲恒可输出。
    gaps = nearest_neighbor_gaps(refined)
    axis = weld_axis((film.x, film.y, film.w, film.h)) if film is not None else "h"
    for d in refined:
        g = quantifier.quantify(d, spacing, image=enhanced, cfg=mrc)
        # 未标定（spacing_known=False）：物理字段置 None，不输出伪物理量；
        # aspect_ratio 为无量纲形状量，恒有效。与 /report grader 熔断保持单一语义。
        cx_px = d.bbox.x + d.bbox.w / 2
        cy_px = d.bbox.y + d.bbox.h / 2
        near = gaps.get(d.id)
        out.append(
            DefectOut(
                id=d.id,
                class_id=d.class_id.value,
                class_name=d.class_id.name,
                shape=d.shape.value if d.shape is not None else "unknown",
                bbox=[d.bbox.x, d.bbox.y, d.bbox.w, d.bbox.h],
                confidence=d.score,
                uncertainty=d.uncertainty,
                calibrated=spacing_known,
                L_mm=g.length_mm if spacing_known else None,
                W_mm=g.width_mm if spacing_known else None,
                area_mm2=g.area_mm2 if spacing_known else None,
                perimeter_mm=g.perimeter_mm if spacing_known else None,
                aspect_ratio=g.aspect_ratio,
                position=[g.position_x_mm, g.position_y_mm] if spacing_known else None,
                mask_ref=d.mask_ref,
                centerline_mm=g.centerline_mm if spacing_known else None,
                clock_position=(
                    clock_position(weld_cx, weld_cy, cx_px, cy_px)
                    if weld_cx is not None and weld_cy is not None
                    else None
                ),
                axial_position_mm=(
                    round((cx_px if axis == "h" else cy_px) * spacing, 3)
                    if spacing_known
                    else None
                ),
                nearest_defect_id=near[0] if near else None,
                nearest_gap_mm=round(near[1] * spacing, 3) if near and spacing_known else None,
            )
        )
    # 标注图叠加在原始灰阶上（真实观感），使用精修后的框。
    return DetectResponse(
        defects=out, annotated_image=_to_b64(_annotate(gray, refined)), mode=detect_mode
    )


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
