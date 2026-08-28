"""可解释性热力图（§12.3，M6 真实实现）。

对指定 defect_id 生成注意力热力图叠加原图（模型无关近似，见
domain/explain.py 的替代声明），供人工复核视图秒懂模型关注区。
默认仅人工复核视图调用，不进入主推理链路（主链路不带 explain 参数）。
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Annotated

import cv2
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from backend.app.dependencies import Registry, get_registry
from backend.app.routers._common import staged_upload
from backend.domain.explain import attention_heatmap
from backend.infra.image_loader import load_image

router = APIRouter(tags=["explain"])


class ExplainResponse(BaseModel):
    heatmap: str  # base64 PNG（热力图叠加原图）


@router.post("/explain", response_model=ExplainResponse)
async def explain(
    image: Annotated[UploadFile, File()],
    reg: Annotated[Registry, Depends(get_registry)],
    defect_id: Annotated[str | None, Form()] = None,
) -> ExplainResponse:
    """对指定缺陷生成注意力热力图叠加原图。

    defect_id 缺省时对**全部检出缺陷**分别叠加（取最高置信缺陷的叠加结果）。
    热力图基于 ROI 局部显著性（§12.3 模型无关近似，非真 Grad-CAM——见
    domain/explain.py 文档的替代声明）。
    """
    async with staged_upload(image, reg.config) as tmp_path:
        heatmap_b64 = await run_in_threadpool(_explain_sync, reg, tmp_path, defect_id)
    return ExplainResponse(heatmap=heatmap_b64)


def _explain_sync(reg: Registry, tmp_path: Path, defect_id: str | None) -> str:
    gray, _meta = load_image(tmp_path)
    dc = reg.config.detect
    pp_cfg = reg.config.preprocess
    enhanced = gray
    if pp_cfg.enabled:
        pp = reg.preprocessor
        enhanced = pp.enhance(pp.denoise(gray), pp_cfg.gamma)
    detections = reg.detector.infer(
        enhanced, conf=dc.infer_conf, iou=dc.infer_iou, class_conf=dc.class_conf
    )
    if not detections:
        raise HTTPException(
            status_code=404,
            detail={"code": "NO_DEFECT", "message": "未检出任何缺陷，无热力图可生成"},
        )
    target = (
        next((d for d in detections if d.id == defect_id), None)
        if defect_id
        else max(detections, key=lambda d: d.score)
    )
    if target is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "DEFECT_NOT_FOUND", "message": f"defect not found: {defect_id}"},
        )
    overlay = attention_heatmap(gray, target)
    ok, buf = cv2.imencode(".png", overlay)
    if not ok:
        raise HTTPException(
            status_code=500,
            detail={"code": "ENCODE_FAILED", "message": "热力图编码失败"},
        )
    return base64.b64encode(buf.tobytes()).decode("ascii")
