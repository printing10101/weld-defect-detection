"""图像预处理。

 真实实现：multipart 上传 → 加载 → 降噪+增强（管线）→ 质量度量 → 返回增强图缩略。
管线以"不损害缺陷边缘"为硬约束；参数走配置。
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
from backend.domain.preprocess.metrics import QualityCfg, assess_quality, estimate_noise, psnr, ssim
from backend.infra.image_loader import load_image

router = APIRouter(tags=["preprocess"])


class PreprocessResponse(BaseModel):
    image_id: str
    thumbnail: str  # base64 PNG（增强后，缩至 ≤256px）
    metrics: dict[str, float]
    quality: dict  # §4.4 质量门禁结论（score/passed/子分/brisque 特征）


@router.post("/preprocess", response_model=PreprocessResponse)
async def preprocess(
    image: Annotated[UploadFile, File()],
    reg: Annotated[Registry, Depends(get_registry)],
    gamma: Annotated[float | None, Form()] = None,  # 缺省走配置
) -> PreprocessResponse:
    # 伽马必须为正：gamma<=0 会让 pow 产生 inf/nan，后续 PSNR/SSIM 全污染。
    if gamma is not None and not 0.0 < gamma <= 10.0:
        raise HTTPException(
            status_code=422,
            detail={"code": "INVALID_GAMMA", "message": "gamma 需在 (0,10] 区间"},
        )
    async with staged_upload(image, reg.config) as tmp_path:
        name = image.filename or "upload"
        # 滤波/CLAHE/度量均为 CPU 密集同步调用，进线程池。
        return await run_in_threadpool(_preprocess_sync, reg, tmp_path, gamma, name)


def _preprocess_sync(
    reg: Registry, tmp_path: Path, gamma: float | None, name: str
) -> PreprocessResponse:
    gray, _meta = load_image(tmp_path)

    cfg = reg.config.preprocess
    pp = reg.preprocessor  # 复用 Registry 单例（与管线同实例）
    gamma_v = gamma if gamma is not None else cfg.gamma

    denoised = pp.denoise(gray)
    enhanced = pp.enhance(denoised, gamma_v)
    metrics = {
        "psnr_db": round(psnr(gray, enhanced), 3),
        "ssim": round(ssim(gray, enhanced), 4),
        "noise_in": round(estimate_noise(gray), 3),
        "noise_out": round(estimate_noise(enhanced), 3),
    }
    # 质量度量门禁（无参考）：BRISQUE 特征 + 复合 RQI 分。
    quality = assess_quality(gray, QualityCfg(**reg.config.quality.model_dump()))
    thumb = _thumbnail(enhanced)
    return PreprocessResponse(
        image_id=name,
        thumbnail=thumb,
        metrics=metrics,
        quality={
            "score": round(quality.score, 2),
            "passed": quality.passed,
            "metrics": quality.metrics,
        },
    )


def _thumbnail(image, max_side: int = 256) -> str:
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:  # 空影像：避免 max(h,w)=0 处除零
        return ""
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        # resize 目标边至少 1px，否则 cv2 抛 error
        image = cv2.resize(
            image,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")
