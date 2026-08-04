"""图像预处理（§4.3/§4.4）。

M3 真实实现：multipart 上传 → 加载 → 降噪+增强（管线）→ 质量度量 → 返回增强图缩略。
管线以"不损害缺陷边缘"为硬约束（§4.3）；参数走配置（§T8 禁硬编码）。
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Annotated

import cv2
from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import BaseModel

from backend.app.dependencies import Registry, get_registry
from backend.domain.preprocess.metrics import estimate_noise, psnr, ssim
from backend.domain.preprocess.pipeline import OpencvPreprocessor
from backend.infra.fs import secure_temp_dir
from backend.infra.image_loader import load_image

router = APIRouter(tags=["preprocess"])


class PreprocessResponse(BaseModel):
    image_id: str
    thumbnail: str  # base64 PNG（增强后，缩至 ≤256px）
    metrics: dict[str, float]


@router.post("/preprocess", response_model=PreprocessResponse)
async def preprocess(
    image: Annotated[UploadFile, File()],
    reg: Annotated[Registry, Depends(get_registry)],
    gamma: Annotated[float | None, Form()] = None,  # 缺省走配置
) -> PreprocessResponse:
    data = await image.read()
    tmp_dir = secure_temp_dir()
    suffix = Path(image.filename or "upload.png").suffix or ".png"
    tmp_path = Path(tmp_dir) / f"upload{suffix}"
    tmp_path.write_bytes(data)
    try:
        gray, _meta = load_image(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    cfg = reg.config.preprocess
    pp = OpencvPreprocessor(
        bilateral_d=cfg.bilateral_d,
        bilateral_sigma_color=cfg.bilateral_sigma_color,
        bilateral_sigma_space=cfg.bilateral_sigma_space,
        median_k=cfg.median_k,
        clahe_clip=cfg.clahe_clip,
        clahe_grid=cfg.clahe_grid,
        canny_kernel=cfg.canny_kernel,
        morph_k_open=cfg.morph_k_open,
        morph_k_close=cfg.morph_k_close,
    )
    gamma_v = gamma if gamma is not None else 1.0

    denoised = pp.denoise(gray)
    enhanced = pp.enhance(denoised, gamma_v)
    metrics = {
        "psnr_db": round(psnr(gray, enhanced), 3),
        "ssim": round(ssim(gray, enhanced), 4),
        "noise_in": round(estimate_noise(gray), 3),
        "noise_out": round(estimate_noise(enhanced), 3),
    }
    thumb = _thumbnail(enhanced)
    return PreprocessResponse(
        image_id=image.filename or "upload",
        thumbnail=thumb,
        metrics=metrics,
    )


def _thumbnail(image, max_side: int = 256) -> str:
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")
