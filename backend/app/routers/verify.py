"""IQI + 黑度校验（§4.2）。

M2 真实实现：multipart 上传 → infra 加载 → 黑度估计 + 线型 IQI 识别 → evaluable。
注：M2 基线为单图轻量计算（同步执行）；批量/重型管线按 §13.11 进线程池（M6）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import BaseModel

from backend.app.dependencies import Registry, get_registry
from backend.domain.density import check_density, estimate_density
from backend.domain.iqi import IqiConfig, verify_wire_iqi
from backend.infra.fs import secure_temp_dir
from backend.infra.image_loader import load_image

router = APIRouter(tags=["verify"])


class IqiOut(BaseModel):
    iqi_type: str
    achieved: str | None
    required: str
    passed: bool


class VerifyResponse(BaseModel):
    iqi: IqiOut
    density: float
    density_ok: bool
    evaluable: bool


@router.post("/verify", response_model=VerifyResponse)
async def verify(
    image: Annotated[UploadFile, File()],
    reg: Annotated[Registry, Depends(get_registry)],
    iqi_roi: Annotated[str | None, Form()] = None,  # "x,y,w,h"（可选，缺省全图）
) -> VerifyResponse:
    data = await image.read()
    tmp_dir = secure_temp_dir()
    suffix = Path(image.filename or "upload.png").suffix or ".png"
    tmp_path = Path(tmp_dir) / f"upload{suffix}"
    tmp_path.write_bytes(data)
    try:
        gray, _meta = load_image(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    density = float(estimate_density(gray))
    density_ok = bool(
        check_density(density, reg.config.density.low, reg.config.density.high)
    )

    iqi_cfg = IqiConfig(
        wire_diameters_mm=tuple(reg.config.iqi.wire_diameters_mm),
        required_wire_no=reg.config.iqi.required_wire_no,
        min_contrast_ratio=reg.config.iqi.min_contrast_ratio,
    )
    iqi = verify_wire_iqi(gray, iqi_cfg, roi=_parse_roi(iqi_roi))

    return VerifyResponse(
        iqi=IqiOut(
            iqi_type=iqi.iqi_type,
            achieved=iqi.achieved,
            required=iqi.required,
            passed=iqi.passed,
        ),
        density=round(density, 3),
        density_ok=density_ok,
        evaluable=density_ok and iqi.passed,
    )


def _parse_roi(raw: str | None) -> tuple[int, int, int, int] | None:
    if not raw:
        return None
    parts = [int(v.strip()) for v in raw.split(",")]
    if len(parts) != 4:
        return None
    return parts[0], parts[1], parts[2], parts[3]
