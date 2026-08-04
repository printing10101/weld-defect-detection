"""标准判定（§6 / §T8 熔断）。

M5 真实实现：defects + 母材厚度 T → Nb47013Grader（多标准适配）。
未授权数值表（authorized=false）→ 熔断：422 GRADING_AMBIGUOUS，不输出级别。
"""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.app.dependencies import Registry, get_registry
from backend.domain.dto import BBox, DefectClass, Detection, ImageMeta, Modality
from backend.domain.errors import AppError, GradingAmbiguousError

router = APIRouter(tags=["judge"])


class JudgeDefectIn(BaseModel):
    id: str
    class_id: int
    bbox: list[float]  # x,y,w,h (px)
    confidence: float = 0.5
    uncertainty: float = 0.5


class JudgeRequest(BaseModel):
    base_metal_thickness_mm: float
    pixel_spacing_mm: float = 1.0
    standard_id: str = "NB/T47013.2-2015"
    defects: list[JudgeDefectIn] = []


class JudgeResponse(BaseModel):
    joint_level: str | None
    per_defect_grade: list[str]
    basis: list[str]
    need_review: bool
    standard_id: str
    standard_version: str


@router.post("/judge", response_model=JudgeResponse)
def judge(
    req: JudgeRequest,
    reg: Annotated[Registry, Depends(get_registry)],
) -> JudgeResponse:
    defects = [
        Detection(
            id=d.id,
            bbox=BBox(d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]),
            class_id=_defect_class(d.class_id),
            score=d.confidence,
            uncertainty=d.uncertainty,
        )
        for d in req.defects
    ]
    context = ImageMeta(
        modality=Modality.GENERIC,
        pixel_spacing_mm=req.pixel_spacing_mm,
        base_metal_thickness_mm=req.base_metal_thickness_mm,
    )
    try:
        result = reg.grader.grade(defects, context)
    except GradingAmbiguousError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": exc.code,
                "message": str(exc),
                "detail": "标准数值未授权或判定信息不足，需人工复核",
            },
        ) from exc
    except AppError as exc:
        raise HTTPException(
            status_code=exc.http_status,
            detail={"code": exc.code, "message": str(exc), "detail": None},
        ) from exc
    return JudgeResponse(
        joint_level=result.joint_level.value,
        per_defect_grade=[g.value for g in result.per_defect_grade],
        basis=list(result.basis),
        need_review=result.need_review,
        standard_id=result.standard_id,
        standard_version=result.standard_version,
    )


def _defect_class(class_id: int) -> DefectClass:
    try:
        return DefectClass(class_id)
    except ValueError:
        return DefectClass.POROSITY  # 未知类别按气孔处理（基线占位）
