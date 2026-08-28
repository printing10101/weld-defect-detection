"""标准判定（§6 / §T8 熔断）。

M5 真实实现：defects + 母材厚度 T → Nb47013Grader（多标准适配）。
未授权数值表（authorized=false）→ 熔断：422 GRADING_AMBIGUOUS，不输出级别。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.app.dependencies import Registry, get_registry
from backend.domain.dto import BBox, DefectClass, Detection, ImageMeta, Modality
from backend.domain.errors import AppError, GradingAmbiguousError

router = APIRouter(tags=["judge"])


class JudgeDefectIn(BaseModel):
    id: str
    class_id: int
    # 严格约束为 4 元素 [x,y,w,h]：长度不足会在构造 Detection 时 IndexError → 500，
    # 这里提前 422，避免把客户端格式错误暴露成服务器内部错误。
    bbox: list[float] = Field(min_length=4, max_length=4)
    confidence: float = 0.5
    uncertainty: float = 0.5


class JudgeRequest(BaseModel):
    base_metal_thickness_mm: float
    # 未标定禁定级（§6/§T8，与 /report 单一真源一致）：缺省 None，
    # 由 grader 对 None 熔断（422 GRADING_AMBIGUOUS），绝不静默 1.0 mm/px 伪物理。
    pixel_spacing_mm: float | None = None
    standard_id: str = "NB/T47013.2-2015"
    defects: list[JudgeDefectIn] = []


class JudgeResponse(BaseModel):
    joint_level: str | None
    per_defect_grade: list[str]
    basis: list[str]
    need_review: bool
    standard_id: str
    standard_version: str
    # 标准来源免责声明（工业过渡路径，T1）：authorized_copy=false 时为强声明，
    # 提示"数值转录自公开解读、非授权正本、不替代责任工程师法定评定"。
    disclaimer: str | None = None


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
        # §6.1 多标准适配：按 standard_id 路由判定器（默认 NB/T47013；骨架/未知标准熔断 422）
        grader = reg.grader_for(req.standard_id)
        result = grader.grade(defects, context)
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
        disclaimer=result.disclaimer,
    )


def _defect_class(class_id: int) -> DefectClass:
    try:
        return DefectClass(class_id)
    except ValueError:
        # 越界 class_id 不得静默回退为 POROSITY：在焊缝缺陷安全系统中，把"裂纹/未熔合"
        # 之类的未知/错误类别悄悄标成无害的"气孔"会掩盖重大缺陷。显式 422 让调用方
        # 修正输入（合法值 0–5），比输出一个错误类别更安全。
        raise HTTPException(
            status_code=422,
            detail={
                "code": "INVALID_CLASS_ID",
                "message": f"未知缺陷类别 id={class_id}，应为 0–5",
            },
        ) from None
