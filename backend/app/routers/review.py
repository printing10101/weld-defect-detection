"""人工复核 / 仲裁闭环。

POST /api/v1/review：提交一次复核（初评/复评/仲裁），后端计算与自动评级的
Cohen's κ 一致性；κ≥阈值则达成共识并落地最终级别（清空 need_review、重生成
PDF/A），否则升级仲裁。详见 domain/review.resolve_review 与 app/pipelines.apply_review。

缺陷增删改（DB50/T 1807-2025 ）：POST /api/v1/review/{image_id}/defects、
PATCH/DELETE /api/v1/review/defects/{defect_id}——全程审计留痕，变更后自动重评级
并重生成报告（操作员与理由必填，审计留痕）。
"""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.app.dependencies import Registry, get_operator_name, get_registry
from backend.app.pipelines import InspectionPipeline

router = APIRouter(tags=["review"])

Role = Literal["initial", "secondary", "arbitrator"]


class DefectGrade(BaseModel):
    defect_id: str
    joint_level: str = Field(pattern="^(I|II|III|IV)$")


class ReviewIn(BaseModel):
    image_id: str
    reviewer: str = Field(min_length=1, description="评片员姓名/工号")
    role: Role
    defect_grades: list[DefectGrade] = Field(
        default_factory=list, description="逐缺陷级别覆盖（可只提交分歧项）"
    )
    overall_level: str | None = Field(
        default=None,
        pattern="^(I|II|III|IV)$",
        description="复核综合级别（可选，优先于按缺陷推算）",
    )
    note: str | None = None


class ReviewOut(BaseModel):
    image_id: str
    reviewer: str
    role: str
    consensus: bool
    kappa: float
    needs_arbitration: bool
    joint_level: str | None
    reviewed_by: str | None
    stage: str
    need_review: bool
    review_count: int


@router.post("/review", response_model=ReviewOut)
def review(
    body: ReviewIn,
    reg: Annotated[Registry, Depends(get_registry)],
    operator: Annotated[str, Depends(get_operator_name)],
) -> ReviewOut:
    # 复核人默认取请求头操作员（闭合 / 操作者身份占位）；显式提供时以显式值为准。
    reviewer = body.reviewer or operator
    pipeline = InspectionPipeline(reg)
    try:
        return ReviewOut(
            **pipeline.apply_review(
                image_id=body.image_id,
                reviewer=reviewer,
                role=body.role,
                actor=operator,
                defect_grades=[g.model_dump() for g in body.defect_grades],
                overall_level=body.overall_level,
                note=body.note,
            )
        )
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail={"code": "NOT_FOUND", "message": f"image not found: {body.image_id}"},
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail={"code": "BAD_REVIEW", "message": str(e)})


# ---------------------------------------------------------------------------
# 缺陷增删改（DB50/T 1807-2025  复核功能：缺陷增删、类型、位置）
# ---------------------------------------------------------------------------


class DefectAddIn(BaseModel):
    """人工添加缺陷框（复核员确认的漏检补录 / 误检订正）。"""

    class_id: int = Field(ge=0, description="缺陷类别（DefectClass，含 6=内凹）")
    bbox_px: list[float] = Field(min_length=4, max_length=4, description="[x,y,w,h] 像素")
    reason: str = Field(min_length=1, description="添加理由（审计必填）")


class DefectEditIn(BaseModel):
    """人工修改缺陷类型和/或位置（字段可选，至少一项）。"""

    class_id: int | None = Field(default=None, ge=0)
    bbox_px: list[float] | None = Field(default=None, min_length=4, max_length=4)
    reason: str = Field(min_length=1, description="修改理由（审计必填）")


def _handle_review_errors(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": str(exc)})
    return HTTPException(status_code=422, detail={"code": "BAD_REVIEW", "message": str(exc)})


@router.post("/review/{image_id}/defects")
def add_defect(
    image_id: str,
    body: DefectAddIn,
    reg: Annotated[Registry, Depends(get_registry)],
    operator: Annotated[str, Depends(get_operator_name)],
) -> dict:
    pipeline = InspectionPipeline(reg)
    try:
        return pipeline.add_defect(
            image_id=image_id,
            class_id=body.class_id,
            bbox_px=body.bbox_px,
            operator=operator,
            reason=body.reason,
        )
    except (KeyError, ValueError) as exc:
        raise _handle_review_errors(exc) from exc


@router.patch("/review/defects/{defect_id}")
def edit_defect(
    defect_id: str,
    body: DefectEditIn,
    reg: Annotated[Registry, Depends(get_registry)],
    operator: Annotated[str, Depends(get_operator_name)],
) -> dict:
    if body.class_id is None and body.bbox_px is None:
        raise HTTPException(
            422, detail={"code": "BAD_REVIEW", "message": "class_id 与 bbox_px 至少提供一项"}
        )
    pipeline = InspectionPipeline(reg)
    try:
        return pipeline.edit_defect(
            defect_id=defect_id,
            operator=operator,
            reason=body.reason,
            class_id=body.class_id,
            bbox_px=body.bbox_px,
        )
    except (KeyError, ValueError) as exc:
        raise _handle_review_errors(exc) from exc


@router.delete("/review/defects/{defect_id}")
def delete_defect(
    defect_id: str,
    reason: str,
    reg: Annotated[Registry, Depends(get_registry)],
    operator: Annotated[str, Depends(get_operator_name)],
) -> dict:
    """软删除缺陷（不物理清除；审计记录 before/after 供追溯）。"""
    pipeline = InspectionPipeline(reg)
    try:
        return pipeline.delete_defect(defect_id=defect_id, operator=operator, reason=reason)
    except (KeyError, ValueError) as exc:
        raise _handle_review_errors(exc) from exc
