"""人工复核 / 仲裁闭环（§12.2 / M7）。

POST /api/v1/review：提交一次复核（初评/复评/仲裁），后端计算与自动评级的
Cohen's κ 一致性；κ≥阈值则达成共识并落地最终级别（清空 need_review、重生成
PDF/A），否则升级仲裁。详见 domain/review.resolve_review 与 app/pipelines.apply_review。
"""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.app.auth import (
    CurrentUser,
    get_current_user,
)
from backend.app.dependencies import Registry, get_registry
from backend.app.pipelines import InspectionPipeline

router = APIRouter(tags=["review"], dependencies=[Depends(get_current_user)])

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
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
) -> ReviewOut:
    # 仲裁（arbitrator）仅审核员/管理员可执行（§12.2 状态机权限分离）
    if body.role == "arbitrator" and not current_user.is_auditor:
        raise HTTPException(
            status_code=403,
            detail={"code": "FORBIDDEN", "message": "仲裁仅限审核员/管理员"},
        )
    # 复核人默认取登录用户（闭合 T1/T2 操作者身份占位）；显式提供时以显式值为准。
    reviewer = body.reviewer or current_user.username
    pipeline = InspectionPipeline(reg)
    try:
        return ReviewOut(
            **pipeline.apply_review(
                image_id=body.image_id,
                reviewer=reviewer,
                role=body.role,
                actor=current_user.username,
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
