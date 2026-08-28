"""合规处置建议（P0-E，POST /api/v1/recommend）。

与 /judge 的区别：judge 在未授权/熔断时 422 硬失败；recommend **永不硬失败**——
评级不可得（未授权/方法标准/信息不足）时降级为「需人工复核」并附免责声明，
保证任何输入都有可追溯的处置建议（工业过渡路径的兜底语义）。

输入复用 JudgeRequest（判定所需全部信息），输出 = 建议 + 可用的评级快照。
评级计算仍走 grader（真实评级），建议引擎只消费结果（domain/recommend，独立适配器）。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.app.dependencies import Registry, get_registry
from backend.app.routers.judge import JudgeRequest, _defect_class
from backend.domain.dto import BBox, Detection, ImageMeta, Modality
from backend.domain.errors import GradingAmbiguousError
from backend.domain.recommend import Recommendation, recommend

router = APIRouter(tags=["recommend"])


class RecommendResponse(BaseModel):
    # 建议（始终有值，不会 422）
    disposition: str
    disposition_label: str
    actions: list[str]
    basis: list[str]
    disclaimer: str | None = None
    # 评级快照（熔断时为 None）
    joint_level: str | None = None
    need_review: bool = False
    standard_id: str
    standard_version: str | None = None


@router.post("/recommend", response_model=RecommendResponse)
def recommend_endpoint(
    req: JudgeRequest,
    reg: Annotated[Registry, Depends(get_registry)],
) -> RecommendResponse:
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

    level: str | None = None
    need_review = False
    standard_version: str | None = None
    disclaimer: str | None = None

    try:
        grader = reg.grader_for(req.standard_id)
        result = grader.grade(defects, context)
        level = result.joint_level.value
        need_review = result.need_review
        standard_version = result.standard_version
        disclaimer = result.disclaimer
    except GradingAmbiguousError:
        # 熔断（未授权/信息不足）：降级为人工复核建议，不硬失败
        standard_version = None
        disclaimer = (
            "评级数值未授权或判定信息不足，已熔断禁止自动评级；"
            "本处置建议仅为「需人工复核」，不构成任何合格/不合格结论。"
        )
        level = None
        need_review = True

    rec: Recommendation = recommend(
        level,
        defects,
        need_review=need_review,
        standard_id=req.standard_id,
        disclaimer=disclaimer,
    )
    return RecommendResponse(
        disposition=rec.disposition,
        disposition_label=rec.disposition_label,
        actions=list(rec.actions),
        basis=list(rec.basis),
        disclaimer=rec.disclaimer,
        joint_level=level,
        need_review=need_review,
        standard_id=req.standard_id,
        standard_version=standard_version,
    )
