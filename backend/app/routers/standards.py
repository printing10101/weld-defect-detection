"""标准能力清单（§6.1，GET /api/v1/standards）。

返回全部已注册判定标准的能力目录（来自 domain/grade/registry 的真实元数据）：
- 是否输出缺陷级别（grades_defects）与级别体系（levels）；
- 状态：enabled（表可用且 authorized）/ unauthorized / tables_missing /
  method_standard（方法标准，不评级）；
- 标准语义说明（note），供前端/审计展示"该标准能做什么、不能做什么"。
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from backend.domain.grade.registry import all_standard_capabilities

router = APIRouter(tags=["standards"])


class StandardCapabilityOut(BaseModel):
    standard_id: str
    name: str
    grades_defects: bool  # 是否输出缺陷级别
    levels: list[str] | None  # 级别体系（I-IV 等；方法标准为 None）
    table_required: bool  # 是否需要评级数值表
    status: str  # enabled | unauthorized | tables_missing | method_standard
    note: str


@router.get("/standards", response_model=list[StandardCapabilityOut])
def list_standards() -> list[StandardCapabilityOut]:
    """标准能力清单（注册表真实元数据，非模拟）。"""
    return [StandardCapabilityOut(**c) for c in all_standard_capabilities()]
