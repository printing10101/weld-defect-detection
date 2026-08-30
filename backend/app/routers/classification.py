"""密级标识管理（C-10）：设定/变更影像密级（仅安全保密管理员）。

- POST /classification/image/{image_id}：设定/变更密级 + 定密依据（必填），
  同步该影像的报告行；动作入主审计链 + 独立安全审计链（C-19 双链）。
- GET  /classification/image/{image_id}：查询当前密级。
- GET  /classification/levels：密级枚举（前端展示用）。

密级嵌入（C-10）在报告渲染层完成：报告 PDF 全页横标 + 页脚定密依据见
infra/reporting/pdf_reporter.py；JSON/CSV 导出的密级管控见 routers/export.py
与 std_eval 误报清单端点。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.app.auth import Principal, require_role
from backend.app.dependencies import Registry, get_registry

router = APIRouter(prefix="/classification", tags=["classification"])

SECRET_LEVEL_NAMES = {0: "非密", 1: "内部", 2: "秘密", 3: "机密"}


class SecretLevelIn(BaseModel):
    secret_level: int = Field(ge=0, le=3)
    classification_basis: str = Field(min_length=1, max_length=256)


class SecretLevelOut(BaseModel):
    image_id: str
    secret_level: int
    secret_level_name: str
    classification_basis: str | None


@router.get("/levels")
def levels(_: Annotated[Principal, Depends(require_role("sysadmin", "secadmin", "auditor"))]) -> dict:
    """密级枚举与语义（C-10）。"""
    return {"levels": [{"level": k, "name": v} for k, v in SECRET_LEVEL_NAMES.items()]}


@router.get("/image/{image_id}", response_model=SecretLevelOut)
def get_secret_level(
    image_id: str,
    reg: Annotated[Registry, Depends(get_registry)],
) -> SecretLevelOut:
    image = reg.repository.get_image(image_id)
    if image is None:
        raise HTTPException(404, detail={"code": "NOT_FOUND", "message": f"image not found: {image_id}"})
    return SecretLevelOut(
        image_id=image_id,
        secret_level=int(image.get("secret_level") or 0),
        secret_level_name=SECRET_LEVEL_NAMES.get(int(image.get("secret_level") or 0), "非密"),
        classification_basis=image.get("classification_basis"),
    )


@router.post("/image/{image_id}", response_model=SecretLevelOut)
def set_secret_level(
    image_id: str,
    body: SecretLevelIn,
    principal: Annotated[Principal, Depends(require_role("secadmin"))],
    reg: Annotated[Registry, Depends(get_registry)],
) -> SecretLevelOut:
    """设定/变更密级（仅安全保密管理员；变更须登记定密依据，双链留痕）。"""
    try:
        snap = reg.repository.set_secret_level(
            image_id,
            secret_level=body.secret_level,
            classification_basis=body.classification_basis,
        )
    except KeyError as exc:
        raise HTTPException(404, detail={"code": "NOT_FOUND", "message": str(exc)}) from None
    except ValueError as exc:
        raise HTTPException(422, detail={"code": "INVALID_LEVEL", "message": str(exc)}) from None
    # C-19 双链：主审计链（统一查询）+ 独立安全审计链（防单链覆盖）
    reg.repository.append_audit(
        actor=principal.username,
        action="secret_level_change",
        object_type="image",
        object_id=image_id,
        before=snap["before"],
        after=snap["after"],
        note=body.classification_basis,
    )
    reg.security_store.append_security_audit(
        actor=principal.username,
        action="secret_level_change",
        object_type="image",
        object_id=image_id,
        before=snap["before"],
        after=snap["after"],
        note=body.classification_basis,
    )
    return SecretLevelOut(
        image_id=image_id,
        secret_level=body.secret_level,
        secret_level_name=SECRET_LEVEL_NAMES.get(body.secret_level, "非密"),
        classification_basis=body.classification_basis,
    )
