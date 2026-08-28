"""影像质量校验 + 报告数字签名校验。

两部分：
1. POST /api/v1/verify（§4.2，原实现）：multipart 上传 → 黑度估计 + 线型 IQI
   识别 + 伪缺陷筛查 → evaluable（单图轻量同步计算）。
2. POST /api/v1/report/{id}/verify（§7.2，新增）：报告数字签名校验——生成时
   PdfReporter.build 计算内容指纹（关键字段 canonical JSON → SHA-256）写入
   reports.report_hash；本端点用同一函数重算比对，防篡改。
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from backend.app.dependencies import Registry, get_registry
from backend.app.routers._common import parse_roi, staged_upload
from backend.domain.density import check_density, estimate_density
from backend.domain.iqi import IqiConfig, enrich_grade, verify_iqi
from backend.domain.pseudo_defect import PseudoDefectCfg, screen_pseudo_defects
from backend.infra.image_loader import load_image
from backend.infra.reporting.pdf_reporter import report_fingerprint

router = APIRouter(tags=["verify"])


class IqiOut(BaseModel):
    iqi_type: str
    achieved: str | None
    required: str
    passed: bool
    grade: str | None  # A/AB/B 影像质量等级（需厚度；None=未定）


class PseudoDefectOut(BaseModel):
    passed: bool
    notes: tuple[str, ...]


class VerifyResponse(BaseModel):
    iqi: IqiOut
    density: float
    density_ok: bool
    pseudo_defect: PseudoDefectOut
    evaluable: bool


@router.post("/verify", response_model=VerifyResponse)
async def verify(
    image: Annotated[UploadFile, File()],
    reg: Annotated[Registry, Depends(get_registry)],
    iqi_roi: Annotated[str | None, Form()] = None,  # "x,y,w,h"（可选，缺省自动定位/全图）
    iqi_type: Annotated[str | None, Form()] = None,  # "wire" | "hole"（可选，缺省按配置）
    thickness_mm: Annotated[float | None, Form()] = None,  # 透照厚度（A/AB/B 等级映射用）
) -> VerifyResponse:
    roi = parse_roi(iqi_roi)
    async with staged_upload(image, reg.config) as tmp_path:
        return await run_in_threadpool(_verify_sync, reg, tmp_path, roi, iqi_type, thickness_mm)


def _verify_sync(
    reg: Registry,
    tmp_path: Path,
    roi: tuple[int, int, int, int] | None,
    iqi_type: str | None = None,
    thickness_mm: float | None = None,
) -> VerifyResponse:
    gray, meta = load_image(tmp_path)

    # 黑度须基于原始存储灰阶（含位深），否则 min-max 拉伸会破坏绝对光学密度。
    density = float(
        estimate_density(
            meta.density_array if meta.density_array is not None else gray,
            bit_depth=meta.bit_depth,
        )
    )
    density_ok = bool(check_density(density, reg.config.density.low, reg.config.density.high))

    iqi_cfg = IqiConfig(
        type=reg.config.iqi.type,
        wire_diameters_mm=tuple(reg.config.iqi.wire_diameters_mm),
        required_wire_no=reg.config.iqi.required_wire_no,
        hole_diameters_mm=tuple(reg.config.iqi.hole_diameters_mm),
        required_hole_no=reg.config.iqi.required_hole_no,
        min_contrast_ratio=reg.config.iqi.min_contrast_ratio,
        auto_locate=reg.config.iqi.auto_locate,
        locate_threshold=reg.config.iqi.locate_threshold,
        sensitivity=tuple(reg.config.iqi.sensitivity),
    )
    iqi = verify_iqi(gray, iqi_cfg, roi=roi, iqi_type=iqi_type)
    # 用透照厚度 + 参考表补全 A/AB/B 等级（厚度缺失则 grade=None，不臆造）。
    iqi = enrich_grade(iqi, thickness_mm, iqi_cfg.sensitivity)

    # 伪缺陷筛查（§4.2：划痕/尘点/显影不均）。仅严重项默认阻断。
    # 将 infra 配置适配为 domain 类型（隔离 pydantic，避免跨层字段耦合）。
    pd_cfg = reg.config.pseudo_defect
    pd_domain = PseudoDefectCfg(
        hough_threshold=pd_cfg.hough_threshold,
        scratch_min_ratio=pd_cfg.scratch_min_ratio,
        scratch_grating_min_lines=pd_cfg.scratch_grating_min_lines,
        canny_lo=pd_cfg.canny_lo,
        canny_hi=pd_cfg.canny_hi,
        uniformity_low_freq=pd_cfg.uniformity_low_freq,
        uniformity_max_ratio=pd_cfg.uniformity_max_ratio,
        dust_tophat_k=pd_cfg.dust_tophat_k,
        dust_min_area=pd_cfg.dust_min_area,
        dust_max_count=pd_cfg.dust_max_count,
        block_on_scratch=pd_cfg.block_on_scratch,
        block_on_uniformity=pd_cfg.block_on_uniformity,
        block_on_dust=pd_cfg.block_on_dust,
    )
    pd = screen_pseudo_defects(gray, pd_domain)

    return VerifyResponse(
        iqi=IqiOut(
            iqi_type=iqi.iqi_type,
            achieved=iqi.achieved,
            required=iqi.required,
            passed=iqi.passed,
            grade=iqi.grade,
        ),
        density=round(density, 3),
        density_ok=density_ok,
        pseudo_defect=PseudoDefectOut(passed=pd.passed, notes=pd.notes),
        evaluable=bool(density_ok and iqi.passed and pd.passed),
    )


# ---------------------------------------------------------------------------
# §7.2 报告数字签名校验（POST /api/v1/report/{id}/verify）
# ---------------------------------------------------------------------------


class VerifyOut(BaseModel):
    report_id: str
    valid: bool | None  # True=一致；False=不一致；None=无法校验（legacy 无指纹）
    hash: str | None
    signer: str | None
    generated_at: str | None
    reason: str | None = None  # legacy | mismatch 说明


@router.post("/report/{report_id}/verify", response_model=VerifyOut)
def verify_report(
    report_id: str,
    reg: Annotated[Registry, Depends(get_registry)],
) -> VerifyOut:
    """校验报告数字签名：重算内容指纹与签发时比对。"""
    repo = reg.repository
    rep = repo.get_report(report_id)
    if rep is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "NOT_FOUND", "message": f"report not found: {report_id}"},
        )
    stored = rep.get("report_hash")
    if not stored:
        return VerifyOut(
            report_id=report_id,
            valid=None,
            hash=None,
            signer=rep.get("signer"),
            generated_at=rep.get("generated_at"),
            reason="legacy",
        )
    image = repo.get_image(rep["image_id"])
    if image is None:  # pragma: no cover - 外键约束下不应出现
        raise HTTPException(
            status_code=404,
            detail={"code": "NOT_FOUND", "message": f"image not found: {rep['image_id']}"},
        )
    recomputed = report_fingerprint(image, image.get("defects") or [], rep)
    valid = recomputed == stored
    return VerifyOut(
        report_id=report_id,
        valid=valid,
        hash=stored,
        signer=rep.get("signer"),
        generated_at=rep.get("generated_at"),
        reason=None if valid else "mismatch",
    )
