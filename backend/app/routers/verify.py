"""影像质量校验 + 报告数字签名校验。

两部分：
1. POST /api/v1/verify：multipart 上传 → 黑度估计 + 线型 IQI
   识别 + 伪缺陷筛查 → evaluable（单图轻量同步计算）。
2. POST /api/v1/report/{id}/verify：报告数字签名校验——生成时
   PdfReporter.build 计算内容指纹（关键字段 canonical JSON → SHA-256）写入
   reports.report_hash，并对指纹做 SM2 数字签名（SM3withSM2）落 sidecar
   文件 <pdf>.sig；本端点重算指纹比对（valid 字段）并用 sidecar 内公钥
   验签（signature.valid 字段），返回双结果。向后兼容：无签名的旧报告
   signature.valid=null 而非报错。
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
from backend.domain.film_region import FilmRegionCfg as DomainFilmRegionCfg
from backend.domain.film_region import detect_film_region
from backend.domain.iqi import IqiConfig, enrich_grade, verify_iqi
from backend.domain.pseudo_defect import PseudoDefectCfg, screen_pseudo_defects
from backend.infra.image_loader import load_image

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
    photo_mode: bool = False  # 翻拍影像（绝对黑度不可测，门禁按 photo_policy 降级）
    warnings: list[str] = []  # 翻拍/门禁告警（与 run_inspection 的 warnings 同源）


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

    # 底片区域分割（与 run_inspection 同语义）：翻拍照片的黑度/IQI/伪缺陷
    # 都须在胶片区上评估，否则灯箱亮背景会稀释整图灰阶。
    fr = reg.config.film_region
    film = None
    if fr.enabled:
        film = detect_film_region(
            gray,
            DomainFilmRegionCfg(
                min_area_frac=fr.min_area_frac,
                max_photo_area_frac=fr.max_photo_area_frac,
                surround_bright_gray=fr.surround_bright_gray,
                surround_min_frac=fr.surround_min_frac,
            ),
        )
    if film is not None and not (film.is_photo or film.area_frac >= 0.7):
        film = None
    photo_mode = bool(
        film is not None and film.is_photo and meta.bit_depth == 8 and meta.density_array is None
    )
    eval_gray = (
        gray[film.y : film.y + film.h, film.x : film.x + film.w] if film is not None else gray
    )

    # 黑度须基于原始存储灰阶（含位深），否则 min-max 拉伸会破坏绝对光学密度；
    # 翻拍影像限定胶片掩膜，排除灯箱亮背景对平均灰阶的稀释。
    density = float(
        estimate_density(
            meta.density_array if meta.density_array is not None else gray,
            bit_depth=meta.bit_depth,
            mask=film.mask if film is not None else None,
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
    # 用户给 ROI 时按原图坐标在全图上验证；无 ROI 时在胶片区上自动定位。
    iqi = verify_iqi(gray if roi is not None else eval_gray, iqi_cfg, roi=roi, iqi_type=iqi_type)
    # 用透照厚度 + 参考表补全 A/AB/B 等级（厚度缺失则 grade=None）。
    iqi = enrich_grade(iqi, thickness_mm, iqi_cfg.sensitivity)

    # 伪缺陷筛查。仅严重项默认阻断。
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
    pd = screen_pseudo_defects(eval_gray, pd_domain)

    # 翻拍影像降级（与 run_inspection 的 photo_advisory 同语义）：
    # 黑度/IQI 门禁不阻断，转为告警 + 人工复核。
    photo_advisory = bool(photo_mode and reg.config.density.photo_policy == "warn")
    warnings: list[str] = []
    if photo_mode:
        warnings.append(
            f"翻拍影像：绝对黑度不可测（8bit 照片黑度上限 2.41），"
            f"胶片区估算 D={density:.2f} 仅供参考"
        )
        if not (density_ok and iqi.passed):
            reasons = []
            if not density_ok:
                reasons.append(f"黑度 {density:.2f} 超出配置范围")
            if not iqi.passed:
                reasons.append(f"IQI 未达要求（要求 {iqi.required}，实测 {iqi.achieved}）")
            warnings.append(
                "翻拍影像质量门禁未通过（" + "；".join(reasons) + "），已降级为人工复核"
            )

    # evaluable = 该影像按当前策略能否免阻断进入检测链路：
    # 翻拍降级时黑度/IQI 不构成阻断（与 run_inspection 的 IQIFailError 语义一致）。
    if photo_advisory:
        evaluable = bool(pd.passed)
    else:
        evaluable = bool(density_ok and iqi.passed and pd.passed)

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
        evaluable=evaluable,
        photo_mode=photo_mode,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# 报告数字签名校验（POST /api/v1/report/{id}/verify）
# ---------------------------------------------------------------------------


class SignatureCheckOut(BaseModel):
    """SM2 签名校验结果（独立于指纹比对的双结果之一）。"""

    valid: bool | None  # True=验签通过；False=不通过；None=无签名（legacy）
    algo: str | None  # 签名算法（SM2）
    public_key: str | None  # 签名方 SM2 公钥（128 hex）
    reason: str | None = None  # missing | invalid_sidecar | fingerprint_mismatch | mismatch


class VerifyOut(BaseModel):
    report_id: str
    valid: bool | None  # True=一致；False=不一致；None=无法校验（legacy 无指纹）
    hash: str | None
    signer: str | None
    generated_at: str | None
    reason: str | None = None  # legacy | mismatch 说明
    signature: SignatureCheckOut | None = None  # SM2 验签结果（None=sidecar 不可用）


def _sm2_verify(public_key_hex: str, signature_hex: str, data: bytes) -> bool:
    """用 sidecar 内公钥做 SM2 验签（SM3withSM2），不依赖本机私钥/主密钥。

    信任模型（诚实声明）：公钥随报告 sidecar 分发，若攻击者能同时替换
    PDF 与 sidecar 则可整体重签——sidecar 签名提供的是完整性/出处校验，
    抵赖性防护需配合硬件托管密钥（PKCS#11 provider）与公钥备案。
    """
    from gmssl import sm2

    try:
        verifier = sm2.CryptSM2(private_key="", public_key=public_key_hex)
        return bool(verifier.verify_with_sm3(signature_hex, data))
    except (ValueError, TypeError, KeyError):
        return False


def _check_report_signature(rep: dict, recomputed: str) -> SignatureCheckOut:
    """读取报告签名 sidecar 并验签；无 sidecar 的旧报告返回 valid=None。"""
    from backend.infra.reporting.pdf_reporter import read_signature_sidecar

    pdf_path = rep.get("pdf_path")
    sidecar = read_signature_sidecar(pdf_path) if pdf_path else None
    if not sidecar:
        return SignatureCheckOut(valid=None, algo=None, public_key=None, reason="missing")
    fingerprint = sidecar.get("fingerprint")
    if not isinstance(fingerprint, str):
        return SignatureCheckOut(
            valid=None,
            algo=sidecar.get("algo") if isinstance(sidecar.get("algo"), str) else None,
            public_key=sidecar.get("public_key"),
            reason="invalid_sidecar",
        )
    ok = _sm2_verify(sidecar["public_key"], sidecar["signature"], fingerprint.encode("utf-8"))
    if ok and fingerprint != recomputed:
        # 拼包防护：签名本身有效，但签的对象不是当前内容指纹
        return SignatureCheckOut(
            valid=False,
            algo=sidecar.get("algo"),
            public_key=sidecar.get("public_key"),
            reason="fingerprint_mismatch",
        )
    return SignatureCheckOut(
        valid=ok,
        algo=sidecar.get("algo"),
        public_key=sidecar.get("public_key"),
        reason=None if ok else "mismatch",
    )


@router.post("/report/{report_id}/verify", response_model=VerifyOut)
def verify_report(
    report_id: str,
    reg: Annotated[Registry, Depends(get_registry)],
) -> VerifyOut:
    """校验报告数字签名：重算内容指纹比对 + SM2 验签，返回双结果。"""
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
    # reportlab 导入较慢，延迟到实际校验时（不占进程导入→端口绑定关键路径）。
    from backend.infra.reporting.pdf_reporter import report_fingerprint

    image = repo.get_image(rep["image_id"])
    if image is None:  # pragma: no cover - 外键约束下不应出现
        raise HTTPException(
            status_code=404,
            detail={"code": "NOT_FOUND", "message": f"image not found: {rep['image_id']}"},
        )
    recomputed = report_fingerprint(image, image.get("defects") or [], rep)
    valid = recomputed == stored
    # SM2 验签（双结果之二）：独立于指纹比对；旧报告无 sidecar → valid=None。
    signature = _check_report_signature(rep, recomputed)
    return VerifyOut(
        report_id=report_id,
        valid=valid,
        hash=stored,
        signer=rep.get("signer"),
        generated_at=rep.get("generated_at"),
        reason=None if valid else "mismatch",
        signature=signature,
    )
