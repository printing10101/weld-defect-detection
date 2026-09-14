"""报告生成。

两种模式：
- 上传新影像 + 表单 → InspectionPipeline 全链路（校验→预处理→检测→判定→落库→PDF）；
- 传 image_id → 对已入库检查重新生成报告（不重跑检测/判定）。
PDF 下载：GET /api/v1/report/{report_id}/pdf。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.app.auth import Principal, get_principal
from backend.app.dependencies import Registry, _resolve_path, get_operator_name, get_registry
from backend.app.llm_narrative import facts_from_report, generate
from backend.app.pipelines import InspectionPipeline
from backend.app.routers._common import parse_roi, staged_upload
from backend.app.routers.export import ensure_export_allowed
from backend.infra.fs import safe_resolve

_LOG = logging.getLogger("scandetection.report")

router = APIRouter(tags=["report"])


class FilmStampOut(BaseModel):
    """底片印字（扫描日期/编号）识别结论（run_inspection 结果快照）。"""

    status: str  # present | missing | unavailable | off
    text: str | None = None
    orientation: str | None = None  # normal | mirrored
    confidence: float | None = None
    need_review: bool = False


class ReportOut(BaseModel):
    report_id: str
    image_id: str
    joint_level: str | None
    need_review: bool
    evaluable: bool
    defect_count: int
    disclaimer: str | None = None
    # 合规处置建议：accept | conditional | rework | recheck
    disposition: str | None = None
    disposition_label: str | None = None
    disposition_actions: list[str] = []
    # 门禁降级/屏蔽告警透出：客户端须能看到"为什么转人工复核"（黑度越界/
    # dpi 未定/翻拍降级/印字区屏蔽等），而非只拿到 need_review 布尔
    warnings: list[str] = []
    # 判定依据/熔断原因快照（报告 PDF"判定依据"章节同源）
    basis: list[str] = []
    # 底片质量与检测姿态快照（密度/翻拍降级/检测工作模式/印字区屏蔽数）
    density: float | None = None
    density_ok: bool | None = None
    iqi_pass: bool | None = None
    iqi_detail: dict | None = None
    photo_mode: bool = False
    detect_mode: str | None = None
    stamp_zone_masked: int = 0
    # 单张评片查重：与历史影像内容哈希相同的记录摘要（批量查重的单张对应物，
    # 仅提示不阻断——是否重复由评片员判断）
    duplicates: list[dict] = []
    # 报告补充信息回显（委托单位/工程名称/工艺参数，键见 meta_fields 白名单）；
    # 前端报告页据此渲染《射线检测报告》样张式首页预览
    report_meta: dict[str, str] = {}
    # 首页预览所需表单回显（工件/焊缝/签字人/标准引用）
    workpiece_no: str | None = None
    weld_no: str | None = None
    signer: str | None = None
    standard_ref: str | None = None
    pdf_url: str
    # 底片印字性质快照（重新生成模式无 fresh 识别结果时为 None）
    stamp: FilmStampOut | None = None
    # AI 预筛级别标记：True 表示 joint_level 来自"底片质量未达标但用户显式请求
    # 预筛"的通道，级别不具合规效力（basis 首条给出降级原因，前端须显著标识）
    grade_preliminary: bool = False


class ReportDetectionsOut(BaseModel):
    """报告对应影像的缺陷明细。

    前端凭此取得逐缺陷的像素 bbox/class_id/置信度/不确定性，经人工复核后
    回流训练池（POST /active/export）。缺陷坐标为存储的 bbox_px 像素值；
    image_w/h 由原图读取，原图不可读时退回缺陷框并集以保证归一化可用。
    """

    report_id: str
    image_id: str
    image_stem: str
    image_w: int
    image_h: int
    defects: list[
        dict[str, Any]
    ]  # 每项：id,class_id,bbox,confidence,uncertainty,reviewed,need_review


@router.post("/report", response_model=ReportOut)
async def report(
    reg: Annotated[Registry, Depends(get_registry)],
    operator: Annotated[str, Depends(get_operator_name)],
    image: Annotated[UploadFile | None, File()] = None,
    image_id: Annotated[str | None, Form()] = None,
    pixel_spacing_mm: Annotated[float | None, Form()] = None,
    base_metal_thickness_mm: Annotated[float | None, Form()] = None,
    standard_id: Annotated[str | None, Form()] = None,
    iqi_roi: Annotated[str | None, Form()] = None,
    workpiece_no: Annotated[str | None, Form()] = None,
    weld_no: Annotated[str | None, Form()] = None,
    signer: Annotated[str | None, Form()] = None,
    template: Annotated[str | None, Form()] = None,
    # S-22 军标见证：可选军代表/见证人署名，透传到报告 PDF 签字栏（不传则不出该行）
    witness: Annotated[str | None, Form()] = None,
    # 报告补充信息（JSON 对象字符串）：委托单位/工程名称/材质/工艺参数等
    # 《射线检测报告》汇总表字段；后端按 meta_fields 白名单清洗，非法 JSON 422
    report_meta: Annotated[str | None, Form()] = None,
    # 底片不合格时的强制出片开关：出片但不输出级别（默认阻断，返回 409 IQI_FAIL）
    force: Annotated[bool, Form()] = False,
    # AI 预筛级别开关：底片质量未达标时仍计算级别，但结果被强标记为非正式级别
    # （basis 首条给出降级原因 + grade_preliminary=true）。与 force 的区别：force
    # 只允许"出片"（仍无级别），本开关额外请求"算出预筛级别"供 AI 辅助预筛；
    # 两者都不构成验收依据。默认 False = 维持"不合格底片不输出级别"的从严语义。
    allow_preliminary_grade: Annotated[bool, Form()] = False,
) -> ReportOut:
    pipeline = InspectionPipeline(reg)
    tpl = template or "standard"
    # 责任工程师签字默认取操作员（闭合 / 占位）；显式提供时优先。
    effective_signer = signer or operator
    actor = operator  # 审计 actor = 操作员（X-Operator-Name 头）
    meta = _parse_report_meta(report_meta)

    if pixel_spacing_mm is not None and pixel_spacing_mm <= 0:
        raise HTTPException(
            status_code=422,
            detail={"code": "INVALID_SPACING", "message": "pixel_spacing_mm 必须为正数"},
        )
    if base_metal_thickness_mm is not None and base_metal_thickness_mm <= 0:
        raise HTTPException(
            status_code=422,
            detail={"code": "INVALID_THICKNESS", "message": "base_metal_thickness_mm 必须为正数"},
        )

    if image_id:
        # 重生成模式：不重跑检测/判定。KeyError 需映射 404，否则落到全局 500。
        try:
            out = await run_in_threadpool(pipeline.regenerate_report, image_id, tpl)
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail={"code": "NOT_FOUND", "message": f"image not found: {image_id}"},
            ) from None
        duplicates: list[dict] = []
    elif image is not None:
        # 新评片模式：全链路（检测+渲染 PDF 为重同步任务，进线程池，）
        roi = parse_roi(iqi_roi)
        async with staged_upload(image, reg.config) as tmp_path:
            # 单张评片查重（批量查重的单张对应物）：在管线落库**前**查历史
            # 内容哈希，命中的必为既有记录；仅提示不阻断（摘要随响应透出）。
            file_hash = await run_in_threadpool(_sha256_of_file, tmp_path)
            duplicates = reg.repository.find_images_by_hashes([file_hash]).get(file_hash, [])
            # 单图链路同样落内容摘要（images.content_hash），批量查重的
            # 历史比对才能覆盖单张评片的影像（文件已落盘，整读一遍）。
            out = await run_in_threadpool(
                lambda: pipeline.run_inspection(
                    tmp_path,
                    pixel_spacing_mm=pixel_spacing_mm,
                    base_metal_thickness_mm=base_metal_thickness_mm,
                    standard_id=standard_id,
                    iqi_roi=roi,
                    workpiece_no=workpiece_no,
                    weld_no=weld_no,
                    signer=effective_signer,
                    actor=actor,
                    template=tpl,
                    force=force,
                    witness=witness,
                    content_sha256=file_hash,
                    report_meta=meta,
                    allow_preliminary_grade=allow_preliminary_grade,
                )
            )
    else:
        raise HTTPException(
            status_code=422,
            detail={"code": "MISSING_INPUT", "message": "需提供 image 文件或 image_id"},
        )

    warnings = list(out.get("warnings") or [])
    if duplicates:
        first = duplicates[0]
        warnings.append(
            f"与 {len(duplicates)} 张历史影像内容完全相同"
            f"（最早 {first.get('created_at') or '未知时间'}，image_id={first.get('image_id')}），"
            "请确认是否重复提交"
        )

    return ReportOut(
        report_id=out["report_id"],
        image_id=out["image_id"],
        joint_level=out["joint_level"],
        need_review=bool(out["need_review"]),
        evaluable=bool(out["evaluable"]),
        defect_count=int(out["defect_count"]),
        disclaimer=out.get("disclaimer"),
        disposition=out.get("disposition"),
        disposition_label=out.get("disposition_label"),
        disposition_actions=list(out.get("disposition_actions") or []),
        warnings=warnings,
        basis=list(out.get("basis") or []),
        density=out.get("density"),
        density_ok=out.get("density_ok"),
        iqi_pass=out.get("iqi_pass"),
        iqi_detail=out.get("iqi_detail"),
        photo_mode=bool(out.get("photo_mode", False)),
        detect_mode=out.get("detect_mode"),
        stamp_zone_masked=int(out.get("stamp_zone_masked") or 0),
        duplicates=duplicates,
        report_meta=dict(out.get("report_meta") or {}),
        workpiece_no=out.get("workpiece_no"),
        weld_no=out.get("weld_no"),
        signer=out.get("signer"),
        standard_ref=out.get("standard_ref"),
        pdf_url=f"/api/v1/report/{out['report_id']}/pdf",
        stamp=FilmStampOut(**out["stamp"]) if out.get("stamp") else None,
        grade_preliminary=bool(out.get("grade_preliminary", False)),
    )


def _parse_report_meta(raw: str | None) -> dict[str, str]:
    """解析表单里的报告补充信息 JSON（缺省/空串 → {}；非法 JSON → 422）。

    白名单清洗（未知键/空值丢弃、超长截断）在 sanitize_report_meta——
    非法 JSON 在此处显式 422（用户输入错误应当可见，而非静默丢弃）。
    """
    if not raw or not raw.strip():
        return {}
    from backend.domain.report.meta_fields import sanitize_report_meta

    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "INVALID_REPORT_META", "message": f"report_meta 不是合法 JSON: {exc}"},
        ) from None
    return sanitize_report_meta(parsed)


@router.get("/report/{report_id}/pdf")
def report_pdf(
    report_id: str,
    request: Request,
    reg: Annotated[Registry, Depends(get_registry)],
    principal: Annotated[Principal, Depends(get_principal)],
) -> FileResponse:
    rep = reg.repository.get_report(report_id)
    if rep is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "NOT_FOUND", "message": f"report not found: {report_id}"},
        )
    # C-14 导出管控：默认需保密员预授权或一次性导出令牌（export.require_approval）
    ensure_export_allowed(f"report:{report_id}", request, principal, reg)
    # 收敛到 reports_dir 之内，避免 DB 中若存了绝对路径导致的越界读取
    pdf = safe_resolve(
        Path(_resolve_path(reg.config.paths.reports_dir)),
        Path(str(rep["pdf_path"])).name,
    )
    if not pdf.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "code": "NOT_FOUND",
                "message": "报告 PDF 文件缺失（可能被移动或删除），请联系系统管理员检查数据目录",
            },
        )
    return FileResponse(str(pdf), media_type="application/pdf", filename=f"{report_id}.pdf")


def _sha256_of_file(path: Path) -> str:
    """整文件流式 SHA256（上传暂存文件为明文，可直读）。"""
    from hashlib import sha256

    digest = sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _image_dims(path: str | None) -> tuple[int, int]:
    """从原图副本读取宽高（像素）；缺失/损坏返回 (0, 0)。

    经 image_loader.read_gray 读取——副本默认为国密密文
    （security.encrypt=true），直接 cv2.imread 密文必然失败。
    """
    if not path:
        return (0, 0)
    try:
        from backend.infra.image_loader import read_gray

        arr = read_gray(path)
    except Exception:  # noqa: BLE001 - 读取失败不应阻断回流，交由调用方兜底
        return (0, 0)
    if arr is None:
        return (0, 0)
    return int(arr.shape[1]), int(arr.shape[0])


@router.get("/report/{report_id}/detections", response_model=ReportDetectionsOut)
def report_detections(
    report_id: str,
    reg: Annotated[Registry, Depends(get_registry)],
) -> ReportDetectionsOut:
    """报告对应影像的缺陷明细（主动学习回流数据源，）。

    闭环链路：评片 → 此处取明细 → 人工复核/改判 → POST /active/export 回流
    训练池（YOLO 标注 + 数据版本指纹）→ 供训练脚本合并重训。
    """
    rep = reg.repository.get_report(report_id)
    if rep is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "NOT_FOUND", "message": f"report not found: {report_id}"},
        )
    image_id = rep.get("image_id")
    if not image_id:
        raise HTTPException(
            status_code=404,
            detail={"code": "NOT_FOUND", "message": "该报告未关联任何影像记录"},
        )
    img = reg.repository.get_image(image_id)
    if img is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "NOT_FOUND", "message": f"image not found: {image_id}"},
        )
    path = img.get("path")
    image_w, image_h = _image_dims(path)
    defects = [
        {
            "id": d["id"],
            "class_id": int(d["class_id"]),
            "bbox": [float(v) for v in (d.get("bbox_px") or [0.0, 0.0, 0.0, 0.0])],
            "confidence": float(d.get("confidence", 0.0)),
            "uncertainty": float(d.get("uncertainty", 1.0)),
            "reviewed": bool(d.get("reviewed_by")),
            "need_review": bool(d.get("need_review", False)),
        }
        for d in (img.get("defects") or [])
    ]
    # 原图不可读（加密/缺失）：以缺陷框并集 + 边距为归一化基准，保证回流可用
    if (image_w <= 0 or image_h <= 0) and defects:
        max_x = max((d["bbox"][0] + d["bbox"][2] for d in defects), default=0.0)
        max_y = max((d["bbox"][1] + d["bbox"][3] for d in defects), default=0.0)
        image_w = max(1, int(max_x) + 1)
        image_h = max(1, int(max_y) + 1)
    return ReportDetectionsOut(
        report_id=report_id,
        image_id=image_id,
        image_stem=Path(path).stem if path else image_id,
        image_w=image_w,
        image_h=image_h,
        defects=defects,
    )


class NarrativeOut(BaseModel):
    """评片结论（本地大模型撰述）。

    status 语义（前端据此决定展示文本 / 原因 / 隐藏）：
    - ok          ：已生成，text 有效
    - disabled    ：配置关闭本地大模型（llm.enabled=false）
    - unavailable ：端点未运行/不可达
    - failed      ：端点可达但调用失败（含"输出未过合规清理"）
    - empty       ：报告或影像不存在（正常路径抛 404，此处为兜底）
    """

    report_id: str
    status: str
    text: str = ""
    model: str = ""
    reason: str = ""
    elapsed_ms: int = 0
    disclaimer: str = ""


@router.get("/report/{report_id}/narrative", response_model=NarrativeOut)
async def report_narrative(
    report_id: str,
    reg: Annotated[Registry, Depends(get_registry)],
    operator: Annotated[str, Depends(get_operator_name)],
) -> NarrativeOut:
    """本地大模型评片结论（按需生成）。

    只做"把算法事实写成可读结论"，**不参与评级**（合规红线，见 domain/narrative）。
    任何生成失败都如实降级为 status≠ok + reason，评片主链路从不因大模型不可用
    而失败——前端据此显示"AI 结论不可用及原因"，而不是空白或报错。
    """
    facts = await run_in_threadpool(facts_from_report, reg, report_id)
    if facts is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "NOT_FOUND", "message": f"report/image not found: {report_id}"},
        )
    outcome = await run_in_threadpool(generate, reg, facts)
    # 审计留痕：AI 撰述是"影响报告的派生内容"，须可追溯到操作者与模型。
    # 失败不阻断返回——审计不可用不应让评片员拿不到结论。
    try:
        reg.repository.append_audit(
            actor=operator,
            action="report_narrative",
            object_type="report",
            object_id=report_id,
            before=None,
            after={
                "status": outcome.status,
                "model": outcome.model,
                "chars": len(outcome.text),
            },
            note="本地大模型评片结论生成（辅助撰述，不参与评级）",
        )
    except Exception as exc:  # noqa: BLE001 - 审计失败不阻断结论返回
        _LOG.warning("评片结论审计写入失败 report=%s: %s", report_id, exc)
    return NarrativeOut(
        report_id=report_id,
        status=outcome.status,
        text=outcome.text,
        model=outcome.model,
        reason=outcome.reason,
        elapsed_ms=outcome.elapsed_ms,
        disclaimer=outcome.disclaimer,
    )
