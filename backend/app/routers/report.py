"""报告生成（§7.2，M6 真实实现）。

两种模式（§14 POST /api/v1/report）：
- 上传新影像 + 表单 → InspectionPipeline 全链路（校验→预处理→检测→判定→落库→PDF）；
- 传 image_id → 对已入库检查重新生成报告（不重跑检测/判定）。
PDF 下载：GET /api/v1/report/{report_id}/pdf。
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.app.dependencies import Registry, get_operator_name, get_registry, _resolve_path
from backend.app.pipelines import InspectionPipeline
from backend.app.routers._common import parse_roi, staged_upload
from backend.infra.fs import safe_resolve

router = APIRouter(tags=["report"])


class ReportOut(BaseModel):
    report_id: str
    image_id: str
    joint_level: str | None
    need_review: bool
    evaluable: bool
    defect_count: int
    disclaimer: str | None = None
    # 合规处置建议（P0-E）：accept | conditional | rework | recheck
    disposition: str | None = None
    disposition_label: str | None = None
    disposition_actions: list[str] = []
    pdf_url: str


class ReportDetectionsOut(BaseModel):
    """报告对应影像的缺陷明细（§5.5/§5.6 主动学习闭环回流用）。

    前端凭此取得逐缺陷的像素 bbox/class_id/置信度/不确定性，经人工复核后
    回流训练池（POST /active/export）。缺陷坐标为存储的 bbox_px 像素值；
    image_w/h 由原图读取，原图不可读时退回缺陷框并集以保证归一化可用。
    """

    report_id: str
    image_id: str
    image_stem: str
    image_w: int
    image_h: int
    defects: list[dict[str, Any]]  # 每项：id,class_id,bbox,confidence,uncertainty,reviewed,need_review


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
    # 底片不合格时的强制出片开关：出片但不输出级别（默认阻断，返回 409 IQI_FAIL）
    force: Annotated[bool, Form()] = False,
) -> ReportOut:
    pipeline = InspectionPipeline(reg)
    tpl = template or "standard"
    # 责任工程师签字默认取操作员（闭合 T1/T2 占位）；显式提供时优先。
    effective_signer = signer or operator
    actor = operator  # 审计 actor = 操作员（X-Operator-Name 头）

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
    elif image is not None:
        # 新评片模式：全链路（检测+渲染 PDF 为重同步任务，进线程池，§13.11）
        roi = parse_roi(iqi_roi)
        async with staged_upload(image, reg.config) as tmp_path:
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
                )
            )
    else:
        raise HTTPException(
            status_code=422,
            detail={"code": "MISSING_INPUT", "message": "需提供 image 文件或 image_id"},
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
        pdf_url=f"/api/v1/report/{out['report_id']}/pdf",
    )


@router.get("/report/{report_id}/pdf")
def report_pdf(
    report_id: str,
    reg: Annotated[Registry, Depends(get_registry)],
) -> FileResponse:
    rep = reg.repository.get_report(report_id)
    if rep is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "NOT_FOUND", "message": f"report not found: {report_id}"},
        )
    # 收敛到 reports_dir 之内，避免 DB 中若存了绝对路径导致的越界读取（§13.9）
    pdf = safe_resolve(
        Path(_resolve_path(reg.config.paths.reports_dir)),
        Path(str(rep["pdf_path"])).name,
    )
    if not pdf.exists():
        raise HTTPException(
            status_code=404, detail={"code": "NOT_FOUND", "message": "pdf file missing"}
        )
    return FileResponse(str(pdf), media_type="application/pdf", filename=f"{report_id}.pdf")


def _image_dims(path: str | None) -> tuple[int, int]:
    """从原图读取宽高（像素）；文件缺失/加密/损坏返回 (0, 0)。"""
    if not path:
        return (0, 0)
    try:
        import cv2

        arr = cv2.imread(path)
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
    """报告对应影像的缺陷明细（主动学习回流数据源，§5.5/§5.6）。

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
            detail={"code": "NOT_FOUND", "message": "report is not linked to any image"},
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
