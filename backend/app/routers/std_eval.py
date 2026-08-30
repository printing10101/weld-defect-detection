"""DB50/T 1807-2025 标准评价 API（人员资质 / 附录A 记录表 / PDF 导出）。

与 CLI（backend/evaluation/run_std_eval.py）配合：CLI 产出指标 JSON
（data/eval/std_eval.json），本路由负责记录表装配与归档输出。
资质不满足或 FRR 未测时记录表标记"参考值"，不出正式分级。
另承载 DB50/T 1807 评价能力补全的端点：
- 三人标注一致性仲裁（E-07，domain/labeling/consensus）；
- I 类漏检风险证据包生成（E-08，evaluation/evidence）；
- 不合格底片留档查询（E-05，evaluation/gate_rejects）；
- 误报底片清单导出（E-10，run_std_eval 产出的 false_report_films）。
"""

from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from backend.app.auth import Principal, get_principal
from backend.app.dependencies import Registry, get_operator_name, get_registry
from backend.app.routers.export import ensure_export_allowed
from backend.domain.labeling.consensus import LabelBox, resolve_consensus
from backend.evaluation.evidence import build_evidence
from backend.evaluation.gate_rejects import GateRejectStore
from backend.evaluation.qualification import (
    Personnel,
    check_personnel,
    load_personnel,
    save_personnel,
)
from backend.evaluation.std_record import build_record
from backend.infra.config import load_config, resolve_config_path
from backend.infra.reporting.std_eval_record import build_record_pdf

router = APIRouter(tags=["std-eval"])


class PersonnelIn(BaseModel):
    name: str
    cert_type: str
    role: str  # "evaluator" | "labeler"
    cert_no: str = ""
    valid_until: str = ""  # ISO 日期，空=未注明


class PersonnelListIn(BaseModel):
    people: list[PersonnelIn]


class RecordIn(BaseModel):
    """评价记录表装配入参（指标来源 = CLI 产出的 std_eval JSON）。"""

    eval_result_path: str = "data/eval/std_eval.json"
    system_name: str = "承压设备射线检测缺陷自动识别系统"
    system_version: str = ""
    developer: str = ""
    contact: str = ""
    address: str = ""
    film_kind: str = ""  # RT / DR / CR
    exposure_layout: str = ""
    weld_form: str = "single"
    weld_method: str = "manual"
    n_defect_images: int = 0
    n_no_defect_images: int = 0
    record_name: str = "std_record"  # 落盘文件名（不含扩展名）
    # 三人标注一致性结果（resolve_consensus().to_dict()，可选；E-07 供附录A 引用）
    consensus: dict[str, Any] | None = None


def _personnel_path() -> Path:
    p = load_config().std_eval.personnel_path
    return Path(p if os.path.isabs(p) else Path.cwd() / p)


def _load_eval_result(rel_path: str) -> dict[str, Any]:
    p = Path(rel_path)
    if not p.is_absolute():
        p = Path.cwd() / rel_path
    if not p.is_file():
        raise HTTPException(
            404,
            f"标准评价结果不存在: {rel_path}（先运行 python -m backend.evaluation.run_std_eval）",
        )
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise HTTPException(422, f"评价结果 JSON 解析失败: {rel_path}") from None
    if "result" not in payload:
        raise HTTPException(422, "评价结果 JSON 缺少 result 字段")
    return payload


@router.get("/std-eval/personnel")
def get_personnel() -> dict[str, Any]:
    people = load_personnel(_personnel_path())
    qual = check_personnel(people)
    return qual


@router.put("/std-eval/personnel")
def put_personnel(body: PersonnelListIn) -> dict[str, Any]:
    if not body.people:
        raise HTTPException(422, "people 不能为空")
    people = [Personnel(**p.model_dump()) for p in body.people]
    save_personnel(people, _personnel_path())
    return check_personnel(people)


@router.post("/std-eval/record")
def create_record(body: RecordIn) -> dict[str, Any]:
    cfg = load_config().std_eval
    payload = _load_eval_result(body.eval_result_path)
    people = load_personnel(_personnel_path())
    result = payload["result"]
    record = build_record(
        result,
        system_name=body.system_name,
        system_version=body.system_version,
        developer=body.developer,
        contact=body.contact,
        address=body.address,
        film_kind=body.film_kind,
        exposure_layout=body.exposure_layout,
        weld_form=body.weld_form or cfg.weld_form,
        weld_method=body.weld_method or cfg.weld_method,
        n_defect_images=body.n_defect_images or payload.get("n_defect_images", 0),
        n_no_defect_images=body.n_no_defect_images or payload.get("n_no_defect_images", 0),
        people=people,
        consensus=body.consensus or payload.get("consensus"),
    )
    out_dir = Path(cfg.eval_dir)
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{body.record_name}.json"
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    record["record_json"] = str(out)
    return record


@router.post("/std-eval/record/pdf")
def create_record_pdf(
    request: Request,
    reg: Annotated[Registry, Depends(get_registry)],
    principal: Annotated[Principal, Depends(get_principal)],
    record_name: str = "std_record",
) -> FileResponse:
    """附录A 记录表 PDF 导出（C-14 受控：需导出审批/令牌）。"""
    ensure_export_allowed(f"std_eval:record_pdf:{record_name}", request, principal, reg)
    cfg = load_config().std_eval
    out_dir = Path(cfg.eval_dir)
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    record_path = out_dir / f"{record_name}.json"
    if not record_path.is_file():
        raise HTTPException(404, f"评价记录不存在: {record_name}（先 POST /std-eval/record）")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    pdf_path = build_record_pdf(record, out_dir / f"{record_name}.pdf")
    return FileResponse(pdf_path, filename=f"{record_name}.pdf", media_type="application/pdf")


# ---------------------------------------------------------------------------
# 三人标注一致性仲裁（DB50/T 1807-2025 E-07）
# ---------------------------------------------------------------------------


class AnnotationIn(BaseModel):
    annotator: str  # "A" | "B" | "C"（或实际姓名，三人各自独立）
    class_id: int
    bbox: list[float] = Field(min_length=4, max_length=4)  # [x,y,w,h] 像素


class ConsensusIn(BaseModel):
    annotations: list[AnnotationIn]
    threshold: float = 0.6  # 两两 IOU 判据（默认严于标准底线 0.5）


def _consensus_verdict(result) -> dict[str, Any]:
    """仲裁建议：作废清单非空即建议标注组长按 DB50/T 1807 §6 复核仲裁。"""
    if result.discarded:
        return {
            "needed": True,
            "suggestion": (
                f"{len(result.discarded)} 个标注框因类型不一致或 IOU<{result.threshold} 作废，"
                "建议标注组长从作废清单中复核仲裁（DB50/T 1807-2025 §6）"
            ),
        }
    return {"needed": False, "suggestion": "三人标注全部达成一致，无需仲裁"}


@router.post("/std-eval/consensus")
def post_consensus(body: ConsensusIn) -> dict[str, Any]:
    """提交三人标注（A/B/C 框+类型），返回仲裁结果（并集/作废清单/仲裁建议）。"""
    if not body.annotations:
        raise HTTPException(422, "annotations 不能为空")
    boxes = [
        LabelBox(annotator=a.annotator, class_id=a.class_id, bbox=tuple(a.bbox))
        for a in body.annotations
    ]
    try:
        result = resolve_consensus(boxes, body.threshold)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    out = result.to_dict()
    out["arbitration"] = _consensus_verdict(result)
    return out


# ---------------------------------------------------------------------------
# I 类漏检风险证据包（DB50/T 1807-2025 E-08）
# ---------------------------------------------------------------------------


class EvidenceDefectIn(BaseModel):
    defect_id: str
    class_id: int
    grade: str | None = None  # NB/T 47013.2 评级（judge per_defect_grade，可缺）


class EvidenceIn(BaseModel):
    film_path: str  # 底片路径（绝对或相对 CWD）
    film_id: str = ""
    defects: list[EvidenceDefectIn] = []  # 漏检缺陷清单
    gt_boxes: list[list[float]] = []  # 人工标注框（叠加左侧，绿）
    det_boxes: list[list[float]] = []  # 系统识别框（叠加右侧，红）


@router.post("/std-eval/evidence/{record_id}")
def create_evidence(record_id: str, body: EvidenceIn) -> dict[str, Any]:
    """生成"标注原图 vs 系统识别图"对照证据图与 manifest（落 eval_dir/evidence）。"""
    film = Path(body.film_path)
    if not film.is_absolute():
        film = Path.cwd() / film
    if not film.is_file():
        raise HTTPException(404, f"底片不存在: {body.film_path}")
    cfg = load_config().std_eval
    out_dir = Path(cfg.eval_dir)
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    try:
        manifest = build_evidence(
            record_id=record_id,
            film_path=film,
            film_id=body.film_id,
            defects=[d.model_dump() for d in body.defects],
            gt_boxes=body.gt_boxes,
            det_boxes=body.det_boxes,
            out_dir=out_dir / "evidence",
        )
    except (OSError, RuntimeError) as exc:
        raise HTTPException(422, f"证据包生成失败: {exc}") from None
    return manifest


# ---------------------------------------------------------------------------
# 不合格底片留档查询（DB50/T 1807-2025 E-05）
# ---------------------------------------------------------------------------


@router.get("/std-eval/gate-rejects")
def list_gate_rejects(
    limit: int = 50,
    offset: int = 0,
    operator: Annotated[str, Depends(get_operator_name)] = "",
) -> dict[str, Any]:
    """拦截留档台账（gate_rejects）：原因/dpi/位深/操作员，按时间降序。"""
    del operator  # 查询不记审计，仅读取台账
    db_path = resolve_config_path(load_config().paths.db_path)
    rows, total = GateRejectStore(str(db_path)).list(limit=limit, offset=offset)
    return {"total": total, "items": rows}


# ---------------------------------------------------------------------------
# 误报底片清单导出（DB50/T 1807-2025 E-10）
# ---------------------------------------------------------------------------


@router.get("/std-eval/false-reports")
def export_false_reports(
    request: Request,
    reg: Annotated[Registry, Depends(get_registry)],
    principal: Annotated[Principal, Depends(get_principal)],
    eval_result_path: str = "data/eval/std_eval.json",
    fmt: str = "json",
) -> Response:
    """导出误报底片清单（id+路径+误报框数；run_std_eval 产出）。fmt=json|csv。

    C-14 受控导出：需导出审批/令牌；C-10 密级管控：清单附加 secret_level 列，
    且含秘密/机密（≥2）底片时直接拒绝导出（高密级底片信息不出系统）。
    """
    ensure_export_allowed("std_eval:false_reports", request, principal, reg)
    payload = _load_eval_result(eval_result_path)
    films = []
    for f in payload.get("false_report_films", []):
        img = reg.repository.get_image(str(f.get("id") or ""))
        films.append({**f, "secret_level": int((img or {}).get("secret_level") or 0)})
    if any(f.get("secret_level", 0) >= 2 for f in films):
        reg.repository.append_audit(
            actor=principal.username,
            action="export_denied",
            object_type="export",
            object_id="std_eval:false_reports",
            before=None,
            after={"reason": "high_secret_level_films"},
            note="清单含秘密/机密底片，按 C-10 拒绝导出",
        )
        raise HTTPException(
            403,
            detail={
                "code": "SECRET_LEVEL_DENIED",
                "message": "误报清单含秘密/机密底片，按密级管控拒绝导出（C-10）",
            },
        )
    if fmt == "csv":
        buf = io.StringIO()
        writer = csv.DictWriter(
            buf, fieldnames=["id", "path", "n_false_reports", "secret_level"]
        )
        writer.writeheader()
        writer.writerows(films)
        return Response(
            content=buf.getvalue(),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": "attachment; filename=false_report_films.csv"},
        )
    return _json_response({"n_films": len(films), "films": films})


# ---------------------------------------------------------------------------
# 评价历史档案（E-15）：聚合 data/eval 评价 JSON + DB 评价记录 → 等级曲线数据源
# ---------------------------------------------------------------------------


@router.get("/std-eval/history")
def get_history(
    limit: int = 200,
    reg: Annotated[Registry, Depends(get_registry)] = None,
) -> dict[str, Any]:
    """评价历史时间线（E-15）：[{evaluated_at, model_version, level, tdr, wdr, frr, ...}]。

    数据源：
    - data/eval/ 下历次评价 JSON（run_std_eval 产出 / Golden 评估报告 / 附录A 记录表）；
    - DB 主审计链中 action=inspect 的评片结论（after.joint_level/级别结论，即
      DB 侧评价记录——每次评片结论随不可变审计链留档）。
    按 evaluated_at 降序；供前端等级曲线（TDR/FRR/级别 随时间）展示。
    """
    from backend.evaluation.std_history import collect_history

    # DB 评价记录：审计链 action=inspect 的评片结论（joint_level 即级别）。
    db_rows, _total = reg.repository.list_audit(action="inspect", limit=max(1, min(limit, 500)))
    db_records = [
        {
            "evaluated_at": r.get("created_at"),
            "model_version": None,
            "level": ((r.get("after") or {}).get("joint_level") or None),
            "source": "db_inspect",
        }
        for r in db_rows
    ]
    items = collect_history(reg.eval_dir, db_records, limit=max(1, min(limit, 500)))
    return {"total": len(items), "items": items}


def _json_response(payload: dict[str, Any]) -> Response:
    return Response(
        content=json.dumps(payload, ensure_ascii=False, indent=2),
        media_type="application/json",
    )
