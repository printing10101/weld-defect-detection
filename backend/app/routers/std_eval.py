"""DB50/T 1807-2025 标准评价 API（人员资质 / 附录A 记录表 / PDF 导出）。

与 CLI（backend/evaluation/run_std_eval.py）配合：CLI 产出指标 JSON
（data/eval/std_eval.json），本路由负责记录表装配与归档输出。
资质不满足或 FRR 未测时记录表标记"参考值"，不出正式分级。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.evaluation.qualification import (
    Personnel,
    check_personnel,
    load_personnel,
    save_personnel,
)
from backend.evaluation.std_record import build_record
from backend.infra.config import load_config
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
def create_record_pdf(record_name: str = "std_record") -> FileResponse:
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
