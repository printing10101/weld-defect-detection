"""附录A 评价记录表数据装配（DB50/T 1807-2025 §11.3 / 表A.1）。

把 std501807.evaluate 的结果 + 系统元数据 + 人员资质装配成表 A.1 的完整字段；
资质不满足（§5）时标记 qualified=False，正式分级结论降级为"参考值"（比标准严）。
"""

from __future__ import annotations

from datetime import date
from typing import Any

from backend.evaluation.qualification import Personnel, check_personnel
from backend.evaluation.std501807 import STD_CLASS_NAMES


def build_record(
    eval_result: dict[str, Any],
    *,
    system_name: str,
    system_version: str,
    developer: str,
    contact: str = "",
    address: str = "",
    film_kind: str = "",  # 检测方法：RT/DR/CR（标准底片类型）
    exposure_layout: str = "",  # 透照布置
    weld_form: str = "single",  # 焊缝形式：单面焊/双面焊
    weld_method: str = "manual",  # 焊接方法：手工焊/自动焊
    n_defect_images: int = 0,
    n_no_defect_images: int = 0,
    people: list[Personnel],
    operator: str = "",
) -> dict[str, Any]:
    """装配表 A.1 记录（返回可 JSON 化 dict，PDF 生成直接消费）。"""
    qual = check_personnel(people)
    std = eval_result["standard"]
    strict = eval_result["strict"]

    n_gt = sum(c["td"] + c["fd"] + c["md"] for c in std["per_class"].values())
    pct = lambda n: round(100 * (std["per_class"][n]["td"] + std["per_class"][n]["fd"] + std["per_class"][n]["md"]) / n_gt, 1) if n_gt else 0.0  # noqa: E731
    dist = "；".join(f"{STD_CLASS_NAMES[int(n)]}: {pct(n)}%" for n in sorted(std["per_class"], key=int))

    def _per_class_row(key: str) -> str:
        return "；".join(
            f"{STD_CLASS_NAMES[int(n)]}: {std['per_class'][n][key] * 100:.1f}%"
            for n in sorted(std["per_class"], key=int)
        )

    level = eval_result["level_recorded"]
    record: dict[str, Any] = {
        "meta": {
            "system_name": system_name,
            "system_version": system_version,
            "developer": developer,
            "contact": contact,
            "address": address,
            "eval_date": date.today().isoformat(),
            "operator": operator,
        },
        "film": {
            "kind": film_kind,
            "exposure_layout": exposure_layout,
            "weld_form": "单面焊" if weld_form == "single" else "双面焊",
            "weld_method": "手工焊" if weld_method == "manual" else "自动焊",
            "n_defect_images": n_defect_images,
            "n_defects": n_gt,
            "class_distribution": dist,
            "n_no_defect_images": n_no_defect_images,
        },
        "personnel": {
            "qualified": qual["qualified"],
            "issues": qual["issues"],
            "evaluators": qual["evaluators"],
            "labelers": qual["labelers"],
        },
        "metrics": {
            "tdr_row": _per_class_row("tdr"),
            "fdr_row": _per_class_row("fdr"),
            "mdr_row": _per_class_row("mdr"),
            "frr_row": _per_class_row("fdr"),  # 见下：FRRn=误报占比，装配时单独覆盖
            "kdr": std["kdr"],
            "wdr": std["wdr"],
            "tdr": std["tdr"],
            "frr": std["frr"],
            "iou_standard": std["iou_threshold"],
            "iou_strict": strict["iou_threshold"],
            "kdr_strict": strict["kdr"],
            "wdr_strict": strict["wdr"],
            "tdr_strict": strict["tdr"],
            "frr_strict": strict["frr"],
        },
        "grading": {
            "level": level,
            "level_standard": std["level"],
            "level_strict": strict["level"],
            "official": bool(qual["qualified"]) and std.get("frr_measured", False),
        },
        "risks": std["risks"],
        "std_result": eval_result,  # 全量结果（混淆矩阵等）随记录归档
    }
    # FRRn（式4）：无缺陷底片误报按预测类别的占比
    fr_total = sum(std["fr_by_class"].values())
    record["metrics"]["frr_row"] = "；".join(
        f"{STD_CLASS_NAMES[int(n)]}: {std['fr_by_class'][n] / fr_total * 100:.1f}%" if fr_total
        else f"{STD_CLASS_NAMES[int(n)]}: —"
        for n in sorted(std["fr_by_class"], key=int)
    )
    if not record["grading"]["official"]:
        record["grading"]["note"] = (
            "资质不符合或 FRR 未测（无缺陷测试集缺失），以下分级仅作参考值，不构成正式结论"
            if not qual["qualified"] or not std.get("frr_measured", False)
            else ""
        )
    return record
