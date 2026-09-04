"""验证与交付物生成器（V-02~V-05，要求文档 §6 表 6-1）。

四类可提交证明材料，全部"活生成"（读真实产物/配置，不预填结论）：

- V-02 信创兼容性验证报告：解析 docs/国产化适配矩阵.md（唯一事实源），
  汇总已验证/待真机条目计数 + 逐条明细，未验证项如实列为"缺口"；
- V-03 稳定性测试报告：聚合 data/compliance/ 下 soak_*.json（S-07 长跑）与
  recovery_drill_*.json（S-13 恢复演练）实测记录 + 看门狗/备份调度配置现状，
  缺失的证据项明确列出（不允许无记录即"通过"）；
- V-04 评价体系合规报告：聚合 data/eval/ 评价产物（std_eval / 附录A 记录表 /
  Golden 评估）与 DB 评片审计，输出 DB50/T 1807 指标/分级/风险一览；
- V-05 边界声明：静态边界文本（第三方资质认定边界），内容固定、可校验。

产物 JSON + PDF（PDF/A-1b）落 data/compliance/，动作入双链审计（路由层）。
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_INSTALL_ROOT = Path(__file__).resolve().parents[3]
_MATRIX_MD = _INSTALL_ROOT / "docs" / "国产化适配矩阵.md"

# 矩阵"状态"列取值 → 归一结论（V-02 计数口径）
_STATUS_MAP: dict[str, str] = {
    "已验证": "verified",
    "代码就绪": "code_ready",
    "预留": "reserved",
    "待真机": "pending_hw",
}


def _now_str() -> str:
    return datetime.now(UTC).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# V-02 信创兼容性验证报告
# ---------------------------------------------------------------------------


def _parse_matrix_rows() -> list[dict[str, str]]:
    """解析适配矩阵 Markdown 的适配表行（| 项目 | 现状 | 状态 |）。"""
    rows: list[dict[str, str]] = []
    try:
        text = _MATRIX_MD.read_text(encoding="utf-8")
    except OSError:
        return rows
    section = ""
    for line in text.splitlines():
        m = re.match(r"^#{2,3}\s+(.*)$", line.strip())
        if m:
            section = m.group(1).strip()
            continue
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 3 or cells[0] in {"项目", "---", ":---"} or set(cells[0]) <= {"-", ":"}:
            continue
        status_raw = cells[2]
        status = "pending_hw"
        for key, norm in _STATUS_MAP.items():
            if key in status_raw:
                status = norm
                break
        rows.append(
            {"section": section, "item": cells[0], "status": status, "status_raw": status_raw}
        )
    return rows


def build_v02() -> dict[str, Any]:
    rows = _parse_matrix_rows()
    counts: dict[str, int] = {"verified": 0, "code_ready": 0, "reserved": 0, "pending_hw": 0}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    gaps = [
        {"item": r["item"], "status": r["status_raw"]} for r in rows if r["status"] != "verified"
    ]
    matrix_exists = _MATRIX_MD.is_file()
    return {
        "spec": "V-02",
        "title": "信创兼容性验证报告",
        "generated_at": _now_str(),
        "matrix_path": str(_MATRIX_MD),
        "matrix_found": matrix_exists,
        "counts": counts,
        "total_items": len(rows),
        # 结论口径：有矩阵且全部已验证才 pass；有矩阵但有未验证项 → warning；
        # 矩阵文档缺失 → fail。
        "overall": (
            "pass"
            if matrix_exists and rows and not gaps
            else ("warning" if matrix_exists and rows else "fail")
        ),
        "rows": rows,
        "gaps": gaps,
        "note": (
            "适配事实源为 docs/国产化适配矩阵.md（区分已验证/待真机）；"
            "“待真机”条目须在目标国产环境实测后回填验证记录表，"
            "本报告不将未实测条目表述为已适配。"
        ),
    }


# ---------------------------------------------------------------------------
# V-03 稳定性测试报告
# ---------------------------------------------------------------------------


def _latest_soaks(compliance_dir: Path, limit: int = 5) -> list[dict[str, Any]]:
    out = []
    for p in sorted(compliance_dir.glob("soak_*.json"), reverse=True)[:limit]:
        data = _read_json(p)
        if data:
            out.append(
                {
                    "file": p.name,
                    "conclusion": data.get("conclusion"),
                    "rounds": data.get("rounds"),
                    "success_rate": data.get("success_rate"),
                    "rss_slope_mb_per_round": data.get("rss_slope_mb_per_round"),
                    "leak_suspected": data.get("leak_suspected"),
                    "failures": data.get("failures"),
                    "planned_hours": data.get("planned_hours"),
                    "elapsed_sec": data.get("elapsed_sec"),
                    "rss_max_mb": data.get("rss_max_mb"),
                }
            )
    return out


def _latest_drills(compliance_dir: Path, limit: int = 5) -> list[dict[str, Any]]:
    out = []
    for p in sorted(compliance_dir.glob("recovery_drill_*.json"), reverse=True)[:limit]:
        data = _read_json(p)
        if data:
            steps = data.get("steps") or []
            out.append(
                {
                    "file": p.name,
                    "conclusion": data.get("conclusion"),
                    "steps_ok": sum(1 for s in steps if s.get("ok")),
                    "steps_total": len(steps),
                    "rto_sec": data.get("rto_sec"),
                    "total_elapsed_sec": data.get("total_elapsed_sec"),
                }
            )
    return out


def build_v03(reg) -> dict[str, Any]:
    compliance_dir = Path(reg.config.paths.data_dir) / "compliance"
    soaks = _latest_soaks(compliance_dir)
    drills = _latest_drills(compliance_dir)
    missing: list[str] = []
    if not soaks:
        missing.append("72h 长跑记录（scripts/soak_72h.py 产出缺失）")
    if not drills:
        missing.append("恢复演练记录（scripts/recovery_drill.py 产出缺失）")
    soak_pass = bool(soaks) and soaks[0].get("conclusion") == "PASS"
    drill_pass = bool(drills) and drills[0].get("conclusion") == "PASS"
    overall = (
        "pass"
        if not missing and soak_pass and drill_pass
        else ("warning" if soaks or drills else "fail")
    )
    return {
        "spec": "V-03",
        "title": "稳定性测试报告",
        "generated_at": _now_str(),
        "overall": overall,
        "watchdog": {
            "enabled": bool(getattr(reg.config, "watchdog", None) and reg.config.watchdog.enabled),
            "running": getattr(reg, "watchdog", None) is not None,
        },
        "backup_scheduler": {
            "interval_hours": getattr(reg.config.backup, "interval_hours", 0),
            "running": getattr(reg, "backup_scheduler", None) is not None,
        },
        "soak_reports": soaks,
        "recovery_drills": drills,
        # 断电续跑/失败隔离/模型回退的证据源：自动化测试套件（S-15/S-16/S-17 专项）
        "test_evidence": (
            "断电续跑（批次快照+retry）、单张失败隔离、模型加载失败回退由"
            " backend/tests 专项用例覆盖（test_batch_queue / test_model_registry 等），"
            " 以 CI 全量通过为准。"
        ),
        "missing": missing,
        "note": ("本报告只聚合实测记录；缺失的证据项列于 missing，无记录的项不表述为通过。"),
    }


# ---------------------------------------------------------------------------
# V-04 评价体系合规报告
# ---------------------------------------------------------------------------


def build_v04(reg) -> dict[str, Any]:
    from backend.evaluation.std_history import collect_history

    eval_dir = Path(reg.eval_dir)
    std_eval = _read_json(eval_dir / "std_eval.json")
    records = sorted(eval_dir.glob("std_record*.json"), reverse=True)
    latest_record = _read_json(records[0]) if records else None
    history_items = collect_history(reg.eval_dir, [], limit=200)
    # DB 侧评价留痕（审计链 action=inspect 计数）
    _, inspect_total = reg.repository.list_audit(action="inspect", limit=1)

    metrics: dict[str, Any] = {}
    level = None
    if std_eval and isinstance(std_eval.get("result"), dict):
        std = std_eval["result"].get("standard") or {}
        metrics = {k: std.get(k) for k in ("tdr", "wdr", "kdr", "frr")}
        level = std.get("level")
    elif latest_record and isinstance(latest_record.get("metrics"), dict):
        m = latest_record["metrics"]
        metrics = {k: m.get(k) for k in ("tdr", "wdr", "kdr", "frr")}
        grading = latest_record.get("grading") or {}
        level = grading.get("level")

    overall = "pass" if metrics else ("warning" if history_items else "fail")
    return {
        "spec": "V-04",
        "title": "评价体系合规报告（DB50/T 1807-2025）",
        "generated_at": _now_str(),
        "overall": overall,
        "metrics": metrics,
        "level": level,
        "history_count": len(history_items),
        "db_inspect_records": inspect_total,
        "std_eval_found": std_eval is not None,
        "std_record_found": latest_record is not None,
        "missing": [
            *([] if std_eval else ["std_eval.json（run_std_eval CLI 产出缺失）"]),
            *([] if latest_record else ["附录A 记录表（POST /std-eval/record 产出缺失）"]),
        ],
        "note": "指标/分级/风险以 data/eval 评价产物与审计链留痕为准；缺失项如实列出。",
    }


# ---------------------------------------------------------------------------
# V-05 边界声明
# ---------------------------------------------------------------------------

_BOUNDARY_TEXT = (
    "本系统为纯软件产品，其自评材料（分级保护自查、密码应用自评估说明、"
    "安全加固自检、信创适配矩阵、稳定性测试记录、评价体系合规报告）"
    "均为研发方自检结论，仅供甲方与测评机构参考。"
    "保密资质认定、商用密码应用安全性评估（密评）、等级保护测评等法定认定，"
    "须由具备相应资质的第三方机构按现行法规与标准另行出具，"
    "本软件随附的任何自检报告均不构成、也不应被解读为上述认定。"
    "涉密事项的最终认定以保密行政管理部门、测评机构及甲方保密办意见为准。"
)


def build_v05() -> dict[str, Any]:
    return {
        "spec": "V-05",
        "title": "边界声明",
        "generated_at": _now_str(),
        "overall": "pass",
        "statement": _BOUNDARY_TEXT,
        "note": "内容为固定边界文本；引用法规以发布机构现行有效版本为准。",
    }


# ---------------------------------------------------------------------------
# PDF 落盘
# ---------------------------------------------------------------------------


def _deliverable_pdf(report: dict[str, Any], out_path: str | Path) -> Path:
    from backend.infra.compliance.doc_pdf import build_doc_pdf

    meta: list[tuple[str, str]] = [
        ("编号", str(report.get("spec", ""))),
        ("生成时间", str(report.get("generated_at", ""))),
        ("总体结论", str(report.get("overall", "—")).upper()),
    ]
    sections: list[dict[str, Any]] = []

    if report["spec"] == "V-02":
        sections.append(
            {
                "heading": "适配矩阵汇总",
                "table": {
                    "head": ["结论", "条目数"],
                    "rows": [[k, str(v)] for k, v in report["counts"].items()],
                },
                "paragraphs": [report["note"]],
            }
        )
        sections.append(
            {
                "heading": "逐条明细",
                "table": {
                    "head": ["维度", "项目", "状态"],
                    "rows": [[r["section"], r["item"], r["status_raw"]] for r in report["rows"]],
                },
            }
        )
    elif report["spec"] == "V-03":
        sections.append(
            {
                "heading": "长跑记录（S-07）",
                "table": {
                    "head": ["文件", "结论", "轮次", "成功率", "RSS斜率(MB/轮)"],
                    "rows": [
                        [
                            s["file"],
                            str(s.get("conclusion")),
                            str(s.get("rounds")),
                            f"{(s.get('success_rate') or 0):.2%}",
                            str(s.get("rss_slope_mb_per_round")),
                        ]
                        for s in report["soak_reports"]
                    ],
                },
            }
        )
        sections.append(
            {
                "heading": "恢复演练记录（S-13）",
                "table": {
                    "head": ["文件", "结论", "步骤", "RTO(秒)"],
                    "rows": [
                        [
                            d["file"],
                            str(d.get("conclusion")),
                            f"{d.get('steps_ok')}/{d.get('steps_total')}",
                            str(d.get("rto_sec")),
                        ]
                        for d in report["recovery_drills"]
                    ],
                },
                "paragraphs": [report["test_evidence"], report["note"]],
            }
        )
    elif report["spec"] == "V-04":
        sections.append(
            {
                "heading": "指标一览",
                "table": {
                    "head": ["指标", "值"],
                    "rows": [
                        [k.upper(), f"{v:.2%}" if isinstance(v, float) else str(v)]
                        for k, v in report["metrics"].items()
                    ]
                    or [["（无评价产物）", "—"]],
                },
                "paragraphs": [
                    f"系统分级：{report['level'] or '未定级'}",
                    f"评价历史条目：{report['history_count']}；DB 评片留痕：{report['db_inspect_records']}",
                    report["note"],
                ],
            }
        )
    else:  # V-05
        sections.append({"heading": "声明", "paragraphs": [report["statement"], report["note"]]})

    if report.get("missing"):
        sections.append(
            {
                "heading": "缺口清单（如实列示，未验证不表述为通过）",
                "paragraphs": [f"· {m}" for m in report["missing"]],
            }
        )
    return build_doc_pdf(report["title"], meta, sections, out_path)


def write_deliverable(report: dict[str, Any], out_dir: str | Path) -> dict[str, str]:
    """交付物落盘（JSON + PDF/A），返回 {json, pdf} 路径。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).replace(tzinfo=None).strftime("%Y%m%d_%H%M%S")
    spec = report["spec"].lower()
    json_path = out / f"deliverable_{spec}_{ts}.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pdf_path = _deliverable_pdf(report, out / f"deliverable_{spec}_{ts}.pdf")
    return {"json": str(json_path), "pdf": str(pdf_path)}
