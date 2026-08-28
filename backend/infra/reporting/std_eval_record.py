"""附录A 评价记录表 PDF 生成（DB50/T 1807-2025 表A.1，reportlab）。

与评片报告（pdf_reporter.py）同一技术栈与字体策略；输出经 pdfa.postprocess_to_pdfa
转写 PDF/A-1b（长期归档）。资质不符合/FRR 未测时在表内明示"参考值"（比标准严）。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from backend.infra.reporting.pdf_reporter import _register_font
from backend.infra.reporting.pdfa import postprocess_to_pdfa

_TITLE = "承压设备射线检测缺陷自动识别系统评价记录表"
_NOTE = (
    "依据 DB50/T 1807-2025《承压设备射线检测缺陷自动识别系统评价方法》评价；"
    "系统分级取标准口径与严格口径（IOU 0.3）中较差者；"
    "评价人员须 RT(D)Ⅱ 级及以上、标注人员须 RTⅡ 级及以上（TSG Z8001）。"
)


def _chk(level: str | None, want: str) -> str:
    """分级勾选：■ 选中 / □ 未选。"""
    return "■" if level == want else "□"


def _risk_chk(risks: dict[str, str], kind: str, want: str) -> str:
    return "■" if risks.get(kind) == want else "□"


def build_record_pdf(record: dict[str, Any], out_pdf: str | Path) -> Path:
    """记录 dict（std_record.build_record）→ 表 A.1 PDF（PDF/A）。"""
    font = _register_font()
    out = Path(out_pdf)
    out.parent.mkdir(parents=True, exist_ok=True)

    rl_path = out.with_suffix(".pdf.rl")
    doc = SimpleDocTemplate(
        str(rl_path),
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title=_TITLE,
    )
    cell = ParagraphStyle("cell", fontName=font, fontSize=9, leading=12)
    head = ParagraphStyle("head", fontName=font, fontSize=14, leading=18, alignment=1)
    small = ParagraphStyle("small", fontName=font, fontSize=8, leading=10)

    m, film, per, met, grad, risks = (
        record["meta"],
        record["film"],
        record["personnel"],
        record["metrics"],
        record["grading"],
        record["risks"],
    )
    level = grad["level"]

    def P(text: str, style: ParagraphStyle = cell) -> Paragraph:
        return Paragraph(str(text), style)

    ev = per["evaluators"] or [{}]
    lb = per["labelers"] or [{}, {}, {}]
    rows: list[list[Any]] = [
        [P("系统名称"), P(m["system_name"]), P("系统版本号"), P(m["system_version"]), P("评价日期"), P(m["eval_date"])],
        [P("开发单位"), P(m["developer"]), P("系统负责人"), P(m.get("operator", "")), P("联系方式"), P(m["contact"])],
        [P("单位地址"), P(m["address"]), P(""), P(""), P(""), P("")],
        [P("评价人员姓名"), P(ev[0].get("name", "—")), P("持证类型"), P(ev[0].get("cert_type", "—")), P("证书有效期"), P(ev[0].get("valid_until") or "未注明")],
        [P("检测方法"), P(film["kind"]), P("透照布置"), P(film["exposure_layout"]), P("焊缝形式/焊接方法"), P(f'{film["weld_form"]} / {film["weld_method"]}')],
        [P("标准测试集缺陷数量（个）"), P(film["n_defects"]), P("缺陷底片数量（张）"), P(film["n_defect_images"]), P("无缺陷底片数量（张）"), P(film["n_no_defect_images"])],
        [P("各类缺陷占比"), P(film["class_distribution"]), P(""), P(""), P(""), P("")],
        [P("标注人员"), P("；".join(x.get("name", "") for x in lb if isinstance(x, dict))), P("持证类型"), P("；".join(x.get("cert_type", "") for x in lb if isinstance(x, dict))), P("资质校验"), P("通过" if per["qualified"] else "不符合：" + "；".join(per["issues"][:3]))],
        [P("单类别 TDRn"), P(met["tdr_row"]), P(""), P(""), P(""), P("")],
        [P("单类别 FDRn"), P(met["fdr_row"]), P(""), P(""), P(""), P("")],
        [P("单类别 MDRn"), P(met["mdr_row"]), P(""), P(""), P(""), P("")],
        [P("单类别 FRRn"), P(met["frr_row"]), P(""), P(""), P(""), P("")],
        [P("KDR 重点关注缺陷识别率"), P(f'{met["kdr"]:.2%}'), P("WDR 综合缺陷识别率"), P(f'{met["wdr"]:.2%}'), P("TDR 综合正检率"), P(f'{met["tdr"]:.2%}')],
        [P("严格口径（IOU≥0.3）"), P(f'KDR={met["kdr_strict"]:.2%}；WDR={met["wdr_strict"]:.2%}；TDR={met["tdr_strict"]:.2%}；FRR={met["frr_strict"]:.2%}'), P(""), P(""), P(""), P("")],
        [P("底片误报率 FRR"), P(f'{met["frr"]:.2%}'), P("标准分级"), P(grad["level_standard"] or "未定级"), P("严格口径分级"), P(grad["level_strict"] or "未定级")],
        [P("系统分级结果"), P(f'{_chk(level, "L1")} L1　{_chk(level, "L2")} L2　{_chk(level, "L3")} L3　{_chk(level, "L4")} L4　{("（记录值，取严）" if level else "（未定级）")}')],
        [P("风险分析结果"), P(
            f'漏检风险：{_risk_chk(risks, "miss", "Ⅰ类")} Ⅰ类　{_risk_chk(risks, "miss", "Ⅱ类")} Ⅱ类　　'
            f'误检风险：{_risk_chk(risks, "false_detect", "Ⅰ类")} Ⅰ类　{_risk_chk(risks, "false_detect", "Ⅱ类")} Ⅱ类　　'
            f'误报风险：{_risk_chk(risks, "false_report", "Ⅰ类")} Ⅰ类　{_risk_chk(risks, "false_report", "Ⅱ类")} Ⅱ类'
        )],
        [P("评价人签字"), P(m.get("operator", "")), P("报告日期"), P(m["eval_date"]), P(""), P("")],
    ]

    tbl = Table(rows, colWidths=[32 * mm, 42 * mm, 28 * mm, 30 * mm, 22 * mm, 26 * mm])
    tbl.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("SPAN", (1, 6), (5, 6)),
        ("SPAN", (1, 8), (5, 8)),
        ("SPAN", (1, 9), (5, 9)),
        ("SPAN", (1, 10), (5, 10)),
        ("SPAN", (1, 11), (5, 11)),
        ("SPAN", (1, 13), (5, 13)),
        ("SPAN", (1, 15), (5, 15)),
        ("SPAN", (1, 16), (5, 16)),
        ("SPAN", (2, 2), (5, 2)),
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
    ]))

    story = [
        Paragraph(_TITLE, head),
        Spacer(1, 4 * mm),
        tbl,
        Spacer(1, 3 * mm),
        Paragraph(_NOTE, small),
    ]
    if grad.get("note"):
        story.append(Paragraph(f"⚠ {grad['note']}", small))
    doc.build(story)

    try:
        postprocess_to_pdfa(str(rl_path), str(out))
    except Exception:  # noqa: BLE001 - PDF/A 转写失败时保留普通 PDF，不阻断记录表输出
        rl_path.replace(out)
        return out
    finally:
        rl_path.unlink(missing_ok=True)
    return out
