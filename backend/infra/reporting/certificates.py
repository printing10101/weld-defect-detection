"""合规证明文档生成（C-12）：载体销毁证明 PDF。

复用 pdf_reporter 的字体注册；单页 A4 证明书版式（编号/载体信息/销毁方式/
双经办人/日期），落盘到指定路径。Minimal reportlab 直绘，不走 platypus 流式。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from backend.infra.reporting.pdf_reporter import _register_font, classification_label

_KIND_CN = {"film": "底片", "report": "报告", "backup": "备份"}


def build_destroy_certificate(carrier: dict[str, Any], out_path: str | Path) -> Path:
    """生成载体销毁证明 PDF（C-12），返回路径。

    证明要素：证明编号、载体编号/类型/关联对象/密级、销毁方式、销毁时间、
    发起（安全保密管理员）与确认（系统管理员）双经办人。
    """
    font = _register_font()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(out),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=25 * mm,
        bottomMargin=20 * mm,
        title=f"载体销毁证明 {carrier.get('carrier_id', '')}",
        author="ScanDetection",
    )
    title = ParagraphStyle("t", fontName=font, fontSize=22, leading=30, alignment=1)
    body = ParagraphStyle("b", fontName=font, fontSize=11, leading=18)
    level = classification_label(int(carrier.get("secret_level") or 0))
    rows = [
        ("载体编号", carrier.get("carrier_id") or "—"),
        ("载体类型", _KIND_CN.get(carrier.get("kind") or "", carrier.get("kind") or "—")),
        ("关联对象", carrier.get("object_id") or "—"),
        ("密级标识", f"{level}（{carrier.get('secret_level', 0)}）" if level else "非密"),
        ("责任人", carrier.get("owner") or "—"),
        ("销毁方式", carrier.get("destroy_method") or "—"),
        ("销毁发起（保密管理员）", carrier.get("destroy_requested_by") or "—"),
        ("销毁确认（系统管理员）", carrier.get("destroy_confirmed_by") or "—"),
        ("销毁时间", carrier.get("destroyed_at") or "—"),
    ]
    data = [[Paragraph(str(k), body), Paragraph(str(v), body)] for k, v in rows]
    t = Table(data, colWidths=[55 * mm, 100 * mm])
    t.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.6, "black"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    flow = [
        Paragraph("涉密载体销毁证明", title),
        Spacer(1, 10 * mm),
        Paragraph("兹证明下列涉密载体已按规定完成销毁，特此证明。", body),
        Spacer(1, 6 * mm),
        t,
        Spacer(1, 12 * mm),
        Paragraph("经办人签字：＿＿＿＿＿＿　　监销人签字：＿＿＿＿＿＿", body),
        Spacer(1, 6 * mm),
        Paragraph("单位（盖章）：＿＿＿＿＿＿　　日期：＿＿＿＿年＿＿月＿＿日", body),
    ]
    doc.build(flow)
    return out
