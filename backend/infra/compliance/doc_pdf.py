"""合规报告通用 PDF 版式（C-23~C-25 共用）。

与评片报告/评价记录表同一技术栈（reportlab + 报告字体注册），输出经
pdfa.postprocess_to_pdfa 转写 PDF/A-1b（长期归档）。版式刻意朴素：
标题 + 摘要表 + 若干小节（段落 / 键值表 / 网格表），满足合规归档查阅即可。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from backend.infra.reporting.pdf_reporter import _register_font
from backend.infra.reporting.pdfa import postprocess_to_pdfa


def _esc(text: Any) -> str:
    """reportlab Paragraph 需要 XML 转义（& < >），其余字符原样。"""
    s = str(text if text is not None else "")
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_doc_pdf(
    title: str,
    meta: list[tuple[str, str]],
    sections: list[dict[str, Any]],
    out_path: str | Path,
) -> Path:
    """构造合规报告 PDF（PDF/A-1b）。

    - title   : 文档标题；
    - meta    : 摘要键值对（两列表格）；
    - sections: 小节列表，每节 {"heading": str,
                "paragraphs": [str,...], "table": {"head":[...], "rows":[[...]]}}；
    - out_path: 输出路径（先写临时 reportlab 文件，再转写 PDF/A 覆盖）。
    """
    font = _register_font()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    rl_path = out.with_name(out.name + ".rl")
    doc = SimpleDocTemplate(
        str(rl_path),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=20 * mm,
        bottomMargin=16 * mm,
        title=title,
        author="ScanDetection",
    )
    title_style = ParagraphStyle("t", fontName=font, fontSize=16, leading=22, alignment=1)
    head_style = ParagraphStyle("h", fontName=font, fontSize=12, leading=16, spaceBefore=6)
    body = ParagraphStyle("b", fontName=font, fontSize=9.5, leading=14)
    small = ParagraphStyle("s", fontName=font, fontSize=8.5, leading=12)

    def P(text: Any, style: ParagraphStyle = body) -> Paragraph:
        return Paragraph(_esc(text), style)

    flow: list[Any] = [Paragraph(_esc(title), title_style), Spacer(1, 6 * mm)]

    if meta:
        data = [[P(k, body), P(v, body)] for k, v in meta]
        t = Table(data, colWidths=[42 * mm, 118 * mm])
        t.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.5, "black"),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        flow += [t, Spacer(1, 5 * mm)]

    for sec in sections:
        flow.append(P(sec.get("heading", ""), head_style))
        for para in sec.get("paragraphs", []):
            flow.append(P(para, body))
        table = sec.get("table")
        if table:
            head = [str(h) for h in table.get("head", [])]
            rows = table.get("rows", [])
            data = [[P(h, small) for h in head]] if head else []
            for row in rows:
                data.append([P(c, small) for c in row])
            if data:
                ncols = max(len(r) for r in data)
                width = 178 * mm / ncols
                t = Table(data, colWidths=[width] * ncols, repeatRows=1 if head else 0)
                t.setStyle(
                    TableStyle(
                        [
                            ("GRID", (0, 0), (-1, -1), 0.4, "black"),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("TOPPADDING", (0, 0), (-1, -1), 3),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                        ]
                    )
                )
                flow.append(t)
        flow.append(Spacer(1, 4 * mm))

    doc.build(flow)
    postprocess_to_pdfa(rl_path, out)
    rl_path.unlink(missing_ok=True)  # 中间 reportlab 文件不留档
    return out
