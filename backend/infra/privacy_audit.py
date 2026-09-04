"""脱敏残留审计（C-13）：DICONDE 患者 PHI 标签 + EXIF 残留扫描与报告。

对指定目录（默认影像库）逐文件扫描：
- DICOM（.dcm/.dicom/.ima）：复用 infra/diconde.audit_dicom_phi，报告患者隐私
  标签残留字段清单；
- 通用影像（.png/.jpg/.jpeg/.tif/.tiff/.bmp）：检查 EXIF 元数据是否剥离
  （拍摄设备/GPS 等属敏感残留）。

产出 JSON 报告 + PDF 报告（落 data/privacy/），供安全审计员归档查阅。
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.infra.diconde import audit_dicom_phi

_LOG = logging.getLogger("scandetection.privacy")

_DICOM_SUFFIXES = {".dcm", ".dicom", ".ima"}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _now_str() -> str:
    return datetime.now(UTC).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")


def audit_directory_phi(directory: str | Path, max_files: int = 5000) -> dict[str, Any]:
    """扫描目录下影像文件的 PHI/EXIF 残留，返回审计报告 dict。

    - clean=True 表示未发现残留；findings 逐文件列出残留字段；
    - max_files 防超大目录拖垮请求；损坏/不可读文件记入 errors 不中断。
    """
    root = Path(directory)
    findings: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    scanned = 0
    if root.is_dir():
        files = [
            p
            for p in sorted(root.rglob("*"))
            if p.is_file() and p.suffix.lower() in (_DICOM_SUFFIXES | _IMAGE_SUFFIXES)
        ][:max_files]
    else:
        files = []
    for p in files:
        scanned += 1
        try:
            if p.suffix.lower() in _DICOM_SUFFIXES:
                residues = audit_dicom_phi(str(p))
                kind = "dicom_phi"
            else:
                residues = _exif_tags(p)
                kind = "exif"
        except Exception as exc:  # noqa: BLE001 - 单文件失败不中断整体审计
            errors.append({"file": str(p), "error": str(exc)[:200]})
            continue
        if residues:
            findings.append({"file": str(p), "kind": kind, "residues": residues})
    return {
        "generated_at": _now_str(),
        "directory": str(root),
        "scanned": scanned,
        "n_findings": len(findings),
        "clean": scanned > 0 and not findings and not errors,
        "findings": findings,
        "errors": errors,
    }


def _exif_tags(path: Path) -> list[str]:
    """读取通用影像的 EXIF 标签清单（无 EXIF/不支持 → 空清单）。"""
    try:
        from PIL import Image

        with Image.open(path) as im:
            exif = im.getexif()
            if not exif:
                return []
            from PIL.ExifTags import TAGS

            return [TAGS.get(k, str(k)) for k in exif]
    except Exception:  # noqa: BLE001 - 非图/损坏 → 无 EXIF 可言
        return []


def build_phi_audit_pdf(report: dict[str, Any], out_path: str | Path) -> Path:
    """脱敏残留审计报告 → PDF（表格版式，复用报告字体注册）。"""
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

    font = _register_font()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(out),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=22 * mm,
        bottomMargin=18 * mm,
        title="脱敏残留审计报告",
        author="ScanDetection",
    )
    title = ParagraphStyle("t", fontName=font, fontSize=18, leading=26, alignment=1)
    body = ParagraphStyle("b", fontName=font, fontSize=10, leading=15)
    summary = [
        ("扫描时间", report.get("generated_at") or "—"),
        ("扫描目录", report.get("directory") or "—"),
        ("扫描文件数", str(report.get("scanned", 0))),
        ("发现残留", str(report.get("n_findings", 0))),
        ("结论", "未发现 PHI/EXIF 残留" if report.get("clean") else "存在残留或异常，需处置"),
    ]
    data = [[Paragraph(str(k), body), Paragraph(str(v), body)] for k, v in summary]
    t = Table(data, colWidths=[40 * mm, 120 * mm])
    t.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.6, "black"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    flow: list = [
        Paragraph("脱敏残留审计报告（C-13）", title),
        Spacer(1, 8 * mm),
        t,
        Spacer(1, 6 * mm),
    ]
    for f in report.get("findings", [])[:100]:
        residues = ", ".join(map(str, f.get("residues") or []))
        flow.append(Paragraph(f"· {f.get('file')}（{f.get('kind')}）：{residues}", body))
    if report.get("errors"):
        flow.append(Spacer(1, 3 * mm))
        flow.append(Paragraph("扫描异常文件：", body))
        for e in report["errors"][:50]:
            flow.append(Paragraph(f"· {e.get('file')}：{e.get('error')}", body))
    doc.build(flow)
    return out


def write_audit_report(report: dict[str, Any], out_dir: str | Path) -> dict[str, str]:
    """报告落盘（JSON + PDF），返回 {json, pdf} 路径。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).replace(tzinfo=None).strftime("%Y%m%d_%H%M%S")
    json_path = out / f"privacy_audit_{ts}.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pdf_path = build_phi_audit_pdf(report, out / f"privacy_audit_{ts}.pdf")
    return {"json": str(json_path), "pdf": str(pdf_path)}
