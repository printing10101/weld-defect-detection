""" 单元测试：PDF/A-1b 后处理。

直接对 reportlab 生成的 PDF 注入 XMP 标识 + sRGB OutputIntent + 文档 ID +
文件头 %PDF-1.4，并校验 is_pdfa_compliant 通过。字体嵌入由 reportlab（TTF）
负责，在本单元外由真实报告路径（test_review_api）覆盖。
"""

from __future__ import annotations

import io

from reportlab.pdfgen import canvas

from backend.infra.reporting.pdfa import is_pdfa_compliant, postprocess_to_pdfa


def _make_pdf(path) -> None:
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(100, 700, "PDF/A probe")
    c.save()
    path.write_bytes(buf.getvalue())


def test_postprocess_to_pdfa(tmp_path) -> None:
    src = tmp_path / "in.pdf"
    out = tmp_path / "out.pdf"
    _make_pdf(src)
    postprocess_to_pdfa(src, out)
    assert out.exists()
    assert out.read_bytes()[:8] == b"%PDF-1.4"

    compliant, info = is_pdfa_compliant(out)
    assert compliant, info
    assert info["has_metadata"] is True
    assert info["has_output_intent"] is True
    assert info["pdfaid"]["part"] == b"1"
    assert info["pdfaid"]["conformance"] == b"B"


def test_is_pdfa_rejects_plain_pdf(tmp_path) -> None:
    src = tmp_path / "plain.pdf"
    _make_pdf(src)
    compliant, info = is_pdfa_compliant(src)
    assert compliant is False
    # 普通 PDF 缺 PDF/A 要素（无 XMP 标识 / OutputIntent）
    assert info["has_metadata"] is False
    assert info["has_output_intent"] is False
