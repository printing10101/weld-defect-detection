"""PDF/A-1b 归档合规后处理。

reportlab 生成 PDF 后，本模块注入长期归档（PDF/A-1b）所必需的要素：
- XMP 元数据（pdfaid part=1, conformance=B）——PDF/A 标识；
- OutputIntent（sRGB ICC 设备无关色彩空间）——PDF/A-1b 强制；
- 文档 ID（trailer /ID）——唯一性 / 可校验；
- 文件头 %PDF-1.4（PDF/A-1 基于 PDF 1.4）。

字体已由 reportlab 以 TTF 全量嵌入（无子集依赖），满足 PDF/A 字体嵌入要求。
本模块不依赖 pypdf 的 XMP 高级封装（其构造需要 stream 参数），改为直接把
XMP 包作为 /Metadata 流挂到 catalog，控制更稳。
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import TypedDict, cast

from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    ArrayObject,
    ByteStringObject,
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
    NumberObject,
    TextStringObject,
)


class PdfAError(RuntimeError):
    """PDF/A-1b 必要要素缺失（如 sRGB ICC 配置文件）。"""


# 打包内置 sRGB 配置文件（离线可用）；缺失时回退系统路径
_BUNDLED_ICC = Path(__file__).resolve().parent / "srgb.icc"
_SYSTEM_ICC_CANDIDATES = (
    Path("C:/Windows/System32/spool/drivers/color/sRGB Color Space Profile.icm"),
    Path("/usr/share/color/icc/sRGB.icc"),
    Path("/Library/ColorSync/Profiles/sRGB Profile.icc"),
)


def _locate_icc() -> Path | None:
    if _BUNDLED_ICC.exists():
        return _BUNDLED_ICC
    for p in _SYSTEM_ICC_CANDIDATES:
        if p.exists():
            return p
    return None


def _build_xmp_packet() -> bytes:
    """构造 PDF/A-1b 的 XMP 包（含 pdfaid 标识）。"""
    return (
        b'<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        b'<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
        b' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        b'  <rdf:Description xmlns:pdfaid="http://www.aiim.org/pdfa/ns/id/" rdf:about="">\n'
        b"   <pdfaid:part>1</pdfaid:part>\n"
        b"   <pdfaid:conformance>B</pdfaid:conformance>\n"
        b"  </rdf:Description>\n"
        b" </rdf:RDF>\n"
        b"</x:xmpmeta>\n"
        b'<?xpacket end="w"?>\n'
    )


def postprocess_to_pdfa(
    src: Path, out: Path, *, icc_path: Path | None = None, require_icc: bool = True
) -> None:
    """将 reportlab 生成的 PDF（src）转写为 PDF/A-1b 并写入 out。

    require_icc=True（默认）时，找不到 sRGB ICC 直接抛 PdfAError：OutputIntent 是
    PDF/A-1b 的强制要素，静默省略会产出"自称 PDF/A 实则不合规"的归档件，
    在合规审计场景下比报错更危险。仅在明确接受非归档件时才置 False。
    """
    reader = PdfReader(str(src))
    writer = PdfWriter()
    writer.append_pages_from_reader(reader)
    # PDF/A-1 基于 PDF 1.4：用公共 API 设定文件头，写出时 xref 偏移自洽
    # （写出后直接改字节，一旦头部长度变化即破坏 xref 偏移）。
    writer.pdf_header = "%PDF-1.4"

    # pypdf 私有 API（已在模块文档说明并经测试验证可用）
    root = cast("DictionaryObject", writer._root_object)

    # 1. XMP 元数据（/Metadata 流，PDF/A 标识）
    meta_stream = DecodedStreamObject()
    meta_stream.set_data(_build_xmp_packet())
    meta_stream[NameObject("/Subtype")] = NameObject("/XML")
    meta_ind = writer._add_object(meta_stream)
    root[NameObject("/Metadata")] = meta_ind

    # 2. OutputIntent（sRGB 设备无关色彩，PDF/A-1b 强制）
    icc = Path(icc_path) if icc_path is not None else _locate_icc()
    if icc is None or not icc.exists():
        if require_icc:
            raise PdfAError(
                "缺少 sRGB ICC 配置文件，无法生成 PDF/A-1b OutputIntent；"
                f"请提供 {_BUNDLED_ICC} 或安装系统 sRGB 配置文件"
            )
    else:
        profile_stream = DecodedStreamObject()
        profile_stream.set_data(icc.read_bytes())
        profile_stream[NameObject("/N")] = NumberObject(3)
        profile_stream[NameObject("/Alternate")] = NameObject("/DeviceRGB")
        profile_ind = writer._add_object(profile_stream)
        intent = DictionaryObject()
        intent[NameObject("/Type")] = NameObject("/OutputIntent")
        intent[NameObject("/S")] = NameObject("/GTS_PDFA1")
        intent[NameObject("/Info")] = TextStringObject("sRGB IEC61966-2.1")
        intent[NameObject("/OutputConditionIdentifier")] = TextStringObject("sRGB")
        intent[NameObject("/DestOutputProfile")] = profile_ind
        root[NameObject("/OutputIntents")] = ArrayObject([writer._add_object(intent)])

    # 3. 文档 ID（trailer /ID，PDF/A 要求），由输出路径 + 页数派生稳定值。
    # 注意属性名是 _ID 且必须是 ArrayObject[ByteStringObject]；写成 _id 或
    # 裸 tuple 都会被 pypdf 忽略——产出的 PDF 将没有 /ID。
    seed = str(out).encode("utf-8") + str(len(reader.pages)).encode("utf-8")
    doc_id = hashlib.sha256(seed).digest()[:16]
    writer._ID = ArrayObject([ByteStringObject(doc_id), ByteStringObject(doc_id)])

    # 4. 原子写出：先落临时文件再 replace，避免转写中断留下半截"报告"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    with open(tmp, "wb") as fh:
        writer.write(fh)
    os.replace(tmp, out)


class _PdfAInfo(TypedDict):
    has_metadata: bool
    has_output_intent: bool
    has_doc_id: bool
    pdfaid: dict[str, bytes | None]
    header_1_4: bool


def is_pdfa_compliant(path: Path) -> tuple[bool, _PdfAInfo]:
    """结构校验：/Metadata(pdfaid) + /OutputIntents + trailer /ID + 1.4 文件头。

    非完整 PDF/A 验证器（不校验字体嵌入、透明度、加密等），仅覆盖本模块负责
    注入的要素，用于自检与回归测试。
    """
    path = Path(path)
    reader = PdfReader(str(path))
    root = cast("DictionaryObject", reader.trailer["/Root"])
    has_meta = "/Metadata" in root
    has_oi = "/OutputIntents" in root
    has_id = bool(reader.trailer.get("/ID"))
    pdfaid: dict[str, bytes | None] = {}
    if has_meta:
        meta = cast("DecodedStreamObject", root["/Metadata"]).get_data()
        pdfaid["part"] = b"1" if b"<pdfaid:part>1</pdfaid:part>" in meta else None
        pdfaid["conformance"] = (
            b"B" if b"<pdfaid:conformance>B</pdfaid:conformance>" in meta else None
        )
    with open(path, "rb") as fh:  # 只需前 8 字节，勿整份读入
        header_ok = fh.read(8) == b"%PDF-1.4"
    compliant = bool(has_meta and has_oi and has_id and pdfaid.get("part") == b"1" and header_ok)
    return compliant, {
        "has_metadata": has_meta,
        "has_output_intent": has_oi,
        "has_doc_id": has_id,
        "pdfaid": pdfaid,
        "header_1_4": header_ok,
    }
