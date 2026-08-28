"""PDF 报告生成（reportlab，§7.2）。

规格书首选 weasyprint（HTML→PDF/A），但其在 Windows 缺 GTK/Pango 系统库
（本机实测 import 失败）；v1 改用 reportlab（纯 Python、全离线）。
中文使用系统黑体（simhei.ttf 已确认存在）。
M7 起报告经 pdfa.postprocess_to_pdfa 转写为 PDF/A-1b（字体已全量嵌入 + XMP
标识 + sRGB OutputIntent + 文档 ID），满足 §7.2 长期归档合规。

实现冻结契约 Reporter.build(image_id, template) -> pdf_path（interfaces.py）。
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable,
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from backend.domain.report.content import build_report_content
from backend.infra.reporting.pdfa import postprocess_to_pdfa
from backend.infra.reporting.templates import ReportTemplate, load_report_template
from backend.infra.repository import InspectionRepository


def _font_candidates() -> list[Path]:
    """跨平台中文字体候选（§部署硬化 #6，去 Windows 硬编码）。

    搜索顺序：
      1) SCAN_FONT_DIR 环境变量（部署时指向合规中文字体目录）；
      2) 捆绑字体目录 backend/infra/reporting/fonts/；
      3) 各平台系统字体（Windows/Linux/macOS 常见中英文字体具体路径 + 树递归）。

    全部缺失时由 _register_font 降级返回 Helvetica（reportlab 内置），
    保证非中文部署环境也能出片（中文退化为占位框，仅告警不崩溃）。
    """
    cands: list[Path] = []

    # 1) 自定义字体目录
    env_dir = os.environ.get("SCAN_FONT_DIR")
    if env_dir:
        d = Path(env_dir)
        if d.is_dir():
            cands.extend(sorted(d.glob("*.tt[cf]")) + sorted(d.glob("*.otf")))

    # 2) 捆绑字体
    bundled = Path(__file__).resolve().parent / "fonts"
    if bundled.is_dir():
        cands.extend(sorted(bundled.glob("*.tt[cf]")) + sorted(bundled.glob("*.otf")))

    # 3) 系统字体具体路径
    names = (
        "simhei.ttf",
        "simfang.ttf",
        "msyh.ttc",
        "msyh.ttf",
        "NotoSansCJK-Regular.ttc",
        "NotoSansCJKsc-Regular.otf",
        "NotoSansSC-Regular.otf",
        "wqy-zenhei.ttc",
        "wqy-microhei.ttc",
        "DroidSansFallback.ttf",
        "DroidSansFallbackFull.ttf",
        "PingFang.ttc",
        "STHeiti-Light.ttc",
        "STHeiti-Medium.ttc",
        "HiraginoSansGB.ttc",
    )
    sys_roots = [
        Path("C:/Windows/Fonts"),
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        Path("/Library/Fonts"),
        Path("/System/Library/Fonts"),
        Path.home() / ".fonts",
        Path.home() / ".local" / "share" / "fonts",
    ]
    for root in sys_roots:
        if not root.exists():
            continue
        for name in names:
            p = root / name
            if p.exists():
                cands.append(p)
        # Linux 常把字体放在子目录，轻量递归（仅文件名匹配）
        if root in (Path("/usr/share/fonts"), Path("/usr/local/share/fonts")):
            cands.extend(root.rglob("*.tt[cf]"))

    # 去重保序
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in cands:
        try:
            key = str(p.resolve())
        except OSError:
            key = str(p)
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


_PAGE_W, _PAGE_H = A4
_MARGIN = 15 * mm
_MAX_ANNOTATIONS = 50  # 图谱最多标注框数（防超大报告与糊图）
_EMBED_MAX_SIDE = 1600  # 嵌入报告的影像最长边（px）：底片常见 4k+，原尺寸会撑爆 PDF
_MAX_IMAGE_H = 95 * mm  # 单张嵌入影像最大高度，防止竖长图撑破版心

_LOG = logging.getLogger("scandetection.reporting")


class PdfReporter:
    """reportlab 实现的报告生成器（实现 Reporter Protocol）。"""

    def __init__(self, repository: InspectionRepository, output_dir: str) -> None:
        self._repo = repository
        self._out = Path(output_dir)
        self._out.mkdir(parents=True, exist_ok=True)
        self._font = _register_font()

    def build(self, image_id: str, template: str = "standard", gray=None) -> str:
        """生成 PDF/A-1b 报告，返回绝对路径。image_id 不存在抛 ValueError。

        §7.2 数字签名：按报告关键字段计算内容指纹（SHA-256），写入 PDF 页脚
        并持久化到 reports.report_hash / signed_at，供 verify 端点防篡改校验。

        gray：可选，pipeline 已加载的灰度底片（numpy）。传入则复用，避免对整张
        大底片二次解码（§优化 F12）；缺省时自行从 image["path"] 解码。
        """
        # 报告模板数据化（§7.2 P2）：模板名 → YAML 数据文件；未知/损坏回退 standard。
        tpl = load_report_template(template)
        image = self._repo.get_image(image_id)
        if image is None:
            raise ValueError(f"image not found: {image_id}")
        defects = image.get("defects") or []
        report = image.get("report")
        # 工业过渡路径（T1）：按报告所用标准表自行生成免责声明（与判定器同源），
        # 表缺失/解析失败时回退默认强声明，不阻断出片。
        disclaimer = _report_disclaimer(image.get("standard_id") or "")
        fingerprint = report_fingerprint(image, defects, report)
        content = build_report_content(
            image, defects, report, disclaimer=disclaimer or None, fingerprint=fingerprint
        )

        # 原图解码：pipeline 已传入灰度图则复用，否则自行解码（F12 避免重复解码）
        if gray is None:
            gray = _read_gray(image["path"])
        orig_bytes = _encode_png(gray)
        graph_bytes = _annotate_png(gray, defects)
        pdf_path = self._out / f"{image_id}.pdf"
        # 先渲染 reportlab PDF 到中间文件，再转写为 PDF/A-1b（§7.2 归档合规）
        rl_path = pdf_path.with_suffix(".rl.pdf")
        _render(rl_path, content, graph_bytes, orig_bytes, self._font, tpl)
        try:
            postprocess_to_pdfa(rl_path, pdf_path)
        finally:
            # 清理失败（只读/占用/回收站不可用）不应连带报告生成一起失败
            try:
                rl_path.unlink(missing_ok=True)
            except OSError:
                pass
        # 指纹写库（§7.2）：报告行存在则回填，缺行/写入失败不阻断出片。
        report_id = (report or {}).get("report_id")
        if report_id:
            try:
                self._repo.update_report(
                    report_id,
                    pdf_path=str(pdf_path),
                    report_hash=fingerprint,
                    signed_at=datetime.now(UTC).replace(tzinfo=None),
                )
            except (KeyError, OSError) as exc:  # pragma: no cover - 持久化尽力而为
                _LOG.warning("fingerprint persist failed report=%s: %s", report_id, exc)
        return str(pdf_path)


def report_fingerprint(image: dict, defects: list[dict], report: dict | None) -> str:
    """报告内容指纹：关键字段 canonical JSON → SHA-256 hex（§7.2 数字签名）。

    覆盖影像标识/工件/厚度/标定、级别/需复核、缺陷明细（类别/几何/当量/级别）、
    判定依据条款与签发人；字段按 key 排序、紧凑序列化，保证同内容稳定可复现。
    verify 端点用同一函数重算比对 → 防篡改。
    """
    payload = {
        "image_id": image.get("image_id") or image.get("id"),
        "workpiece_no": image.get("workpiece_no"),
        "weld_no": image.get("weld_no"),
        "pixel_spacing_mm": image.get("pixel_spacing_mm"),
        "base_metal_thickness_mm": image.get("base_metal_thickness_mm"),
        "joint_level": image.get("joint_level"),
        "need_review": bool(image.get("need_review", False)),
        "standard_id": image.get("standard_id"),
        "standard_version": image.get("standard_version"),
        "defects": [
            {
                "class_id": d.get("class_id"),
                "shape": d.get("shape"),
                "length_mm": d.get("length_mm"),
                "width_mm": d.get("width_mm"),
                "area_mm2": d.get("area_mm2"),
                "perimeter_mm": d.get("perimeter_mm"),
                "joint_level": d.get("joint_level"),
                "need_review": bool(d.get("need_review", False)),
            }
            for d in (defects or [])
        ],
        "report": {
            "joint_level": (report or {}).get("joint_level"),
            "standard_ref": (report or {}).get("standard_ref"),
            "signer": (report or {}).get("signer"),
            "basis": list((report or {}).get("basis") or []),
        },
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _report_disclaimer(standard_id: str) -> str:
    """按报告所用标准表生成免责声明（工业过渡路径，T1）。

    与后端判定器共用 disclaimer_for，保证 API / PDF / 报告一致。
    """
    from backend.domain.standards.tables.loader import disclaimer_for, load_standard_tables

    if standard_id:
        try:
            tables = load_standard_tables(standard_id)
            return disclaimer_for(tables)
        except Exception:  # noqa: BLE001, S110 - 表缺失/解析失败 → 回退默认，不阻断出片
            pass
    return ""


def _register_font() -> str:
    """注册第一个可用的中文字体，返回字体名（找不到则返回 Helvetica 并降级）。"""
    for path in _font_candidates():
        if path.exists():
            name = f"CN-{path.stem}"
            try:
                pdfmetrics.registerFont(TTFont(name, str(path)))
                return name
            except (OSError, ValueError, TypeError):
                pass  # 字体损坏/不兼容 → 尝试下一个候选
    return "Helvetica"


def _read_gray(image_path: str) -> np.ndarray | None:
    """以 unicode 安全方式读取灰度图（cv2.imread 在中文路径上会失败）。

    §7.5 静态加密兼容：影像副本可能为 AES-256-GCM 密文（魔数 b"SDC1"），
    检测到则用 SCAN_CRYPTO_KEY 解密后再解码；明文旧数据直接解码。
    密钥缺失/解密失败时返回 None（报告图谱降级为空，不抛 500）。
    """
    try:
        with open(image_path, "rb") as fh:
            buf = fh.read()
    except OSError:
        return None
    if not buf:
        return None
    _MAGIC = b"SDC1"
    if buf.startswith(_MAGIC):
        try:
            from backend.infra.crypto import AesCrypto, CryptoKeyError

            try:
                cipher = AesCrypto()
            except CryptoKeyError:
                return None
            buf = cipher.decrypt(buf)
        except Exception:  # noqa: BLE001  # 解密失败（密钥不符/密文损坏）→ 读图降级
            return None
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    return img


def _downscale(img: np.ndarray) -> np.ndarray:
    """长边限幅到 _EMBED_MAX_SIDE（等比）。工业底片常 4k+，原尺寸嵌两张会让
    PDF 膨胀到数十 MB 且渲染缓慢；限幅后仍足以人工核对缺陷位置。"""
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= _EMBED_MAX_SIDE or longest <= 0:
        return img
    scale = _EMBED_MAX_SIDE / float(longest)
    return cv2.resize(
        img,
        (max(1, round(w * scale)), max(1, round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _encode_png(img: np.ndarray | None) -> bytes | None:
    """送检原始影像（不标注）→ PNG bytes，用于报告『送检原始影像』对照。"""
    if img is None or img.size == 0:
        return None
    ok, buf = cv2.imencode(".png", _downscale(img))
    return buf.tobytes() if ok else None


def _annotate_png(img: np.ndarray | None, defects: list[dict]) -> bytes | None:
    """原图带缺陷框标注 → PNG bytes（报告缺陷图谱；无影像返回 None）。"""
    if img is None or img.size == 0:
        return None
    src = _downscale(img)
    h, w = src.shape[:2]
    sx = w / float(img.shape[1]) if img.shape[1] else 1.0
    sy = h / float(img.shape[0]) if img.shape[0] else 1.0
    canvas = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    for i, d in enumerate(defects[:_MAX_ANNOTATIONS], 1):
        bb = d.get("bbox_px")
        if not bb or len(bb) < 4:
            continue
        try:
            x, y, bw, bh = (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
        except (TypeError, ValueError):
            continue  # 脏数据不应中断整份报告
        # 框随降采样同步缩放，并裁剪到画布内（越界坐标会画到图外/不可见）
        x0 = max(0, min(w - 1, round(x * sx)))
        y0 = max(0, min(h - 1, round(y * sy)))
        x1 = max(0, min(w - 1, round((x + bw) * sx)))
        y1 = max(0, min(h - 1, round((y + bh) * sy)))
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.putText(
            canvas,
            _defect_label(i, d),
            (x0, max(y0 - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
    ok, buf = cv2.imencode(".png", canvas)
    return buf.tobytes() if ok else None


def _build_graph_bytes(image_path: str, defects: list[dict]) -> bytes | None:
    """按路径生成标注图（保留供外部/测试直接调用）。"""
    return _annotate_png(_read_gray(image_path), defects)


def _build_original_bytes(image_path: str) -> bytes | None:
    """按路径生成原始影像图（保留供外部/测试直接调用）。"""
    return _encode_png(_read_gray(image_path))


def _defect_label(idx: int, d: dict) -> str:
    """图上标签只用 ASCII：OpenCV Hershey 字体无中文字形，中文会被画成 '?'。

    中文类别名在"缺陷清单"表中以序号一一对应给出。
    """
    level = str(d.get("joint_level") or "").strip()
    return f"#{idx} {level}".strip() if level.isascii() else f"#{idx}"


def _comparison_flow(
    orig_bytes: bytes | None,
    anno_bytes: bytes | None,
    styles: dict[str, ParagraphStyle],
) -> list[Flowable]:
    """生成『送检原始影像 vs 检测标注影像』对照排版（并排，附图注）。"""
    items: list[tuple[str, bytes]] = []
    if orig_bytes:
        items.append(("送检原始影像（未标注）", orig_bytes))
    if anno_bytes:
        items.append(("检测标注影像（AI 定位缺陷，红框）", anno_bytes))

    if not items:
        return [Paragraph("（影像不可用，对比图省略）", styles["body"])]
    if len(items) == 1:
        cap, b = items[0]
        return _image_with_caption(b, _PAGE_W - 2 * _MARGIN, cap, styles)

    # 并排：两列等宽，中间留 4mm 间隙
    gap = 4 * mm
    col = (_PAGE_W - 2 * _MARGIN - gap) / 2
    cells: list[Flowable] = []
    caps: list[Paragraph] = []
    for cap, b in items:
        img = _scaled_image(b, col)
        cells.append(img)
        caps.append(Paragraph(cap, styles["caption"]))
    table = Table([cells, caps], colWidths=[col, col])
    table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, 0), 0),
                ("BOTTOMPADDING", (0, 1), (-1, 1), 2),
            ]
        )
    )
    return [table]


def _scaled_image(b: bytes, max_w: float, max_h: float = _MAX_IMAGE_H) -> Image:
    # reportlab>=5 不再接受裸 bytes，且 Image 不再接受 ImageReader 对象；
    # 必须用 file-like（BytesIO）直传给 Image，另用一个 BytesIO 取尺寸。
    reader = ImageReader(io.BytesIO(b))
    iw, ih = reader.getSize()
    if iw <= 0 or ih <= 0:
        raise ValueError("嵌入影像尺寸非法")
    # 同时受宽、高约束：仅按宽度缩放时，竖长底片会超出版心导致 LayoutError
    scale = min(1.0, max_w / iw, max_h / ih)
    return Image(io.BytesIO(b), width=iw * scale, height=ih * scale)


def _image_with_caption(
    b: bytes, max_w: float, cap: str, styles: dict[str, ParagraphStyle]
) -> list[Flowable]:
    return [_scaled_image(b, max_w), Paragraph(cap, styles["caption"])]


def _render(
    pdf_path: Path,
    content: object,
    graph_bytes: bytes | None,
    orig_bytes: bytes | None,
    font: str,
    tpl: ReportTemplate,
) -> None:
    """渲染报告 PDF（reportlab platypus 流式排版）。

    版式文案由模板数据驱动（§7.2 P2）：标题/小节/字段标签/回退文案/表头
    均取自 ReportTemplate，渲染算法不变——换模板 = 换 YAML，不改代码。
    """
    c = _cast(content)
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=_MARGIN,
        rightMargin=_MARGIN,
        topMargin=_MARGIN,
        bottomMargin=_MARGIN,
        title=f"{tpl.doc_title_prefix} {c.image_id}",
        author=tpl.author,
    )
    styles = _make_styles(font)
    flow: list[Flowable] = []

    # 封面标题
    flow.append(Paragraph(tpl.cover_title, styles["title"]))
    flow.append(Spacer(1, 4 * mm))
    flow.append(
        Paragraph(tpl.meta_report_no.format(v=c.report_id or tpl.fallback), styles["meta"])
    )
    flow.append(Paragraph(tpl.meta_image_id.format(v=c.image_id), styles["meta"]))
    flow.append(Paragraph(tpl.meta_generated_at.format(v=c.generated_at), styles["meta"]))
    flow.append(Spacer(1, 6 * mm))

    # 工件信息
    wf = tpl.workpiece_fields
    flow.append(_section(tpl.section_workpiece, styles))
    flow.append(
        _kv_table(
            [
                (wf.get("workpiece_no", "工件号"), c.workpiece_no or tpl.fallback),
                (wf.get("weld_no", "焊口编号"), c.weld_no or tpl.fallback),
                (wf.get("modality", "影像模态"), f"{c.modality}（{c.source_type}）"),
            ],
            styles,
        )
    )

    # 检测参数
    pf = tpl.params_fields
    flow.append(_section(tpl.section_params, styles))
    flow.append(
        _kv_table(
            [
                (
                    pf.get("pixel_spacing", "像素标定"),
                    f"{c.pixel_spacing_mm:.4f} mm/px" if c.pixel_spacing_mm else tpl.not_provided,
                ),
                (
                    pf.get("thickness", "母材厚度 T"),
                    f"{c.base_metal_thickness_mm} mm" if c.base_metal_thickness_mm else tpl.not_provided,
                ),
                (pf.get("standard", "执行标准"), c.standard_ref or tpl.fallback),
            ],
            styles,
        )
    )

    # 影像质量校验
    iqf = tpl.iqi_fields
    flow.append(_section(tpl.section_iqi, styles))
    iqi_txt = "通过" if c.iqi_pass else ("不通过" if c.iqi_pass is False else "未校验")
    achieved = (c.iqi_detail or {}).get("achieved") or tpl.fallback
    required = (c.iqi_detail or {}).get("required") or tpl.fallback
    flow.append(
        _kv_table(
            [
                (
                    iqf.get("iqi", "像质计"),
                    f"{iqi_txt}（达到丝号 {achieved} / 要求 {required}）",
                ),
                (
                    iqf.get("density", "黑度 D"),
                    f"{c.density:.3f}" if c.density is not None else tpl.fallback,
                ),
                (
                    iqf.get("density_limit", "黑度门限"),
                    "AB 级 2.0–4.5" if c.density_ok is not None else tpl.fallback,
                ),
                (
                    iqf.get("evaluable", "可评片性"),
                    "可评片" if c.evaluable else "不可评片（影像质量不达标）",
                ),
            ],
            styles,
        )
    )

    # 缺陷清单
    flow.append(_section(tpl.section_defects, styles))
    if c.defects:
        flow.append(_defects_table(c.defects, styles, tpl.defect_columns))
    else:
        flow.append(Paragraph(tpl.defects_empty, styles["body"]))

    # 检测影像对比：送检原始影像 vs 检测标注影像（并排，便于人工判断）
    flow.append(_section(tpl.section_comparison, styles))
    flow.extend(_comparison_flow(orig_bytes, graph_bytes, styles))

    # 判定依据
    flow.append(_section(tpl.section_basis, styles))
    if c.basis:
        for i, b in enumerate(c.basis, 1):
            flow.append(Paragraph(f"{i}. {b}", styles["body"]))
    else:
        flow.append(Paragraph(tpl.basis_empty, styles["body"]))

    # 结论
    flow.append(_section(tpl.section_conclusion, styles))
    if c.joint_level:
        flow.append(Paragraph(tpl.level_text.format(level=c.joint_level), styles["verdict"]))
    else:
        flow.append(Paragraph(tpl.no_level_text, styles["verdict"]))
    if c.need_review:
        flow.append(Paragraph(tpl.review_warn, styles["warn"]))
    flow.append(Spacer(1, 4 * mm))
    flow.append(
        Paragraph(tpl.signer_text.format(signer=c.signer or "____________"), styles["meta"])
    )
    if c.fingerprint:
        flow.append(
            Paragraph(tpl.fingerprint_text.format(fp=c.fingerprint[:12]), styles["meta"])
        )
    flow.append(Spacer(1, 4 * mm))
    flow.append(Paragraph(tpl.disclaimer_label.format(text=c.disclaimer), styles["fine"]))

    doc.build(flow)


def _cast(content: object):
    from backend.domain.report.content import ReportContent

    if not isinstance(content, ReportContent):
        raise TypeError("PdfReporter._render 需要 ReportContent")
    return content


def _make_styles(font: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "t",
            parent=base["Title"],
            fontName=font,
            fontSize=18,
            leading=24,
            alignment=TA_CENTER,
            spaceAfter=6,
        ),
        "meta": ParagraphStyle(
            "m", parent=base["Normal"], fontName=font, fontSize=9, leading=13, textColor=colors.grey
        ),
        "section": ParagraphStyle(
            "s",
            parent=base["Heading2"],
            fontName=font,
            fontSize=13,
            leading=17,
            spaceBefore=6,
            spaceAfter=4,
            textColor=colors.HexColor("#1a3a5c"),
        ),
        "body": ParagraphStyle("b", parent=base["Normal"], fontName=font, fontSize=10, leading=15),
        "verdict": ParagraphStyle(
            "v",
            parent=base["Normal"],
            fontName=font,
            fontSize=13,
            leading=18,
            textColor=colors.HexColor("#8b0000"),
            spaceBefore=2,
        ),
        "warn": ParagraphStyle(
            "w",
            parent=base["Normal"],
            fontName=font,
            fontSize=10,
            leading=15,
            textColor=colors.HexColor("#b8860b"),
        ),
        "fine": ParagraphStyle(
            "f", parent=base["Normal"], fontName=font, fontSize=8, leading=11, textColor=colors.grey
        ),
        "cell": ParagraphStyle("c", parent=base["Normal"], fontName=font, fontSize=9, leading=12),
        "caption": ParagraphStyle(
            "cap",
            parent=base["Normal"],
            fontName=font,
            fontSize=8,
            leading=11,
            alignment=TA_CENTER,
            textColor=colors.grey,
        ),
    }


def _section(title: str, styles: dict[str, ParagraphStyle]) -> KeepTogether:
    return KeepTogether([Paragraph(title, styles["section"]), Spacer(1, 1 * mm)])


def _kv_table(rows: list[tuple[str, str]], styles: dict[str, ParagraphStyle]) -> Table:
    data = [[Paragraph(k, styles["cell"]), Paragraph(v, styles["cell"])] for k, v in rows]
    t = Table(data, colWidths=[40 * mm, None])
    t.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef2f7")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return t


def _defects_table(
    defects: tuple[dict, ...],
    styles: dict[str, ParagraphStyle],
    columns: tuple[str, ...] = (),
) -> Table:
    header = (
        list(columns)
        if columns
        else ["#", "类别", "形状", "长 L(mm)", "宽 W(mm)", "面积(mm²)", "周长(mm)", "评级"]
    )
    rows = [header]
    for i, d in enumerate(defects, 1):
        rows.append(
            [
                str(i),
                str(d.get("class_name") or "—"),
                str(d.get("shape") or "—"),
                _fmt(d.get("length_mm")),
                _fmt(d.get("width_mm")),
                _fmt(d.get("area_mm2")),
                _fmt(d.get("perimeter_mm")),
                str(d.get("joint_level") or "—"),
            ]
        )
    data = [[Paragraph(str(c), styles["cell"]) for c in r] for r in rows]
    t = Table(
        data, colWidths=[8 * mm, 22 * mm, 14 * mm, 20 * mm, 20 * mm, 22 * mm, 22 * mm, 14 * mm]
    )
    t.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3a5c")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f7fa")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return t


def _fmt(v: float | None) -> str:
    return f"{v:.2f}" if v is not None else "—"
