"""PDF 报告生成（reportlab，）。

设计文档首选 weasyprint（HTML→PDF/A），但其在 Windows 缺 GTK/Pango 系统库
（本机实测 import 失败）；v1 改用 reportlab（纯 Python、全离线）。
中文优先系统宋体（simsun.ttc，与正式 RT 报告样张同款），缺失时按候选链降级。
 起报告经 pdfa.postprocess_to_pdfa 转写为 PDF/A-1b（字体已全量嵌入 + XMP
标识 + sRGB OutputIntent + 文档 ID），满足  长期归档合规。

实现冻结契约 Reporter.build(image_id, template) -> pdf_path（interfaces.py）。
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import os
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from backend.domain.report.content import build_report_content
from backend.infra.reporting.pdfa import postprocess_to_pdfa
from backend.infra.reporting.qrcode import qr_png_bytes, trace_code
from backend.infra.reporting.templates import ReportTemplate, load_report_template
from backend.infra.repository import InspectionRepository


def _font_candidates() -> list[Path]:
    """跨平台中文字体候选。

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

    # 3) 系统字体具体路径（宋体优先：正式 RT 报告样张使用宋体）
    names = (
        "simsun.ttc",
        "simfang.ttf",
        "simhei.ttf",
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
_MARGIN = 12 * mm
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

    def build(
        self, image_id: str, template: str = "standard", gray=None, witness: str | None = None
    ) -> str:
        """生成 PDF/A-1b 报告，返回绝对路径。image_id 不存在抛 ValueError。

         数字签名：按报告关键字段计算内容指纹（SHA-256），写入 PDF 页脚
        并持久化到 reports.report_hash / signed_at，供 verify 端点防篡改校验；
        国密化（C-03）后在此之外叠加 SM2 数字签名（SM3withSM2，对指纹签名），
        签名值落 sidecar 文件 <pdf>.sig（指纹本体已入 reports.report_hash，
        表结构不变故签名落文件）。未配置 SCAN_CRYPTO_KEY 时签名降级为仅
        指纹（不阻断出片，与静态加密的降级策略一致）。

        gray：可选，pipeline 已加载的灰度底片（numpy）。传入则复用，避免对整张
        大底片二次解码；缺省时自行从 image["path"] 解码。

        witness（S-22）：可选军代表/见证人署名，透传到签字栏（不传则不出该行）。
        诚实边界：witness 仅在本次生成时生效（不落库），重生成报告（regenerate）
        需再次传入。
        """
        # 报告模板数据化：模板名 → YAML 数据文件；未知/损坏回退 standard。
        tpl = load_report_template(template)
        image = self._repo.get_image(image_id)
        if image is None:
            raise ValueError(f"image not found: {image_id}")
        defects = image.get("defects") or []
        report = image.get("report")
        # 工业过渡路径：按报告所用标准表自行生成免责声明（与判定器同源），
        # 表缺失/解析失败时回退默认强声明，不阻断出片。
        disclaimer = _report_disclaimer(image.get("standard_id") or "")
        fingerprint = report_fingerprint(image, defects, report)
        content = build_report_content(
            image,
            defects,
            report,
            disclaimer=disclaimer or None,
            fingerprint=fingerprint,
            witness=witness,
        )

        # 原图解码：pipeline 已传入灰度图则复用，否则自行解码（ 避免重复解码）
        if gray is None:
            gray = _read_gray(image["path"])
        orig_bytes = _encode_png(gray)
        graph_bytes = _annotate_png(gray, defects)
        pdf_path = self._out / f"{image_id}.pdf"
        # 追溯二维码（G01）：编码 report_id + 指纹前缀；生成失败 fail-soft
        # 不阻断出片（二维码是增强能力，报告是主业务）。
        rid = (report or {}).get("report_id") or image_id
        qr_bytes = qr_png_bytes(trace_code(rid, fingerprint))
        # 先渲染 reportlab PDF 到中间文件，再转写为 PDF/A-1b
        rl_path = pdf_path.with_suffix(".rl.pdf")
        _render(rl_path, content, graph_bytes, orig_bytes, self._font, tpl, qr_bytes)
        try:
            postprocess_to_pdfa(rl_path, pdf_path)
        finally:
            # 清理失败（只读/占用/回收站不可用）不应连带报告生成一起失败
            try:
                rl_path.unlink(missing_ok=True)
            except OSError:
                pass
        # 指纹写库：报告行存在则回填，缺行/写入失败不阻断出片。
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
        # SM2 签名落 sidecar（C-03）：在 SHA-256 指纹之外叠加国密签名；
        # 未配置密钥时降级为仅指纹（写 sidecar 返回 False，不阻断出片）。
        write_signature_sidecar(pdf_path, fingerprint)
        return str(pdf_path)


def report_fingerprint(image: dict, defects: list[dict], report: dict | None) -> str:
    """报告内容指纹：关键字段 canonical JSON → SHA-256 hex。

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
        "secret_level": int(image.get("secret_level") or 0),  # C-10：密级纳入防篡改指纹
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


# ---------------------------------------------------------------------------
# SM2 报告签名（C-03）：对 SHA-256 指纹做 SM3withSM2 签名，值落 sidecar。
# ---------------------------------------------------------------------------

_SIDE_SUFFIX = ".sig"  # 签名 sidecar 后缀（<pdf>.sig，JSON）


def report_signature(fingerprint: str) -> dict[str, str] | None:
    """对报告指纹做 SM2 数字签名（SM3withSM2），返回 sidecar 元数据。

    签名对象是指纹字符串本身，指纹覆盖报告全部关键字段（见
    report_fingerprint），SM2 签名由此间接覆盖全内容。未配置
    SCAN_CRYPTO_KEY（或 provider 初始化失败）时返回 None：签名降级为仅
    指纹，不阻断出片——与静态加密的降级策略一致。
    """
    try:
        from backend.infra.crypto import CryptoKeyError, get_provider

        try:
            provider = get_provider()
        except CryptoKeyError as exc:
            _LOG.warning("SM2 签名未生效（%s）：报告仅落 SHA-256 指纹", exc)
            return None
    except ImportError as exc:  # pragma: no cover - gmssl 为硬依赖，防御性降级
        _LOG.warning("国密库不可用（%s）：报告仅落 SHA-256 指纹", exc)
        return None
    return {
        "algo": "SM2",
        "hash_algo": "SHA-256",
        "fingerprint": fingerprint,
        "signature": provider.sign(fingerprint.encode("utf-8")),
        "public_key": provider.public_key_hex,
        "signed_at": datetime.now(UTC).replace(tzinfo=None).isoformat(timespec="seconds"),
    }


def signature_sidecar_path(pdf_path: str | Path) -> Path:
    """签名 sidecar 文件路径（<pdf>.sig）。"""
    return Path(str(pdf_path) + _SIDE_SUFFIX)


def write_signature_sidecar(pdf_path: str | Path, fingerprint: str) -> bool:
    """SM2 签名落 sidecar 文件（<pdf>.sig，JSON）。

    指纹本体已入 reports.report_hash，表结构不变，故签名值落文件；sidecar
    内附带公钥，验签方无需持有签名私钥即可校验。返回是否写入成功。
    """
    meta = report_signature(fingerprint)
    if meta is None:
        return False
    meta["report_pdf"] = Path(pdf_path).name
    try:
        signature_sidecar_path(pdf_path).write_text(
            json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        return True
    except OSError as exc:  # 落盘失败不阻断出片（签名尽力而为）
        _LOG.warning("SM2 签名 sidecar 写入失败 %s: %s", pdf_path, exc)
        return False


def read_signature_sidecar(pdf_path: str | Path) -> dict | None:
    """读取签名 sidecar；文件不存在/损坏/格式不符返回 None（旧报告无签名）。"""
    try:
        data = json.loads(signature_sidecar_path(pdf_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    sig, pub = data.get("signature"), data.get("public_key")
    if not isinstance(sig, str) or not isinstance(pub, str):
        return None
    return data


def _report_disclaimer(standard_id: str) -> str:
    """按报告所用标准表生成免责声明（工业过渡路径，）。

    与后端判定器共用 disclaimer_for，保证 API / PDF / 报告一致。
    """
    from backend.domain.standards.tables.loader import disclaimer_for, load_standard_tables

    if standard_id:
        try:
            tables = load_standard_tables(standard_id)
            return disclaimer_for(tables)
        except Exception as exc:  # noqa: BLE001 - 表缺失/解析失败 → 回退默认，不阻断出片
            # 表损坏恰是最需要强声明（authorized 表不可读）的场景：声明在
            # 正式交付的合规 PDF 里无声消失不可接受，至少留痕告警。
            _LOG.warning(
                "报告免责声明加载失败 standard=%s（PDF 将无声明输出）: %s", standard_id, exc
            )
    return ""


def _register_font() -> str:
    """注册第一个可用的中文字体，返回字体名（找不到则返回 Helvetica 并降级）。

    .ttc 为字体集合，取第一个子字体（simsun.ttc[0] 即宋体）。
    """
    for path in _font_candidates():
        if path.exists():
            name = f"CN-{path.stem}"
            try:
                if path.suffix.lower() == ".ttc":
                    pdfmetrics.registerFont(TTFont(name, str(path), subfontIndex=0))
                else:
                    pdfmetrics.registerFont(TTFont(name, str(path)))
                return name
            except (OSError, ValueError, TypeError):
                pass  # 字体损坏/不兼容 → 尝试下一个候选
    return "Helvetica"


def _read_gray(image_path: str) -> np.ndarray | None:
    """读取灰度图（含密文副本解密）。实现已下移 image_loader.read_gray，
    本名保留为兼容别名（既有测试/内部调用点使用）。"""
    from backend.infra.image_loader import read_gray

    return read_gray(image_path)


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


# ---------------------------------------------------------------------------
# 正式 RT 报告版式（1:1 对齐传统《射线检测报告》样张）：
# 第1页 《射线检测报告》：大标题 + NO: 报告编号 + 全字段汇总表（委托单位/
#   工程名称/工件概况/技术要求/检测器材及工艺参数/检测情况/检测结论及说明/
#   检测·审核签字 + 检测单位检测专用章），软件已知字段自动填入，未知留空
#   供机构手工补填；
# 第2页起 《射线检测底片评定表》：序号/焊缝管口编号/片号/黑度/识别丝号/
#   缺陷性质与缺陷尺寸/缺陷部位/评定等级/备注，一行一缺陷，同焊缝同片号
#   纵向合并，空行补满整页（与样张一致）；
# 末页 附图：射线检测位置示意图（标注影像 + 送检底片）+ 判定依据 + 防伪指纹。
# 每页页脚『共 N 页 第 M 页』（数字带下划线）；密级横标覆盖全部页面（C-10）。
# 字体：宋体（样张同款），正文 12pt、大标题 18pt；列宽行高取自样张 docx 网格。
# ---------------------------------------------------------------------------

_TWIP_PT = 0.05  # 1 twip = 1/20 pt
_PAGE_MARGIN_X = 0.9 * cm  # 样张汇总表宽 19.2cm 居中（略宽于常规正文边距）
_PAGE_MARGIN_TOP = 16 * mm
_PAGE_MARGIN_BOTTOM = 14 * mm
_CONTENT_W = _PAGE_W - 2 * _PAGE_MARGIN_X

# 《射线检测报告》汇总表：21 列网格宽（twips，逐列取自样张 w:tblGrid）
_SUMMARY_GRID = (
    810,
    465,
    264,
    1034,
    582,
    930,
    175,
    741,
    159,
    61,
    599,
    159,
    657,
    800,
    58,
    152,
    498,
    132,
    1057,
    705,
    872,
)
_SUMMARY_ROWS = 21  # 不含可选军代表/见证行
_CONCLUSION_ROW_H = 4.9 * cm  # 样张结论行 2804 twips ≈ 4.9cm

# 《射线检测底片评定表》：9 列网格宽（twips，逐列取自样张）
_EVAL_GRID = (465, 1785, 751, 1155, 780, 2400, 1590, 795, 1268)
_EVAL_ROWS_PER_PAGE = 30  # 每页数据行（含空行补位），与样张一致

_EVAL_HEADER = (
    "序号",
    "焊缝/管口\n编     号",
    "片号",
    "黑度",
    "识别\n丝号",
    "缺陷性质与缺陷尺寸",
    "缺陷\n部位",
    "评定\n等级",
    "备\n注",
)

# 缺陷代号（样张结论栏第 3 条）：A裂纹 B未焊透 C未熔合 D圆形 E条形 F内凹 G咬边
_CLASS_CODE = {0: "D", 1: "D", 2: "B", 3: "C", 4: "A", 5: "G", 6: "F"}
_NAME_CODE = {
    "裂纹": "A",
    "未焊透": "B",
    "未熔合": "C",
    "气孔": "D",
    "夹渣": "D",
    "内凹": "F",
    "咬边": "G",
}
_IQI_TYPE_CN = {"wire": "丝型", "hole": "孔型"}

_RT_QUAL = "RTⅡ"  # 评片/审核人员资格级别（版式占位）
_ROMAN = {"I": "Ⅰ", "II": "Ⅱ", "III": "Ⅲ", "IV": "Ⅳ"}
_SECRET_LEVEL_NAMES = {0: "非密", 1: "内部", 2: "秘密", 3: "机密"}


def classification_label(secret_level: int) -> str:
    """密级数值 → 页面横标文本（C-10）；非密（0）返回空串（不绘制横标）。"""
    level = int(secret_level or 0)
    return _SECRET_LEVEL_NAMES.get(level, "") if level > 0 else ""


class _ReportCanvas(pdfcanvas.Canvas):
    """两遍渲染页脚：『共 N 页 第 M 页』在 save 时才知总页数，先快照再统一补画。

    每页（首页/评定表/附图页）页脚居中绘制共/第页码，数字带下划线（样张格式）；
    密级横标（C-10）secret_level>0 时绘制于每页顶部，页脚左下角另附定密依据。
    """

    def __init__(
        self,
        *args,
        font: str = "Helvetica",
        classification: str = "",
        basis: str = "",
        qr_bytes: bytes | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._chrome_font = font
        self._classification = classification  # 如 "秘密"（空串=非密不绘制）
        self._basis = basis
        # 追溯二维码（G01）：PNG bytes，None=不绘制；随页脚每页绘制
        self._qr_image = ImageReader(io.BytesIO(qr_bytes)) if qr_bytes else None
        self._saved_states: list[dict] = []

    def showPage(self) -> None:
        self._saved_states.append(dict(self.__dict__))
        self._startPage()  # type: ignore[attr-defined]  # reportlab Canvas 私有 API，stub 未声明

    def save(self) -> None:
        total = len(self._saved_states)
        for state in self._saved_states:
            self.__dict__.update(state)
            self._draw_chrome(total)
            pdfcanvas.Canvas.showPage(self)
        pdfcanvas.Canvas.save(self)

    def _draw_chrome(self, total: int) -> None:
        page = self._pageNumber  # type: ignore[attr-defined]  # reportlab Canvas 私有属性
        if self._classification:
            self.saveState()
            self.setFont(self._chrome_font, 12)
            self.setFillColor(colors.red)
            self.drawCentredString(_PAGE_W / 2.0, _PAGE_H - 24, f"密级：{self._classification}")
            self.restoreState()
        self._draw_page_footer(page, total)
        if self._classification and self._basis:
            self.saveState()
            self.setFont(self._chrome_font, 7)
            self.drawString(_PAGE_MARGIN_X, 8, f"定密依据：{self._basis[:60]}")
            self.restoreState()

    def _draw_page_footer(self, page: int, total: int) -> None:
        """『共 N 页　第 M 页』：数字加下划线，居中（样张页脚格式）；右下角附追溯二维码。"""
        self.saveState()
        self.setFont(self._chrome_font, 11)
        parts = (
            ("共 ", False),
            (str(total), True),
            (" 页　　第 ", False),
            (str(page), True),
            (" 页", False),
        )
        widths = [self.stringWidth(text, self._chrome_font, 11) for text, _ in parts]
        x = (_PAGE_W - sum(widths)) / 2.0
        y = 22
        for (text, underline), w in zip(parts, widths):
            self.drawString(x, y, text)
            if underline:
                self.line(x, y - 2.5, x + w, y - 2.5)
            x += w
        self.restoreState()
        if self._qr_image is not None:
            # 每页右下角二维码（~10.6mm）：扫码定位档案 + 指纹前缀人工比对
            qr_size = 30
            self.saveState()
            self.drawImage(
                self._qr_image,
                _PAGE_W - _PAGE_MARGIN_X - qr_size,
                8,
                qr_size,
                qr_size,
            )
            self.setFont(self._chrome_font, 5)
            self.drawCentredString(_PAGE_W - _PAGE_MARGIN_X - qr_size / 2, 3, "扫码追溯")
            self.restoreState()


def _render(
    pdf_path: Path,
    content: object,
    graph_bytes: bytes | None,
    orig_bytes: bytes | None,
    font: str,
    tpl: ReportTemplate,
    qr_bytes: bytes | None = None,
) -> None:
    """渲染报告 PDF（样张同版式：射线检测报告 + 底片评定表 + 附图）。"""
    c = _cast(content)
    styles = _make_styles(font)
    doc = BaseDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=_PAGE_MARGIN_X,
        rightMargin=_PAGE_MARGIN_X,
        topMargin=_PAGE_MARGIN_TOP,
        bottomMargin=_PAGE_MARGIN_BOTTOM,
        title=f"射线检测报告 {c.image_id}",
        author=tpl.author,
    )
    pad = {"leftPadding": 0, "rightPadding": 0, "topPadding": 0, "bottomPadding": 0}
    doc.addPageTemplates(
        [
            PageTemplate(
                id="body",
                frames=[
                    Frame(
                        _PAGE_MARGIN_X,
                        _PAGE_MARGIN_BOTTOM,
                        _CONTENT_W,
                        _PAGE_H - _PAGE_MARGIN_TOP - _PAGE_MARGIN_BOTTOM,
                        id="full",
                        **pad,
                    )
                ],
            )
        ]
    )

    flow: list[Flowable] = []
    flow.extend(_summary_flow(c, styles))
    flow.append(PageBreak())
    flow.extend(_film_eval_flow(c, styles))
    flow.extend(_attachment_flow(c, graph_bytes, orig_bytes, styles))

    doc.build(
        flow,
        canvasmaker=lambda *a, **k: _ReportCanvas(
            *a,
            font=font,
            classification=classification_label(getattr(c, "secret_level", 0)),
            basis=getattr(c, "classification_basis", "") or "",
            qr_bytes=qr_bytes,
            **k,
        ),
    )


def _summary_flow(c, styles: dict[str, ParagraphStyle]) -> list[Flowable]:
    """第1页《射线检测报告》：大标题 + NO: 编号行 + 汇总表。"""
    return [
        Paragraph("射 线 检 测 报 告", styles["title"]),
        Spacer(1, 3 * mm),
        Paragraph(f"NO:{c.report_id or c.image_id}", styles["no_line"]),
        Spacer(1, 1.5 * mm),
        _summary_table(c, styles),
    ]


def _summary_table(c, styles: dict[str, ParagraphStyle]) -> Table:
    """《射线检测报告》汇总表：21 列合并网格，行列结构逐格对齐样张。

    软件已知字段自动填入（工件/标准/黑度/丝号/级别/缺陷统计/签字日期），
    未知工艺字段留空供检测机构打印后手工补填（样张本身也留空委托单位等）。
    """
    iqi = c.iqi_detail or {}
    std = c.standard_ref or ""
    date_dot = _dot_date(c.generated_at)
    spec = f"{c.base_metal_thickness_mm:g}mm" if c.base_metal_thickness_mm else ""
    density = f"{c.density:.1f}" if c.density is not None else ""
    wire_req = f"{iqi.get('required')}#" if iqi.get("required") else ""
    iqi_model = _IQI_TYPE_CN.get(str(iqi.get("type") or ""), "")
    part = c.workpiece_no or ""
    meta = dict(getattr(c, "report_meta", None) or {})

    def m(key: str) -> str:
        return meta.get(key, "")

    # 最终评定结果统计（单张底片：评为哪级记 1 张、总计 1；无级别以 / 占位）
    counts = {"I": "/", "II": "/", "III": "/", "IV": "/"}
    total_films = "/"
    lv = str(c.joint_level or "").strip().upper()
    if lv in counts:
        counts[lv] = "1"
        total_films = "1"

    witness = getattr(c, "witness", None)
    nrows = _SUMMARY_ROWS + (1 if witness else 0)
    grid: list[list] = [[""] * 21 for _ in range(nrows)]
    spans: list[tuple] = []

    def put(r: int, col: int, text, cs: int = 1, rs: int = 1, style: str = "cell") -> None:
        s = str(text)
        if s:
            # 长值（长标准号/编号）缩号排布，避免在窄列内截断换行
            use = "cell_sm" if style == "cell" and len(s) > 14 else style
            grid[r][col] = Paragraph(s, styles[use])
        if cs > 1 or rs > 1:
            spans.append(("SPAN", (col, r), (col + cs - 1, r + rs - 1)))

    # 表头两行：委托单位/工程名称 × 工程类别/检测时机 × 检测地址
    put(0, 0, "委托单位", cs=3)
    put(0, 3, m("client_unit"), cs=7)
    put(0, 10, "工程类别/检测时机", cs=7)
    put(0, 17, m("project_category"), cs=4)
    put(1, 0, "工程名称", cs=3)
    put(1, 3, m("project_name"), cs=7)
    put(1, 10, "检测地址", cs=7)
    put(1, 17, m("test_address"), cs=4)

    # 工件概况（行 2-4）
    put(2, 0, "工件<br/>概况", rs=3)
    put(2, 1, "工件名称", cs=3)
    put(2, 4, part, cs=3)
    put(2, 7, "材    质", cs=4)
    put(2, 11, m("material"), cs=5)
    put(2, 16, "坡口形式", cs=3)
    put(2, 19, m("groove_type"), cs=2)
    put(3, 1, "工件编号", cs=3)
    put(3, 4, m("part_no"), cs=3)
    put(3, 7, "规    格", cs=4)
    put(3, 11, spec, cs=5)
    put(3, 16, "表面状况", cs=3)
    put(3, 19, m("surface_status"), cs=2)
    put(4, 1, "检测部位", cs=3)
    put(4, 4, "焊接接头", cs=3)
    put(4, 7, "焊接方式", cs=4)
    put(4, 11, m("weld_process"), cs=5)
    put(4, 16, "热处理状态", cs=3)
    put(4, 19, m("heat_treatment"), cs=2)

    # 技术要求（行 5-7）
    put(5, 0, "技术<br/>要求", rs=3)
    put(5, 1, "验收标准", cs=3)
    put(5, 4, std, cs=3)
    put(5, 7, "检测标准", cs=4)
    put(5, 11, std, cs=5)
    put(5, 16, "检测比例", cs=3)
    put(5, 19, "100%", cs=2)
    put(6, 1, "检测技术等级", cs=3)
    put(6, 4, m("tech_level"), cs=3)
    # 合格级别=验收要求级别（用户在表单提供；未提供留空）；评定级别见"最终评定结果"
    put(6, 7, "合格级别", cs=4)
    put(6, 11, m("accept_level"), cs=5)
    put(6, 16, "原始记录编号", cs=3)
    put(6, 19, m("record_no"), cs=2)
    put(7, 1, "黑度范围", cs=3)
    put(7, 4, density, cs=3)
    put(7, 7, "应识别丝号", cs=4)
    put(7, 11, wire_req, cs=5)
    put(7, 16, "散射线控制", cs=3)
    put(7, 19, m("scatter_control"), cs=2)

    # 检测器材及工艺参数（行 8-14，共 7 行 × 3 组）
    put(8, 0, "检测器材<br/>及工艺<br/>参数", rs=7)
    put(8, 1, "源种类", cs=3)
    put(8, 4, m("source_kind"), cs=3)
    put(8, 7, "设备型号/编号", cs=4)
    put(8, 11, m("device_no"), cs=5)
    put(8, 16, "焦点尺寸", cs=3)
    put(8, 19, m("focus_size"), cs=2)
    put(9, 1, "胶片型号", cs=3)
    put(9, 4, m("film_model"), cs=3)
    put(9, 7, "胶片规格", cs=4)
    put(9, 11, m("film_size"), cs=5)
    put(9, 16, "胶片分类等级", cs=3)
    put(9, 19, m("film_class"), cs=2)
    put(10, 1, "增感方式", cs=3)
    put(10, 4, m("screen_way"), cs=3)
    put(10, 7, "像质计型号", cs=4)
    put(10, 11, iqi_model, cs=5)
    put(10, 16, "像质计摆放", cs=3)
    put(10, 19, m("iqi_position"), cs=2)
    put(11, 1, "前屏/后屏", cs=3)
    put(11, 4, m("screens"), cs=3)
    put(11, 7, "透照方式", cs=4)
    put(11, 11, m("technique"), cs=5)
    put(11, 16, "透照厚度", cs=3)
    put(11, 19, spec, cs=2)
    put(12, 1, "F（焦距）", cs=3)
    put(12, 4, m("focus_distance"), cs=3)
    put(12, 7, "f（源至工件）", cs=4)
    put(12, 11, m("source_distance"), cs=5)
    put(12, 16, "冲洗条件", cs=3)
    put(12, 19, m("develop_method"), cs=2)
    put(13, 1, "b（工件至胶片）", cs=3)
    put(13, 4, m("film_distance"), cs=3)
    put(13, 7, "管电压", cs=4)
    put(13, 11, m("tube_voltage"), cs=5)
    put(13, 16, "显影液配方", cs=3)
    put(13, 19, m("developer"), cs=2)
    put(14, 1, "管电流", cs=3)
    put(14, 4, m("tube_current"), cs=3)
    put(14, 7, "曝光时间", cs=4)
    put(14, 11, m("exposure_time"), cs=5)
    put(14, 16, "洗片温度", cs=3)
    put(14, 19, m("develop_temp"), cs=2)

    # 检测情况（行 15-17）：焊缝统计 + 最终评定结果分级张数
    put(15, 0, "检测<br/>情况", rs=3)
    put(15, 1, "焊缝总数", cs=3)
    put(15, 4, "1道", cs=3)
    put(15, 7, "最终评定结果", rs=3)
    put(15, 8, "Ｉ级（张）", cs=3, rs=2)
    put(15, 11, "Ⅱ级（张）", cs=2, rs=2)
    put(15, 13, "Ⅲ级（张）", rs=2)
    put(15, 14, "Ⅳ级（张）", cs=4, rs=2)
    put(15, 18, "总计（张）", rs=2)
    put(15, 19, "返修数量（张）", rs=2)
    put(15, 20, "最高返修次数（次）", rs=2)
    put(16, 1, "检测数量", cs=3)
    put(16, 4, "1道", cs=3)
    put(17, 1, "检测比例", cs=3)
    put(17, 4, "100%", cs=3)
    put(17, 8, counts["I"], cs=3)
    put(17, 11, counts["II"], cs=2)
    put(17, 13, counts["III"])
    put(17, 14, counts["IV"], cs=4)
    put(17, 18, total_films)
    put(17, 19, "/")
    put(17, 20, "/")

    # 检测结论及说明（行 18，整行合并、固定高）
    grid[18][0] = _conclusion_flowables(c, styles)
    spans.append(("SPAN", (0, 18), (20, 18)))

    # 签字栏（行 19-20 [+21]）+ 检测单位检测专用章（右侧整块合并）
    grid[19][15] = _seal_flowables(c, styles)
    spans.append(("SPAN", (15, 19), (20, nrows - 1)))
    put(19, 0, "检 测", cs=2)
    put(19, 2, c.signer or "", cs=3)
    put(19, 5, "资 格")
    put(19, 6, _RT_QUAL, cs=3)
    put(19, 9, "日 期", cs=3)
    put(19, 12, date_dot, cs=3)
    put(20, 0, "审 核", cs=2)
    put(20, 2, "", cs=3)
    put(20, 5, "资 格")
    put(20, 6, _RT_QUAL, cs=3)
    put(20, 9, "日 期", cs=3)
    put(20, 12, date_dot, cs=3)
    if witness:
        put(21, 0, "军代表/见证", cs=2)
        put(21, 2, str(witness), cs=3)
        put(21, 5, "资 格")
        put(21, 6, _RT_QUAL, cs=3)
        put(21, 9, "日 期", cs=3)
        put(21, 12, date_dot, cs=3)

    raw_w = sum(_SUMMARY_GRID) * _TWIP_PT
    scale = _CONTENT_W / raw_w
    row_heights: list = [None] * nrows
    row_heights[18] = _CONCLUSION_ROW_H
    t = Table(
        grid,
        colWidths=[tw * _TWIP_PT * scale for tw in _SUMMARY_GRID],
        rowHeights=row_heights,
    )
    cmds = [
        ("GRID", (0, 0), (-1, -1), 0.75, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 1),
        ("RIGHTPADDING", (0, 0), (-1, -1), 1),
        ("VALIGN", (0, 18), (-1, 18), "TOP"),
    ]
    cmds.extend(spans)
    t.setStyle(TableStyle(cmds))
    return t


def _conclusion_flowables(c, styles: dict[str, ParagraphStyle]) -> list:
    """检测结论及说明（样张文案 + AI 辅助声明第 4 条）。

    合格判定：表单提供合格级别（验收要求）时按级别序比较（Ⅰ<Ⅱ<Ⅲ<Ⅳ）；
    未提供时按 NB/T47013 惯例（Ⅰ/Ⅱ 合格，Ⅲ/Ⅳ 不合格）。
    """
    meta = dict(getattr(c, "report_meta", None) or {})
    part = c.workpiece_no or f"影像 {c.image_id[-8:].upper()}"
    std = c.standard_ref or "验收标准"
    if not c.evaluable:
        first = "1、影像质量校验未通过（IQI/黑度不达标），本片不可评片，需人工复核处理。"
    elif c.joint_level:
        grade = _grade_rank(c.joint_level)
        accept = _grade_rank(meta.get("accept_level"))
        if grade is not None and accept is not None:
            ok = grade <= accept
            first = (
                f"1、本工件（{part}）焊缝质量经检测，依据{std}评为{_roman_level(c.joint_level)}，"
                f"{'满足' if ok else '未满足'}验收要求（合格级别{_roman_level(meta.get('accept_level'))}），"
                f"结果{'合格' if ok else '不合格'}。"
            )
        else:
            verdict = "合格" if grade is not None and grade <= 2 else "不合格"
            first = (
                f"1、本工件（{part}）焊缝质量经检测，依据{std} "
                f"评为{_roman_level(c.joint_level)}，结果{verdict}。"
            )
    else:
        first = "1、本片暂无法自动评级（置信度不足或未标定），需人工评定。"
    if c.need_review and c.joint_level and c.evaluable:
        # 有级别但标记复核（翻拍降级/初评对评分歧等）：结论不得以正式口吻
        # 收尾，否则与判定依据里的降级声明互相矛盾。
        first += "（本片尚须经人工复核确认，暂不作为正式评定结论）"
    return [
        Paragraph("检测结论及说明：", styles["cell_left"]),
        Spacer(1, 8),
        Paragraph(first, styles["cell_left"]),
        Paragraph(
            "2、检测位置，返修部位，底片评定情况见射线检测位置示意图和底片评定表。",
            styles["cell_left"],
        ),
        Paragraph("3、缺陷代号", styles["cell_left"]),
        Paragraph(
            "A裂纹、B未焊透、C未熔合、D圆形缺陷（气孔、夹渣、夹钨、夹铜等）、"
            "E条形缺陷、F内凹、G咬边",
            styles["cell_left"],
        ),
        Paragraph(
            "4、本报告为AI辅助评定，级别须经责任工程师复核签核后方可采信。", styles["cell_left"]
        ),
    ]


def _seal_flowables(c, styles: dict[str, ParagraphStyle]) -> list:
    """检测单位检测专用章占位区（章 + 日期）。"""
    out: list = [Paragraph("检测单位检测专用章", styles["seal"])]
    date_cn = _cn_date(c.generated_at)
    if date_cn:
        out.append(Spacer(1, 14))
        out.append(Paragraph(f"日期：{date_cn}", styles["cell"]))
    return out


def _film_eval_flow(c, styles: dict[str, ParagraphStyle]) -> list[Flowable]:
    """《射线检测底片评定表》：一行一缺陷，同焊缝/同片号合并，空行补满页。"""
    vals = _eval_row_values(c)
    flow: list[Flowable] = []
    pages = max(1, math.ceil(len(vals) / _EVAL_ROWS_PER_PAGE))
    for p in range(pages):
        chunk = vals[p * _EVAL_ROWS_PER_PAGE : (p + 1) * _EVAL_ROWS_PER_PAGE]
        if p:
            flow.append(PageBreak())
        flow.append(Paragraph("射线检测底片评定表", styles["title"]))
        flow.append(Spacer(1, 2 * mm))
        flow.append(Paragraph(f"NO:{c.report_id or c.image_id}", styles["no_line"]))
        flow.append(Spacer(1, 1.5 * mm))
        flow.append(_eval_table(styles, chunk))
    return flow


def _eval_table(styles: dict[str, ParagraphStyle], vals: list[list[str]]) -> Table:
    """评定表分页块：vals 为本页 8 列数据行，纵向合并同焊缝/同片号单元格。"""
    data: list[list] = [
        [Paragraph(h.replace("\n", "<br/>"), styles["ehead"]) for h in _EVAL_HEADER]
    ]
    for row in vals:
        cells: list = [""]
        for ci, v in enumerate(row):
            if not v:
                cells.append("")
                continue
            # 片号等窄列长串缩号，保证单行不折行
            style = "cell_film" if ci == 1 and len(v) > 5 else "cell"
            cells.append(Paragraph(v, styles[style]))
        data.append(cells)
    for _ in range(_EVAL_ROWS_PER_PAGE - len(vals)):
        data.append([""] * 9)

    spans: list[tuple] = []
    i, seq = 0, 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1][0] == vals[i][0]:
            j += 1
        seq += 1
        data[i + 1][0] = Paragraph(str(seq), styles["cell"])
        spans.append(("SPAN", (0, i + 1), (0, j + 1)))
        spans.append(("SPAN", (1, i + 1), (1, j + 1)))
        k = i
        while k <= j:
            m = k
            while m + 1 <= j and vals[m + 1][1] == vals[k][1]:
                m += 1
            if m > k:
                spans.append(("SPAN", (2, k + 1), (2, m + 1)))
            k = m + 1
        i = j + 1

    raw_w = sum(_EVAL_GRID) * _TWIP_PT
    scale = _CONTENT_W / raw_w
    t = Table(
        data,
        colWidths=[tw * _TWIP_PT * scale for tw in _EVAL_GRID],
        repeatRows=1,
    )
    cmds = [
        ("GRID", (0, 0), (-1, -1), 0.75, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 1),
        ("RIGHTPADDING", (0, 0), (-1, -1), 1),
    ]
    cmds.extend(spans)
    t.setStyle(TableStyle(cmds))
    return t


def _eval_row_values(c) -> list[list[str]]:
    """评定表数据行（8 列：焊缝/片号/黑度/丝号/缺陷代号/部位/等级/备注）。"""
    weld = c.weld_no or "—"
    # 片号（G05）：优先印字抽取的独立片号；未识别到时回退影像短号口径
    film = (c.film_no or c.image_id[-6:]).upper()
    dens = f"{c.density:.1f}" if c.density is not None else ""
    wire = str((c.iqi_detail or {}).get("achieved") or "")
    remark = f"{c.base_metal_thickness_mm:g}mm" if c.base_metal_thickness_mm else ""
    rows = [
        [
            weld,
            film,
            dens,
            wire,
            _defect_code_txt(d),
            _defect_pos_txt(d),
            _roman_plain(d.get("joint_level")),
            remark,
        ]
        for d in c.defects
    ]
    if not rows:
        rows.append([weld, film, dens, wire, "未检出缺陷", "", _roman_plain(c.joint_level), remark])
    return rows


def _defect_code_txt(d: dict) -> str:
    """缺陷性质与尺寸（样张代号记法）：D:Φ1.2（圆形）/ E:L=6.5（条形）等。"""
    name = str(d.get("class_name") or "")
    cid = d.get("class_id")
    code = _NAME_CODE.get(name) or (_CLASS_CODE.get(cid, "D") if isinstance(cid, int) else "D")
    length, width = d.get("length_mm"), d.get("width_mm")
    if code == "D" and str(d.get("shape") or "") == "linear":
        code = "E"  # 条形缺陷（气孔/夹渣呈长条状时按条形代号）
    if code == "D":
        try:
            dia = max(float(length or 0), float(width or 0))
        except (TypeError, ValueError):
            dia = 0.0
        return f"D:Φ{dia:.1f}" if dia > 0 else "D"
    if length is not None:
        try:
            return f"{code}:L={float(length):.1f}"
        except (TypeError, ValueError):
            return code
    return code


def _defect_pos_txt(d: dict) -> str:
    """缺陷部位：优先标定坐标（mm），未标定时给像素中心。"""
    px, py = d.get("position_x"), d.get("position_y")
    if px is not None and py is not None:
        try:
            return f"{float(px):.0f},{float(py):.0f}"
        except (TypeError, ValueError):
            pass
    bb = d.get("bbox_px")
    if bb and len(bb) >= 4:
        try:
            cx = float(bb[0]) + float(bb[2]) / 2
            cy = float(bb[1]) + float(bb[3]) / 2
            return f"{cx:.0f},{cy:.0f}px"
        except (TypeError, ValueError):
            return ""
    return ""


def _attachment_flow(
    c, graph_bytes: bytes | None, orig_bytes: bytes | None, styles: dict[str, ParagraphStyle]
) -> list[Flowable]:
    """附图页：射线检测位置示意图（标注影像 + 送检底片）+ 依据/声明/指纹。"""
    out: list[Flowable] = []
    if graph_bytes or orig_bytes:
        out.append(PageBreak())
        out.append(Paragraph("射线检测位置示意图（附图）", styles["att_title"]))
        out.append(Spacer(1, 3 * mm))
        if graph_bytes:
            out.append(_scaled_image(graph_bytes, _CONTENT_W, 100 * mm))
            out.append(Paragraph("检测标注影像（缺陷位置示意）", styles["caption"]))
            out.append(Spacer(1, 3 * mm))
        if orig_bytes:
            out.append(_scaled_image(orig_bytes, _CONTENT_W, 70 * mm))
            out.append(Paragraph("送检底片（未标注）", styles["caption"]))
        out.append(Spacer(1, 3 * mm))
    else:
        out.append(Spacer(1, 6 * mm))
    if c.basis:
        out.append(Paragraph("判定依据条款", styles["att_section"]))
        for i, b in enumerate(c.basis, 1):
            out.append(Paragraph(f"{i}. {b}", styles["fine"]))
        out.append(Spacer(1, 2 * mm))
    if c.disclaimer:
        out.append(Paragraph(c.disclaimer, styles["fine"]))
        out.append(Spacer(1, 2 * mm))
    if c.fingerprint:
        out.append(
            Paragraph(f"数字指纹：SHA-256:{c.fingerprint}（报告内容防篡改校验）", styles["fine"])
        )
    return out


def _roman_level(level: object) -> str:
    """级别（I/II/III/IV）→ 罗马数字带级（Ⅰ级…）；空值返回 '—'。"""
    lv = str(level or "").strip().upper()
    return f"{_ROMAN.get(lv, lv)}级" if lv else "—"


def _roman_plain(level: object) -> str:
    """级别（I/II/III/IV）→ 裸罗马数字（Ⅱ，样张评定等级栏格式）；空返回 ''。"""
    lv = str(level or "").strip().upper()
    return _ROMAN.get(lv, lv) if lv else ""


def _grade_rank(level: object) -> int | None:
    """级别（I/II/III/IV，兼容全角罗马数字）→ 序数 1..4（合格级别比较用）。"""
    lv = str(level or "").strip().upper()
    lv = {"Ⅰ": "I", "Ⅱ": "II", "Ⅲ": "III", "Ⅳ": "IV"}.get(lv, lv)
    rank = {"I": 1, "II": 2, "III": 3, "IV": 4}.get(lv)
    return int(rank) if rank else None


def _dot_date(value: str) -> str:
    """ISO/时间戳 → 'YYYY.M.D'（样张签字栏日期格式；解析失败返回 ''）。"""
    s = str(value or "").strip()
    if not s:
        return ""
    for candidate in (s[:19].replace(" ", "T"), s[:10]):
        try:
            dt = datetime.fromisoformat(candidate)
        except ValueError:
            continue
        return f"{dt.year}.{dt.month}.{dt.day}"
    try:
        dt = datetime.fromtimestamp(float(s), tz=UTC)
    except (ValueError, OSError):
        return ""
    return f"{dt.year}.{dt.month}.{dt.day}"


def _cn_date(value: str) -> str:
    """ISO/时间戳 → 'YYYY年MM月DD日'（解析失败返回原串前 10 位）。"""
    s = str(value or "").strip()
    if not s:
        return ""
    for candidate in (s[:19].replace(" ", "T"), s[:10]):
        try:
            return datetime.fromisoformat(candidate).strftime("%Y年%m月%d日")
        except ValueError:
            continue
    try:
        return datetime.fromtimestamp(float(s), tz=UTC).strftime("%Y年%m月%d日")
    except (ValueError, OSError):
        return s[:10]


def _make_styles(font: str) -> dict[str, ParagraphStyle]:
    """样张版式段落样式（宋体：正文 12pt，大标题 18pt；CJK 换行）。"""
    return {
        "title": ParagraphStyle(
            "rt_title",
            fontName=font,
            fontSize=18,
            leading=26,
            alignment=TA_CENTER,
            wordWrap="CJK",
        ),
        "att_title": ParagraphStyle(
            "rt_att",
            fontName=font,
            fontSize=14,
            leading=20,
            alignment=TA_CENTER,
            wordWrap="CJK",
        ),
        "att_section": ParagraphStyle(
            "rt_secs", fontName=font, fontSize=12, leading=17, wordWrap="CJK"
        ),
        "no_line": ParagraphStyle(
            "rt_no",
            fontName=font,
            fontSize=12,
            leading=16,
            alignment=TA_RIGHT,
            wordWrap="CJK",
        ),
        "cell": ParagraphStyle(
            "rt_cell",
            fontName=font,
            fontSize=12,
            leading=15,
            alignment=TA_CENTER,
            wordWrap="CJK",
        ),
        "cell_left": ParagraphStyle(
            "rt_cl",
            fontName=font,
            fontSize=12,
            leading=17,
            alignment=TA_LEFT,
            wordWrap="CJK",
        ),
        "cell_sm": ParagraphStyle(
            "rt_csm",
            fontName=font,
            fontSize=8,
            leading=11,
            alignment=TA_CENTER,
            wordWrap="CJK",
        ),
        "cell_film": ParagraphStyle(
            "rt_cfilm",
            fontName=font,
            fontSize=9,
            leading=12,
            alignment=TA_CENTER,
            wordWrap="CJK",
        ),
        "seal": ParagraphStyle(
            "rt_seal",
            fontName=font,
            fontSize=13,
            leading=18,
            alignment=TA_CENTER,
            wordWrap="CJK",
        ),
        "caption": ParagraphStyle(
            "rt_cap",
            fontName=font,
            fontSize=9,
            leading=13,
            alignment=TA_CENTER,
            textColor=colors.grey,
            wordWrap="CJK",
        ),
        "ehead": ParagraphStyle(
            "rt_eh",
            fontName=font,
            fontSize=12,
            leading=15,
            alignment=TA_CENTER,
            wordWrap="CJK",
        ),
        "fine": ParagraphStyle("rt_fine", fontName=font, fontSize=8, leading=12, wordWrap="CJK"),
    }


def _scaled_image(b: bytes, max_w: float, max_h: float) -> Image:
    # reportlab>=5 不再接受裸 bytes，且 Image 不再接受 ImageReader 对象；
    # 必须用 file-like（BytesIO）直传给 Image，另用一个 BytesIO 取尺寸。
    reader = ImageReader(io.BytesIO(b))
    iw, ih = reader.getSize()
    if iw <= 0 or ih <= 0:
        raise ValueError("嵌入影像尺寸非法")
    # 同时受宽、高约束：仅按宽度缩放时，竖长底片会超出版心导致 LayoutError
    scale = min(1.0, max_w / iw, max_h / ih)
    return Image(io.BytesIO(b), width=iw * scale, height=ih * scale)


def _cast(content: object):
    from backend.domain.report.content import ReportContent

    if not isinstance(content, ReportContent):
        raise TypeError("PdfReporter._render 需要 ReportContent")
    return content
