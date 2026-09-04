"""PDF 报告生成（reportlab，）。

设计文档首选 weasyprint（HTML→PDF/A），但其在 Windows 缺 GTK/Pango 系统库
（本机实测 import 失败）；v1 改用 reportlab（纯 Python、全离线）。
中文使用系统黑体（simhei.ttf 已确认存在）。
 起报告经 pdfa.postprocess_to_pdfa 转写为 PDF/A-1b（字体已全量嵌入 + XMP
标识 + sRGB OutputIntent + 文档 ID），满足  长期归档合规。

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
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from backend.domain.report.content import build_report_content
from backend.infra.reporting.pdfa import postprocess_to_pdfa
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
        # 先渲染 reportlab PDF 到中间文件，再转写为 PDF/A-1b
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

     静态加密兼容：影像副本可能为国密密文（SM4，魔数 b"SDC2"）或历史
    AES-256-GCM 密文（b"SDC1"），检测到任一魔数则用 SCAN_CRYPTO_KEY 委托
    crypto 模块按魔数分流解密后再解码；明文旧数据直接解码。
    密钥缺失/解密失败时返回 None（报告图谱降级为空，不抛 500）。
    """
    try:
        with open(image_path, "rb") as fh:
            buf = fh.read()
    except OSError:
        return None
    if not buf:
        return None
    # C-01 国密化：新副本 SDC2（SM4-CTR+HMAC-SM3），存量副本 SDC1（AES-GCM）
    if buf.startswith((b"SDC2", b"SDC1")):
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


# ---------------------------------------------------------------------------
# 正式检测报告版式（对齐特检院无损检测报告格式）：
# 第1页 封面（报告编号 / 大标题 / 工件字段 / 防伪指纹框）
# 第2页 注意事项（含单位信息与 AI 辅助声明）
# 第3页起 正文（页眉机构名 + 质量文件/报告编号 + 六列信息表 +
# 评定表 + 检测结论框 + 检测/审核/审批签字栏 + 页脚页码）
# 末页 附图（缺陷位置示意图 + 送检原始影像 + 判定依据 + 指纹）
# 页码『第x页 共y页』仅计正文与附图页，封面/注意事项不编号（与参考一致）。
# ---------------------------------------------------------------------------

_CONTENT_START_PAGE = 3  # 封面、注意事项不计页码，正文从此页起算
_SECRET_LEVEL_NAMES = {0: "非密", 1: "内部", 2: "秘密", 3: "机密"}


def classification_label(secret_level: int) -> str:
    """密级数值 → 页面横标文本（C-10）；非密（0）返回空串（不绘制横标）。"""
    level = int(secret_level or 0)
    return _SECRET_LEVEL_NAMES.get(level, "") if level > 0 else ""


_QUALITY_DOC_NO = "SD-RT-R01-1.00"  # 质量文件编号（版式占位）
_RT_LEVEL = "RT-Ⅱ"  # 评片/审核人员资格级别（版式占位）
_ROMAN = {"I": "Ⅰ", "II": "Ⅱ", "III": "Ⅲ", "IV": "Ⅳ"}
_CLASS_CN = {0: "气孔", 1: "夹渣", 2: "未焊透", 3: "未熔合", 4: "裂纹", 5: "咬边", 6: "内凹"}

_NOTES = (
    "1、本报告书适用于焊缝射线检测数字化智能评片；",
    "2、报告书应由计算机打印输出，字迹要工整，涂改无效；",
    "3、本报告书采用电子版模式发放，请使用单位自行打印和保存；",
    "4、本报告书无检测、审核、批准人员签字无效；",
    "5、受检单位对本报告结论如有异议，请在收到报告书之日起15日内，向检测方提出书面意见；",
    "6、本报告评级结果由人工智能辅助评定生成，最终级别须经责任工程师复核并签核后方可采信。",
)


class _ReportCanvas(pdfcanvas.Canvas):
    """两遍渲染页眉/页脚：总页数在 save 时才可知，故先快照各页再统一补画。

    正文页页眉绘制机构名（模板 cover_title），页脚绘制『第x页 共y页』；
    封面与注意事项页不绘制（与参考报告一致）。
    密级标识（C-10）：secret_level>0 时在**每页顶部居中**绘制密级横标
    （军工合规要求密级标识覆盖全部页面，含封面/注意事项），正文页页脚
    另附定密依据。
    """

    def __init__(
        self,
        *args,
        header_text: str = "",
        font: str = "Helvetica",
        classification: str = "",
        basis: str = "",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._header_text = header_text
        self._chrome_font = font
        self._classification = classification  # 如 "秘密"（空串=非密不绘制）
        self._basis = basis  # 定密依据（正文页页脚附注）
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
        page = self._pageNumber  # type: ignore[attr-defined]  # reportlab Canvas 私有属性，stub 未声明
        # 密级横标（C-10）：全部页面绘制，红色加粗，位于页眉文字上方
        if self._classification:
            self.saveState()
            self.setFont(self._chrome_font, 12)
            self.setFillColor(colors.red)
            self.drawCentredString(_PAGE_W / 2.0, _PAGE_H - 16, f"密级：{self._classification}")
            self.restoreState()
        if page < _CONTENT_START_PAGE:
            return
        n = page - _CONTENT_START_PAGE + 1
        n_total = total - _CONTENT_START_PAGE + 1
        self.saveState()
        if self._header_text:
            self.setFont(self._chrome_font, 14)
            self.drawCentredString(_PAGE_W / 2.0, _PAGE_H - 32, self._header_text)
        self.setFont(self._chrome_font, 9)
        self.drawCentredString(_PAGE_W / 2.0, 22, f"第{n}页 共{n_total}页")
        if self._classification and self._basis:
            self.setFont(self._chrome_font, 8)
            self.drawCentredString(_PAGE_W / 2.0, 12, f"定密依据：{self._basis[:80]}")
        self.restoreState()


def _render(
    pdf_path: Path,
    content: object,
    graph_bytes: bytes | None,
    orig_bytes: bytes | None,
    font: str,
    tpl: ReportTemplate,
) -> None:
    """渲染报告 PDF（正式检测报告版式，reportlab platypus 流式排版）。"""
    c = _cast(content)
    styles = _make_styles(font)
    doc = BaseDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=_MARGIN,
        rightMargin=_MARGIN,
        topMargin=_MARGIN,
        bottomMargin=_MARGIN,
        title=f"{tpl.doc_title_prefix} {c.image_id}",
        author=tpl.author,
    )
    pad = {"leftPadding": 0, "rightPadding": 0, "topPadding": 0, "bottomPadding": 0}
    full_frame = Frame(
        _MARGIN, _MARGIN, _PAGE_W - 2 * _MARGIN, _PAGE_H - 2 * _MARGIN, id="full", **pad
    )
    # 正文帧顶部让出页眉区（页眉绘制于 y ≈ _PAGE_H-30）
    main_frame = Frame(
        _MARGIN,
        _MARGIN,
        _PAGE_W - 2 * _MARGIN,
        _PAGE_H - 2 * _MARGIN - 12 * mm,
        id="main",
        **pad,
    )
    doc.addPageTemplates(
        [
            PageTemplate(id="cover", frames=[full_frame]),
            PageTemplate(id="notes", frames=[full_frame]),
            PageTemplate(id="main", frames=[main_frame]),
        ]
    )

    flow: list[Flowable] = []
    flow.extend(_cover_flow(c, tpl, styles))
    flow.append(NextPageTemplate("notes"))
    flow.append(PageBreak())
    flow.extend(_notes_flow(c, tpl, styles))
    flow.append(NextPageTemplate("main"))
    flow.append(PageBreak())
    flow.extend(_main_flow(c, styles))
    flow.extend(_attachment_flow(c, graph_bytes, orig_bytes, styles))

    doc.build(
        flow,
        canvasmaker=lambda *a, **k: _ReportCanvas(
            *a,
            header_text=tpl.cover_title,
            font=font,
            classification=classification_label(getattr(c, "secret_level", 0)),
            basis=getattr(c, "classification_basis", "") or "",
            **k,
        ),
    )


def _cover_flow(c, tpl: ReportTemplate, styles: dict[str, ParagraphStyle]) -> list[Flowable]:
    """封面：报告编号（右上）→ 大标题 → 工件字段 → 防伪指纹框（右下）。"""
    out: list[Flowable] = []
    out.append(Paragraph(f"报告编号：{c.report_id or '—'}", styles["cover_report_no"]))
    out.append(Spacer(1, 48 * mm))
    out.append(Paragraph(tpl.cover_title, styles["cover_sub"]))
    out.append(Spacer(1, 8 * mm))
    out.append(Paragraph("检 测 报 告", styles["cover_big"]))
    out.append(Spacer(1, 40 * mm))
    for label, value in (
        ("工件编号", c.workpiece_no),
        ("焊口编号", c.weld_no),
        ("影像编号", c.image_id),
        ("评定标准", c.standard_ref),
        ("检测时间", _cn_date(c.generated_at)),
    ):
        out.append(Paragraph(f"{label}：{value or '—'}", styles["cover_field"]))
        out.append(Spacer(1, 6 * mm))
    out.append(Spacer(1, 28 * mm))
    # 参考报告同位置为防伪二维码；此处以内容指纹（SHA-256）承担同等防伪校验职责
    box = Table(
        [
            [Paragraph("防伪校验指纹", styles["fp_cap"])],
            [Paragraph(c.fingerprint or "—", styles["fp_val"])],
        ],
        colWidths=[64 * mm],
    )
    box.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.8, colors.black),
                ("TOPPADDING", (0, 0), (-1, 0), 4),
                ("BOTTOMPADDING", (0, -1), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    box.hAlign = "RIGHT"
    out.append(box)
    return out


def _notes_flow(c, tpl: ReportTemplate, styles: dict[str, ParagraphStyle]) -> list[Flowable]:
    """注意事项页：报告使用须知 + AI 辅助声明 + 单位信息。"""
    out: list[Flowable] = [Paragraph("注 意 事 项", styles["notes_title"]), Spacer(1, 8 * mm)]
    out.extend(Paragraph(n, styles["notes_body"]) for n in _NOTES)
    out.append(Spacer(1, 10 * mm))
    if c.disclaimer:
        out.append(Paragraph(c.disclaimer, styles["fine"]))
    out.append(Spacer(1, 30 * mm))
    for line in (
        f"单位名称：{tpl.author}",
        "单位地址：—",
        "邮政编码：—",
        "联系电话：—",
        "电子邮箱：—",
    ):
        out.append(Paragraph(line, styles["notes_body"]))
    return out


def _main_flow(c, styles: dict[str, ParagraphStyle]) -> list[Flowable]:
    """正文页：编号行 + 信息表 + 评定表 + 检测结论 + 签字栏。"""
    w = _PAGE_W - 2 * _MARGIN
    out: list[Flowable] = []
    out.append(
        Paragraph(
            f"质量文件编号：{_QUALITY_DOC_NO}　报告编号：{c.report_id or '—'}",
            styles["doc_no"],
        )
    )
    out.append(Spacer(1, 4 * mm))
    out.append(_meta_table(c, styles, w))
    out.append(Spacer(1, 5 * mm))
    out.append(Paragraph("射线检测结果评定表", styles["table_title"]))
    out.append(_eval_table(c, styles, w))
    out.append(Spacer(1, 5 * mm))
    out.extend(_conclusion_flow(c, styles, w))
    out.append(Spacer(1, 8 * mm))
    out.append(_signature_table(c, styles, w))
    return out


def _meta_table(c, styles: dict[str, ParagraphStyle], w: float) -> Table:
    """六列信息表（标签|值 ×3 一行，共 6 行，对齐参考报告首页表格）。"""
    iqi = c.iqi_detail or {}
    achieved = iqi.get("achieved") or "—"
    required = iqi.get("required") or "—"
    iqi_txt = ("通过" if c.iqi_pass else "不通过") if c.iqi_pass is not None else "未校验"
    rows = [
        (
            "检件名称",
            c.workpiece_no or "—",
            "工件材质",
            "—",
            "工件规格",
            f"{c.base_metal_thickness_mm} mm" if c.base_metal_thickness_mm else "—",
        ),
        ("检测部位", "焊缝", "检测时机", "—", "热处理状态", "—"),
        ("仪器名称", "智能评片系统", "仪器型号", "—", "仪器编号", "—"),
        (
            "影像模态",
            c.modality,
            "像素标定",
            f"{c.pixel_spacing_mm:.4f} mm/px" if c.pixel_spacing_mm else "—",
            "黑度 D",
            f"{c.density:.3f}" if c.density is not None else "—",
        ),
        (
            "像质计",
            f"{iqi_txt}（{achieved}/{required}）",
            "可评片性",
            "可评片" if c.evaluable else "不可评片",
            "检测比例",
            "100%",
        ),
        ("检测标准", c.standard_ref or "—", "合格级别", "—", "操作指导书编号", "—"),
    ]
    data = [
        [
            Paragraph(str(cell), styles["mlabel"] if i % 2 == 0 else styles["mval"])
            for i, cell in enumerate(r)
        ]
        for r in rows
    ]
    t = Table(data, colWidths=[w * 0.125, w * 0.21, w * 0.125, w * 0.21, w * 0.125, w * 0.205])
    t.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.6, colors.black),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def _eval_table(c, styles: dict[str, ParagraphStyle], w: float) -> Table:
    """『射线检测结果评定表』：区段编号/缺陷位置/尺寸/性质/评定/备注。"""
    head = ["区段编号", "缺陷位置", "缺陷尺寸(mm)", "缺陷性质", "评定", "备注"]
    body = [
        [
            f"D{i}",
            _defect_position(d, c.pixel_spacing_mm),
            _defect_size(d),
            _defect_class_name(d),
            _roman_level(d.get("joint_level")),
            "需人工复核" if d.get("need_review") else "—",
        ]
        for i, d in enumerate(c.defects, 1)
    ]
    if not body:
        body = [["见附图", "—", "—", "未检出缺陷", _roman_level(c.joint_level), "—"]]
    data = [
        [Paragraph(str(cell), styles["ehead"] if r == 0 else styles["ecell"]) for cell in row]
        for r, row in enumerate([head] + body)
    ]
    t = Table(data, colWidths=[w * 0.11, w * 0.23, w * 0.15, w * 0.17, w * 0.11, w * 0.23])
    t.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.6, colors.black),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def _conclusion_flow(c, styles: dict[str, ParagraphStyle], w: float) -> list[Flowable]:
    """检测结论框：符合标准要求 / 需人工复核 + 不可评片/复核提示。"""
    out: list[Flowable] = [Paragraph("检测结论：", styles["concl_label"])]
    lines: list[Flowable] = []
    if c.joint_level:
        std = c.standard_ref or "检测标准"
        lines.append(
            Paragraph(f"符合{std}标准{_roman_level(c.joint_level)}要求", styles["concl_val"])
        )
    else:
        lines.append(Paragraph("无法自动评级，需人工复核。", styles["concl_val"]))
    if not c.evaluable:
        lines.append(Paragraph("影像质量不达标，不可评片。", styles["concl_sub"]))
    if c.need_review:
        lines.append(Paragraph("本报告标注需要人工复核。", styles["concl_sub"]))
    box = Table([[lines]], colWidths=[w])
    box.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.8, colors.black),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    out.append(box)
    return out


def _signature_table(c, styles: dict[str, ParagraphStyle], w: float) -> Table:
    """签字栏：检测/审核/审批（左）+ 检验机构检验专用章区（右，占位）。

    S-22 军标见证：content.witness（军代表/见证人）可选；传入时在审批行下
    增加一行"军代表/见证人"，不传则不出现在版式（默认版式不变）。
    """
    date = _cn_date(c.generated_at) or "　年　月　日"
    signer = c.signer or "（签字）"
    witness = getattr(c, "witness", None)
    rows = [
        [
            Paragraph(f"检测（级别）<br/>{_RT_LEVEL}<br/>{signer}<br/>{date}", styles["sig"]),
            Paragraph("检 验 机 构<br/><br/>检 验 专 用 章", styles["stamp"]),
        ],
        [Paragraph(f"审核（级别）<br/>{_RT_LEVEL}<br/>（签字）<br/>{date}", styles["sig"]), ""],
        [Paragraph(f"审批<br/>（签字）<br/>{date}", styles["sig"]), ""],
    ]
    if witness:
        rows.append(
            [
                Paragraph(
                    f"军代表/见证人<br/>{witness}<br/>{date}",
                    styles["sig"],
                ),
                "",
            ]
        )
    data = rows
    t = Table(data, colWidths=[w * 0.58, w * 0.42])
    t.setStyle(
        TableStyle(
            [
                ("SPAN", (1, 0), (1, 2)),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return t


def _attachment_flow(
    c, graph_bytes: bytes | None, orig_bytes: bytes | None, styles: dict[str, ParagraphStyle]
) -> list[Flowable]:
    """附图页：缺陷位置示意图（清单+标注图）+ 原始影像 + 判定依据 + 指纹。"""
    w = _PAGE_W - 2 * _MARGIN
    out: list[Flowable] = [PageBreak()]
    out.append(Paragraph("一 检测部位及缺陷位置示意图：附图", styles["section"]))
    out.append(Spacer(1, 3 * mm))
    # 清单条数上限：与影像并排的单元格不可分页，过多缺陷会撑爆版心；
    # 超出部分以总数提示收尾（完整明细已在正文『评定表』逐行列出）。
    _MAX_LIST = 15
    shown = list(c.defects)[:_MAX_LIST]
    lines = [
        Paragraph(
            f"D{i} {_defect_class_name(d)}：{_defect_size(d)}，"
            f"位于 {_defect_position(d, c.pixel_spacing_mm)}",
            styles["ecell_l"],
        )
        for i, d in enumerate(shown, 1)
    ] or [Paragraph("未检出缺陷。", styles["ecell_l"])]
    if len(c.defects) > _MAX_LIST:
        lines.append(Paragraph(f"……共 {len(c.defects)} 处缺陷，明细见评定表。", styles["ecell_l"]))
    if graph_bytes:
        img = _scaled_image(graph_bytes, w * 0.6, 75 * mm)
        grid = Table([[lines, img]], colWidths=[w * 0.38, w * 0.62])
        grid.setStyle(
            TableStyle(
                [
                    ("BOX", (0, 0), (-1, -1), 0.6, colors.black),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        out.append(grid)
    else:
        out.extend(lines)
    out.append(Spacer(1, 4 * mm))
    if orig_bytes:
        out.append(
            KeepTogether(
                [
                    _scaled_image(orig_bytes, w, 55 * mm),
                    Paragraph("送检原始影像（未标注）", styles["caption"]),
                ]
            )
        )
        out.append(Spacer(1, 4 * mm))
    if c.basis:
        out.append(Paragraph("二 判定依据条款", styles["section"]))
        for i, b in enumerate(c.basis, 1):
            out.append(Paragraph(f"{i}. {b}", styles["fine_l"]))
        out.append(Spacer(1, 4 * mm))
    if c.fingerprint:
        out.append(
            Paragraph(f"数字指纹：SHA-256:{c.fingerprint}（报告内容防篡改校验）", styles["fine_l"])
        )
    out.append(Spacer(1, 6 * mm))
    date = _cn_date(c.generated_at)
    out.append(Paragraph(f"检测：{_RT_LEVEL}　　审核：{_RT_LEVEL}　　{date}", styles["sig"]))
    return out


def _roman_level(level: object) -> str:
    """级别（I/II/III/IV）→ 罗马数字带级（Ⅰ级…）；空值返回 '—'。"""
    lv = str(level or "").strip().upper()
    return f"{_ROMAN.get(lv, lv)}级" if lv else "—"


def _defect_class_name(d: dict) -> str:
    """缺陷性质中文名：class_name 优先，缺省按 class_id 映射。"""
    name = d.get("class_name")
    if name:
        return str(name)
    cid = d.get("class_id")
    return _CLASS_CN.get(cid, "—") if isinstance(cid, int) else "—"


def _defect_position(d: dict, spacing: float | None) -> str:
    """缺陷位置：bbox 中心坐标（有像素标定换算为 mm，否则 px）。"""
    bb = d.get("bbox_px")
    if not bb or len(bb) < 4:
        return "—"
    try:
        x, y, bw, bh = (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
    except (TypeError, ValueError):
        return "—"
    cx, cy = x + bw / 2.0, y + bh / 2.0
    if spacing:
        return f"({cx * spacing:.1f}, {cy * spacing:.1f}) mm"
    return f"({cx:.0f}, {cy:.0f}) px"


def _defect_size(d: dict) -> str:
    """缺陷尺寸：长×宽（mm）。"""
    length, width = d.get("length_mm"), d.get("width_mm")
    if length is not None and width is not None:
        return f"{length:.1f}×{width:.1f}"
    return "—"


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
    """正式报告版式的段落样式（全部使用注册中文字体）。"""
    return {
        # 封面
        "cover_report_no": ParagraphStyle(
            "crn", fontName=font, fontSize=11, leading=15, alignment=TA_RIGHT
        ),
        "cover_sub": ParagraphStyle(
            "cs", fontName=font, fontSize=16, leading=22, alignment=TA_CENTER
        ),
        "cover_big": ParagraphStyle(
            "cb", fontName=font, fontSize=30, leading=40, alignment=TA_CENTER
        ),
        "cover_field": ParagraphStyle("cf", fontName=font, fontSize=14, leading=20),
        "fp_cap": ParagraphStyle("fpc", fontName=font, fontSize=9, leading=12, alignment=TA_CENTER),
        "fp_val": ParagraphStyle(
            "fpv", fontName=font, fontSize=6.5, leading=9, alignment=TA_CENTER
        ),
        # 注意事项
        "notes_title": ParagraphStyle(
            "nt", fontName=font, fontSize=16, leading=22, alignment=TA_CENTER
        ),
        "notes_body": ParagraphStyle("nb", fontName=font, fontSize=10.5, leading=20),
        # 正文
        "doc_no": ParagraphStyle("dn", fontName=font, fontSize=9, leading=13, alignment=TA_RIGHT),
        "table_title": ParagraphStyle(
            "tt",
            fontName=font,
            fontSize=12,
            leading=16,
            alignment=TA_CENTER,
            spaceBefore=4,
            spaceAfter=4,
        ),
        "mlabel": ParagraphStyle("ml", fontName=font, fontSize=9, leading=12, alignment=TA_CENTER),
        "mval": ParagraphStyle("mv", fontName=font, fontSize=9, leading=12, alignment=TA_CENTER),
        "ehead": ParagraphStyle("eh", fontName=font, fontSize=9.5, leading=13, alignment=TA_CENTER),
        "ecell": ParagraphStyle("ec", fontName=font, fontSize=9.5, leading=13, alignment=TA_CENTER),
        "ecell_l": ParagraphStyle(
            "ecl", fontName=font, fontSize=9.5, leading=14, alignment=TA_LEFT
        ),
        "concl_label": ParagraphStyle("cl", fontName=font, fontSize=11, leading=15),
        "concl_val": ParagraphStyle(
            "cv", fontName=font, fontSize=12, leading=18, alignment=TA_CENTER
        ),
        "concl_sub": ParagraphStyle(
            "cs2", fontName=font, fontSize=10, leading=15, alignment=TA_CENTER
        ),
        "sig": ParagraphStyle("sg", fontName=font, fontSize=10.5, leading=16),
        "stamp": ParagraphStyle("st", fontName=font, fontSize=11, leading=18, alignment=TA_CENTER),
        "section": ParagraphStyle("sec", fontName=font, fontSize=12, leading=16),
        "caption": ParagraphStyle(
            "cap",
            fontName=font,
            fontSize=8,
            leading=11,
            alignment=TA_CENTER,
            textColor=colors.grey,
        ),
        "fine": ParagraphStyle("f", fontName=font, fontSize=8, leading=11),
        "fine_l": ParagraphStyle("fl", fontName=font, fontSize=8, leading=12),
    }


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


def _cast(content: object):
    from backend.domain.report.content import ReportContent

    if not isinstance(content, ReportContent):
        raise TypeError("PdfReporter._render 需要 ReportContent")
    return content
