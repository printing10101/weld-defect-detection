"""软著（计算机软件著作权登记）鉴别材料生成器。

产出符合 CPCC 形式要求的 PDF：
  - 程序鉴别材料：60 页（前 30 + 后 30），每页恰好 50 行，
    页眉为「软件全称 + 版本号」，页脚页码。
  - 文档鉴别材料：用户手册排版 PDF，同一页眉页脚样式，支持插入界面截图。

用法：
  python scripts/make_copyright_materials.py program --out output/ruanzhu
  python scripts/make_copyright_materials.py manual  --out output/ruanzhu \
      --screenshots output/ruanzhu/shots

注意：程序节选文件清单 SOFT_NAME / PROGRAM_FILES 变更后需重跑并核对页数。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader, simpleSplit
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as canvas_module
from reportlab.platypus import (
    HRFlowable,
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

REPO = Path(__file__).resolve().parent.parent
OUT_DEFAULT = REPO / "output" / "ruanzhu"

SOFT_NAME = "射线焊缝缺陷智能检测系统"
SOFT_VERSION = "V1.0"
PAGE_HEADER = f"{SOFT_NAME} {SOFT_VERSION}"

LINES_PER_PAGE = 50
TOTAL_PAGES = 60
FONT_NAME = "SimHei"
FONT_FILE = "C:/Windows/Fonts/simhei.ttf"
FONT_SIZE = 8.5
LEADING = 14.4
MARGIN_X = 48.0
PAGE_W, PAGE_H = A4  # 595.27 x 841.89

# 程序鉴别材料节选清单（按出现顺序拼接；前 1500 行进前 30 页，
# 末 1500 行进后 30 页）。只收录业务核心代码，排除测试/迁移/生成物，
# 也排除训练数据接入（外部数据集）与 GGUF 元数据工具等外围文件。
PROGRAM_FILES: list[str] = [
    "backend/app/main.py",
    "backend/app/security.py",
    "backend/app/batch_queue.py",
    "backend/app/pipelines.py",
    "backend/domain/detect/yolo_detector.py",
    "backend/domain/quantify.py",
    "backend/infra/crypto.py",
    "backend/infra/repository.py",
    "backend/infra/reporting/pdf_reporter.py",
    "src/electron/main.cjs",
    "src/src/main.ts",
    "src/src/stores/backend.ts",
    "src/src/App.vue",
]

# 防御性过滤：命中即丢弃该行并记录（正常应为 0 命中，见生成日志）。
TRACE_PATTERNS = re.compile(
    r"(?i)(copilot|chatgpt|gpt-|claude|anthropic|gemini|codium|tabnine"
    r"|windsurf|deepseek|zcode|generated\s+by|auto[-_ ]?generated|@generated"
    r"|do\s+not\s+edit|AI\s*生成|大模型代写)"
)


def _register_fonts() -> None:
    if FONT_NAME not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_FILE))


def _sanitize_glyphs(line: str) -> str:
    """SimHei 字形表没有的字符（上标²、⇒、数学减号等）打印成豆腐块，替换为 ?。"""
    _register_fonts()
    cmap = pdfmetrics.getFont(FONT_NAME).face.charToGlyph
    return "".join(ch if ord(ch) < 127 or ord(ch) in cmap else "?" for ch in line)


def _read_lines(rel_path: str) -> list[str]:
    """读入单个源文件为展示行：tab 展开、去尾空白、UTF-8 容错。"""
    raw = (REPO / rel_path).read_bytes().decode("utf-8", errors="replace")
    lines = []
    for line in raw.splitlines():
        line = line.replace("\t", "    ").rstrip()
        if TRACE_PATTERNS.search(line):
            print(f"  [过滤] {rel_path}: {line[:60]}")
            continue
        lines.append(_sanitize_glyphs(line))
    return lines


def collect_program_lines() -> list[str]:
    collected: list[str] = []
    for rel in PROGRAM_FILES:
        lines = _read_lines(rel)
        print(f"  {rel}: {len(lines)} 行")
        collected.extend(lines)
    return collected


def wrap_physical(lines: list[str]) -> list[str]:
    """超宽行折行为物理行，保证版心内完整可读、不截断。"""
    _register_fonts()
    usable = PAGE_W - 2 * MARGIN_X
    physical: list[str] = []
    for line in lines:
        if not line:
            physical.append("")
            continue
        # simpleSplit 按字体宽度切分；中文按整字断行足够鉴别材料使用
        physical.extend(simpleSplit(line, FONT_NAME, FONT_SIZE, usable))
    return physical


def _draw_chrome(c: canvas.Canvas, page_no: int, total_pages: int | None) -> None:
    """页眉（软件全称+版本号，居中）与页脚（页码）。"""
    c.setFont(FONT_NAME, 9)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 40, PAGE_HEADER)
    c.setLineWidth(0.5)
    c.line(MARGIN_X, PAGE_H - 46, PAGE_W - MARGIN_X, PAGE_H - 46)
    if total_pages:
        label = f"第 {page_no} 页 共 {total_pages} 页"
    else:
        label = f"第 {page_no} 页"
    c.setFont(FONT_NAME, 8.5)
    c.drawCentredString(PAGE_W / 2, 30, label)


def render_pdf(
    pages: list[list[str]],
    out_path: Path,
    title: str,
    total_pages: int | None,
) -> None:
    _register_fonts()
    c = canvas_module.Canvas(str(out_path), pagesize=A4)
    c.setTitle(title)
    c.setAuthor("ScanDetection")
    c.setCreator(SOFT_NAME)
    try:
        c.setProducer(SOFT_NAME)  # 覆盖默认 ReportLab 生产者元数据
    except AttributeError:
        c._doc.info.producer = SOFT_NAME

    first_content_y = PAGE_H - 64
    for idx, page_lines in enumerate(pages, start=1):
        _draw_chrome(c, idx, total_pages)
        y = first_content_y
        for line in page_lines:
            if line:
                c.drawString(MARGIN_X, y, line)
            y -= LEADING
        c.showPage()
    c.save()


def build_program(out_dir: Path) -> Path:
    print("[1/3] 收集源代码行…")
    lines = collect_program_lines()
    half = TOTAL_PAGES // 2 * LINES_PER_PAGE
    selected = lines[:half] + lines[-half:]
    print(f"[2/3] 源代码共 {len(lines)} 行，节选 {len(selected)} 行（前 {half} + 后 {half}）")

    physical = wrap_physical(selected)
    # 超出 60 页的折行尾部截去，保证恰好 60 页、每页 50 物理行（末页可少）
    physical = physical[: TOTAL_PAGES * LINES_PER_PAGE]
    pages = [
        physical[i : i + LINES_PER_PAGE]
        for i in range(0, len(physical), LINES_PER_PAGE)
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"程序鉴别材料_{SOFT_NAME}{SOFT_VERSION}.pdf"
    render_pdf(pages, out_path, f"{PAGE_HEADER} 程序鉴别材料", TOTAL_PAGES)
    print(f"[3/3] 生成 {out_path}（{len(pages)} 页）")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="软著鉴别材料生成器")
    parser.add_argument("kind", choices=["program", "manual"], help="材料类型")
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument(
        "--screenshots",
        type=Path,
        default=OUT_DEFAULT / "shots",
        help="手册截图目录（manual 用）",
    )
    args = parser.parse_args()

    if args.kind == "program":
        build_program(args.out)
    elif args.kind == "manual":
        build_manual(args.out, args.screenshots)


# ---------------------------------------------------------------------------
# 文档鉴别材料（用户手册 → 排版 PDF，同一页眉页脚样式，按章节插入界面截图）
# ---------------------------------------------------------------------------

MANUAL_MD = REPO / "docs" / "用户手册.md"

# 章节标题前缀 → (截图文件, 图注)。截图由 scripts 之外的截图流程产出。
MANUAL_IMAGES: list[tuple[str, str, str]] = [
    ("## 2. 启动与登录", "01_login.png", "图 1 登录与三员分岗身份认证界面"),
    ("## 3. 界面总览", "02_journey_upload.png", "图 2 单幅评定工作区（底片导入与工艺参数录入）"),
    ("### 4.2 质量门禁（自动）", "12_quality_gate_reject.png", "图 3 底片质量门禁拦截提示（黑度/像质计/位深不达标时阻断并提示重拍）"),
    ("### 4.3 检测与量化（自动）", "03_journey_processing.png", "图 4 自动评定进行中（六阶段本地推理流水线）"),
    ("### 4.4 结论与处置建议", "04_journey_result_full.png", "图 5 评定结果与《射线检测报告》首页预览"),
    ("## 6. 批量检测", "06_batch.png", "图 6 批量评定：批量导入与公共工艺参数"),
    ("## 8. 档案检索 / 底片查看 / 设备标定", "07_archive.png", "图 7 检测档案检索"),
    ("## 9. 系统评价（DB50/T 1807-2025）", "08_std_eval.png", "图 10 系统评价与附录 A 记录表"),
]
MANUAL_IMAGES_MULTI: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "## 8. 档案检索 / 底片查看 / 设备标定",
        [
            ("09_viewer.png", "图 8 底片观察：缩放/窗位窗宽/正反片转换"),
            ("10_device.png", "图 9 设备标定台账与一致性核查"),
        ],
    ),
]

DOC_WIDTH = A4[0] - 2 * 56


def _md_inline(text: str) -> str:
    """行内 Markdown → reportlab 段落内联标记（先转义 XML）。"""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r'<font face="Courier" size="9">\1</font>', text)
    # 链接只保留文字；源文本已带书名号的不再叠加（避免《《x》》）
    text = re.sub(r"《\[([^\]]+)\]\([^)]*\)》", r"《\1》", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"《\1》", text)
    return text


def _styles() -> dict[str, ParagraphStyle]:
    _register_fonts()
    body = ParagraphStyle("body", fontName="DengXian", fontSize=10.5, leading=17, spaceAfter=5)
    h1 = ParagraphStyle("h1", fontName=FONT_NAME, fontSize=17, leading=24, spaceAfter=10, spaceBefore=4, keepWithNext=1)
    h2 = ParagraphStyle("h2", fontName=FONT_NAME, fontSize=13.5, leading=20, spaceBefore=12, spaceAfter=6, keepWithNext=1)
    h3 = ParagraphStyle("h3", fontName=FONT_NAME, fontSize=11.5, leading=17, spaceBefore=8, spaceAfter=4, keepWithNext=1)
    quote = ParagraphStyle("quote", parent=body, leftIndent=14, textColor=colors.HexColor("#444444"), fontSize=9.8, leading=15)
    caption = ParagraphStyle("caption", parent=body, alignment=1, textColor=colors.HexColor("#555555"), fontSize=9, spaceBefore=2, spaceAfter=8)
    cell = ParagraphStyle("cell", parent=body, fontSize=9.5, leading=13, spaceAfter=0)
    return {"body": body, "h1": h1, "h2": h2, "h3": h3, "quote": quote, "caption": caption, "cell": cell}


def _image_flowables(shots_dir: Path, filename: str, caption: str, st: dict) -> list:
    path = shots_dir / filename
    if not path.exists():
        print(f"  [缺图] {path}")
        return []
    iw, ih = ImageReader(str(path)).getSize()
    w = min(430.0, DOC_WIDTH)
    h = w * ih / iw
    max_h = A4[1] - 150
    if h > max_h:  # fullPage 长图缩到一页内
        h = max_h
        w = h * iw / ih
    return [
        Image(str(path), width=w, height=h),
        Paragraph(_md_inline(caption), st["caption"]),
    ]


_STRUCTURAL = re.compile(r"^(#{1,3} |>|\||- |\* |\d+\. |---$)")


def _merge_soft_wrapped(lines: list[str]) -> list[str]:
    """把源 Markdown 里为排版手工折行的段落/列表续行并回上一行。

    Markdown 源里中文段落手工换行硬拷贝成独立段落会出现"断句"观感；
    空行分隔段落、结构行（标题/列表/表格/引用）不动。
    """
    merged: list[str] = []
    for line in lines:
        s = line.strip()
        if not s or not merged:
            merged.append(line)
            continue
        prev = merged[-1].strip()
        if _STRUCTURAL.match(s) or _STRUCTURAL.match(prev):
            merged.append(line)
        elif line[:1] in (" ", "\t") and (prev.startswith("- ") or re.match(r"^\d+\. ", prev)):
            # 列表项的缩进续行：并入列表项（分隔一个空格，避免词语粘连）
            merged[-1] = merged[-1].rstrip() + " " + s
        else:
            # 中文段落软换行：直接拼接
            merged[-1] = merged[-1].rstrip() + s
    return merged


def build_manual(out_dir: Path, shots_dir: Path) -> Path:
    print("[manual] 渲染用户手册…")
    if "DengXian" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("DengXian", "C:/Windows/Fonts/Deng.ttf"))
    st = _styles()
    raw = MANUAL_MD.read_text(encoding="utf-8")
    raw = raw.replace("适用版本：0.1.0", f"适用版本：{SOFT_VERSION}")

    for m in TRACE_PATTERNS.finditer(raw):
        print(f"  [痕迹警告] 手册命中: {m.group(0)}")

    story: list = []
    lines = _merge_soft_wrapped(raw.splitlines())
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith("|"):  # 表格块
            rows: list[list[str]] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                    rows.append([Paragraph(_md_inline(c), st["cell"]) for c in cells])
                i += 1
            if rows:
                ncol = max(len(r) for r in rows)
                table = Table(rows, colWidths=[DOC_WIDTH / ncol] * ncol)
                table.setStyle(TableStyle([
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EFEFEF")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ]))
                story.append(table)
                story.append(Spacer(1, 6))
            continue
        if stripped.startswith("### "):
            key = next((k for k, _, _ in MANUAL_IMAGES if k.startswith("###") and stripped.startswith(k)), None)
            story.append(Paragraph(_md_inline(stripped[4:]), st["h3"]))
            if key:
                _, f, cap = next(x for x in MANUAL_IMAGES if x[0] == key)
                story.extend(_image_flowables(shots_dir, f, cap, st))
        elif stripped.startswith("## "):
            story.append(Paragraph(_md_inline(stripped[3:]), st["h2"]))
            for key, f, cap in MANUAL_IMAGES:
                if key.startswith("##") and stripped.startswith(key) and "###" not in key:
                    story.extend(_image_flowables(shots_dir, f, cap, st))
            for key, imgs in MANUAL_IMAGES_MULTI:
                if stripped.startswith(key):
                    for f, cap in imgs:
                        story.extend(_image_flowables(shots_dir, f, cap, st))
        elif stripped.startswith("# "):
            story.append(Paragraph(_md_inline(stripped[2:]), st["h1"]))
            story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#666666")))
        elif stripped.startswith("> "):
            quote_lines = [stripped[2:]]
            while i + 1 < len(lines) and lines[i + 1].strip().startswith("> "):
                i += 1
                quote_lines.append(lines[i].strip()[2:])
            story.append(Paragraph(_md_inline(" ".join(quote_lines)), st["quote"]))
        elif stripped == "---":
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#AAAAAA")))
        elif re.match(r"^[-*] ", stripped):
            story.append(Paragraph(_md_inline(stripped[2:]), st["body"], bulletText="•"))
        elif re.match(r"^\d+\. ", stripped):
            story.append(Paragraph(_md_inline(stripped), st["body"]))
        else:
            story.append(Paragraph(_md_inline(stripped), st["body"]))
        i += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"文档鉴别材料_用户手册_{SOFT_NAME}{SOFT_VERSION}.pdf"
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=56,
        rightMargin=56,
        topMargin=64,
        bottomMargin=52,
        title=f"{PAGE_HEADER} 用户手册",
        author="ScanDetection",
        creator=SOFT_NAME,
        producer=SOFT_NAME,
    )

    def draw_header(canv, _doc):
        canv.saveState()
        canv.setFont(FONT_NAME, 9)
        canv.drawCentredString(A4[0] / 2, A4[1] - 40, PAGE_HEADER)
        canv.setLineWidth(0.5)
        canv.line(56, A4[1] - 46, A4[0] - 56, A4[1] - 46)
        canv.restoreState()

    class NumberedCanvas(canvas_module.Canvas):
        """两通道渲染：页脚拿到总页数后统一补画。"""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.setTitle(f"{PAGE_HEADER} 用户手册")
            self.setAuthor("ScanDetection")
            self.setCreator(SOFT_NAME)
            self.setProducer(SOFT_NAME)
            self._saved = []

        def showPage(self):
            self._saved.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            total = len(self._saved)
            for state in self._saved:
                self.__dict__.update(state)
                self.setFont(FONT_NAME, 8.5)
                self.drawCentredString(A4[0] / 2, 30, f"第 {self._pageNumber} 页 共 {total} 页")
                super().showPage()
            super().save()

    doc.build(story, onFirstPage=draw_header, onLaterPages=draw_header, canvasmaker=NumberedCanvas)
    print(f"[manual] 生成 {out_path}")
    return out_path


if __name__ == "__main__":
    main()
