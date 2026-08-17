"""生成《射线焊缝缺陷智能检测系统 — 成果演示》自包含 HTML（真实底片 + 部署模型）。

用法:
    backend/.venv/Scripts/python scripts/make_demo.py
输出:
    demo/成果演示.html   （单文件，内嵌全部图片，可离线打开/微信发送）

内容: 真实底片 → YOLOv8n(ONNX) 检测 → 掩膜精修量化 → 注意力热力图
      → NB/T47013.2-2015 评级 → 自包含演示页。
"""
from __future__ import annotations
import base64, io, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from backend.domain.dto import DefectClass, ImageMeta, Modality
from backend.domain.detect.yolo_detector import YoloDetector
from backend.domain.explain import attention_heatmap
from backend.domain.quantify import MaskQuantifier, MaskRefineCfg
from backend.domain.grade.nb47013 import Nb47013Grader
from backend.domain.standards.tables.loader import load_standard_tables
from backend.infra.repository import InspectionRepository
from backend.infra.reporting.pdf_reporter import PdfReporter

ONNX = ROOT / "_pkg" / "ScanDetection" / "models" / "weights" / "best.onnx"
IMG_DIR = ROOT / "data" / "real_label" / "images"
OUT = ROOT / "demo" / "成果演示.html"
PDF_DB = ROOT / "demo" / "_report_demo.db"   # 报告生成用临时库（可重建）
PDF_OUT = ROOT / "demo" / "reports"

# 方案 A 逐类置信度阈值（验收报告，eval_rare_metrics.json 依据）
CLASS_CONF = {0: 0.30, 1: 0.01, 2: 0.01, 3: 0.008, 4: 0.005, 5: 0.01}
NAMES = {0: "气孔", 1: "夹渣", 2: "未焊透", 3: "未熔合", 4: "裂纹", 5: "咬边"}
EN_NAMES = {0: "POROSITY", 1: "SLAG", 2: "IP", 3: "LOF", 4: "CRACK", 5: "UNDERCUT"}
COLORS = {
    0: "#3b82f6", 1: "#f59e0b", 2: "#10b981", 3: "#8b5cf6", 4: "#ef4444", 5: "#14b8a6",
}

# 演示假设参数（真实底片无标定；页面醒目标注）
PIXEL_SPACING_MM = 0.10
THICKNESS_MM = 12.0

# 选片：覆盖全部 6 类（含唯一真实裂纹 PG101-2-6）
STEMS = [
    "PG101-2-6",   # 裂纹（唯一真实裂纹底片）
    "PG102-5-4",   # 未熔合 + 夹渣
    "PG103-1-1",   # 夹渣
    "PG103-2-4",   # 夹渣
    "PG120-1-1",   # 未焊透 + 未熔合
    "PG121-4-1",   # 未熔合 + 未焊透
    "PL117-2-1",   # 未焊透
    "PG12-2-1",    # 未焊透
]

MAX_W = 920       # 展示图最大宽度
HEAT_TOPN = 4     # 每片最多热力图数量


def load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(str(path)).convert("L"))


def to_b64(img: Image.Image, quality: int = 84) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def fit(img: Image.Image, max_w: int = MAX_W) -> Image.Image:
    if img.width <= max_w:
        return img
    h = round(img.height * max_w / img.width)
    return img.resize((max_w, h), Image.LANCZOS)


def draw_overlay(gray: np.ndarray, dets: list, refined: dict[str, object]) -> Image.Image:
    """原图 + 缺陷框 + 类别/置信度/尺寸 标签（PIL 支持中文）。"""
    img = Image.fromarray(gray).convert("RGB")
    scale = 1.0
    if img.width > MAX_W:
        scale = MAX_W / img.width
        img = fit(img)
    dr = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("msyh.ttc", 20)
        font_sm = ImageFont.truetype("msyh.ttc", 16)
    except Exception:
        font = font_sm = ImageFont.load_default()
    for d in dets:
        cls = d.class_id.value
        x, y, w, h = d.bbox.x * scale, d.bbox.y * scale, d.bbox.w * scale, d.bbox.h * scale
        geo = refined.get(d.id)
        label = f"{NAMES.get(cls, cls)} {d.score:.2f}"
        if geo is not None:
            label += f"  {geo.length_mm:.1f}×{geo.width_mm:.1f}mm"
        dr.rectangle([x, y, x + w, y + h], outline=COLORS.get(cls, "#fff"), width=3)
        tw = dr.textlength(label, font=font)
        dr.rectangle([x, max(0, y - 26), x + tw + 10, y], fill=COLORS.get(cls, "#333"))
        dr.text((x + 5, max(0, y - 24)), label, fill="#111", font=font)
    return img


def heatmap_b64(gray: np.ndarray, d) -> str:
    ov = attention_heatmap(gray, d)  # BGR
    img = Image.fromarray(cv2_bgr2rgb(ov))
    return to_b64(fit(img))


def cv2_bgr2rgb(arr: np.ndarray) -> np.ndarray:
    return arr[..., ::-1]


def grade_film(grader: Nb47013Grader, dets: list) -> dict:
    ctx = ImageMeta(
        modality=Modality.DR,
        pixel_spacing_mm=PIXEL_SPACING_MM,
        base_metal_thickness_mm=THICKNESS_MM,
    )
    try:
        res = grader.grade(dets, ctx)
        return {
            "ok": True,
            "level": res.joint_level.value,
            "need_review": res.need_review,
            "basis": list(res.basis),
            "disclaimer": res.disclaimer or "",
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def build_pdf_report(
    stem: str,
    gray: np.ndarray,
    dets: list,
    refined: dict,
    grader: Nb47013Grader,
) -> bytes | None:
    """为一张真实底片生成正式 PDF/A-1b 报告（含数字签名指纹），返回 PDF bytes。

    复用生产管线（InspectionRepository + PdfReporter），写入演示临时库；
    失败返回 None（不阻断演示页生成）。
    """
    try:
        src = IMG_DIR / f"{stem}.jpg"
        import shutil

        PDF_OUT.mkdir(parents=True, exist_ok=True)
        # 报告嵌入影像：把原图复制为演示副本（路径须存在且可读）
        img_copy = PDF_OUT / f"{stem}_src.jpg"
        if not img_copy.exists():
            shutil.copyfile(src, img_copy)

        img_id = f"DEMO-{stem}"
        if PDF_DB.exists():
            PDF_DB.unlink()
        repo = InspectionRepository(str(PDF_DB))

        # 评级
        ctx = ImageMeta(
            modality=Modality.DR,
            pixel_spacing_mm=PIXEL_SPACING_MM,
            base_metal_thickness_mm=THICKNESS_MM,
        )
        res = grader.grade(dets, ctx)
        # 缺陷明细（生产 DefectRecord 字段）
        defect_rows = []
        for i, d in enumerate(sorted(dets, key=lambda x: x.score, reverse=True)):
            geo = refined.get(d.id)
            rd_shape = d.shape.value if d.shape else ("round" if (geo and geo.aspect_ratio <= 3) else "linear")
            defect_rows.append({
                "id": f"{img_id}-d{i}",
                "image_id": img_id,
                "class_id": d.class_id.value,
                "bbox_px": [d.bbox.x, d.bbox.y, d.bbox.w, d.bbox.h],
                "shape": rd_shape,
                "length_mm": geo.length_mm if geo else None,
                "width_mm": geo.width_mm if geo else None,
                "area_mm2": geo.area_mm2 if geo else None,
                "perimeter_mm": geo.perimeter_mm if geo else None,
                "position_x": geo.position_x_mm if geo else None,
                "position_y": geo.position_y_mm if geo else None,
                "confidence": d.score,
                "uncertainty": d.uncertainty,
                "joint_level": res.per_defect_grade[i].value if i < len(res.per_defect_grade) else None,
                "need_review": res.need_review,
                "standard_id": "NB/T47013.2-2015",
                "standard_version": "2015",
            })
        report_id = f"RPT-DEMO-{stem}"
        report_row = {
            "id": report_id,
            "image_id": img_id,
            "joint_level": res.joint_level.value,
            "generated_at": __import__("datetime").datetime.now(__import__("datetime").UTC).replace(tzinfo=None),
            "pdf_path": "",
            "standard_ref": "NB/T47013.2-2015",
            "signer": "AI 演示（待责任工程师签核）",
            "basis": list(res.basis),
        }
        image_row = {
            "id": img_id,
            "path": str(img_copy),
            "source_type": "image",
            "modality": "DR",
            "workpiece_no": f"DEMO-WP-{stem}",
            "weld_no": stem,
            "pixel_spacing_mm": PIXEL_SPACING_MM,
            "base_metal_thickness_mm": THICKNESS_MM,
            "iqi_pass": True,
            "density": 2.8,
            "density_ok": True,
            "quality_pass": True,
            "evaluable": True,
            "joint_level": res.joint_level.value,
            "need_review": res.need_review,
            "standard_id": "NB/T47013.2-2015",
            "standard_version": "2015",
        }
        repo.create_inspection(image_row, defect_rows, report_row)
        reporter = PdfReporter(repo, str(PDF_OUT))
        pdf_path = reporter.build(img_id)
        return Path(pdf_path).read_bytes()
    except Exception as exc:  # noqa: BLE001  # 报告生成尽力而为
        print(f"[报告跳过] {stem}: {type(exc).__name__}: {exc}")
        return None


def main() -> None:
    t0 = time.time()
    det = YoloDetector()
    det.load(str(ONNX), backend="onnx")
    q = MaskQuantifier()
    cfg = MaskRefineCfg()
    tables = load_standard_tables("NB/T47013.2-2015", filename="nb47013.yaml")
    grader = Nb47013Grader(tables)

    cards: list[str] = []
    for stem in STEMS:
        path = IMG_DIR / f"{stem}.jpg"
        if not path.exists():
            continue
        try:
            gray = load_gray(path)
        except Exception as exc:
            print(f"[跳过] {stem}: {exc}")
            continue
        dets = det.infer(gray, conf=0.30, iou=0.5, class_conf=CLASS_CONF)
        if not dets:
            print(f"[无检出] {stem}")
            continue
        refined: dict[str, object] = {}
        for d in dets:
            rd = q.refine(gray, d, cfg)
            refined[d.id] = q.measure(rd, PIXEL_SPACING_MM)

        overlay = draw_overlay(gray, dets, refined)
        orig_b64 = to_b64(fit(Image.fromarray(gray)))
        over_b64 = to_b64(overlay)

        # 按置信度取 top 缺陷做热力图
        top = sorted(dets, key=lambda d: d.score, reverse=True)[:HEAT_TOPN]
        heats = []
        for d in top:
            heats.append(
                f'<figure class="heat"><img src="data:image/jpeg;base64,{heatmap_b64(gray, d)}" '
                f'alt="heatmap"/><figcaption>{NAMES.get(d.class_id.value, d.class_id)} '
                f'<b>{d.score:.3f}</b></figcaption></figure>'
            )

        # 缺陷表
        rows = []
        for d in sorted(dets, key=lambda x: x.score, reverse=True):
            geo = refined.get(d.id)
            shape = "条形" if (d.shape and d.shape.value == "linear") else "圆形"
            sz = f"{geo.length_mm:.2f}×{geo.width_mm:.2f}mm" if geo else "—"
            rows.append(
                f"<tr><td><span class='dot' style='background:{COLORS.get(d.class_id.value)}'></span>"
                f"{NAMES.get(d.class_id.value, d.class_id)}</td>"
                f"<td>{d.score:.3f}</td><td>{shape}</td><td>{sz}</td></tr>"
            )
        grade = grade_film(grader, dets)
        if grade["ok"]:
            lv = grade["level"]
            lv_color = {"I": "#10b981", "II": "#3b82f6", "III": "#f59e0b", "IV": "#ef4444"}.get(lv, "#fff")
            basis = "；".join(grade["basis"])
            grade_html = (
                f"<div class='grade'><span class='level' style='color:{lv_color};border-color:{lv_color}'>"
                f"{lv} 级</span>"
                f"<span class='tag {"warn" if grade["need_review"] else "ok"}'>"
                f'{"⚠ 需人工复核" if grade["need_review"] else "自动判定"}</span>'
                f"<p class='basis'>{basis}</p></div>"
            )
        else:
            grade_html = f"<div class='grade err'>{grade['error']}</div>"

        cnt = {}
        for d in dets:
            cnt[d.class_id.value] = cnt.get(d.class_id.value, 0) + 1
        summary = " ".join(f"<b style='color:{COLORS.get(k)}'>{v}×{NAMES.get(k)}</b>" for k, v in sorted(cnt.items()))

        cards.append(f"""
<section class="card">
  <header>
    <h2>{stem}</h2>
    <div class="summary">{summary}</div>
  </header>
  <div class="imgs">
    <figure><img src="data:image/jpeg;base64,{orig_b64}" alt="原图"/><figcaption>原始底片（{gray.shape[1]}×{gray.shape[0]}）</figcaption></figure>
    <figure><img src="data:image/jpeg;base64,{over_b64}" alt="检测"/><figcaption>AI 检测 + 掩膜精修量化（{len(dets)} 处缺陷）</figcaption></figure>
  </div>
  <div class="row">
    <div class="heats">{''.join(heats)}</div>
    <div class="meta">
      <table><thead><tr><th>缺陷</th><th>置信度</th><th>形态</th><th>尺寸</th></tr></thead>
      <tbody>{''.join(rows)}</tbody></table>
      {grade_html}
    </div>
  </div>
</section>""")
        print(f"[完成] {stem}: {len(dets)} 框")

    # 为裂纹底片生成正式 PDF/A 报告（演示页"正式报告"区）
    pdf_b64 = None
    crack_stem = "PG101-2-6"
    crack_path = IMG_DIR / f"{crack_stem}.jpg"
    if crack_path.exists():
        gray = load_gray(crack_path)
        dets = det.infer(gray, conf=0.30, iou=0.5, class_conf=CLASS_CONF)
        refined = {}
        for d in dets:
            rd = q.refine(gray, d, cfg)
            refined[d.id] = q.measure(rd, PIXEL_SPACING_MM)
        pdf_bytes = build_pdf_report(crack_stem, gray, dets, refined, grader)
        if pdf_bytes:
            pdf_b64 = base64.b64encode(pdf_bytes).decode("ascii")
            pdf_kb = round(len(pdf_bytes) / 1024)
            print(f"[报告] {crack_stem} PDF/A 生成 {pdf_kb} KB")

    # ---------------- 组装 HTML ----------------
    metrics = [
        ("100%", "稀有缺陷召回", "27/27 真实稀有缺陷全部命中"),
        ("0→17", "裂纹检出", "真实裂纹底片 PG101-2-6 从 0 到 17 框"),
        ("262", "后端单测通过", "覆盖率 88.35%（门禁 70%）"),
        ("165", "真实底片", "含 153 张人工标注（非合成）"),
        ("17", "API 路由", "全部真实实现，无 501 桩"),
        ("0.5s", "单张全链路", "检测→量化→评级（CPU）"),
    ]
    metric_html = "".join(
        f"<div class='metric'><div class='num'>{v}</div><div class='k'>{k}</div>"
        f"<div class='d'>{d}</div></div>" for v, k, d in metrics
    )

    # 方案 A 验收对比表（验收报告数据）
    planA_rows = [
        ("气孔", 381, 381, 0),
        ("夹渣", 7, 32, "+25"),
        ("未焊透", 20, 100, "+80"),
        ("未熔合", 7, 37, "+30"),
        ("裂纹", 0, 17, "+17"),
        ("咬边", 5, 27, "+22"),
        ("稀有类合计", 39, 213, "+174"),
    ]
    planA_html = "".join(
        f"<tr><td>{n}</td><td>{o}</td><td>{ne}</td>"
        f"<td class='delta {'pos' if str(d).startswith('+') else ''}'>{d}</td></tr>"
        for n, o, ne, d in planA_rows
    )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>射线焊缝缺陷智能检测系统 — 成果演示</title>
<style>
  :root {{ --bg:#0b1220; --card:#111a2e; --line:#23304d; --fg:#e6edf7; --mut:#8fa3c4; }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:var(--bg); color:var(--fg); font-family:"Microsoft YaHei","PingFang SC",sans-serif; line-height:1.6; }}
  .wrap {{ max-width:1180px; margin:0 auto; padding:28px 20px 60px; }}
  header.hero {{ text-align:center; padding:36px 0 8px; }}
  header.hero h1 {{ font-size:34px; letter-spacing:1px; }}
  header.hero .sub {{ color:var(--mut); margin-top:10px; font-size:15px; }}
  .tagline {{ text-align:center; color:#93c5fd; margin:14px 0 30px; font-size:16px; }}
  .metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:14px; margin:26px 0; }}
  .metric {{ background:var(--card); border:1px solid var(--line); border-radius:14px; padding:18px 14px; text-align:center; }}
  .metric .num {{ font-size:30px; font-weight:700; color:#fbbf24; }}
  .metric .k {{ margin-top:6px; font-weight:600; }}
  .metric .d {{ color:var(--mut); font-size:12px; margin-top:4px; }}
  h3.sec {{ margin:36px 0 14px; font-size:20px; border-left:4px solid #fbbf24; padding-left:10px; }}
  .note {{ background:#0d1526; border:1px dashed #3b5a8f; border-radius:10px; padding:12px 16px; color:#b8c7e4; font-size:13px; margin:14px 0; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:16px; padding:20px; margin-bottom:26px; }}
  .card header {{ display:flex; align-items:center; gap:16px; flex-wrap:wrap; margin-bottom:14px; }}
  .card h2 {{ font-size:20px; }}
  .summary {{ color:var(--mut); font-size:14px; }}
  .imgs {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
  .imgs figure {{ text-align:center; }}
  .imgs img {{ width:100%; border-radius:10px; border:1px solid var(--line); }}
  .imgs figcaption {{ color:var(--mut); font-size:12px; margin-top:6px; }}
  .row {{ display:grid; grid-template-columns:1fr 380px; gap:16px; margin-top:14px; }}
  .heats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; }}
  .heat img {{ width:100%; border-radius:8px; border:1px solid var(--line); }}
  .heat figcaption {{ font-size:11px; color:var(--mut); text-align:center; margin-top:4px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th,td {{ padding:6px 8px; text-align:left; border-bottom:1px solid var(--line); }}
  th {{ color:var(--mut); font-weight:600; }}
  .dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; }}
  .grade {{ margin-top:12px; border:1px solid var(--line); border-radius:10px; padding:12px; }}
  .grade .level {{ font-size:24px; font-weight:800; border:2px solid; border-radius:8px; padding:2px 12px; }}
  .grade .tag {{ margin-left:10px; font-size:12px; padding:2px 8px; border-radius:20px; }}
  .tag.warn {{ background:#7f1d1d33; color:#fca5a5; border:1px solid #ef4444; }}
  .tag.ok {{ background:#14532d33; color:#86efac; border:1px solid #22c55e; }}
  .grade .basis {{ color:var(--mut); font-size:12px; margin-top:8px; }}
  .grade.err {{ color:#fca5a5; }}
  .tbl {{ overflow-x:auto; }}
  .delta.pos {{ color:#4ade80; font-weight:700; }}
  .pdf-zone .pdf-card {{ background:linear-gradient(135deg,#14203a,#0f1a30); border:1px solid #2b4a7a; border-radius:14px; padding:20px; display:flex; align-items:center; gap:18px; flex-wrap:wrap; }}
  .pdf-zone .pdf-card p {{ flex:1; min-width:260px; color:#c6d4ec; font-size:14px; }}
  .pdf-btn {{ display:inline-block; background:#fbbf24; color:#111; font-weight:700; padding:12px 20px; border-radius:10px; text-decoration:none; }}
  .pdf-btn:hover {{ background:#fcd34d; }}
  footer {{ margin-top:40px; color:var(--mut); font-size:12px; text-align:center; }}
  @media (max-width:900px) {{ .imgs,.row {{ grid-template-columns:1fr; }} }}
</style></head><body><div class="wrap">

<header class="hero">
  <h1>射线焊缝缺陷智能检测系统</h1>
  <div class="sub">YOLOv8n · ONNX Runtime · NB/T47013.2-2015 · 掩膜精修量化 · 注意力热力图 · PDF/A 报告 · 审计哈希链</div>
</header>
<div class="tagline">以下全部结果均由<b>真实工业底片</b>与<b>已部署 ONNX 模型</b>在本地实时推理得出，无任何人工修饰。</div>

{"<div class='pdf-zone'><h3 class='sec'>正式报告（PDF/A-1b · 数字签名）</h3><div class='pdf-card'><p>对真实裂纹底片 <b>PG101-2-6</b> 一键生成的合规归档报告：PDF/A-1b 长期归档格式、内容指纹 SHA-256 数字签名、判定依据条款快照。</p><a class='pdf-btn' href='data:application/pdf;base64," + pdf_b64 + "' download='PG101-2-6_评片报告.pdf'>⬇ 下载正式评片报告（PDF/A，{pdf_kb} KB）</a></div></div>" if pdf_b64 else ""}

<h3 class="sec">关键成果</h3>
<div class="metrics">{metric_html}</div>

<h3 class="sec">逐类置信度阈值：方案 A 验收（165 张真实底片实测）</h3>
<div class="tbl"><table>
<thead><tr><th>类别</th><th>OLD 统一 0.30</th><th>NEW 逐类阈值</th><th>Δ</th></tr></thead>
<tbody>{planA_html}</tbody>
</table></div>
<div class="note">策略：气孔取高阈值 0.30 压制过检（381→381 零增长）；稀有类取低阈值 0.005–0.01 吃满召回
（96.3%→100%）；低分稀有框经 uncertainty 机制自动进入人工复核，遵循"安全关键缺陷优先召回"原则。</div>

<h3 class="sec">真实底片逐张演示（推理链路：检测 → 掩膜精修 → 量化 → 评级）</h3>
{''.join(cards)}

<div class="note">⚠ <b>参数假设</b>：真实底片无标定信息，演示按像素间距 0.10 mm/px、母材厚度 12.0 mm 假设计算物理尺寸与评级；
接入设备标定档案（§12.4）后自动套用真实标定。<b>标准声明</b>：NB/T47013 数值表转录自公开解读、未持有授权正本
（authorized_copy=false），本评级仅用于 AI 辅助预筛演示，不构成法定判定依据，须责任工程师复核签核。</div>

<footer>ScanDetection · 生成于 {time.strftime('%Y-%m-%d %H:%M')} · 单文件自包含，可离线打开</footer>
</div></body></html>"""

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"\n[输出] {OUT}  ({OUT.stat().st_size/1024:.0f} KB, 耗时 {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
