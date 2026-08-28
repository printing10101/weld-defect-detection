"""SWRD → YOLO 格式转换（，零训练前的 ingestion）。

SWRD 官方标注为**多边形实例级**（porosity/inclusion/crack/undercut/
lack_of_fusion/lack_of_penetration）。本模块把任意常见格式归一为 YOLO txt：
    <class_id> <x_center> <y_center> <w> <h>   （均归一化 0-1）

支持的输入布局（自动探测，落盘于 data/external/swrd/）：
1. COCO 风格 instances_*.json（images/annotations/categories）
2. 每图同名的 .json（{"annotations":[{"label":..,"points":[[x,y],...]}]}）
3. Pascal VOC .xml（<object><name><polygon><x>/<y></polygon>）
4. 已是 YOLO .txt（直接复制）

转换后写到 data/training/raw/swrd/{images,labels}。
"""

from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from backend.training.class_map import map_source_label

_DEFAULT_SRC = Path("data/external/swrd")
_DEFAULT_DST = Path("data/training/raw/swrd")


def _flatten_points(pts: list) -> list[float]:
    """把缺陷坐标统一为展平列表 [x1,y1,x2,y2,...]。

    pts 可能是成对坐标 [[x,y],...]（labelme/COCO 多边形）或展平坐标 [x,y,x,y,...]。
    旧实现误用未定义变量 p 且只取首点两点 → 成对格式下 NameError 且结果错误，
    这里抽取为纯函数便于单元测试。
    """
    if pts and isinstance(pts[0], (list, tuple)):
        return [float(coord) for pt in pts for coord in pt]
    return [float(v) for v in pts]


def _polygon_to_yolo(
    points: list[float], img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    """多边形顶点 (x0,y0,x1,y1,...) → YOLO 归一化中心/宽高。"""
    xs = points[0::2]
    ys = points[1::2]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    cx = ((x0 + x1) / 2) / img_w
    cy = ((y0 + y1) / 2) / img_h
    w = (x1 - x0) / img_w
    h = (y1 - y0) / img_h
    return cx, cy, w, h


def _coco_load(path: Path, dst_img: Path, dst_lbl: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    images = {im["id"]: im for im in data.get("images", [])}
    cats = {c["id"]: c["name"] for c in data.get("categories", [])}
    by_img: dict[int, list[str]] = {}
    for ann in data.get("annotations", []):
        img_id = ann["image_id"]
        cls = map_source_label(cats.get(ann["category_id"], ""))
        if cls is None:
            continue
        seg = ann.get("segmentation")
        if isinstance(seg, list) and seg and isinstance(seg[0], list):
            pts = [float(v) for v in seg[0]]
        elif isinstance(seg, list) and seg:
            pts = [float(v) for v in seg]
        else:
            bbox = ann.get("bbox")  # [x,y,w,h]
            if not bbox:
                continue
            x, y, w, h = bbox
            pts = [x, y, x + w, y, x + w, y + h, x, y + h]
        im = images[img_id]
        cx, cy, w, h = _polygon_to_yolo(pts, im["width"], im["height"])
        by_img.setdefault(img_id, []).append(f"{cls.value} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    count = 0
    for img_id, lines in by_img.items():
        im = images[img_id]
        src = _find_image(_DEFAULT_SRC, im["file_name"])
        if src is None:
            continue
        shutil.copy(src, dst_img / Path(im["file_name"]).name)
        (dst_lbl / (Path(im["file_name"]).stem + ".txt")).write_text("\n".join(lines))
        count += 1
    return count


def _perimage_json(path: Path, dst_img: Path, dst_lbl: Path) -> int:
    """单图同名 .json：{"annotations":[{"label":..,"points":[[x,y],...]}]}。"""
    stem = path.stem
    img = _find_image(_DEFAULT_SRC, stem)
    if img is None:
        return 0
    obj = json.loads(path.read_text(encoding="utf-8"))
    anns = obj.get("annotations") or obj.get("shapes") or []
    lines: list[str] = []
    from PIL import Image  # 延迟导入，避免无 PIL 时报错

    with Image.open(img) as im:
        w, h = im.size
    for a in anns:
        label = a.get("label") or (a.get("points") and None)
        pts = a.get("points") or a.get("polygon") or []
        if not label or len(pts) < 3:
            continue
        cls = map_source_label(label)
        if cls is None:
            continue
        # pts 可能是成对坐标 [[x,y],...]（labelme/COCO 多边形）或展平坐标 [x,y,x,y,...]。
        flat = _flatten_points(pts)
        cx, cy, bw, bh = _polygon_to_yolo(flat, w, h)
        lines.append(f"{cls.value} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    if not lines:
        return 0
    shutil.copy(img, dst_img / Path(img).name)
    (dst_lbl / (stem + ".txt")).write_text("\n".join(lines))
    return 1


def _voc_load(path: Path, dst_img: Path, dst_lbl: Path) -> int:
    stem = path.stem
    img = _find_image(_DEFAULT_SRC, stem)
    if img is None:
        return 0
    root = ET.parse(path).getroot()
    from PIL import Image

    with Image.open(img) as im:
        w, h = im.size
    lines: list[str] = []
    for obj in root.iter("object"):
        name = obj.findtext("name", "").strip()
        cls = map_source_label(name)
        if cls is None:
            continue
        poly = obj.find("polygon")
        pts: list[float] = []
        if poly is not None:
            xs = [float(v.text) for v in poly.iter("x")]
            ys = [float(v.text) for v in poly.iter("y")]
            pts = [v for pair in zip(xs, ys) for v in pair]
        else:
            bnd = obj.find("bndbox")
            if bnd is None:
                continue
            x = float(bnd.findtext("xmin"))
            y = float(bnd.findtext("ymin"))
            xx = float(bnd.findtext("xmax"))
            yy = float(bnd.findtext("ymax"))
            pts = [x, y, xx, y, xx, yy, x, yy]
        if len(pts) < 6:
            continue
        cx, cy, bw, bh = _polygon_to_yolo(pts, w, h)
        lines.append(f"{cls.value} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    if not lines:
        return 0
    shutil.copy(img, dst_img / Path(img).name)
    (dst_lbl / (stem + ".txt")).write_text("\n".join(lines))
    return 1


def _find_image(src: Path, stem: str) -> Path | None:
    stem_p = Path(stem)
    candidates = [
        src / stem_p.name,
        src / (stem_p.stem + ".jpg"),
        src / (stem_p.stem + ".png"),
        src / (stem_p.stem + ".tif"),
        src / (stem_p.stem + ".bmp"),
    ]
    for c in candidates:
        if c.exists():
            return c
    # 递归搜索
    for ext in ("*.jpg", "*.png", "*.tif", "*.bmp", "*.jpeg"):
        hits = list(src.rglob(ext))
        for hit in hits:
            if hit.stem == stem_p.stem:
                return hit
    return None


def convert(src: Path | None = None, dst: Path | None = None) -> int:
    """把 SWRD 多边形标注转换为 YOLO。返回转换图像数。"""
    src = Path(src or _DEFAULT_SRC)
    dst = Path(dst or _DEFAULT_DST)
    dst_img = dst / "images"
    dst_lbl = dst / "labels"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    total = 0
    # 1) COCO
    coco = list(src.rglob("instances_*.json")) + list(src.rglob("*annotations*.json"))
    for f in coco:
        total += _coco_load(f, dst_img, dst_lbl)
    if total:
        return total
    # 2) 逐图 json / voc xml
    for f in sorted(src.rglob("*.json")):
        if f.stem.startswith("instances_") or "annotation" in f.name.lower():
            continue
        total += _perimage_json(f, dst_img, dst_lbl)
    for f in sorted(src.rglob("*.xml")):
        total += _voc_load(f, dst_img, dst_lbl)
    # 3) 已是 YOLO
    if total == 0:
        for f in sorted(src.rglob("*.txt")):
            stem = f.stem
            img = _find_image(src, stem)
            if img is None:
                continue
            shutil.copy(img, dst_img / Path(img).name)
            shutil.copy(f, dst_lbl / f.name)
            total += 1
    return total


if __name__ == "__main__":
    n = convert()
    print(f"converted {n} SWRD images -> YOLO at {_DEFAULT_DST}")
