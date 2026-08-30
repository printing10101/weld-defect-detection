"""I 类漏检风险证据包（DB50/T 1807-2025 E-08）。

I 类漏检风险需附带"标注原图 vs 系统识别图"对照证据：
- 证据图：原图复制两份并排，左侧叠加人工标注框（绿）、右侧叠加系统检测框
  （红），供评审直观比对漏检差异；
- 证据包 manifest：JSON 清单（底片 id、缺陷 id、评级、文件 hash），hash 用
  hashlib SHA-256（文件完整性留痕，非密码学承诺）。

产出目录 data/eval/evidence/<record_id>/，由 /std-eval/evidence/{record_id} 调用。
纯 cv2/hashlib，可离线单测。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# 框颜色（BGR）：绿=人工标注（真值），红=系统识别（检测）
_GT_COLOR = (0, 200, 0)
_DET_COLOR = (0, 0, 230)
_THICKNESS = 2


def _imread_unicode(p: Path) -> np.ndarray | None:
    """Unicode 安全解码（cv2.imread 在 Windows 非 ASCII 路径返回 None）。"""
    buf = np.fromfile(str(p), dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _imwrite_unicode(p: Path, img: np.ndarray) -> None:
    """Unicode 安全编码落盘。"""
    ok, buf = cv2.imencode(p.suffix or ".png", img)
    if not ok:
        raise RuntimeError(f"证据图编码失败: {p.name}")
    buf.tofile(str(p))


def _draw_boxes(img: np.ndarray, boxes: list[list[float]], color) -> np.ndarray:
    out = img.copy()
    for b in boxes:
        x, y, w, h = (round(float(v)) for v in b[:4])
        cv2.rectangle(out, (x, y), (x + w, y + h), color, _THICKNESS)
    return out


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_evidence(
    *,
    record_id: str,
    film_path: str | Path,
    film_id: str = "",
    defects: list[dict[str, Any]] | None = None,
    gt_boxes: list[list[float]] | None = None,
    det_boxes: list[list[float]] | None = None,
    out_dir: str | Path,
) -> dict[str, Any]:
    """生成漏检风险证据图 + manifest，返回 manifest dict。

    defects: 漏检缺陷清单 [{defect_id, class_id, grade?}]（grade = NB/T 47013.2
    评级，来自 judge API per_defect_grade；缺失留空）。
    """
    film = Path(film_path)
    img = _imread_unicode(film)
    if img is None:
        raise FileNotFoundError(f"底片无法解码: {film}")
    # 并排：左=标注原图（绿框），右=系统识别图（红框）
    left = _draw_boxes(img, gt_boxes or [], _GT_COLOR)
    right = _draw_boxes(img, det_boxes or [], _DET_COLOR)
    sep = np.full((img.shape[0], 4, 3), 255, np.uint8)
    combo = np.hstack([left, sep, right])

    out = Path(out_dir) / record_id
    out.mkdir(parents=True, exist_ok=True)
    img_file = out / "evidence.png"
    _imwrite_unicode(img_file, combo)

    manifest: dict[str, Any] = {
        "record_id": record_id,
        "film_id": film_id,
        "film_path": str(film),
        "film_sha256": _sha256(film),
        "defects": defects or [],
        "gt_boxes": gt_boxes or [],
        "det_boxes": det_boxes or [],
        "evidence_image": str(img_file),
        "evidence_image_sha256": _sha256(img_file),
        "layout": "left=标注原图(绿框) | right=系统识别图(红框)",
        "generated_at": datetime.now().isoformat(timespec="seconds"),  # noqa: DTZ005
    }
    manifest_file = out / "manifest.json"
    manifest_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def load_manifest(record_id: str, evidence_dir: str | Path) -> dict[str, Any] | None:
    """读取已生成证据包的 manifest；不存在返回 None。"""
    p = Path(evidence_dir) / record_id / "manifest.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))
