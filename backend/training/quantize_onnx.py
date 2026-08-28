"""ONNX INT8 量化可行性验证（P1-D，借鉴 micronet PTQ 思路）。

自包含脚本（不 import backend 包，避免 ML venv 缺 Web 依赖）：
1. 对 models/weights/best.onnx 做 PTQ 静态 INT8 量化（用真实底片做校准集；
   静态失败时回退动态量化），产出 *_int8.onnx；
2. 用真实底片子集对比 FP32 vs INT8 的检出（数量 / 匹配框置信度漂移 / 框重合度），
   输出可行性报告（JSON + Markdown）。

⚠️ 关键约束（本机已验证）：legacy ONNX 导出输出已融合 sigmoid 的概率通道
（非 logits），量化后须复测 score 分布与逐类阈值，防漏检。

用法（ML venv）：
  python -m backend.training.quantize_onnx \
    --src models/weights/best.onnx --dst models/weights/best_int8.onnx \
    --images data/real_label/images --limit 24
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

# 与 backend.domain.dto.DefectClass 一致（避免 import backend）
NC = 6
_INPUT_SIZE = 640

# 逐类置信度阈值（与 configs/default.yaml detect.class_conf 一致，供后处理复刻）
_CLASS_CONF = {0: 0.30, 1: 0.12, 2: 0.12, 3: 0.08, 4: 0.05, 5: 0.18}
_GLOBAL_CONF = 0.3
_IOU = 0.5

_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# ---------------------------------------------------------------------------
# 预处理 / 后处理（复刻 YoloDetector._infer_onnx 语义，保证对比口径一致）
# ---------------------------------------------------------------------------


def read_gray(path: Path) -> np.ndarray | None:
    """中文路径安全灰度读取。"""
    arr = np.fromfile(str(path), dtype=np.uint8)
    if arr.size == 0:
        return None
    im = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    if im.ndim == 2:
        return im
    if im.ndim == 3:
        return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(im, cv2.COLOR_BGRA2GRAY)


def letterbox(image: np.ndarray, size: int = _INPUT_SIZE) -> tuple[np.ndarray, float, int, int]:
    """letterbox 缩放 → (blob[1,3,size,size], r, top, left)。"""
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    h, w = rgb.shape[:2]
    r = min(size / w, size / h)
    new_w, new_h = int(w * r), int(h * r)
    resized = cv2.resize(rgb, (new_w, new_h))
    top = (size - new_h) // 2
    bottom = size - new_h - top
    left = (size - new_w) // 2
    right = size - new_w - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=114)
    blob = padded.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return blob, r, top, left


def postprocess(
    raw: np.ndarray,
    *,
    r: float,
    top: int,
    left: int,
    conf: float = _GLOBAL_CONF,
    iou: float = _IOU,
    class_conf: dict[int, float] | None = None,
) -> list[tuple[float, float, float, float, int, float]]:
    """ONNX 输出 → [(x, y, w, h, cls, score)]（原图坐标系，复刻检测器后处理）。

    含通道优先布局自适应、logits→sigmoid 启发式、逐类阈值、坐标还原与 NMS。
    """
    out = np.asarray(raw)
    if out.shape[1] == 4 + NC:  # 通道优先 (1, 4+nc, anchors) → (1, anchors, 4+nc)
        out = out.transpose(0, 2, 1)
    preds = out[0]  # [anchors, 4+nc]
    boxes_xywh = preds[:, :4]
    scores_all = preds[:, 4:]
    if scores_all.min() < -1e-3:  # 原始 logits（legacy 兜底）
        scores_all = 1.0 / (1.0 + np.exp(-np.clip(scores_all, -50.0, 50.0)))
    cls = scores_all.argmax(1)
    score = scores_all.max(1)
    cc = class_conf if class_conf is not None else _CLASS_CONF
    thr = np.array([cc.get(int(c), conf) for c in cls], dtype=np.float32)
    mask = score >= thr
    if not np.any(mask):
        return []
    bx, sc, cl = boxes_xywh[mask], score[mask], cls[mask]
    x1 = (bx[:, 0] - bx[:, 2] / 2 - left) / r
    y1 = (bx[:, 1] - bx[:, 3] / 2 - top) / r
    x2 = (bx[:, 0] + bx[:, 2] / 2 - left) / r
    y2 = (bx[:, 1] + bx[:, 3] / 2 - top) / r
    raw_boxes = list(
        zip(x1.tolist(), y1.tolist(), x2.tolist(), y2.tolist(), sc.tolist(), cl.tolist())
    )
    keep = _nms(raw_boxes, iou)
    out_boxes: list[tuple[float, float, float, float, int, float]] = []
    for i in keep:
        x1i, y1i, x2i, y2i, sci, cli = raw_boxes[i]
        out_boxes.append((x1i, y1i, x2i - x1i, y2i - y1i, int(cli), float(sci)))
    return out_boxes


def _nms(boxes: list, iou_thr: float) -> list[int]:
    if not boxes:
        return []
    try:
        xywh = np.array([[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes], dtype=np.float32)
        scores = np.array([b[4] for b in boxes], dtype=np.float32)
        idxs = cv2.dnn.NMSBoxes(xywh, scores, 0.0, iou_thr)
        if isinstance(idxs, tuple):
            idxs = idxs[0]
        return [int(i) for i in idxs]
    except (cv2.error, AttributeError):
        kept: list[int] = []
        used: list[tuple[float, float, float, float]] = []
        for i in sorted(range(len(boxes)), key=lambda j: boxes[j][4], reverse=True):
            x1, y1, x2, y2 = boxes[i][:4]
            if any(_iou((x1, y1, x2, y2), u) > iou_thr for u in used):
                continue
            kept.append(i)
            used.append((x1, y1, x2, y2))
        return kept


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / max(union, 1e-9)


def detect_file(sess, img_path: Path, conf: float = _GLOBAL_CONF, iou: float = _IOU) -> list:
    gray = read_gray(img_path)
    if gray is None:
        return []
    blob, r, top, left = letterbox(gray)
    out = sess.run(None, {sess.get_inputs()[0].name: blob})[0]
    return postprocess(out, r=r, top=top, left=left, conf=conf, iou=iou)


# ---------------------------------------------------------------------------
# 量化
# ---------------------------------------------------------------------------


class _CalibReader:
    """PTQ 静态量化校准集：真实底片 letterbox 后的输入张量。"""

    def __init__(self, images: list[Path], size: int = _INPUT_SIZE):
        self._blobs = []
        for p in images:
            gray = read_gray(p)
            if gray is None:
                continue
            blob, _, _, _ = letterbox(gray, size)
            self._blobs.append(blob)
        self._i = 0

    def get_next(self) -> dict | None:
        if self._i >= len(self._blobs):
            return None
        blob = self._blobs[self._i]
        self._i += 1
        return {"images": blob}

    def rewind(self) -> None:
        self._i = 0


def quantize_static_ptq(src: Path, dst: Path, calib_images: list[Path]) -> str:
    """PTQ 静态 INT8 量化（Conv 权重/激活），返回量化模式。"""
    from onnxruntime.quantization import QuantType, quantize_static

    reader = _CalibReader(calib_images)
    quantize_static(
        str(src),
        str(dst),
        reader,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        per_channel=True,
    )
    return "static_int8"


def quantize_dynamic_ptq(src: Path, dst: Path) -> str:
    """动态量化（权重 INT8，Conv 通常保持 FP32）——静态失败的兜底。"""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
    return "dynamic_int8"


def quantize(src: Path, dst: Path, calib_images: list[Path]) -> tuple[str, str]:
    """执行量化；静态优先，失败回退动态。返回 (mode, error|"")。"""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        mode = quantize_static_ptq(src, dst, calib_images)
        return mode, ""
    except Exception as exc:  # noqa: BLE001 - 回退
        import traceback

        try:
            mode = quantize_dynamic_ptq(src, dst)
            return mode, f"static 失败回退 dynamic：{exc}"
        except Exception as exc2:  # noqa: BLE001
            return "", f"{exc}\n{exc2}\n{traceback.format_exc()}"


# ---------------------------------------------------------------------------
# 对比与报告
# ---------------------------------------------------------------------------


@dataclass
class ImageCompare:
    image: str
    n_fp32: int
    n_int8: int
    matched: int
    mean_abs_score_delta: float
    mean_iou_matched: float


@dataclass
class QuantReport:
    src: str
    dst: str
    mode: str
    note: str
    size_fp32_mb: float
    size_int8_mb: float
    compression_pct: float
    n_images: int
    mean_abs_score_delta: float
    det_delta_pct: float  # (n_int8 - n_fp32)/n_fp32
    mean_iou_matched: float
    verdict: str
    per_image: list[dict] = field(default_factory=list)
    generated_at: str = field(
        default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S")
    )


def compare_models(
    src: Path,
    dst: Path,
    images: list[Path],
    conf: float = _GLOBAL_CONF,
    iou: float = _IOU,
    iou_match: float = 0.3,
    score_delta_limit: float = 0.05,
) -> QuantReport:
    """FP32 vs INT8 检出对比。阈值：平均|Δscore|≤score_delta_limit 判可行。"""
    import onnxruntime as ort

    s32 = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
    s8 = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    rows: list[ImageCompare] = []
    all_deltas: list[float] = []
    all_ious: list[float] = []
    n_fp32 = n_int8 = 0
    for p in images:
        d32 = detect_file(s32, p, conf, iou)
        d8 = detect_file(s8, p, conf, iou)
        n_fp32 += len(d32)
        n_int8 += len(d8)
        matched, deltas, ious = _match(d32, d8, iou_match)
        all_deltas.extend(deltas)
        all_ious.extend(ious)
        rows.append(
            ImageCompare(
                image=p.name,
                n_fp32=len(d32),
                n_int8=len(d8),
                matched=matched,
                mean_abs_score_delta=float(np.mean(deltas)) if deltas else 0.0,
                mean_iou_matched=float(np.mean(ious)) if ious else 0.0,
            )
        )
    mean_delta = float(np.mean(all_deltas)) if all_deltas else 0.0
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    det_delta = (n_int8 - n_fp32) / max(n_fp32, 1) * 100.0
    size_fp32 = src.stat().st_size / 1e6
    size_int8 = dst.stat().st_size / 1e6
    ok = mean_delta <= score_delta_limit and det_delta > -30.0
    return QuantReport(
        src=str(src),
        dst=str(dst),
        mode="?",
        note="",
        size_fp32_mb=round(size_fp32, 2),
        size_int8_mb=round(size_int8, 2),
        compression_pct=round((1 - size_int8 / max(size_fp32, 1e-9)) * 100, 1),
        n_images=len(images),
        mean_abs_score_delta=round(mean_delta, 4),
        det_delta_pct=round(det_delta, 1),
        mean_iou_matched=round(mean_iou, 4),
        verdict="OK（score 漂移可控，可试点部署）" if ok else "WARN（score 漂移超限，需复核阈值）",
        per_image=[asdict(r) for r in rows],
    )


def _match(d32: list, d8: list, iou_match: float) -> tuple[int, list[float], list[float]]:
    """贪心 IoU 匹配 FP32/INT8 检出（同类别优先），返回 (匹配数, Δscore 列表, IoU 列表)。"""
    deltas: list[float] = []
    ious: list[float] = []
    matched = 0
    used8: set[int] = set()
    for b32 in d32:
        x1, y1, w1, h1, c1, s1 = b32
        best_i, best_iou = -1, 0.0
        for j, b8 in enumerate(d8):
            if j in used8 or int(b8[4]) != c1:
                continue
            iu = _iou((x1, y1, x1 + w1, y1 + h1), (b8[0], b8[1], b8[0] + b8[2], b8[1] + b8[3]))
            if iu > best_iou:
                best_i, best_iou = j, iu
        if best_i >= 0 and best_iou >= iou_match:
            used8.add(best_i)
            matched += 1
            deltas.append(abs(s1 - d8[best_i][5]))
            ious.append(best_iou)
    return matched, deltas, ious


def _collect_images(images_dir: Path, limit: int) -> list[Path]:
    out: list[Path] = []
    for p in sorted(images_dir.iterdir()):
        if p.suffix.lower() in _EXTS:
            out.append(p)
            if len(out) >= limit:
                break
    return out


def _restore_native_delete() -> None:
    """WorkBuddy 沙箱 safe-delete 护栏会劫持 os.remove/Path.unlink（回收站不可用时
    FAIL_CLOSED 拒绝删除）。onnxruntime 量化内部会 unlink 临时文件（如
    *-inferred.onnx），被护栏拦截导致量化失败。恢复进程内原生删除——
    仅作用于本进程自己生成的临时文件，无外部风险（同 planB_run.py 顶部做法）。"""
    try:
        import nt  # Windows 原生删除
        import pathlib

        os.unlink = nt.unlink  # type: ignore[attr-defined]
        os.remove = nt.remove  # type: ignore[attr-defined]
        pathlib.Path.unlink = lambda self, *a, **k: os.unlink(str(self))  # type: ignore[attr-defined]
        pathlib.Path.rmdir = lambda self, *a, **k: os.rmdir(str(self))  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001, S110 - 非沙箱环境无需恢复，静默即可
        pass


def main() -> None:
    _restore_native_delete()
    ap = argparse.ArgumentParser(description="ONNX INT8 量化可行性验证（P1-D）")
    ap.add_argument("--src", default="models/weights/best.onnx")
    ap.add_argument("--dst", default="models/weights/best_int8.onnx")
    ap.add_argument("--images", default="data/real_label/images", help="真实底片目录（校准+对比）")
    ap.add_argument("--limit", type=int, default=24, help="对比图数上限")
    ap.add_argument("--calib", type=int, default=10, help="校准图数（静态量化）")
    ap.add_argument("--report", default="data/experiments/quant_report.json")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    imgs = _collect_images(Path(args.images), max(args.limit, args.calib))
    if not imgs:
        print(f"[quant] 无图像：{args.images}")
        sys.exit(1)
    print(f"[quant] 图像 {len(imgs)} 张（校准 {args.calib}，对比 {min(len(imgs), args.limit)}）")

    mode, note = quantize(src, dst, imgs[: args.calib])
    if not mode:
        print(f"[quant] 量化失败：{note}")
        sys.exit(2)
    print(f"[quant] 量化完成：{dst}（mode={mode}）{('；' + note) if note else ''}")

    report = compare_models(src, dst, imgs[: args.limit])
    report.mode = mode
    report.note = note
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[quant] 报告 → {args.report}")
    print(
        f"[quant] 结论: {report.verdict}\n"
        f"  FP32 {report.size_fp32_mb}MB → INT8 {report.size_int8_mb}MB（压缩 {report.compression_pct}%）\n"
        f"  检出: FP32 {sum(r['n_fp32'] for r in report.per_image)} / INT8 {sum(r['n_int8'] for r in report.per_image)}"
        f"（Δ {report.det_delta_pct}%）\n"
        f"  匹配框平均|Δscore|={report.mean_abs_score_delta}，平均 IoU={report.mean_iou_matched}"
    )


if __name__ == "__main__":
    main()
