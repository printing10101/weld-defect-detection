# !/usr/bin/env python3
"""真实全集检测质量基线。

在全部真实底片（data/real_label/{images,labels}）上用**部署 ONNX 路径**
（onnxruntime CPU，与后端推理一致）跑推理，计算逐类 mAP@0.5 / 召回 / 精确，
产出 JSON 指标 + Markdown 报告，作为"数据扩量 /  校准 / 罕见类召回"的事实基线。

设计要点：
- 输入图像走字节读取 + cv2.imdecode（规避 Windows 非 ASCII 路径静默失败）。
- YOLO ONNX 输出 [1, 4+nc, 8400]：前 4 通道为 (cx,cy,w,h) 绝对像素（640 -letterbox 空间），
  其后 nc 通道为各类 logits/scores；置信 = 各类最大值，类别 = argmax。
- 坐标还原：640 空间 → 原图像素空间（撤销 letterbox: 减 pad、除 scale）→ 与真值同处原图像素系
  （真值由归一化×原图宽高还原），IoU 在几何正确的原图像素空间计算。
- 逐类置信阈值采用部署配置 class_conf（稀有/重大缺陷低阈值优先召回）；
  逐类 NMS(IoU=0.5) 去重后再算指标，避免重复框虚增 FP。
- 指标复用 backend/evaluation/harness.detection_metrics（单一事实源，纯 numpy）。

用法：
  python scripts/eval_real_onnx.py \
      --model models/weights/best.onnx \
      --img-dir data/real_label/images \
      --label-dir data/real_label/labels \
      --out data/eval/real_baseline.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# ---- 指标：优先复用后端 harness，失败则内联等价实现（纯 numpy） ----
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
    from backend.evaluation.harness import detection_metrics  # type: ignore
except Exception:  # noqa: BLE001 — harness 不可导入时内联等价实现

    def iou(a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[0] + a[2], b[0] + b[2])
        y2 = min(a[1] + a[3], b[1] + b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        ua = max(0.0, a[2]) * max(0.0, a[3]) + max(0.0, b[2]) * max(0.0, b[3]) - inter
        return inter / ua if ua > 0 else 0.0

    def _ap(p, r):
        if len(p) == 0:
            return 0.0
        s = 0.0
        for t in np.linspace(0.0, 1.0, 11):
            s += float(p[r >= t].max()) if len(p[r >= t]) else 0.0
        return s / 11.0

    def detection_metrics(preds, targets, iou_threshold=0.5):
        bc = {}
        for x in preds:
            bc.setdefault(x["class_id"], {"p": [], "t": []})["p"].append(x)
        for x in targets:
            bc.setdefault(x["class_id"], {"p": [], "t": []})["t"].append(x)
        cm = {}
        ta = tr = tp = tg = 0
        for cid, g in bc.items():
            ps = sorted(g["p"], key=lambda d: -d["score"])
            ts = g["t"]
            used = [False] * len(ts)
            tpf = []
            for d in ps:
                hit = False
                for i, t in enumerate(ts):
                    if used[i]:
                        continue
                    if iou(d["bbox"], t["bbox"]) >= iou_threshold:
                        used[i] = True
                        hit = True
                        break
                tpf.append(hit)
            cum = np.cumsum(np.array(tpf, float))
            n = len(ts)
            rec = cum / n if n else np.zeros_like(cum)
            prec = cum / np.maximum(cum, 1e-9)
            a = _ap(prec, rec)
            r = float(cum[-1] / n) if n else 0.0
            pr = float(cum[-1] / max(cum[-1] + (len(ps) - cum[-1]), 1e-9)) if ps else 0.0
            cm[str(cid)] = {
                "ap50": round(a, 4),
                "recall": round(r, 4),
                "precision": round(pr, 4),
                "gt_count": n,
            }
            ta += a * n
            tr += r * n
            tp += pr * n
            tg += n
        if tg == 0:
            return {"mAP50": 0.0, "recall": 0.0, "precision": 0.0, "gt_total": 0, "by_class": cm}
        return {
            "mAP50": round(ta / tg, 4),
            "recall": round(tr / tg, 4),
            "precision": round(tp / tg, 4),
            "gt_total": tg,
            "by_class": cm,
        }


IMGSZ = 640
CLASS_NAMES = ["POROSITY", "SLAG", "INCOMPLETE_PENETRATION", "LACK_OF_FUSION", "CRACK", "UNDERCUT"]
# 部署配置 class_conf（DetectCfg），稀有/重大缺陷低阈值优先召回
CLASS_CONF = {0: 0.30, 1: 0.12, 2: 0.12, 3: 0.08, 4: 0.05, 5: 0.18}
NMS_IOU = 0.5


def read_image(p: Path) -> np.ndarray:
    data = np.fromfile(str(p), dtype=np.uint8)
    if data.size == 0:
        raise RuntimeError(f"空文件: {p}")
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"imdecode 失败: {p}")
    return bgr


def letterbox(img, new=IMGSZ):
    h, w = img.shape[:2]
    scale = min(new / h, new / w)
    nh, nw = round(h * scale), round(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    t = (new - nh) // 2
    b = new - nh - t
    l = (new - nw) // 2
    r = new - nw - l
    padded = cv2.copyMakeBorder(resized, t, b, l, r, cv2.BORDER_CONSTANT, value=114)
    return padded, scale, (l, t)


def preprocess(p: Path):
    bgr = read_image(p)
    padded, scale, (left, top) = letterbox(bgr)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[None, ...]
    return blob, scale, (left, top), (bgr.shape[1], bgr.shape[0])


def _nms(boxes, scores, iou_thr):
    """boxes: Nx4 (cx,cy,w,h) 同坐标系；返回保留索引。"""
    if len(boxes) == 0:
        return []
    idx = np.argsort(-np.array(scores))
    keep = []
    while len(idx):
        i = idx[0]
        keep.append(int(i))
        if len(idx) == 1:
            break
        rest = idx[1:]
        bi = boxes[i]
        ious = np.array([_iou_xywh(bi, boxes[j]) for j in rest])
        idx = rest[ious < iou_thr]
    return keep


def _iou_xywh(a, b):
    ax1, ay1, aw, ah = a[0] - a[2] / 2, a[1] - a[3] / 2, a[2], a[3]
    bx1, by1, bw, bh = b[0] - b[2] / 2, b[1] - b[3] / 2, b[2], b[3]
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax1 + aw, bx1 + bw)
    y2 = min(ay1 + ah, by1 + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = aw * ah + bw * bh - inter
    return inter / ua if ua > 0 else 0.0


def parse_and_filter(out, scale, pad, orig_wh):
    """ONNX 输出 → 原图像素空间预测框（NMS 后）。"""
    out = out[0] if isinstance(out, (list, tuple)) else out
    if out.ndim == 3:
        out = out[0]
    preds = np.transpose(out, (1, 0))  # N x (4+nc)
    boxes = preds[:, :4]
    scores = preds[:, 4:]
    confs = scores.max(1)
    cls = scores.argmax(1)
    kept = []
    for c in range(scores.shape[1]):
        m = (cls == c) & (confs >= CLASS_CONF.get(int(c), 0.3))
        if not m.any():
            continue
        bi = np.where(m)[0]
        bboxes_c = boxes[bi]
        keep_idx = _nms(bboxes_c, confs[bi], NMS_IOU)
        for k in keep_idx:
            i = bi[k]
            cx, cy, w, h = bboxes_c[k]
            # 撤销 letterbox → 原图像素
            cx_o = (cx - pad[0]) / scale
            cy_o = (cy - pad[1]) / scale
            w_o = w / scale
            h_o = h / scale
            kept.append(
                {
                    "class_id": int(c),
                    "score": float(confs[i]),
                    "bbox": [float(cx_o - w_o / 2), float(cy_o - h_o / 2), float(w_o), float(h_o)],
                }
            )
    return kept


def load_gt(stem: str, label_dir: Path, orig_wh):
    p = label_dir / (stem + ".txt")
    W, H = orig_wh
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cid, cx, cy, w, h = (
                int(parts[0]),
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
            )
        except ValueError:
            continue
        # 归一化 → 原图像素 [x,y,w,h]
        out.append(
            {
                "class_id": cid,
                "bbox": [(cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H],
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/weights/best.onnx")
    ap.add_argument("--img-dir", default="data/real_label/images")
    ap.add_argument("--label-dir", default="data/real_label/labels")
    ap.add_argument("--out", default="data/eval/real_baseline.json")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    model = root / args.model
    img_dir = root / args.img_dir
    label_dir = root / args.label_dir
    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    outn = sess.get_outputs()[0].name

    stems = sorted(p.name[:-4] for p in img_dir.glob("*.jpg"))
    preds_all, targets_all = [], []
    n_images = 0
    for stem in stems:
        ip = img_dir / (stem + ".jpg")
        if not ip.exists():
            continue
        n_images += 1
        blob, scale, pad, orig_wh = preprocess(ip)
        out = sess.run([outn], {inp: blob})[0]
        preds = parse_and_filter(out, scale, pad, orig_wh)
        targets = load_gt(stem, label_dir, orig_wh)
        preds_all.extend(preds)
        targets_all.extend(targets)

    metrics = detection_metrics(preds_all, targets_all, 0.5)
    payload = {
        "model": str(model),
        "n_images": n_images,
        "class_conf": CLASS_CONF,
        "nms_iou": NMS_IOU,
        "metrics": metrics,
        "generated_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # 控制台摘要
    print(f"模型: {model}")
    print(f"图像数: {n_images}  预测框: {len(preds_all)}  真值框: {len(targets_all)}")
    print(
        f"整体 mAP@0.5={metrics['mAP50']:.4f}  召回={metrics['recall']:.4f}  精确={metrics['precision']:.4f}"
    )
    print(f"{'类别':<24}{'GT':>5}{'AP50':>9}{'召回':>9}{'精确':>9}")
    for cid, m in metrics["by_class"].items():
        name = CLASS_NAMES[int(cid)] if int(cid) < len(CLASS_NAMES) else f"cls{cid}"
        print(
            f"{name:<24}{m['gt_count']:>5}{m['ap50']:>9.4f}{m['recall']:>9.4f}{m['precision']:>9.4f}"
        )
    print(f"\n已写 {out_path}")
    return payload


if __name__ == "__main__":
    main()
