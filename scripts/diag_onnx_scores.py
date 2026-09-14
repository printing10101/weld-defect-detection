"""诊断：部署 ONNX 权重在底片上的**原始输出分数分布**（不套任何阈值）。

回答的问题：评片链路"检出 0 个缺陷"到底是
  (a) 权重失效（输出恒定低分 / 退化为常数），
  (b) 预处理把缺陷抹掉，
  (c) 阈值配置过严。

做法：与生产推理同一条 letterbox 路径直接跑 onnxruntime，打印
输出张量形状、box 回归范围、逐类最大置信、top-10 锚框置信，
再用 YoloDetector 走完整后处理（极低阈值）对照检出数。

用法（仓库根目录，后端 venv）：
  python scripts/diag_onnx_scores.py
  python scripts/diag_onnx_scores.py --model models/weights/best.onnx --n-real 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def imread_any(path: Path) -> np.ndarray:
    """Windows 非 ASCII 路径安全的灰度读图（cv2.imread 会静默返回 None）。"""
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法解码：{path}")
    return img


def letterbox(rgb: np.ndarray, nh: int, nw: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    r = min(nw / w, nh / h)
    nwr, nhr = max(1, int(w * r)), max(1, int(h * r))
    resized = cv2.resize(rgb, (nwr, nhr))
    top = (nh - nhr) // 2
    bottom = nh - nhr - top
    left = (nw - nwr) // 2
    right = nw - nwr - left
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=114
    )
    return padded.transpose(2, 0, 1)[None].astype(np.float32) / 255.0


def synth_pores(seed: int = 42) -> np.ndarray:
    """合成密集气孔底片（与 e2e_api_smoke._make_films 同款）。"""
    base = np.full((480, 640), 165.0, np.float32)
    base[:, 220:420] += 35.0
    base = cv2.GaussianBlur(base, (0, 0), 9)
    rng = np.random.default_rng(seed)
    img = base.copy()
    for _ in range(12):
        x, y = int(rng.integers(240, 400)), int(rng.integers(60, 420))
        r = int(rng.integers(3, 7))
        cv2.circle(img, (x, y), r, -30.0, -1)
    return np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype("uint8")


def synth_slag() -> np.ndarray:
    base = np.full((480, 640), 165.0, np.float32)
    base[:, 220:420] += 35.0
    base = cv2.GaussianBlur(base, (0, 0), 9)
    img = base.copy()
    cv2.rectangle(img, (300, 150), (316, 330), -45.0, -1)
    rng = np.random.default_rng(7)
    return np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype("uint8")


def analyze(sess, inp_name: str, nh: int, nw: int, gray: np.ndarray, label: str) -> None:
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    blob = letterbox(rgb, nh, nw)
    out = np.asarray(sess.run(None, {inp_name: blob})[0])
    print(f"\n### {label}")
    print(f"  图像 {gray.shape}  灰度 min/med/max={gray.min()}/{int(np.median(gray))}/{gray.max()}")
    print(f"  原始输出 shape={out.shape} dtype={out.dtype}")
    if out.ndim == 3 and out.shape[1] < out.shape[2]:
        out = out.transpose(0, 2, 1)
    preds = out[0]
    boxes, sc = preds[:, :4], preds[:, 4:]
    print(f"  box 回归 min/max = {boxes.min():.1f} / {boxes.max():.1f}")
    print(f"  原始分数 min/max = {sc.min():.4f} / {sc.max():.4f}")
    is_logit = bool(sc.min() < -1e-3)
    print(f"  含负值→按 logits 处理（sigmoid）：{is_logit}")
    prob = 1.0 / (1.0 + np.exp(-np.clip(sc, -50.0, 50.0))) if is_logit else sc
    per_class = prob.max(0)
    print("  逐类最大置信：" + "  ".join(f"c{i}={v:.4f}" for i, v in enumerate(per_class)))
    top = np.sort(prob.max(1))[::-1][:8]
    print("  top-8 锚框置信：" + "  ".join(f"{v:.4f}" for v in top))
    n_over = int((prob.max(1) >= 0.30).sum())
    print(f"  ≥0.30 的锚框数：{n_over}   （0 = 权重无有效输出 / 阈值问题不成立）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/weights/best.onnx")
    ap.add_argument("--n-real", type=int, default=4)
    args = ap.parse_args()

    import onnxruntime as ort

    model_path = ROOT / args.model
    print(f"模型：{model_path}  存在={model_path.is_file()}  "
          f"大小={model_path.stat().st_size if model_path.is_file() else 0} bytes")
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    print(f"输入：name={inp.name} shape={inp.shape} type={inp.type}")
    print(f"输出：{[(o.name, o.shape) for o in sess.get_outputs()]}")
    nh = inp.shape[2] if isinstance(inp.shape[2], int) else 640
    nw = inp.shape[3] if isinstance(inp.shape[3], int) else 640

    analyze(sess, inp.name, nh, nw, synth_pores(), "合成·密集气孔")
    analyze(sess, inp.name, nh, nw, synth_slag(), "合成·条状夹渣")

    imgs = sorted((ROOT / "data" / "images").glob("*"))
    imgs = [p for p in imgs if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif"}]
    print(f"\n真实底片 {len(imgs)} 张，取前 {args.n_real} 张：")
    for p in imgs[: args.n_real]:
        try:
            analyze(sess, inp.name, nh, nw, imread_any(p), f"真实·{p.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] {p.name}: {exc}")

    # 完整后处理对照（极低阈值 0.01）：确认「模型有分但被阈值滤掉」还是「真无输出」
    print("\n### 完整后处理对照（YoloDetector，conf=0.01）")
    from backend.domain.detect.yolo_detector import YoloDetector

    det = YoloDetector()
    det.load(str(model_path), "onnx")
    for name, img in (
        ("合成·密集气孔", synth_pores()),
        ("合成·条状夹渣", synth_slag()),
    ):
        dets = det.infer(img, conf=0.01, iou=0.5, class_conf={i: 0.01 for i in range(7)})
        print(f"  {name}: 检出 {len(dets)} 个" + (
            "  " + ", ".join(f"{d.class_id.name}@{d.score:.3f}" for d in dets[:8]) if dets else ""
        ))


if __name__ == "__main__":
    main()
