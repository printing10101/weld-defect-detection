"""ONNX 原始分类分数剖析：模型是否会为任何锚框把稀有类作为 argmax？

若 100% 锚框 argmax=气孔(0)，则逐类阈值无法释放稀有类（argmax 已锁死气孔），
必须重训平衡模型。若稀有类在某些锚框是 argmax 但被高统一阈值压制，则逐类阈值有效。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent
ONNX = ROOT / "_pkg" / "ScanDetection" / "models" / "weights" / "best.onnx"
IMG_DIR = ROOT / "data" / "real_label" / "images"
NAMES = ["气孔", "夹渣", "未焊透", "未熔合", "裂纹", "咬边"]

def main():
    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    stems = sorted(p.stem for p in IMG_DIR.glob("*.jpg") if not p.name.startswith("."))[:20]
    argmax_counter = np.zeros(6, dtype=int)
    total_anchors = 0
    top_rare = []  # 记录稀有类作为 argmax 的最高分数
    for stem in stems:
        arr = np.asarray(Image.open(str(IMG_DIR / (stem + ".jpg"))).convert("L"))
        h, w = arr.shape
        r = min(640 / w, 640 / h)
        nw, nh = int(w * r), int(h * r)
        img = np.asarray(Image.open(str(IMG_DIR / (stem + ".jpg"))).convert("RGB").resize((nw, nh)))
        pad = np.full((640, 640, 3), 114, dtype=np.uint8)
        top = (640 - nh) // 2; left = (640 - nw) // 2
        pad[top:top+nh, left:left+nw] = img
        blob = pad.transpose(2,0,1)[None].astype(np.float32) / 255.0
        out = sess.run(None, {inp: blob})[0]
        print(f"[debug] raw out.shape={out.shape}", flush=True)
        preds = out[0]
        F = 4 + 6
        if preds.shape[1] != F:
            preds = preds.transpose()
        print(f"[debug] oriented preds.shape={preds.shape}", flush=True)
        scores = preds[:, 4:]
        if float(np.max(scores)) > 1.0:
            scores = 1.0 / (1.0 + np.exp(-np.clip(scores, -50, 50)))
        cls = scores.argmax(1)
        argmax_counter += np.bincount(cls, minlength=6)
        total_anchors += len(cls)
        # 收集稀有类作为 argmax 的锚框最高分
        for c in (1,2,3,4,5):
            idx = np.where(cls == c)[0]
            if len(idx):
                top_rare.append((stem, c, float(scores[idx].max())))
    print(f"样本 {len(stems)} 张, 锚框总数 {total_anchors}")
    print(f"\n{'类别':<10}{'argmax 锚框数':>14}{'占比':>10}")
    print("-"*36)
    for c in range(6):
        n = int(argmax_counter[c])
        print(f"{NAMES[c]:<10}{n:>14}{n/total_anchors:>9.1%}")
    print(f"\n稀有类作为 argmax 的锚框总数: {int(argmax_counter[1:].sum())}")
    if top_rare:
        print("稀有类 argmax 锚框中的最高置信度（前 10）:")
        for s,c,v in sorted(top_rare, key=lambda x:-x[2])[:10]:
            print(f"   {s:<20} {NAMES[c]} score={v:.4f}")
    else:
        print("结论: 没有任何锚框把稀有类作为 argmax —— 模型分类头气孔独大，")
        print("      逐类阈值无法释放稀有类，必须重训平衡模型（方案 B 实质必要）。")

if __name__ == "__main__":
    main()
