"""决定性问题探针：在全部 165 张真实底片上跑 ONNX 推理，
   统计各模型在部署阈值(0.3)与低阈值(0.05)下的检出能力与置信分布，
   以判定 ONNX 部署路径是否在真实底片上实际可用。
   仅用于决策，不作为交付物。"""
from pathlib import Path
import numpy as np
import cv2
import onnxruntime as ort
from collections import Counter

ROOT = Path(r"C:\Users\Lenovo\Desktop\扫描检测软件")
CANDIDATES = {
    "default": ROOT / "models" / "weights" / "best.onnx",
    "yolo11n_real2": ROOT / "data" / "real_label" / "runs" / "yolo11n_real2" / "train" / "weights" / "best.onnx",
}
IMG_DIR = ROOT / "data" / "real_label" / "images"
IMGSZ = 640
PROVIDERS = ["CPUExecutionProvider"]


def read_image(p):
    data = np.fromfile(str(p), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"imdecode failed: {p}")
    return bgr


def letterbox(img, new=IMGSZ):
    h, w = img.shape[:2]
    scale = min(new / h, new / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = (new - nh) // 2, (new - nh) - (new - nh) // 2
    left, right = (new - nw) // 2, (new - nw) - (new - nw) // 2
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=114)
    return padded, scale, (left, top)


def preprocess(p):
    bgr = read_image(p)
    padded, scale, (left, top) = letterbox(bgr)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[None, ...]
    return blob


def parse(out, conf):
    out = out[0] if isinstance(out, (list, tuple)) else out
    if out.ndim == 3:
        out = out[0]
    preds = np.transpose(out, (1, 0))
    scores = preds[:, 4:]
    cls = scores.argmax(1)
    confs = scores.max(1)
    mask = confs >= conf
    return int(mask.sum()), float(confs.max()) if len(confs) else 0.0, cls[mask]


def main():
    stems = sorted(p.stem for p in IMG_DIR.glob("*.jpg"))
    print(f"真实底片总数: {len(stems)}")
    for name, path in CANDIDATES.items():
        print("\n" + "=" * 64)
        print(f"模型: {name}")
        sess = ort.InferenceSession(str(path), providers=PROVIDERS)
        inp = sess.get_inputs()[0].name
        outn = sess.get_outputs()[0].name
        max_confs = []
        n_det_03 = 0
        n_det_005 = 0
        n_imgs_with_any_005 = 0
        for i, stem in enumerate(stems):
            blob = preprocess(IMG_DIR / (stem + ".jpg"))
            res = sess.run([outn], {inp: blob})[0]
            n05, mc05, _ = parse(res, 0.05)
            n03, mc03, _ = parse(res, 0.3)
            max_confs.append(mc05)
            n_det_005 += n05
            n_det_03 += n03
            if n05 > 0:
                n_imgs_with_any_005 += 1
        max_confs = np.array(max_confs)
        print(f"  部署阈值(0.3): 总检出框={n_det_03}  有检出图=0?{n_det_03 == 0}")
        print(f"  低阈值(0.05): 总检出框={n_det_005}  有检出图={n_imgs_with_any_005}/{len(stems)}")
        print(f"  逐图 max_conf: 均值={max_confs.mean():.4f} 中位={np.median(max_confs):.4f} "
              f"P90={np.percentile(max_confs,90):.4f} 最大={max_confs.max():.4f} "
              f"≥0.3图数={(max_confs>=0.3).sum()}  ≥0.1图数={(max_confs>=0.1).sum()}")


if __name__ == "__main__":
    main()
