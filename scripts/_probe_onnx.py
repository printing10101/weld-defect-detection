"""临时探针：检查候选 ONNX 模型的 I/O 形状，并在单张真实底片上跑一次推理，
   统计检出框数 / 最大置信，以判定哪个模型在 CPU/ONNX 部署路径下真正可用。
   仅用于决策，不作为交付物。"""
from pathlib import Path
import numpy as np
import cv2
import onnxruntime as ort

ROOT = Path(r"C:\Users\Lenovo\Desktop\扫描检测软件")
CANDIDATES = {
    "default_models_weights": ROOT / "models" / "weights" / "best.onnx",
    "real_synth2": ROOT / "data" / "real_label" / "runs" / "real_synth2" / "weights" / "best.onnx",
    "yolo11n_real2": ROOT / "data" / "real_label" / "runs" / "yolo11n_real2" / "train" / "weights" / "best.onnx",
}
IMG = ROOT / "data" / "real_label" / "images" / "PG101-1-1.jpg"
IMGSZ = 640


def letterbox(img, new=IMGSZ):
    h, w = img.shape[:2]
    scale = min(new / h, new / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_h = new - nh
    pad_w = new - nw
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=114)
    return padded, scale, (left, top)


def read_image(img_path):
    # cv2.imread 在 Windows 含非 ASCII 路径下会静默失败 → 改用字节读取 + imdecode
    data = np.fromfile(str(img_path), dtype=np.uint8)
    if data.size == 0:
        raise RuntimeError(f"空文件: {img_path}")
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"imdecode failed: {img_path}")
    return bgr


def preprocess(img_path):
    bgr = read_image(img_path)
    padded, scale, (left, top) = letterbox(bgr)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[None, ...]  # 1x3x640x640
    return blob, (scale, left, top), (bgr.shape[1], bgr.shape[0])


def parse(out, conf=0.05):
    # 兼容 ultralytics 导出: [1, 4+nc, 8400]
    out = out[0] if isinstance(out, (list, tuple)) else out
    if out.ndim == 3:
        out = out[0]
    # out: (C, N)
    preds = np.transpose(out, (1, 0))  # N x C
    boxes = preds[:, :4]
    scores = preds[:, 4:]
    cls = scores.argmax(1)
    confs = scores.max(1)
    mask = confs >= conf
    return boxes[mask], confs[mask], cls[mask]


def main():
    blob, _, orig = preprocess(IMG)
    print(f"输入 blob: {blob.shape}, 原图尺寸={orig}")
    for name, path in CANDIDATES.items():
        print("\n" + "=" * 60)
        print(f"模型: {name}  ({path})")
        if not path.exists():
            print("  !! 文件不存在")
            continue
        try:
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        except Exception as e:
            print(f"  !! 加载失败: {e}")
            continue
        inp = sess.get_inputs()[0]
        outp = sess.get_outputs()[0]
        print(f"  input:  name={inp.name} shape={inp.shape} dtype={inp.type}")
        print(f"  output: name={outp.name} shape={outp.shape} dtype={outp.type}")
        try:
            res = sess.run([outp.name], {inp.name: blob})
            boxes, confs, cls = parse(res[0], conf=0.05)
            print(f"  @conf>=0.05 检出框数={len(boxes)}  max_conf={float(confs.max()) if len(confs) else 0:.4f}")
            print(f"  类别分布={dict(zip(*np.unique(cls, return_counts=True)))}" if len(cls) else "  无检出")
            # 更低阈值再探一次，看是否均匀低置信
            boxes2, confs2, _ = parse(res[0], conf=0.001)
            print(f"  @conf>=0.001 检出框数={len(boxes2)} max_conf={float(confs2.max()) if len(confs2) else 0:.4f}")
        except Exception as e:
            print(f"  !! 推理失败: {e}")


if __name__ == "__main__":
    main()
