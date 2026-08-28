"""DB50/T 1807-2025 标准评价 CLI。

两种预测来源（二选一）：
- --pred-dir  : YOLO txt 预测目录（归一化 class cx cy w h，与真值同格式）；
- --model     : 部署 ONNX，复用 scripts/eval_real_onnx.py 的推理链路
                （letterbox + class_conf 逐类阈值 + NMS），口径与生产一致。

无缺陷测试集：--clean-img-dir；若带 --clean-label-dir
会校验其标签全空（混入缺陷标注即拒绝评价）。

用法：
  python -m backend.evaluation.run_std_eval \
      --img-dir data/real_label/images --label-dir data/real_label/labels \
      --model models/weights/best.onnx \
      [--clean-img-dir data/std_test/clean/images] \
      [--out data/eval/std_eval.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

from backend.evaluation.std501807 import STD_CLASS_NAMES, StdEvalConfig, evaluate
from backend.infra.config import load_config

_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _config_from_app() -> StdEvalConfig:
    """AppConfig.std_eval（扁平字段）→ StdEvalConfig（frr 聚合成 dict）。"""
    c = load_config().std_eval
    return StdEvalConfig(
        iou_standard=c.iou_standard,
        iou_strict=c.iou_strict,
        weld_form=c.weld_form,
        weld_method=c.weld_method,
        strict_frr=c.strict_frr,
        frr_l1={"auto": c.frr_l1_auto, "manual": c.frr_l1_manual},
        frr_strict={"auto": c.frr_strict_auto, "manual": c.frr_strict_manual},
        aspect_round_max=c.aspect_round_max,
    )


def _image_size(p: Path) -> tuple[int, int]:
    """读图像宽高（走字节读，规避 Windows 非 ASCII 路径）。"""
    import cv2

    import numpy as np

    data = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"图像解码失败: {p}")
    return img.shape[1], img.shape[0]


def _list_images(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in _EXT)


def _load_yolo_labels(p: Path, wh: tuple[int, int]) -> list[dict]:
    """YOLO 归一化标签 → 绝对像素 [{bbox,class_id}]（与 harness 协议一致）。"""
    if not p.exists():
        return []
    W, H = wh
    out = []
    for line in p.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cid, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
        except ValueError:
            continue
        out.append(
            {
                "class_id": cid,
                "bbox": [(cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H],
            }
        )
    return out


def _predict_with_onnx(img_paths: list[Path], model_path: Path) -> dict[str, list[dict]]:
    """ONNX 推理（复用 eval_real_onnx 链路，class_conf 取部署配置，含内凹 0.10）。"""
    import onnxruntime as ort

    sys.path.insert(0, str(_ROOT / "scripts"))
    import eval_real_onnx as base

    from backend.infra.config import DetectCfg

    base.CLASS_CONF = dict(DetectCfg().class_conf)
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    out: dict[str, list[dict]] = {}
    for ip in img_paths:
        blob, scale, pad, orig_wh = base.preprocess(ip)
        raw = sess.run(None, {inp: blob})[0]
        out[ip.stem] = base.parse_and_filter(raw, scale, pad, orig_wh)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="DB50/T 1807-2025 标准评价")
    ap.add_argument("--img-dir", required=True, help="缺陷测试集图像目录")
    ap.add_argument("--label-dir", required=True, help="缺陷测试集真值标签（YOLO 归一化）")
    ap.add_argument("--pred-dir", help="预测标签目录（YOLO txt，与真值同格式）")
    ap.add_argument("--model", help="部署 ONNX 路径（与 --pred-dir 二选一）")
    ap.add_argument("--clean-img-dir", help="无缺陷测试集图像目录（§9.1.2 底片级误报）")
    ap.add_argument("--clean-pred-dir", help="无缺陷集预测目录（pred-dir 模式下）")
    ap.add_argument("--weld-form", choices=["single", "double"])
    ap.add_argument("--weld-method", choices=["manual", "auto"])
    ap.add_argument("--out", default="data/eval/std_eval.json")
    args = ap.parse_args()
    if bool(args.pred_dir) == bool(args.model):
        ap.error("--pred-dir 与 --model 必须二选一")

    cfg = _config_from_app()
    if args.weld_form:
        cfg.weld_form = args.weld_form
    if args.weld_method:
        cfg.weld_method = args.weld_method

    img_dir = _ROOT / args.img_dir
    lbl_dir = _ROOT / args.label_dir
    img_paths = _list_images(img_dir)

    if args.model:
        preds_by_stem = _predict_with_onnx(img_paths, _ROOT / args.model)
    else:
        pred_dir = _ROOT / args.pred_dir
        preds_by_stem = {
            p.stem: _load_yolo_labels(pred_dir / f"{p.stem}.txt", _image_size(p))
            for p in img_paths
        }

    defect_set = []
    for p in img_paths:
        gts = _load_yolo_labels(lbl_dir / f"{p.stem}.txt", _image_size(p))
        defect_set.append((p.stem, gts, preds_by_stem.get(p.stem, [])))

    no_defect_set: list[tuple[str, list[dict]]] = []
    if args.clean_img_dir:
        clean_dir = _ROOT / args.clean_img_dir
        clean_paths = _list_images(clean_dir)
        if args.model:
            clean_preds = _predict_with_onnx(clean_paths, _ROOT / args.model)
        else:
            cpd = _ROOT / args.clean_pred_dir if args.clean_pred_dir else None
            clean_preds = {
                p.stem: (_load_yolo_labels(cpd / f"{p.stem}.txt", _image_size(p)) if cpd else [])
                for p in clean_paths
            }
        for p in clean_paths:
            if args.clean_pred_dir is None and not args.model:
                lbl = _ROOT / args.clean_img_dir / "labels" / f"{p.stem}.txt"
            else:
                lbl = (_ROOT / args.clean_img_dir).parent / "labels" / f"{p.stem}.txt"
            if args.clean_pred_dir is None and lbl.exists() and _load_yolo_labels(lbl, _image_size(p)):
                raise SystemExit(
                    f"无缺陷测试集发现缺陷标注: {p.stem}（§9.1.2 口径被污染，拒绝评价）"
                )
            no_defect_set.append((p.stem, clean_preds.get(p.stem, [])))

    result = evaluate(defect_set, no_defect_set, cfg)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_defect_images": len(defect_set),
        "n_no_defect_images": len(no_defect_set),
        "result": result,
    }
    out = _ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    std = result["standard"]
    strict = result["strict"]
    print(f"标准口径  (IOU≥{std['iou_threshold']}): TDR={std['tdr']:.2%} WDR={std['wdr']:.2%} "
          f"KDR={std['kdr']:.2%} FRR={std['frr']:.2%} 分级={std['level']}")
    print(f"严格口径  (IOU≥{strict['iou_threshold']}): TDR={strict['tdr']:.2%} WDR={strict['wdr']:.2%} "
          f"KDR={strict['kdr']:.2%} FRR={strict['frr']:.2%} 分级={strict['level']}")
    print(f"记录分级（从严）: {result['level_recorded']}")
    print(f"风险: 漏检={std['risks']['miss']} 误检={std['risks']['false_detect']} "
          f"误报={std['risks']['false_report']}")
    for n in sorted(STD_CLASS_NAMES):
        c = std["per_class"][str(n)]
        print(f"  {n} {c['name']}: TDRn={c['tdr']:.2%} FDRn={c['fdr']:.2%} MDRn={c['mdr']:.2%}")
    print(f"结果已写入 {out}")


if __name__ == "__main__":
    main()
