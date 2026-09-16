"""资格评定协议基准：一次运行产出 AI 评片系统资格评定所需的全部实测指标。

覆盖五个维度的可复现测量（全部为项目内已实现模块的**首次落盘实测**）：

1. **DB50/T 1807-2025 标准评价**（``evaluation.std501807``）——TDRn/FDRn/MDRn、
   KDR/WDR/TDR/FRR、L1–L4 分级、三类风险判定；
2. **检测指标**（``evaluation.harness.detection_metrics``）——mAP50、逐类 AP、召回、精确；
3. **形状保真**（``evaluation.harness.shape_fidelity_metrics``）——中心/尺度/长宽比解耦误差；
4. **POD 曲线**（尺寸条件化检出概率，MIL-HDBK-1823A hit/miss 口径）+ Wilson 95% CI；
5. **置信度校准**（``evaluation.calibration``）——校准前/后 ECE，逐类温度来自
   ``models/weights/best.calibration.json``；
6. **灰度扰动鲁棒性**（``evaluation.robustness``）——7 条件 + 恒等对照。

设计约束
--------
- **推理路径与部署一致**：复用 ``scripts/eval_real_onnx.py`` 的 letterbox + 逐类阈值 +
  分类别 NMS，``class_conf`` 取自 ``load_config().detect.class_conf``（而非模块默认值，
  后者停留在 6 类时代，会让评估口径与线上漂移）。
- **数据集级 POD**：``harness.pod_curve`` 是单图口径（分位数在单图内取）；本脚本在数据集
  层面按同口径重算（同类别、IoU≥阈值的检出判定不变，仅把尺寸分位数与 Wilson CI 提到
  数据集级），以便得到的 POD 反映整批底片而非单张平均。Wilson CI 直接复用
  ``harness._wilson_ci`` 以保证与库内语义逐位一致。
- **鲁棒性扰动施加在亮度通道**：部署链路读彩色图；本脚本把 BGR 转灰度后在亮度通道施加
  扰动再还原为 3 通道，模拟扫描仪/曝光差异。恒等条件（gain=1.0）一并评估，作为
  保持率的同路径基线，避免把"绝对检出率"误读成"相对保持率"。
- **不制造数字**：任何算不出的量如实留空/NULL（如无缺陷测试集缺失时 FRR 未测，
  标准评价按库内语义拒绝定级），绝不以 0 冒充。

用法::

    backend/.venv/Scripts/python.exe scripts/bench_qualification_protocol.py \
        --img-dir data/training/test/images \
        --label-dir data/training/test/labels \
        --domain synthetic \
        --out data/reports/qualification_protocol_synthetic.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# ---------------------------------------------------------------------------
# 推理器：与部署同一条链路（letterbox + 逐类阈值 + 分类别 NMS）
# ---------------------------------------------------------------------------
class OnnxRunner:
    """ONNX 推理封装；形参为 BGR 数组，便于鲁棒性测试注入扰动图。"""

    def __init__(self, model_path: Path, class_conf: dict[int, float] | None = None) -> None:
        import onnxruntime as ort

        import eval_real_onnx as base

        self.base = base
        # 以部署配置为准（default.yaml + 环境覆盖），而非模块内 6 类时代的默认值
        from backend.infra.config import load_config

        base.CLASS_CONF = dict(class_conf or load_config().detect.class_conf)
        self.sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def infer_bgr(self, bgr: np.ndarray) -> list[dict]:
        padded, scale, pad = self.base.letterbox(bgr)
        rgb = np.ascontiguousarray(padded[:, :, ::-1])
        blob = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
        raw = self.sess.run(None, {self.inp: blob})[0]
        return self.base.parse_and_filter(raw, scale, pad, (bgr.shape[1], bgr.shape[0]))

    def infer_path(self, p: Path) -> list[dict]:
        return self.infer_bgr(self.base.read_image(p))


# ---------------------------------------------------------------------------
# 数据集级 POD（口径与 harness.pod_curve 一致，分箱提到数据集层面）
# ---------------------------------------------------------------------------
def dataset_pod(
    per_image: list[tuple[list[dict], list[dict]]],
    *,
    iou_threshold: float = 0.5,
    n_bins: int = 4,
) -> dict:
    """尺寸条件化检出概率（数据集级）。

    per_image: [(preds, targets)]。逐真值判定：同类别且与任一预测 IoU≥阈值即记检出
    （工作点 POD）。特征长度取 sqrt(面积)，按分位数等频分箱，Wilson 95% CI。
    """
    from backend.evaluation.harness import _bbox_area, _wilson_ci, iou

    sizes: list[float] = []
    flags: list[bool] = []
    for preds, targets in per_image:
        for t in targets:
            sizes.append(float(np.sqrt(_bbox_area(t["bbox"]))))
            flags.append(
                any(
                    p["class_id"] == t["class_id"] and iou(p["bbox"], t["bbox"]) >= iou_threshold
                    for p in preds
                )
            )
    if not sizes:
        return {"overall": {"n": 0, "detected": 0, "pod": 0.0, "ci95": [0.0, 1.0]}, "bins": []}

    arr = np.asarray(sizes)
    edges = sorted({round(float(e), 6) for e in np.quantile(arr, np.linspace(0.0, 1.0, n_bins + 1))})
    if len(edges) == 1:
        edges = [edges[0], edges[0]]

    bins: list[dict] = []
    last = edges[-1]
    for lo, hi in zip(edges, edges[1:]):
        idx = [i for i, s in enumerate(sizes) if lo <= s < hi or (hi == last and s >= hi)]
        n = len(idx)
        if n == 0:
            continue
        k = sum(flags[i] for i in idx)
        ci_lo, ci_hi = _wilson_ci(k, n)
        bins.append(
            {
                "size_min_px": round(lo, 2),
                "size_max_px": round(hi, 2),
                "n": n,
                "detected": int(k),
                "pod": round(k / n, 4),
                "ci95": [round(ci_lo, 4), round(ci_hi, 4)],
            }
        )
    k_all, n_all = sum(flags), len(flags)
    ci_lo, ci_hi = _wilson_ci(k_all, n_all)
    return {
        "overall": {
            "n": n_all,
            "detected": int(k_all),
            "pod": round(k_all / n_all, 4),
            "ci95": [round(ci_lo, 4), round(ci_hi, 4)],
        },
        "bins": bins,
    }


# ---------------------------------------------------------------------------
# ECE（校准前 / 校准后，逐类温度）
# ---------------------------------------------------------------------------
def _ece_at(
    per_image: list[tuple[list[dict], list[dict]]],
    temps: dict[int, float] | None,
) -> dict:
    """在给定逐类温度下计算 ECE；temps=None 即校准前（原始置信度）。

    **工作点保持原则（必须遵守）**：贪心匹配由**原始分数**决定，温度只改变
    参与 ECE 统计的置信度**数值**，不改变"哪个预测配哪个 GT、是否算正确"。
    若允许校准后的分数重新决定匹配次序，正确性标签会随校准变化——那测出来的
    就不是校准误差，而是"校准 + 重匹配"的复合效应，会掩盖真实校准质量。
    """
    from backend.domain.detect.calibration import temperature_transform

    _MATCH_IOU = 0.5
    confs: list[float] = []
    correct: list[bool] = []
    for preds, targets in per_image:
        order = sorted(range(len(preds)), key=lambda i: -float(preds[i].get("score", 0.0)))
        used: set[int] = set()
        for i in order:
            p = preds[i]
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(targets):
                if j in used or int(g.get("class_id", -1)) != int(p.get("class_id", -2)):
                    continue
                v = _iou_xywh(p["bbox"], g["bbox"])
                if v > best_iou:
                    best_iou, best_j = v, j
            ok = best_j >= 0 and best_iou >= _MATCH_IOU
            if ok:
                used.add(best_j)
            score = float(p.get("score", 0.0))
            if temps:
                t = temps.get(int(p["class_id"]))
                if t and t > 0:
                    score = float(temperature_transform(np.asarray([score]), float(t))[0])
            confs.append(score)
            correct.append(ok)

    from backend.evaluation.calibration import expected_calibration_error

    return expected_calibration_error(confs, correct)


def _iou_xywh(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def _find_images(img_dir: Path) -> list[Path]:
    if (img_dir / "images").is_dir():
        img_dir = img_dir / "images"
    return sorted(p for p in img_dir.iterdir() if p.suffix.lower() in _EXT)


def _resolve_dirs(img_dir: Path, label_dir: Path | None) -> tuple[Path, Path]:
    if (img_dir / "images").is_dir():
        return img_dir / "images", (label_dir or img_dir / "labels")
    return img_dir, (label_dir or img_dir.parent / "labels")


def main() -> None:
    ap = argparse.ArgumentParser(description="AI 评片系统资格评定协议基准")
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--label-dir", default=None)
    ap.add_argument("--model", default="models/weights/best.onnx")
    ap.add_argument("--calibration", default="models/weights/best.calibration.json")
    ap.add_argument("--domain", default="unknown", help="数据域标注：synthetic / real")
    ap.add_argument("--weld-form", default="single", choices=["single", "double"])
    ap.add_argument("--weld-method", default="manual", choices=["manual", "auto"])
    ap.add_argument("--n-bins-pod", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-robustness", action="store_true")
    args = ap.parse_args()

    img_dir, label_dir = _resolve_dirs(_ROOT / args.img_dir, Path(args.label_dir) if args.label_dir else None)
    if not label_dir.is_absolute():
        label_dir = _ROOT / label_dir
    model_path = _ROOT / args.model
    calib_path = _ROOT / args.calibration

    imgs = _find_images(img_dir)
    if not imgs:
        raise SystemExit(f"未找到图像: {img_dir}")

    from backend.evaluation.run_eval import _aggregate, _aggregate_shape_fidelity

    runner = OnnxRunner(model_path)

    from backend.evaluation.harness import detection_metrics, shape_fidelity_metrics

    per_image: list[tuple[list[dict], list[dict]]] = []
    per_metrics: list[dict] = []
    per_shape: list[dict] = []
    raw_cache: list[tuple[Path, np.ndarray, list[dict]]] = []
    for p in imgs:
        bgr = runner.base.read_image(p)
        gts = runner.base.load_gt(p.stem, label_dir, (bgr.shape[1], bgr.shape[0]))
        preds = runner.infer_bgr(bgr)
        per_image.append((preds, gts))
        per_metrics.append(detection_metrics(preds, gts))
        per_shape.append(shape_fidelity_metrics(preds, gts))
        raw_cache.append((p, bgr, gts))

    # ---- 1/2/3 检测 + 形状保真 ----
    metrics = _aggregate(per_metrics)
    shape = _aggregate_shape_fidelity(per_shape)

    # ---- 4 POD ----
    pod = dataset_pod(per_image, n_bins=args.n_bins_pod)

    # ---- 5 ECE（校准前/后）----
    temps: dict[int, float] | None = None
    calib_meta: dict = {"file": str(calib_path), "active": False}
    if calib_path.exists():
        from backend.domain.detect.calibration import parse_calibration_payload

        payload = json.loads(calib_path.read_text(encoding="utf-8"))
        temps = parse_calibration_payload(payload, _model_fingerprint(model_path))
        if temps is not None:
            calib_meta.update({"active": True, "temperatures": {str(k): v for k, v in temps.items()}})
    ece_uncal = _ece_at(per_image, None)
    ece_cal = _ece_at(per_image, temps) if temps else None

    # ---- 6 鲁棒性 ----
    robustness = None
    if not args.skip_robustness:
        from backend.evaluation.robustness import DEFAULT_CONDITIONS, robustness_metrics

        def infer_fn(gray: np.ndarray) -> list[dict]:
            import cv2

            return runner.infer_bgr(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

        import cv2

        samples = [
            (cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), gts) for _, bgr, gts in raw_cache
        ]
        # 恒等条件一并评估，作为同路径基线（否则"保持率"会被误读为绝对检出率）
        conditions = (("brightness_gain", 1.0), *DEFAULT_CONDITIONS)
        rep = robustness_metrics(samples, infer_fn, conditions=conditions)
        robustness = {
            "conditions": rep.conditions,
            "retention_worst": rep.retention_worst,
            "quant_dev_worst": rep.quant_dev_worst,
            "fp_increase_worst": rep.fp_increase_worst,
            "passed": rep.passed,
            "failures": rep.failures,
        }

    # ---- DB50/T 1807-2025 标准评价 ----
    from backend.evaluation.std501807 import StdEvalConfig, evaluate

    cfg = StdEvalConfig(weld_form=args.weld_form, weld_method=args.weld_method)
    std_defect_set = [(p.stem, gts, preds) for (p, _b, gts), (preds, _g) in zip(raw_cache, per_image)]
    # 无缺陷测试集未提供 → 库内语义拒绝定级（FRR 未测），如实透出 frr_measured=False
    std = evaluate(std_defect_set, [], cfg)

    # ---- 汇总落盘 ----
    gt_counts = Counter(int(g["class_id"]) for _p, _b, gts in raw_cache for g in gts)
    payload = {
        "domain": args.domain,
        "model": {
            "path": str(model_path.relative_to(_ROOT)),
            "sha256": _sha256(model_path),
            "fingerprint": _model_fingerprint(model_path),
        },
        "dataset": {
            "img_dir": str(img_dir.relative_to(_ROOT)),
            "label_dir": str(label_dir.relative_to(_ROOT)),
            "n_images": len(imgs),
            "n_gt": sum(len(g) for _p, _b, g in raw_cache),
            "gt_by_class": {str(k): v for k, v in sorted(gt_counts.items())},
        },
        "calibration": calib_meta,
        "detection": {"metrics": metrics, "shape_fidelity": shape},
        "pod": pod,
        "ece": {"uncalibrated": ece_uncal, "calibrated": ece_cal},
        "robustness": robustness,
        "std501807": {
            "weld_form": std["weld_form"],
            "weld_method": std["weld_method"],
            "standard": std["standard"],
            "strict": std["strict"],
            "level_recorded": std["level_recorded"],
        },
    }
    out = _ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 控制台摘要 ----
    print(f"[{args.domain}] 图像 {len(imgs)} 张 / GT {payload['dataset']['n_gt']} 个")
    print(f"  mAP50={metrics['mAP50']:.4f} recall={metrics['recall']:.4f} precision={metrics['precision']:.4f}")
    print(
        f"  POD overall={pod['overall']['pod']:.4f} "
        f"CI={pod['overall']['ci95']} n={pod['overall']['n']}"
    )
    for b in pod["bins"]:
        print(
            f"    bin [{b['size_min_px']:>6.2f},{b['size_max_px']:>6.2f}) px "
            f"n={b['n']:>3d} POD={b['pod']:.3f} CI={b['ci95']}"
        )
    print(f"  ECE 未校准={ece_uncal['ece']:.4f} 校准后={ece_cal['ece'] if ece_cal else 'N/A'}")
    if robustness:
        print(f"  鲁棒性 passed={robustness['passed']} 最差保持率={robustness['retention_worst']:.4f}")
        for name, c in robustness["conditions"].items():
            print(f"    {name:<28s} retention={c['retention']:.3f} qdev={c['quant_dev_mean']:.3f}")
    s, st = std["standard"], std["strict"]
    print(
        f"  DB50 标准口径 TDR={s['tdr']:.2%} WDR={s['wdr']:.2%} KDR={s['kdr']:.2%} "
        f"FRR={s['frr']:.2%} 分级={s['level']}（frr_measured={s.get('frr_measured')}）"
    )
    print(f"  DB50 记录分级（从严）={std['level_recorded']} 风险={s['risks']}")
    print(f"  误检方向 fd_pairs={s['fd_pairs']}")
    print(f"  结果已写入 {out}")


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _model_fingerprint(p: Path) -> str:
    """与 model_registry 同口径的模型标识：<stem>::<sha256 前 12 位>。"""
    return f"{p.stem}::{_sha256(p)[:12]}"


if __name__ == "__main__":
    main()
