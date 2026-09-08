"""逐类温度校准拟合（§15.4 ECE 门禁的修复闭环）。

协议（防乐观偏差 + 最大化留出数据利用）：
1. 在**训练留出划分**（--data，默认 val+test 合并；train 绝不参与——在训练
   数据上拟合校准会被过拟合记忆污染）上以部署同参推理收集逐类
   (score, correct, image) 对，correct 口径与 ECE harness 一致；
2. **图像级 2 折交叉验证**估计泛化：按图像名奇偶分折，A 拟合→B 验证与
   B 拟合→A 验证各跑一遍，取验证折 ECE 均值——这是无偏的泛化数字；
3. 最终部署表用**全部**留出对重新拟合（CV 只做估计，不浪费数据）；
4. 分层拟合策略：样本充足的类单独拟合；不足的类**池化**共享一个温度；
   池化仍不足则恒等 T=1（诚实不拟合优于小样本过拟合）；
5. 校准表落盘权重旁并绑定 model_id 指纹：部署加载时校验，换权重自动失效。

用法（后端 venv）：
  python -m backend.training.fit_calibration
  python -m backend.training.fit_calibration --data data/real_label   # 真实域重拟合
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "models" / "weights" / "best.onnx"
DEFAULT_DATA = [ROOT / "data" / "training" / "val", ROOT / "data" / "training" / "test"]
# 与部署配置 configs/default.yaml detect.class_conf 同步（含 6: 0.10 内凹）
_CLASS_CONF = {0: 0.30, 1: 0.12, 2: 0.12, 3: 0.08, 4: 0.05, 5: 0.18, 6: 0.10}

# 校准对记录：(class_id, score, correct, image_stem)
Record = tuple[int, float, bool, str]


def collect_records(
    detector: Any,
    data_dirs: list[Path],
    *,
    limit_per_dir: int = 0,
    conf: float = 0.25,
    iou: float = 0.5,
) -> list[Record]:
    """在带标注评估集上收集校准对记录（部署同参推理）。"""
    from backend.evaluation.calibration import match_confidences
    from backend.training.post_deploy_eval import _read_gray_unicode_safe, load_yolo_labels

    records: list[Record] = []
    for data_dir in data_dirs:
        images_dir = data_dir / "images"
        labels_dir = data_dir / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            raise FileNotFoundError(f"评估集缺失：{images_dir} / {labels_dir}")
        imgs = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".png"))
        if limit_per_dir > 0:
            imgs = imgs[:limit_per_dir]
        for p in imgs:
            label = labels_dir / (p.stem + ".txt")
            if not label.exists():
                continue
            g = _read_gray_unicode_safe(p)
            if g is None:
                continue
            ih, iw = g.shape[:2]
            gts = load_yolo_labels(label, iw, ih)
            dets = detector.infer(g, conf=conf, iou=iou, class_conf=_CLASS_CONF)
            preds = [
                {
                    "class_id": d.class_id.value,
                    "bbox": [d.bbox.x, d.bbox.y, d.bbox.w, d.bbox.h],
                    "score": d.score,
                }
                for d in dets
            ]
            # match_confidences 按分数降序（稳定排序）返回对，与同序 preds 逐条对齐
            ordered = sorted(preds, key=lambda q: -float(q["score"]))
            for pr, (c, ok) in zip(
                ordered, match_confidences(preds, gts, iou_thr=iou), strict=True
            ):
                records.append((int(pr["class_id"]), c, ok, p.stem))
    return records


def fit_table(records: list[Record], *, min_pairs: int) -> tuple[dict[int, float], dict[str, Any]]:
    """分层拟合温度表：逐类 → 池化 → 恒等（返回 temperatures 与逐类报告）。"""
    by_class: dict[int, list[tuple[float, bool]]] = {}
    for cid, c, ok, _ in records:
        by_class.setdefault(cid, []).append((c, ok))

    temperatures: dict[int, float] = {}
    per_class: dict[str, Any] = {}

    def _apply(cid: int, r: dict[str, Any], pooled: bool) -> None:
        temperatures[cid] = float(r["best_temp"])
        entry = dict(r)
        if pooled:
            entry["pooled"] = True
        per_class[str(cid)] = entry

    fit_ids = sorted(cid for cid, ps in by_class.items() if len(ps) >= min_pairs)
    pooled_ids = sorted(cid for cid, ps in by_class.items() if len(ps) < min_pairs)
    for cid in fit_ids:
        r = grid_search(by_class[cid])
        _apply(cid, r, pooled=False)
        print(
            f"  class {cid}: n={r['n']}  T={r['best_temp']:.2f}  "
            f"ECE {r['ece_before']} → {r['ece_after']}"
        )
    pooled_pairs = [p for cid in pooled_ids for p in by_class[cid]]
    if pooled_ids:
        if len(pooled_pairs) >= min_pairs:
            pr = grid_search(pooled_pairs)
            for cid in pooled_ids:
                _apply(cid, {**pr, "pooled_group_n": len(pooled_pairs)}, pooled=True)
            print(
                f"  classes {pooled_ids}: 池化 n={pr['n']}  T={pr['best_temp']:.2f}  "
                f"ECE {pr['ece_before']} → {pr['ece_after']}"
            )
        else:
            for cid in pooled_ids:
                temperatures[cid] = 1.0
                per_class[str(cid)] = {
                    "best_temp": 1.0,
                    "ece_before": None,
                    "ece_after": None,
                    "n": len(by_class[cid]),
                    "skipped": True,
                }
            print(f"  classes {pooled_ids}: 池化后仍不足（n={len(pooled_pairs)}），保持恒等")
    return temperatures, per_class


def grid_search(pairs: list[tuple[float, bool]]) -> dict[str, Any]:
    from backend.domain.detect.calibration import grid_search_temperature

    return grid_search_temperature([c for c, _ in pairs], [ok for _, ok in pairs])


def apply_temps_ece(records: list[Record], temperatures: dict[int, float]) -> tuple[float, float]:
    """整表 ECE before/after（温度变换保持单调，逐对变换后重算）。"""
    import numpy as np

    from backend.domain.detect.calibration import temperature_transform
    from backend.evaluation.calibration import expected_calibration_error

    confs = [c for _, c, _, _ in records]
    oks = [ok for _, _, ok, _ in records]
    ece_before = float(expected_calibration_error(confs, oks)["ece"])
    cal = [
        float(temperature_transform(np.array([c]), temperatures.get(cid, 1.0))[0])
        for cid, c, _, _ in records
    ]
    ece_after = float(expected_calibration_error(cal, oks)["ece"])
    return ece_before, ece_after


def main() -> None:
    ap = argparse.ArgumentParser(description="逐类温度校准拟合（2 折 CV 估计 + 全量留出拟合）")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument(
        "--data",
        action="append",
        default=[],
        help="留出评估集目录（可多次指定；默认 val+test 合并）",
    )
    ap.add_argument("--out", default="", help="校准表输出路径（默认权重旁 best.calibration.json）")
    ap.add_argument("--limit-per-dir", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    from backend.domain.detect.calibration import MIN_PAIRS_PER_CLASS
    from backend.domain.detect.yolo_detector import YoloDetector
    from backend.evaluation.tracking import ExperimentTracker
    from backend.training.post_deploy_eval import model_id_of

    model_path = Path(args.model)
    model_id = model_id_of(model_path)
    data_dirs = [Path(d) for d in args.data] or DEFAULT_DATA

    det = YoloDetector()
    det.load(str(model_path), backend="onnx")
    print(f"[fit-calibration] model={model_id}")
    print(f"[fit-calibration] 收集校准对（留出划分合并）：{[str(d) for d in data_dirs]}")
    records = collect_records(det, data_dirs, limit_per_dir=args.limit_per_dir, conf=args.conf)
    print(f"[fit-calibration] 校准对 n={len(records)}")

    # 图像级 2 折 CV：折间互验给出泛化估计（拟合表不用于部署）
    stems = sorted({img for _, _, _, img in records})
    fold_a = set(stems[::2])
    rec_a = [r for r in records if r[3] in fold_a]
    rec_b = [r for r in records if r[3] not in fold_a]
    cv_ece_after: list[float] = []
    cv_ece_before: list[float] = []
    for fit_rec, eval_rec in ((rec_a, rec_b), (rec_b, rec_a)):
        if not fit_rec or not eval_rec:
            continue
        temps_fold, _ = fit_table(fit_rec, min_pairs=MIN_PAIRS_PER_CLASS)
        eb, ea = apply_temps_ece(eval_rec, temps_fold)
        cv_ece_before.append(eb)
        cv_ece_after.append(ea)
        print(f"[fit-calibration] CV 折：验证 ECE {eb:.4f} → {ea:.4f}（n={len(eval_rec)}）")
    cv_mean_before = sum(cv_ece_before) / len(cv_ece_before) if cv_ece_before else None
    cv_mean_after = sum(cv_ece_after) / len(cv_ece_after) if cv_ece_after else None

    # 最终部署表：全部留出对拟合（CV 只做估计）
    print("[fit-calibration] 全量留出拟合（最终表）：")
    temperatures, per_class = fit_table(records, min_pairs=MIN_PAIRS_PER_CLASS)
    ece_before, ece_after = apply_temps_ece(records, temperatures)

    payload: dict[str, Any] = {
        "model_id": model_id,
        "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "method": "per_class_temperature_ece_grid_cv2",
        "data": {
            "dirs": [str(d) for d in data_dirs],
            "n_images": len(stems),
            "n_pairs": len(records),
        },
        "cv2": {
            "ece_before_mean": cv_mean_before,
            "ece_after_mean": cv_mean_after,
            "note": "图像级 2 折交叉验证（折间互验均值）= 泛化估计",
        },
        "insample_ece": {"before": ece_before, "after": ece_after},
        "temperatures": {str(k): v for k, v in sorted(temperatures.items())},
        "per_class": per_class,
    }

    out = Path(args.out) if args.out else model_path.parent / "best.calibration.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[fit-calibration] 校准表 -> {out}")
    if cv_mean_before is not None:
        print(
            f"[fit-calibration] 泛化 ECE（CV 均值）：{cv_mean_before:.4f} → {cv_mean_after:.4f}"
            "（门槛 0.05）"
        )

    tracker = ExperimentTracker(ROOT / "data" / "experiments")
    run_id = tracker.start_run(
        "fit_calibration",
        params={"model_id": model_id, "data": [str(d) for d in data_dirs], "n_pairs": len(records)},
    )
    tracker.log_metrics(
        run_id,
        {
            "cv_ece_before": float(cv_mean_before or 0.0),
            "cv_ece_after": float(cv_mean_after or 0.0),
            "n_active_temperatures": float(
                sum(1 for v in temperatures.values() if abs(v - 1.0) > 1e-9)
            ),
        },
    )
    tracker.log_artifact(run_id, str(out.relative_to(ROOT)))


if __name__ == "__main__":
    main()
