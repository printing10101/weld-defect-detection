"""部署后评估闭环：部署权重 × 带标注评估集 → 实测指标 + 模型卡 + 实验记录。

规格书 §7.4/§15 的评估 harness（mAP/ECE/门禁）此前"只有实现没有数字"——
本脚本把闭环接上：对当前部署权重在与**部署一致的推理路径**（domain
YoloDetector）下产出 mAP50/逐类 AP/召回/精确 + ECE 校准 + 与上次评估的
回归对比，并落盘三处：

1. ``data/eval/deployed_eval.json``          —— 最新评估报告（含历史指针）；
2. ``data/model_cards/<model_id>.json``      —— 模型卡（实测指标/数据分布/局限）；
3. ``data/experiments/experiments.jsonl``    —— ExperimentTracker 实验记录。

诚实性铁律：评估域（synthetic/real）按数据路径**如实标注**并在模型卡
局限中声明，禁止把同源合成域指标冒充真实域性能；``data/real_label`` 恢复
后以 ``--data data/real_label`` 重跑即得真实域数字。

用法（后端 venv）：
  python -m backend.training.post_deploy_eval                 # 默认部署权重 × 合成测试集
  python -m backend.training.post_deploy_eval --data data/real_label   # 真实域
  python -m backend.training.post_deploy_eval --limit 50      # 快速抽样
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "models" / "weights" / "best.onnx"
DEFAULT_DATA = ROOT / "data" / "training" / "test"

# 部署语义逐类置信度阈值——与 configs/default.yaml detect.class_conf 保持同步
# （7 类，含 6: 0.10 内凹）。
_CLASS_CONF: dict[int, float] = {0: 0.30, 1: 0.12, 2: 0.12, 3: 0.08, 4: 0.05, 5: 0.18, 6: 0.10}

_CLASS_NAMES_ZH = {
    0: "气孔",
    1: "夹渣",
    2: "未焊透",
    3: "未熔合",
    4: "裂纹",
    5: "咬边",
    6: "内凹",
}


def _read_gray_unicode_safe(path: Path) -> np.ndarray | None:
    """Windows 上 cv2.imread 不支持非 ASCII 路径，与 infra.image_loader 同口径：
    np.fromfile + imdecode（此处不引 infra，保持训练脚本对运行时零依赖）。"""
    import cv2
    import numpy as np

    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf.size == 0:
        return None
    im = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im


def model_id_of(path: Path) -> str:
    """``best::<sha256[:12]>``——与 infra.model_registry 的 id 语义一致。"""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return f"{path.stem}::{h.hexdigest()[:12]}"


def domain_label(data_dir: Path) -> str:
    """评估域如实标注：路径含 real 视为真实域，否则合成域。"""
    return "real" if "real" in str(data_dir).lower() else "synthetic"


def load_yolo_labels(label_path: Path, img_w: int, img_h: int) -> list[dict[str, Any]]:
    """YOLO txt（cls cx cy w h 归一化）→ harness 真值 [{class_id, bbox:[x,y,w,h]}]。"""
    targets: list[dict[str, Any]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(float(parts[0]))
        cx, cy, w, h = (float(v) for v in parts[1:5])
        targets.append(
            {
                "class_id": cid,
                "bbox": [
                    (cx - w / 2) * img_w,
                    (cy - h / 2) * img_h,
                    w * img_w,
                    h * img_h,
                ],
            }
        )
    return targets


def run_post_deploy_eval(
    *,
    model_path: Path = DEFAULT_MODEL,
    data_dir: Path = DEFAULT_DATA,
    limit: int = 0,
    conf: float = 0.25,
    iou: float = 0.5,
    verbose: bool = True,
) -> dict[str, Any] | None:
    """执行一次部署后评估并落盘三件套，返回评估报告（无标注数据时返回 None）。"""

    from backend.domain.detect.calibration import parse_calibration_payload
    from backend.domain.detect.yolo_detector import YoloDetector
    from backend.evaluation.calibration import (
        expected_calibration_error,
        match_confidences,
    )
    from backend.evaluation.harness import check_regression, detection_metrics
    from backend.evaluation.tracking import ExperimentTracker, build_model_card

    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        if verbose:
            print(f"[post-deploy-eval] 评估集缺失（{images_dir} / {labels_dir}），跳过")
        return None
    imgs = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".png"))
    if limit > 0:
        imgs = imgs[:limit]
    if not imgs:
        if verbose:
            print("[post-deploy-eval] 评估集为空，跳过")
        return None

    detector = YoloDetector()
    detector.load(str(model_path), backend="onnx")

    def _collect(
        d: Any,
    ) -> tuple[
        list[dict[str, Any]], list[dict[str, Any]], list[tuple[float, bool]], dict[str, int], int
    ]:
        """单遍推理收集：预测 / 真值 / (score, correct) 对 / 类别计数 / 有效图数。"""
        preds_all: list[dict[str, Any]] = []
        targets_all: list[dict[str, Any]] = []
        pairs: list[tuple[float, bool]] = []
        counts: dict[str, int] = {}
        used = 0
        for p in imgs:
            label = labels_dir / (p.stem + ".txt")
            if not label.exists():
                continue
            arr = _read_gray_unicode_safe(p)
            if arr is None:
                continue
            ih, iw = arr.shape[:2]
            targets = load_yolo_labels(label, iw, ih)
            dets = d.infer(arr, conf=conf, iou=iou, class_conf=_CLASS_CONF)
            preds = [
                {
                    "class_id": x.class_id.value,
                    "bbox": [x.bbox.x, x.bbox.y, x.bbox.w, x.bbox.h],
                    "score": x.score,
                }
                for x in dets
            ]
            preds_all.extend(preds)
            targets_all.extend(targets)
            pairs.extend(match_confidences(preds, targets, iou_thr=iou))
            for t in targets:
                key = str(t["class_id"])
                counts[key] = counts.get(key, 0) + 1
            used += 1
        return preds_all, targets_all, pairs, counts, used

    all_preds, all_targets, conf_pairs, class_counts, n_used = _collect(detector)
    if n_used == 0:
        if verbose:
            print("[post-deploy-eval] 无带标注图像，跳过")
        return None

    metrics = detection_metrics(all_preds, all_targets, iou_threshold=iou)
    ece = expected_calibration_error([c for c, _ in conf_pairs], [ok for _, ok in conf_pairs])

    mid = model_id_of(model_path)
    domain = domain_label(data_dir)
    # 与上次评估（同域）的回归对比：信息性输出，不设阻断（门禁属 CI Golden 职责）
    prev = _load_previous(domain)
    regression = None
    if prev is not None and prev.get("metrics", {}).get("gt_total", 0) > 0:
        gate = check_regression(metrics, prev["metrics"])
        regression = {
            "against": prev.get("created_at"),
            "passed": gate.passed,
            "deltas": gate.deltas,
            "violations": gate.violations,
        }

    # 双通道：权重旁存在匹配指纹的温度校准表时，加跑校准通道，并把**校准后**
    # 作为部署口径（与生产推理一致——app 装配会自动注入同一张表）。原始通道
    # 数字保留为 *_uncalibrated，两版并存供对比。
    metrics_uncal = metrics
    ece_uncal = ece
    calibration_info: dict[str, Any] | None = None
    cal_file = model_path.parent / f"{model_path.stem}.calibration.json"
    if cal_file.is_file():
        try:
            payload = json.loads(cal_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        temps = parse_calibration_payload(payload, mid) if payload is not None else None
        if temps:
            detector.class_temperature = temps
            preds_cal, targets_cal, pairs_cal, _, _ = _collect(detector)
            metrics_cal = detection_metrics(preds_cal, targets_cal, iou_threshold=iou)
            ece_cal = expected_calibration_error(
                [c for c, _ in pairs_cal], [ok for _, ok in pairs_cal]
            )
            metrics, ece = metrics_cal, ece_cal
            calibration_info = {
                "file": str(cal_file),
                "temperatures": {str(k): v for k, v in sorted(temps.items())},
                "active": True,
            }
            if verbose:
                print(f"  校准通道生效（{len(temps)} 类）：ECE {ece_uncal['ece']} → {ece['ece']}")
        elif verbose:
            print(f"  校准表指纹不匹配/非法，按未校准口径评估：{cal_file.name}")

    report: dict[str, Any] = {
        "model": str(model_path),
        "model_id": mid,
        "domain": domain,
        "data_dir": str(data_dir),
        "n_images": n_used,
        "conf": conf,
        "iou": iou,
        "class_conf": {str(k): v for k, v in sorted(_CLASS_CONF.items())},
        "metrics": metrics,
        "ece": {k: ece[k] for k in ("n_samples", "ece", "mce", "verdict")},
        "metrics_uncalibrated": metrics_uncal,
        "ece_uncalibrated": {k: ece_uncal[k] for k in ("ece", "verdict")},
        "calibration": calibration_info,
        "regression": regression,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    eval_dir = ROOT / "data" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    latest = eval_dir / f"deployed_eval_{domain}.json"
    stamp = eval_dir / f"deployed_eval_{domain}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    for out in (latest, stamp):
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    card_dir = ROOT / "data" / "model_cards"
    card_dir.mkdir(parents=True, exist_ok=True)
    per_class_zh = {
        (_CLASS_NAMES_ZH.get(int(cid), cid)): m for cid, m in metrics.get("by_class", {}).items()
    }
    card = build_model_card(
        model_id=mid,
        version=time.strftime("%Y%m%d"),
        metrics={
            "domain": domain,
            "n_images": n_used,
            "mAP50": metrics["mAP50"],
            "recall": metrics["recall"],
            "precision": metrics["precision"],
            "ece": ece["ece"],
            "ece_verdict": ece["verdict"],
            "ece_uncalibrated": ece_uncal["ece"],
            "calibration": calibration_info,
            "per_class": per_class_zh,
            "regression": regression,
        },
        data_summary={
            "source_dir": str(data_dir),
            "domain": domain,
            "n_images": n_used,
            "gt_total": metrics["gt_total"],
            "class_counts": class_counts,
        },
        limitations=_limitations_for(
            domain, metrics, ece, calibration_active=bool(calibration_info)
        ),
        ethics=[
            "评级/评价输出仅为辅助参考，不构成法定无损检测结论（NB/T 47013 授权表未复核）",
            "置信度经 ECE 口径校准评估；低置信/高不确定性检出强制进入人工复核",
        ],
    )
    (card_dir / f"{mid.replace('::', '_')}.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    tracker = ExperimentTracker(ROOT / "data" / "experiments")
    run_id = tracker.start_run(
        "deploy_eval",
        params={
            "model_id": mid,
            "domain": domain,
            "n_images": n_used,
            "conf": conf,
            "iou": iou,
        },
    )
    tracker.log_metrics(
        run_id,
        {
            "mAP50": float(metrics["mAP50"]),
            "recall": float(metrics["recall"]),
            "precision": float(metrics["precision"]),
            "ece": float(ece["ece"]),
        },
    )
    tracker.log_artifact(run_id, str(latest.relative_to(ROOT)))

    if verbose:
        print(f"[post-deploy-eval] model={mid} domain={domain} n={n_used}")
        print(
            f"  mAP50={metrics['mAP50']}  R={metrics['recall']}  "
            f"P={metrics['precision']}  ECE={ece['ece']} (passed={ece['verdict']['passed']})"
        )
        if regression:
            print(f"  regression vs {regression['against']}: passed={regression['passed']}")
        print(f"  report -> {latest.name}  card+tracker 已更新")
    return report


def _limitations_for(
    domain: str,
    metrics: dict[str, Any],
    ece: dict[str, Any],
    calibration_active: bool = False,
) -> list[str]:
    """按评估域如实生成模型卡局限——禁止把合成域指标冒充真实域性能。"""
    lim: list[str] = []
    if domain == "synthetic":
        lim.append(
            "当前指标来自**与训练同源的合成底片**，不代表真实射线底片性能；"
            "data/real_label 恢复后须以真实域重跑评估"
        )
    else:
        lim.append(
            "真实域指标基于 data/real_label 现有标注量，稀有类（裂纹等）样本不足时置信区间宽"
        )
    weak = [
        f"{_CLASS_NAMES_ZH.get(int(cid), cid)}(gt={m['gt_count']})"
        for cid, m in metrics.get("by_class", {}).items()
        if m["gt_count"] < 20
    ]
    if weak:
        lim.append("稀有类 GT 不足 20，逐类 AP 无统计意义：" + "、".join(weak))
    if not ece["verdict"]["passed"]:
        lim.append(
            f"ECE={ece['ece']} 超出门槛 0.05：置信度分流人工复核的阈值语义需重校"
            "（training/fit_calibration 可重新拟合校准表）"
        )
    elif calibration_active:
        lim.append(f"ECE={ece['ece']}（逐类温度校准后达标）；校准表与权重指纹绑定，换权重须重拟合")
    lim.append("部署推理为 ONNX 路径：不确定性为单视角启发式 + TTA 集成近似（非 MC Dropout）")
    return lim


def _load_previous(domain: str) -> dict[str, Any] | None:
    path = ROOT / "data" / "eval" / f"deployed_eval_{domain}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="部署后评估闭环（实测指标+模型卡+实验记录）")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="部署 ONNX 路径")
    ap.add_argument("--data", default=str(DEFAULT_DATA), help="带标注评估集（含 images/ labels/）")
    ap.add_argument("--limit", type=int, default=0, help="最多评估图像数（0=全部）")
    ap.add_argument("--conf", type=float, default=0.25, help="全局置信度阈值")
    ap.add_argument("--iou", type=float, default=0.5, help="NMS/匹配 IoU")
    args = ap.parse_args()
    report = run_post_deploy_eval(
        model_path=Path(args.model),
        data_dir=Path(args.data),
        limit=args.limit,
        conf=args.conf,
        iou=args.iou,
    )
    if report is None:
        raise SystemExit(0)
    if report["regression"] and not report["regression"]["passed"]:
        print("[post-deploy-eval] ⚠️ 与上次评估相比指标退化（详见 regression.violations）")
        raise SystemExit(3)


if __name__ == "__main__":
    main()
