# -*- coding: utf-8 -*-
"""训练模型回归门禁（本地有权重环境专用）。

背景：backend/evaluation/gate.py 的 CI 门禁用 BlobDetector（CI 无 ML 依赖），
只监测基线检测器，监测不到训练权重的退化。本脚本在**有权重的开发/部署机**上
用生产协议（生产 conf/iou/class_conf + 逐类温度校准 + 生产预处理链）对训练
模型跑同一 run_golden_evaluation，与基线做 check_regression 比较。

与 CI 门禁的关系：互补而非替代——CI 继续跑 BlobDetector 保证管线可复现；
本脚本回答「这版权重比基线权重差了吗」，用于重训/回退/激活前的人工把关
（配合 models API 的 activate 门禁使用）。

基线语义：
- 首跑或 --update-baseline：把当前权重的指标写入基线（记录 model_id 指纹）；
- 常规跑：当前权重 vs 基线，任一指标超容差（mAP50/召回/精确，容差同 CI 门禁）
  → 退出码 1；
- 基线由**另一版权重**建立时照常比较（这正是「候选权重激活前验证」的用法），
  但会打印两条 model_id 以便人工确认比较对象。

用法：
  python scripts/eval_gate_trained.py                       # 当前权重 vs 基线
  python scripts/eval_gate_trained.py --update-baseline     # 重建基线
  python scripts/eval_gate_trained.py --weights <path.onnx> # 验证候选权重
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from backend.domain.detect import get_detector
from backend.domain.detect.calibration import parse_calibration_payload
from backend.evaluation.harness import GateResult, check_regression
from backend.evaluation.run_eval import run_golden_evaluation
from backend.infra.config import load_config
from backend.infra.model_registry import ModelRegistry
from backend.infra.paths import resolve_model_uri

DEFAULT_GOLDEN = _ROOT / "data" / "eval" / "golden_v3"
DEFAULT_BASELINE = _ROOT / "data" / "eval" / "baseline_trained.json"


def _class_temperature_for(weight_path: Path):
    """与 dependencies.Registry._class_temperature_for 同语义：指纹不符不启用。"""
    cal_file = _ROOT / "models" / "weights" / "best.calibration.json"
    if not cal_file.is_file():
        return None
    try:
        payload = json.loads(cal_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return parse_calibration_payload(payload, ModelRegistry.entry_id_for(str(weight_path)))


def _build_detector(cfg, weight_path: Path):
    dc = cfg.detect
    return get_detector(
        "trained_yolo",
        model_uri=str(weight_path),
        backend=cfg.model.backend,
        providers=cfg.model.providers,
        tile_size=dc.tile_size,
        tile_overlap=dc.tile_overlap,
        tile_trigger_side=dc.tile_trigger_side,
        tile_max_count=dc.tile_max_count,
        tile_merge_iou=dc.tile_merge_iou,
        class_temperature=_class_temperature_for(weight_path),
    )


def _preprocess_fn(cfg):
    pc = cfg.preprocess
    if not pc.enabled:
        return None
    from backend.domain.preprocess.pipeline import OpencvPreprocessor

    pp = OpencvPreprocessor(
        bilateral_d=pc.bilateral_d, bilateral_sigma_color=pc.bilateral_sigma_color,
        bilateral_sigma_space=pc.bilateral_sigma_space, median_k=pc.median_k,
        clahe_clip=pc.clahe_clip, clahe_grid=pc.clahe_grid,
        canny_kernel=pc.canny_kernel, morph_k_open=pc.morph_k_open,
        morph_k_close=pc.morph_k_close,
    )

    def fn(gray):
        return pp.enhance(pp.denoise(gray), pc.gamma)

    return fn


def main() -> None:
    ap = argparse.ArgumentParser(description="训练模型 Golden 回归门禁（本地权重）")
    ap.add_argument("--golden", default=str(DEFAULT_GOLDEN))
    ap.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    ap.add_argument("--weights", default=None, help="默认取 config.model.default_uri（当前部署权重）")
    ap.add_argument("--update-baseline", action="store_true")
    args = ap.parse_args()

    golden_dir = Path(args.golden)
    meta_path = golden_dir / "_META.json"
    if not meta_path.is_file():
        raise SystemExit(
            f"GATE ERROR: {golden_dir} 缺 _META.json——先用 scripts/make_golden_set_v3.py 生成评估集"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    cfg = load_config()
    weight_path = Path(args.weights) if args.weights else Path(resolve_model_uri(cfg.model.default_uri))
    if not weight_path.is_file():
        raise SystemExit(f"GATE ERROR: 权重不存在: {weight_path}")
    model_id = ModelRegistry.entry_id_for(str(weight_path))

    det = _build_detector(cfg, weight_path)
    work = _ROOT / "data" / "tmp" / "trained_gate_work"
    result = run_golden_evaluation(
        model_id=model_id,
        detector=det,
        golden_dir=str(golden_dir),
        eval_dir=str(work / "eval"),
        experiments_dir=str(work / "experiments"),
        drift_baseline_path=str(work / "drift_baseline.json"),
        conf=cfg.detect.infer_conf,
        iou=cfg.detect.infer_iou,
        class_conf=cfg.detect.class_conf,
        preprocess_fn=_preprocess_fn(cfg),
        spacing_mm=1.0,
    )
    metrics = result["metrics"]

    baseline_path = Path(args.baseline)
    if args.update_baseline or not baseline_path.is_file():
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(
            json.dumps(
                {
                    "model_id": model_id,
                    "weights": str(weight_path),
                    "golden": str(golden_dir),
                    "golden_fingerprint": result.get("golden_fingerprint"),
                    "metrics": metrics,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        gate = GateResult(True, {"mAP50": 0.0, "recall": 0.0, "precision": 0.0}, [])
        mode = "基线已重建"
    else:
        prev = json.loads(baseline_path.read_text(encoding="utf-8"))
        gate = check_regression(metrics, prev["metrics"])
        mode = f"基线权重={prev.get('model_id')}（{prev.get('created_at')}）"

    payload = {
        "model_id": model_id,
        "weights": str(weight_path),
        "golden_version": meta.get("version"),
        "metrics": metrics,
        "gate": {"passed": gate.passed, "deltas": gate.deltas, "violations": gate.violations},
        "mode": mode,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(0 if gate.passed else 1)


if __name__ == "__main__":
    main()
