"""评估回归门禁。

提供 `run_eval_gate`：用固定 Golden Set + BlobDetector（零训练、无 ML 依赖，
CI 可跑）跑 `run_golden_evaluation`，再与已提交的基线 (baseline.json) 做
`check_regression` 比较。mAP@0.5 / 召回 / 精确任一退化超容差 → 门禁失败。

目的：把"检测 → 指标 → 报告 → 漂移 → 跟踪 → 回归"全链路锁定为强制回归门禁，
任何静默破坏（预处理、检测、harness 聚合、报告落盘）都会被 PR 阻断。

为什么用 BlobDetector 而非训练 YOLO：
- 真实 X 光底片与标准授权文本缺失（用户明确"真实授权文本拿不到"），Golden Set
  只能走合成数据；
- 训练 YOLO 需权重 + torch/onnxruntime，CI 默认无 ML 依赖。BlobDetector 为纯 CV
  （opencv/skimage），可在 CI 无 ML 环境真实跑通检测流水线，提供可复现基准；
- 真实 YOLO 模型评估走同一 `run_golden_evaluation`（--backend yolo），不阻塞 CI。

Golden Set 为合成缺陷图（暗团块 + class 0 真值），确定性生成
（scripts/make_golden_set.py），规避授权与真实数据约束，同时提供可复现回归基准。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.domain.detect.blob_detector import BlobConfig, BlobDetector
from backend.evaluation.harness import GateResult, check_regression
from backend.evaluation.run_eval import run_golden_evaluation

GOLDEN_MODEL_ID = "baseline::blob"
DEFAULT_CONF = 0.3
DEFAULT_IOU = 0.5


def run_eval_gate(
    *,
    golden_dir: str | Path,
    baseline_path: str | Path,
    work_dir: str | Path,
    model_id: str = GOLDEN_MODEL_ID,
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    class_conf: dict[int, float] | None = None,
    update_baseline: bool = False,
) -> tuple[GateResult, dict[str, Any]]:
    """跑 Golden Set 评估并与基线比较，返回 (门禁结果, 当前指标)。

    - work_dir：评估/实验/漂移基线等中间产物的落盘处（建议用临时目录）。
    - baseline_path 缺失或 update_baseline=True：仅建立基线并写出，门禁视为通过。
    - 否则与已提交基线比较，任一指标超容差即 GateResult.passed=False。
    """
    work = Path(work_dir)
    eval_dir = work / "eval"
    experiments_dir = work / "experiments"
    drift_baseline_path = work / "drift_baseline.json"

    # BlobDetector 纯 CV、无权重、无 ML 依赖；dark_only 仅检暗缺陷，
    # 与 Golden Set 的 class 0 暗团块真值对齐，使召回/mAP 有意义。
    detector = BlobDetector(BlobConfig(dark_only=True))
    detector.load("baseline://gate", "blob")

    result = run_golden_evaluation(
        model_id,
        detector,
        golden_dir=golden_dir,
        eval_dir=eval_dir,
        experiments_dir=experiments_dir,
        drift_baseline_path=drift_baseline_path,
        conf=conf,
        iou=iou,
        class_conf=class_conf,
        spacing_mm=1.0,
    )
    metrics = result["metrics"]

    baseline_path = Path(baseline_path)
    if update_baseline or not baseline_path.exists():
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        gate = GateResult(
            passed=True,
            deltas={"mAP50": 0.0, "recall": 0.0, "precision": 0.0},
            violations=[],
        )
        return gate, metrics

    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"基线文件损坏或无法读取: {baseline_path} ({exc})") from exc

    gate = check_regression(metrics, baseline)
    return gate, metrics
