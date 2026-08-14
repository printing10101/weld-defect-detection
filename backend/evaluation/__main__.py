"""评估回归门禁 CLI：python -m backend.evaluation [options]

跑 Golden Set 评估并与提交基线比较。退出码 0=通过，1=回归/错误。
CI 步骤：python -m backend.evaluation --golden data/eval/golden --baseline data/eval/baseline.json
本地重建基线：python -m backend.evaluation --update-baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from backend.evaluation.gate import GOLDEN_MODEL_ID, run_eval_gate

# 仓库根：backend/evaluation/__main__.py -> parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="射线焊缝缺陷检测评估回归门禁")
    parser.add_argument(
        "--golden",
        default=str(_REPO_ROOT / "data" / "eval" / "golden"),
        help="Golden Set 目录（images/ + labels/）",
    )
    parser.add_argument(
        "--baseline",
        default=str(_REPO_ROOT / "data" / "eval" / "baseline.json"),
        help="已提交基线指标文件（mAP/召回/精确）",
    )
    parser.add_argument(
        "--work-dir",
        default=str(_REPO_ROOT / "data" / "eval" / ".gate_work"),
        help="中间产物落盘目录（评估/实验/漂移基线）",
    )
    parser.add_argument("--model-id", default=GOLDEN_MODEL_ID)
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="重建并提交基线（仅在有意提升指标后使用）",
    )
    args = parser.parse_args(argv)

    try:
        gate, metrics = run_eval_gate(
            golden_dir=args.golden,
            baseline_path=args.baseline,
            work_dir=args.work_dir,
            model_id=args.model_id,
            update_baseline=args.update_baseline,
        )
    except FileNotFoundError as exc:
        print(f"EVAL GATE ERROR: Golden Set 缺失: {exc}", file=sys.stderr)
        return 1

    payload = {
        "metrics": metrics,
        "gate": {
            "passed": gate.passed,
            "deltas": gate.deltas,
            "violations": gate.violations,
        },
        "baseline_updated": args.update_baseline,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if gate.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
