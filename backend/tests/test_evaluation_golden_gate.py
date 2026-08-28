"""#4 评估回归门禁集成测试。

用已提交的确定性 Golden Set + 基线 (baseline.json)，以 BlobDetector 跑完整评估
流水线，并断言 check_regression 通过（即当前指标未退化超容差）。

本测试使评估回归门禁成为 pytest 强制门禁的一部分：
- 固定 Golden Set（data/eval/golden，确定性生成）→ 指标可复现；
- 提交基线 (data/eval/baseline.json) → 与实际运行比较；
- 任一指标退化 > 容差（mAP 1.0 点 / 召回 2.0 点 / 精确 2.0 点）→ 阻断合并。

Golden Set / 基线缺失视为门禁环境被破坏（fail），而非 skip，避免静默放行。
"""

from __future__ import annotations

import json
from pathlib import Path

from backend.evaluation.gate import run_eval_gate

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GOLDEN = _REPO_ROOT / "data" / "eval" / "golden"
_BASELINE = _REPO_ROOT / "data" / "eval" / "baseline.json"


def test_eval_gate_passes_on_committed_golden(tmp_path: Path) -> None:
    assert _GOLDEN.is_dir(), f"Golden Set 缺失：{_GOLDEN}（运行 scripts/make_golden_set.py）"
    assert _BASELINE.exists(), (
        f"评估基线缺失：{_BASELINE}（运行 python -m backend.evaluation --update-baseline）"
    )
    gate, metrics = run_eval_gate(
        golden_dir=_GOLDEN,
        baseline_path=_BASELINE,
        work_dir=tmp_path / "gate_work",
    )
    assert metrics["gt_total"] > 0, "Golden Set 应含真值缺陷（gt_total > 0）"
    assert gate.passed, f"评估回归门禁失败：{gate.violations}（deltas={gate.deltas}）"


def test_eval_gate_detects_regression(tmp_path: Path) -> None:
    """回归检测正向校验：把基线指标人为压低，门禁应判定不通过。"""
    assert _GOLDEN.is_dir()
    baseline = {
        "mAP50": 0.95,
        "recall": 0.95,
        "precision": 0.95,
        "gt_total": 999,
        "by_class": {},
    }
    fake_baseline = tmp_path / "fake_baseline.json"
    fake_baseline.write_text(json.dumps(baseline), encoding="utf-8")

    gate, _ = run_eval_gate(
        golden_dir=_GOLDEN,
        baseline_path=fake_baseline,
        work_dir=tmp_path / "gate_work2",
    )
    # 真实 Golden Set 指标远低于 0.95（合成数据+基线 blob 检测器上限），
    # 与人为高基线比较必然触发 mAP/召回/精确 退化告警。
    assert not gate.passed, "门禁应捕获人为压低的基线导致的回归"
    assert gate.violations, "回归时应产生违约描述"
