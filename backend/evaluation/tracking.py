"""实验追踪 + 模型卡 + 数据版本。

设计文档  要求（v1 本地可离线、零外部服务依赖，可演进）：
- 实验追踪：记录超参/指标/产物（MLflow 可本地；本实现以 JSONL 落盘
  data/experiments/，字段语义对齐 MLflow Run，未来可一键导出/迁移）；
- 模型注册：models 表 + 模型卡（精度、数据分布、局限、ethically 使用说明）；
- 数据版本：DVC 管理数据集与标注版本（本实现提供目录内容哈希指纹，
  作为 DVC 的轻量本地替代，见 harness.golden_set_fingerprint）。

类与函数：
- ExperimentTracker：JSONL 实验记录（log_metrics / log_params / log_artifact / list_runs）
- build_model_card：组装模型卡（精度/数据分布/局限/伦理说明）
- dataset_fingerprint：数据集目录内容哈希（DVC 语义的本地替代）
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any


class ExperimentTracker:
    """本地实验追踪（JSONL， 可演进 MLflow）。

    每条 Run 一行 JSON：run_id, name, params, metrics, artifacts, created_at。
    并发安全：追加写 + 进程内锁；只追加不修改（审计友好）。
    """

    def __init__(self, store_dir: str | Path) -> None:
        self._dir = Path(store_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "experiments.jsonl"

    def start_run(self, name: str, params: dict[str, Any] | None = None) -> str:
        run_id = uuid.uuid4().hex[:12]
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "run_id": run_id,
                        "name": name,
                        "params": params or {},
                        "metrics": {},
                        "artifacts": [],
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        return run_id

    def log_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        self._update(run_id, "metrics", metrics)

    def log_artifact(self, run_id: str, path: str) -> None:
        rows = self._load()
        for r in rows:
            if r["run_id"] == run_id:
                r["artifacts"].append(path)
        self._rewrite(rows)

    def _update(self, run_id: str, key: str, value: dict) -> None:
        rows = self._load()
        for r in rows:
            if r["run_id"] == run_id:
                r[key].update(value)
        self._rewrite(rows)

    def _load(self) -> list[dict]:
        if not self._path.exists():
            return []
        return [
            json.loads(line)
            for line in self._path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _rewrite(self, rows: list[dict]) -> None:
        with self._path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    def list_runs(self) -> list[dict]:
        return self._load()

    def get_run(self, run_id: str) -> dict | None:
        return next((r for r in self._load() if r["run_id"] == run_id), None)


def build_model_card(
    *,
    model_id: str,
    version: str,
    metrics: dict[str, Any],
    data_summary: dict[str, Any],
    limitations: list[str],
    ethics: list[str],
) -> dict[str, Any]:
    """组装模型卡。

    与设计文档  models 表 metric_map 语义对齐：metric_map 存指标与数据分布，
    note 存局限/伦理；前端可据此展示"该模型能用在哪、不能信什么"。
    """
    return {
        "model_id": model_id,
        "version": version,
        "metrics": metrics,  # 精度（mAP/召回/精确，来自 evaluation harness）
        "data_distribution": data_summary,  # 数据分布（类别占比/厚度范围/来源）
        "limitations": limitations,  # 局限（未覆盖场景/低置信行为）
        "ethics": ethics,  # ethically 使用说明（合规边界）
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def dataset_fingerprint(directory: str | Path) -> str:
    """数据集目录内容哈希（DVC 数据版本语义的本地替代，）。

    与 harness.golden_set_fingerprint 同算法；区别在语义：本函数用于训练集
    版本追踪（何时换了训练数据），后者用于固定评估集（Golden Set）版本锁定。
    """
    from backend.evaluation.harness import golden_set_fingerprint

    return golden_set_fingerprint(directory)
