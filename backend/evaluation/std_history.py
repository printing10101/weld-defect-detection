"""评价历史档案聚合（E-15）。

把散落的评价产物聚合成统一时间线，供等级曲线（GET /std-eval/history）与
稳定性测试报告引用：

- data/eval/std_eval.json          ：run_std_eval CLI 产出（DB50/T 1807 分级：
                                     TDR/WDR/KDR/FRR + level）；
- data/eval/{model_id}__ver.json   ：Golden 评估报告（harness.save_eval_report，
                                     mAP/recall/precision + Golden 指纹）；
- data/eval/std_record*.json       ：附录A 记录表（POST /std-eval/record 落盘）；
- DB 记录（可选注入）              ：调用方传入的 DB 侧评价记录（如报告评级行）。

输出统一条目形状：{evaluated_at, model_version, level, tdr, wdr, frr,
map50, recall, source}，缺省字段为 None；按 evaluated_at 降序。
解析失败的单个文件跳过（best-effort 聚合，不因单个坏档拖垮整个历史）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _entry(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "evaluated_at": None,
        "model_version": None,
        "level": None,
        "tdr": None,
        "wdr": None,
        "frr": None,
        "map50": None,
        "recall": None,
        "source": None,
    }
    base.update({k: v for k, v in kw.items() if v is not None})
    return base


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _from_std_eval(payload: dict[str, Any]) -> dict[str, Any] | None:
    """run_std_eval 产出：{generated_at, result: {standard: {...}, strict: {...}}}。"""
    result = payload.get("result")
    if not isinstance(result, dict):
        return None
    std = result.get("standard") or {}
    return _entry(
        evaluated_at=payload.get("generated_at"),
        level=std.get("level"),
        tdr=std.get("tdr"),
        wdr=std.get("wdr"),
        frr=std.get("frr"),
        source="std_eval",
    )


def _from_golden_report(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Golden 评估报告（harness.save_eval_report）：{model_id, metrics, created_at?}。"""
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or "mAP50" not in metrics:
        return None
    return _entry(
        evaluated_at=payload.get("created_at") or payload.get("generated_at"),
        model_version=payload.get("model_id"),
        map50=metrics.get("mAP50"),
        recall=metrics.get("recall"),
        source="golden_eval",
    )


def _from_std_record(payload: dict[str, Any]) -> dict[str, Any] | None:
    """附录A 记录表（POST /std-eval/record 落盘）：含 standard/strict 指标摘要。"""
    std = payload.get("standard")
    if not isinstance(std, dict):
        return None
    return _entry(
        evaluated_at=payload.get("generated_at") or payload.get("evaluated_at"),
        model_version=payload.get("model_version"),
        level=payload.get("level") or std.get("level"),
        tdr=std.get("tdr"),
        wdr=std.get("wdr"),
        frr=std.get("frr"),
        source="std_record",
    )


def collect_history(
    eval_dir: str | Path,
    db_records: list[dict[str, Any]] | None = None,
    *,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """聚合评价历史（按 evaluated_at 降序，limit 截断）。

    db_records：DB 侧评价记录（调用方装配，形状同输出条目，source 建议标
    明来源如 "db_report"），原样并入统一排序。
    """
    root = Path(eval_dir)
    entries: list[dict[str, Any]] = []
    if root.is_dir():
        for f in sorted(root.glob("*.json")):
            payload = _read_json(f)
            if payload is None:
                continue
            entry = None
            name = f.name.lower()
            if "result" in payload and isinstance(payload.get("result"), dict):
                entry = _from_std_eval(payload)
            elif "metrics" in payload and isinstance(payload.get("metrics"), dict):
                entry = _from_golden_report(payload)
            elif name.startswith("std_record"):
                entry = _from_std_record(payload)
            if entry is not None:
                entries.append(entry)
    for rec in db_records or []:
        entries.append(_entry(**rec))
    # 降序排序（缺时间戳的排最后，稳定排序保证同键顺序确定）。
    entries.sort(key=lambda e: e.get("evaluated_at") or "", reverse=True)
    return entries[: max(0, limit)]
