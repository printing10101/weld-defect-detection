"""评估 / 漂移 / 实验追踪 API。

- POST /api/v1/evaluation/drift：用新数据样本与参考基线比较，返回漂移结果
  （drift/alerts/metrics）。基线缺失（尚未跑过评估）返回 409，提示先评估建立基线。
- GET  /api/v1/evaluation/drift/baseline：读取当前漂移基线（供前端展示参考分布）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.app.dependencies import Registry, get_registry
from backend.evaluation.drift import estimate_drift
from backend.infra.fs import safe_resolve

router = APIRouter(tags=["evaluation"])


class DriftSample(BaseModel):
    class_id: int
    score: float
    L_mm: float | None = None


class DriftRequest(BaseModel):
    samples: list[DriftSample]
    baseline_path: str | None = None  # 缺省用配置中的 drift_baseline_path


class DriftResponse(BaseModel):
    drift: bool
    alerts: list[str]
    metrics: dict[str, float]


@router.post("/evaluation/drift", response_model=DriftResponse)
def evaluate_drift(
    body: DriftRequest,
    reg: Annotated[Registry, Depends(get_registry)],
) -> DriftResponse:
    # 基线必须位于评估目录之内（防任意文件读取）；用户可指定相对名，缺省用配置基线。
    eval_dir = Path(reg.config.eval.drift_baseline_path).resolve().parent
    if body.baseline_path:
        try:
            p = safe_resolve(eval_dir, body.baseline_path)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "INVALID_BASELINE_PATH",
                    "message": "baseline_path 必须位于评估目录内",
                },
            )
    else:
        p = Path(reg.config.eval.drift_baseline_path).resolve()
    if not p.exists():
        raise HTTPException(
            status_code=409,
            detail={
                "code": "NO_BASELINE",
                "message": "漂移基线尚未建立，请先运行一次模型评估（POST /api/v1/models/{id}/evaluate）",
            },
        )
    try:
        baseline: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise HTTPException(
            status_code=500,
            detail={"code": "BASELINE_CORRUPT", "message": "基线文件损坏，无法解析"},
        )

    samples = [s.model_dump() for s in body.samples]
    result = estimate_drift(samples, baseline)
    return DriftResponse(drift=result.drift, alerts=result.alerts, metrics=result.metrics)


@router.get("/evaluation/drift/baseline")
def get_drift_baseline(
    reg: Annotated[Registry, Depends(get_registry)],
) -> dict[str, Any]:
    p = Path(reg.config.eval.drift_baseline_path)
    if not p.exists():
        raise HTTPException(
            status_code=404,
            detail={"code": "NO_BASELINE", "message": "漂移基线尚未建立"},
        )
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise HTTPException(
            status_code=500,
            detail={"code": "BASELINE_CORRUPT", "message": "基线文件损坏，无法解析"},
        )
