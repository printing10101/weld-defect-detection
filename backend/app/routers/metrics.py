"""可观测性导出（GET /api/v1/metrics，架构升级：进程内指标）。

返回进程内指标快照（JSON, content-type application/json）：
- counters: 计数型（http_requests_total 等）
- histograms: 直方图聚合（http_request_duration_seconds 的 count/sum/min/max）

边界：导出端点是可观测的唯一出口；接入 Prometheus/OTEL 时代替的是"如何序列化"
这一层，业务采集点（get_metrics.inc/observe）保持不变。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from backend.app.dependencies import Registry, try_get_registry
from backend.infra.metrics import get_metrics
from backend.infra.timeutil import fmt_naive_utc

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
def metrics_endpoint(
    request: Request, reg: Annotated[Registry | None, Depends(try_get_registry)]
) -> JSONResponse:
    """导出指标快照。reg 未就绪时仅返回已采集的 HTTP 层指标（启动期也可观测）。"""
    del reg  # 仅用于保持与其它健康类端点一致的依赖签名；指标采集不依赖业务装配
    metrics = get_metrics()
    snapshot = metrics.snapshot()
    return JSONResponse(
        {
            "ts": fmt_naive_utc(),
            "metrics_enabled": metrics.enabled,
            "counters": snapshot["counters"],
            "histograms": snapshot["histograms"],
        },
        media_type="application/json",
    )
