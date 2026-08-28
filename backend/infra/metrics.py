"""轻量进程内指标注册表（可观测性基础设施，无第三方依赖）。

为"未来接 Prometheus / OpenTelemetry"预留的边界：
- 业务代码只调用 `get_metrics().inc / observe`，不感知导出方式；
- 导出端点是唯一出口（`/api/v1/metrics`，见 app/routers/metrics.py）；
- 替换实现（如接入 OTEL）只改本模块与导出端点，不动业务调用点。

Thread-safe：计数器/直方图聚合均在锁内更新。
直方图只保留聚合量（count/sum/min/max），不保存原始样本，内存恒定、
支持任意长期运行（十年级进程无累积泄漏），代价是无法精确分位点——
对本地单机可观测（定位 qps/时延/偶发错误）足够；若需精确分位再接时序库。
"""

from __future__ import annotations

import threading
import time
from typing import Any, Self

# 单例访问：independent of app registry，构造中间件/端点时可即取即用。
_METRICS: MetricsRegistry | None = None
_METRICS_LOCK = threading.Lock()


class MetricsRegistry:
    """进程内指标注册表（计数器 + 直方图聚合）。"""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._lock = threading.Lock()
        # 计数器：name -> {标签键值元组 -> 累计值}
        self._counters: dict[str, dict[tuple[tuple[str, str], ...], int]] = {}
        # 直方图聚合：name -> {标签元组 -> {count,sum,min,max}}
        self._hist = {}

    # -- 写入 ------------------------------------------------------------
    def inc(self, name: str, value: int = 1, labels: dict[str, str] | None = None) -> None:
        """累加计数器（默认 +1）。value 应 >= 0。"""
        if not self.enabled:
            return
        lab = _frozen(labels)
        with self._lock:
            bucket = self._counters.setdefault(name, {})
            bucket[lab] = bucket.get(lab, 0) + value

    def observe(self, name: str, seconds: float, labels: dict[str, str] | None = None) -> None:
        """记录一次耗时观测（秒），并入直方图聚合。"""
        if not self.enabled:
            return
        lab = _frozen(labels)
        with self._lock:
            bucket = self._hist.setdefault(name, {})
            agg = bucket.get(lab)
            if agg is None:
                bucket[lab] = {"count": 1, "sum": seconds, "min": seconds, "max": seconds}
            else:
                agg["count"] += 1
                agg["sum"] += seconds
                agg["min"] = min(agg["min"], seconds)
                agg["max"] = max(agg["max"], seconds)

    def timer(self, name: str, labels: dict[str, str] | None = None) -> Timer:
        """返回一个上下文管理器，with 块结束自动记录耗时。"""
        return Timer(self, name, labels)

    # -- 读取 ------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        """导出当前全部指标（快照，线程安全；JSON 友好：字典键均为字符串）。

        结构：
          counters  {名称: [ {labels:{...}, value:int} ... ]}
          histograms{名称: [ {labels:{...}, count,sum,min,max} ... ]}
        """
        if not self.enabled:
            return {"counters": {}, "histograms": {}}
        with self._lock:
            return {
                "counters": {
                    name: [{"labels": _unfrozen(k), "value": v} for k, v in sorted(bucket.items())]
                    for name, bucket in sorted(self._counters.items())
                },
                "histograms": {
                    name: [{"labels": _unfrozen(k), **agg} for k, agg in sorted(bucket.items())]
                    for name, bucket in sorted(self._hist.items())
                },
            }

    def reset(self) -> None:
        """清空全部指标（测试/生命周期重置用）。"""
        with self._lock:
            self._counters.clear()
            self._hist.clear()


class Timer:
    """with 语义的耗时记录器。"""

    __slots__ = ("_labels", "_name", "_registry", "_start")

    def __init__(self, registry: MetricsRegistry, name: str, labels: dict[str, str] | None) -> None:
        self._registry = registry
        self._name = name
        self._labels = labels
        self._start = time.perf_counter()

    def __enter__(self) -> Self:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self._registry.observe(self._name, time.perf_counter() - self._start, self._labels)


def _frozen(labels: dict[str, str] | None) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((labels or {}).items()))


def _unfrozen(lab: tuple[tuple[str, str], ...]) -> dict[str, str]:
    return dict(lab)


class MetricsMiddleware:
    """ASGI 中间件：按路由组（URI 首段）+ 状态码统计请求数/耗时。

    加入 app 中间件栈即可零侵入采集 HTTP 可观测性：
    - http_requests_total{route,method,status}
    - http_request_duration_seconds{route}
    5xx 另行以 http_errors_total 计数（便于告警/定位偶发失败）。
    """

    def __init__(self, app: Any, metrics: MetricsRegistry | None = None) -> None:
        self.app = app
        self.metrics: MetricsRegistry = metrics or get_metrics()

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path") or "/"
        # 路由组 = URI 首段（/api/v1/<group>/... -> <group>；根路径记 root）
        route = path.strip("/").split("/", 1)[0] or "root"
        # 仅统计 API 命中，静态 SPA 资源（/assets 等）避免噪声；见注释
        if route not in {"api", "root"}:
            # 静态前端/SPA 请求也低开销计入 route 组，但单独归类避免污染业务指标
            route = "static"
        method = (scope.get("method") or "GET").upper()
        start = time.perf_counter()
        status_holder: dict[str, int] = {}

        async def _send(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                status_holder["status"] = int(message.get("status", 0))
            await send(message)

        status = 0
        try:
            await self.app(scope, receive, _send)
        except Exception:
            self.metrics.inc("http_errors_total", labels={"route": route, "method": method})
            raise
        finally:
            status = status_holder.get("status", status)
            dur = time.perf_counter() - start
            labels = {"route": route, "method": method, "status": str(status)}
            self.metrics.inc("http_requests_total", labels=labels)
            self.metrics.observe("http_request_duration_seconds", dur, labels={"route": route})


def get_metrics() -> MetricsRegistry:
    """获取进程级指标单例（懒初始化；调用方持引用即可，勿重复构造）。"""
    global _METRICS
    if _METRICS is None:
        with _METRICS_LOCK:
            if _METRICS is None:
                _METRICS = MetricsRegistry()
    return _METRICS
