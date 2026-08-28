"""可观测性基础设施测试（架构升级：进程内指标 + 结构化日志）。

覆盖：
- MetricsRegistry 计数/直方图聚合/快照；
- Timer 上下文管理器自动记录耗时；
- /api/v1/metrics 端点返回合法结构与 HTTP 层指标；
- JsonFormatter 输出单行合法 JSON 且 extra 并入顶层。

这些测试触发真实 TestClient 请求，验证 MetricsMiddleware 也确实接入（HTTP 计数）。
"""

from __future__ import annotations

import json
import logging
import time

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.infra.logging import JsonFormatter
from backend.infra.metrics import MetricsRegistry, get_metrics


def test_counter_and_snapshot() -> None:
    m = MetricsRegistry()
    m.inc("my_counter")
    m.inc("my_counter", labels={"kind": "a"})
    m.reset()
    m.inc("my_counter")
    m.inc("my_counter", labels={"kind": "a"})
    snap = m.snapshot()
    by_labels = {
        item["labels"].get("kind", ""): item["value"] for item in snap["counters"]["my_counter"]
    }
    assert by_labels == {"": 1, "a": 1}


def test_observe_histogram_aggregates() -> None:
    m = MetricsRegistry()
    m.observe("latency", 0.2, labels={"route": "detect"})
    m.observe("latency", 0.8, labels={"route": "detect"})
    agg = m.snapshot()["histograms"]["latency"][0]
    assert agg["count"] == 2
    assert abs(agg["sum"] - 1.0) < 1e-9
    assert agg["min"] == 0.2
    assert agg["max"] == 0.8


def test_timer_context_records() -> None:
    m = MetricsRegistry()
    with m.timer("op", labels={"kind": "x"}):
        time.sleep(0.01)
    hist = m.snapshot()["histograms"]["op"][0]
    assert hist["count"] == 1
    assert hist["sum"] > 0


def test_disabled_registry_empty() -> None:
    m = MetricsRegistry(enabled=False)
    m.inc("nope")
    snap = m.snapshot()
    assert snap["counters"] == {}
    assert snap["histograms"] == {}


def test_metrics_endpoint_shape() -> None:
    with TestClient(app) as client:
        # 先随意打一发业务请求，确保 HTTP 中间件有数据可报
        client.get("/api/v1/health")
        resp = client.get("/api/v1/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert "ts" in body
    assert "counters" in body
    assert "histograms" in body
    # 中间件接入证明：指标端点快照会记录"此前已完成"的请求
    # （当前 /metrics 请求由中间件在响应后计入，故此处已包含 health 那条）。
    counts = body["counters"].get("http_requests_total", [])
    total = sum(item["value"] for item in counts)
    assert total >= 1  # health 探测请求已被采集
    route_labels = {item["labels"].get("route") for item in counts}
    assert "api" in route_labels
    assert "http_request_duration_seconds" in body["histograms"]


def test_json_formatter_single_line_with_extra() -> None:
    fmt = JsonFormatter()
    rec = logging.LogRecord(
        name="scandetection",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hi",
        args=(),
        exc_info=None,
    )
    rec.model_uri = "models/weights/best.onnx"  # extra 应并入顶层
    out = fmt.format(rec)
    parsed = json.loads(out)  # 必须是单行合法 JSON
    assert parsed["logger"] == "scandetection"
    assert parsed["level"] == "INFO"
    assert parsed["message"] == "hi"
    assert parsed["model_uri"] == "models/weights/best.onnx"
    # ensure_ascii=False + separators 紧凑：不应含换行
    assert "\n" not in out


def test_metrics_singleton_is_stable() -> None:
    assert get_metrics() is get_metrics()
