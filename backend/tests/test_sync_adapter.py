"""P2-6：HttpSyncAdapter 测试。

- HttpSyncAdapter 必须带 endpoint 构造，否则 ValueError；
- push 在网络不可达时本地留档、不抛（尽力而为）；
- push 成功时端点真实收到记录（含 Bearer token）并计数指标；
- LocalAdapter 行为不变（数据不出本机）。
"""

from __future__ import annotations

import json
import threading

import pytest

from backend.domain.sync import CloudAdapter, HttpSyncAdapter, LocalAdapter
from backend.infra.sync_io import JsonlQueue, UrllibJsonPoster


class _CaptureReceiver:
    """极简线程化 HTTP 接收器：记录收到的路径/头/请求体，供成功路径断言。"""

    def __init__(self):
        import http.server

        self._seen: list[dict] = []
        self._lock = threading.Lock()
        # 内部 Handler 通过闭包捕获接收器实例状态（请求体在 handler 线程写入）。
        seen = self._seen
        lock = self._lock

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # 协议方法名，ruff N 类规则未启用，无需 noqa
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                with lock:
                    seen.append(
                        {"path": self.path, "auth": self.headers.get("Authorization"), "body": body}
                    )
                self.send_response(200)
                self.end_headers()

            def log_message(self, format, *args):  # 抑制测试期冗长访问日志（覆写基类签名）
                return

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()

    def requests(self) -> list[dict]:
        with self._lock:
            return list(self._seen)


def test_http_adapter_requires_endpoint() -> None:
    with pytest.raises(ValueError):
        HttpSyncAdapter(endpoint=None)


def test_http_adapter_local_fallback_on_network_failure(tmp_path) -> None:
    queue = tmp_path / "pending.jsonl"
    # 指向不可达端点：push 应捕获异常、本地留档、不抛（IO 经 infra 注入，Task #9）
    adapter = HttpSyncAdapter(
        endpoint="http://127.0.0.1:9/unreachable",
        token=None,
        queue=JsonlQueue(queue),
        transport=UrllibJsonPoster(),
    )
    assert adapter.name == "http"
    adapter.push({"image_id": "x", "level": "II"})
    assert queue.exists()
    assert adapter.pending_count == 1


def test_http_adapter_push_success_hits_endpoint(tmp_path) -> None:
    """成功路径：端点真实收到 JSON 负载、正确路径与 Bearer token，并计数成功指标。"""
    from backend.infra.metrics import get_metrics

    queue = tmp_path / "pending.jsonl"
    get_metrics().reset()
    with _CaptureReceiver() as receiver:
        adapter = HttpSyncAdapter(
            endpoint=f"http://127.0.0.1:{receiver.port}/api/sync",
            token="s3cret",
            queue=JsonlQueue(queue),
            transport=UrllibJsonPoster(),
        )
        adapter.push({"image_id": "img_9", "level": "II"})
        assert adapter.pending_count == 1  # 本地仍留档（双写语义）
        got = receiver.requests()
        assert len(got) == 1
        assert got[0]["path"] == "/api/sync"
        assert got[0]["auth"] == "Bearer s3cret"
        assert json.loads(got[0]["body"]) == {"image_id": "img_9", "level": "II"}

    snap = get_metrics().snapshot()
    success = [
        i
        for i in snap["counters"]["sync_push_total"]
        if i["labels"] == {"adapter": "http", "result": "success"}
    ]
    assert len(success) == 1 and success[0]["value"] == 1


def test_local_adapter_unchanged(tmp_path) -> None:
    queue = tmp_path / "pending.jsonl"
    a = LocalAdapter(JsonlQueue(queue))
    assert a.name == "local"
    a.push({"k": 1})
    assert a.pending_count == 1
    assert a.pull() == []


def test_jsonl_queue_io_extracted_to_infra(tmp_path) -> None:
    """QueuePort IO 外移 infra（Task #9）：JsonlQueue 独立可测（追加/计数/缺失为 0）。"""
    q = JsonlQueue(tmp_path / "sub" / "pending.jsonl")
    assert q.count() == 0  # 文件不存在 → 0
    q.append({"image_id": "a", "level": "II"})
    q.append({"image_id": "b"})
    assert q.count() == 2
    # 持久化真实落盘（可跨实例恢复）
    q2 = JsonlQueue(tmp_path / "sub" / "pending.jsonl")
    assert q2.count() == 2


def test_http_timeout_and_cors_origins_centralized() -> None:
    """ 配置中心化（P2）：HTTP 同步超时与 CORS 源均为配置单一权威。"""
    from backend.infra.config import load_config

    cfg = load_config()
    assert cfg.sync.http_timeout == 10.0
    assert "tauri://localhost" in cfg.server.cors_origins
    assert "https://tauri.localhost" in cfg.server.cors_origins
    # 超时经构造注入传输实现（不再是硬编码 10）
    poster = UrllibJsonPoster(timeout=cfg.sync.http_timeout)
    assert poster.timeout == 10.0
    poster_custom = UrllibJsonPoster(timeout=3.0)
    assert poster_custom.timeout == 3.0


# ---------------------------------------------------------------------------
# P3：v3 CloudAdapter 联邦占位
# ---------------------------------------------------------------------------


def test_cloud_adapter_requires_endpoint() -> None:
    with pytest.raises(ValueError):
        CloudAdapter(endpoint="")


def test_cloud_adapter_placeholder_fail_loud() -> None:
    """v3 占位：push/pull/federate 显式 NotImplementedError，绝不静默假装已同步/已联邦。"""
    adapter = CloudAdapter(endpoint="https://cloud.example/fed", token="tok")
    assert adapter.name == "cloud"
    assert adapter.pending_count == 0
    with pytest.raises(NotImplementedError):
        adapter.push({"image_id": "x"})
    with pytest.raises(NotImplementedError):
        adapter.pull()
    with pytest.raises(NotImplementedError):
        adapter.federate({"layer": 1})


def test_cloud_kind_wired_to_placeholder() -> None:
    """sync.kind=cloud → 装配 CloudAdapter；endpoint 缺失 → 回退 local（不阻断启动）。"""
    from types import SimpleNamespace
    from typing import cast

    from backend.app.dependencies import Registry

    stub = SimpleNamespace(
        config=SimpleNamespace(
            sync=SimpleNamespace(
                kind="cloud", http_endpoint="https://cloud.example/fed", http_token=None
            ),
            paths=SimpleNamespace(data_dir="data"),
        )
    )
    assert Registry._build_syncer(cast(Registry, stub)).name == "cloud"

    stub_bad = SimpleNamespace(
        config=SimpleNamespace(
            sync=SimpleNamespace(kind="cloud", http_endpoint=None, http_token=None),
            paths=SimpleNamespace(data_dir="data"),
        )
    )
    assert Registry._build_syncer(cast(Registry, stub_bad)).name == "local"  # 回退，不阻断启动
