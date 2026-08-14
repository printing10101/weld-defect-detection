"""P2-6：HttpSyncAdapter 测试（§7.6 端边云同步适配器选择）。

- HttpSyncAdapter 必须带 endpoint 构造，否则 ValueError；
- push 在网络不可达时本地留档、不抛（尽力而为）；
- LocalAdapter 行为不变（数据不出本机）。
"""

from __future__ import annotations

import pytest

from backend.domain.sync import HttpSyncAdapter, LocalAdapter


def test_http_adapter_requires_endpoint() -> None:
    with pytest.raises(ValueError):
        HttpSyncAdapter(endpoint=None)


def test_http_adapter_local_fallback_on_network_failure(tmp_path) -> None:
    queue = tmp_path / "pending.jsonl"
    # 指向不可达端点：push 应捕获异常、本地留档、不抛
    adapter = HttpSyncAdapter(
        endpoint="http://127.0.0.1:9/unreachable",
        token=None,
        queue_path=queue,
    )
    assert adapter.name == "http"
    adapter.push({"image_id": "x", "level": "II"})
    assert queue.exists()
    assert adapter.pending_count == 1


def test_local_adapter_unchanged(tmp_path) -> None:
    queue = tmp_path / "pending.jsonl"
    a = LocalAdapter(queue)
    assert a.name == "local"
    a.push({"k": 1})
    assert a.pending_count == 1
    assert a.pull() == []
