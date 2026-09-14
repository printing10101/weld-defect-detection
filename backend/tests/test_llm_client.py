"""``infra.llm_client`` 单测：回环约束、失败降级、响应解析。

不依赖真实服务：用假 HTTP 层注入（monkeypatch ``_request``）验证各分支，
确保"本地大模型不可用绝不抛异常到调用方"这一契约。
"""

from __future__ import annotations

import json

import pytest

from backend.infra.llm_client import LlmClient, parse_base_url


class TestParseBaseUrl:
    def test_loopback_ok(self) -> None:
        assert parse_base_url("http://127.0.0.1:8080") == ("127.0.0.1", 8080)
        assert parse_base_url("http://localhost:1234/") == ("localhost", 1234)

    @pytest.mark.parametrize(
        "url",
        [
            "https://127.0.0.1:8080",  # 非 http
            "http://api.openai.com",  # 非回环（涉密信息不得出机）
            "http://192.168.1.10:8080",  # 非回环
            "http://",  # 缺主机
            "",
        ],
    )
    def test_reject_non_loopback(self, url: str) -> None:
        with pytest.raises(ValueError):
            parse_base_url(url)


class TestChat:
    def test_unreachable_returns_not_ok(self) -> None:
        """端点不可达 → ok=False + reason，绝不抛异常（评片主链路不能被拖垮）。"""
        client = LlmClient("http://127.0.0.1:9")  # 9 = discard 端口，必然连不上
        result = client.chat("sys", "user")
        assert result.ok is False
        assert result.reason

    def test_non_loopback_returns_not_ok(self) -> None:
        client = LlmClient("http://10.0.0.1:8080")
        result = client.chat("sys", "user")
        assert result.ok is False
        assert "回环" in result.reason

    def test_parses_openai_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = LlmClient("http://127.0.0.1:8080", model="m1")
        payload = {
            "model": "m1",
            "choices": [{"message": {"role": "assistant", "content": "  结论文本  "}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        monkeypatch.setattr(client, "_request", lambda *a, **k: (200, payload))
        result = client.chat("sys", "user")
        assert result.ok is True
        assert result.text == "结论文本"  # 首尾空白已清理
        assert result.model == "m1"
        assert result.completion_tokens == 5

    @pytest.mark.parametrize(
        "payload,keyword",
        [
            ({"choices": []}, "choices"),
            ({"choices": [{"message": {"content": "   "}}]}, "空内容"),
            ({"choices": [{}]}, "空内容"),
        ],
    )
    def test_malformed_payload_not_ok(
        self, monkeypatch: pytest.MonkeyPatch, payload: dict, keyword: str
    ) -> None:
        client = LlmClient("http://127.0.0.1:8080")
        monkeypatch.setattr(client, "_request", lambda *a, **k: (200, payload))
        result = client.chat("sys", "user")
        assert result.ok is False
        assert keyword in result.reason

    def test_http_error_status_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = LlmClient("http://127.0.0.1:8080")
        monkeypatch.setattr(client, "_request", lambda *a, **k: (500, "boom"))
        result = client.chat("sys", "user")
        assert result.ok is False
        assert "500" in result.reason

    def test_available_requires_models(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """可用判据看 /v1/models（防"活着但用不了"的假绿，与 llm_server 同口径）。"""
        client = LlmClient("http://127.0.0.1:8080")
        monkeypatch.setattr(client, "_request", lambda *a, **k: (200, {"data": []}))
        assert client.available() is False
        monkeypatch.setattr(
            client, "_request", lambda *a, **k: (200, {"data": [{"id": "qwen3-4b"}]})
        )
        assert client.available() is True

    def test_resolve_model_falls_back_to_first(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = LlmClient("http://127.0.0.1:8080")
        monkeypatch.setattr(
            client, "_request", lambda *a, **k: (200, {"data": [{"id": "a"}, {"id": "b"}]})
        )
        assert client.resolve_model() == "a"

    def test_list_models_handles_garbage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = LlmClient("http://127.0.0.1:8080")
        monkeypatch.setattr(client, "_request", lambda *a, **k: (200, "not json dict"))
        assert client.list_models() == []

    def test_request_body_is_valid_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """请求体必须是可解析 JSON（中文提示词不得破坏编码）。"""
        captured: dict = {}
        client = LlmClient("http://127.0.0.1:8080", model="m1")

        def fake_request(method: str, path: str, body: dict | None = None):
            captured["method"] = method
            captured["path"] = path
            captured["body"] = body
            return 200, {"choices": [{"message": {"content": "ok"}}]}

        monkeypatch.setattr(client, "_request", fake_request)
        client.chat("系统", "用户")
        assert captured["method"] == "POST"
        assert captured["path"] == "/v1/chat/completions"
        assert json.dumps(captured["body"], ensure_ascii=False)
        assert captured["body"]["messages"][0]["content"] == "系统"
