"""本地大模型（OpenAI 兼容）HTTP 客户端。

职责边界：本模块**只做 HTTP I/O**——连接本机 llama.cpp / Ollama / LM Studio
等 OpenAI 兼容端点，收发 ``/v1/models`` 与 ``/v1/chat/completions``。评片结论
的提示词构造与输出清理属业务逻辑，在 ``domain/narrative.py``（infra 不 import
业务判定模块，见 pyproject.toml 的 import-linter 契约）。

安全约束（与 ``llm_server`` 同口径）：
- 仅接受 http + **回环** host：不引入 TLS/远端语义，"把评片内容发到外部大模型"
  是被明确排除的拓扑（涉密影像衍生信息不得出机）。
- 失败一律降级为 ``LlmChatResult(ok=False, reason=...)``，**不抛异常**——调用方
  在评片主链路上，本地大模型不可用绝不能阻断评片。

连接失败/超时/非 200/响应体不合法，全部收敛为 ok=False + 可读 reason，供前端与
审计如实展示"为什么没有 AI 结论"。
"""

from __future__ import annotations

import http.client
import json
import logging
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

_LOG = logging.getLogger("scandetection.llm.client")

# 默认超时：本地 4B 模型单轮结论生成（~400 token 输出）在 CPU 上也可能数十秒，
# 故默认给足；调用方按机型/模型可调。
_DEFAULT_TIMEOUT_SEC = 120.0
_DEFAULT_MAX_TOKENS = 768
_DEFAULT_TEMPERATURE = 0.2


@dataclass(frozen=True)
class LlmChatResult:
    """一次对话调用结果（永不抛错，失败以 ok=False + reason 表达）。"""

    ok: bool
    text: str = ""
    model: str = ""
    reason: str = ""
    elapsed_ms: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


def _loopback_host(host: str) -> bool:
    """仅回环（与 infra.llm_server.is_loopback_host 同语义，此处独立实现避免耦合）。"""
    return host in {"127.0.0.1", "localhost", "::1", "0:0:0:0:0:0:0:1"}


def parse_base_url(base_url: str) -> tuple[str, int]:
    """解析 ``http://127.0.0.1:8080`` → ``(host, port)``；非法或非回环抛 ValueError。"""
    parts = urlsplit((base_url or "").strip())
    if parts.scheme != "http":
        raise ValueError(f"端点仅支持 http://（本地服务），收到: {base_url!r}")
    host = parts.hostname or ""
    if not host:
        raise ValueError(f"端点缺少主机名: {base_url!r}")
    if not _loopback_host(host):
        raise ValueError(f"仅允许回环端点（本地模型），拒绝: {host!r}")
    if host == "::1":
        host = "127.0.0.1"
    port = parts.port or 80
    return host, int(port)


class LlmClient:
    """OpenAI 兼容端点的最小客户端（仅回环、仅所需两个接口）。"""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT_SEC,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        temperature: float = _DEFAULT_TEMPERATURE,
    ) -> None:
        self._base_url = (base_url or "").rstrip("/")
        self._api_key = (api_key or "").strip() or None
        self._model = (model or "").strip() or None
        self._timeout = float(timeout)
        self._max_tokens = int(max_tokens)
        self._temperature = float(temperature)

    # ---- 内部 HTTP ---------------------------------------------------------
    def _request(
        self, method: str, path: str, body: dict | None = None
    ) -> tuple[int | None, object]:
        """发一次请求，返回 ``(status, parsed_json_or_raw_text)``；连不上返回 (None, reason)。"""
        try:
            host, port = parse_base_url(self._base_url)
        except ValueError as exc:
            return None, str(exc)
        conn: http.client.HTTPConnection | None = None
        try:
            conn = http.client.HTTPConnection(host, port, timeout=self._timeout)
            headers = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            payload = json.dumps(body, ensure_ascii=False).encode("utf-8") if body else None
            conn.request(method, path, body=payload, headers=headers)
            resp = conn.getresponse()
            raw = resp.read().decode("utf-8", errors="replace")
            if resp.status != 200:
                # 401 单独给可读原因：正是「本地服务活着但客户端用不了」的典型形态
                hint = "（端点要求鉴权，请核对 api_key_env 配置）" if resp.status == 401 else ""
                return resp.status, f"HTTP {resp.status}{hint}: {raw[:200]}"
            try:
                return resp.status, json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, raw
        except OSError as exc:
            return None, f"端点不可达: {exc.__class__.__name__}: {exc}"
        finally:
            if conn is not None:
                conn.close()

    # ---- 公开接口 ---------------------------------------------------------
    def list_models(self) -> list[str]:
        """``GET /v1/models`` 的模型 id 列表；失败返回空列表。"""
        status, data = self._request("GET", "/v1/models")
        if status != 200 or not isinstance(data, dict):
            return []
        items = data.get("data")
        if not isinstance(items, list):
            return []
        return [str(m.get("id", "")) for m in items if isinstance(m, dict) and m.get("id")]

    def resolve_model(self) -> str:
        """显式配置优先；否则取端点首个模型 id（llama-server 为权重文件名）。"""
        if self._model:
            return self._model
        models = self.list_models()
        return models[0] if models else ""

    def available(self) -> bool:
        """端点可用判据 = ``/v1/models`` 返回 200 且有模型（不看 /health，防假绿）。"""
        return bool(self.list_models())

    def chat(self, system: str, user: str) -> LlmChatResult:
        """单轮对话。任何失败都以 ``ok=False`` 返回，**不抛异常**。"""
        t0 = time.monotonic()
        model = self.resolve_model()
        body: dict = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "stream": False,
        }
        if model:
            body["model"] = model
        status, data = self._request("POST", "/v1/chat/completions", body)
        elapsed = int((time.monotonic() - t0) * 1000)
        if status != 200 or not isinstance(data, dict):
            # 状态码必须出现在原因里：否则运维只看到 "boom" 无法判断是 500 还是 401
            if status is None:
                reason = str(data)
            else:
                detail = data if isinstance(data, str) else str(data)
                reason = f"HTTP {status}: {detail}"[:300]
            _LOG.warning("本地大模型调用失败: %s", reason)
            return LlmChatResult(ok=False, reason=reason, elapsed_ms=elapsed)
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return LlmChatResult(
                ok=False, reason="响应缺少 choices", model=model, elapsed_ms=elapsed
            )
        # 显式标注：字面量 {} 会让类型推断把值类型判成 None，触发 reportOptionalMemberAccess
        first: dict[str, Any] = choices[0] if isinstance(choices[0], dict) else {}
        msg_raw = first.get("message")
        message: dict[str, Any] = msg_raw if isinstance(msg_raw, dict) else {}
        text = str(message.get("content") or "").strip()
        if not text:
            return LlmChatResult(ok=False, reason="模型返回空内容", model=model, elapsed_ms=elapsed)
        usage_raw = data.get("usage")
        usage: dict[str, Any] = usage_raw if isinstance(usage_raw, dict) else {}
        return LlmChatResult(
            ok=True,
            text=text,
            model=str(data.get("model") or model),
            elapsed_ms=elapsed,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )


__all__ = ["LlmChatResult", "LlmClient", "parse_base_url"]
