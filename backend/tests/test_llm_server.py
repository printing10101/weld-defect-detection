"""本地大模型（llama-server）随软件启停：LlamaServerManager 单元测试。

不依赖真实 llama.cpp：以"venv 自身 Python + 内联 HTTP 小服务"冒充
llama-server（经 monkeypatch 替换 _build_command），验证状态机、
启停回收、看门狗重启与各降级路径。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from backend.infra.config import LlmCfg
from backend.infra.llm_server import LlamaServerManager, is_loopback_host

# 冒充 llama-server：/health 恒 200 的最小 HTTP 服务（端口取 argv[1]）
_FAKE_SERVER = (
    "import http.server, sys\n"
    "class H(http.server.BaseHTTPRequestHandler):\n"
    "    def do_GET(self):\n"
    "        self.send_response(200)\n"
    "        self.end_headers()\n"
    "    def log_message(self, *a):\n"
    "        pass\n"
    "http.server.HTTPServer(('127.0.0.1', int(sys.argv[1])), H).serve_forever()\n"
)

# 复现 llama.cpp 读到 LLAMA_API_KEY 时的行为：/health 免鉴权仍 200，
# 但 /v1/* 一律 401——正是"状态 ready、调用全废"的假绿场景。
_FAKE_AUTH_SERVER = (
    "import http.server, sys\n"
    "class H(http.server.BaseHTTPRequestHandler):\n"
    "    def do_GET(self):\n"
    "        code = 401 if self.path.startswith('/v1/') else 200\n"
    "        self.send_response(code)\n"
    "        self.end_headers()\n"
    "        self.wfile.write(b'{}')\n"
    "    def log_message(self, *a):\n"
    "        pass\n"
    "http.server.HTTPServer(('127.0.0.1', int(sys.argv[1])), H).serve_forever()\n"
)


def _cfg(tmp_path: Path, port: int, **kw) -> LlmCfg:
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"gguf-stub")
    return LlmCfg(
        model_file=str(model),
        # 缺省用当前解释器顶替 llama 运行时（monkeypatch _build_command 后并不
        # 真正执行它）——否则 managed 路径落到 tools/llama 默认路径，CI 等无
        # 运行时的环境全部退成 unavailable，测试密不可移植
        server_exe=kw.pop("server_exe", sys.executable),
        port=port,
        startup_timeout_sec=kw.pop("startup_timeout_sec", 8.0),
        watch_interval_sec=kw.pop("watch_interval_sec", 0.3),
        **kw,
    )


def _fake_command(server_code: str):
    def _build(self, exe, model):
        return [sys.executable, "-c", server_code, str(self._cfg.port)]

    return _build


def _fake_probe(*, reachable: bool, needs_auth: bool = False, models: list[str] | None = None):
    """构造端点探测结果，避免 external 模式测试依赖真实网络。"""
    from backend.infra.llm_discovery import ServiceProbe

    probe = ServiceProbe(host="127.0.0.1", port=0)
    probe.reachable = reachable
    probe.needs_auth = needs_auth
    probe.models = list(models or [])
    return probe


def _wait_state(mgr: LlamaServerManager, state: str, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if mgr.status()["state"] == state:
            return
        time.sleep(0.1)
    pytest.fail(f"等待状态 {state} 超时，当前: {mgr.status()}")


def test_disabled_config_never_spawns(tmp_path) -> None:
    mgr = LlamaServerManager(_cfg(tmp_path, 18791, enabled=False), tmp_path)
    mgr.start()
    assert mgr.status()["state"] == "disabled"
    assert mgr.status()["enabled"] is False


def test_missing_binary_degrades_without_crash(tmp_path) -> None:
    # 环境不具备（安装版未随包分发运行时）→ unavailable（预期状态），不是 error（真故障）
    cfg = _cfg(tmp_path, 18792, server_exe=str(tmp_path / "no_such_llama-server.exe"))
    mgr = LlamaServerManager(cfg, tmp_path)
    mgr.start()
    st = mgr.status()
    assert st["state"] == "unavailable"
    assert "不存在" in (st["error"] or "")
    assert st["advice"]  # 必须给出可执行的下一步，而不是只报一个状态码


def test_missing_model_degrades_without_crash(tmp_path) -> None:
    cfg = LlmCfg(
        server_exe=sys.executable,  # 存在的文件，过二进制检查
        model_file=str(tmp_path / "missing.gguf"),
        port=18793,
    )
    mgr = LlamaServerManager(cfg, tmp_path)
    mgr.start()
    st = mgr.status()
    assert st["state"] == "unavailable"
    assert "模型文件不存在" in (st["error"] or "")


def test_ready_requires_api_surface_not_just_health(tmp_path, monkeypatch) -> None:
    """回归：/health 200 但 /v1/* 401 时**不得**报 ready（假绿防护）。

    这正是 llama.cpp 读到用户环境变量 LLAMA_API_KEY 时的行为——进程加载完模型、
    /health 正常，客户端却一个请求都发不出去。若只看 /health 就会误报就绪。
    """
    monkeypatch.setattr(LlamaServerManager, "_build_command", _fake_command(_FAKE_AUTH_SERVER))
    mgr = LlamaServerManager(_cfg(tmp_path, 18796), tmp_path)
    mgr.start()
    st = mgr.status()
    assert st["state"] == "auth_required"
    assert st["state"] != "ready"
    assert "401" in (st["error"] or "")
    assert "LLAMA_API_KEY" in (st["advice"] or "")
    mgr.stop()


def test_child_env_strips_llama_api_key(monkeypatch) -> None:
    """托管实例的子进程环境必须剔除会静默开启鉴权的密钥变量。"""
    from backend.infra.llm_server import _child_env

    monkeypatch.setenv("LLAMA_API_KEY", "sk-should-not-propagate")
    monkeypatch.setenv("LLAMA_SERVER_API_KEY", "sk-also-not")
    monkeypatch.setenv("KEEP_ME", "yes")
    env = _child_env()
    assert "LLAMA_API_KEY" not in env
    assert "LLAMA_SERVER_API_KEY" not in env
    assert env.get("KEEP_ME") == "yes"


def test_start_async_reports_starting_not_disabled(tmp_path) -> None:
    """启动在途不得报 disabled——那是"被配置关掉"的语义，会让前端误判。"""
    mgr = LlamaServerManager(_cfg(tmp_path, 18797), tmp_path)
    st = mgr.status()
    assert st["enabled"] is True
    assert st["state"] == "starting"  # 构造完成、尚未拉起
    mgr.stop()


def test_external_mode_connects_existing_service(tmp_path, monkeypatch) -> None:
    """external 模式：端点已有服务 → ready 收编，绝不拉起子进程、退出不回收。"""
    cfg = LlmCfg(
        mode="external",
        external_endpoint="http://127.0.0.1:18801",
        probe_timeout_sec=1.0,
        watch_interval_sec=0.3,
    )
    mgr = LlamaServerManager(cfg, tmp_path)
    monkeypatch.setattr(
        LlamaServerManager,
        "_probe_external",
        lambda self: _fake_probe(reachable=True, models=["qwen3-30b"]),
    )
    mgr.start()
    st = mgr.status()
    assert st["state"] == "ready"
    assert st["mode"] == "external"
    assert st["adopted"] is True
    assert st["endpoint"] == "http://127.0.0.1:18801"
    assert mgr._proc is None  # 没有拉起任何子进程
    mgr.stop()
    assert mgr.status()["state"] == "stopped"


def test_external_mode_endpoint_down_is_unavailable_not_error(tmp_path, monkeypatch) -> None:
    """端点未运行 → unavailable（环境事实）+ 可执行建议，而不是 error（故障）。"""
    cfg = LlmCfg(mode="external", external_endpoint="http://127.0.0.1:18802")
    mgr = LlamaServerManager(cfg, tmp_path)
    monkeypatch.setattr(
        LlamaServerManager, "_probe_external", lambda self: _fake_probe(reachable=False)
    )
    mgr.start()
    st = mgr.status()
    assert st["state"] == "unavailable"
    assert "未运行" in (st["error"] or "")
    assert "请先启动本机 llama 服务" in (st["advice"] or "")


def test_external_mode_auth_required_state(tmp_path, monkeypatch) -> None:
    cfg = LlmCfg(mode="external", external_endpoint="http://127.0.0.1:18803")
    mgr = LlamaServerManager(cfg, tmp_path)
    monkeypatch.setattr(
        LlamaServerManager,
        "_probe_external",
        lambda self: _fake_probe(reachable=True, needs_auth=True),
    )
    mgr.start()
    st = mgr.status()
    assert st["state"] == "auth_required"
    assert "SCANDETECTION_LLM_API_KEY" in (st["advice"] or "")


def test_external_mode_rejects_non_loopback_endpoint(tmp_path) -> None:
    """端点非回环 → 构造即拒绝（安全边界，不是运行期可选项）。"""
    cfg = LlmCfg(mode="external", external_endpoint="http://192.168.1.9:8080")
    with pytest.raises(ValueError):
        LlamaServerManager(cfg, tmp_path)


def test_external_mode_rejects_non_http_endpoint(tmp_path) -> None:
    cfg = LlmCfg(mode="external", external_endpoint="https://127.0.0.1:8080")
    with pytest.raises(ValueError):
        LlamaServerManager(cfg, tmp_path)


def test_non_loopback_host_rejected(tmp_path) -> None:
    with pytest.raises(ValueError):
        LlamaServerManager(_cfg(tmp_path, 18794, host="192.168.1.5"), tmp_path)


def test_stop_before_start_is_idempotent(tmp_path) -> None:
    mgr = LlamaServerManager(_cfg(tmp_path, 18795), tmp_path)
    mgr.stop()
    assert mgr.status()["state"] == "stopped"


def test_full_start_ready_stop_lifecycle(tmp_path, monkeypatch) -> None:
    cfg = _cfg(tmp_path, 18796)
    mgr = LlamaServerManager(cfg, tmp_path)
    monkeypatch.setattr(LlamaServerManager, "_build_command", _fake_command(_FAKE_SERVER))
    mgr.start()
    st = mgr.status()
    assert st["state"] == "ready"
    assert st["adopted"] is False
    proc = mgr._proc
    assert proc is not None and proc.poll() is None  # 存活
    mgr.stop(timeout=5.0)
    assert mgr.status()["state"] == "stopped"
    assert proc.poll() is not None  # 进程已被回收


def test_start_timeout_reaps_process_and_degrades(tmp_path, monkeypatch) -> None:
    # 占住端口但永不就绪的"哑进程"（sleep，不监听）→ 触发启动超时回收
    sleeper = "import time; time.sleep(60)\n"
    cfg = _cfg(tmp_path, 18797, startup_timeout_sec=2.0)
    mgr = LlamaServerManager(cfg, tmp_path)
    monkeypatch.setattr(LlamaServerManager, "_build_command", _fake_command(sleeper))
    mgr.start()
    st = mgr.status()
    assert st["state"] == "error"
    assert "超时" in (st["error"] or "")
    proc = mgr._proc
    assert proc is None  # 已回收


def test_watchdog_restarts_crashed_server(tmp_path, monkeypatch) -> None:
    cfg = _cfg(tmp_path, 18798, watch_interval_sec=0.3)
    mgr = LlamaServerManager(cfg, tmp_path)
    monkeypatch.setattr(LlamaServerManager, "_build_command", _fake_command(_FAKE_SERVER))
    mgr.start()
    _wait_state(mgr, "ready")
    first_proc = mgr._proc
    assert first_proc is not None
    first_pid = first_proc.pid
    first_proc.kill()  # 模拟意外崩溃
    first_proc.wait(timeout=5)
    # 等看门狗拉起"新"实例：state=ready 且进程存活且 pid 变化
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        st = mgr.status()
        proc = mgr._proc
        if (
            st["state"] == "ready"
            and proc is not None
            and proc.poll() is None
            and proc.pid != first_pid
        ):
            break
        time.sleep(0.1)
    else:
        pytest.fail(f"看门狗未拉起新实例，当前: {mgr.status()}")
    restarted = mgr._proc
    assert restarted is not None and restarted.pid != first_pid
    mgr.stop(timeout=5.0)
    assert mgr.status()["state"] == "stopped"


def test_loopback_host_validation() -> None:
    assert is_loopback_host("127.0.0.1")
    assert is_loopback_host("127.8.8.8")
    assert is_loopback_host("::1")
    assert is_loopback_host("localhost")
    assert not is_loopback_host("192.168.1.1")
    assert not is_loopback_host("0.0.0.0")
    assert not is_loopback_host("example.com")
    assert not is_loopback_host("")
