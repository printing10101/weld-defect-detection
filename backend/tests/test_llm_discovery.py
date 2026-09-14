"""本地 LLM 资源发现：unit 测试（临时目录 + mock 网络/子进程，不依赖真实部署）。

覆盖：
- 扫描收集 ``.gguf`` 且**跳过系统/易变目录**（``temp`` / ``node_modules`` …）；
- **不跟随符号链接**（同一模型不因 link 被重复计入）；
- 同权重复本归并（保留路径最短者）与不同尺寸不误并；
- 服务探测：非回环**必须被拒**（防外发）、401 视为「活着但要凭据」而非不可达、
  OpenAI ``data[].id`` 解析、API Key 经环境变量注入；
- 扫描缓存：往返 / TTL 过期 / 版本不符 / 坏 JSON 一律安全失效；
- ``LlmDiscovery`` 编排：``start`` 幂等、取消状态、GPU 探测的失败兜底。
"""

from __future__ import annotations

import struct
import threading
import time
from pathlib import Path

import pytest

from backend.infra import llm_discovery
from backend.infra.gguf_meta import GgufMeta
from backend.infra.llm_discovery import (
    LlmDiscovery,
    dedupe_models,
    load_cache,
    probe_service,
    read_gpu_vram,
    save_cache,
    scan_gguf_files,
)


def _gguf_bytes() -> bytes:
    """最小合法 GGUF（magic + 版本 + 0 tensor + 0 KV）。"""
    return b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 0)


# --------------------------------------------------------------------------- #
# 目录扫描
# --------------------------------------------------------------------------- #


def test_scan_finds_gguf_and_skips_system_dirs(tmp_path: Path):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "a.gguf").write_bytes(_gguf_bytes())
    for skipped in ("temp", "node_modules", "$RECYCLE.BIN"):
        d = tmp_path / skipped
        d.mkdir()
        (d / "should_not_be_found.gguf").write_bytes(_gguf_bytes())

    metas, stats = scan_gguf_files([tmp_path])
    assert {Path(m.path).name for m in metas} == {"a.gguf"}
    assert stats["skipped_dirs"] == 3
    assert stats["found"] == 1


def test_scan_ignores_non_gguf_files(tmp_path: Path):
    (tmp_path / "readme.txt").write_text("x")
    (tmp_path / "m.gguf.bak").write_bytes(_gguf_bytes())
    metas, _ = scan_gguf_files([tmp_path])
    assert metas == []


def test_scan_does_not_follow_symlink(tmp_path: Path):
    """符号链接不被跟随：同一模型不因 link 重复计入（防目录循环）。"""
    real = tmp_path / "real"
    real.mkdir()
    (real / "a.gguf").write_bytes(_gguf_bytes())
    try:
        (tmp_path / "link").symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("当前环境无创建符号链接权限")
    link = tmp_path / "link"
    if not link.is_symlink() and not link.exists():
        # 本机实测过：杀软/minifilter 会把 symlink「静默吞掉」——创建调用成功、
        # 不抛异常，但条目从目录里消失。这是环境限制而非被测行为，诚实跳过。
        pytest.skip("当前环境符号链接被系统层静默丢弃（无法构造测试前提）")

    metas, stats = scan_gguf_files([tmp_path])
    assert len(metas) == 1
    assert stats["skipped_dirs"] >= 1


def test_scan_reports_elapsed_and_cancel_flag(tmp_path: Path):
    (tmp_path / "a.gguf").write_bytes(_gguf_bytes())
    _, stats = scan_gguf_files([tmp_path], parse_meta=False)
    assert stats["cancelled"] is False
    assert isinstance(stats["elapsed_sec"], float)
    assert stats["roots"] == [str(tmp_path)]


def test_scan_can_cancel_early(tmp_path: Path):
    # 需要多个目录才能触发第二轮循环顶部的取消检查（tmp_path 本身只有一轮）
    for i in range(5):
        (tmp_path / f"d{i}").mkdir()
    calls = {"n": 0}

    def should_cancel() -> bool:
        calls["n"] += 1
        return calls["n"] > 1

    _, stats = scan_gguf_files([tmp_path], should_cancel=should_cancel)
    assert stats["cancelled"] is True


# --------------------------------------------------------------------------- #
# 归并
# --------------------------------------------------------------------------- #


def test_dedupe_keeps_shortest_path():
    short = GgufMeta(
        path=r"D:\m\x.gguf", size_bytes=100, name="X", architecture="qwen3", quant="Q4_K_M"
    )
    deep = GgufMeta(
        path=r"D:\very\deep\dir\x.gguf",
        size_bytes=100,
        name="X",
        architecture="qwen3",
        quant="Q4_K_M",
    )
    uniq, dupes = dedupe_models([deep, short])
    assert len(uniq) == 1
    assert len(dupes) == 1
    assert uniq[0].path == short.path


def test_dedupe_keeps_different_sizes_separate():
    a = GgufMeta(path="a.gguf", size_bytes=100, name="X")
    b = GgufMeta(path="b.gguf", size_bytes=200, name="X")
    uniq, dupes = dedupe_models([a, b])
    assert len(uniq) == 2
    assert dupes == []


# --------------------------------------------------------------------------- #
# 服务探测
# --------------------------------------------------------------------------- #


def test_probe_rejects_non_loopback():
    """硬约束：非回环端点一律拒绝探测，绝不外发。"""
    probe = probe_service("8.8.8.8", 8080)
    assert probe.reachable is False
    assert probe.error == "non_loopback"


def test_probe_unreachable(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(llm_discovery, "_http_get_json", lambda *a, **k: (0, None))
    probe = probe_service("127.0.0.1", 8080)
    assert probe.reachable is False
    assert probe.error == "unreachable"


def test_probe_parses_openai_models(monkeypatch: pytest.MonkeyPatch):
    def fake(url, *, timeout, api_key=None):  # type: ignore[no-untyped-def]
        if url.endswith("/health"):
            return 200, {"status": "ok"}
        return 200, {"object": "list", "data": [{"id": "m1"}, {"id": "m2"}, {"bad": 1}, "junk"]}

    monkeypatch.setattr(llm_discovery, "_http_get_json", fake)
    probe = probe_service("127.0.0.1", 8080)
    assert probe.reachable is True
    assert probe.health_ok is True
    assert probe.models == ["m1", "m2"]
    assert probe.openai_base_url == "http://127.0.0.1:8080/v1"


def test_probe_401_means_alive_but_needs_auth(monkeypatch: pytest.MonkeyPatch):
    """401 是「服务活着但要凭据」，不能当成不可达（否则误报服务缺失）。"""
    monkeypatch.setattr(llm_discovery, "_http_get_json", lambda *a, **k: (401, None))
    probe = probe_service("127.0.0.1", 8081)
    assert probe.reachable is True
    assert probe.needs_auth is True
    assert probe.error == "auth_required"


def test_probe_injects_api_key_from_env(monkeypatch: pytest.MonkeyPatch):
    seen: dict[str, object] = {}

    def fake(url, *, timeout, api_key=None):  # type: ignore[no-untyped-def]
        seen["key"] = api_key
        return 200, {"data": []}

    monkeypatch.setattr(llm_discovery, "_http_get_json", fake)
    monkeypatch.setenv("MY_LLM_API_KEY", "s3cr3t")
    probe_service("127.0.0.1", 8080, api_key_env="MY_LLM_API_KEY")
    assert seen["key"] == "s3cr3t"


# --------------------------------------------------------------------------- #
# 缓存
# --------------------------------------------------------------------------- #


def _payload(**over: object) -> dict:
    base: dict = {"version": 1, "scanned_at": time.time(), "roots": [r"D:\llama"], "files": []}
    base.update(over)
    return base


def test_cache_roundtrip(tmp_path: Path):
    f = tmp_path / "cache.json"
    assert save_cache(f, _payload(files=[{"path": "x.gguf", "ok": True}])) is True
    back = load_cache(f)
    assert back is not None
    assert len(back["files"]) == 1


def test_cache_expires_by_ttl(tmp_path: Path):
    f = tmp_path / "cache.json"
    save_cache(f, _payload(scanned_at=time.time() - 100000))
    assert load_cache(f, max_age_sec=3600) is None
    assert load_cache(f, max_age_sec=0) is not None  # 0 = 忽略 TTL


def test_cache_rejects_version_and_corruption(tmp_path: Path):
    f = tmp_path / "cache.json"
    save_cache(f, _payload(version=99))
    assert load_cache(f) is None
    f.write_text("{not json", encoding="utf-8")
    assert load_cache(f) is None
    assert load_cache(tmp_path / "missing.json") is None


# --------------------------------------------------------------------------- #
# GPU 探测
# --------------------------------------------------------------------------- #


class _FakeProc:
    def __init__(self, *, rc: int = 0, out: str = "") -> None:
        self.returncode = rc
        self.stdout = out


def test_read_gpu_vram_parses_smi_output(monkeypatch: pytest.MonkeyPatch):
    fake = _FakeProc(out="NVIDIA GeForce RTX 3080 Laptop GPU, 16384, 15949\n")
    monkeypatch.setattr(llm_discovery.subprocess, "run", lambda *a, **k: fake)
    gpu = read_gpu_vram()
    assert gpu is not None
    assert gpu.name.startswith("NVIDIA")
    assert gpu.total_bytes == 16384 * 1024**2
    assert gpu.free_bytes == 15949 * 1024**2


def test_read_gpu_vram_none_on_failure(monkeypatch: pytest.MonkeyPatch):
    def boom(*a, **k):  # type: ignore[no-untyped-def]
        raise OSError("nvidia-smi not found")

    monkeypatch.setattr(llm_discovery.subprocess, "run", boom)
    assert read_gpu_vram() is None

    monkeypatch.setattr(llm_discovery.subprocess, "run", lambda *a, **k: _FakeProc(rc=1))
    assert read_gpu_vram() is None

    monkeypatch.setattr(llm_discovery.subprocess, "run", lambda *a, **k: _FakeProc(out="garbage"))
    assert read_gpu_vram() is None


# --------------------------------------------------------------------------- #
# 编排
# --------------------------------------------------------------------------- #


def test_start_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """已在扫描中时再次 start 返回 False，不重复起线程。"""
    entered = threading.Event()
    release = threading.Event()

    def slow_scan(*a, **k):  # type: ignore[no-untyped-def]
        entered.set()
        release.wait(5)
        return [], {
            "dirs": 1,
            "skipped_dirs": 0,
            "permission_errors": 0,
            "cancelled": False,
            "roots": [],
            "elapsed_sec": 0.0,
            "found": 0,
        }

    monkeypatch.setattr(llm_discovery, "scan_gguf_files", slow_scan)
    monkeypatch.setattr(llm_discovery, "read_gpu_vram", lambda: None)

    d = LlmDiscovery(roots=[str(tmp_path)], cache_file=tmp_path / "c.json")
    try:
        assert d.start() is True
        assert entered.wait(5) is True
        assert d.start() is False
    finally:
        release.set()
        d.join(5)


def test_cancel_when_idle_returns_false(tmp_path: Path):
    d = LlmDiscovery(roots=[str(tmp_path)], cache_file=tmp_path / "c.json")
    assert d.cancel() is False


def test_scan_now_reports_cancelled_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def fake_scan(*a, **k):  # type: ignore[no-untyped-def]
        return [], {
            "dirs": 1,
            "skipped_dirs": 0,
            "permission_errors": 0,
            "cancelled": True,
            "roots": [],
            "elapsed_sec": 0.1,
            "found": 0,
        }

    monkeypatch.setattr(llm_discovery, "scan_gguf_files", fake_scan)
    monkeypatch.setattr(llm_discovery, "read_gpu_vram", lambda: llm_discovery.GpuInfo("gpu", 8, 4))

    d = LlmDiscovery(roots=[str(tmp_path)], cache_file=tmp_path / "c.json")
    status = d.scan_now()
    assert status["state"] == "cancelled"
    _, gpu, _ = d.results()
    assert gpu is not None and gpu.name == "gpu"


def test_scan_now_survives_unexpected_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """发现过程出错必须被吞掉并落 error 状态，不能拖垮调用方（服务启动路径）。"""

    def boom(*a, **k):  # type: ignore[no-untyped-def]
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(llm_discovery, "scan_gguf_files", boom)
    d = LlmDiscovery(roots=[str(tmp_path)], cache_file=tmp_path / "c.json")
    status = d.scan_now()
    assert status["state"] == "error"
    assert "disk on fire" in str(status["error"])


def test_cached_requires_root_cover(tmp_path: Path):
    """缓存必须覆盖当前扫描根，否则失效重扫（换了范围就该重来）。"""
    f = tmp_path / "c.json"
    save_cache(f, _payload(roots=[r"D:\llama"], files=[]))
    hit = LlmDiscovery(roots=[r"D:\llama"], cache_file=f)
    miss = LlmDiscovery(roots=[r"E:\llama-cpp"], cache_file=f)
    assert hit.cached() is not None
    assert miss.cached() is None
