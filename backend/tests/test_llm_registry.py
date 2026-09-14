"""本地大模型注册表（LlmRegistry）：发现 / 选中 / 持久化 / 目录 单元测试。

不依赖真实权重与网络：
- 用最小 GGUF 字节流构造"模型文件"（与 test_gguf_meta 同法，本文件自带构造器，
  避免跨测试模块耦合）；
- 服务探测结果以 :class:`ServiceProbe` 直接注入，不发起任何请求；
- 显存判定用显式 ``free_vram_bytes``，不调 nvidia-smi。

覆盖要点
--------
- **不折叠、不过滤**：跑不动的模型（``infeasible``）照列，同模型多副本照列
  且标出 ``duplicate_of`` / ``duplicate_count``；
- 选中本地 → ``managed`` + ``model_file``；选中服务 → ``external`` + 端点；
- 不可用条目（GGUF 头损坏 / 端点要鉴权）选中被拒（ValueError）；
- 状态文件持久化往返、损坏回落默认、目录增删幂等、盘根不被同步直扫（防病态阻塞）；
- 非回环端点拒绝探测（安全边界）。
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from backend.infra.config import LlmCfg
from backend.infra.llm_discovery import ServiceProbe
from backend.infra.llm_registry import (
    MODE_EXTERNAL,
    MODE_MANAGED,
    SOURCE_LOCAL,
    SOURCE_SERVICE,
    LlmRegistry,
)

_GIB = 1024**3

_T_UINT32, _T_STRING = 4, 8


def _enc(vtype: int, val: object) -> bytes:
    if vtype == _T_STRING:
        raw = str(val).encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw
    if vtype == _T_UINT32:
        return struct.pack("<I", int(val))  # type: ignore[arg-type]
    raise AssertionError(f"unsupported vtype {vtype}")


def _gguf_bytes(arch: str = "qwen3", name: str = "Qwen3 4B", ctx: int = 40960) -> bytes:
    kvs: list[tuple[str, int, object]] = [
        ("general.architecture", _T_STRING, arch),
        ("general.name", _T_STRING, name),
        ("general.file_type", _T_UINT32, 15),
        (f"{arch}.context_length", _T_UINT32, ctx),
        (f"{arch}.block_count", _T_UINT32, 36),
        (f"{arch}.embedding_length", _T_UINT32, 2560),
        (f"{arch}.attention.head_count", _T_UINT32, 32),
        (f"{arch}.attention.head_count_kv", _T_UINT32, 8),
    ]
    buf = bytearray(b"GGUF")
    buf += struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kvs))
    for key, vtype, val in kvs:
        raw = key.encode("utf-8")
        buf += struct.pack("<Q", len(raw)) + raw
        buf += struct.pack("<I", vtype) + _enc(vtype, val)
    return bytes(buf)


def _make_model(dirpath: Path, fname: str = "model.gguf", size_pad: int = 0) -> Path:
    """写一个最小合法 GGUF（可选尾部填充以模拟不同体积）。"""
    dirpath.mkdir(parents=True, exist_ok=True)
    target = dirpath / fname
    target.write_bytes(_gguf_bytes() + b"\x00" * size_pad)
    return target


def _cfg(tmp_path: Path, **kw) -> LlmCfg:
    """测试用配置：扫描根锁定到空目录，避免误触发全盘扫描。"""
    empty = tmp_path / "_empty_scan_root"
    empty.mkdir(exist_ok=True)
    kw.setdefault("scan_roots", [str(empty)])
    kw.setdefault("model_dirs", [])
    kw.setdefault("service_presets", [])
    return LlmCfg(**kw)


def _reg(tmp_path: Path, cfg: LlmCfg | None = None) -> LlmRegistry:
    cfg = cfg or _cfg(tmp_path)
    return LlmRegistry(
        cfg,
        state_file=tmp_path / "state" / "llm_registry.json",
        cache_file=tmp_path / "state" / "llm_scan_cache.json",
    )


def _probe(models: list[str], *, needs_auth: bool = False, port: int = 8080) -> ServiceProbe:
    p = ServiceProbe(host="127.0.0.1", port=port)
    p.reachable = True
    p.needs_auth = needs_auth
    p.models = models
    return p


# --------------------------------------------------------------------------- #
# 清单：不过滤、不折叠
# --------------------------------------------------------------------------- #


def test_local_models_listed_from_declared_dir(tmp_path) -> None:
    """声明目录后应立刻能看到模型（不必等全盘扫描）。"""
    models_dir = tmp_path / "models"
    _make_model(models_dir, "Qwen3-4B-Q4_K_M.gguf")
    reg = _reg(tmp_path, _cfg(tmp_path, model_dirs=[str(models_dir)]))

    entries = reg.list_models(include_services=False, free_vram_bytes=16 * _GIB)
    assert len(entries) == 1
    e = entries[0]
    assert e.source == SOURCE_LOCAL
    assert e.available is True
    assert e.architecture == "qwen3"
    assert e.quant == "Q4_K_M"
    assert e.max_ctx == 40960
    assert e.id.startswith("gguf:")
    assert e.vram is not None


def test_infeasible_model_is_still_listed(tmp_path) -> None:
    """跑不动的模型必须照列（只标注判定，不隐藏）——用户明确要求。"""
    models_dir = tmp_path / "models"
    _make_model(models_dir, "big.gguf")
    reg = _reg(tmp_path, _cfg(tmp_path, model_dirs=[str(models_dir)]))

    entries = reg.list_models(include_services=False, free_vram_bytes=1)  # 显存远小于权重
    assert len(entries) == 1
    assert entries[0].available is True  # 可用（只是跑不动）
    assert entries[0].vram is not None
    assert entries[0].vram["verdict"] == "infeasible"


def test_duplicate_copies_are_listed_not_merged(tmp_path) -> None:
    """同模型多副本全部列出：主条目带 duplicate_count，副本带 duplicate_of。"""
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    body = _gguf_bytes()
    (a_dir).mkdir(parents=True, exist_ok=True)
    (b_dir).mkdir(parents=True, exist_ok=True)
    (a_dir / "same.gguf").write_bytes(body)
    (b_dir / "same.gguf").write_bytes(body)
    reg = _reg(tmp_path, _cfg(tmp_path, model_dirs=[str(a_dir), str(b_dir)]))

    entries = reg.list_models(include_services=False, free_vram_bytes=16 * _GIB)
    assert len(entries) == 2  # 没被折叠掉
    primaries = [e for e in entries if e.primary]
    copies = [e for e in entries if not e.primary]
    assert len(primaries) == 1 and len(copies) == 1
    assert primaries[0].duplicate_count == 1
    assert copies[0].duplicate_of == primaries[0].id


def test_services_merged_into_same_list(tmp_path) -> None:
    models_dir = tmp_path / "models"
    _make_model(models_dir, "local.gguf")
    reg = _reg(tmp_path, _cfg(tmp_path, model_dirs=[str(models_dir)]))

    entries = reg.list_models(
        probes=[_probe(["qwen3-instruct-30b", "gpt-oss-20b"])],
        free_vram_bytes=16 * _GIB,
    )
    by_source = {e.source for e in entries}
    assert by_source == {SOURCE_LOCAL, SOURCE_SERVICE}
    svc = [e for e in entries if e.source == SOURCE_SERVICE]
    assert len(svc) == 2
    assert all(e.endpoint == "http://127.0.0.1:8080" for e in svc)
    assert all(e.vram is None for e in svc)  # 服务模型的显存由服务自己承担


def test_corrupt_gguf_listed_as_unavailable(tmp_path) -> None:
    """坏文件不能拖垮列表：照列 + available=false + 原因。"""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "broken.gguf").write_bytes(b"this is not a gguf file at all")
    reg = _reg(tmp_path, _cfg(tmp_path, model_dirs=[str(models_dir)]))

    entries = reg.list_models(include_services=False, free_vram_bytes=16 * _GIB)
    assert len(entries) == 1
    assert entries[0].available is False
    assert entries[0].error == "not_gguf"


# --------------------------------------------------------------------------- #
# 选中
# --------------------------------------------------------------------------- #


def test_select_local_yields_managed_cfg(tmp_path) -> None:
    models_dir = tmp_path / "models"
    model = _make_model(models_dir, "Qwen3-4B-Q4_K_M.gguf")
    cfg = _cfg(tmp_path, model_dirs=[str(models_dir)])
    reg = _reg(tmp_path, cfg)

    entry = reg.select(
        next(e.id for e in reg.list_models(include_services=False, free_vram_bytes=16 * _GIB))
    )
    assert entry.active is True
    eff = reg.effective_llm_cfg(cfg)
    assert eff.mode == MODE_MANAGED
    assert eff.model_file == str(model)


def test_select_service_yields_external_cfg(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    reg = _reg(tmp_path, cfg)
    reg.list_models(probes=[_probe(["qwen3-coder-30b"], port=8080)], free_vram_bytes=16 * _GIB)

    svc_id = next(
        e.id
        for e in reg.list_models(probes=[_probe(["qwen3-coder-30b"])], free_vram_bytes=None)
        if e.source == SOURCE_SERVICE
    )
    reg.select(svc_id)
    eff = reg.effective_llm_cfg(cfg)
    assert eff.mode == MODE_EXTERNAL
    assert eff.external_endpoint == "http://127.0.0.1:8080"


def test_select_unavailable_model_rejected(tmp_path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "broken.gguf").write_bytes(b"not gguf at all")
    reg = _reg(tmp_path, _cfg(tmp_path, model_dirs=[str(models_dir)]))

    bad = reg.list_models(include_services=False, free_vram_bytes=16 * _GIB)[0]
    with pytest.raises(ValueError):
        reg.select(bad.id)


def test_select_auth_required_service_rejected(tmp_path) -> None:
    reg = _reg(tmp_path)
    probes = [_probe(["locked-model"], needs_auth=True)]
    entry = reg.list_models(probes=probes, free_vram_bytes=None)[0]
    assert entry.available is False
    assert entry.error == "auth_required"
    with pytest.raises(ValueError):
        reg.select(entry.id)


def test_select_unknown_id_raises_keyerror(tmp_path) -> None:
    reg = _reg(tmp_path)
    with pytest.raises(KeyError):
        reg.select("gguf:deadbeef0000")


def test_select_local_rejects_deleted_file(tmp_path) -> None:
    """清单快照可能陈旧：文件被删后选中必须被拒，而不是把失败留到引擎启动。"""
    models_dir = tmp_path / "models"
    model = _make_model(models_dir, "gone.gguf")
    reg = _reg(tmp_path, _cfg(tmp_path, model_dirs=[str(models_dir)]))
    entry = reg.list_models(include_services=False, free_vram_bytes=16 * _GIB)[0]

    model.unlink()
    with pytest.raises(ValueError):
        reg.select(entry.id)


def test_select_service_survives_probe_hiccup(tmp_path) -> None:
    """端点一次探测失败时，用户"看着列表点下去"仍应选得中（回落清单快照）。"""
    reg = _reg(tmp_path)  # service_presets 为空 → 再次探测拿不到任何服务条目
    entry = reg.list_models(probes=[_probe(["qwen3-coder-30b"])], free_vram_bytes=None)[0]
    picked = reg.select(entry.id)
    assert picked.id == entry.id
    assert reg.selection()["mode"] == MODE_EXTERNAL


def test_selection_survives_reload(tmp_path) -> None:
    models_dir = tmp_path / "models"
    _make_model(models_dir, "m.gguf")
    reg = _reg(tmp_path, _cfg(tmp_path, model_dirs=[str(models_dir)]))
    entry = reg.list_models(include_services=False, free_vram_bytes=16 * _GIB)[0]
    reg.select(entry.id)

    reloaded = _reg(tmp_path, _cfg(tmp_path, model_dirs=[str(models_dir)]))
    assert reloaded.active_id == entry.id
    assert reloaded.selection()["mode"] == MODE_MANAGED
    assert reloaded.selection()["model_path"] == entry.path


def test_corrupt_state_file_falls_back_to_defaults(tmp_path) -> None:
    state = tmp_path / "state" / "llm_registry.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text("{ this is not json", encoding="utf-8")
    reg = LlmRegistry(_cfg(tmp_path), state_file=state, cache_file=tmp_path / "c.json")
    assert reg.active_id is None
    assert reg.extra_model_dirs() == []


def test_unknown_state_version_falls_back(tmp_path) -> None:
    state = tmp_path / "state" / "llm_registry.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps({"version": 999, "active_id": "gguf:x"}), encoding="utf-8")
    reg = LlmRegistry(_cfg(tmp_path), state_file=state, cache_file=tmp_path / "c.json")
    assert reg.active_id is None


# --------------------------------------------------------------------------- #
# 目录
# --------------------------------------------------------------------------- #


def test_add_and_remove_dir(tmp_path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    reg = _reg(tmp_path)

    resolved = reg.add_dir(str(models_dir))
    assert Path(resolved) == models_dir
    assert [Path(d) for d in reg.extra_model_dirs()] == [models_dir]

    reg.add_dir(str(models_dir))  # 幂等
    assert len(reg.extra_model_dirs()) == 1

    assert reg.remove_dir(str(models_dir)) is True
    assert reg.extra_model_dirs() == []
    assert reg.remove_dir(str(models_dir)) is False  # 幂等


def test_add_dir_rejects_missing_path(tmp_path) -> None:
    reg = _reg(tmp_path)
    with pytest.raises(ValueError):
        reg.add_dir(str(tmp_path / "nope"))
    with pytest.raises(ValueError):
        reg.add_dir("   ")


def test_config_dirs_not_duplicated_by_add(tmp_path) -> None:
    """配置文件里的内置目录再手动添加一次，不应产生重复项。"""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    reg = _reg(tmp_path, _cfg(tmp_path, model_dirs=[str(models_dir)]))
    reg.add_dir(str(models_dir))
    assert reg.extra_model_dirs() == []
    assert len(reg.all_model_dirs()) == 1


def test_drive_root_excluded_from_direct_scan(tmp_path) -> None:
    """盘根声明交给全盘扫描，不做同步直扫（否则一次请求阻塞数分钟）。"""
    root = "C:\\" if len(str(tmp_path)) > 3 else "/"
    reg = _reg(tmp_path, _cfg(tmp_path, model_dirs=[root]))
    assert reg.declared_dirs()  # 声明仍在
    assert reg._dir_metas() == []  # 但不做直扫


def test_engine_model_dirs_and_scan_roots_are_distinct(tmp_path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    scan_root = tmp_path / "scan"
    scan_root.mkdir()
    reg = _reg(tmp_path, _cfg(tmp_path, model_dirs=[str(models_dir)], scan_roots=[str(scan_root)]))
    assert reg.declared_dirs() == [str(models_dir)]
    assert reg.scan_roots() == [str(scan_root)]


# --------------------------------------------------------------------------- #
# 探测：仅回环
# --------------------------------------------------------------------------- #


def test_presets_skip_invalid_entries(tmp_path) -> None:
    reg = _reg(tmp_path, _cfg(tmp_path, service_presets=["127.0.0.1:8080", "garbage", ":80"]))
    assert reg.presets() == [("127.0.0.1", 8080)]


def test_non_loopback_preset_refused_without_outbound(tmp_path) -> None:
    """非回环端点直接拒绝（不发任何请求）——安全边界回归。"""
    reg = _reg(tmp_path, _cfg(tmp_path, service_presets=["8.8.8.8:80"], probe_timeout_sec=0.2))
    probes = reg.probe_all()
    assert len(probes) == 1
    assert probes[0].reachable is False
    assert probes[0].error == "non_loopback"
