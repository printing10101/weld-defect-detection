"""GGUF 元数据解析 + 显存估算：unit 测试（构造字节流，不依赖真实权重文件）。

覆盖：
- 字符串型 KV（``general.architecture`` / ``general.name``）能正确读出
  —— 回归：早期版本「只看数值类型」导致架构名恒空、上下文随之落空；
- 量化（``general.file_type`` 编号 → 名称）与 ``{arch}.context_length`` 解析；
- ``attention.head_count_kv`` 为逐层数组时取最大值兜底并置 from_array；
- embedding 模型经 ``pooling_type`` 识别；
- 头部畸形的四类失败路径（非 GGUF / 仅 magic / 文件缺失 / 目录当文件）
  一律返回 ``ok=False`` **且不抛异常**；
- KV cache 估算公式与 ``estimate_vram`` 五档判定（full_gpu / tight /
  partial_offload / infeasible / unknown）。
"""

from __future__ import annotations

import struct
from pathlib import Path

from backend.infra.gguf_meta import (
    GgufMeta,
    estimate_kv_cache_bytes,
    estimate_vram,
    read_gguf_meta,
)

_GIB = 1024**3

# GGUF 值类型（llama.cpp gguf.h）——本地定义，避免测试随私有常量重构而静默失效
_T_UINT8, _T_INT8, _T_UINT16, _T_INT16, _T_UINT32, _T_INT32 = 0, 1, 2, 3, 4, 5
_T_STRING, _T_ARRAY, _T_UINT64 = 8, 9, 10


def _enc(vtype: int, val: object) -> bytes:
    if vtype == _T_STRING:
        raw = str(val).encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw
    if vtype == _T_UINT32:
        return struct.pack("<I", int(val))  # type: ignore[arg-type]
    if vtype == _T_INT32:
        return struct.pack("<i", int(val))  # type: ignore[arg-type]
    if vtype == _T_UINT64:
        return struct.pack("<Q", int(val))  # type: ignore[arg-type]
    if vtype == _T_ARRAY:
        elem_type, items = val  # type: ignore[misc]
        out = struct.pack("<I", int(elem_type)) + struct.pack("<Q", len(items))
        for item in items:
            out += _enc(int(elem_type), item)
        return out
    raise AssertionError(f"unsupported vtype {vtype}")


def _gguf(kvs: list[tuple[str, int, object]], *, version: int = 3, tensors: int = 0) -> bytes:
    buf = bytearray(b"GGUF")
    buf += struct.pack("<I", version)
    buf += struct.pack("<Q", tensors)
    buf += struct.pack("<Q", len(kvs))
    for key, vtype, val in kvs:
        name = key.encode("utf-8")
        buf += struct.pack("<Q", len(name)) + name
        buf += struct.pack("<I", vtype)
        buf += _enc(vtype, val)
    return bytes(buf)


def _write(tmp_path: Path, data: bytes, name: str = "m.gguf") -> Path:
    target = tmp_path / name
    target.write_bytes(data)
    return target


_QWEN_KVS: list[tuple[str, int, object]] = [
    ("general.architecture", _T_STRING, "qwen3"),
    ("general.name", _T_STRING, "Qwen3 4B Test"),
    ("general.file_type", _T_UINT32, 15),  # Q4_K_M
    ("qwen3.context_length", _T_UINT32, 40960),
    ("qwen3.block_count", _T_UINT32, 36),
    ("qwen3.embedding_length", _T_UINT32, 2560),
    ("qwen3.attention.head_count", _T_UINT32, 32),
    ("qwen3.attention.head_count_kv", _T_UINT32, 8),
]


def test_parses_string_kv(tmp_path: Path):
    """回归：字符串型 KV 必须能读出（此前只判数值类型导致架构名恒空）。"""
    meta = read_gguf_meta(_write(tmp_path, _gguf(_QWEN_KVS)))
    assert meta.ok
    assert meta.architecture == "qwen3"
    assert meta.name == "Qwen3 4B Test"
    assert meta.gguf_version == 3


def test_quant_and_ctx_resolved(tmp_path: Path):
    meta = read_gguf_meta(_write(tmp_path, _gguf(_QWEN_KVS)))
    assert meta.quant == "Q4_K_M"
    assert meta.max_ctx == 40960
    assert meta.block_count == 36
    assert meta.head_count == 32
    assert meta.head_count_kv == 8


def test_unknown_file_type_falls_back_to_number(tmp_path: Path):
    kvs = [("general.architecture", _T_STRING, "qwen3"), ("general.file_type", _T_UINT32, 999)]
    meta = read_gguf_meta(_write(tmp_path, _gguf(kvs)))
    assert meta.quant == "file_type_999"


def test_head_count_kv_array_takes_max(tmp_path: Path):
    """逐层数组（GQA 分层 / 滑动窗口模型）取最大值兜底并标记 from_array。"""
    kvs = [
        ("general.architecture", _T_STRING, "gpt-oss"),
        ("gpt-oss.attention.head_count_kv", _T_ARRAY, (_T_UINT32, [4, 8, 2, 8])),
    ]
    meta = read_gguf_meta(_write(tmp_path, _gguf(kvs)))
    assert meta.head_count_kv == 8
    assert meta.head_count_kv_from_array is True


def test_embedding_detected_by_pooling_type(tmp_path: Path):
    kvs = [
        ("general.architecture", _T_STRING, "bert"),
        ("bert.pooling_type", _T_UINT32, 1),
    ]
    meta = read_gguf_meta(_write(tmp_path, _gguf(kvs)))
    assert meta.is_embedding is True


def test_not_gguf_for_plain_file(tmp_path: Path):
    meta = read_gguf_meta(_write(tmp_path, b"hello world, definitely not gguf", "plain.gguf"))
    assert meta.ok is False
    assert meta.error == "not_gguf"


def test_truncated_header_when_magic_only(tmp_path: Path):
    """magic 正确但固定头不全 → 报 truncated_header（而非笼统的 not_gguf）。"""
    meta = read_gguf_meta(_write(tmp_path, b"GGUF" + b"\x00" * 4, "short.gguf"))
    assert meta.ok is False
    assert meta.error == "truncated_header"


def test_missing_file_does_not_raise(tmp_path: Path):
    meta = read_gguf_meta(tmp_path / "nope.gguf")
    assert meta.ok is False
    assert meta.error is not None and meta.error.startswith("stat_failed")


def test_directory_path_does_not_raise(tmp_path: Path):
    meta = read_gguf_meta(tmp_path)
    assert meta.ok is False
    assert meta.error is not None


def test_kv_cache_formula():
    """KV cache = 层数 × KV头数 × head_dim × ctx × 2(K,V) × 2B(f16)。"""
    meta = GgufMeta(
        path="x",
        ok=True,
        block_count=32,
        embedding_length=4096,
        head_count=32,
        head_count_kv=8,
    )
    # head_dim = 4096/32 = 128；32*8*128*8192*2*2 = 1 GiB
    assert estimate_kv_cache_bytes(meta, 8192) == _GIB
    # 缺失关键字段 → None（而非编造数字）
    assert estimate_kv_cache_bytes(GgufMeta(path="x", ok=True), 8192) is None


def _meta_4g() -> GgufMeta:
    """4 GiB 权重 + 1 GiB KV(8K) + 0.5 GiB 固定开销 = 需要 5.5 GiB。"""
    return GgufMeta(
        path="m.gguf",
        size_bytes=4 * _GIB,
        ok=True,
        block_count=32,
        embedding_length=4096,
        head_count=32,
        head_count_kv=8,
    )


def test_vram_verdict_full_gpu():
    est = estimate_vram(_meta_4g(), 8192, free_vram_bytes=int(8.5 * _GIB))
    assert est.verdict == "full_gpu"
    assert est.needed_bytes == int(5.5 * _GIB)


def test_vram_verdict_tight():
    est = estimate_vram(_meta_4g(), 8192, free_vram_bytes=int(5.5 * _GIB))
    assert est.verdict == "tight"


def test_vram_verdict_partial_offload():
    """权重放得下、含上下文放不下 → 需降上下文或部分层走 CPU。"""
    est = estimate_vram(_meta_4g(), 8192, free_vram_bytes=int(5.5 * _GIB) - 1)
    assert est.verdict == "partial_offload"


def test_vram_verdict_infeasible_when_weights_exceed():
    """权重本身超显存（如 16.5 GB 的 30B MoE 对上 16 GB 卡）→ 直接不可行。"""
    est = estimate_vram(_meta_4g(), 8192, free_vram_bytes=3 * _GIB)
    assert est.verdict == "infeasible"
    assert "无法全量上卡" in est.advice


def test_vram_verdict_unknown_without_gpu():
    """未探测到 GPU 时只给需求侧数字，不猜硬件。"""
    est = estimate_vram(_meta_4g(), 8192)
    assert est.verdict == "unknown"
    assert est.needed_bytes == int(5.5 * _GIB)


def test_vram_bad_meta_is_unknown():
    est = estimate_vram(GgufMeta(path="x", ok=False, error="not_gguf"), 8192)
    assert est.verdict == "unknown"
    assert est.needed_bytes is None
