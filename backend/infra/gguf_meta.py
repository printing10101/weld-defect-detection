"""GGUF 模型元数据解析 + 运行显存估算（只读文件头部，绝不整文件载入）。

用途
----
「本地大模型发现」的底座能力：给定一个 ``.gguf`` 路径，解析出模型名 / 架构 /
量化 / 上下文上限 / 层数 / 注意力头数，并据此估算运行时显存需求。上层
（``infra.llm_discovery`` / ``api.v1.llm``）据此回答两个问题：
**本机有哪些模型**、**哪些跑得动**。

资源边界（硬约束）
------------------
- **只读**：不修改、不移动、不删除任何文件；
- 先校验 ``GGUF`` magic，非 GGUF 一律拒绝解析；
- **只读文件头部** ``HEADER_READ_BYTES``（默认 64 MiB），绝不整文件读入——
  本机最大权重单文件 16.5 GB（Qwen3-30B-A3B），整读会直接把内存打爆；
- 解析器全程带边界（``_Reader`` 越界即抛 ``EOFError``），畸形/截断头部一律
  返回 ``ok=False``，**绝不向调用方抛异常**（列表接口不能因一个坏文件整体 500）。

为什么头部预算是 64 MiB
-----------------------
KV 元数据区在文件头部，但**大 tokenizer 词表 + MoE 模型的 KV 区会超 8 MiB**：
实测 ``gpt-oss-20b`` 在 8 MiB 预算下被截断（``meta_truncated=True``、架构读不到），
升到 64 MiB 后本机全部大模型完整读出，单文件仅 0.3–0.4 s。串行读取，峰值可控。

估算的置信度
------------
KV cache 按 ``层数 × KV头数 × head_dim × ctx × 2(K,V) × 2B(f16)`` 估算，
是**上界**。两处会导致实际占用更低：
- 采用滑动窗口注意力的模型（如 gpt-oss）只有部分层缓存全长上下文；
- ``attention.head_count_kv`` 为逐层数组时（GQA 分层配置），取数组最大值兜底，
  偏保守。
故 ``advice`` 里对这类情况显式提示，避免把「估算够」说成「实测够」。
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_LOG = logging.getLogger("scandetection.gguf")

GGUF_MAGIC = b"GGUF"

# 元数据 KV 区在文件头部；64 MiB 覆盖本机全部大模型（含 30B MoE 与 gpt-oss 词表）。
HEADER_READ_BYTES = 64 * 1024 * 1024

_GIB = 1024 * 1024 * 1024

# GGUF 元数据类型枚举（llama.cpp gguf.h）
_T_UINT8, _T_INT8, _T_UINT16, _T_INT16 = 0, 1, 2, 3
_T_UINT32, _T_INT32, _T_FLOAT32, _T_BOOL = 4, 5, 6, 7
_T_STRING, _T_ARRAY, _T_UINT64, _T_INT64, _T_FLOAT64 = 8, 9, 10, 11, 12

_SCALAR_FMT: dict[int, tuple[str, int]] = {
    _T_UINT8: ("<B", 1),
    _T_INT8: ("<b", 1),
    _T_UINT16: ("<H", 2),
    _T_INT16: ("<h", 2),
    _T_UINT32: ("<I", 4),
    _T_INT32: ("<i", 4),
    _T_FLOAT32: ("<f", 4),
    _T_BOOL: ("<B", 1),
    _T_UINT64: ("<Q", 8),
    _T_INT64: ("<q", 8),
    _T_FLOAT64: ("<d", 8),
}

# general.file_type → 量化方案（llama.cpp ggml_type 的 LLAMA_FTYPE_* 编号）
_QUANT_BY_FILE_TYPE: dict[int, str] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    19: "IQ2_XXS",
    20: "IQ2_XS",
    21: "IQ3_XXS",
    22: "IQ1_S",
    23: "IQ4_NL",
    24: "IQ3_S",
    25: "IQ2_S",
    26: "IQ4_XS",
    27: "I8",
    28: "I16",
    29: "I32",
    30: "I64",
    31: "F64",
    32: "IQ1_M",
    33: "BF16",
}

# 精确匹配的键（非 arch 前缀）
_EXACT_STR_KEYS = {"general.name", "general.architecture"}
_EXACT_INT_KEYS = {"general.file_type", "general.parameter_count", "split.count"}

# 前缀在架构名之后的键（qwen3.context_length / qwen3.attention.head_count …）
_WANTED_SUFFIXES = (
    ".context_length",
    ".block_count",
    ".embedding_length",
    ".attention.head_count",
    ".attention.head_count_kv",
    ".attention.key_length",
    ".attention.value_length",
    ".pooling_type",
)

# 可能以「逐层数组」形式出现的键：取最大值兜底（偏保守）
_ARRAY_MAX_SUFFIXES = (".attention.head_count_kv",)

_KV_DTYPE_BYTES = 2  # llama.cpp 默认 KV cache 为 f16
_DEFAULT_OVERHEAD_BYTES = 512 * 1024 * 1024  # CUDA context + 计算缓冲经验值
_TIGHT_RESERVE_BYTES = 2 * _GIB  # 余量低于此值即认为「与检测模型并存会紧」


class _Reader:
    """带边界的顺序读取器：越界抛 ``EOFError``，由调用方统一兜底。"""

    __slots__ = ("buf", "pos")

    def __init__(self, buf: bytes) -> None:
        self.buf = buf
        self.pos = 0

    def read(self, n: int) -> bytes:
        if n < 0 or self.pos + n > len(self.buf):
            raise EOFError
        out = self.buf[self.pos : self.pos + n]
        self.pos += n
        return out

    def scalar(self, fmt: str) -> Any:
        """按 struct 格式读标量（GGUF 值为异构二进制，类型由 fmt 决定）。"""
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        n = int(self.scalar("<Q"))
        return self.read(n).decode("utf-8", "replace")

    def skip_value(self, vtype: int) -> None:
        """跳过非目标值（数组递归跳过）；未知类型抛 ``ValueError``。"""
        if vtype in _SCALAR_FMT:
            self.read(_SCALAR_FMT[vtype][1])
            return
        if vtype == _T_STRING:
            self.read(int(self.scalar("<Q")))
            return
        if vtype == _T_ARRAY:
            elem_type = int(self.scalar("<I"))
            count = int(self.scalar("<Q"))
            for _ in range(count):
                self.skip_value(elem_type)
            return
        raise ValueError(f"unknown gguf value type {vtype}")


@dataclass
class GgufMeta:
    """单个 GGUF 文件的元数据摘要（解析失败时 ``ok=False`` 且 ``error`` 有值）。"""

    path: str
    size_bytes: int = 0
    ok: bool = False
    error: str | None = None

    gguf_version: int = 0
    kv_count: int = 0
    tensor_count: int = 0

    name: str = ""
    architecture: str = ""
    quant: str = ""

    max_ctx: int | None = None  # 模型**支持上限**，不等于服务实际 -c
    block_count: int | None = None
    embedding_length: int | None = None
    head_count: int | None = None
    head_count_kv: int | None = None
    head_count_kv_from_array: bool = False  # 逐层数组取 max 兜底（估算偏保守）
    key_length: int | None = None

    is_embedding: bool = False
    meta_truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "ok": self.ok,
            "error": self.error,
            "gguf_version": self.gguf_version,
            "kv_count": self.kv_count,
            "tensor_count": self.tensor_count,
            "name": self.name,
            "architecture": self.architecture,
            "quant": self.quant,
            "max_ctx": self.max_ctx,
            "block_count": self.block_count,
            "embedding_length": self.embedding_length,
            "head_count": self.head_count,
            "head_count_kv": self.head_count_kv,
            "key_length": self.key_length,
            "is_embedding": self.is_embedding,
            "meta_truncated": self.meta_truncated,
        }


def _max_scalar_array(r: _Reader) -> int | None:
    """读完整数组并返回最大值（逐层 KV 头数配置的保守兜底）；非数值元素返回 None。"""
    elem_type = int(r.scalar("<I"))
    count = int(r.scalar("<Q"))
    if elem_type not in _SCALAR_FMT:
        for _ in range(count):
            r.skip_value(elem_type)
        return None
    fmt = _SCALAR_FMT[elem_type][0]
    best: int | None = None
    for _ in range(count):
        val = int(r.scalar(fmt))
        if best is None or val > best:
            best = val
    return best


def _consume_kv(
    r: _Reader, key: str, vtype: int, out: dict[str, Any], array_keys: set[str]
) -> None:
    """消费一个 KV：命中关注键则记录，否则跳过（保持读取位置对齐）。"""
    wanted = key in _EXACT_STR_KEYS or key in _EXACT_INT_KEYS or key.endswith(_WANTED_SUFFIXES)
    if not wanted:
        r.skip_value(vtype)
        return
    if vtype == _T_STRING:
        out[key] = r.string()
        return
    if vtype in _SCALAR_FMT:
        out[key] = r.scalar(_SCALAR_FMT[vtype][0])
        return
    if vtype == _T_ARRAY and key.endswith(_ARRAY_MAX_SUFFIXES):
        best = _max_scalar_array(r)
        if best is not None:
            out[key] = best
            array_keys.add(key)
        return
    r.skip_value(vtype)


def _arch_scalar(kv: dict[str, Any], arch: str, suffix: str) -> int | None:
    """取 ``{arch}{suffix}``；架构名缺失/不匹配时退化为「任一以 suffix 结尾的键」。"""
    if arch:
        val = kv.get(f"{arch}{suffix}")
        if isinstance(val, int):
            return val
    for key, val in kv.items():
        if key.endswith(suffix) and isinstance(val, int):
            return val
    return None


def read_gguf_meta(path: str | Path) -> GgufMeta:
    """解析 GGUF 头部 → :class:`GgufMeta`。任何异常都转成 ``ok=False``，不向外抛。"""
    p = Path(path)
    try:
        size = p.stat().st_size
    except OSError as exc:
        return GgufMeta(path=str(p), error=f"stat_failed:{exc.__class__.__name__}")

    try:
        with p.open("rb") as fh:
            head = fh.read(HEADER_READ_BYTES)
    except OSError as exc:
        return GgufMeta(path=str(p), size_bytes=size, error=f"read_failed:{exc.__class__.__name__}")

    if len(head) < 4 or head[:4] != GGUF_MAGIC:
        return GgufMeta(path=str(p), size_bytes=size, error="not_gguf")
    if len(head) < 24:
        # magic 正确但连固定头（version/tensor_count/kv_count）都不完整
        return GgufMeta(path=str(p), size_bytes=size, error="truncated_header")

    r = _Reader(head)
    try:
        r.read(4)  # magic
        version = int(r.scalar("<I"))
        tensor_count = int(r.scalar("<Q"))
        kv_count = int(r.scalar("<Q"))
    except (EOFError, struct.error):
        return GgufMeta(path=str(p), size_bytes=size, error="truncated_header")

    kv: dict[str, Any] = {}
    array_keys: set[str] = set()
    truncated = False
    for _ in range(kv_count):
        try:
            key = r.string()
            vtype = int(r.scalar("<I"))
            _consume_kv(r, key, vtype, kv, array_keys)
        except (EOFError, struct.error, ValueError, MemoryError):
            # 头部预算内未读完全部 KV：已读到的字段仍然可用，标记 truncated 即可
            truncated = True
            break

    arch = str(kv.get("general.architecture") or "")
    file_type = kv.get("general.file_type")
    name = str(kv.get("general.name") or "")

    meta = GgufMeta(
        path=str(p),
        size_bytes=size,
        ok=True,
        gguf_version=version,
        kv_count=kv_count,
        tensor_count=tensor_count,
        name=name,
        architecture=arch,
        quant=_QUANT_BY_FILE_TYPE.get(file_type, f"file_type_{file_type}")
        if isinstance(file_type, int)
        else "?",
        max_ctx=_arch_scalar(kv, arch, ".context_length"),
        block_count=_arch_scalar(kv, arch, ".block_count"),
        embedding_length=_arch_scalar(kv, arch, ".embedding_length"),
        head_count=_arch_scalar(kv, arch, ".attention.head_count"),
        head_count_kv=_arch_scalar(kv, arch, ".attention.head_count_kv"),
        key_length=_arch_scalar(kv, arch, ".attention.key_length"),
        meta_truncated=truncated,
    )
    meta.head_count_kv_from_array = f"{arch}.attention.head_count_kv" in array_keys
    meta.is_embedding = bool(
        _arch_scalar(kv, arch, ".pooling_type") is not None
        or "embed" in name.lower()
        or "embed" in arch.lower()
    )
    if truncated:
        _LOG.debug("gguf 元数据在头部预算内未读完: %s (kv=%d)", p, kv_count)
    return meta


def estimate_kv_cache_bytes(meta: GgufMeta, n_ctx: int) -> int | None:
    """估算 KV cache 字节数（f16，上界）；关键字段缺失返回 None。"""
    layers = meta.block_count
    embed = meta.embedding_length
    heads = meta.head_count
    if not layers or not embed or not heads or heads <= 0 or n_ctx <= 0:
        return None
    head_dim = meta.key_length or max(1, embed // heads)
    kv_heads = meta.head_count_kv or heads
    return layers * kv_heads * head_dim * n_ctx * 2 * _KV_DTYPE_BYTES


@dataclass
class VramEstimate:
    """单个模型在给定上下文下的运行显存估算与可行性判定。"""

    weights_bytes: int = 0
    kv_cache_bytes: int | None = None
    overhead_bytes: int = _DEFAULT_OVERHEAD_BYTES
    needed_bytes: int | None = None
    free_vram_bytes: int | None = None
    verdict: str = "unknown"  # full_gpu / tight / partial_offload / infeasible / unknown
    advice: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights_bytes": self.weights_bytes,
            "kv_cache_bytes": self.kv_cache_bytes,
            "overhead_bytes": self.overhead_bytes,
            "needed_bytes": self.needed_bytes,
            "free_vram_bytes": self.free_vram_bytes,
            "verdict": self.verdict,
            "advice": self.advice,
            "notes": list(self.notes),
        }


def _gib(n: int) -> str:
    return f"{n / _GIB:.2f} GB"


def estimate_vram(
    meta: GgufMeta,
    n_ctx: int,
    *,
    free_vram_bytes: int | None = None,
    overhead_bytes: int = _DEFAULT_OVERHEAD_BYTES,
) -> VramEstimate:
    """估算模型在 ``n_ctx`` 下所需显存，并对 ``free_vram_bytes`` 给出可行性判定。

    ``free_vram_bytes=None``（未探测到 NVIDIA GPU）时只返回需求侧数字，
    ``verdict`` 保持 ``unknown``——不猜硬件。
    """
    est = VramEstimate(
        weights_bytes=meta.size_bytes,
        overhead_bytes=overhead_bytes,
        free_vram_bytes=free_vram_bytes,
    )
    if not meta.ok or meta.size_bytes <= 0:
        est.verdict = "unknown"
        est.advice = "模型元数据不可用，无法估算显存。"
        return est

    kv_bytes = estimate_kv_cache_bytes(meta, n_ctx)
    if kv_bytes is None:
        est.notes.append("缺少层数/隐藏维/头数元数据，KV cache 未计入，实际占用会更高。")
        kv_bytes = 0
    est.kv_cache_bytes = kv_bytes
    if meta.head_count_kv_from_array:
        est.notes.append(
            "KV 头数按逐层数组取最大值估算（偏保守）；滑动窗口注意力模型实际占用更低。"
        )
    if meta.meta_truncated:
        est.notes.append("元数据未在头部预算内读完，部分字段可能缺失。")

    needed = meta.size_bytes + kv_bytes + overhead_bytes
    est.needed_bytes = needed

    if free_vram_bytes is None:
        est.verdict = "unknown"
        est.advice = f"预计需要 {_gib(needed)}（含 {n_ctx} 上下文），未探测到 GPU 显存无法判定。"
        return est

    if meta.size_bytes > free_vram_bytes:
        est.verdict = "infeasible"
        est.advice = (
            f"权重 {_gib(meta.size_bytes)} 已超可用显存 {_gib(free_vram_bytes)}，"
            "无法全量上卡；即使强行启动也会退回 CPU 推理，速度下降一个量级。"
        )
        return est

    if needed > free_vram_bytes:
        est.verdict = "partial_offload"
        est.advice = (
            f"权重能放下（{_gib(meta.size_bytes)}），但含 {n_ctx} 上下文共需 "
            f"{_gib(needed)}，超可用 {_gib(free_vram_bytes)}；需降低上下文或部分层走 CPU。"
        )
        return est

    reserve = free_vram_bytes - needed
    if reserve < _TIGHT_RESERVE_BYTES:
        est.verdict = "tight"
        est.advice = f"可全量上卡，但仅剩 {_gib(reserve)} 余量，与检测模型（YOLO）并存会紧张。"
        return est

    est.verdict = "full_gpu"
    est.advice = f"可全量上卡，预计占用 {_gib(needed)}，余量 {_gib(reserve)}。"
    return est


__all__ = [
    "GGUF_MAGIC",
    "HEADER_READ_BYTES",
    "GgufMeta",
    "VramEstimate",
    "estimate_kv_cache_bytes",
    "estimate_vram",
    "read_gguf_meta",
]
