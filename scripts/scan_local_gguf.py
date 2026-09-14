"""本地 GGUF 模型发现（探路版，零第三方依赖）。

用途：扫描本机固定盘定位全部 .gguf，并解析 GGUF 头部元数据（名称/架构/量化/上下文）。
同时作为后端 `backend/infra/gguf_meta.py` 的算法验证原型。

安全边界（与项目硬化口径一致）：
- 只读：不修改、不移动、不删除任何文件；
- 只认 *.gguf，先校验 GGUF magic 再解析元数据；
- 不跟随符号链接 / junction（防目录循环与越界）；
- 跳过系统与易变目录（Windows / $Recycle.Bin / System Volume Information 等）；
- 权限错误计数跳过，不中断整体扫描；
- 仅读文件头部若干 MB，绝不整文件读入（14B 权重单文件 8GB+）。

用法：python scripts/scan_local_gguf.py [--roots C: D: E:]
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import time
from pathlib import Path

# 跳过：系统目录 / 回收站 / 卷影 / 临时 / 版本控制 / 依赖目录
SKIP_DIR_NAMES = {
    "$recycle.bin",
    "system volume information",
    "$windows.~bt",
    "$windows.~ws",
    "recovery",
    "perflogs",
    "msocache",
    "config.msi",
    "winsxs",
    "temp",
    "node_modules",
    ".git",
    ".cargo",
    ".rustup",
}

GGUF_MAGIC = b"GGUF"
HEADER_READ_BYTES = 64 * 1024 * 1024  # 元数据在文件前部；大 tokenizer 词表 + MoE 模型 KV 区
# 可达数十 MB（实测 gpt-oss-20b 超 8MB 即截断，30B MoE 需 64MB 才完整；单文件串行读取，峰值可控）

# GGUF 元数据类型枚举（llama.cpp gguf.h）
_T_UINT8, _T_INT8, _T_UINT16, _T_INT16 = 0, 1, 2, 3
_T_UINT32, _T_INT32, _T_FLOAT32, _T_BOOL = 4, 5, 6, 7
_T_STRING, _T_ARRAY, _T_UINT64, _T_INT64, _T_FLOAT64 = 8, 9, 10, 11, 12

_SCALAR_FMT = {
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

_QUANT_BY_FILE_TYPE = {
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

# 我们关心的元数据键（其余跳过，控制解析成本）
WANTED_KEYS = {
    "general.name",
    "general.architecture",
    "general.file_type",
    "general.parameter_count",
    "general.size_label",
    "split.count",
}


class _Reader:
    """带边界的顺序读取器（越界即停，不抛到调用方）。"""

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

    def scalar(self, fmt: str) -> object:
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.read(size))[0]

    def string(self) -> str:
        n = self.scalar("<Q")
        return self.read(int(n)).decode("utf-8", "replace")

    def skip_value(self, vtype: int) -> None:
        """跳过非目标值；数组递归跳过。"""
        if vtype in _SCALAR_FMT:
            fmt, size = _SCALAR_FMT[vtype]
            self.read(size)
            return
        if vtype == _T_STRING:
            n = self.scalar("<Q")
            self.read(int(n))
            return
        if vtype == _T_ARRAY:
            elem_type = self.scalar("<I")
            count = self.scalar("<Q")
            for _ in range(int(count)):
                self.skip_value(int(elem_type))
            return
        raise ValueError(f"unknown gguf value type {vtype}")


def read_gguf_meta(path: Path) -> dict:
    """解析 GGUF 头部 → 元数据摘要。失败返回 {'error': 原因}，绝不抛异常。"""
    try:
        with path.open("rb") as fh:
            head = fh.read(HEADER_READ_BYTES)
    except OSError as exc:
        return {"error": f"read_failed: {exc.__class__.__name__}"}

    if len(head) < 24 or head[:4] != GGUF_MAGIC:
        return {"error": "not_gguf"}

    r = _Reader(head)
    try:
        r.read(4)  # magic
        version = int(r.scalar("<I"))
        tensor_count = int(r.scalar("<Q"))
        kv_count = int(r.scalar("<Q"))
    except (EOFError, struct.error):
        return {"error": "truncated_header"}

    meta: dict = {
        "gguf_version": version,
        "tensor_count": tensor_count,
        "kv_count": kv_count,
    }
    arch = ""
    for _ in range(kv_count):
        try:
            key = r.string()
            vtype = int(r.scalar("<I"))
            interesting = key in WANTED_KEYS or key.endswith(
                (".context_length", ".block_count", ".embedding_length")
            )
            if interesting and vtype == _T_STRING:
                # 架构名/模型名是字符串；此前只判数值类型导致 architecture 恒空
                meta[key] = r.string()
                if key == "general.architecture":
                    arch = str(meta[key])
            elif interesting and vtype in _SCALAR_FMT:
                fmt, _size = _SCALAR_FMT[vtype]
                meta[key] = r.scalar(fmt)
            else:
                r.skip_value(vtype)
        except (EOFError, struct.error, ValueError, MemoryError):
            meta["meta_truncated"] = True
            break

    file_type = meta.get("general.file_type")
    if isinstance(file_type, int):
        meta["quant"] = _QUANT_BY_FILE_TYPE.get(file_type, f"file_type_{file_type}")
    if arch:
        meta.setdefault("ctx_len", meta.get(f"{arch}.context_length"))
        meta.setdefault("block_count", meta.get(f"{arch}.block_count"))
        meta.setdefault("embedding_length", meta.get(f"{arch}.embedding_length"))
    return meta


def is_reparse_point(entry: os.DirEntry) -> bool:
    """junction / symlink 检测：不跟随，避免循环与越界。"""
    if entry.is_symlink():
        return True
    try:
        return bool(entry.stat(follow_symlinks=False).st_file_attributes & 0x400)  # REPARSE_POINT
    except (OSError, AttributeError):
        return False


def scan(roots: list[Path]) -> tuple[list[dict], dict]:
    found: list[dict] = []
    stats = {"dirs": 0, "permission_errors": 0, "skipped_dirs": 0}
    stack = [r for r in roots if r.exists()]

    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if entry.name.lower() in SKIP_DIR_NAMES or is_reparse_point(entry):
                                stats["skipped_dirs"] += 1
                                continue
                            stack.append(Path(entry.path))
                            stats["dirs"] += 1
                        elif entry.name.lower().endswith(".gguf"):
                            st = entry.stat(follow_symlinks=False)
                            found.append(
                                {
                                    "path": entry.path,
                                    "size_bytes": st.st_size,
                                    "mtime": st.st_mtime,
                                }
                            )
                    except (OSError, PermissionError):
                        stats["permission_errors"] += 1
        except (PermissionError, OSError):
            stats["permission_errors"] += 1

    return found, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=None, help="扫描根（默认全部固定盘）")
    args = ap.parse_args()

    if args.roots:
        roots = [Path(r) for r in args.roots]
    else:
        roots = [Path(f"{d}:\\") for d in "CDEFGH" if Path(f"{d}:\\").exists()]

    print(f"扫描根: {[str(r) for r in roots]}", flush=True)
    t0 = time.perf_counter()
    found, stats = scan(roots)
    elapsed = time.perf_counter() - t0

    print(f"\n耗时 {elapsed:.1f}s | 遍历目录 {stats['dirs']} | 跳过 {stats['skipped_dirs']} | 权限错误 {stats['permission_errors']}")
    print(f"命中 .gguf：{len(found)} 个\n")

    for item in sorted(found, key=lambda d: -d["size_bytes"]):
        p = Path(item["path"])
        meta = read_gguf_meta(p)
        size_gb = item["size_bytes"] / 1073741824
        if "error" in meta:
            detail = f"解析失败({meta['error']})"
        else:
            detail = "{arch} {quant} ctx={ctx} layers={layers}".format(
                arch=meta.get("general.architecture", "?"),
                quant=meta.get("quant", "?"),
                ctx=meta.get("ctx_len", "?"),
                layers=meta.get("block_count", "?"),
            )
        print(f"  {size_gb:7.2f} GB  {detail}")
        print(f"            {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
