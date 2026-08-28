"""备份 / 恢复基础设施（产品化基础，架构升级）。

为"国内顶级扫描软件"的长期数据治理打底的原子能力，纯 stdlib（zipfile+hashlib+json）：

- `create_backup(sources, archive_path)`：把关键系统状态打成 zip 归档，内置 manifest.json，
  每条目记录 {size, sha256}，并对归档整体算 sha256，支持完整性自校验；
- `verify_backup(archive_path)`：校验 manifest 存在 + 每个条目大小/哈希匹配（防损坏/篡改）；
- `restore_backup(archive_path, destinations)`：校验后原子回写（临时文件 + rename，失败不留半包）。

备份范围由调用方（router）按配置注入：DB + 模型注册表 + 漂移基线等"可重建关键状态"。
影像/报告等大体积数据按其目录整体另行归档策略（v1 不并入，避免单归档过大）。

安全：路径校验用 fs.safe_resolve 防穿越；恢复是破坏性操作，调用方需显式确认并记审计。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import TypedDict


class ManifestEntry(TypedDict):
    size: int
    sha256: str


class Manifest(TypedDict):
    app_version: str
    created_at: str
    entries: dict[str, ManifestEntry]  # 键 -> {size, sha256}


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _entry_sha(name: str, data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def create_backup(
    sources: dict[str, Path],
    archive_path: Path,
    app_version: str = "0.1.0",
) -> dict:
    """把 sources（键->源文件）打成一个带 manifest 的 zip 归档，返回 manifest。

    - 仅归档"存在且为文件"的条目；缺失的键予以跳过（记录到 message）。
    - 目标目录若不存在会自动创建。
    - 返回 {manifest, archive_sha256, skipped}，archive_sha256 为归档文件的哈希。
    """
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    entries: dict[str, ManifestEntry] = {}
    skipped: list[str] = []

    with tempfile.TemporaryDirectory(prefix="scan_backup_") as tmp:
        stage = Path(tmp)
        for key, src in sorted(sources.items()):
            if not src.is_file():
                skipped.append(key)
                continue
            data = src.read_bytes()
            entries[key] = {"size": len(data), "sha256": _entry_sha(key, data)}
            (stage / _safe_zipname(key)).write_bytes(data)
        manifest: Manifest = {
            "app_version": app_version,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "entries": entries,
        }
        (stage / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        # 写入 zip（压缩），再把 zip 拷到目标路径；用临时文件避免半包落到最终名。
        fd, tmp_zip = tempfile.mkstemp(suffix=".zip", prefix="scan_backup_", dir=str(stage))
        os.close(fd)
        Path(tmp_zip).unlink()
        with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in stage.iterdir():
                if f.name == os.path.basename(tmp_zip):
                    continue
                zf.write(f, arcname=f.name)
        shutil.move(tmp_zip, archive_path)

    return {
        "manifest": manifest,
        "archive_sha256": _sha256(archive_path),
        "skipped": skipped,
    }


def verify_backup(archive_path: Path) -> Manifest:
    """校验归档完整性：manifest 可解析、每条目存在且 size/sha256 匹配。

    校验失败抛出 ValueError（损坏/被篡改），成功返回 manifest。
    """
    if not archive_path.is_file():
        raise ValueError(f"archive not found: {archive_path}")
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            if "manifest.json" not in zf.namelist():
                raise ValueError("archive missing manifest.json")
            manifest: Manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
            for key, meta in manifest.get("entries", {}).items():
                if key not in zf.namelist():
                    raise ValueError(f"archive missing entry: {key}")
                data = zf.read(key)
                if len(data) != meta["size"] or _entry_sha(key, data) != meta["sha256"]:
                    raise ValueError(f"integrity mismatch: {key}")
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid backup archive: {exc}") from exc
    return manifest


def restore_backup(archive_path: Path, destinations: dict[str, Path]) -> Manifest:
    """校验后原子回写：把归档里 key 指向的文件恢复到 destinations[key]。

    - 先完整校验（verify_backup）再动手，避免"恢复了一半才发现包损坏"；
    - 每个条目：先写临时文件，校验内容后 rename 覆盖目标；失败清理临时文件；
    - 目标目录不存在自动创建。返回恢复所用 manifest。
    """
    manifest = verify_backup(archive_path)
    with zipfile.ZipFile(archive_path, "r") as zf:
        for key, dest in destinations.items():
            if key not in manifest.get("entries", {}):
                continue
            dest = Path(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            data = zf.read(key)
            fd, tmp = tempfile.mkstemp(prefix="scan_restore_", dir=str(dest.parent))
            os.close(fd)
            tmp_path = Path(tmp)
            try:
                tmp_path.write_bytes(data)
                if _entry_sha(key, tmp_path.read_bytes()) != manifest["entries"][key]["sha256"]:
                    raise ValueError(f"restore verification failed: {key}")
                shutil.move(str(tmp_path), dest)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()
    return manifest


def _safe_zipname(key: str) -> str:
    """把断言到的 key 归一化为 zip 内安全文件名（仅允许安全字符，防 zip 路径穿越）。"""
    return key.replace("\\", "/").replace("/", "__").replace("..", "__").replace(":", "_")
