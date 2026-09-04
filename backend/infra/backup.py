"""备份 / 恢复基础设施（产品化基础，架构升级 + S-12 增强）。

为"国内顶级扫描软件"的长期数据治理打底的原子能力（zipfile+hashlib+json，
SM3 经 infra/crypto 复用）：

- `create_backup(sources, archive_path, ...)`：把关键系统状态打成 zip 归档，
  内置 manifest.json。S-12 起条目哈希默认 SM3（国密，复用 infra.crypto.sm3_hex），
  manifest 记 ``hash_algo`` 字段；旧 manifest（无该字段）按 sha256 校验——
  双算法兼容读取。归档整体哈希仍为 sha256（传输层校验值，与 API 字段
  archive_sha256 契约一致）。
  可选 ``dirs``（键→目录）：把影像等目录逐文件纳入归档（S-12c，键前缀/相对路径），
  每个文件独立 SM3 校验。
- `verify_backup(archive_path)`：校验 manifest 存在 + 每个条目大小/哈希匹配
  （按 manifest.hash_algo 选算法，防损坏/篡改）；
- `restore_backup(archive_path, destinations)`：校验后原子回写（临时文件 + rename，
  失败不留半包）；
- `BackupScheduler`：S-12a 定期备份调度（后台线程，interval_hours>0 时由应用
  lifespan 启动；0=关闭。动作经回调入审计）。

备份范围由调用方（router）按配置注入：DB + 模型注册表 + 漂移基线等"可重建关键状态"。

安全：恢复是破坏性操作，调用方需显式确认并记审计。SM3 为纯 Python 实现
（约 1~2 MB/s 量级），大体积影像纳入备份时归档/校验耗时显著上升，属诚实边界。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

from backend.infra.timeutil import fmt_naive_utc


class ManifestEntry(TypedDict):
    size: int
    sha256: str  # 字段名为历史兼容名；实际算法由 manifest.hash_algo 决定（S-12）


class Manifest(TypedDict):
    app_version: str
    created_at: str
    hash_algo: str  # 条目哈希算法：sm3（新）| sha256（旧 manifest 缺省）
    entries: dict[str, ManifestEntry]  # 键 -> {size, <hash_algo>}


def _sm3_hex(data: bytes) -> str:
    """SM3 摘要（复用国密 provider 层原语，S-12b）。"""
    from backend.infra.crypto import sm3_hex

    return sm3_hex(data)


def _hash_bytes(data: bytes, algo: str) -> str:
    if algo == "sm3":
        return _sm3_hex(data)
    if algo == "sha256":
        return hashlib.sha256(data).hexdigest()
    raise ValueError(f"不支持的备份哈希算法: {algo}（支持 sha256 | sm3）")


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def create_backup(
    sources: dict[str, Path],
    archive_path: Path,
    app_version: str = "0.1.0",
    *,
    hash_algo: str = "sm3",
    dirs: dict[str, Path] | None = None,
) -> dict:
    """把 sources（键->源文件）打成一个带 manifest 的 zip 归档，返回 manifest。

    - 仅归档"存在且为文件"的条目；缺失的键予以跳过（记录到 message）；
    - ``dirs``（S-12c）：键->目录，递归纳入目录下全部文件，条目键为
      ``<键>/<相对路径>``；目录不存在整个键跳过；
    - 条目哈希算法由 ``hash_algo`` 指定（默认 sm3，S-12b），manifest 记 hash_algo；
    - 目标目录若不存在会自动创建；
    - 返回 {manifest, archive_sha256, skipped}，archive_sha256 为归档文件的
      sha256（整体传输校验，字段名与既有 API 契约保持一致）。
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
            entries[key] = {"size": len(data), "sha256": _hash_bytes(data, hash_algo)}
            (stage / _safe_zipname(key)).write_bytes(data)
        for key, d in sorted((dirs or {}).items()):
            if not d.is_dir():
                skipped.append(key)
                continue
            for f in sorted(p for p in d.rglob("*") if p.is_file()):
                rel = f.relative_to(d).as_posix()
                ekey = f"{key}/{rel}"
                data = f.read_bytes()
                entries[ekey] = {"size": len(data), "sha256": _hash_bytes(data, hash_algo)}
                (stage / _safe_zipname(ekey)).write_bytes(data)
        manifest: Manifest = {
            "app_version": app_version,
            "created_at": fmt_naive_utc(),
            "hash_algo": hash_algo,
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
    """校验归档完整性：manifest 可解析、每条目存在且 size/哈希匹配。

    哈希算法按 manifest ``hash_algo`` 选择（S-12b）：缺省（旧 manifest）按
    sha256 校验，存量归档仍可验证。校验失败抛 ValueError（损坏/被篡改），
    成功返回 manifest。
    """
    if not archive_path.is_file():
        raise ValueError(f"archive not found: {archive_path}")
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            if "manifest.json" not in zf.namelist():
                raise ValueError("archive missing manifest.json")
            manifest: Manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
            algo = manifest.get("hash_algo") or "sha256"
            for key, meta in manifest.get("entries", {}).items():
                if key not in zf.namelist():
                    raise ValueError(f"archive missing entry: {key}")
                data = zf.read(key)
                if len(data) != meta["size"] or _hash_bytes(data, algo) != meta["sha256"]:
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
    algo = manifest.get("hash_algo") or "sha256"
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
                if _hash_bytes(tmp_path.read_bytes(), algo) != manifest["entries"][key]["sha256"]:
                    raise ValueError(f"restore verification failed: {key}")
                shutil.move(str(tmp_path), dest)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()
    return manifest


def _safe_zipname(key: str) -> str:
    """把断言到的 key 归一化为 zip 内安全文件名（仅允许安全字符，防 zip 路径穿越）。"""
    return key.replace("\\", "/").replace("/", "__").replace("..", "__").replace(":", "_")


class BackupScheduler:
    """S-12a 定期备份调度：后台守护线程按固定间隔执行备份任务。

    - interval_hours<=0 时不启动（调用方判断，默认 0=关闭，不影响测试）；
    - job 由调用方注入（含审计/告警副作用），调度器只管节拍与生命周期；
    - stop() 置位事件，线程在下次等待点退出（不硬杀正在执行的备份）。
    """

    def __init__(self, interval_hours: float, job: Callable[[], None]) -> None:
        if interval_hours <= 0:
            raise ValueError("interval_hours 必须 > 0（0 表示关闭调度，不构造本类）")
        self._interval = float(interval_hours) * 3600.0
        self._job = job
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_run_at: str | None = None
        self.last_error: str | None = None
        self.run_count = 0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="backup-scheduler", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        # 首轮先等待一个完整间隔再执行（应用启动即备份属冷启动负担，且
        # 手动 POST /system/backup 已覆盖"立即备份"诉求）。
        while not self._stop.wait(self._interval):
            try:
                self._job()
                self.run_count += 1
                self.last_run_at = fmt_naive_utc()
                self.last_error = None
            except Exception as exc:  # noqa: BLE001 - 定时备份失败只记录，不拖垮进程
                self.last_error = str(exc)[:200]

    def snapshot(self) -> dict:
        """调度器状态快照（/health watchdog 相邻字段同风格，可观测）。"""
        return {
            "enabled": self._thread is not None and self._thread.is_alive(),
            "interval_hours": self._interval / 3600.0,
            "run_count": self.run_count,
            "last_run_at": self.last_run_at,
            "last_error": self.last_error,
        }
