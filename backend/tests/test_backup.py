"""备份/恢复基础设施测试（产品化基础：数据治理原子能力）。

覆盖：
- create_backup 打包含 manifest + 每条目 size/sha256，缺失源跳过；
- verify_backup 对损坏/被篡改/缺条目归档抛错；
- restore_backup 校验后原子回写，篡改内容不落盘；
- /api/v1/system/backup + /restore 端点到端到到跑通（round-trip 后清理产物）。

纯函数部分不依赖全局 registry，完全隔离；API 部分复用 TestClient 共享 registry，
因 restore 只动模型注册表/漂移基线（回写与备份内容一致，幂等），并把产生的
归档在断言后删除，避免污染真实 data/backups。
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.infra.backup import create_backup, restore_backup, verify_backup


def test_create_verify_restore_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("hello", encoding="utf-8")
    (src / "b.json").write_text(json.dumps({"k": 1}), encoding="utf-8")
    archive = tmp_path / "out" / "backup.zip"
    sink = tmp_path / "sink"
    sink.mkdir()

    sources = {"a.txt": src / "a.txt", "b.json": src / "b.json"}
    result = create_backup(sources, archive, app_version="1.2.3")

    assert archive.is_file()
    assert result["skipped"] == []
    assert result["manifest"]["app_version"] == "1.2.3"
    assert set(result["manifest"]["entries"]) == {"a.txt", "b.json"}

    # 归档整体哈希可复算
    manifest = verify_backup(archive)
    assert manifest["entries"]["a.txt"]["size"] == 5

    restored = restore_backup(archive, {"a.txt": sink / "a.txt", "b.json": sink / "b.json"})
    assert set(restored["entries"]) == {"a.txt", "b.json"}
    assert (sink / "a.txt").read_text(encoding="utf-8") == "hello"
    assert json.loads((sink / "b.json").read_text(encoding="utf-8")) == {"k": 1}


def test_create_backup_skips_missing_source(tmp_path: Path) -> None:
    archive = tmp_path / "backup.zip"
    existing = tmp_path / "existing.json"
    existing.write_text(json.dumps({1: 2}), encoding="utf-8")
    result = create_backup(
        {"existing": existing, "missing": tmp_path / "nope.txt"},
        archive,
    )
    assert result["skipped"] == ["missing"]
    assert set(result["manifest"]["entries"]) == {"existing"}


def test_verify_backup_rejects_tampered_entry(tmp_path: Path) -> None:
    archive = tmp_path / "backup.zip"
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("original", encoding="utf-8")
    create_backup({"a.txt": src / "a.txt"}, archive)

    # 篡改归档内条目内容（保持文件名不变）
    with zipfile.ZipFile(archive, "a") as zf:
        zf.writestr("a.txt", b"TAMPERED!!")

    try:
        verify_backup(archive)
    except ValueError as exc:
        assert "integrity mismatch" in str(exc) or "missing entry" in str(exc)
    else:
        raise AssertionError("expected tamper detection to fail")


def test_verify_backup_rejects_missing_manifest(tmp_path: Path) -> None:
    archive = tmp_path / "no_manifest.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("data.png", b"x")
    try:
        verify_backup(archive)
    except ValueError as exc:
        assert "manifest" in str(exc)
    else:
        raise AssertionError("expected missing-manifest failure")


def test_restore_backup_does_not_partially_overwrite_on_tamper(tmp_path: Path) -> None:
    archive = tmp_path / "backup.zip"
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("good", encoding="utf-8")
    create_backup({"a.txt": src / "a.txt"}, archive)

    # 篡改归档后再恢复：restore_backup 先整体校验，须在写任何目标前失败
    with zipfile.ZipFile(archive, "a") as zf:
        zf.writestr("a.txt", b"EVIL")
    sink = tmp_path / "sink"
    sink.mkdir()
    try:
        restore_backup(archive, {"a.txt": sink / "a.txt"})
    except ValueError:
        pass
    else:
        raise AssertionError("expected restore to fail on invalid archive")
    # 目标不得被创建/覆盖（半包不留盘）
    assert not (sink / "a.txt").exists()


_APP_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _APP_ROOT / "data"


def test_backup_restore_api_roundtrip() -> None:
    with TestClient(app) as client:
        bk = client.post("/api/v1/system/backup")
        assert bk.status_code == 200, bk.text
        body = bk.json()
        assert body["archive"].startswith("scan_backup_")
        assert body["archive"].endswith(".zip")
        assert isinstance(body["archive_sha256"], str) and len(body["archive_sha256"]) == 64
        assert "scan.db" in body["entries"]

        archive_name = body["archive"]
        try:
            # restore 即时回写非 DB 状态；DB 需重启生效
            rs = client.post("/api/v1/system/restore", json={"archive": archive_name})
            assert rs.status_code == 200, rs.text
            rbody = rs.json()
            assert rbody["integrity_ok"] is True
            assert rbody["db_restore"] == "pending_restart"
            assert "model_registry.json" in rbody["restored"]
        finally:
            created = _DATA_DIR / "backups" / archive_name
            if created.is_file():
                created.unlink()

    # 篡改后的归档应被 restore 拒绝（404/校验失败层面）
    with TestClient(app) as client:
        bad = client.post("/api/v1/system/restore", json={"archive": "does_not_exist.zip"})
        assert bad.status_code == 404


def test_api_rejects_traversal_archive_name() -> None:
    """路径穿越防护：归档名不得越出 backups 目录。"""
    with TestClient(app) as client:
        res = client.post("/api/v1/system/restore", json={"archive": "../evil.zip"})
        assert res.status_code == 422
