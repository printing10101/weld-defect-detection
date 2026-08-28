"""系统运维端点（产品化基础，架构升级）：备份 / 恢复。

POST /api/v1/system/backup   打包关键系统状态（DB + 模型注册表 + 漂移基线）→ data/backups/*.zip，
                              返回 manifest + 归档整体哈希（可校验完整性）；
POST /api/v1/system/restore  按归档文件名恢复：非 DB 状态即时原子回写，
                              DB 因运行时占用改由"重启后生效"（仅校验完整性，不冒险热替换）。

边界：这些是"数据治理"入口，为将来接入机构版/远程管理留位；当前本地单机，操作均记审计。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.app.dependencies import Registry, get_registry
from backend.infra.backup import create_backup, restore_backup, verify_backup
from backend.infra.config import resolve_config_path
from backend.infra.fs import safe_resolve

router = APIRouter(prefix="/system", tags=["system"])


class BackupResponse(BaseModel):
    archive: str  # 归档文件名（相对 backups 目录）
    archive_sha256: str
    entries: dict[str, dict]
    skipped: list[str]
    created_at: str


class RestoreRequest(BaseModel):
    archive: str = Field(description="data/backups 下的归档文件名")


class RestoreResponse(BaseModel):
    restored: list[str]  # 已即时恢复的状态键
    db_restore: str  # "done"=已恢复 | "pending_restart"=需重启生效
    integrity_ok: bool


def _resolve_sources(reg: Registry) -> dict[str, Path]:
    """备份目标（"可重建关键状态"）：DB + 模型注册表 + 漂移基线。"""
    return {
        "scan.db": resolve_config_path(reg.config.paths.db_path),
        "model_registry.json": resolve_config_path(reg.config.model.registry_state_file),
        "drift_baseline.json": resolve_config_path(reg.config.eval.drift_baseline_path),
    }


@router.post("/backup", response_model=BackupResponse)
def backup(reg: Annotated[Registry, Depends(get_registry)]) -> BackupResponse:
    backups_dir = resolve_config_path(reg.config.paths.data_dir) / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    archive_path = backups_dir / f"scan_backup_{ts}.zip"

    result = create_backup(_resolve_sources(reg), archive_path, app_version="0.1.0")
    manifest = result["manifest"]

    reg.repository.append_audit(
        actor="system",
        action="backup_create",
        object_type="backup",
        object_id=archive_path.name,
        before=None,
        after={"sha256": result["archive_sha256"], "entries": len(manifest["entries"])},
        note="system backup created",
    )
    return BackupResponse(
        archive=archive_path.name,
        archive_sha256=result["archive_sha256"],
        entries=manifest["entries"],
        skipped=result["skipped"],
        created_at=manifest["created_at"],
    )


@router.post("/restore", response_model=RestoreResponse)
def restore(
    req: RestoreRequest,
    reg: Annotated[Registry, Depends(get_registry)],
) -> RestoreResponse:
    backups_dir = resolve_config_path(reg.config.paths.data_dir) / "backups"
    try:
        archive_path = safe_resolve(backups_dir, req.archive)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not archive_path.is_file():
        raise HTTPException(status_code=404, detail="archive not found")

    verify_backup(archive_path)  # 先完整校验（含 SHA），失败不落任何破坏

    sources = _resolve_sources(reg)
    # 即时恢复安全的非 DB 状态；DB 因运行时占用改由重启后生效（仅校验完整性）。
    db_key = "scan.db"
    non_db = {k: v for k, v in sources.items() if k != db_key}
    restored = list(restore_backup(archive_path, non_db).get("entries", {}).keys())

    reg.repository.append_audit(
        actor="system",
        action="backup_restore",
        object_type="backup",
        object_id=archive_path.name,
        before=None,
        after={"restored": restored, "db": "pending_restart"},
        note="system restore (db requires restart)",
    )
    return RestoreResponse(
        restored=restored,
        db_restore="pending_restart",
        integrity_ok=True,
    )
