"""系统运维端点（产品化基础，架构升级）：备份 / 恢复 / 网络状态自证。

POST /api/v1/system/backup   打包关键系统状态（DB + 模型注册表 + 漂移基线）→ data/backups/*.zip，
                              返回 manifest + 归档整体哈希（可校验完整性）；
POST /api/v1/system/restore  按归档文件名恢复：非 DB 状态即时原子回写，
                              DB 因运行时占用改由"重启后生效"（仅校验完整性，不冒险热替换）；
GET  /api/v1/system/network-status  纯离线自证（C-15）：离线模式结论 + 外联事件计数。

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
from backend.infra.config import AppConfig, resolve_config_path
from backend.infra.fs import safe_resolve

router = APIRouter(prefix="/system", tags=["system"])


def offline_conclusion(config: AppConfig) -> dict:
    """C-15 纯离线自检：由静态配置得出"无外网依赖"结论（启动自检与端点共用）。

    判定（诚实边界：这是软件侧配置层的自证，物理断网须由 OS/主机防火墙
    出站默认拒绝兜底，见 docs/deployment-baseline.md）：
    - offline_mode = sync.kind == local（同步通道不留任何外发出口）；
    - egress_guard_enabled = egress.enabled（进程级外联拦截是否在岗）。
    """
    return {
        "offline_mode": config.sync.kind == "local",
        "sync_kind": config.sync.kind,
        "egress_guard_enabled": bool(config.egress.enabled),
    }


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


class NetworkStatusResponse(BaseModel):
    """C-15 纯离线自证快照（GET /system/network-status）。"""

    offline_mode: bool  # sync.kind=local 即离线模式（数据不出本机）
    sync_kind: str  # 当前同步通道种类
    egress_guard_enabled: bool  # 进程级外联防护是否在岗（C-16）
    egress_blocked_events: int  # 外联拦截事件数（alerts 表持久计数，跨重启累计）


def _resolve_sources(reg: Registry) -> dict[str, Path]:
    """备份目标（"可重建关键状态"）：DB + 模型注册表 + 漂移基线。"""
    return {
        "scan.db": resolve_config_path(reg.config.paths.db_path),
        "model_registry.json": resolve_config_path(reg.config.model.registry_state_file),
        "drift_baseline.json": resolve_config_path(reg.config.eval.drift_baseline_path),
    }


def _image_dirs(reg: Registry) -> dict[str, Path]:
    """S-12c：备份.include_images=true 时纳入影像目录（逐文件 SM3 校验）。"""
    if not reg.config.backup.include_images:
        return {}
    images = resolve_config_path(reg.config.paths.images_dir)
    return {"images": images} if images.is_dir() else {}


def run_backup(reg: Registry, actor: str = "system", note: str = "system backup created") -> dict:
    """执行一次备份并记审计（手动端点与 S-12a 定期调度共用）。返回结果 dict。"""
    backups_dir = resolve_config_path(reg.config.paths.data_dir) / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    archive_path = backups_dir / f"scan_backup_{ts}.zip"

    result = create_backup(
        _resolve_sources(reg),
        archive_path,
        app_version="0.1.0",
        dirs=_image_dirs(reg),
    )
    manifest = result["manifest"]

    reg.repository.append_audit(
        actor=actor,
        action="backup_create",
        object_type="backup",
        object_id=archive_path.name,
        before=None,
        after={
            "sha256": result["archive_sha256"],
            "entries": len(manifest["entries"]),
            "hash_algo": manifest.get("hash_algo"),
        },
        note=note,
    )
    result["archive"] = archive_path.name  # 供端点响应（restore 按 data/backups 下文件名定位）
    return result


@router.post("/backup", response_model=BackupResponse)
def backup(reg: Annotated[Registry, Depends(get_registry)]) -> BackupResponse:
    result = run_backup(reg)
    manifest = result["manifest"]
    return BackupResponse(
        archive=result["archive"],
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


@router.get("/network-status", response_model=NetworkStatusResponse)
def network_status(reg: Annotated[Registry, Depends(get_registry)]) -> NetworkStatusResponse:
    """纯离线自证（C-15）：返回离线模式结论与外联拦截事件计数。

    结论由静态配置得出（sync.kind / egress.enabled），拦截计数取 alerts 表
    持久统计（kind=egress_blocked，含历史进程），供保密检查现场一键取证。
    """
    conclusion = offline_conclusion(reg.config)
    return NetworkStatusResponse(
        **conclusion,
        egress_blocked_events=reg.security_store.count_alerts(kind="egress_blocked"),
    )
