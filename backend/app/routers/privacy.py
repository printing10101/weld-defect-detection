"""脱敏残留审计端点（C-13）：对影像库/指定目录跑 PHI + EXIF 残留扫描。

POST /privacy/audit {directory?}：扫描（默认 paths.images_dir），生成
JSON + PDF 残留审计报告落 data/privacy/，动作入主审计链；报告供安全
审计员归档查阅（C-06 权限矩阵：自检报告）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from backend.app.auth import Principal, require_role
from backend.app.dependencies import Registry, get_registry
from backend.infra.config import resolve_config_path
from backend.infra.privacy_audit import audit_directory_phi, write_audit_report

router = APIRouter(prefix="/privacy", tags=["privacy"])


class PrivacyAuditIn(BaseModel):
    directory: str | None = None  # 缺省扫描影像库目录


@router.post("/audit")
async def run_privacy_audit(
    reg: Annotated[Registry, Depends(get_registry)],
    principal: Annotated[Principal, Depends(require_role("sysadmin", "secadmin", "auditor"))],
    body: PrivacyAuditIn | None = None,
) -> dict[str, Any]:
    """运行脱敏残留扫描（缺省目录=影像库），生成 JSON+PDF 报告并留痕。

    目录参数锚定 data/ 根：待扫底片夹应置于数据目录（入库前暂存亦然）。
    无锚定时空接受任意绝对路径会把端点变成全盘目录枚举/文件清点探测面。
    """
    data_root = resolve_config_path(reg.config.paths.data_dir).resolve()
    if body is not None and body.directory:
        candidate = Path(body.directory)
        candidate = candidate if candidate.is_absolute() else resolve_config_path(str(candidate))
        candidate = candidate.resolve()
        if not candidate.is_relative_to(data_root):
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "PATH_OUT_OF_SCOPE",
                    "message": f"扫描目录必须位于数据目录内: {data_root}",
                },
            )
        directory = str(candidate)
    else:
        directory = str(resolve_config_path(reg.config.paths.images_dir))
    report = await run_in_threadpool(audit_directory_phi, directory)
    paths = write_audit_report(report, resolve_config_path(reg.config.paths.data_dir) / "privacy")
    report["report_files"] = paths
    reg.repository.append_audit(
        actor=principal.username,
        action="privacy_audit",
        object_type="privacy",
        object_id=paths["json"],
        before=None,
        after={"scanned": report["scanned"], "n_findings": report["n_findings"]},
        note="C-13 脱敏残留审计",
    )
    return report
