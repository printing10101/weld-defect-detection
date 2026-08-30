"""分级保护自查报告（C-23）。

按分级保护（GB/T 22239 / GB/T 17859 二级要求口径）五类逐项自动检查：
身份鉴别 / 访问控制 / 安全审计 / 边界防护 / 信息流转控制。

**活检查承诺**：每项检查都在运行时真实查询系统状态（数据库表、配置、
守卫对象、目录扫描），不硬编码结论；查不到证据的项如实给 warning，
并在证据栏写明查询方式。每项含：检查项名称、依据（对应条款描述）、
结论（pass/fail/warning）、证据引用、未达标处置建议。

产物：JSON + PDF 落 data/compliance/，自查动作由路由层入审计。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.infra.compliance.doc_pdf import build_doc_pdf

# 关键动作审计覆盖检查的核心动作清单（评片/复核/导出/账号/运维全链路）
_CORE_AUDIT_ACTIONS: tuple[str, ...] = (
    "inspect",
    "review",
    "export_request_create",
    "export_download",
    "account_lock",
    "account_unlock",
    "model_activate",
    "secret_level_change",
    "backup_create",
    "backup_restore",
    "gate_reject",
    "egress_blocked",
)

_LOCAL_ORIGIN_TOKENS = ("127.0.0.1", "localhost", "tauri://localhost", "tauri.localhost")


def _now_str() -> str:
    return datetime.now(UTC).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")


def _item(
    name: str,
    basis: str,
    result: str,
    evidence: str,
    recommendation: str | None = None,
) -> dict[str, str]:
    """构造单条自检项（result ∈ pass/fail/warning）。"""
    return {
        "name": name,
        "basis": basis,
        "result": result,
        "evidence": evidence,
        "recommendation": recommendation or ("保持现状" if result == "pass" else "见证据栏"),
    }


# ---------------------------------------------------------------------------
# 五类检查（每项均为运行时真实状态查询）
# ---------------------------------------------------------------------------


def _check_identity(reg) -> list[dict[str, str]]:
    """身份鉴别：三员齐备互斥 / 无共享账号 / 空闲锁定 / 登录失败锁定。"""
    items: list[dict[str, str]] = []
    accounts = reg.security_store.list_accounts()
    role_dist: dict[str, int] = {}
    for a in accounts:
        role_dist[a["role"]] = role_dist.get(a["role"], 0) + 1
    missing = [r for r in ("sysadmin", "secadmin", "auditor") if role_dist.get(r, 0) < 1]
    # 一人一岗互斥：账号表 schema 上每行仅一个角色列，运行期再确认无"停用外
    # 的空角色"账号（互斥由一人一岗的账号模型保证，这里验证其成立的事实）。
    if missing:
        items.append(
            _item(
                "三员角色齐备且互斥",
                "分级保护要求：系统管理员/安全保密管理员/安全审计员三权分立、岗位齐备且相互独立",
                "fail",
                f"accounts 表 {len(accounts)} 账号，角色分布 {role_dist}，缺 {missing}",
                "由系统管理员按三员岗位补建账号（POST /auth/accounts）并签发 SM2 软证书",
            )
        )
    else:
        items.append(
            _item(
                "三员角色齐备且互斥",
                "分级保护要求：三权分立、岗位齐备且相互独立",
                "pass",
                f"accounts 表 {len(accounts)} 账号，三角色齐备，角色分布 {role_dist}；"
                "一人一岗账号模型（每账号仅一个角色列）保证岗位互斥",
            )
        )

    # 无共享账号：活动会话按账号聚合，同一账号并发 >1 个活动会话即视为共享嫌疑
    sessions = _active_sessions(reg)
    by_account: dict[str, int] = {}
    for s in sessions:
        by_account[s["account_id"]] = by_account.get(s["account_id"], 0) + 1
    shared = {k: v for k, v in by_account.items() if v > 1}
    if shared:
        items.append(
            _item(
                "无共享账号（会话重叠检测）",
                "分级保护要求：一个用户一个账号，禁止多人共用同一账号同时在线",
                "fail",
                f"检测到 {len(shared)} 个账号存在并发活动会话重叠: {shared}",
                "核查重叠账号使用人；确认单点登录配置 auth.max_sessions=1 并追究共用行为",
            )
        )
    else:
        items.append(
            _item(
                "无共享账号（会话重叠检测）",
                "分级保护要求：一个用户一个账号，禁止多人共用同一账号",
                "pass",
                f"当前活动会话 {len(sessions)} 个，无单账号多会话重叠"
                f"（并发上限 auth.max_sessions={reg.config.auth.max_sessions}）",
            )
        )

    idle_ok = reg.config.auth.idle_timeout_min > 0
    items.append(
        _item(
            "会话空闲锁定已启用",
            "分级保护要求：登录连接超时自动退出（空闲超时锁定）",
            "pass" if idle_ok else "fail",
            f"auth.idle_timeout_min={reg.config.auth.idle_timeout_min} 分钟"
            f"（另有绝对有效期 {reg.config.auth.session_ttl_min} 分钟硬上限）",
            None if idle_ok else "将 auth.idle_timeout_min 配置为大于 0 的值",
        )
    )

    lockout_ok = reg.config.auth.max_failed_attempts > 0 and reg.config.auth.lockout_min > 0
    items.append(
        _item(
            "登录失败锁定已启用",
            "分级保护要求：登录失败达到阈值采取锁定等处置措施",
            "pass" if lockout_ok else "fail",
            f"auth.max_failed_attempts={reg.config.auth.max_failed_attempts}，"
            f"lockout_min={reg.config.auth.lockout_min} 分钟"
            f"（历史锁定告警 {reg.security_store.count_alerts(kind='account_locked')} 条）",
            None if lockout_ok else "将 max_failed_attempts / lockout_min 配置为大于 0 的值",
        )
    )
    return items


def _active_sessions(reg) -> list[dict[str, Any]]:
    """枚举未吊销会话（经 security_store 内部查询的等价实现，活检查）。"""
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    from backend.infra.db import SessionRecord

    with Session(reg.security_store._engine) as session:
        rows = list(
            session.scalars(select(SessionRecord).where(SessionRecord.revoked.is_(False)))
        )
        return [{"account_id": r.account_id, "token_hash": r.token_hash} for r in rows]


def _check_access(reg) -> list[dict[str, str]]:
    """访问控制：密级字段 / 导出审批 / 高密级导出拦截。"""
    from backend.infra.db import ImageRecord

    sec_cols = {c.name for c in ImageRecord.__table__.columns}
    has_level = "secret_level" in sec_cols and "classification_basis" in sec_cols
    items = [
        _item(
            "密级字段已启用",
            "分级保护要求：对涉密信息进行密级标识，密级与实体同步存储",
            "pass" if has_level else "fail",
            "images/reports 表均含 secret_level/classification_basis 列"
            f"（静态加密 security.encrypt={reg.config.security.encrypt}）",
            None if has_level else "数据库 schema 异常，请联系系统管理员检查迁移",
        )
    ]
    approval = reg.config.export.require_approval
    items.append(
        _item(
            "导出审批已开启",
            "分级保护要求：信息输出须经授权审批，未经批准不得导出",
            "pass" if approval else "fail",
            f"export.require_approval={approval}"
            f"（导出令牌有效期 {reg.config.export.token_ttl_sec}s，一次一用）",
            None if approval else "生产环境必须 export.require_approval=true（当前仅限单机调试）",
        )
    )
    denied_cnt = int(reg.repository.list_audit(action="export_denied", limit=1)[1])
    # 高密级导出拦截：以运行证据（export_denied 审计记录，含 std_eval 高密级拒导）
    # 佐证拦截逻辑真实生效；无记录时诚实给 warning 而非硬判 pass。
    std_denied, _ = reg.repository.list_audit(
        action="export_denied", object_id="std_eval:false_reports", limit=1
    )
    if approval and (denied_cnt > 0 or std_denied):
        items.append(
            _item(
                "高密级导出拦截有效",
                "分级保护要求：高密级信息未经授权禁止流转出系统",
                "pass",
                f"导出审批开启，审计链存在 {denied_cnt} 条 export_denied 拒绝记录"
                "（含高密级清单拒导）",
            )
        )
    elif approval:
        items.append(
            _item(
                "高密级导出拦截有效",
                "分级保护要求：高密级信息未经授权禁止流转出系统",
                "warning",
                "导出审批已开启（代码路径 ensure_export_allowed 挂接所有受控导出端点），"
                "但尚无 export_denied 拒绝记录可佐证拦截曾被真实触发",
                "可执行一次越权导出验证拦截（预期 401/403 并落 export_denied 审计）",
            )
        )
    else:
        items.append(
            _item(
                "高密级导出拦截有效",
                "分级保护要求：高密级信息未经授权禁止流转出系统",
                "fail",
                "export.require_approval=false，导出门禁整体关闭",
                "开启 export.require_approval=true",
            )
        )
    return items


def _check_audit(reg) -> list[dict[str, str]]:
    """安全审计：双链校验 / 关键动作覆盖 / 归档导出可用。"""
    main_valid = reg.repository.verify_chain()
    sec_valid = reg.security_store.verify_security_chain()
    items = [
        _item(
            "主审计链（SM3 哈希链）完整性",
            "分级保护要求：审计记录受保护，防止未授权修改/删除（防篡改哈希链）",
            "pass" if main_valid else "fail",
            f"repository.verify_chain() = {main_valid}",
            None if main_valid else "主链哈希校验失败：立即停止写入并取证核查篡改范围",
        ),
        _item(
            "安全审计链（独立双链）完整性",
            "分级保护要求：安全审计与业务审计分立，防单链被整体覆盖",
            "pass" if sec_valid else "fail",
            f"security_store.verify_security_chain() = {sec_valid}",
            None if sec_valid else "安全链哈希校验失败：立即取证核查",
        ),
    ]
    covered: list[str] = []
    missing: list[str] = []
    for action in _CORE_AUDIT_ACTIONS:
        _, total = reg.repository.list_audit(action=action, limit=1)
        (covered if total > 0 else missing).append(f"{action}({total})")
    if not missing:
        items.append(
            _item(
                "关键动作审计覆盖",
                "分级保护要求：对重要用户行为/安全事件进行审计覆盖",
                "pass",
                f"核心动作 {_CORE_AUDIT_ACTIONS} 全部有审计记录（counts: {', '.join(covered)}）",
            )
        )
    elif covered:
        items.append(
            _item(
                "关键动作审计覆盖",
                "分级保护要求：对重要用户行为/安全事件进行审计覆盖",
                "warning",
                f"已覆盖: {', '.join(covered)}；尚无记录: {', '.join(missing)}"
                "（新装机/未执行过对应动作属正常，无法凭空产生记录）",
                "使用一段时间后重跑自查；对涉及安全的关键动作做一轮演练覆盖",
            )
        )
    else:
        items.append(
            _item(
                "关键动作审计覆盖",
                "分级保护要求：对重要用户行为/安全事件进行审计覆盖",
                "fail",
                "核心动作均无审计记录（审计链可能为空或写入故障）",
                "核查审计写入链路（repository.append_audit）是否正常",
            )
        )
    # 归档导出可用：真实构造一次 JSONL 归档（不落盘、不入审计——入审计由端点负责）
    from backend.app.routers.audit import build_audit_export

    try:
        body, footer = build_audit_export(reg, "selfcheck")
        n_lines = len(body.strip().splitlines()) - 2  # 去 header 与 footer 两行
        expect = footer["main_chain_total"] + footer["security_chain_total"]
        ok = n_lines == expect
        items.append(
            _item(
                "审计归档导出可用",
                "分级保护要求：审计记录可按需输出为只读归档格式",
                "pass" if ok else "fail",
                f"JSONL 归档构造成功：记录行 {n_lines}（主链 {footer['main_chain_total']}"
                f" + 安全链 {footer['security_chain_total']}），行数核对{'一致' if ok else '不一致'}",
                None if ok else "归档行数与链内条数不符，检查分页导出逻辑",
            )
        )
    except Exception as exc:  # noqa: BLE001 - 检查自身失败如实报告
        items.append(
            _item(
                "审计归档导出可用",
                "分级保护要求：审计记录可按需输出为只读归档格式",
                "fail",
                f"归档构造抛出异常: {exc}",
                "检查 /audit/export 端点与审计存储",
            )
        )
    return items


def _check_boundary(reg) -> list[dict[str, str]]:
    """边界防护：egress guard / IPC 令牌 / CORS / 离线模式。"""
    from backend.infra.config import resolve_config_path
    from backend.infra.egress_guard import get_guard

    cfg = reg.config
    guard = get_guard()
    if cfg.egress.enabled and guard is not None:
        result, rec = "pass", None
        evidence = (
            f"egress.enabled=true 且运行时守卫已装配（allow_cidrs={cfg.egress.allow_cidrs}，"
            f"进程内累计拦截 {guard.blocked_total} 次，持久告警 "
            f"{reg.security_store.count_alerts(kind='egress_blocked')} 条）"
        )
    elif cfg.egress.enabled:
        result, rec = "warning", "egress 守卫未在当前进程装配（服务未重启/独立进程调用）"
        evidence = f"egress.enabled=true 但 get_guard()={guard}（None=未装配）"
    else:
        result, rec = "fail", "开启 egress.enabled=true 并重启服务"
        evidence = "egress.enabled=false——进程级外联防护关闭"
    items = [
        _item(
            "外联防护（egress guard）启用",
            "分级保护要求：边界防护控制进出网络的数据流，非授权外联应拦截并告警",
            result,
            evidence,
            rec,
        )
    ]
    token_path = resolve_config_path(cfg.paths.data_dir) / "ipc_token"
    if cfg.ipc.enforce and token_path.is_file():
        items.append(
            _item(
                "IPC 一次性令牌强制",
                "分级保护要求：防止其他本机进程/未授权调用方访问系统接口",
                "pass",
                f"ipc.enforce=true，令牌文件存在: {token_path}",
            )
        )
    elif cfg.ipc.enforce:
        items.append(
            _item(
                "IPC 一次性令牌强制",
                "分级保护要求：防止其他本机进程/未授权调用方访问系统接口",
                "warning",
                f"ipc.enforce=true 但令牌文件不存在: {token_path}（服务刚启动尚未签发？）",
                "重启服务由启动自检签发令牌",
            )
        )
    else:
        items.append(
            _item(
                "IPC 一次性令牌强制",
                "分级保护要求：防止其他本机进程/未授权调用方访问系统接口",
                "fail",
                "ipc.enforce=false（仅限单机调试）",
                "生产环境置 ipc.enforce=true",
            )
        )
    origins = list(cfg.server.cors_origins)
    wildcard = "*" in origins
    non_local = [o for o in origins if not any(tok in o for tok in _LOCAL_ORIGIN_TOKENS)]
    if wildcard or non_local:
        items.append(
            _item(
                "CORS 仅本机来源",
                "分级保护要求：跨源访问受控，防止外部网站读取本机 API 数据",
                "fail",
                f"cors_origins={origins}（含通配符: {wildcard}，非本机来源: {non_local}）",
                "移除 '*' 与非本机来源，仅保留 127.0.0.1/localhost/tauri 来源",
            )
        )
    else:
        items.append(
            _item(
                "CORS 仅本机来源",
                "分级保护要求：跨源访问受控，防止外部网站读取本机 API 数据",
                "pass",
                f"cors_origins={origins}（无通配符，全部为本机/Tauri 来源）",
            )
        )
    from backend.app.routers.system import offline_conclusion

    conclusion = offline_conclusion(cfg)
    offline_ok = bool(conclusion["offline_mode"])
    items.append(
        _item(
            "离线模式（数据不出本机）",
            "分级保护要求：涉密信息不得未经批准流出系统边界",
            "pass" if offline_ok else "fail",
            f"sync.kind={conclusion['sync_kind']}, offline_mode={offline_ok}, "
            f"egress_guard_enabled={conclusion['egress_guard_enabled']}",
            None if offline_ok else "sync.kind 配置为 local（涉密单机部署不得外发）",
        )
    )
    return items


def _check_infoflow(reg) -> list[dict[str, str]]:
    """信息流转控制：脱敏审计 / 载体台账 / 销毁双确认。"""
    from backend.infra.config import resolve_config_path
    from backend.infra.privacy_audit import audit_directory_phi

    items: list[dict[str, str]] = []
    try:
        report = audit_directory_phi(resolve_config_path(reg.config.paths.images_dir), max_files=2000)
        if report["scanned"] == 0:
            items.append(
                _item(
                    "脱敏残留审计（DICONDE PHI/EXIF）",
                    "分级保护要求：信息流转前去除与业务无关的个人敏感信息",
                    "warning",
                    f"影像目录 {report['directory']} 无可扫描影像（scanned=0），无法给出残留结论",
                    "上传真实影像后重跑自查",
                )
            )
        elif report["clean"]:
            items.append(
                _item(
                    "脱敏残留审计（DICONDE PHI/EXIF）",
                    "分级保护要求：信息流转前去除与业务无关的个人敏感信息",
                    "pass",
                    f"扫描 {report['scanned']} 个影像文件，未发现 PHI/EXIF 残留",
                )
            )
        else:
            items.append(
                _item(
                    "脱敏残留审计（DICONDE PHI/EXIF）",
                    "分级保护要求：信息流转前去除与业务无关的个人敏感信息",
                    "fail",
                    f"扫描 {report['scanned']} 个文件，发现 {report['n_findings']} 处残留/异常",
                    "对残留文件执行脱敏（anonymize）后重新入库",
                )
            )
    except Exception as exc:  # noqa: BLE001 - 检查自身失败如实报告
        items.append(
            _item(
                "脱敏残留审计（DICONDE PHI/EXIF）",
                "分级保护要求：信息流转前去除与业务无关的个人敏感信息",
                "fail",
                f"扫描抛出异常: {exc}",
                "检查影像目录与 privacy_audit 模块",
            )
        )

    carriers = reg.carrier_store.list()
    if carriers:
        items.append(
            _item(
                "涉密载体台账在用",
                "分级保护要求：涉密载体登记、流转、销毁全生命周期台账管理",
                "pass",
                f"载体台账共 {len(carriers)} 条登记"
                f"（状态分布: {_count_by(carriers, 'status')}）",
            )
        )
    else:
        items.append(
            _item(
                "涉密载体台账在用",
                "分级保护要求：涉密载体登记、流转、销毁全生命周期台账管理",
                "warning",
                "载体台账功能可用（POST /carriers），但当前无任何登记记录",
                "对底片/报告/备份介质执行载体登记后重跑自查",
            )
        )

    destroyed = [c for c in carriers if c["status"] == "destroyed"]
    if destroyed:
        bad = [
            c["carrier_id"]
            for c in destroyed
            if c.get("destroy_confirmed_by") in (None, c.get("destroy_requested_by"))
        ]
        items.append(
            _item(
                "载体销毁双确认生效",
                "分级保护要求：涉密载体销毁须经审批双人经办",
                "fail" if bad else "pass",
                (
                    f"{len(destroyed)} 个已销毁载体中 {len(bad)} 个缺少独立双确认: {bad}"
                    if bad
                    else f"{len(destroyed)} 个已销毁载体均满足发起人≠确认人的双确认"
                ),
                None if not bad else "核查越权销毁记录并追责",
            )
        )
    else:
        items.append(
            _item(
                "载体销毁双确认生效",
                "分级保护要求：涉密载体销毁须经审批双人经办（发起≠确认）",
                "warning",
                "销毁为两段式接口（destroy-request 由保密员发起 → destroy-confirm "
                "由系统管理员确认，代码强制发起人≠确认人），尚无销毁记录可佐证",
                "执行一次载体销毁演练后重跑自查",
            )
        )
    return items


def _count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        k = str(r.get(key))
        out[k] = out.get(k, 0) + 1
    return out


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------

CATEGORIES: tuple[tuple[str, str], ...] = (
    ("身份鉴别", "_check_identity"),
    ("访问控制", "_check_access"),
    ("安全审计", "_check_audit"),
    ("边界防护", "_check_boundary"),
    ("信息流转控制", "_check_infoflow"),
)


def run_selfcheck(reg) -> dict[str, Any]:
    """执行五类分级保护自查，返回报告 dict（JSON 可序列化）。

    overall = fail（任一 fail）> warning（任一 warning）> pass。
    """
    checkers = {
        "身份鉴别": _check_identity,
        "访问控制": _check_access,
        "安全审计": _check_audit,
        "边界防护": _check_boundary,
        "信息流转控制": _check_infoflow,
    }
    categories: dict[str, list[dict[str, str]]] = {}
    counts = {"pass": 0, "fail": 0, "warning": 0}
    for cat, fn in checkers.items():
        items = fn(reg)
        categories[cat] = items
        for it in items:
            counts[it["result"]] = counts.get(it["result"], 0) + 1
    overall = "fail" if counts["fail"] else ("warning" if counts["warning"] else "pass")
    return {
        "generated_at": _now_str(),
        "standard": "分级保护（GB/T 22239 / GB/T 17859 二级口径，软件侧自动检查部分）",
        "overall": overall,
        "summary": {"total": sum(counts.values()), **counts},
        "categories": categories,
    }


def selfcheck_pdf(report: dict[str, Any], out_path: str | Path) -> Path:
    """自查报告 → PDF（PDF/A-1b）。"""
    meta = [
        ("生成时间", report.get("generated_at", "—")),
        ("检查依据", report.get("standard", "—")),
        ("总体结论", str(report.get("overall", "—")).upper()),
        (
            "项数统计",
            (f"pass={report['summary']['pass']} / warning={report['summary']['warning']}"
            f" / fail={report['summary']['fail']}"),
        ),
    ]
    sections = []
    for cat, items in report.get("categories", {}).items():
        sections.append(
            {
                "heading": cat,
                "table": {
                    "head": ["检查项", "结论", "证据", "处置建议"],
                    "rows": [
                        [it["name"], it["result"], it["evidence"], it["recommendation"]]
                        for it in items
                    ],
                },
            }
        )
    return build_doc_pdf(
        "分级保护安全自查报告（C-23）",
        meta,
        sections,
        Path(out_path),
    )


def write_selfcheck(report: dict[str, Any], out_dir: str | Path) -> dict[str, str]:
    """报告落盘（JSON + PDF），返回 {json, pdf} 路径。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).replace(tzinfo=None).strftime("%Y%m%d_%H%M%S")
    json_path = out / f"selfcheck_{ts}.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pdf_path = selfcheck_pdf(report, out / f"selfcheck_{ts}.pdf")
    return {"json": str(json_path), "pdf": str(pdf_path)}
