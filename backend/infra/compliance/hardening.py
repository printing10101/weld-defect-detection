"""安全加固自检（C-25）。

五类检查（结果分级 high/medium/low，检查项含修复建议）：
1. 默认口令：本系统账号体系无静态口令（SM2 挑战-响应证书/UKey 模式），
   "默认口令"风险转化为两件事的真实检查——引导窗口是否关闭、是否存在
   未绑定公钥的账号；
2. 开放端口：配置监听地址必须 127.0.0.1；psutil 可用时核验本进程实际
   监听列表（未安装则如实降级为仅核验配置，不造假）；
3. 未授权接口：内省 FastAPI app.routes，枚举未挂 get_principal/require_role
   的 API 路由（health/metrics/auth 豁免，其余应为空集）；
4. TLS/传输：IPC 令牌强制与否 + 本机回环明文 HTTP 的诚实声明；
5. 文件权限：数据目录 / DB / IPC 令牌文件的 POSIX 权限位（Windows 平台
   ACL 无法经 POSIX 位核验，如实注明）。

产物：JSON + PDF 落 data/compliance/，动作由路由层入审计。
"""

from __future__ import annotations

import json
import os
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.infra.compliance.doc_pdf import build_doc_pdf
from backend.infra.config import resolve_config_path


def _now_str() -> str:
    return datetime.now(UTC).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")


def _finding(
    name: str,
    severity: str,
    result: str,
    evidence: str,
    recommendation: str | None = None,
) -> dict[str, str]:
    """单条检查结果（severity ∈ high/medium/low；result ∈ pass/fail/warning）。"""
    return {
        "name": name,
        "severity": severity,
        "result": result,
        "evidence": evidence,
        "recommendation": recommendation or ("保持现状" if result == "pass" else "见证据栏"),
    }


# ---------------------------------------------------------------------------
# 1. 默认口令
# ---------------------------------------------------------------------------


def _check_default_credential(reg) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    accounts = reg.security_store.list_accounts()
    n_accounts = len(accounts)
    if n_accounts == 0:
        items.append(
            _finding(
                "默认口令/引导窗口",
                "high",
                "fail",
                "accounts 表为空——POST /auth/bootstrap 引导窗口仍开放（任何本机调用方可创建首个账号）",
                "立即完成首次部署引导（创建三员账号），关闭引导窗口",
            )
        )
    else:
        items.append(
            _finding(
                "默认口令/引导窗口",
                "high",
                "pass",
                f"accounts 表 {n_accounts} 个账号，引导窗口已关闭（bootstrap 返回 409）；"
                "登录为 SM2 挑战-响应（无静态口令可猜/可默认）",
            )
        )
    no_key = [a["username"] for a in accounts if not a.get("sm2_public_key")]
    if no_key:
        items.append(
            _finding(
                "未绑定公钥账号",
                "high",
                "fail",
                f"{len(no_key)} 个账号未登记 SM2 公钥（无法登录，属悬空账号）: {no_key}",
                "为悬空账号签发软证书或停用删除",
            )
        )
    else:
        items.append(
            _finding(
                "未绑定公钥账号",
                "high",
                "pass",
                "全部账号均已绑定 SM2 公钥（挑战-响应凭据，无私钥/口令留存于系统）",
            )
        )
    return items


# ---------------------------------------------------------------------------
# 2. 开放端口
# ---------------------------------------------------------------------------


def _listening_of_current_process() -> list[dict[str, int | str]] | None:
    """本进程监听中的 INET 套接字（psutil 可用才返回，否则 None 如实降级）。"""
    try:
        import psutil
    except ImportError:
        return None
    pid = os.getpid()
    seen: set[tuple[str, int]] = set()
    out: list[dict[str, int | str]] = []
    try:
        for c in psutil.net_connections(kind="inet"):
            if c.status != psutil.CONN_LISTEN or c.pid != pid or not c.laddr:
                continue
            key = (str(c.laddr.ip), int(c.laddr.port))
            if key in seen:
                continue
            seen.add(key)
            out.append({"addr": key[0], "port": key[1]})
    except Exception:  # noqa: BLE001 - 平台限制（如 macOS 权限）→ 如实降级
        return None
    return out


def _is_loopback(addr: str) -> bool:
    return addr in ("127.0.0.1", "::1") or addr.startswith("127.")


def _check_ports(reg) -> list[dict[str, str]]:
    cfg = reg.config
    items: list[dict[str, str]] = []
    hosts = {"server": cfg.server.host, "annotator": cfg.annotator.host}
    non_local = {k: v for k, v in hosts.items() if not _is_loopback(v)}
    if non_local:
        items.append(
            _finding(
                "监听地址仅本机",
                "high",
                "fail",
                f"配置的监听地址含非回环: {non_local}",
                "server.host / annotator.host 必须为 127.0.0.1（涉密单机不得对外监听）",
            )
        )
    else:
        items.append(
            _finding(
                "监听地址仅本机",
                "high",
                "pass",
                f"配置监听地址: {hosts}（均为回环）",
            )
        )
    listeners = _listening_of_current_process()
    if listeners is None:
        items.append(
            _finding(
                "本进程实际监听核验",
                "low",
                "warning",
                "psutil 未安装，无法核验本进程实际监听套接字；仅完成配置层检查（不造假降级）",
                "安装 psutil 后重跑自查可核验真实监听列表",
            )
        )
    else:
        bad = [l for l in listeners if not _is_loopback(str(l["addr"]))]
        if bad:
            items.append(
                _finding(
                    "本进程实际监听核验",
                    "high",
                    "fail",
                    f"检测到本进程监听非回环地址: {bad}",
                    "核查监听代码路径，立即修正为 127.0.0.1 绑定",
                )
            )
        else:
            items.append(
                _finding(
                    "本进程实际监听核验",
                    "high",
                    "pass",
                    f"本进程监听套接字 {listeners}（全部回环）",
                )
            )
    return items


# ---------------------------------------------------------------------------
# 3. 未授权接口扫描（内省 app.routes）
# ---------------------------------------------------------------------------

_API_EXEMPT_PREFIXES = ("/api/v1/health", "/api/v1/metrics", "/api/v1/auth")


def _iter_effective_api_routes(app):
    """遍历 app 的全部生效 API 路由（跨 FastAPI 版本兼容）。

    新版 FastAPI（≥0.115 惰性 include）在 app.routes 中存放 _IncludedRouter，
    需递归 effective_candidates 还原 _EffectiveRouteContext（含完整前缀路径）；
    旧版 app.routes 直接是 APIRoute。统一产出 (完整路径, 路由对象)。
    """
    from fastapi.routing import APIRoute

    for item in app.routes:
        if isinstance(item, APIRoute):
            yield item.path, item
            continue
        cand = getattr(item, "effective_candidates", None)
        if cand is None:
            continue
        for ctx in cand():
            route = getattr(ctx, "original_route", None)
            if isinstance(route, APIRoute):
                yield str(getattr(ctx, "path", route.path)), ctx


def _dep_callables(deps) -> list:
    """从依赖对象列表提取可调用体（兼容 Depends.dependency 与内部 Dependency.call）。"""
    out = []
    for d in deps or []:
        fn = getattr(d, "dependency", None) or getattr(d, "call", None)
        if fn is not None:
            out.append(fn)
    return out


def unauthenticated_api_routes(app) -> list[str]:
    """枚举无鉴权依赖的 API 路由（C-25 未授权接口扫描）。

    判定：路由级依赖（include_router dependencies）或端点签名依赖中出现
    backend.app.auth 模块的鉴权依赖（get_principal / require_role 闭包）即视为
    已管控；health/metrics/auth（登录入口本身）豁免。返回 "METHOD path" 列表，
    **应为空集**。
    """
    from backend.app.auth import get_principal

    out: list[str] = []
    for path, route in _iter_effective_api_routes(app):
        if not path.startswith("/api/v1"):
            continue
        if any(path.startswith(p) for p in _API_EXEMPT_PREFIXES):
            continue
        deps = _dep_callables(getattr(route, "dependencies", None))
        deps += (
            [d.call for d in getattr(route, "dependant", None).dependencies]
            if getattr(route, "dependant", None)
            else []
        )
        authed = any(
            getattr(d, "__module__", "") == "backend.app.auth"
            and (d is get_principal or getattr(d, "__name__", "") == "_dep")
            for d in deps
        )
        if not authed:
            methods = getattr(route, "methods", None) or {"GET"}
            out.append(
                f"{','.join(sorted(m for m in methods if m not in ('HEAD', 'OPTIONS')))} {path}"
            )
    return out


def _check_endpoints(app) -> list[dict[str, str]]:
    unauthed = unauthenticated_api_routes(app)
    if unauthed:
        return [
            _finding(
                "未授权接口扫描",
                "high",
                "fail",
                f"发现 {len(unauthed)} 个未挂鉴权依赖的 API 路由: {unauthed}",
                "为列出的路由补挂 get_principal/require_role 依赖",
            )
        ]
    return [
        _finding(
            "未授权接口扫描",
            "high",
            "pass",
            "app.routes 内省：除 health/metrics/auth 豁免外，全部 API 路由均已挂 "
            "get_principal 或 require_role 鉴权依赖（空集）",
        )
    ]


# ---------------------------------------------------------------------------
# 4. TLS / 传输
# ---------------------------------------------------------------------------


def _check_transport(reg) -> list[dict[str, str]]:
    cfg = reg.config
    items = []
    if cfg.ipc.enforce:
        items.append(
            _finding(
                "IPC 一次性令牌",
                "medium",
                "pass",
                "ipc.enforce=true——非认证调用方（其他本机进程/网页）被令牌中间件拦截",
            )
        )
    else:
        items.append(
            _finding(
                "IPC 一次性令牌",
                "medium",
                "fail",
                "ipc.enforce=false——本机任意进程可直接调用业务 API（仅限单机调试）",
                "生产环境置 ipc.enforce=true",
            )
        )
    # 诚实声明：本机回环为明文 HTTP，无 TLS；靠单机部署边界 + IPC 令牌缓解。
    items.append(
        _finding(
            "传输加密（TLS）",
            "medium",
            "warning",
            "本机前后端通信为明文 HTTP（127.0.0.1 回环），未启用 TLS；"
            "缓解措施：仅回环监听 + IPC 令牌 + 会话鉴权。此为本版本如实声明的设计边界",
            "如需传输加密，挂本机证书经 TLS 反代接入（部署基线文档注明）",
        )
    )
    return items


# ---------------------------------------------------------------------------
# 5. 文件权限
# ---------------------------------------------------------------------------


def _check_file_permissions(reg) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    data_dir = resolve_config_path(reg.config.paths.data_dir)
    db_path = resolve_config_path(reg.config.paths.db_path)
    token_path = data_dir / "ipc_token"
    targets = [
        ("数据目录", data_dir, True),
        ("数据库文件", db_path, False),
        ("IPC 令牌文件", token_path, False),
    ]
    for label, path, is_dir in targets:
        if not path.exists():
            items.append(
                _finding(
                    f"文件权限: {label}",
                    "low",
                    "warning",
                    f"{path} 不存在（服务未初始化或路径变更）",
                    "确认服务已启动并完成运行时目录创建",
                )
            )
            continue
        mode = stat.S_IMODE(path.stat().st_mode)
        if os.name == "nt":
            # Windows 平台 POSIX 位不具 ACL 语义（st_mode 不反映 icacls 授权），
            # 写位恒置属平台映射而非真实放权——如实降级为 warning，不误报 fail。
            items.append(
                _finding(
                    f"文件权限: {label}",
                    "low",
                    "warning",
                    f"{path} 权限位 {stat.filemode(mode)}（Windows 平台 POSIX 位仅参考，"
                    "ACL 需以 icacls 核验——本检查不据此下 fail 结论）",
                    "以 icacls 核验并收紧到仅服务账号可访问",
                )
            )
            continue
        group_other_write = bool(mode & (stat.S_IWGRP | stat.S_IWOTH))
        sev, result = ("medium", "fail") if group_other_write else ("low", "pass")
        rec = None if not group_other_write else "收紧权限：移除组/其他用户的写权限（icacls/chmod）"
        if (
            label == "IPC 令牌文件"
            and not group_other_write
            and (mode & (stat.S_IRGRP | stat.S_IROTH))
        ):
            sev, result = "medium", "warning"
            rec = "令牌文件可被本机其他用户读取，收紧为仅当前用户可读"
        items.append(
            _finding(
                f"文件权限: {label}",
                sev,
                result,
                f"{path} 权限位 {stat.filemode(mode)}",
                rec,
            )
        )
    return items


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------


def run_hardening_check(reg, app) -> dict[str, Any]:
    """执行安全加固自检，返回报告 dict。"""
    findings = (
        _check_default_credential(reg)
        + _check_ports(reg)
        + _check_endpoints(app)
        + _check_transport(reg)
        + _check_file_permissions(reg)
    )
    high_failed = [f for f in findings if f["severity"] == "high" and f["result"] == "fail"]
    counts: dict[str, int] = {}
    for f in findings:
        key = f"{f['severity']}_{f['result']}"
        counts[key] = counts.get(key, 0) + 1
    overall = (
        "fail"
        if high_failed
        else ("warning" if any(f["result"] != "pass" for f in findings) else "pass")
    )
    return {
        "generated_at": _now_str(),
        "overall": overall,
        "summary": {"total": len(findings), **counts},
        "findings": findings,
        "high_findings": [
            {"name": f["name"], "evidence": f["evidence"], "recommendation": f["recommendation"]}
            for f in high_failed
        ],
    }


def hardening_pdf(report: dict[str, Any], out_path: str | Path) -> Path:
    """加固自检报告 → PDF（PDF/A-1b）。"""
    meta = [
        ("生成时间", report.get("generated_at", "—")),
        ("总体结论", str(report.get("overall", "—")).upper()),
        (
            "高危项",
            f"{len(report.get('high_findings', []))} 项",
        ),
    ]
    sections = [
        {
            "heading": "检查明细",
            "table": {
                "head": ["检查项", "级别", "结论", "证据", "修复建议"],
                "rows": [
                    [f["name"], f["severity"], f["result"], f["evidence"], f["recommendation"]]
                    for f in report.get("findings", [])
                ],
            },
        },
        {
            "heading": "高危项与修复建议",
            "paragraphs": (
                [
                    f"· {h['name']}：{h['evidence']} → 建议：{h['recommendation']}"
                    for h in report.get("high_findings", [])
                ]
                or ["无高危失败项。"]
            ),
        },
    ]
    return build_doc_pdf("安全加固自检报告（C-25）", meta, sections, Path(out_path))


def write_hardening_report(report: dict[str, Any], out_dir: str | Path) -> dict[str, str]:
    """报告落盘（JSON + PDF），返回 {json, pdf} 路径。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).replace(tzinfo=None).strftime("%Y%m%d_%H%M%S")
    json_path = out / f"hardening_{ts}.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pdf_path = hardening_pdf(report, out / f"hardening_{ts}.pdf")
    return {"json": str(json_path), "pdf": str(pdf_path)}
