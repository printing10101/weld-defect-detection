"""审计归档 JSONL 构造（C-20）——纯 infra 逻辑，供审计路由与合规自检共用。

行格式（每行一个 JSON 对象，只追加语义、按链序排列）：
- 首行 header：导出元数据（时间/操作者/两条链的整链校验结论/总条数）；
- 记录行：{"type":"record","chain":"main"|"security","seq":n,
  "record":{...},"chain_valid":<该链整链校验结论>}；
- 末行 footer：两条链的条数与校验结论汇总（供归档校验程序核对）。

分层说明：原实现放在 app.routers.audit，导致 infra.compliance.selfcheck
反向 import app 层（import-linter 分层合约红）。构造归档件只依赖两个
infra 仓储对象，故下移至本模块；路由层仅做鉴权/审计留痕编排。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from typing import Any

# list_audit/list_security_audit 单页上限 500；全量导出按 offset 翻页取整链。
_EXPORT_PAGE = 500


def _iter_full_chain(list_fn: Callable[..., tuple], total: int) -> Iterator[dict[str, Any]]:
    """按页取完整审计链（列表接口按 seq 降序，这里还原为 seq 升序输出）。"""
    fetched: list[dict[str, Any]] = []
    offset = 0
    while offset < total:
        page, _ = list_fn(limit=_EXPORT_PAGE, offset=offset)
        if not page:
            break
        fetched.extend(page)
        offset += len(page)
    fetched.reverse()  # 降序 → 升序（归档件按链序只追加）
    return iter(fetched)


def build_audit_export(
    repository: Any, security_store: Any, actor: str
) -> tuple[str, dict[str, Any]]:
    """构造审计归档 JSONL。返回 (jsonl 文本, footer dict)。

    参数为两个 infra 仓储对象（主审计链 repository / 安全链 security_store），
    不依赖 app 层 Registry——调用方从各自持有的 registry 取用即可。
    """
    main_valid = repository.verify_chain()
    sec_valid = security_store.verify_security_chain()
    main_entries, main_total = repository.list_audit(limit=_EXPORT_PAGE, offset=0)
    del main_entries
    sec_entries, sec_total = security_store.list_security_audit(limit=_EXPORT_PAGE, offset=0)
    del sec_entries

    now = datetime.now(UTC).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = [
        json.dumps(
            {
                "type": "export_header",
                "format": "scandetection-audit-export/1",
                "exported_at": now,
                "exported_by": actor,
                "main_chain_total": main_total,
                "security_chain_total": sec_total,
                "main_chain_valid": main_valid,
                "security_chain_valid": sec_valid,
            },
            ensure_ascii=False,
        )
    ]
    for chain, entries, valid in (
        ("main", _iter_full_chain(repository.list_audit, main_total), main_valid),
        (
            "security",
            _iter_full_chain(security_store.list_security_audit, sec_total),
            sec_valid,
        ),
    ):
        for e in entries:
            lines.append(
                json.dumps(
                    {
                        "type": "record",
                        "chain": chain,
                        "seq": e["seq"],
                        "record": e,
                        "chain_valid": valid,
                    },
                    ensure_ascii=False,
                )
            )
    footer = {
        "type": "export_footer",
        "main_chain_total": main_total,
        "security_chain_total": sec_total,
        "main_chain_valid": main_valid,
        "security_chain_valid": sec_valid,
        "exported_at": now,
    }
    lines.append(json.dumps(footer, ensure_ascii=False))
    return "\n".join(lines) + "\n", footer
