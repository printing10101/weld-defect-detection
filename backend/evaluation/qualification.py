"""评价/标注人员资质管理（DB50/T 1807-2025  / TSG Z8001）。

标准要求：
- 系统评价人员：RT(D)Ⅱ 级及以上；
- 标准测试集标注人员：RTⅡ 级及以上。

本实现口径（较标准收紧）：
- 持证类型/证书编号/有效期全部落盘（data/eval/std_personnel.json，路径可配），
  记录表须引用实际记录，不接受前端随手填姓名；
- 有效期过期 → 视为无资质；
- 资质不满足 → 评价记录表标记"资质不符合"，只出参考值不出正式分级结论。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

_CERT_RE = re.compile(r"RT\s*(?:\(D\))?\s*[-‐–]?\s*(Ⅰ|Ⅱ|III|II|I|1|2|3)\s*级?", re.IGNORECASE)


def parse_cert_level(cert_type: str) -> int | None:
    """持证类型 → 级别数值（1/2/3）；(D) 资格同样有效；无法解析返回 None。

    接受 "RT-Ⅱ"、"RT II"、"RT(D)II"、"rt2" 等常见写法。
    """
    m = _CERT_RE.search(cert_type or "")
    if not m:
        return None
    token = m.group(1).upper()
    # 全角罗马数字归一化（Ⅰ/Ⅱ/Ⅲ → I/II/III）
    token = {"Ⅰ": "I", "Ⅱ": "II", "Ⅲ": "III"}.get(token, token)
    return {"I": 1, "1": 1, "II": 2, "2": 2, "III": 3, "3": 3}[token]


@dataclass(frozen=True)
class Personnel:
    """一名持证人员（评价人员或标注人员）。"""

    name: str
    cert_type: str  # 如 "RT(D)-II"
    role: str  # "evaluator" | "labeler"
    cert_no: str = ""
    valid_until: str = ""  # ISO 日期；空 = 未注明（保守视为长期有效但记录表注明）

    @property
    def level(self) -> int | None:
        return parse_cert_level(self.cert_type)

    @property
    def expired(self) -> bool:
        if not self.valid_until:
            return False
        try:
            return date.fromisoformat(self.valid_until) < date.today()  # noqa: DTZ011 — 本地时区语义即所需
        except ValueError:
            return True  # 日期格式非法按过期处理


def required_level(role: str) -> int:
    """岗位最低级别要求：评价/标注人员均为Ⅱ级及以上；未知岗位取最严（3 级）。

    评价人员与标注人员的差别在资格类别（评价须 RT(D)，），不在级别。
    """
    return 2 if role in ("evaluator", "labeler") else 3


def check_personnel(people: list[Personnel]) -> dict[str, Any]:
    """资质校验：评价人员须 RT(D)Ⅱ+、标注人员须 RTⅡ+。

    返回 {"qualified": bool, "issues": [str], "evaluators": [...], "labelers": [...]}。
    任何岗位缺员、级别不足、证书过期、证书类型无法解析均计入 issues。
    """
    issues: list[str] = []
    evaluators = [p for p in people if p.role == "evaluator"]
    labelers = [p for p in people if p.role == "labeler"]
    if not evaluators:
        issues.append("缺少评价人员（§5.1：RT(D)Ⅱ 级及以上）")
    if not labelers:
        issues.append("缺少标注人员（§5.2：RTⅡ 级及以上）")
    for p in people:
        lv = p.level
        if lv is None:
            issues.append(f"{p.name}: 持证类型无法解析（{p.cert_type!r}）")
            continue
        if lv < required_level(p.role):
            issues.append(f"{p.name}: 持证 {p.cert_type} 低于岗位要求 {required_level(p.role)} 级")
        if p.expired:
            issues.append(f"{p.name}: 证书已过期/有效期非法（{p.valid_until or '未注明'}）")
    return {
        "qualified": not issues,
        "issues": issues,
        "evaluators": [p.__dict__ | {"level": p.level} for p in evaluators],
        "labelers": [p.__dict__ | {"level": p.level} for p in labelers],
    }


def load_personnel(path: str | Path) -> list[Personnel]:
    """从 JSON 文件读取人员资质（不存在/损坏 → 空表，由 check_personnel 报缺员）。"""
    p = Path(path)
    if not p.is_file():
        return []
    try:
        rows = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [
        Personnel(
            name=str(r.get("name", "")),
            cert_type=str(r.get("cert_type", "")),
            role=str(r.get("role", "")),
            cert_no=str(r.get("cert_no", "")),
            valid_until=str(r.get("valid_until", "")),
        )
        for r in rows
        if isinstance(r, dict)
    ]


def save_personnel(people: list[Personnel], path: str | Path) -> Path:
    """人员资质落盘（评价记录表的姓名/持证类型来源，）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps([p.__dict__ for p in people], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return p
