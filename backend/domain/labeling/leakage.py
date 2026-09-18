"""数据集泄漏审计（DB50/T 1807-2025 §8.3.1 延伸口径）。

dataset_guard 回答"两个 split 目录里有没有同一张图"（md5 + 感知哈希）；
本模块回答更难的问题——**同源底片泄漏**：

1. 一张物理底片的多张衍生图（裁剪 patch、copy-paste 合成、过采样副本）
   是否跨越 train/val/test。评估图只要与训练图同源，指标即被乐观污染
   （RIAWELC 隐患：patch 级随机划分令同一底片的 patch 跨 split，
   表观性能可被同源泄漏推高；截至 2026-09 该虚高幅度尚无公开量化审计）；
2. 字节/感知重复按"源底片组"归因落盘，供部署后评估闭环引用审计。

同源等价类（assign_groups，并查集，对 filename 主干）：
- `os{N}_` 过采样前缀、`dup{N}_` 跨源同名去重前缀剥离（副本与原图同组，
  且两者只会在同一 split 内出现）；
- `cp_/rcp_{序号}_{src}_x_{tgt}`（augment.generate_copy_paste 命名）→
  合成图与两个亲本 src、tgt 并入同一等价类：任一亲本落在评估集而
  合成图落在训练集即判泄漏；
- 其余按主干自身分组；等价类代表取类内最小编号底片名（可读、确定性）。
单名解析见 film_group（不跨名归并，池级等价类必须用 assign_groups）。
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from backend.domain.labeling.dataset_guard import IMG_EXTS, dhash, hamming, md5_of

_OS_PREFIX = re.compile(r"^(?:(?:os|dup)\d+_)+")
_CP_NAME = re.compile(r"^(?:rcp|cp)_\d+_(.+)_x_(.+)$")


def film_group(stem: str) -> str:
    """单名派生解析：剥离过采样前缀；copy-paste 图返回 `src|tgt`。

    注意合成图与亲本的等价关系需池级归并（assign_groups），本函数只做
    单名规范化——`cp_a_x_b` 返回 "a|b" 而非 "a" 或 "b"。
    """
    s = _OS_PREFIX.sub("", stem)
    m = _CP_NAME.match(s)
    if m:
        return "|".join(sorted((m.group(1), m.group(2))))
    return s


def assign_groups(stems: Iterable[str]) -> dict[str, str]:
    """池级同源等价类：过采样副本、copy-paste 合成图与亲本并查集归并。

    返回 {原始stem: 等价类代表}；代表取类内最小编号底片名（合成图名
    不作代表，保证报告可读），与元素枚举顺序无关。
    """
    stems = list(stems)  # 三轮遍历（归并/成员/映射），生成器入参先物化
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # 路径压缩
            parent[x], x = root, parent[x]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            if rb < ra:
                ra, rb = rb, ra
            parent[rb] = ra

    for s in stems:
        base = _OS_PREFIX.sub("", s)
        union(s, base)
        m = _CP_NAME.match(base)
        if m:
            union(base, m.group(1))
            union(base, m.group(2))

    members: dict[str, list[str]] = {}
    for s in stems:
        members.setdefault(find(s), []).append(s)
    rep_of: dict[str, str] = {}
    for root, names in members.items():
        derived = {n for n in names if _CP_NAME.match(_OS_PREFIX.sub("", n))}
        plain = [n for n in names if n not in derived]
        rep_of[root] = min(plain) if plain else min(derived)
    return {s: rep_of[find(s)] for s in stems}


@dataclass
class PoolEntry:
    """单张图像的审计条目。"""

    name: str  # "<split>/<filename>"
    split: str
    stem: str
    md5: str
    phash: int | None = None
    group: str = ""  # audit_leakage 池级归并后填充


@dataclass
class LeakageAudit:
    """泄漏审计报告：重复簇 + 感知疑似对 + 跨 split 分组。"""

    per_split: dict[str, int] = field(default_factory=dict)
    duplicate_clusters: list[dict[str, object]] = field(default_factory=list)
    perceptual_pairs: list[dict[str, object]] = field(default_factory=list)
    cross_split_groups: list[dict[str, object]] = field(default_factory=list)
    phash_hamming: int = 4
    enforce_groups: bool = False

    @property
    def n_files(self) -> int:
        return sum(self.per_split.values())

    @property
    def passed(self) -> bool:
        exact_leak = any(c["cross_split"] for c in self.duplicate_clusters)
        group_breach = self.enforce_groups and bool(self.cross_split_groups)
        return not (exact_leak or self.perceptual_pairs or group_breach)

    def to_dict(self) -> dict:
        return {
            "per_split": self.per_split,
            "n_files": self.n_files,
            "duplicate_clusters": self.duplicate_clusters,
            "perceptual_pairs": self.perceptual_pairs,
            "cross_split_groups": self.cross_split_groups,
            "phash_hamming": self.phash_hamming,
            "enforce_groups": self.enforce_groups,
            "passed": self.passed,
        }


def scan_split(split: str, img_dir: str | Path) -> list[PoolEntry]:
    """扫描一个 split 的图像目录，计算 md5 / 感知哈希（分组键由池级归并定）。"""
    root = Path(img_dir)
    entries: list[PoolEntry] = []
    if not root.is_dir():
        return entries
    for p in sorted(root.iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        entries.append(
            PoolEntry(
                name=f"{split}/{p.name}",
                split=split,
                stem=p.stem,
                md5=md5_of(p),
                phash=dhash(p),
            )
        )
    return entries


def audit_leakage(
    splits: Mapping[str, str | Path],
    *,
    phash_hamming: int = 4,
    enforce_groups: bool = False,
) -> LeakageAudit:
    """对若干命名 split 的图像目录做泄漏审计（纯计算，不落盘）。"""
    pool: list[PoolEntry] = []
    for split, d in splits.items():
        pool.extend(scan_split(split, d))

    audit = LeakageAudit(
        per_split={s: sum(1 for e in pool if e.split == s) for s in splits},
        phash_hamming=phash_hamming,
        enforce_groups=enforce_groups,
    )

    # 池级同源归并：合成图/副本与亲本同一等价类，再挂到各条目
    group_of = assign_groups(e.stem for e in pool)
    for e in pool:
        e.group = group_of[e.stem]

    # 1) 字节重复簇：跨 split 即泄漏；簇内归因分组键
    by_md5: dict[str, list[PoolEntry]] = {}
    for e in pool:
        by_md5.setdefault(e.md5, []).append(e)
    for h, members in by_md5.items():
        if len(members) < 2:
            continue
        splits_hit = sorted({m.split for m in members})
        audit.duplicate_clusters.append(
            {
                "md5": h,
                "files": [m.name for m in members],
                "groups": sorted({m.group for m in members}),
                "splits": splits_hit,
                "cross_split": len(splits_hit) > 1,
            }
        )

    # 2) 感知疑似对：仅跨 split 判定（split 内部相似不构成评估泄漏）
    for i, a in enumerate(pool):
        if a.phash is None:
            continue
        for b in pool[i + 1 :]:
            if b.split == a.split or b.phash is None:
                continue
            dist = hamming(a.phash, b.phash)
            if dist <= phash_hamming:
                audit.perceptual_pairs.append(
                    {
                        "a": a.name,
                        "b": b.name,
                        "hamming": dist,
                        "same_group": a.group == b.group,
                    }
                )

    # 3) 跨 split 分组：同一源底片的衍生图散落多个 split（结构性泄漏）
    by_group: dict[str, dict[str, list[str]]] = {}
    for e in pool:
        by_group.setdefault(e.group, {}).setdefault(e.split, []).append(e.name)
    for g, per in by_group.items():
        if len(per) > 1:
            audit.cross_split_groups.append({"group": g, "splits": per})
    audit.cross_split_groups.sort(key=lambda c: sorted(c["splits"].keys()))  # type: ignore[arg-type,return-value]
    audit.duplicate_clusters.sort(key=lambda c: not c["cross_split"])  # type: ignore[arg-type,return-value]
    return audit


def assert_no_leakage(
    splits: Mapping[str, str | Path],
    *,
    phash_hamming: int = 4,
    enforce_groups: bool = False,
) -> LeakageAudit:
    """强校验：存在泄漏（或分组越界未放行）→ RuntimeError 阻断。"""
    audit = audit_leakage(splits, phash_hamming=phash_hamming, enforce_groups=enforce_groups)
    if audit.passed:
        return audit
    exact = [c for c in audit.duplicate_clusters if c["cross_split"]]
    msg = (
        f"数据泄漏审计未通过（DB50/T 1807-2025 §8.3.1）："
        f"跨split重复簇={len(exact)} 感知疑似对={len(audit.perceptual_pairs)} "
        f"跨split同源分组={len(audit.cross_split_groups)}"
    )
    if exact:
        msg += f" 首例: {exact[0]['files']}"
    elif audit.perceptual_pairs:
        msg += f" 首例: {audit.perceptual_pairs[0]['a']} ~ {audit.perceptual_pairs[0]['b']}"
    elif audit.cross_split_groups:
        msg += f" 首例: 组 {audit.cross_split_groups[0]['group']}"
    raise RuntimeError(msg)
