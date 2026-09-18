"""RIAWELC 公开基准同源泄漏审计（研究用，方向 2-A 的量化证据）。

背景：RIAWELC（Totino et al., 2022）24,407 张 224×224 焊缝射线 patch 来自
约 72 张数字化底片的裁切，官方按 patch 级随机划分 train/valid/test——
同一物理底片的 patch 可能同时出现在训练与测试集，令基准表观性能虚高。
截至 2026-09 无公开的量化审计；本脚本用项目既有口径（md5 字节簇 +
dHash 64bit 汉明≤4 感知对，与 backend.domain.labeling.dataset_guard 一致）
加上「源底片分组」维度补上这个数字。

审计维度：
1. 字节重复簇：跨 split 的 md5 完全相同（硬泄漏）；
2. 感知疑似对：跨 split dHash 汉明≤阈值（8 带分桶检索，避免 O(n²)）；
3. 源底片分组：文件名可解析出底片 ID 时（多 patch 共享前缀），统计
   跨 split 组数与受污染图像占比；不可解析时如实报告并给出分桶基数证据。

用法：
  python scripts/audit_riawelc_leakage.py --root data/external/riawelc/Dataset_partitioned \
      --json data/eval/riawelc_leakage_audit.json
存在跨 split 泄漏 → 退出码 1（与 backend.training.audit_dataset_leakage 一致）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
# RIAWELC 官方分区目录名 → 规范 split 名
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "valid": "valid",
    "validation": "valid",
    "val": "valid",
    "test": "test",
    "testing": "test",
}
# 底片分组键：RRT-101R_Img1_A80_S1_[12][8] → 底片=RRT-101R_Img1_A80
_RIAWELC_PATCH = re.compile(r"^(?P<film>.+?)_S\d+_\[\d+\]\[\d+\]$")


def _dhash_of(path: Path, hash_size: int = 8) -> int | None:
    """dHash 感知哈希，算法与 dataset_guard.dhash 完全一致（灰度 9×8 横向差值）。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    resized = cv2.resize(img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    return int(np.packbits(diff.flatten()).tobytes().hex(), 16)


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _band_values(h: int, n_bands: int = 8) -> list[int]:
    """64bit 拆 8 带（每带 8bit）。汉明≤4 时必有≥1 带完全相同（鸽笼原理）。"""
    return [(h >> (8 * i)) & 0xFF for i in range(n_bands)]


def _find_image_root(root: Path) -> Path | None:
    """定位同时含 train/valid/test 任两别名的目录层级（兼容 class 子目录夹层）。"""
    if not root.is_dir():
        return None

    def hit(d: Path) -> int:
        return sum(1 for s in d.iterdir() if d.is_dir() and s.name.lower() in SPLIT_ALIASES)

    if hit(root) >= 2:
        return root
    for child in sorted(root.iterdir()):
        if child.is_dir() and hit(child) >= 2:
            return child
    return None


def _iter_images(split_dir: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(split_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out.append(p)
    return out


def detect_group_key(stems: list[str]) -> tuple[str | None, dict]:
    """探测文件名中的底片分组键：找一个分隔段，其基数远小于图像数。

    依据：约 72 张底片裁出 2.4 万 patch → 合法组键的基数应在 10² 量级、
    且平均组内 ≥30 张。返回 (组键正则描述, 诊断信息)；找不到返回 (None, ...)。
    """
    diag: dict = {"n_stems": len(stems), "candidates": []}
    if not stems:
        return None, diag
    sample = stems[: min(20, len(stems))]
    diag["sample"] = sample
    # 候选切分：下划线/点/横杠分隔的首段、前两段
    for sep_idx, desc in ((1, "第一段"), (2, "前两段")):
        keys = ["_".join(s.split("_")[:sep_idx]) if "_" in s else None for s in stems]
        keys = [k for k in keys if k]
        if not keys:
            continue
        card = len(set(keys))
        diag["candidates"].append({"key": desc, "cardinality": card})
        if 8 <= card <= 500 and len(stems) / card >= 20:
            return f"prefix{sep_idx}", diag
    return None, diag


def group_of(stem: str, mode: str | None) -> str:
    if mode is None:
        return stem
    if mode == "riawelc_film":
        m = _RIAWELC_PATCH.match(stem)
        return m.group("film") if m else stem
    if mode == "riawelc_section":
        # 保留 _S{n} 分段、剥掉网格坐标 [r][c]
        return re.sub(r"_\[\d+\]\[\d+\]$", "", stem)
    if mode == "prefix1":
        return stem.split("_")[0]
    if mode == "prefix2":
        return "_".join(stem.split("_")[:2])
    raise ValueError(f"未知分组模式: {mode}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RIAWELC 同源泄漏审计（md5 + dHash + 底片分组）")
    ap.add_argument(
        "--root", required=True, help="Dataset_partitioned 根（自动定位 train/test 层级）"
    )
    ap.add_argument("--json", default=None, help="报告落盘路径")
    ap.add_argument("--phash-hamming", type=int, default=4, help="感知哈希汉明阈值")
    ap.add_argument("--max-candidates", type=int, default=4000, help="单带桶最大候选对比数（防爆）")
    args = ap.parse_args(argv)

    root = _find_image_root(Path(args.root))
    if root is None:
        print(f"[audit] 未在 {args.root} 下找到含 train/valid/test 别名的目录层级")
        return 2
    split_dirs = {
        SPLIT_ALIASES[d.name.lower()]: d
        for d in sorted(root.iterdir())
        if d.is_dir() and d.name.lower() in SPLIT_ALIASES
    }
    print(f"[audit] splits: {list(split_dirs)}  root={root}")

    # 单次扫描：路径 / 字节 md5 / dHash
    t0 = time.time()
    meta: list[dict] = []
    for split, d in split_dirs.items():
        imgs = _iter_images(d)
        for i, p in enumerate(imgs):
            raw_md5 = hashlib.md5(p.read_bytes()).hexdigest()
            ph = _dhash_of(p)
            cls = p.parent.name if p.parent != d else "unknown"
            meta.append(
                {
                    "name": f"{split}/{p.relative_to(d).as_posix()}",
                    "split": split,
                    "cls": cls,
                    "stem": p.stem,
                    "md5": raw_md5,
                    "phash": ph,
                }
            )
            if i % 2000 == 0:
                print(f"[audit] {split}: {i}/{len(imgs)}  ({time.time() - t0:.0f}s)")
    n = len(meta)
    per_split = {s: sum(1 for m in meta if m["split"] == s) for s in split_dirs}
    print(f"[audit] 扫描完成 {n} 张（{time.time() - t0:.0f}s）: {per_split}")

    report: dict = {
        "dataset": "RIAWELC",
        "source": "github.com/stefyste/RIAWELC Dataset_partitioned",
        "n_images": n,
        "per_split": per_split,
        "per_class_split": {
            c: dict(Counter(m["split"] for m in meta if m["cls"] == c))
            for c in sorted({m["cls"] for m in meta})
        },
        "phash_hamming": args.phash_hamming,
    }

    # 1) 字节重复簇
    by_md5: dict[str, list[dict]] = defaultdict(list)
    for m in meta:
        by_md5[m["md5"]].append(m)
    exact_cross = [
        {"files": [m["name"] for m in members], "splits": sorted({m["split"] for m in members})}
        for members in by_md5.values()
        if len(members) > 1 and len({m["split"] for m in members}) > 1
    ]
    n_inner_dup = sum(1 for members in by_md5.values() if len(members) > 1) - len(exact_cross)
    report["exact_cross_split_clusters"] = exact_cross
    report["exact_cross_split_images"] = sum(len(c["files"]) for c in exact_cross)
    report["exact_within_split_clusters"] = n_inner_dup
    print(f"[audit] 跨split字节重复簇={len(exact_cross)}（split内 {n_inner_dup} 簇）")

    # 2) 感知疑似对（8 带分桶，仅跨 split）
    bands: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, m in enumerate(meta):
        if m["phash"] is None:
            continue
        for bi, bv in enumerate(_band_values(m["phash"])):
            bands[(bi, bv)].append(i)
    seen: set[tuple[int, int]] = set()
    pair_list: list[dict] = []
    dist_hist: Counter = Counter()
    skipped = 0
    for (bi, bv), idxs in bands.items():
        if len(idxs) < 2 or len(idxs) > args.max_candidates:
            skipped += max(0, len(idxs) - args.max_candidates)
            continue
        for a_pos in range(len(idxs)):
            ia = idxs[a_pos]
            for ib in idxs[a_pos + 1 :]:
                if meta[ia]["split"] == meta[ib]["split"]:
                    continue
                key = (min(ia, ib), max(ia, ib))
                if key in seen:
                    continue
                d = _hamming(meta[ia]["phash"], meta[ib]["phash"])
                if d <= args.phash_hamming:
                    seen.add(key)
                    dist_hist[d] += 1
                    if len(pair_list) < 200:
                        pair_list.append(
                            {"a": meta[ia]["name"], "b": meta[ib]["name"], "hamming": d}
                        )
    report["perceptual_cross_split_pairs"] = len(seen)
    report["perceptual_distance_hist"] = {str(k): v for k, v in sorted(dist_hist.items())}
    report["perceptual_examples"] = pair_list
    print(
        f"[audit] 跨split感知疑似对={len(seen)}  距离分布={dict(sorted(dist_hist.items()))}"
        f"  超限桶截断={skipped}"
    )

    # 3) 源底片分组：RIAWELC 命名（…_S{n}_[r][c]）直接解析；否则退回前缀探测
    stems = [m["stem"] for m in meta]
    n_match = sum(1 for s in stems if _RIAWELC_PATCH.match(s))
    if n_match >= len(stems) * 0.9:
        mode = "riawelc_film"
        diag = {
            "n_stems": len(stems),
            "regex_match": n_match,
            "sample": stems[:10],
        }
    else:
        mode, diag = detect_group_key(stems)
    report["group_key_detection"] = {"mode": mode, **diag}
    if mode:
        by_group: dict[str, set[str]] = defaultdict(set)
        for m in meta:
            by_group[group_of(m["stem"], mode)].add(m["split"])
        cross = {g: sorted(s) for g, s in by_group.items() if len(s) > 1}
        n_cross_imgs = sum(1 for m in meta if len(by_group[group_of(m["stem"], mode)]) > 1)
        report["film_groups_total"] = len(by_group)
        report["film_groups_cross_split"] = len(cross)
        report["film_groups_cross_ratio"] = round(len(cross) / max(len(by_group), 1), 4)
        report["images_in_cross_split_groups"] = n_cross_imgs
        report["images_in_cross_split_ratio"] = round(n_cross_imgs / max(n, 1), 4)
        report["cross_split_group_examples"] = dict(list(cross.items())[:20])
        print(
            f"[audit] 底片组={len(by_group)}  跨split组={len(cross)}"
            f"（{report['film_groups_cross_ratio']:.1%}）  受污染图像={n_cross_imgs}"
            f"（{report['images_in_cross_split_ratio']:.1%}）"
        )
        # 二级口径：分段级（保留 _S{n}），区分"同底片异段"与"同段"泄漏
        by_sec: dict[str, set[str]] = defaultdict(set)
        for m in meta:
            by_sec[group_of(m["stem"], "riawelc_section" if mode == "riawelc_film" else mode)].add(
                m["split"]
            )
        sec_cross = {g: sorted(s) for g, s in by_sec.items() if len(s) > 1}
        report["section_groups_total"] = len(by_sec)
        report["section_groups_cross_split"] = len(sec_cross)
        print(
            f"[audit] 分段组={len(by_sec)}  跨split分段组={len(sec_cross)}"
            f"（{len(sec_cross) / max(len(by_sec), 1):.1%}）"
        )
    else:
        report["film_groups_total"] = None
        print("[audit] 文件名未探测到底片分组键——组级统计不可用，以重复/感知维度为准")

    leaked = bool(exact_cross or seen or (mode and report.get("film_groups_cross_split")))
    report["leaked"] = leaked
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[audit] 报告: {args.json}")
    print("[audit] 结论:", "存在跨split泄漏" if leaked else "未发现跨split泄漏")
    return 1 if leaked else 0


if __name__ == "__main__":
    sys.exit(main())
