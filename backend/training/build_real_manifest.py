"""
构建真实图标注清单 manifest.json（方案 B 关键修正）

问题背景：
  - data/real_label/images 下 165 张图里有 19 对"字节级完全相同"的重复
    （如 PG101-1-2.jpg 与 PG101-1-2dcn.jpg 是同一张；另有 PG101-1-5jpg.jpg 这类
    命名错乱的副本）。
  - 原种子集按文件名排序取前 25 张，全部来自 PG101/PG102 两个试板族，
    完全没有覆盖其余 7 个试板族（PG103 42、PL118 41…），代表性差。

本脚本做两件事：
  1) 内容哈希去重 → 150 张唯一底片（每对重复只保留"命名更干净"的一张）。
  2) 跨试板族分层抽样选 25 张种子：每个族至少 1 张，剩余按族大小比例分配，
     使种子覆盖全部 9 个试板族且大族占更多。

产出 data/real_label/manifest.json：
  { "n_total_files", "n_canonical", "n_seed", "families",
    "images": [ {"name", "family", "is_seed"}, ... 150 项，按族排序 ] }

标注服务与 real_finetune 都只读这份清单，保证去重+分层抽样一致生效。
"""

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

REAL = Path("data/real_label")
IMG = REAL / "images"
MANIFEST = REAL / "manifest.json"
SEED_N = 25


# 命名清洁度评分（越低越优先保留）：dcn 副本、双扩展名副本都降级
def clean_score(name: str) -> int:
    s = 0
    if "dcn" in name.lower():
        s += 2
    if re.search(r"jpg\.jpg$", name.lower()):  # 形如 xxx5jpg.jpg
        s += 2
    return s


def family_of(name: str) -> str:
    # PG101-1-2.jpg -> PG101 ; PG12-2-1.jpg -> PG12 ; PL117-... -> PL117
    m = re.match(r"^([A-Za-z]+\d+)", name)
    return m.group(1) if m else name.split("-")[0]


def main():
    files = sorted(p.name for p in IMG.glob("*.jpg"))
    if not files:
        raise SystemExit("未发现任何图像：data/real_label/images/ 为空。")

    # 1) 内容哈希分组 → 每组合并为一唯一底片
    groups = defaultdict(list)  # hash -> [names]
    for name in files:
        h = hashlib.md5((IMG / name).read_bytes()).hexdigest()
        groups[h].append(name)

    canonical = []
    dropped = []
    for h, names in groups.items():
        if len(names) == 1:
            canonical.append(names[0])
        else:
            # 选清洁度最高（评分最低）、同分时字典序最小的一张保留
            keep = min(names, key=lambda n: (clean_score(n), n))
            canonical.append(keep)
            dropped.extend(n for n in names if n != keep)

    canonical = sorted(canonical)
    print(
        f"[manifest] 文件总数={len(files)}  去重后唯一底片={len(canonical)}  "
        f"丢弃重复={len(dropped)}"
    )
    if dropped:
        print("  丢弃的重复（保留其干净副本）：")
        for d in dropped:
            print("   -", d)

    # 2) 跨试板族分层抽样 25 张种子
    by_fam = defaultdict(list)
    for n in canonical:
        by_fam[family_of(n)].append(n)
    for f in by_fam:
        by_fam[f].sort()

    fam_counts = {f: len(v) for f, v in by_fam.items()}
    families = sorted(by_fam.keys(), key=lambda f: -fam_counts[f])
    print("[manifest] 试板族分布:", fam_counts)

    # 每族至少 1 张；剩余 (SEED_N - n_families) 按族大小比例分配
    n_fam = len(families)
    base = dict.fromkeys(families, 1)
    remain = SEED_N - n_fam
    total = sum(fam_counts.values())
    # 比例分配（最大余数法）
    alloc = {f: remain * fam_counts[f] / total for f in families}
    floor = {f: int(alloc[f]) for f in families}
    extras = dict(floor)  # 先取整数部分，再补余数
    used = sum(floor.values())
    rem = remain - used
    # 余数从大到小补
    order = sorted(families, key=lambda f: -(alloc[f] - floor[f]))
    for f in order:
        if rem <= 0:
            break
        extras[f] += 1
        rem -= 1
    seed_quota = {f: base[f] + extras[f] for f in families}
    # 修正：若某族配额超过其规模，截断
    for f in families:
        seed_quota[f] = min(seed_quota[f], fam_counts[f])

    seed_set = set()
    for f in families:
        members = by_fam[f]
        k = seed_quota[f]
        if k <= 0:
            continue
        if k >= len(members):
            picks = members
        else:
            # 在族内均匀抽取 k 张，提升族内多样性
            idx = [round(i * (len(members) - 1) / (k - 1)) for i in range(k)] if k > 1 else [0]
            picks = [members[i] for i in idx]
        for p in picks:
            seed_set.add(p)
        print(f"  族 {f}: 规模={fam_counts[f]} 种子配额={k} 抽取={[x[:-4] for x in picks]}")

    # 3) 写清单（按族排序，种子在前）
    images = []
    for f in families:
        for n in by_fam[f]:
            images.append({"name": n, "family": f, "is_seed": n in seed_set})

    manifest = {
        "generated_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "n_total_files": len(files),
        "n_canonical": len(canonical),
        "n_seed": len(seed_set),
        "families": fam_counts,
        "seed_quota": seed_quota,
        "images": images,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[manifest] 已写 {MANIFEST}  唯一底片={len(canonical)}  种子={len(seed_set)}")


if __name__ == "__main__":
    main()
