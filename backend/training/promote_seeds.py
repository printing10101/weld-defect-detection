"""
将用户已目视确认的 25 张种子预标注提升为正式标注。

规则（安全优先）：
  - 若 labels/{stem}.txt 已存在（用户手动保存过，含空标注），整文件保持不动，也不改 .review；
  - 否则将 prelabels/{stem}.txt 复制到 labels/，并写 .review="0"（用户已目视确认没问题）。

绝不 以预标注覆盖用户已保存的标注。
"""

import json
import shutil
from pathlib import Path

REAL = Path("data/real_label")
IMG = REAL / "images"
LBL = REAL / "labels"
PRE = REAL / "prelabels"
MANIFEST = REAL / "manifest.json"
LBL.mkdir(parents=True, exist_ok=True)

data = json.loads(MANIFEST.read_text(encoding="utf-8"))
seeds = [it["name"] for it in data["images"] if it.get("is_seed")]
print(f"种子总数: {len(seeds)}")

promoted = 0
kept = 0
skipped = 0
for name in seeds:
    stem = Path(name).stem
    lt = LBL / (stem + ".txt")
    pt = PRE / (stem + ".txt")
    if lt.exists():
        # 用户已保存（含空标注）→ 保留，不触碰
        kept += 1
        continue
    if pt.exists():
        shutil.copy(pt, lt)
        (LBL / (stem + ".review")).write_text("0", encoding="utf-8")
        promoted += 1
    else:
        skipped += 1
        print(f"  [warn] 无预标注可提升: {name}")

print(f"保留用户已存: {kept} | 提升预标注: {promoted} | 无预标注跳过: {skipped}")
print(f"正式标注总数: {kept + promoted}")
