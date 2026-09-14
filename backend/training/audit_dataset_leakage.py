"""数据集泄漏审计 CLI（DB50/T 1807-2025 §8.3.1 延伸）。

用法：
  python -m backend.training.audit_dataset_leakage \
      --train data/training/train --val data/training/val --test data/training/test \
      --json data/training/leakage_audit.json

目录既可指向 split 根（自动取其 images/ 子目录），也可直接指向图像目录。
审计三维：跨 split 字节重复簇、感知哈希疑似对、同源底片分组越界
（copy-paste 合成图双亲 / 过采样副本归属，见 domain.labeling.leakage）。
存在泄漏 → 退出码 1，训练前 CI/人工拦截；--enforce-groups 时分组越界同样阻断。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from backend.domain.labeling.leakage import audit_leakage


def _images_dir(p: str) -> Path:
    root = Path(p)
    img = root / "images"
    return img if img.is_dir() else root


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="训练集泄漏审计（字节/感知/同源底片分组）")
    ap.add_argument("--train", required=True, help="训练集目录（split 根或 images 目录）")
    ap.add_argument("--val", default=None, help="验证集目录（可选）")
    ap.add_argument("--test", default=None, help="测试集目录（可选）")
    ap.add_argument("--phash-hamming", type=int, default=4, help="感知哈希汉明距离阈值")
    ap.add_argument("--enforce-groups", action="store_true", help="同源底片组跨 split 时判失败")
    ap.add_argument("--json", default=None, help="报告落盘路径（可选）")
    args = ap.parse_args(argv)

    splits = {"train": args.train}
    if args.val:
        splits["val"] = args.val
    if args.test:
        splits["test"] = args.test
    if len(splits) < 2:
        ap.error("至少需要两个 split 才能审计跨 split 泄漏")

    audit = audit_leakage(
        {s: _images_dir(d) for s, d in splits.items()},
        phash_hamming=args.phash_hamming,
        enforce_groups=args.enforce_groups,
    )
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps(audit.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(f"split 规模: {audit.per_split} 共 {audit.n_files} 张")
    cross_dup = [c for c in audit.duplicate_clusters if c["cross_split"]]
    inner_dup = len(audit.duplicate_clusters) - len(cross_dup)
    print(
        f"跨split重复簇={len(cross_dup)}（split内 {inner_dup} 簇仅记录） "
        f"感知疑似对={len(audit.perceptual_pairs)} "
        f"跨split同源组={len(audit.cross_split_groups)}"
    )
    for c in cross_dup[:10]:
        print("  [exact]", c["files"])
    for p in audit.perceptual_pairs[:10]:
        print("  [perceptual]", p["a"], "~", p["b"], f"(hamming={p['hamming']})")
    for g in audit.cross_split_groups[:10]:
        print("  [group]", g["group"], "→", sorted(g["splits"]))  # type: ignore[arg-type]
    if args.json:
        print(f"报告: {args.json}")
    ok = audit.passed
    print("结论:", "通过" if ok else "未通过（存在泄漏）")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
