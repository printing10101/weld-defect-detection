"""训练前数据互斥校验 CLI（DB50/T 1807-2025 ）。

用法：
  python -m backend.training.check_dataset_disjoint \
      --train data/training/train/images --test data/training/test/images
重叠（字节或感知哈希疑似）→ 退出码 1，训练前 CI/人工拦截。
"""

from __future__ import annotations

import argparse
import json
import sys

from backend.domain.labeling.dataset_guard import find_overlaps


def main() -> int:
    ap = argparse.ArgumentParser(description="训练/测试集互斥校验（§8.3.1）")
    ap.add_argument("--train", required=True, help="训练集图像目录")
    ap.add_argument("--test", required=True, help="测试集图像目录")
    ap.add_argument("--phash-hamming", type=int, default=4, help="感知哈希汉明距离阈值")
    ap.add_argument("--allow-perceptual", action="store_true", help="放行疑似重复（仍记录清单）")
    ap.add_argument("--json", default=None, help="报告落盘路径（可选）")
    args = ap.parse_args()

    report = find_overlaps(args.train, args.test, phash_hamming=args.phash_hamming)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, ensure_ascii=False, indent=2)
    print(f"train={report.n_train} test={report.n_test} "
          f"字节重叠={len(report.exact)} 疑似重复={len(report.perceptual)}")
    for e in report.exact[:10]:
        print("  [exact]", e)
    for p in report.perceptual[:10]:
        print("  [perceptual]", p)
    ok = report.passed or (args.allow_perceptual and not report.exact)
    print("结论:", "通过" if ok else "未通过（互斥性被破坏）")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
