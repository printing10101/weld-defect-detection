"""python_embed 裁剪：剔除运行时不需要的内容，压缩安装包体积。

用途：``tauri build`` 打包前对 ``src/python_embed``（tauri.conf resources 直接
分发该目录）执行一次。目录不入 git，本脚本是可复现的裁剪产物——每台打包机
在构建前各自执行。

裁剪项（均可安全剔除，uvicorn 后端不依赖）：
- ``__pycache__``/``*.pyc``：字节码缓存。首次启动由 Python 自动重建
  （冷启动略慢一次，之后无差异）；
- 科学栈包内 tests/testing 目录（scipy/numpy/skimage 等）：仅包自测用；
- pip / setuptools / wheel：运行期不安装任何包；
- pytest 一族（pytest/_pytest/iniconfig/pluggy/py/pygments）：开发期测试工具，
  测试请用 `python -m pip install -e ".[dev]"` 的开发环境，不用本嵌入目录。
  （py 是 pytest 家族的兼容 shim，其内容 `import _pytest._py.error` 在
  _pytest 被删后即炸——留下它等于埋一个"存在却不可用"的毒模块。）
- 散落 ``*.whl``：site-packages 内不应有 wheel 文件（历史操作残留，
  单个可达数十 MB），一并清除。

用法：
    python scripts/slim_python_embed.py            # 裁剪 src/python_embed
    python scripts/slim_python_embed.py --target <dir>
    python scripts/slim_python_embed.py --dry-run  # 只打印将删除的内容与体积

注意：裁剪后该目录不能再用于运行后端单测（pytest 已被剔除）。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# 开发工具包（连同其 dist-info 一并剔除）
DEV_PACKAGES = [
    "pytest",
    "_pytest",
    "iniconfig",
    "pluggy",
    "py",
    "pygments",
    "pip",
    "setuptools",
    "wheel",
]
# 科学栈包内可剔除的测试目录名（相对于包根）
TEST_DIR_NAMES = {"tests", "testing"}
# 仅对这些大包做 tests 目录剔除（其他包 tests 目录极小，不值得遍历）
TEST_TRIM_PACKAGES = ["scipy", "numpy", "skimage", "cv2", "pandas", "matplotlib"]


def _dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def plan_removals(root: Path) -> list[Path]:
    """收集待删除目标（目录/文件），不执行删除。"""
    site = root / "Lib" / "site-packages"
    targets: list[Path] = []
    if not site.is_dir():
        return targets

    # 1) 全部 __pycache__ 与其外散落的 .pyc
    targets.extend(root.rglob("__pycache__"))
    targets.extend(p for p in root.rglob("*.pyc") if p.is_file())

    # 1b) 散落的 wheel 残留（site-packages 内不应存在 .whl；历史操作残留单个
    # 可达数十 MB）。注意 rglob 对目录/文件通吃，此处只要文件。
    targets.extend(p for p in root.rglob("*.whl") if p.is_file())

    # 2) 科学栈包内测试目录
    for pkg in TEST_TRIM_PACKAGES:
        pkg_dir = site / pkg
        if not pkg_dir.is_dir():
            continue
        for d in pkg_dir.rglob("*"):
            if d.is_dir() and d.name in TEST_DIR_NAMES and ".dist-info" not in str(d):
                targets.append(d)

    # 3) 开发工具包 + dist-info + Scripts 入口
    # 注意：pytest 家族的兼容 shim "py" 是 site-packages 下的单文件 py.py
    # （非目录），必须单独命中，否则删了 _pytest 却留下 import 即炸的毒 shim。
    scripts_dir = root / "Scripts"
    for name in DEV_PACKAGES:
        pkg_dir = site / name
        if pkg_dir.is_dir():
            targets.append(pkg_dir)
        elif (site / f"{name}.py").is_file():
            targets.append(site / f"{name}.py")
        targets.extend(site.glob(f"{name}-*.dist-info"))
    if scripts_dir.is_dir():
        for entry in scripts_dir.iterdir():
            stem = entry.name.lower()
            if any(stem.startswith(f"{n}") for n in ("pip", "pytest")):
                targets.append(entry)

    # 去重（嵌套目录可能重复出现）并保序
    seen: set[Path] = set()
    unique: list[Path] = []
    for t in targets:
        rp = t.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(t)
    return unique


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--target",
        default=str(Path(__file__).resolve().parents[1] / "src" / "python_embed"),
        help="python_embed 目录（默认 <repo>/src/python_embed）",
    )
    ap.add_argument("--dry-run", action="store_true", help="只统计，不删除")
    args = ap.parse_args()

    root = Path(args.target)
    if not (root / "python.exe").is_file() and not (root / "python3").is_file():
        print(f"错误：{root} 不像 python_embed 目录（缺 python.exe）", file=sys.stderr)
        return 2

    targets = plan_removals(root)
    total = sum(_dir_size(t) if t.is_dir() else t.stat().st_size for t in targets)
    before = _dir_size(root)
    print(f"目标：{root}")
    print(
        f"将删除 {len(targets)} 项，合计 {total / 1024 / 1024:.1f} MB（目录总 {before / 1024 / 1024:.1f} MB）"
    )
    if args.dry_run:
        for t in targets[:50]:
            print(f"  - {t.relative_to(root)}")
        if len(targets) > 50:
            print(f"  … 及另外 {len(targets) - 50} 项")
        return 0

    for t in targets:
        if t.is_dir():
            shutil.rmtree(t, ignore_errors=True)
        elif t.exists():
            t.unlink()
    after = _dir_size(root)
    print(
        f"完成：{before / 1024 / 1024:.1f} MB → {after / 1024 / 1024:.1f} MB"
        f"（省 {(before - after) / 1024 / 1024:.1f} MB）"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
