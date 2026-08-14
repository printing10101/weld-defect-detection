"""SWRD 自动获取 + 转换（CC BY 4.0，可训练 + 可用于比赛，须署名）。

SWRD 官方（tz-ndt.com）为浏览器登录下载，无法脚本直链。本模块职责：
1. print_download_guide()：给出人工下载步骤与署名要求（合规红线）。
2. verify(src)：校验 data/external/swrd 是否含图像与标注。
3. ingest()：用户浏览器下载落盘后，调用 swrd_converter.convert() 转 YOLO，
   再 dataset_builder.build_dataset() 装配训练集；返回 data.yaml 路径。
4. 可选：若设置了环境变量 SWRD_MIRROR_URL 且可达，尝试自动下载（未来扩展点）。

引用（比赛/论文须署名，DATA_LICENSE.md 已存 BibTeX）：
  Gao, Y. et al. "SWRD: Ship weld X-ray image defect detection dataset."
"""

from __future__ import annotations

import os
from pathlib import Path

from backend.training import dataset_builder, swrd_converter

_SRC = Path("data/external/swrd")
_DST = Path("data/training/raw/swrd")


def print_download_guide() -> None:
    print(
        "── SWRD 下载指引（CC BY 4.0，可训练 + 可用于比赛，必须署名）──\n"
        "1. 浏览器打开 https://tz-ndt.com/SWRD 并登录/申请下载。\n"
        "2. 把下载到的压缩包解压到：data/external/swrd/  （保持图像的原始目录结构）。\n"
        "3. 标注格式支持：COCO instances_*.json / 逐图同名 .json / Pascal VOC .xml / 已是 YOLO .txt。\n"
        "4. 运行：python -m backend.training.download_swrd --ingest\n"
        "5. 比赛/论文中按 DATA_LICENSE.md 的 BibTeX 署名 SWRD 作者。\n"
        "⚠️ 切勿把 SWRD 原始数据重新分发或闭源商用；仅用于训练你自己的模型与合规参赛。"
    )


def verify(src: Path | None = None) -> bool:
    src = Path(src or _SRC)
    if not src.exists():
        print(f"[swrd] 未找到 {src}，请先下载（--guide）。")
        return False
    imgs = list(src.rglob("*.jpg")) + list(src.rglob("*.png")) + list(src.rglob("*.tif"))
    ann = (
        list(src.rglob("instances_*.json"))
        + list(src.rglob("*.json"))
        + list(src.rglob("*.xml"))
        + list(src.rglob("*.txt"))
    )
    print(f"[swrd] 图像 {len(imgs)} 张，标注/候选 {len(ann)} 个。")
    return len(imgs) > 0 and len(ann) > 0


def ingest(src: Path | None = None, dst: Path | None = None) -> Path:
    src = Path(src or _SRC)
    dst = Path(dst or _DST)
    if not verify(src):
        raise RuntimeError("SWRD 校验未通过，请先下载（--guide）。")
    n = swrd_converter.convert(src, dst)
    print(f"[swrd] 已转换 {n} 张 → {dst}")
    if n == 0:
        raise RuntimeError("转换 0 张，请检查 data/external/swrd 下的标注格式。")
    return dataset_builder.build_dataset()


def _try_mirror() -> bool:
    url = os.environ.get("SWRD_MIRROR_URL")
    if not url:
        return False
    # 未来扩展：自动下载镜像（需确认镜像同样为 CC BY 4.0）。当前不实现以免误用未授权源。
    print(f"[swrd] 发现 SWRD_MIRROR_URL 但未启用自动下载（合规待确认）：{url}")
    return False


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="SWRD 获取/转换（CC BY 4.0）")
    ap.add_argument("--guide", action="store_true", help="打印下载指引")
    ap.add_argument("--ingest", action="store_true", help="转换 + 装配训练集")
    args = ap.parse_args()
    if args.guide or not args.ingest:
        print_download_guide()
    if args.ingest:
        if _try_mirror():
            return
        yaml_path = ingest()
        print(f"[swrd] 完成，训练集配置：{yaml_path}")


if __name__ == "__main__":
    main()
