"""Roboflow X 光焊缝数据集接入（，绕开 SWRD 115GB 死穴）。

已核实许可、可用于比赛的候选集：
- Danila "X-ray Weld Defect" : Public Domain, 416 张, YOLOv8, 640px（推荐主训练集补充）
- Cassius Fro "XrayWeld"      : CC BY 4.0, 619 张, 实例分割（类别匿名 0-4，需补语义映射）

获取（Roboflow 有 Cloudflare 防护，脚本直链会被拦，需用户侧下载）：
  浏览器：universe.roboflow.com/<owner>/<project> → Export → YOLOv8 → Download
  或 CLI：roboflow download --dataset <url> --model <ver> --format yolov8 --api_key <KEY>
把导出的 zip 解压到 data/external/roboflow/<name>/（含 data.yaml + {train,valid,test}）。

类别映射复用 class_map.map_source_label（英文/中文/关键字容错）。
未匹配类别的行被跳过并告警；缺类（如咬边）由用户 165 张目标域补齐。
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from backend.training import dataset_builder
from backend.training.class_map import map_source_label

_EXTERNAL = Path("data/external/roboflow")
_RAW = Path("data/training/raw/roboflow")


def _load_names(export_dir: Path) -> list[str]:
    """解析 Roboflow 生成的 data.yaml 的 names（支持单行 list 与块 list 两种）。"""
    text = (export_dir / "data.yaml").read_text(encoding="utf-8")
    m = re.search(r"^names:\s*\[(.+?)\]\s*$", text, re.MULTILINE)
    if m:
        return [x.strip().strip("'").strip('"') for x in m.group(1).split(",") if x.strip()]
    names: list[str] = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^names:\s*$", line):
            j = i + 1
            while j < len(lines) and re.match(r"^\s*-\s+", lines[j]):
                names.append(lines[j].split("-", 1)[1].strip().strip("'").strip('"'))
                j += 1
            if names:
                return names
    return names


def _find_img(img_dir: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        p = img_dir / (stem + ext)
        if p.exists():
            return p
    return None


def ingest(name: str, export_dir: Path | None = None) -> Path:
    export_dir = Path(export_dir or _EXTERNAL / name)
    yaml_path = export_dir / "data.yaml"
    if not yaml_path.exists():
        raise RuntimeError(f"未找到 {yaml_path}。请先把 Roboflow 导出的 YOLOv8 zip 解压到该目录。")
    names = _load_names(export_dir)
    print(f"[roboflow] {name} 原始类别({len(names)}): {names}")

    dst_img = _RAW / "images"
    dst_lbl = _RAW / "labels"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    mapped = 0
    skipped = 0
    unmapped: set[str] = set()
    copied_imgs = 0

    for split in ("train", "valid", "test"):
        lbl_dir = export_dir / split / "labels"
        img_dir = export_dir / split / "images"
        if not lbl_dir.exists():
            continue
        for lbl in sorted(lbl_dir.glob("*.txt")):
            img = _find_img(img_dir, lbl.stem)
            if img is None:
                print(f"[roboflow] 警告: 缺图像 {lbl.stem}（{split}），跳过")
                continue
            new_name = f"{name}_{split}_{img.name}"
            shutil.copy(img, dst_img / new_name)
            copied_imgs += 1
            out_lines: list[str] = []
            for line in lbl.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                try:
                    cid = int(parts[0])
                except ValueError:
                    continue
                src_name = names[cid] if cid < len(names) else f"idx{cid}"
                cls = map_source_label(src_name)
                if cls is None:
                    skipped += 1
                    unmapped.add(src_name)
                    continue
                out_lines.append(f"{cls.value} {' '.join(parts[1:])}")
                mapped += 1
            (dst_lbl / (Path(new_name).stem + ".txt")).write_text("\n".join(out_lines))

    print(
        f"[roboflow] 复制图像 {copied_imgs} 张；映射框 {mapped} 个；"
        f"跳过未匹配 {skipped} 个（类别: {sorted(unmapped) or '无'}）"
    )
    if mapped == 0:
        print(
            "[roboflow] ⚠️ 映射框为 0：该集类别名无法识别（如 XrayWeld 匿名 0-4）。"
            "请提供类别语义映射，或仅在域预训练中使用（不作为检测训练集）。"
        )
    return dataset_builder.build_dataset()


def main() -> None:
    ap = argparse.ArgumentParser(description="Roboflow X 光焊缝集接入（CC/PD，可比赛）")
    ap.add_argument("--name", required=True, help="数据集名，对应 data/external/roboflow/<name>/")
    ap.add_argument("--export-dir", default=None, help="可选：覆盖导出目录")
    args = ap.parse_args()
    yaml_out = ingest(args.name, args.export_dir)
    print(f"[roboflow] 完成，训练集配置：{yaml_out}")


if __name__ == "__main__":
    main()
