"""steel-tube 钢管焊缝缺陷数据集接入（M4b，来源 huangyebiaoke/steel-pipe-weld-defect-detection）。

GitHub Releases 773MB zip，已解压到 data/external/steel_tube/extracted/steel-tube-dataset-all/。
YOLO 格式：yolo/images/{train2021,val2021} + yolo/labels/{train2021,val2021}，8 类：
  0 air-hole 气孔     1 bite-edge 咬边     2 broken-arc 断弧     3 crack 裂缝
  4 hollow-bead 夹珠  5 overlap 焊瘤       6 slag-inclusion 夹渣  7 unfused 未融合

映射到 ScanDetection 6 类（DefectClass，ADR-010）：
  0 -> POROSITY, 1 -> UNDERCUT, 3 -> CRACK, 6 -> SLAG, 7 -> LACK_OF_FUSION
  2/4/5 无对应类 -> 跳过（断弧/夹珠/焊瘤，避免错误标注污染训练）
一张图若全部行都被跳过（只有不可映射缺陷）则整图跳过；
部分映射的图保留有效框，丢弃不可映射框。

输出 data/training/raw/steel/{images,labels}（dataset_builder 兼容）。
许可：GPL-3.0（代码）；数据用于训练/研究。商用请注意 GPL 传染风险。
"""

from __future__ import annotations

import shutil
from pathlib import Path

from backend.domain.dto import DefectClass

_SRC = Path("data/external/steel_tube/extracted/steel-tube-dataset-all/yolo")
_DST = Path("data/training/raw/steel")

# steel-tube 8 类 -> DefectClass；None = 跳过（无对应类）
_STEEL_TO_DEFECTCLASS: dict[int, DefectClass | None] = {
    0: DefectClass.POROSITY,
    1: DefectClass.UNDERCUT,
    2: None,  # broken-arc 断弧（电弧缺陷，非射线缺陷类别）
    3: DefectClass.CRACK,
    4: None,  # hollow-bead 夹珠
    5: None,  # overlap 焊瘤
    6: DefectClass.SLAG,
    7: DefectClass.LACK_OF_FUSION,
}

_NAMES = [
    "air-hole", "bite-edge", "broken-arc", "crack",
    "hollow-bead", "overlap", "slag-inclusion", "unfused",
]


def _map_line(line: str) -> tuple[int, str] | None:
    """解析 YOLO 行 'cid cx cy w h'，映射到 DefectClass；跳过返回 None。"""
    parts = line.split()
    if len(parts) != 5:
        return None
    try:
        cid = int(parts[0])
        rest = " ".join(parts[1:])
    except ValueError:
        return None
    cls = _STEEL_TO_DEFECTCLASS.get(cid)
    if cls is None:
        return None
    return cls.value, rest


def ingest(src: Path | None = None, dst: Path | None = None) -> int:
    """转换 steel-tube YOLO 标签到项目 6 类；返回有效图像数。"""
    src = Path(src or _SRC)
    dst = Path(dst or _DST)
    dst_img = dst / "images"
    dst_lbl = dst / "labels"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    total_imgs = 0
    total_boxes = 0
    skipped_boxes = 0
    skipped_imgs = 0
    by_cls: dict[str, int] = {}

    for split in ("train2021", "val2021"):
        img_dir = src / "images" / split
        lbl_dir = src / "labels" / split
        if not lbl_dir.exists():
            continue
        for lbl in sorted(lbl_dir.glob("*.txt")):
            lines: list[str] = []
            for raw in lbl.read_text(encoding="utf-8").splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                mapped = _map_line(raw)
                if mapped is None:
                    skipped_boxes += 1
                    continue
                cid, rest = mapped
                lines.append(f"{cid} {rest}")
                by_cls[str(cid)] = by_cls.get(str(cid), 0) + 1
            if not lines:  # 全部为不可映射缺陷 -> 整图跳过
                skipped_imgs += 1
                continue
            # 图像（yolo/images 下与标签同名，扩展名可能 jpg/png）
            img = None
            for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                cand = img_dir / (lbl.stem + ext)
                if cand.exists():
                    img = cand
                    break
            if img is None:
                skipped_imgs += 1
                continue
            new_name = f"steel_{lbl.stem}{img.suffix}"
            shutil.copy(img, dst_img / new_name)
            (dst_lbl / (Path(new_name).stem + ".txt")).write_text("\n".join(lines))
            total_imgs += 1
            total_boxes += len(lines)

    print(
        f"[steel] 图像 {total_imgs} 张 / 有效框 {total_boxes} 个 / 跳过框 {skipped_boxes} / "
        f"整图跳过 {skipped_imgs}"
    )
    print(f"[steel] 映射后类别分布: {dict(sorted(by_cls.items()))} (0气孔 1夹渣 2未焊透 3未熔合 4裂纹 5咬边)")
    return total_imgs


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="steel-tube 接入（8 类 → 6 类 YOLO）")
    ap.add_argument("--src", default=str(_SRC))
    ap.add_argument("--dst", default=str(_DST))
    args = ap.parse_args()
    n = ingest(args.src, args.dst)
    print(f"[steel] 完成，输出到 {_DST}（{n} 张）")


if __name__ == "__main__":
    main()
