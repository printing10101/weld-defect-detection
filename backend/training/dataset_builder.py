"""M4b 训练数据集装配（合并多源 → YOLO 分层划分 → data.yaml）。

数据源（落盘于 data/training/raw/）：
- swrd/      : SWRD 转换结果（swrd_converter，CC BY 4.0，3675 张；本机 115GB 装不下时改用 roboflow）
- user/      : 用户 165 张 定检 标注（Label Studio 导出，域适应集，最重要）
- synthetic/ : copy-paste / Blender 合成增强（可选，见 augment.generate_copy_paste）
- roboflow/  : Roboflow X 光焊缝集（Danila Public Domain / XrayWeld CC BY 4.0，绕开 SWRD 体积死穴）

输出 data/training/{train,val,test}/{images,labels} + data/training/data.yaml
（nc=6，names 与 DefectClass 严格一致）。

划分：按"图像所含类别集合"分层抽样，保证每类在 train/val/test 均有代表；
固定 seed 可复现（§15.6 Golden Set 隔离）。
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

from backend.domain.dto import DefectClass
from backend.training.class_map import YOLO_CLASSES

_RAW_ROOT = Path("data/training/raw")
_OUT_ROOT = Path("data/training")
_SOURCES = ["swrd", "user", "synthetic", "roboflow"]

# DefectClass 枚举名 → 中文名（YOLO names，供训练日志/可视化）
_CLASS_ZH = {
    "POROSITY": "气孔",
    "SLAG": "夹渣",
    "INCOMPLETE_PENETRATION": "未焊透",
    "LACK_OF_FUSION": "未熔合",
    "CRACK": "裂纹",
    "UNDERCUT": "咬边",
}


def _discover(raw_root: Path, source: str) -> list[tuple[Path, Path | None]]:
    img_dir = raw_root / source / "images"
    lbl_dir = raw_root / source / "labels"
    if not img_dir.exists():
        return []
    pairs: list[tuple[Path, Path | None]] = []
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
            continue
        lbl = lbl_dir / (img.stem + ".txt")
        pairs.append((img, lbl if lbl.exists() else None))
    return pairs


def _classes_in_label(lbl: Path | None) -> frozenset[int]:
    if lbl is None or not lbl.exists() or lbl.stat().st_size == 0:
        return frozenset()  # 无缺陷 / 背景
    ids: set[int] = set()
    for line in lbl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ids.add(int(line.split()[0]))
        except (ValueError, IndexError):
            continue
    return frozenset(ids)


def build_dataset(
    out_root: Path | None = None,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> Path:
    """合并多源 → 分层划分 → 写出 data.yaml。返回 data.yaml 路径。"""
    out_root = Path(out_root or _OUT_ROOT)
    raw_root = _RAW_ROOT
    all_pairs: list[tuple[Path, Path | None]] = []
    for src in _SOURCES:
        ps = _discover(raw_root, src)
        if ps:
            print(f"[dataset] 源 {src}: {len(ps)} 张")
        all_pairs.extend(ps)
    if not all_pairs:
        raise RuntimeError(
            "未找到任何训练图像。请先将 SWRD 落到 data/external/swrd 并运行 "
            "python -m backend.training.download_swrd --ingest，或将用户标注放到 "
            "data/training/raw/user/{images,labels}。"
        )

    # 按类别集合分层抽样
    strata: dict[frozenset[int], list[tuple[Path, Path | None]]] = {}
    for p in all_pairs:
        key = _classes_in_label(p[1])
        strata.setdefault(key, []).append(p)

    rnd = random.Random(seed)
    splits: dict[str, list[tuple[Path, Path | None]]] = {"train": [], "val": [], "test": []}
    for key, items in strata.items():
        rnd.shuffle(items)
        n = len(items)
        n_train = round(n * ratios[0])
        n_val = round(n * ratios[1])
        if n >= 3:
            n_train = max(1, min(n_train, n - 2))
            n_val = max(0, min(n_val, n - n_train))
        else:
            n_train = max(1, n - 1)  # 极小层：保 train，其余给 test
            n_val = 0
        n - n_train - n_val
        splits["train"].extend(items[:n_train])
        splits["val"].extend(items[n_train : n_train + n_val])
        splits["test"].extend(items[n_train + n_val :])

    for split, pairs in splits.items():
        d_img = out_root / split / "images"
        d_lbl = out_root / split / "labels"
        d_img.mkdir(parents=True, exist_ok=True)
        d_lbl.mkdir(parents=True, exist_ok=True)
        for img, lbl in pairs:
            shutil.copy(img, d_img / img.name)
            if lbl is not None and lbl.exists():
                shutil.copy(lbl, d_lbl / (img.stem + ".txt"))

    nc = len(YOLO_CLASSES)
    assert nc == len(DefectClass), "YOLO_CLASSES 与 DefectClass 数量不一致"
    names = [_CLASS_ZH.get(c, c) for c in YOLO_CLASSES]
    yaml_text = (
        f"# 自动生成（dataset_builder.build_dataset）。nc 与 DefectClass 严格一致（ADR-010）。\n"
        f"path: {out_root.resolve()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"test: test/images\n"
        f"nc: {nc}\n"
        f"names: {names}\n"
    )
    (out_root / "data.yaml").write_text(yaml_text, encoding="utf-8")
    print(
        f"[dataset] 划分完成 train={len(splits['train'])} val={len(splits['val'])} "
        f"test={len(splits['test'])} → {out_root / 'data.yaml'}"
    )
    return out_root / "data.yaml"


def ensure_dataset(out_root: Path | None = None, **kw) -> Path:
    """data.yaml 已存在且 train 非空则复用，否则重建。"""
    out_root = Path(out_root or _OUT_ROOT)
    data_yaml = out_root / "data.yaml"
    train_imgs = out_root / "train" / "images"
    if data_yaml.exists() and train_imgs.exists() and any(train_imgs.iterdir()):
        return data_yaml
    return build_dataset(out_root, **kw)


if __name__ == "__main__":
    print(build_dataset())
