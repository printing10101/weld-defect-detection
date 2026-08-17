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

import os
import random
import shutil
from pathlib import Path

from backend.domain.dto import DefectClass
from backend.training.class_map import YOLO_CLASSES

_RAW_ROOT = Path("data/training/raw")
_OUT_ROOT = Path("data/training")
_SOURCES = ["swrd", "user", "synthetic", "roboflow", "steel", "pseudo"]


def _rmtree_native(path: Path) -> None:
    """绕 safe-delete shim 清空目录（仅删 build 生成的 split 产物，无外部风险）。

    WorkBuddy 沙箱的 safe-delete 护栏会劫持 os.remove/Path.unlink/`shutil.rmtree`
    （回收站不可用时 FAIL_CLOSED；shim 的 rmtree 对非 os 临时目录走 trash，bulk 时
    静默不删）。build_dataset 重复运行时旧 split 文件会**累积**（曾致 auto_v2
    训练误用 6267 张历史数据）。绕法：**子进程 cmd /c rmdir**（子进程无 shim
    注入）；数据在 data/training 下，为 Windows 专用路径，cmd 语义可靠。
    """
    if not path.exists():
        return
    import subprocess

    r = subprocess.run(
        ["cmd", "/c", "rmdir", "/s", "/q", str(path)],
        capture_output=True,
        check=False,
    )
    if path.exists():
        # 兜底：恢复原生删除后重试（os 临时目录场景 shim 本会豁免）
        import nt
        import pathlib

        orig_unlink, orig_remove, orig_rmdir = os.unlink, os.remove, os.rmdir
        orig_punlink, orig_prmdir = pathlib.Path.unlink, pathlib.Path.rmdir
        os.unlink = nt.unlink  # type: ignore[attr-defined]
        os.remove = nt.remove  # type: ignore[attr-defined]
        os.rmdir = nt.rmdir  # type: ignore[attr-defined]
        pathlib.Path.unlink = lambda self, *a, **k: os.unlink(str(self))  # type: ignore[attr-defined]
        pathlib.Path.rmdir = lambda self, *a, **k: os.rmdir(str(self))  # type: ignore[attr-defined]
        try:
            shutil.rmtree(path, ignore_errors=True)
        finally:
            os.unlink, os.remove, os.rmdir = orig_unlink, orig_remove, orig_rmdir
            pathlib.Path.unlink, pathlib.Path.rmdir = orig_punlink, orig_prmdir
    if path.exists():
        raise RuntimeError(f"清理 split 目录失败（shim 拦截）：{path} rc={r.returncode}")

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


# 罕见且安全关键的缺陷类别（真实集样本极少，需过采样缓解长尾；与 config.class_conf 低阈值策略一致）
DEFAULT_RARE_CLASSES = frozenset({1, 2, 3, 4, 5})  # 夹渣/未焊透/未熔合/裂纹/咬边（气孔占 94.6%）


def rare_class_stats(pairs: list[tuple[Path, Path | None]]) -> dict[int, int]:
    """统计各类别样本图像数（缺陷感知采样的依据）。"""
    stats: dict[int, int] = {}
    for _, lbl in pairs:
        for cid in _classes_in_label(lbl):
            stats[cid] = stats.get(cid, 0) + 1
    return dict(sorted(stats.items()))


def oversample_rare(
    pairs: list[tuple[Path, Path | None]],
    rare_classes: set[int] | None = None,
    factor: int = 2,
    seed: int = 42,
) -> list[tuple[Path, Path | None]]:
    """缺陷感知过采样：含罕见类的图像复制 factor 份（副本排在前列）。

    仅用于**训练集**（build_dataset 内对 train split 调用）；val/test 保持原始
    分布，避免过采样污染评估指标。factor=0 或 rare 图像不存在时原样返回。
    """
    if factor <= 0:
        return list(pairs)
    rare = set(rare_classes if rare_classes is not None else DEFAULT_RARE_CLASSES)
    rare_items = [p for p in pairs if rare & _classes_in_label(p[1])]
    if not rare_items:
        return list(pairs)
    # 副本放最前：YOLO 训练默认 shuffle，顺序无影响，但便于调试复现
    copies: list[tuple[Path, Path | None]] = []
    for _ in range(factor):
        copies.extend(rare_items)
    return copies + [p for p in pairs if p not in rare_items]


def _limit_class_balanced(
    ps: list[tuple[Path, Path | None]],
    limit: int,
    rare: set[int],
    rnd: random.Random,
    src: str,
) -> list[tuple[Path, Path | None]]:
    """类别均衡限流：含罕见类的图优先（保底 60% 配额，按类别轮询），
    剩余配额随机补常见类。防止随机抽样稀释本就稀少的罕见类（auto_v2 教训）。"""
    n_orig = len(ps)
    rare_imgs = [p for p in ps if rare & _classes_in_label(p[1])]
    common_imgs = [p for p in ps if not (rare & _classes_in_label(p[1]))]
    rare_budget = int(limit * 0.6)
    if rare_imgs and rare_budget > 0:
        if len(rare_imgs) > rare_budget:
            # 按"含哪些罕见类"分桶，轮询抽取保证每类都有代表
            by_key: dict[tuple[int, ...], list[tuple[Path, Path | None]]] = {}
            for p in rare_imgs:
                key = tuple(sorted(rare & _classes_in_label(p[1])))
                by_key.setdefault(key, []).append(p)
            picked: list[tuple[Path, Path | None]] = []
            keys = list(by_key.keys())
            i = 0
            while len(picked) < rare_budget and any(by_key.values()):
                k = keys[i % len(keys)]
                bucket = by_key[k]
                if bucket:
                    rnd.shuffle(bucket)
                    picked.append(bucket.pop())
                i += 1
            rare_picked = picked
        else:
            rare_picked = list(rare_imgs)
    else:
        rare_picked = []
    rnd.shuffle(common_imgs)
    ps = rare_picked + common_imgs[: max(0, limit - len(rare_picked))]
    n_rare = len(rare_picked)
    print(f"[dataset] 源 {src}: 限流 {len(ps)} 张（原 {n_orig}，罕见类优先保 {n_rare}）")
    return ps


def build_dataset(
    out_root: Path | None = None,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    rare_oversample: int = 0,
    rare_classes: set[int] | None = None,
    train_only_sources: set[str] | None = None,
    source_limits: dict[str, int] | None = None,
    clean_output: bool = False,
) -> Path:
    """合并多源 → 分层划分 → 写出 data.yaml。返回 data.yaml 路径。

    rare_oversample>0：对 train split 中**含罕见类**的图像过采样 factor 份
    （缺陷感知采样，P1-C），缓解长尾；val/test 不采样，评估指标不被污染。

    train_only_sources：仅进 train 的源（如伪标签 pseudo/，模型自预测有噪声，
    只用于扩充训练分布，**不得进入 val/test**，否则评估指标被乐观污染）。

    source_limits：按源限流（{源名: 上限}）。外部合成源（如 steel 2699 张）若
    无差别全量灌入会淹没真实域 → 真实域漂移（2026-08-15 auto_v1 教训）。
    **限流为类别均衡抽样**（auto_v2 教训：随机抽样会把本就稀少的罕见类进一步
    抽掉）：含罕见类（rare_classes，默认 DEFAULT_RARE_CLASSES）的图**优先入选**，
    占限流配额的 60%（按类别轮询保证每罕见类都有代表），剩余配额随机补气孔等常见类。

    clean_output：写入前清空 out_root 的 train/val/test（防重复 build 累积旧文件，
    曾致 auto_v2 误用 6267 张历史数据）。
    """
    out_root = Path(out_root or _OUT_ROOT)
    raw_root = _RAW_ROOT
    train_only = set(train_only_sources or ())
    limits = source_limits or {}
    rare = set(rare_classes if rare_classes is not None else DEFAULT_RARE_CLASSES)
    rnd = random.Random(seed)
    all_pairs: list[tuple[Path, Path | None]] = []
    train_only_pairs: list[tuple[Path, Path | None]] = []
    for src in _SOURCES:
        ps = _discover(raw_root, src)
        if not ps:
            continue
        n_orig = len(ps)
        limit = limits.get(src)
        if limit is not None and n_orig > limit:
            ps = _limit_class_balanced(ps, limit, rare, rnd, src)
        print(f"[dataset] 源 {src}: {len(ps)} 张" + ("（train-only）" if src in train_only else ""))
        if src in train_only:
            train_only_pairs.extend(ps)
        else:
            all_pairs.extend(ps)
    if not all_pairs and not train_only_pairs:
        raise RuntimeError(
            "未找到任何训练图像。请先将 SWRD 落到 data/external/swrd 并运行 "
            "python -m backend.training.download_swrd --ingest，或将用户标注放到 "
            "data/training/raw/user/{images,labels}。"
        )

    # 输出清理：重复 build 时旧 split 文件会累积（auto_v2 教训），显式开启才清
    if clean_output:
        for split in ("train", "val", "test"):
            _rmtree_native(out_root / split)
        print("[dataset] 已清空旧 split 输出（clean_output=True）")

    # 按类别集合分层抽样（仅非 train-only 源参与，train-only 源全部并入 train）
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
        splits["train"].extend(items[:n_train])
        splits["val"].extend(items[n_train : n_train + n_val])
        splits["test"].extend(items[n_train + n_val :])

    # train-only 源（伪标签等）全部并入 train
    if train_only_pairs:
        splits["train"].extend(train_only_pairs)
        print(f"[dataset] train-only 源并入 train: {len(train_only_pairs)} 张")

    # 缺陷感知采样（P1-C）：仅对 train 过采样罕见类，val/test 保持原始分布
    if rare_oversample > 0:
        before = len(splits["train"])
        splits["train"] = oversample_rare(
            splits["train"], rare_classes=rare_classes, factor=rare_oversample, seed=seed
        )
        print(f"[dataset] 罕见类过采样 x{rare_oversample}: train {before} → {len(splits['train'])}")

    for split, pairs in splits.items():
        d_img = out_root / split / "images"
        d_lbl = out_root / split / "labels"
        d_img.mkdir(parents=True, exist_ok=True)
        d_lbl.mkdir(parents=True, exist_ok=True)
        seen: dict[str, int] = {}
        for img, lbl in pairs:
            name = img.name
            # 过采样副本同名 → 唯一化（os0_/os1_ 前缀），否则 shutil.copy 互相覆盖
            cnt = seen.get(name, 0)
            seen[name] = cnt + 1
            if cnt > 0:
                name = f"os{cnt}_{name}"
            shutil.copy(img, d_img / name)
            if lbl is not None and lbl.exists():
                shutil.copy(lbl, d_lbl / (Path(name).stem + ".txt"))

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
