"""训练/测试集互斥校验（DB50/T 1807-2025 ）。

标准要求：测试数据集中不应包含训练数据集中的数据。

本实现口径（较标准收紧）：
- 字节级 md5 完全相同 → 判重叠（直接违规）；
- 感知哈希（dHash 64bit，汉明距离≤阈值默认 4）相似 → 判"疑似重复"，
  默认同样判失败（可配置放行，但必须留清单供审计）；
- 提供 CLI（backend/training/check_dataset_disjoint.py）与 build_dataset
  自动挂接（划分写盘后强制校验，重叠即抛异常阻断）。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass
class OverlapReport:
    """互斥校验报告。"""

    n_train: int = 0
    n_test: int = 0
    exact: list[str] = field(default_factory=list)  # 字节完全相同的文件对
    perceptual: list[dict[str, object]] = field(default_factory=list)  # 疑似重复（含距离）
    phash_hamming: int = 4

    @property
    def passed(self) -> bool:
        return not self.exact and not self.perceptual

    def to_dict(self) -> dict:
        return {
            "n_train": self.n_train,
            "n_test": self.n_test,
            "exact": self.exact,
            "perceptual": self.perceptual,
            "phash_hamming": self.phash_hamming,
            "passed": self.passed,
        }


def _image_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.suffix.lower() in _IMG_EXTS)


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(path: Path, hash_size: int = 8) -> int | None:
    """dHash 感知哈希（差值哈希）：灰度缩 9×8，相邻像素横向差值 → 64bit。

    图像解码失败返回 None（跳过感知判定，不阻断——md5 判定仍有效）。
    """
    data = np.fromfile(str(path), dtype=np.uint8)  # 字节读，规避中文路径
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    resized = cv2.resize(img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = diff.flatten()
    return int(np.packbits(bits).tobytes().hex(), 16)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def find_overlaps(
    train_dir: str | Path,
    test_dir: str | Path,
    *,
    phash_hamming: int = 4,
) -> OverlapReport:
    """校验两个图像目录的互斥性。"""
    train_dir, test_dir = Path(train_dir), Path(test_dir)
    train_files = _image_files(train_dir)
    test_files = _image_files(test_dir)
    report = OverlapReport(
        n_train=len(train_files), n_test=len(test_files), phash_hamming=phash_hamming
    )
    train_md5 = {f.name: md5_of(f) for f in train_files}
    train_hash: dict[str, int] = {}
    for f in train_files:
        h = dhash(f)
        if h is not None:
            train_hash[f.name] = h

    test_md5: dict[str, str] = {}
    for f in test_files:
        m = md5_of(f)
        test_md5[f.name] = m
        dup = [name for name, tm in train_md5.items() if tm == m]
        if dup:
            report.exact.append(f"test/{f.name} == train/{dup[0]}")
    if report.exact:
        return report  # 字节已重叠，感知判定无意义

    for f in test_files:
        h = dhash(f)
        if h is None:
            continue
        for name, th in train_hash.items():
            d = hamming(h, th)
            if d <= phash_hamming:
                report.perceptual.append({"test": f.name, "train": name, "hamming": d})
                break
    return report


def assert_disjoint(
    train_dir: str | Path,
    test_dir: str | Path,
    *,
    phash_hamming: int = 4,
    allow_perceptual: bool = False,
) -> OverlapReport:
    """强校验：重叠（或疑似重复未放行）→ raise RuntimeError 阻断。"""
    report = find_overlaps(train_dir, test_dir, phash_hamming=phash_hamming)
    if report.exact or (report.perceptual and not allow_perceptual):
        details = "; ".join(report.exact[:3])
        percept = "; ".join(str(p) for p in report.perceptual[:3])
        raise RuntimeError(
            f"数据互斥校验失败（DB50/T 1807-2025 §8.3.1）：train={train_dir} test={test_dir} "
            f"字节重叠: [{details}] 疑似重复: [{percept}]"
        )
    return report


# 双侧样本量都达到该值才强制互斥（重叠即抛异常）；低于该值的小样本合成集
# （单测夹具、冒烟数据）文件必然来回复用，只告警不阻断——真实数据集治理
# 由 CLI（check_dataset_disjoint）与本函数共同兜底。
MIN_SPLIT_FOR_ENFORCE = 10


def enforce_split_disjoint(*dirs: Path) -> list[OverlapReport]:
    """build_dataset 划分后的互斥门禁：两两 split 逐一校验。

    - 双侧 ≥ MIN_SPLIT_FOR_ENFORCE 张：重叠（字节或感知）→ RuntimeError 阻断；
    - 小样本：仅打印告警（真实治理以 CLI 全量校验为准）。
    """
    from itertools import combinations

    reports: list[OverlapReport] = []
    for a, b in combinations(dirs, 2):
        report = find_overlaps(a, b)
        reports.append(report)
        if report.passed:
            continue
        msg = (
            f"split 互斥校验失败（§8.3.1）: {a} vs {b} 字节重叠={len(report.exact)} "
            f"疑似重复={len(report.perceptual)}"
        )
        if report.n_train >= MIN_SPLIT_FOR_ENFORCE and report.n_test >= MIN_SPLIT_FOR_ENFORCE:
            raise RuntimeError(msg)
        print(f"[dataset] ⚠ {msg}（小样本集仅告警）")
    return reports
