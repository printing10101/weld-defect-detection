"""P1-C 缺陷感知采样测试（dataset_builder.oversample_rare + build_dataset 集成）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from backend.training.dataset_builder import (
    DEFAULT_RARE_CLASSES,
    oversample_rare,
    rare_class_stats,
)


def _write_pair(root: Path, stem: str, classes: list[int]) -> tuple[Path, Path]:
    """写一张带标注的图（YOLO normalized，仅类别不同）。"""
    img_dir = root / "images"
    lbl_dir = root / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(stem)) % (2**31))
    img = rng.integers(40, 180, (64, 64), dtype=np.uint8)
    img_p = img_dir / f"{stem}.png"
    import cv2

    _, buf = cv2.imencode(".png", img)
    img_p.write_bytes(buf.tobytes())
    lbl_p = lbl_dir / f"{stem}.txt"
    lines = []
    for i, c in enumerate(classes):
        cx = 0.2 + 0.1 * i
        lines.append(f"{c} {cx:.4f} 0.5 0.1 0.1")
    lbl_p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return img_p, lbl_p


def _make_raw(root: Path) -> list[tuple[Path, Path]]:
    pairs = [
        _write_pair(root, "rare1", [4]),  # 裂纹
        _write_pair(root, "rare2", [2]),  # 未焊透
        _write_pair(root, "common1", [0]),  # 仅气孔
        _write_pair(root, "common2", [0, 0]),  # 仅气孔
    ]
    return pairs


class TestOversampleRare:
    def test_factor_expands_rare_only(self, tmp_path: Path):
        pairs = _make_raw(tmp_path)  # 真实落盘：rare1 裂纹 / rare2 未焊透 / common1-2 气孔
        out = oversample_rare(pairs, factor=2)
        stems = [p[0].stem for p in out]
        assert stems.count("rare1") == 2
        assert stems.count("rare2") == 2
        assert stems.count("common1") == 1
        assert stems.count("common2") == 1

    def test_factor_zero_returns_same(self):
        pairs = [(Path("a.png"), None), (Path("b.png"), None)]
        assert oversample_rare(pairs, factor=0) == pairs

    def test_no_rare_returns_same(self):
        pairs = [(Path("a.png"), None), (Path("b.png"), None)]
        assert oversample_rare(pairs, factor=3) == pairs

    def test_default_rare_classes(self):
        assert DEFAULT_RARE_CLASSES == frozenset({1, 2, 3, 4, 5})


class TestBuildDatasetIntegration:
    def _make_raw_dir(self, tmp: Path) -> Path:
        raw = tmp / "data" / "training" / "raw"
        for src in ("user", "synthetic"):
            _make_raw(raw / src)
        return raw

    def test_oversample_writes_unique_files(self, tmp_path: Path, monkeypatch):
        raw = self._make_raw_dir(tmp_path)
        # 让 _RAW_ROOT 指向临时 raw（monkeypatch 模块常量）
        import backend.training.dataset_builder as db

        monkeypatch.setattr(db, "_RAW_ROOT", raw)
        out = tmp_path / "out"
        yaml_path = db.build_dataset(out_root=out, rare_oversample=2, seed=1)
        assert yaml_path.exists()
        train_imgs = list((out / "train" / "images").iterdir())
        train_lbls = list((out / "train" / "labels").iterdir())
        # 4 张源图 × 2 源 = 8，train ≈ 6-7（含 os_ 副本）→ 至少存在 os 副本文件
        names = [p.name for p in train_imgs]
        assert any(n.startswith("os") for n in names), f"no oversampled files: {names}"
        assert len(train_imgs) == len(train_lbls)
        # 每个图像都有对应标签（stem 对齐）
        for img in train_imgs:
            assert (out / "train" / "labels" / f"{img.stem}.txt").exists()

    def test_default_no_oversample(self, tmp_path: Path, monkeypatch):
        raw = self._make_raw_dir(tmp_path)
        import backend.training.dataset_builder as db

        monkeypatch.setattr(db, "_RAW_ROOT", raw)
        out = tmp_path / "out2"
        db.build_dataset(out_root=out, rare_oversample=0, seed=1)
        names = [p.name for p in (out / "train" / "images").iterdir()]
        assert not any(n.startswith("os") for n in names)

    def test_rare_class_stats(self, tmp_path: Path):
        pairs = _make_raw(tmp_path)
        stats = rare_class_stats(pairs)
        assert stats[4] == 1
        assert stats[2] == 1
        assert stats[0] == 2

    def test_source_limits_cap_synthetic_dominance(self, tmp_path: Path, monkeypatch):
        """A 方案：外部合成源限流（防真实域漂移）——steel 全量灌入会淹没 user。"""
        import backend.training.dataset_builder as db

        raw = tmp_path / "raw"
        # 构造：真实 user 4 张 + 合成 steel 60 张（模拟 79% 淹没场景）
        for s in ("u1", "u2", "u3", "u4"):
            _write_pair(raw / "user", s, [0])
        for i in range(60):
            _write_pair(raw / "steel", f"s{i}", [0])
        monkeypatch.setattr(db, "_RAW_ROOT", raw)
        out = tmp_path / "out3"
        db.build_dataset(out_root=out, source_limits={"steel": 12}, seed=1)
        train = {p.name for p in (out / "train" / "images").iterdir()}
        # steel 被限到 12 张以内（且都是 s* 前缀）
        steel_in_train = [n for n in train if n.startswith("s")]
        assert 0 < len(steel_in_train) <= 12
        assert len([n for n in train if n.startswith("u")]) >= 2  # 真实域仍保留

    def test_clean_output_prevents_accumulation(self, tmp_path: Path, monkeypatch):
        """clean_output=True 时重复 build 不累积旧 split 文件（auto_v2 教训）。"""
        import backend.training.dataset_builder as db

        raw = tmp_path / "raw"
        _write_pair(raw / "user", "u1", [0])
        _write_pair(raw / "user", "u2", [0])
        monkeypatch.setattr(db, "_RAW_ROOT", raw)
        out = tmp_path / "out4"
        db.build_dataset(out_root=out, clean_output=True, seed=1)
        n1 = len(list((out / "train" / "images").iterdir()))
        # 再 build 一次（clean_output=True）：数量应一致，不叠加
        db.build_dataset(out_root=out, clean_output=True, seed=1)
        n2 = len(list((out / "train" / "images").iterdir()))
        assert n1 == n2
        assert n1 <= 2
