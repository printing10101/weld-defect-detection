"""伪标签自动晋升 + train-only 源测试（auto_pseudo_label + dataset_builder）。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from backend.domain.dto import BBox, DefectClass, Detection
from backend.training.auto_pseudo_label import promote_high_conf
from backend.training.dataset_builder import _discover


class ConfDetector:
    """测试桩：按输入返回固定置信检出（高置信气孔 + 低置信裂纹）。"""

    def __init__(self) -> None:
        self.calls = 0

    def infer(self, image, conf=0.3, iou=0.5, class_conf=None) -> list[Detection]:
        self.calls += 1
        return [
            Detection(
                id="hi",
                bbox=BBox(10, 10, 20, 20),
                class_id=DefectClass.POROSITY,
                score=0.95,
                uncertainty=0.1,  # 高置信低不确定 → 采纳
            ),
            Detection(
                id="lo",
                bbox=BBox(50, 50, 20, 20),
                class_id=DefectClass.CRACK,
                score=0.3,
                uncertainty=0.7,  # 低置信高不确定 → 拒绝（安全关键更须人工）
            ),
        ]


def _make_real_root(tmp: Path) -> Path:
    real = tmp / "real_label"
    (real / "images").mkdir(parents=True)
    (real / "labels").mkdir(parents=True)
    for i, name in enumerate(("PG103-1-1", "PG103-2-3", "PL118-12-1")):
        img = np.full((60, 80), 100 + i, dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        (real / "images" / f"{name}.jpg").write_bytes(buf.tobytes())
    # PG103-1-1 已有用户真实标注（空）→ 不可覆盖
    (real / "labels" / "PG103-1-1.txt").write_text("", encoding="utf-8")
    return real


class TestPromoteHighConf:
    def test_promotes_high_conf_only(self, tmp_path: Path):
        real = _make_real_root(tmp_path)
        out = tmp_path / "pseudo"
        stats = promote_high_conf(real, out, ConfDetector(), conf=0.3, iou=0.5)
        # 3 张图：1 张已标注跳过，2 张推理；每张只有 1 条采纳框（高置信气孔）
        assert stats["skipped_existing"] == 1
        assert stats["promoted_images"] == 2
        assert stats["promoted_boxes"] == 2
        assert stats["filtered_images"] == 0
        imgs = sorted((out / "images").glob("*.jpg"))
        lbls = sorted((out / "labels").glob("*.txt"))
        assert len(imgs) == 2 and len(lbls) == 2
        # 只写采纳框（每行 5 列，类别=气孔 0，不包含低置信裂纹）
        for lbl in lbls:
            lines = [ln for ln in lbl.read_text(encoding="utf-8").splitlines() if ln.strip()]
            assert lines and all(int(ln.split()[0]) == 0 for ln in lines)

    def test_no_promote_when_all_low_conf(self, tmp_path: Path):
        real = _make_real_root(tmp_path)

        class LowConf:
            def infer(self, image, conf=0.3, iou=0.5, class_conf=None):
                return [
                    Detection(
                        id="x",
                        bbox=BBox(0, 0, 10, 10),
                        class_id=DefectClass.POROSITY,
                        score=0.2,
                        uncertainty=0.6,
                    )
                ]

        out = tmp_path / "pseudo2"
        stats = promote_high_conf(real, out, LowConf(), conf=0.3, iou=0.5)
        assert stats["promoted_images"] == 0
        assert not (out / "images").exists() or not any((out / "images").iterdir())


def _seed_raw(tmp: Path, src: str, stems: list[str]) -> None:
    root = tmp / "raw" / src
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    for s in stems:
        img = np.full((48, 48), 120, dtype=np.uint8)
        _, buf = cv2.imencode(".png", img)
        (root / "images" / f"{s}.png").write_bytes(buf.tobytes())
        (root / "labels" / f"{s}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")


class TestTrainOnlySources:
    def test_pseudo_source_only_in_train(self, tmp_path: Path, monkeypatch):
        import backend.training.dataset_builder as db

        raw = tmp_path / "raw"
        _seed_raw(tmp_path, "user", ["u1", "u2", "u3", "u4"])
        _seed_raw(tmp_path, "pseudo", ["p1", "p2", "p3", "p4", "p5", "p6"])
        monkeypatch.setattr(db, "_RAW_ROOT", raw)
        out = tmp_path / "out"
        db.build_dataset(out_root=out, train_only_sources={"pseudo"}, seed=1)
        train = {p.name for p in (out / "train" / "images").iterdir()}
        val = {p.name for p in (out / "val" / "images").iterdir()}
        test = {p.name for p in (out / "test" / "images").iterdir()}
        # 伪标签全部进 train
        assert {"p1.png", "p2.png", "p3.png", "p4.png", "p5.png", "p6.png"} <= train
        # 伪标签绝不进 val/test
        assert not ({"p1.png", "p2.png", "p3.png", "p4.png", "p5.png", "p6.png"} & (val | test))

    def test_discover_finds_pseudo(self, tmp_path: Path):
        _seed_raw(tmp_path, "pseudo", ["p1"])
        pairs = _discover(tmp_path / "raw", "pseudo")
        assert len(pairs) == 1
