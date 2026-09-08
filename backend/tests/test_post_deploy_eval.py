"""部署后评估闭环的纯函数单测（不依赖权重/推理）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.training.post_deploy_eval import (
    domain_label,
    load_yolo_labels,
    model_id_of,
)


class TestModelIdOf:
    def test_matches_registry_semantics(self, tmp_path: Path):
        p = tmp_path / "best.onnx"
        p.write_bytes(b"fake-onnx-bytes")
        mid = model_id_of(p)
        assert mid.startswith("best::")
        assert len(mid.split("::")[1]) == 12
        # 同内容同哈希（stem 可不同），异内容异 id
        q = tmp_path / "other.onnx"
        q.write_bytes(b"fake-onnx-bytes")
        assert model_id_of(q).split("::")[1] == mid.split("::")[1]
        q.write_bytes(b"different-bytes")
        assert model_id_of(q).split("::")[1] != mid.split("::")[1]


class TestDomainLabel:
    def test_real_path_labeled_real(self, tmp_path: Path):
        assert domain_label(Path("data/real_label/images")) == "real"
        assert domain_label(Path("d:/data/REAL_SET")) == "real"

    def test_other_paths_labeled_synthetic(self, tmp_path: Path):
        assert domain_label(Path("data/training/test")) == "synthetic"


class TestLoadYoloLabels:
    def test_normalizes_to_pixels(self, tmp_path: Path):
        p = tmp_path / "a.txt"
        p.write_text("0 0.5 0.5 0.25 0.5\n4 0.1 0.2 0.1 0.1\n", encoding="utf-8")
        gts = load_yolo_labels(p, 200, 100)
        assert len(gts) == 2
        assert gts[0]["class_id"] == 0
        # cx=0.5, w=0.25 → x1=(0.5-0.125)*200=75；cy=0.5, h=0.5 → y1=25
        assert gts[0]["bbox"] == pytest.approx([75.0, 25.0, 50.0, 50.0])
        assert gts[1]["class_id"] == 4

    def test_skips_malformed_lines(self, tmp_path: Path):
        p = tmp_path / "b.txt"
        p.write_text("0 0.5 0.5 0.25\nbad line\n\n0 0.5 0.5 0.25 0.5 0 0\n", encoding="utf-8")
        gts = load_yolo_labels(p, 100, 100)
        assert len(gts) == 1  # 4 字段行与坏行跳过；6 字段行取前 5 项
