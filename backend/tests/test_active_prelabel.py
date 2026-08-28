"""P0-G 主动预标注 + 不确定性采样测试（backend/training/active_prelabel）。

用测试桩 detector（不依赖 ONNX/训练权重）验证：
- 未标注图被推理并写入预标注；
- 用户已标注的图**不被覆盖**；
- 队列按价值降序、字段完整；
- export_top 导出到训练池。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from backend.domain.dto import BBox, DefectClass, Detection
from backend.training.active_prelabel import (
    _family_of,
    _unlabeled_images,
    active_prelabel,
    export_top,
)


class StubDetector:
    """固定返回两条检测的测试桩：高不确定裂纹 + 低置信气孔。"""

    def __init__(self) -> None:
        self.calls = 0

    def infer(self, image, conf=0.3, iou=0.5, class_conf=None) -> list[Detection]:
        self.calls += 1
        return [
            Detection(
                id="d0",
                bbox=BBox(10, 10, 20, 20),
                class_id=DefectClass.CRACK,
                score=0.9,
                uncertainty=0.8,  # 高不确定（安全关键基线）
            ),
            Detection(
                id="d1",
                bbox=BBox(50, 50, 10, 10),
                class_id=DefectClass.POROSITY,
                score=0.2,
                uncertainty=0.15,
            ),
        ]


class EmptyDetector:
    """无检出桩：验证 no_detection 标记。"""

    def infer(self, image, conf=0.3, iou=0.5, class_conf=None) -> list[Detection]:
        return []


def _make_image(path: Path, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    img = rng.integers(50, 150, (60, 80), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    path.write_bytes(buf.tobytes())


def _make_real_root(tmp: Path) -> Path:
    real = tmp / "real_label"
    (real / "images").mkdir(parents=True)
    (real / "labels").mkdir(parents=True)
    (real / "prelabels").mkdir(parents=True)
    for i, name in enumerate(("PG103-1-1", "PG103-2-3", "PL118-12-1")):
        _make_image(real / "images" / f"{name}.jpg", seed=i)
    # PG103-1-1 已有用户正式标注（空标注也视为已标注）
    (real / "labels" / "PG103-1-1.txt").write_text("", encoding="utf-8")
    return real


class TestActivePrelabel:
    def test_prelabels_written_and_user_labels_kept(self, tmp_path: Path):
        real = _make_real_root(tmp_path)
        det = StubDetector()
        queue = active_prelabel(real, det, conf=0.3, iou=0.5, class_conf={0: 0.3, 4: 0.05})
        # 3 张图，1 张已标注 → 只处理 2 张
        assert queue["total"] == 2
        # 预标注写入（PG103-2-3 / PL118-12-1）
        assert (real / "prelabels" / "PG103-2-3.txt").exists()
        assert (real / "prelabels" / "PL118-12-1.txt").exists()
        # 用户已标注图不写预标注、不动 labels
        assert not (real / "prelabels" / "PG103-1-1.txt").exists()
        assert (real / "labels" / "PG103-1-1.txt").read_text(encoding="utf-8") == ""
        # 预标注内容：两条 Detection → 两行合法 YOLO
        lines = (real / "prelabels" / "PG103-2-3.txt").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        for ln in lines:
            p = ln.split()
            assert len(p) == 5
            assert 0 <= int(p[0]) <= 5
            assert all(0.0 <= float(v) <= 1.0 for v in p[1:])

    def test_queue_sorted_by_value_and_fields(self, tmp_path: Path):
        real = _make_real_root(tmp_path)
        queue = active_prelabel(real, StubDetector(), conf=0.3, iou=0.5)
        entries = queue["queue"]
        values = [e["max_value"] for e in entries]
        assert values == sorted(values, reverse=True)
        # 裂纹（安全关键）价值 ≥ 0.5（safety_base），气孔低置信近边界也 > 0
        assert all(e["max_value"] >= 0.5 for e in entries)
        for e in entries:
            assert {
                "stem",
                "family",
                "n_detections",
                "max_uncertainty",
                "max_value",
                "classes",
            } <= set(e)
            assert e["n_detections"] == 2
        # active_queue.json 落盘
        assert (real / "active_queue.json").exists()

    def test_no_detection_marked(self, tmp_path: Path):
        real = _make_real_root(tmp_path)
        queue = active_prelabel(real, EmptyDetector(), conf=0.3, iou=0.5)
        assert queue["total"] == 2
        assert all(e["n_detections"] == 0 and e["no_detection"] for e in queue["queue"])
        # 空预标注文件已写（供人工确认"合格/漏检"）
        assert (real / "prelabels" / "PG103-2-3.txt").read_text(encoding="utf-8") == ""

    def test_export_top(self, tmp_path: Path):
        real = _make_real_root(tmp_path)
        det = StubDetector()
        queue = active_prelabel(real, det, conf=0.3, iou=0.5)
        pool = tmp_path / "pool"
        n = export_top(real, queue, top_k=1, pool_dir=pool, detector=det, conf=0.3, iou=0.5)
        assert n == 1
        files = list(pool.glob("*.txt"))
        assert len(files) == 1
        assert files[0].read_text(encoding="utf-8").splitlines()  # 非空标注


class TestHelpers:
    def test_family_of(self):
        assert _family_of("PG103-1-1") == "PG103"
        assert _family_of("PL118-12-1") == "PL118"
        assert _family_of("foo-1") == "unknown"

    def test_unlabeled_images_excludes_labeled(self, tmp_path: Path):
        real = _make_real_root(tmp_path)
        unlabeled = _unlabeled_images(real)
        stems = {p.stem for p in unlabeled}
        assert stems == {"PG103-2-3", "PL118-12-1"}
