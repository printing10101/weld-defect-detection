"""三人标注一致性（§8.4.2）+ 数据互斥校验（§8.3.1）单测。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.domain.labeling.consensus import (
    LabelBox,
    arbitrate,
    resolve_consensus,
)
from backend.domain.labeling.dataset_guard import (
    assert_disjoint,
    find_overlaps,
)


def _box(a: str, cid: int, bbox) -> LabelBox:
    return LabelBox(annotator=a, class_id=cid, bbox=tuple(bbox))


# ---------------------------------------------------------------------------
# 三人一致性
# ---------------------------------------------------------------------------


def test_consensus_accepts_union():
    # 三人类型一致且两两 IOU≥0.6 → 接受，取三人外接并集框
    anns = [
        _box("A", 4, [10, 10, 20, 20]),
        _box("B", 4, [12, 10, 20, 20]),
        _box("C", 4, [11, 12, 20, 20]),
    ]
    res = resolve_consensus(anns, threshold=0.5)
    assert len(res.accepted) == 1 and res.discarded == []
    acc = res.accepted[0]
    assert acc["class_id"] == 4
    assert acc["bbox"] == [10, 10, 22, 22]  # 外接并集
    assert set(acc["annotators"]) == {"A", "B", "C"}


def test_consensus_type_mismatch_discards():
    # C 标成别的类型 → 该缺陷作废，三人框全部不入标注样本
    anns = [
        _box("A", 4, [10, 10, 20, 20]),
        _box("B", 4, [12, 10, 20, 20]),
        _box("C", 0, [11, 12, 20, 20]),
    ]
    res = resolve_consensus(anns, threshold=0.5)
    assert res.accepted == []
    assert len(res.discarded) == 3
    assert all("未见同类型" in d["reason"] for d in res.discarded)


def test_consensus_iou_below_threshold_discards():
    # A-B 重叠 0.9 但 B-C 仅 0.55（阈值 0.6）→ 两两判据不通过（严于三方模糊解释）
    anns = [
        _box("A", 4, [0, 0, 10, 10]),
        _box("B", 4, [1, 0, 10, 10]),
        _box("C", 4, [11, 0, 10, 10]),  # 与 B IOU = 1/(21-1)... 计算：交 0? 调整
    ]
    # 构造 B-C IOU = 0.55：B=[1,0,10,10] 与 C=[6,0,10,10] 交=5, 并=15 → 0.333
    anns[2] = _box("C", 4, [5, 0, 10, 10])  # B-C IOU = 6/14 ≈ 0.43 < 0.6
    res = resolve_consensus(anns, threshold=0.6)
    assert res.accepted == []
    assert len(res.discarded) == 3


def test_consensus_missing_annotator_discards():
    # 只有两人画了该缺陷 → 三方齐备才接受，缺者作废
    anns = [
        _box("A", 1, [0, 0, 10, 10]),
        _box("B", 1, [0, 0, 10, 10]),
    ]
    res = resolve_consensus(anns, threshold=0.5)
    assert res.accepted == [] and len(res.discarded) == 2


def test_consensus_two_defects_mixed():
    # 一个三人一致（气孔）+ 一个只有 A 画（咬边）→ 1 接受 + 1 作废
    anns = [
        _box("A", 0, [0, 0, 8, 8]),
        _box("B", 0, [1, 1, 8, 8]),
        _box("C", 0, [0, 1, 8, 8]),
        _box("A", 5, [50, 50, 10, 4]),
    ]
    res = resolve_consensus(anns, threshold=0.5)
    assert len(res.accepted) == 1 and res.accepted[0]["class_id"] == 0
    assert len(res.discarded) == 1 and res.discarded[0]["class_id"] == 5
    # 一致率 = 3×1 / 4
    assert res.agreement_rate == 0.75


def test_consensus_unknown_annotator_rejected():
    with pytest.raises(ValueError, match="未知标注员"):
        resolve_consensus([_box("D", 0, [0, 0, 1, 1])])


def test_arbitrate_recovers_discarded():
    anns = [
        _box("A", 4, [0, 0, 10, 10]),
        _box("B", 4, [1, 0, 10, 10]),
    ]
    res = resolve_consensus(anns, threshold=0.5)
    sink: list[list[dict]] = []
    out = arbitrate(
        res,
        [{"annotator": "A", "index": 0, "class_id": 4}],
        arbitrator="组长",
        reason="C 缺席，组长确认保留",
        label_sink=sink.append,
    )
    assert out["recovered"][0]["source"] == "arbitration"
    assert out["recovered"][0]["arbitrator"] == "组长"
    assert sink and sink[0][0]["class_id"] == 4


def test_arbitrate_requires_reason():
    res = resolve_consensus([_box("A", 0, [0, 0, 1, 1])])
    with pytest.raises(ValueError, match="理由"):
        arbitrate(res, [{"annotator": "A", "index": 0}], arbitrator="x", reason=" ")


def test_arbitrate_invalid_target():
    res = resolve_consensus([_box("A", 0, [0, 0, 1, 1])])
    with pytest.raises(ValueError, match="不存在"):
        arbitrate(res, [{"annotator": "A", "index": 5}], arbitrator="x", reason="r")


# ---------------------------------------------------------------------------
# 数据互斥校验
# ---------------------------------------------------------------------------


@pytest.fixture()
def img_dirs(tmp_path: Path) -> tuple[Path, Path]:
    import cv2

    tr, te = tmp_path / "train", tmp_path / "test"
    (tr / "images").mkdir(parents=True)
    (te / "images").mkdir(parents=True)
    rng = np.random.default_rng(2)
    for i in range(3):
        img = rng.integers(0, 255, (64, 64), dtype=np.uint8)
        cv2.imwrite(str(tr / "images" / f"t{i}.png"), img)
    for i in range(2):
        img = rng.integers(0, 255, (64, 64), dtype=np.uint8)  # 不同内容（非平移/微扰）
        cv2.imwrite(str(te / "images" / f"s{i}.png"), img)
    return tr / "images", te / "images"


def test_disjoint_ok(img_dirs):
    tr, te = img_dirs
    report = find_overlaps(tr, te)
    assert report.passed and report.n_train == 3 and report.n_test == 2


def test_disjoint_exact_duplicate(img_dirs):
    import shutil

    tr, te = img_dirs
    shutil.copy(tr / "t0.png", te / "copy.png")  # 字节完全相同
    report = find_overlaps(tr, te)
    assert report.exact and not report.passed
    with pytest.raises(RuntimeError, match="互斥校验失败"):
        assert_disjoint(tr, te)


def test_disjoint_perceptual_duplicate(tmp_path):
    import cv2

    tr, te = tmp_path / "train", tmp_path / "test"
    (tr / "images").mkdir(parents=True)
    (te / "images").mkdir(parents=True)
    # 平滑结构图（dHash 对纯噪声图极其敏感，须用平滑图模拟真实底片）
    yy, xx = np.mgrid[0:128, 0:128]
    smooth = ((xx * 1.7 + yy * 0.9) % 256).astype(np.uint8)
    cv2.imwrite(str(tr / "images" / "a.png"), smooth)
    cv2.imwrite(str(te / "images" / "near.png"), cv2.GaussianBlur(smooth, (3, 3), 0.8))
    report = find_overlaps(tr / "images", te / "images")
    assert report.perceptual and not report.passed
    with pytest.raises(RuntimeError, match="互斥校验失败"):
        assert_disjoint(tr / "images", te / "images")
    # 放行疑似（仍留清单）：仅字节重复才阻断
    report2 = assert_disjoint(tr / "images", te / "images", allow_perceptual=True)
    assert report2.perceptual


def test_disjoint_empty_dirs_pass(tmp_path):
    d1 = tmp_path / "a"
    d2 = tmp_path / "b"
    d1.mkdir()
    d2.mkdir()
    assert find_overlaps(d1, d2).passed
