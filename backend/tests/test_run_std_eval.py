"""run_std_eval CLI 单测：参数约定 / YOLO 标签加载 / 无缺陷集污染拒绝。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.evaluation.run_std_eval import _load_yolo_labels, main


def _image(path: Path, w: int = 100, h: int = 50) -> None:
    img = np.full((h, w), 128, np.uint8)
    assert cv2.imwrite(str(path), img)


@pytest.fixture()
def eval_tree(tmp_path: Path) -> dict[str, Path]:
    """缺陷集 2 张（GT 各 1 个气孔）+ 无缺陷集 1 张；预测 = 真值全对。"""
    root = tmp_path / "set"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    (root / "preds").mkdir()
    clean = tmp_path / "clean"
    (clean / "images").mkdir(parents=True)
    (clean / "labels").mkdir()
    for stem in ("a", "b"):
        _image(root / "images" / f"{stem}.png")
        # 归一化 GT：类 0（气孔），中心 (0.5, 0.5)，框 10x10 像素
        (root / "labels" / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.2\n", encoding="utf-8")
        (root / "preds" / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.2\n", encoding="utf-8")
    _image(clean / "images" / "c.png")
    return {"root": root, "clean": clean, "out": tmp_path / "out.json"}


def _argv(tree: dict[str, Path], *extra: str) -> list[str]:
    return [
        "run_std_eval",
        "--img-dir",
        str(tree["root"] / "images"),
        "--label-dir",
        str(tree["root"] / "labels"),
        "--pred-dir",
        str(tree["root"] / "preds"),
        "--out",
        str(tree["out"]),
        *extra,
    ]


def test_load_yolo_labels_absolute_pixels(tmp_path: Path):
    img = tmp_path / "x.png"
    _image(img, w=200, h=100)  # 归一化 0.1x0.2 → 20x20 像素
    lbl = tmp_path / "x.txt"
    lbl.write_text("6 0.5 0.5 0.1 0.2\n", encoding="utf-8")
    out = _load_yolo_labels(lbl, (200, 100))
    assert len(out) == 1
    assert out[0]["class_id"] == 6  # 内凹
    x, y, w, h = out[0]["bbox"]
    assert (x, y, w, h) == (90.0, 40.0, 20.0, 20.0)


def test_missing_label_file_means_empty(tmp_path: Path):
    assert _load_yolo_labels(tmp_path / "none.txt", (100, 100)) == []


def test_basic_eval_writes_result(eval_tree: dict[str, Path], capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", _argv(eval_tree))
    main()
    payload = json.loads(eval_tree["out"].read_text(encoding="utf-8"))
    assert payload["n_defect_images"] == 2
    std = payload["result"]["standard"]
    assert std["per_class"]["1"]["td"] == 2  # 两个气孔全正检
    assert std["tdr"] == 1.0
    # 未提供无缺陷集 → FRR 未测，不给出分级（比标准严）
    assert std["frr_measured"] is False and std["level"] is None
    assert "记录分级" in capsys.readouterr().out


def test_clean_set_pollution_rejected(eval_tree: dict[str, Path], monkeypatch):
    # 无缺陷集里混入缺陷标注 → 拒绝评价（不产出结果文件）
    (eval_tree["clean"] / "labels" / "c.txt").write_text("4 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv", _argv(eval_tree, "--clean-img-dir", str(eval_tree["clean"] / "images"))
    )
    with pytest.raises(SystemExit, match="拒绝评价"):
        main()
    assert not eval_tree["out"].exists()


def test_clean_set_ok_counts_frr(eval_tree: dict[str, Path], monkeypatch):
    # 无缺陷集干净 + 一张误报预测 → FRR = 1/1
    (eval_tree["root"] / "preds" / "clean_c.txt").write_text(
        "0 0.5 0.5 0.1 0.1\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(
            eval_tree,
            "--clean-img-dir",
            str(eval_tree["clean"] / "images"),
            "--clean-pred-dir",
            str(eval_tree["root"] / "preds"),
        ),
    )
    main()
    payload = json.loads(eval_tree["out"].read_text(encoding="utf-8"))
    std = payload["result"]["standard"]
    assert std["no_defect"]["reported_a"] == 0  # clean_c 预测键不匹配无缺陷集，未计入
    assert payload["n_no_defect_images"] == 1


def test_weld_form_override(eval_tree: dict[str, Path], monkeypatch):
    monkeypatch.setattr(
        sys, "argv", _argv(eval_tree, "--weld-form", "double", "--weld-method", "auto")
    )
    main()
    payload = json.loads(eval_tree["out"].read_text(encoding="utf-8"))
    assert payload["result"]["weld_form"] == "double"
    assert payload["result"]["weld_method"] == "auto"


def test_pred_and_model_mutually_exclusive(eval_tree: dict[str, Path], monkeypatch):
    monkeypatch.setattr(sys, "argv", _argv(eval_tree, "--model", "x.onnx"))
    with pytest.raises(SystemExit):
        main()
