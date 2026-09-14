"""数据集泄漏审计测试（domain.labeling.leakage + dataset_builder 集成 + CLI）。

场景对齐 "Revisiting RIAWELC"：同一物理底片的衍生图（copy-paste 合成、
过采样副本）跨越 train/val/test 即构成评估泄漏，指标被乐观污染。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.domain.labeling.leakage import (
    assert_no_leakage,
    assign_groups,
    audit_leakage,
    film_group,
    scan_split,
)
from backend.training.audit_dataset_leakage import main as cli_main


def _write_img(path: Path, seed: int, size: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (size, size), dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)
    path.write_bytes(buf.tobytes())


class TestFilmGroup:
    def test_identity(self):
        assert film_group("wf_0123") == "wf_0123"

    def test_oversample_prefix_stripped(self):
        assert film_group("os1_rare1") == "rare1"
        assert film_group("os2_os1_rare1") == "rare1"

    def test_copy_paste_dual_parent(self):
        assert film_group("cp_000042_filma_x_filmb") == "filma|filmb"

    def test_rare_copy_paste_dual_parent(self):
        assert film_group("rcp_000007_b_x_a") == "a|b"

    def test_group_order_normalized(self):
        assert film_group("cp_1_bbb_x_aaa") == film_group("cp_2_aaa_x_bbb")


class TestAssignGroups:
    def test_identity_grouping(self):
        g = assign_groups(["a", "b"])
        assert g["a"] == "a"
        assert g["b"] == "b"

    def test_copy_paste_merges_parents(self):
        g = assign_groups(["filmb", "filma", "cp_000001_filma_x_filmb"])
        assert g["filma"] == g["filmb"] == g["cp_000001_filma_x_filmb"]

    def test_rep_prefers_plain_film_name(self):
        g = assign_groups(["filmb", "cp_000001_filma_x_filmb", "filma"])
        assert g["cp_000001_filma_x_filmb"] == "filma"

    def test_oversample_copy_same_group(self):
        g = assign_groups(["rare1", "os1_rare1", "os2_os1_rare1"])
        assert len(set(g.values())) == 1

    def test_chained_derivations_transitive(self):
        # cp 的亲本本身是另一张 cp 的亲本 → 传递归并
        g = assign_groups(["f1", "f2", "f3", "cp_1_f1_x_f2", "cp_2_f2_x_f3"])
        assert len(set(g.values())) == 1

    def test_order_independent(self):
        stems = ["b", "cp_1_a_x_b", "a", "c"]
        assert assign_groups(stems) == assign_groups(list(reversed(stems)))


class TestAuditLeakage:
    def _setup(self, tmp: Path) -> dict[str, Path]:
        d = {
            "train": tmp / "train" / "images",
            "val": tmp / "val" / "images",
            "test": tmp / "test" / "images",
        }
        _write_img(d["train"] / "film_a.png", 1)
        _write_img(d["train"] / "cp_000001_film_a_x_film_b.png", 2)
        _write_img(d["val"] / "film_c.png", 3)
        _write_img(d["test"] / "film_b.png", 4)
        return d

    def test_clean_dataset_passes(self, tmp_path: Path):
        d = self._setup(tmp_path)
        audit = audit_leakage(d)
        assert audit.per_split == {"train": 2, "val": 1, "test": 1}
        assert not audit.duplicate_clusters
        assert not audit.perceptual_pairs
        assert audit.passed  # 分组越界存在但默认仅报告

    def test_cross_split_group_detected(self, tmp_path: Path):
        d = self._setup(tmp_path)
        # cp 图在 train、亲本 film_b 在 test → 同源等价类跨 split
        audit = audit_leakage(d)
        (grp,) = audit.cross_split_groups
        assert grp["group"] == "film_a"  # 代表取类内最小编号底片名
        assert set(grp["splits"]) == {"train", "test"}  # type: ignore[arg-type]

    def test_enforce_groups_fails(self, tmp_path: Path):
        d = self._setup(tmp_path)
        audit = audit_leakage(d, enforce_groups=True)
        assert not audit.passed
        with pytest.raises(RuntimeError, match="跨split同源分组"):
            assert_no_leakage(d, enforce_groups=True)

    def test_exact_duplicate_across_splits(self, tmp_path: Path):
        d = self._setup(tmp_path)
        _write_img(d["val"] / "dup_of_a.png", 1)  # 与 train/film_a.png 同字节
        audit = audit_leakage(d)
        cross = [c for c in audit.duplicate_clusters if c["cross_split"]]
        assert len(cross) == 1
        assert set(cross[0]["splits"]) == {"train", "val"}  # type: ignore[arg-type]
        assert not audit.passed

    def test_within_split_duplicate_report_only(self, tmp_path: Path):
        d = self._setup(tmp_path)
        _write_img(d["train"] / "same_as_a.png", 1)  # train 内部重复：不构成泄漏
        audit = audit_leakage(d)
        assert audit.duplicate_clusters
        assert not any(c["cross_split"] for c in audit.duplicate_clusters)
        assert audit.passed

    def test_perceptual_near_duplicate_fails(self, tmp_path: Path):
        d = self._setup(tmp_path)
        # 全图 +1 灰度：dHash 比较关系不变（hamming=0），md5 不同 → 仅感知命中
        rng = np.random.default_rng(99)
        img = rng.integers(0, 255, (64, 64), dtype=np.uint8)
        shifted = (img.astype(np.int16) + 1).astype(np.uint8)
        _, buf = cv2.imencode(".png", img)
        (d["train"] / "orig.png").write_bytes(buf.tobytes())
        _, buf = cv2.imencode(".png", shifted)
        (d["test"] / "noisy.png").write_bytes(buf.tobytes())
        audit = audit_leakage(d)
        assert audit.perceptual_pairs
        assert audit.perceptual_pairs[0]["hamming"] == 0
        assert not audit.passed


class TestScanSplit:
    def test_suffix_filter_and_naming(self, tmp_path: Path):
        d = tmp_path / "images"
        _write_img(d / "a.png", 5)
        (d / "notes.txt").write_text("x", encoding="utf-8")
        entries = scan_split("train", d)
        assert [e.name for e in entries] == ["train/a.png"]
        assert len(entries[0].md5) == 32

    def test_missing_dir_returns_empty(self, tmp_path: Path):
        assert scan_split("train", tmp_path / "nope") == []


class TestBuildDatasetGroupAware:
    def _make_raw(self, raw: Path, n_films: int = 12) -> None:
        (raw / "user" / "labels").mkdir(parents=True, exist_ok=True)
        for i in range(n_films):
            _write_img(raw / "user" / "images" / f"film_{i:02d}.png", 100 + i)
            (raw / "user" / "labels" / f"film_{i:02d}.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
            )
        # 同类别层的合成图：必须与亲本 film_00 / film_01 同 split
        _write_img(raw / "user" / "images" / "cp_000001_film_00_x_film_01.png", 200)
        (raw / "user" / "labels" / "cp_000001_film_00_x_film_01.txt").write_text(
            "0 0.3 0.3 0.1 0.1\n", encoding="utf-8"
        )

    def test_same_group_never_across_splits(self, tmp_path: Path, monkeypatch):
        import backend.training.dataset_builder as db

        raw = tmp_path / "raw"
        self._make_raw(raw)
        monkeypatch.setattr(db, "_RAW_ROOT", raw)
        out = tmp_path / "out"
        db.build_dataset(out_root=out, seed=7)

        def stems(split: str) -> set[str]:
            return {p.stem for p in (out / split / "images").iterdir()}

        s = {k: stems(k) for k in ("train", "val", "test")}
        cp = "cp_000001_film_00_x_film_01"
        cp_split = next(k for k, v in s.items() if cp in v)
        assert "film_00" in s[cp_split], f"亲本 film_00 与合成图分离: {s}"
        assert "film_01" in s[cp_split], f"亲本 film_01 与合成图分离: {s}"

    def test_audit_report_landed(self, tmp_path: Path, monkeypatch):
        import backend.training.dataset_builder as db

        raw = tmp_path / "raw"
        self._make_raw(raw)
        monkeypatch.setattr(db, "_RAW_ROOT", raw)
        out = tmp_path / "out"
        db.build_dataset(out_root=out, seed=7)
        report = json.loads((out / "leakage_audit.json").read_text(encoding="utf-8"))
        assert report["n_files"] == 13
        assert report["passed"]  # 分组划分后无泄漏

    def test_train_only_synthetic_leak_blocked(self, tmp_path: Path, monkeypatch):
        """train-only 合成源（伪标签同款语义）：cp 恒进 train，亲本组被划入
        test → 同源组跨 split；enforce_groups=True 时阻断且报告先落盘。"""
        import backend.training.dataset_builder as db

        raw = tmp_path / "raw"
        self._make_raw(raw, n_films=12)
        syn_lbl = raw / "synthetic" / "labels"
        syn_lbl.mkdir(parents=True, exist_ok=True)
        _write_img(raw / "synthetic" / "images" / "cp_000002_film_00_x_film_01.png", 201)
        (syn_lbl / "cp_000002_film_00_x_film_01.txt").write_text(
            "0 0.6 0.6 0.1 0.1\n", encoding="utf-8"
        )
        monkeypatch.setattr(db, "_RAW_ROOT", raw)
        out = tmp_path / "out"
        with pytest.raises(RuntimeError, match="enforce_groups"):
            db.build_dataset(
                out_root=out, seed=2, train_only_sources={"synthetic"}, enforce_groups=True
            )
        # 报告先于异常落盘：失败现场仍可审计
        report = json.loads((out / "leakage_audit.json").read_text(encoding="utf-8"))
        assert report["cross_split_groups"]


class TestCli:
    def _setup(self, tmp: Path) -> tuple[dict[str, Path], Path]:
        d = {
            "train": tmp / "train" / "images",
            "test": tmp / "test" / "images",
        }
        _write_img(d["train"] / "a.png", 11)
        _write_img(d["test"] / "b.png", 12)
        return d, tmp / "report.json"

    def test_pass(self, tmp_path: Path, capsys):
        d, report = self._setup(tmp_path)
        rc = cli_main(["--train", str(d["train"]), "--test", str(d["test"]), "--json", str(report)])
        assert rc == 0
        assert json.loads(report.read_text(encoding="utf-8"))["passed"]
        assert "通过" in capsys.readouterr().out

    def test_fail_on_duplicate(self, tmp_path: Path):
        d, report = self._setup(tmp_path)
        _write_img(d["test"] / "dup.png", 11)  # 与 train/a.png 同字节
        rc = cli_main(["--train", str(d["train"]), "--test", str(d["test"]), "--json", str(report)])
        assert rc == 1

    def test_split_root_autodetect(self, tmp_path: Path):
        """传 split 根目录时自动取 images/ 子目录。"""
        d, _ = self._setup(tmp_path)
        rc = cli_main(["--train", str(d["train"].parent), "--test", str(d["test"].parent)])
        assert rc == 0

    def test_single_split_rejected(self, tmp_path: Path):
        d, _ = self._setup(tmp_path)
        with pytest.raises(SystemExit):
            cli_main(["--train", str(d["train"])])

    def test_module_entrypoint(self, tmp_path: Path):
        d, _ = self._setup(tmp_path)
        r = subprocess.run(
            [
                sys.executable,
                "-m",
                "backend.training.audit_dataset_leakage",
                "--train",
                str(d["train"]),
                "--test",
                str(d["test"]),
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=Path(__file__).resolve().parents[2],
        )
        assert r.returncode == 0, r.stderr
        assert "结论: 通过" in r.stdout
