"""底片印字识别（扫描日期/编号，正/镜像）测试。

覆盖：
- 文本模式判定（日期合理性/编号形制）纯单元；
- read_stamp 域链路：正向/镜像/空白/关闭（真实 RapidOCR 引擎，合成底片）；
- summary 快照契约；
- 仓储 flag_missing_stamps：批次收尾缺印字复核补标（延迟合并，只增不减）；
- Registry 印字占比裁决：大批普遍无印字豁免 / 占比达标补标 / 小批量不豁免；
- BatchManager on_finished 收尾钩子（摘要入批次快照、fail-soft）。
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.app.batch_queue import BatchItem, BatchManager
from backend.domain.stamp import StampCfg, StampResult, is_date_token, is_id_token, read_stamp
from backend.infra.repository import InspectionRepository

# ---------------------------------------------------------------------------
# 文本模式（纯单元，不依赖 OCR 引擎）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["2023-08-12", "2023.08.12", "2023/08/12", "2023年08月12日", "20230812", "19991231"],
)
def test_is_date_token_accepts_common_formats(text: str) -> None:
    assert is_date_token(text)


@pytest.mark.parametrize("text", ["20231308", "20231332", "18990101", "abc"])
def test_is_date_token_rejects_invalid_dates(text: str) -> None:
    # 单位数月日（2023-8-12）是合法印字格式，放行
    assert is_date_token("2023-8-12")


@pytest.mark.parametrize("text", ["0421", "No.0421", "B3-042", "W12-0345"])
def test_is_id_token_accepts_number_like(text: str) -> None:
    assert is_id_token(text)


@pytest.mark.parametrize("text", ["ABC", "ab12", "12", "2023-08-12"])
def test_is_id_token_rejects_non_number_like(text: str) -> None:
    # 纯日期串不算编号（避免同一段文字重复计入）；纯字母/数字过短不算编号
    assert not is_id_token(text)


def test_stamp_result_summary_contract() -> None:
    s = StampResult(status="present", text="2023-08-12", orientation="normal", confidence=0.9)
    assert s.summary(need_review=False) == {
        "status": "present",
        "text": "2023-08-12",
        "orientation": "normal",
        "confidence": 0.9,
        "need_review": False,
    }
    missing = StampResult(status="missing").summary(need_review=True)
    assert missing["need_review"] is True and missing["text"] is None


# ---------------------------------------------------------------------------
# read_stamp（真实引擎；引擎缺失时整组 skip，不影响无 OCR 环境）
# ---------------------------------------------------------------------------


def _stamped_film(mirror: bool = False, with_text: bool = True, seed: int = 7) -> np.ndarray:
    """合成暗底片 + 亮印字（日期 + 编号），mirror=True 模拟背面扫描镜像件。"""
    rng = np.random.default_rng(seed)
    img = rng.normal(60, 6.0, (900, 1600)).astype(np.uint8)
    cv2.rectangle(img, (0, 0), (1599, 899), 110, 8)
    if with_text:
        cv2.putText(img, "2023-08-12", (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 200, 4)
        cv2.putText(img, "No.0421", (80, 260), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 200, 4)
        if mirror:
            img = cv2.flip(img, 1)
    return img


@pytest.fixture(scope="module")
def _ocr_engine():
    pytest.importorskip("rapidocr_onnxruntime")


@pytest.mark.usefixtures("_ocr_engine")
class TestReadStamp:
    def test_normal_film_recognized(self) -> None:
        r = read_stamp(_stamped_film(), StampCfg())
        assert r.status == "present"
        assert r.orientation == "normal"
        assert r.text is not None and "2023-08-12" in r.text
        assert r.confidence is not None and r.confidence >= 0.6

    def test_mirrored_film_recognized_as_mirrored(self) -> None:
        for seed in (7, 99):
            r = read_stamp(_stamped_film(mirror=True, seed=seed), StampCfg())
            assert r.status == "present", f"seed={seed}: {r}"
            assert r.orientation == "mirrored", f"seed={seed}: {r}"
            assert r.text is not None and "2023-08-12" in r.text

    def test_blank_film_missing(self) -> None:
        r = read_stamp(_stamped_film(with_text=False), StampCfg())
        assert r.status == "missing"
        assert r.orientation is None

    def test_disabled_off(self) -> None:
        r = read_stamp(_stamped_film(), StampCfg(enabled=False))
        assert r.status == "off"

    def test_summary_of_missing_flags_review(self) -> None:
        r = read_stamp(_stamped_film(with_text=False), StampCfg())
        assert r.summary(need_review=True)["need_review"] is True


def test_read_stamp_fail_soft_on_engine_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """引擎抛异常 → 降级 unavailable，绝不向评片主链路传播异常。"""

    def _boom(_img):
        raise RuntimeError("engine exploded")

    monkeypatch.setattr("backend.domain.stamp._get_engine", lambda: _boom)
    r = read_stamp(_stamped_film(), StampCfg())
    assert r.status == "unavailable"
    assert "engine exploded" in (r.note or "")


# ---------------------------------------------------------------------------
# 仓储：批次收尾缺印字复核补标
# ---------------------------------------------------------------------------


def _tmp_repo() -> InspectionRepository:
    # mkdtemp 生成不可预测目录（避免 mktemp 可预测路径被抢占/跨用户读取）
    db_dir = tempfile.mkdtemp(prefix="stamp_test_")
    return InspectionRepository(str(Path(db_dir) / "test.db"))


def _add_image(
    repo: InspectionRepository,
    image_id: str,
    *,
    batch_no: str | None,
    stamp_status: str | None,
    stamp_need_review: bool = False,
    need_review: bool = False,
) -> None:
    repo.create_inspection(
        {
            "id": image_id,
            "path": f"{image_id}.png",
            "source_type": "image",
            "modality": "GENERIC",
            "batch_no": batch_no,
            "stamp_status": stamp_status,
            "stamp_need_review": stamp_need_review,
            "need_review": need_review,
        },
        [],
    )


def test_flag_missing_stamps_batch_scoped_and_idempotent() -> None:
    repo = _tmp_repo()
    _add_image(repo, "m1", batch_no="b1", stamp_status="missing")  # 应补标
    _add_image(repo, "p1", batch_no="b1", stamp_status="present")  # 有印字不动
    _add_image(
        repo, "f1", batch_no="b1", stamp_status="missing", stamp_need_review=True
    )  # 已标不动
    _add_image(repo, "m2", batch_no="b2", stamp_status="missing")  # 别的批次不动
    _add_image(repo, "m3", batch_no=None, stamp_status="missing")  # 单图（无批次）不动

    assert repo.flag_missing_stamps("b1") == 1
    m1 = repo.get_image("m1")
    assert m1 is not None
    assert m1["need_review"] is True
    assert m1["stamp_need_review"] is True
    for img, field in (("p1", "need_review"), ("f1", "need_review"), ("m2", "need_review")):
        row = repo.get_image(img)
        assert row is not None and row[field] is False
    m3 = repo.get_image("m3")
    assert m3 is not None and m3["need_review"] is False
    # 幂等：第二次无可补标行
    assert repo.flag_missing_stamps("b1") == 0
    # 不存在的批次
    assert repo.flag_missing_stamps("nope") == 0


def test_flag_missing_stamps_preserves_existing_review_flag() -> None:
    """延迟合并只增不减：已有其它来源 need_review 的缺印字片不被覆盖。"""
    repo = _tmp_repo()
    _add_image(repo, "x1", batch_no="b1", stamp_status="missing", need_review=True)
    assert repo.flag_missing_stamps("b1") == 1
    x1 = repo.get_image("x1")
    assert x1 is not None and x1["need_review"] is True


# ---------------------------------------------------------------------------
# Registry 印字占比裁决（批量豁免规则）
# ---------------------------------------------------------------------------


def _stamp_res(status: str) -> dict:
    return {"status": status, "text": None, "orientation": None, "confidence": None}


@pytest.fixture()
def registry():
    from backend.app import dependencies as deps

    return deps.get_registry()


class TestBatchStampPolicy:
    def test_majority_missing_suppressed(self, registry, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(registry.config.stamp, "batch_flag_min", 4)
        monkeypatch.setattr(registry.config.stamp, "batch_flag_ratio", 0.5)
        results = [{"stamp": _stamp_res("missing")} for _ in range(9)]
        results.append({"stamp": _stamp_res("present")})  # 1/10 有印字
        summary = registry._apply_batch_stamp_policy("batchA", results)
        assert summary is not None
        assert summary["evaluated"] == 10
        assert summary["present"] == 1
        assert summary["suppressed"] is True
        assert summary["flagged"] == 0

    def test_majority_present_flags_missing(
        self, registry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(registry.config.stamp, "batch_flag_min", 4)
        monkeypatch.setattr(registry.config.stamp, "batch_flag_ratio", 0.5)
        results = [{"stamp": _stamp_res("present")} for _ in range(4)]
        results.append({"stamp": _stamp_res("missing")})
        summary = registry._apply_batch_stamp_policy("batchB", results)
        assert summary is not None
        assert summary["suppressed"] is False
        assert summary["flagged"] == 0  # 测试库无该批次影像，补标 0 行但走达裁决

    def test_small_batch_never_suppressed(self, registry, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(registry.config.stamp, "batch_flag_min", 4)
        monkeypatch.setattr(registry.config.stamp, "batch_flag_ratio", 0.5)
        results = [{"stamp": _stamp_res("missing")} for _ in range(3)]
        summary = registry._apply_batch_stamp_policy("batchC", results)
        assert summary is not None
        assert summary["suppressed"] is False

    def test_no_evaluable_results_returns_none(self, registry) -> None:
        results = [{"stamp": _stamp_res("off")}, {"stamp": _stamp_res("unavailable")}, {}]
        assert registry._apply_batch_stamp_policy("batchD", results) is None


# ---------------------------------------------------------------------------
# BatchManager 收尾钩子
# ---------------------------------------------------------------------------


class _FakePipeline:
    """返回固定结果的假流水线（批量收尾钩子集成用）。"""

    def __init__(self, results: list[dict]) -> None:
        self._results = list(results)

    def run_inspection(self, **_kw) -> dict:
        return self._results.pop(0)


def _wait_finished(bm: BatchManager, batch_id: str, timeout_loops: int = 200) -> dict:
    """轮询批量直至 finished（超时 fail），返回终态快照。"""
    for _ in range(timeout_loops):
        snap = bm.status(batch_id)
        assert snap is not None
        if snap["status"] == "finished":
            return snap
        time.sleep(0.05)
    raise AssertionError("batch 未在限时内完成")


def test_batch_finish_hook_runs_and_stores_summary(tmp_path: Path) -> None:
    seen: dict = {}

    def hook(batch_id: str, results: list[dict]) -> dict:
        seen["batch_id"] = batch_id
        seen["results"] = results
        return {"evaluated": 1, "present": 1, "ratio": 1.0, "suppressed": False, "flagged": 0}

    bm = BatchManager(
        lambda: _FakePipeline([{"image_id": "i1", "need_review": False}]),
        workers=1,
        per_image_estimate_sec=0.1,
        batch_dir=tmp_path,
        on_finished=hook,
    )
    film = tmp_path / "f.png"
    film.write_bytes(b"fake")
    batch_id = bm.submit([BatchItem(image_path=film, options={})])
    snap = _wait_finished(bm, batch_id)
    assert seen["batch_id"] == batch_id
    assert seen["results"][0]["image_id"] == "i1"
    assert snap["stamp_summary"]["present"] == 1


def test_batch_finish_hook_failure_is_soft(tmp_path: Path) -> None:
    def bad_hook(_batch_id: str, _results: list[dict]) -> dict:
        raise RuntimeError("policy exploded")

    bm = BatchManager(
        lambda: _FakePipeline([{"image_id": "i1"}]),
        workers=1,
        per_image_estimate_sec=0.1,
        batch_dir=tmp_path,
        on_finished=bad_hook,
    )
    film = tmp_path / "f.png"
    film.write_bytes(b"fake")
    batch_id = bm.submit([BatchItem(image_path=film, options={})])
    snap = _wait_finished(bm, batch_id)
    assert "stamp_summary" not in snap
