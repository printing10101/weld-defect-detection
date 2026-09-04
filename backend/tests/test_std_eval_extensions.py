"""std_eval 路由扩展（DB50/T 1807-2025 E-05/E-07/E-08/E-10）测试。

- POST /std-eval/consensus      三人标注一致性仲裁（E-07）；
- POST /std-eval/record         consensus 摘要写入附录A 记录（E-07）；
- POST /std-eval/evidence/{id}  漏检风险证据包（E-08）；
- GET  /std-eval/gate-rejects   不合格底片留档查询（E-05）；
- GET  /std-eval/false-reports  误报底片清单导出 CSV/JSON（E-10）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from skimage.io import imsave

from backend.app.routers.std_eval import router as std_eval_router
from backend.evaluation.gate_rejects import GateRejectStore
from backend.evaluation.std501807 import StdEvalConfig, evaluate
from backend.infra.config import load_config, resolve_config_path


@pytest.fixture()
def eval_result() -> dict:
    defect_set = [
        (
            "img1",
            [{"bbox": [0, 0, 10, 10], "class_id": 4}],
            [{"bbox": [0, 0, 10, 10], "class_id": 4, "score": 0.9}],
        ),
    ]
    no_defect = [("c1", []), ("c2", [])]
    return evaluate(defect_set, no_defect, StdEvalConfig())


@pytest.fixture()
def client(tmp_path: Path, monkeypatch, eval_result) -> TestClient:
    app = FastAPI()
    app.include_router(std_eval_router)

    # C-14 起导出端点需鉴权 + registry：本文件用独立 app 装配 std_eval 路由，
    # 这里注入测试 principal（sysadmin）并复用全局 registry（require_approval
    # 由 conftest 环境放宽为 false）。
    from backend.app import auth as _auth

    def _fake_principal(request: Request):
        p = _auth.Principal("test-account", "测试管理员", "sysadmin")
        request.state.principal = p
        return p

    app.dependency_overrides[_auth.get_principal] = _fake_principal
    # 隔离：评价产物/人员记录用环境变量指向 tmp（生产代码已废除 CWD 相对解析，
    # chdir 隔离不再有效——这正是本次路径统一要消除的语义）
    eval_dir = tmp_path / "data" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SCAN_STD_EVAL__EVAL_DIR", str(eval_dir))
    monkeypatch.setenv("SCAN_STD_EVAL__PERSONNEL_PATH", str(eval_dir / "std_personnel.json"))
    eval_json = eval_dir / "std_eval.json"
    eval_json.write_text(
        json.dumps(
            {
                "n_defect_images": 1,
                "n_no_defect_images": 2,
                "false_report_films": [
                    {"id": "c3", "path": str(tmp_path / "c3.png"), "n_false_reports": 2}
                ],
                "result": eval_result,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return TestClient(app)


# ---------------------------------------------------------------------------
# E-07 三人标注一致性
# ---------------------------------------------------------------------------


def test_consensus_api_accepts_and_suggests(client: TestClient) -> None:
    anns = [
        {"annotator": "A", "class_id": 4, "bbox": [10, 10, 20, 20]},
        {"annotator": "B", "class_id": 4, "bbox": [12, 10, 20, 20]},
        {"annotator": "C", "class_id": 4, "bbox": [11, 12, 20, 20]},
    ]
    r = client.post("/std-eval/consensus", json={"annotations": anns, "threshold": 0.5})
    assert r.status_code == 200
    body = r.json()
    assert len(body["accepted"]) == 1 and body["discarded"] == []
    assert body["accepted"][0]["bbox"] == [10, 10, 22, 22]  # 三人并集外接框
    assert body["arbitration"]["needed"] is False


def test_consensus_api_discard_triggers_arbitration(client: TestClient) -> None:
    anns = [
        {"annotator": "A", "class_id": 4, "bbox": [10, 10, 20, 20]},
        {"annotator": "B", "class_id": 4, "bbox": [12, 10, 20, 20]},
        {"annotator": "C", "class_id": 0, "bbox": [11, 12, 20, 20]},  # 类型不一致
    ]
    r = client.post("/std-eval/consensus", json={"annotations": anns, "threshold": 0.5})
    body = r.json()
    assert body["accepted"] == [] and len(body["discarded"]) == 3
    assert body["agreement_rate"] == 0.0
    assert body["arbitration"]["needed"] is True and "仲裁" in body["arbitration"]["suggestion"]


def test_consensus_api_unknown_annotator_422(client: TestClient) -> None:
    anns = [{"annotator": "D", "class_id": 4, "bbox": [10, 10, 20, 20]}]
    assert client.post("/std-eval/consensus", json={"annotations": anns}).status_code == 422
    assert client.post("/std-eval/consensus", json={"annotations": []}).status_code == 422


def test_record_includes_labeling_consensus(client: TestClient, tmp_path: Path) -> None:
    client.put(
        "/std-eval/personnel",
        json={
            "people": [
                {"name": "张三", "cert_type": "RT(D)-II", "role": "evaluator"},
                {"name": "李四", "cert_type": "RT-II", "role": "labeler"},
            ]
        },
    )
    r = client.post(
        "/std-eval/record",
        json={
            "eval_result_path": str(tmp_path / "data" / "eval" / "std_eval.json"),
            "system_name": "s",
            "system_version": "v",
            "developer": "d",
            "consensus": {
                "accepted": [{"class_id": 4}],
                "discarded": [],
                "agreement_rate": 1.0,
                "threshold": 0.6,
            },
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["labeling_consensus"]["agreement_rate"] == 1.0
    assert body["labeling_consensus"]["needs_arbitration"] is False
    saved = json.loads((tmp_path / "data/eval/std_record.json").read_text(encoding="utf-8"))
    assert saved["labeling_consensus"]["accepted"] == 1


def test_record_without_consensus_has_no_block(client: TestClient, tmp_path) -> None:
    r = client.post(
        "/std-eval/record",
        json={
            "eval_result_path": str(tmp_path / "data" / "eval" / "std_eval.json"),
            "system_name": "s",
        },
    )
    assert r.status_code == 200
    assert "labeling_consensus" not in r.json()


# ---------------------------------------------------------------------------
# E-08 风险证据包
# ---------------------------------------------------------------------------


def _film(path: Path) -> Path:
    imsave(str(path), np.full((64, 80), 128, np.uint8))
    return path


def test_evidence_api_builds_image_and_manifest(client: TestClient, tmp_path: Path) -> None:
    film = _film(tmp_path / "film.png")
    r = client.post(
        "/std-eval/evidence/rec-1",
        json={
            "film_path": str(film),
            "film_id": "film-1",
            "defects": [{"defect_id": "d1", "class_id": 0, "grade": "I"}],
            "gt_boxes": [[10, 10, 10, 10]],
            "det_boxes": [[12, 10, 10, 10]],
        },
    )
    assert r.status_code == 200
    manifest = r.json()
    assert manifest["film_id"] == "film-1"
    assert manifest["defects"][0]["grade"] == "I"
    ev_img = Path(manifest["evidence_image"])
    assert ev_img.exists() and ev_img.stat().st_size > 0
    # manifest hash 与落盘证据图一致（SHA-256，hashlib）
    assert manifest["evidence_image_sha256"] == hashlib.sha256(ev_img.read_bytes()).hexdigest()
    mf = tmp_path / "data/eval/evidence/rec-1/manifest.json"
    assert mf.exists()


def test_evidence_api_missing_film_404(client: TestClient) -> None:
    r = client.post(
        "/std-eval/evidence/rec-2",
        json={"film_path": "no/such/film.png", "defects": []},
    )
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# E-05 留档查询
# ---------------------------------------------------------------------------


def test_gate_rejects_query(client: TestClient) -> None:
    db_path = str(resolve_config_path(load_config().paths.db_path))
    store = GateRejectStore(db_path)
    store.add(
        reject_id="r-query-1",
        reject_reason="位深 8bit 低于 16bit 硬门禁",
        detail={"reasons": ["位深 8bit 低于 16bit 硬门禁"]},
        dpi=None,
        bit_depth=8,
        operator="tester",
    )
    r = client.get("/std-eval/gate-rejects")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] >= 1
    assert any(item["id"] == "r-query-1" for item in body["items"])
    assert body["items"][0]["bit_depth"] == 8


# ---------------------------------------------------------------------------
# E-10 误报底片清单导出
# ---------------------------------------------------------------------------


def test_false_reports_export_json(client: TestClient) -> None:
    r = client.get("/std-eval/false-reports")
    assert r.status_code == 200
    body = r.json()
    assert body["n_films"] == 1
    film = body["films"][0]
    assert film["id"] == "c3" and film["n_false_reports"] == 2
    assert film["path"].endswith("c3.png")


def test_false_reports_export_csv(client: TestClient) -> None:
    r = client.get("/std-eval/false-reports", params={"fmt": "csv"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    lines = r.text.strip().splitlines()
    assert lines[0] == "id,path,n_false_reports,secret_level"
    assert "c3" in lines[1] and lines[1].endswith(",2,0")  # C-10：库内无此影像 → 密级默认 0
