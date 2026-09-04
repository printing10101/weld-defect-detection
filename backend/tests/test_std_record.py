"""附录A 评价记录表 + 人员资质 单测（DB50/T 1807-2025  / ）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from backend.app.routers.std_eval import router as std_eval_router
from backend.evaluation.qualification import (
    Personnel,
    check_personnel,
    parse_cert_level,
)
from backend.evaluation.std_record import build_record
from backend.infra.reporting.std_eval_record import build_record_pdf

# ---------------------------------------------------------------------------
# 测试夹具：一次小规模标准评价结果（evaluate 产物的同构字典）
# ---------------------------------------------------------------------------


@pytest.fixture()
def eval_result() -> dict:
    from backend.evaluation.std501807 import StdEvalConfig, evaluate

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
def people() -> list[Personnel]:
    return [
        Personnel(
            name="张三",
            cert_type="RT(D)-II",
            role="evaluator",
            cert_no="RTD-001",
            valid_until="2099-12-31",
        ),
        Personnel(name="李四", cert_type="RT-Ⅱ", role="labeler", cert_no="RT-002"),
        Personnel(name="王五", cert_type="RT-Ⅱ", role="labeler", cert_no="RT-003"),
    ]


# ---------------------------------------------------------------------------
# 资质
# ---------------------------------------------------------------------------


def test_parse_cert_level_variants():
    assert parse_cert_level("RT(D)-II") == 2
    assert parse_cert_level("RT-Ⅱ") == 2
    assert parse_cert_level("rt 3") == 3
    assert parse_cert_level("RT III") == 3
    assert parse_cert_level("MT-II") is None
    assert parse_cert_level("") is None


def test_check_personnel_qualified():
    res = check_personnel(
        [Personnel("张三", "RT(D)-II", "evaluator"), Personnel("李四", "RT-II", "labeler")]
    )
    assert res["qualified"] is True and res["issues"] == []


def test_check_personnel_insufficient_level():
    res = check_personnel(
        [Personnel("张三", "RT-I", "evaluator"), Personnel("李四", "RT-II", "labeler")]
    )
    assert res["qualified"] is False
    assert any("低于岗位要求" in i for i in res["issues"])


def test_check_personnel_expired_and_missing():
    res = check_personnel([Personnel("张三", "RT(D)-II", "evaluator", valid_until="2020-01-01")])
    assert res["qualified"] is False
    assert any("过期" in i for i in res["issues"])
    assert any("缺少标注人员" in i for i in res["issues"])


def test_check_personnel_bad_date_conservative():
    # 有效期格式非法 → 保守判过期（从严）
    res = check_personnel(
        [
            Personnel("张三", "RT(D)-II", "evaluator", valid_until="not-a-date"),
            Personnel("李四", "RT-II", "labeler"),
        ]
    )
    assert res["qualified"] is False
    assert any("过期" in i for i in res["issues"])


# ---------------------------------------------------------------------------
# 记录表装配（表A.1）
# ---------------------------------------------------------------------------


def test_build_record_fields(eval_result, people, tmp_path: Path):
    rec = build_record(
        eval_result,
        system_name="测试系统",
        system_version="1.0",
        developer="测试单位",
        film_kind="RT",
        weld_form="single",
        weld_method="manual",
        n_defect_images=1,
        n_no_defect_images=2,
        people=people,
    )
    assert rec["meta"]["system_name"] == "测试系统"
    assert rec["film"]["weld_form"] == "单面焊"
    assert rec["film"]["n_defects"] == 1
    assert rec["metrics"]["kdr"] == 1.0
    assert rec["grading"]["level_standard"] == "L4"  # 小样本全对 + FRR 0%
    assert rec["grading"]["official"] is True
    assert "裂纹" in rec["film"]["class_distribution"]


def test_build_record_unqualified_reference_only(eval_result, tmp_path: Path):
    # 无人员资质 → 官方结论降级为参考值
    rec = build_record(
        eval_result,
        system_name="s",
        system_version="v",
        developer="d",
        people=[],
        operator="",
    )
    assert rec["grading"]["official"] is False
    assert "参考值" in rec["grading"]["note"]


# ---------------------------------------------------------------------------
# PDF（reportlab 实际生成）
# ---------------------------------------------------------------------------


def test_build_record_pdf(eval_result, people, tmp_path: Path):
    rec = build_record(
        eval_result,
        system_name="s",
        system_version="v",
        developer="d",
        people=people,
        operator="张三",
    )
    out = tmp_path / "rec.pdf"
    path = build_record_pdf(rec, out)
    assert path.exists() and path.stat().st_size > 1000
    assert path.read_bytes()[:5] == b"%PDF-"


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


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
    # 隔离：评价产物/人员记录都指向 tmp_path
    # 隔离：评价产物/人员记录用环境变量指向 tmp（生产代码已废除 CWD 相对解析，
    # chdir 隔离不再有效——这正是本次路径统一要消除的语义）
    eval_dir = tmp_path / "data" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SCAN_STD_EVAL__EVAL_DIR", str(eval_dir))
    monkeypatch.setenv("SCAN_STD_EVAL__PERSONNEL_PATH", str(eval_dir / "std_personnel.json"))
    eval_json = eval_dir / "std_eval.json"
    eval_json.write_text(
        json.dumps(
            {"n_defect_images": 1, "n_no_defect_images": 2, "result": eval_result},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return TestClient(app)


def test_personnel_roundtrip(client: TestClient, tmp_path: Path):
    r = client.get("/std-eval/personnel")
    assert r.status_code == 200 and r.json()["qualified"] is False  # 尚未录入
    r = client.put(
        "/std-eval/personnel",
        json={
            "people": [
                {
                    "name": "张三",
                    "cert_type": "RT(D)-II",
                    "role": "evaluator",
                    "valid_until": "2099-01-01",
                },
                {"name": "李四", "cert_type": "RT-II", "role": "labeler"},
                {"name": "王五", "cert_type": "RT-II", "role": "labeler"},
            ]
        },
    )
    assert r.status_code == 200 and r.json()["qualified"] is True
    assert (tmp_path / "data/eval/std_personnel.json").exists()


def test_personnel_empty_rejected(client: TestClient):
    assert client.put("/std-eval/personnel", json={"people": []}).status_code == 422


def test_record_flow_json_and_pdf(client: TestClient, tmp_path: Path):
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
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["grading"]["level_standard"] == "L4"
    assert (tmp_path / "data/eval/std_record.json").exists()
    r2 = client.get("/std-eval/record/pdf", params={"record_name": "std_record"})
    assert r2.status_code == 200
    assert r2.content[:5] == b"%PDF-"


def test_record_missing_eval_result(client: TestClient):
    r = client.post("/std-eval/record", json={"eval_result_path": "data/eval/none.json"})
    assert r.status_code == 404
