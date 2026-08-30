"""pytest 根配置：backend 包导入 + 测试环境隔离。

 起涉及真实 DB/报告目录/标准表：session 级 autouse fixture 将
SCAN_* 环境变量指向临时目录 + authorized 测试表副本，保证：
- 测试写库/写报告不污染 data/（生产 data/scan.db 不受影响）；
- 全链路评级可测（生产表 authorized=false 会熔断，测试注入 true）。
环境变量在 get_registry（懒单例）首次调用前生效。
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest
from fastapi import Request  # noqa: TCH002 - 须在模块级：get_type_hints 解析 override 注解需要

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TMP_ROOT = Path(tempfile.mkdtemp(prefix="scan_m6_test_"))

# C-17 IPC 一次性令牌：默认关闭（对测试侵入最小的方案——免于给全部既有
# TestClient 注入 X-IPC-Token 头）。注意 create_app 在 pytest 收集期（导入
# 测试模块）即构造中间件，因此必须在 conftest 模块体设置，晚于收集期的
# fixture 内设置无效。令牌真实链路（缺失 401 / 正确放行 / health 豁免）由
# test_ipc_token 专项覆盖（临时置 true 构造独立 app 实例）。
os.environ.setdefault("SCAN_IPC__ENFORCE", "false")

# P2-9：测试禁用限流（TestClient 共享计数会误伤套件）；安全头中间件保持生效。
# 同上：create_app 在收集期构造中间件，必须模块体设置（fixture 内设置无效——
# 曾因此导致全量运行时限流 60 req/min 跨测试共享，登录请求偶发 429）。
os.environ.setdefault("SCAN_RATE_LIMIT", "0")


@pytest.fixture(scope="session")
def auth_table() -> Path:
    """authorized=true 的测试表副本（report 全链路评级测试注入用）。

    生产表 authorized=false 会熔断，需要评级时由测试注入本副本。
    """
    src = _PROJECT_ROOT / "backend" / "domain" / "standards" / "tables" / "nb47013.yaml"
    dst = _TMP_ROOT / "nb47013_authorized_test.yaml"
    if not dst.exists():
        text = src.read_text(encoding="utf-8").replace("authorized: false", "authorized: true")
        dst.write_text(text, encoding="utf-8")
    return dst


@pytest.fixture(scope="session", autouse=True)
def _test_env(auth_table: Path) -> None:
    # 仅隔离数据目录（db/影像/报告），不动 standard——熔断语义测试依赖真实表
    del auth_table
    os.environ["SCAN_PATHS__DB_PATH"] = str(_TMP_ROOT / "test.db")
    os.environ["SCAN_PATHS__IMAGES_DIR"] = str(_TMP_ROOT / "images")
    os.environ["SCAN_PATHS__REPORTS_DIR"] = str(_TMP_ROOT / "reports")
    # 检测器确定性：默认强制 基线（blob），不依赖开发机是否存在训练权重。
    # 集成测试用合成底片断言 ≥N 缺陷，训练 YOLO 在合成图上 0 检出；
    # 训练模型路径由 test_yolo_detector_ml（@pytest.mark.ml）等直接实例化 YoloDetector 覆盖。
    # 设 SCAN_TEST_REAL_DETECTOR=1 可关闭强制，改用真实 YOLO（需 ML 依赖+权重），
    # 用于本地验证真实检测链路——此时部分集成测试（断言 ≥N 缺陷）可能需相应调整。
    if not os.environ.get("SCAN_TEST_REAL_DETECTOR"):
        os.environ["SCAN_DETECT__BASELINE_ENABLED"] = "true"
    # DB50/T 1807-2025 §5 扫描参数门禁（E-05）默认硬拦截 8bit 底片；历史夹具
    # 均为 8bit 合成 PNG，统一降级放行避免整组集成测试被新门禁误杀，
    # 硬拦截/留档路径由 test_gate_reject 专项覆盖（显式 monkeypatch 配置）。
    os.environ.setdefault("SCAN_GATE__ALLOW_8BIT", "true")
    # 留档目录同步隔离（默认 data/rejects 相对安装根目录，会写进项目 data/）
    os.environ.setdefault("SCAN_GATE__REJECTS_DIR", str(_TMP_ROOT / "rejects"))
    # C-14 导出管控默认按生产开启（require_approval=true）；既有 28 个测试
    # 文件直接 TestClient(app) 且以 sysadmin 注入身份，为避免整组测试被
    # 审批门禁误杀，这里放宽为 false；真实审批流由 test_export_control
    # 专项覆盖（临时置 true）。
    os.environ.setdefault("SCAN_EXPORT__REQUIRE_APPROVAL", "false")


@pytest.fixture(autouse=True)
def _auth_principal_override():
    """三员鉴权兼容（C-06/C-07）。

    鉴权依赖集中在 backend.app.auth.get_principal，生产由 main.py 以路由级
    依赖挂到全部业务路由；这里经 app.dependency_overrides 注入测试 principal
    （sysadmin 角色 + 固定测试账号名），使既有测试无需改造即可通过。
    鉴权真实生效性（无 token 401 / 越权 403 / 三员互斥）由 test_auth 专项
    覆盖（临时摘除本覆盖走真实链路）。
    """
    from backend.app import auth as _auth
    from backend.app.main import app

    def _fake_principal(request: Request):
        principal = _auth.Principal(
            account_id="test-account-id",
            username="测试管理员",
            role="sysadmin",
        )
        request.state.principal = principal  # get_operator_name 以账号为准
        return principal

    app.dependency_overrides[_auth.get_principal] = _fake_principal
    yield
    app.dependency_overrides.pop(_auth.get_principal, None)
