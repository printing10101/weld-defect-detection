"""pytest 根配置：backend 包导入 + 测试环境隔离。

M6 起涉及真实 DB/报告目录/标准表：session 级 autouse fixture 将
SCAN_* 环境变量指向临时目录 + authorized 测试表副本，保证：
- 测试写库/写报告不污染 data/（生产 data/scan.db 不受影响）；
- 全链路评级可测（生产表 authorized=false 会熔断，测试注入 true）。
环境变量在 get_registry()（懒单例）首次调用前生效。
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TMP_ROOT = Path(tempfile.mkdtemp(prefix="scan_m6_test_"))


@pytest.fixture(scope="session")
def auth_table() -> Path:
    """authorized=true 的测试表副本（report 全链路评级测试注入用）。

    生产表 authorized=false 会熔断（§T8），需要评级时由测试注入本副本。
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
    # P2-9：测试禁用限流（TestClient 共享计数会误伤套件）；安全头中间件保持生效
    os.environ.setdefault("SCAN_RATE_LIMIT", "0")
    # 检测器确定性：默认强制 M4a 基线（blob），不依赖开发机是否存在训练权重。
    # 集成测试用合成底片断言 ≥N 缺陷，训练 YOLO 在合成图上 0 检出；
    # 训练模型路径由 test_yolo_detector_ml（@pytest.mark.ml）等直接实例化 YoloDetector 覆盖。
    # 设 SCAN_TEST_REAL_DETECTOR=1 可关闭强制，改用真实 YOLO（需 ML 依赖+权重），
    # 用于本地验证真实检测链路——此时部分集成测试（断言 ≥N 缺陷）可能需相应调整。
    if not os.environ.get("SCAN_TEST_REAL_DETECTOR"):
        os.environ["SCAN_DETECT__BASELINE_ENABLED"] = "true"
