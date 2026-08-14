"""部署硬化（#6）回归测试：配置去硬编码 + 运行时目录确定性创建。

覆盖：
  - 字体跨平台解析（去 Windows 硬编码），缺失时优雅降级 Helvetica；
  - 标注器数据目录锚定安装根（不再相对 CWD）；
  - ensure_runtime_dirs 在干净环境创建全部运行时目录。
"""

from __future__ import annotations

from pathlib import Path

from backend.infra.config import AppConfig, ensure_runtime_dirs
from backend.infra.reporting import pdf_reporter


def test_font_resolution_returns_usable_font() -> None:
    """当前平台应至少能解析到一个字体名（Windows 本机为 simhei/simfang 之一）。"""
    name = pdf_reporter._register_font()
    assert isinstance(name, str) and name != ""
    # 若本机无中文字体，应降级为 Helvetica 而非崩溃
    assert name in ("Helvetica",) or name.startswith("CN-")


def test_font_fallback_when_no_candidates(monkeypatch) -> None:
    """候选列表为空时，_register_font 必须降级返回 Helvetica 且不抛异常。"""
    monkeypatch.setattr(pdf_reporter, "_font_candidates", list)
    assert pdf_reporter._register_font() == "Helvetica"


def test_annotator_root_anchored_to_install_root() -> None:
    """标注器数据目录必须锚定安装根，且不依赖 CWD（Tauri/安装包场景）。"""
    from backend.annotator import server as annotator_server

    root = annotator_server.ROOT
    assert root.is_absolute(), "ROOT 必须绝对，否则 CWD 不同会导致路径漂移"
    assert root.parts[-2:] == ("data", "real_label")


def test_ensure_runtime_dirs_creates_all(tmp_path: Path) -> None:
    """干净环境传入绝对路径配置，ensure_runtime_dirs 必须创建全部运行时目录。"""
    cfg = AppConfig(
        paths={
            "data_dir": str(tmp_path / "data"),
            "tmp_dir": str(tmp_path / "data" / "tmp"),
            "db_path": str(tmp_path / "data" / "scan.db"),
            "images_dir": str(tmp_path / "data" / "images"),
            "reports_dir": str(tmp_path / "data" / "reports"),
        },
        model={"weights_dir": str(tmp_path / "models" / "weights")},
    )
    created = ensure_runtime_dirs(cfg)
    expected = {
        tmp_path / "data",
        tmp_path / "data" / "tmp",
        tmp_path / "data" / "images",
        tmp_path / "data" / "reports",
        tmp_path / "models" / "weights",
        tmp_path / "data" / "batch",
        tmp_path / "data" / "sync",
    }
    for d in expected:
        assert d.is_dir(), f"缺失运行时目录: {d}"
    assert set(created) == expected
