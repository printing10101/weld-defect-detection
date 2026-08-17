"""静态加密测试（§7.5，P0-2 修复验证）。

覆盖：
1. AesCrypto 加解密往返 + 完整性校验（篡改/密钥不符检测）；
2. 影像副本落盘加密：encrypt=True + 密钥 → data/images 下为密文（SDC1 魔数），
   报告生成（_read_gray）能解密读取 → PDF 缺陷图谱正常；
3. 密钥缺失降级：encrypt=True 但无 SCAN_CRYPTO_KEY → 明文落盘 + 日志告警（不崩）。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.infra.crypto import AesCrypto, CryptoIntegrityError, CryptoKeyError

_TEST_KEY = AesCrypto.generate_key()


def _png_bytes() -> bytes:
    img = np.full((60, 80), 128, dtype=np.uint8)
    cv2.circle(img, (30, 30), 10, 60, -1)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


# ---------------------------------------------------------------------------
# 单元：AesCrypto
# ---------------------------------------------------------------------------


def test_encrypt_decrypt_roundtrip() -> None:
    cipher = AesCrypto(bytes.fromhex("00" * 32))
    plaintext = b"film image bytes"
    ct = cipher.encrypt(plaintext)
    assert ct.startswith(b"SDC1")
    assert ct != plaintext
    assert cipher.decrypt(ct) == plaintext


def test_decrypt_tampered_raises() -> None:
    cipher = AesCrypto(bytes.fromhex("11" * 32))
    ct = bytearray(cipher.encrypt(b"secret"))
    ct[-1] ^= 0xFF  # 篡改最后一个字节
    with pytest.raises(CryptoIntegrityError):
        cipher.decrypt(bytes(ct))


def test_decrypt_wrong_key_raises() -> None:
    ct = AesCrypto(bytes.fromhex("22" * 32)).encrypt(b"secret")
    with pytest.raises(CryptoIntegrityError):
        AesCrypto(bytes.fromhex("33" * 32)).decrypt(ct)


def test_env_key_roundtrip(monkeypatch) -> None:
    monkeypatch.setenv("SCAN_CRYPTO_KEY", _TEST_KEY)
    cipher = AesCrypto()
    assert cipher.decrypt(cipher.encrypt(b"x")) == b"x"


def test_missing_key_raises(monkeypatch) -> None:
    monkeypatch.delenv("SCAN_CRYPTO_KEY", raising=False)
    with pytest.raises(CryptoKeyError):
        AesCrypto()


# ---------------------------------------------------------------------------
# 集成：影像副本加密落盘 + 报告解密读取
# ---------------------------------------------------------------------------


def _build_report_with_env(tmp_path: Path, monkeypatch, key: str | None) -> tuple[Path, bytes]:
    """在注入 SCAN_CRYPTO_KEY 的环境下跑一次 report 全链路，返回影像副本路径与 PDF。"""
    from fastapi.testclient import TestClient

    from backend.app import dependencies as deps
    from backend.app.main import app
    from backend.domain.grade.nb47013 import Nb47013Grader
    from backend.domain.standards.tables.loader import load_standard_tables

    # 测试环境隔离（与 conftest 一致，指向独立 tmp 目录）
    monkeypatch.setenv("SCAN_PATHS__DB_PATH", str(tmp_path / "db" / "test.db"))
    monkeypatch.setenv("SCAN_PATHS__IMAGES_DIR", str(tmp_path / "images"))
    monkeypatch.setenv("SCAN_PATHS__REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("SCAN_PATHS__TMP_DIR", str(tmp_path / "tmp"))
    if key is not None:
        monkeypatch.setenv("SCAN_CRYPTO_KEY", key)
    else:
        monkeypatch.delenv("SCAN_CRYPTO_KEY", raising=False)

    deps._registry = None
    # authorized 测试表（须先写盘，grader 装配时读取）。锚定仓库根，避免依赖 pytest CWD。
    src = Path(__file__).resolve().parents[2] / "backend" / "domain" / "standards" / "tables" / "nb47013.yaml"
    text = src.read_text(encoding="utf-8").replace("authorized: false", "authorized: true")
    (tmp_path / "nb.yaml").write_text(text, encoding="utf-8")
    reg = deps.get_registry()
    reg.grader = Nb47013Grader(
        load_standard_tables("NB/T47013.2-2015", filename=str(tmp_path / "nb.yaml"))
    )
    # 放宽黑度 + 关质量门禁（合成底片不过硬门禁）
    orig_low, orig_block = reg.config.density.low, reg.config.quality.block_on_quality
    reg.config.density.low = 0.0
    reg.config.quality.block_on_quality = False
    try:
        with TestClient(app) as client:
            resp = client.post(
                "/api/v1/report",
                files={"image": ("film.png", _png_bytes(), "image/png")},
                data={"pixel_spacing_mm": "0.1", "base_metal_thickness_mm": "20", "force": "true"},
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        image_id = body["image_id"]
        pdf_url = body["pdf_url"]
        with TestClient(app) as client:
            pdf_resp = client.get(pdf_url)
        assert pdf_resp.status_code == 200, pdf_resp.text
        copies = list(Path(tmp_path / "images").glob(f"{image_id}.*"))
        assert copies, "影像副本应已落盘"
        return copies[0], pdf_resp.content
    finally:
        reg.config.density.low = orig_low
        reg.config.quality.block_on_quality = orig_block
        deps._registry = None


def test_persist_encrypts_with_key(tmp_path: Path, monkeypatch) -> None:
    """encrypt=True + 密钥 → 副本为密文（SDC1），报告 PDF 正常生成（解密读取路径有效）。"""
    copy_path, pdf = _build_report_with_env(tmp_path, monkeypatch, key=_TEST_KEY)
    raw = copy_path.read_bytes()
    assert raw.startswith(b"SDC1"), "落盘副本应为 AES 密文"
    assert pdf.startswith(b"%PDF"), "报告 PDF 应正常生成（解密读取路径打通）"
    # 用同一密钥可解密回 PNG
    from backend.infra.crypto import AesCrypto as AC

    monkeypatch.setenv("SCAN_CRYPTO_KEY", _TEST_KEY)
    plain = AC().decrypt(raw)
    assert plain.startswith(b"\x89PNG")


def test_persist_plaintext_without_key(tmp_path: Path, monkeypatch) -> None:
    """encrypt=True 但无密钥 → 明文落盘 + 不崩（桌面单机默认可运行）。"""
    copy_path, _pdf = _build_report_with_env(tmp_path, monkeypatch, key=None)
    raw = copy_path.read_bytes()
    assert raw.startswith(b"\x89PNG"), "无密钥时应明文落盘（降级不崩）"


def test_read_gray_decrypts(tmp_path: Path, monkeypatch) -> None:
    """_read_gray 对密文副本可解密读取（报告图谱数据源）。"""
    from backend.infra.reporting.pdf_reporter import _read_gray

    monkeypatch.setenv("SCAN_CRYPTO_KEY", _TEST_KEY)
    cipher = AesCrypto()
    p = tmp_path / "film.png.enc"
    p.write_bytes(cipher.encrypt(_png_bytes()))
    img = _read_gray(str(p))
    assert img is not None and img.shape == (60, 80)


def test_read_gray_fallback_plaintext(tmp_path: Path, monkeypatch) -> None:
    """明文旧数据（无魔数）直接解码，不受加密逻辑影响。"""
    from backend.infra.reporting.pdf_reporter import _read_gray

    monkeypatch.delenv("SCAN_CRYPTO_KEY", raising=False)
    p = tmp_path / "old.png"
    p.write_bytes(_png_bytes())
    img = _read_gray(str(p))
    assert img is not None and img.shape == (60, 80)
