"""静态加密与国密算法测试。

覆盖：
1. 默认 provider（SM4-CTR + HMAC-SM3，信封 SDC2）加解密往返 + 完整性校验
   （篡改/错钥/aad 检测）；SM3 已知向量；SM2 签名验签往返/篡改检测；
2. 历史 AES 信封兼容：SDC1（AES-256-GCM）旧密文仍可解，新写入一律 SDC2；
3. provider 切换：soft-sm 默认 / pkcs11 未配置抛明确错误 / 未知名报错；
4. 影像副本落盘加密：encrypt=True + 密钥 → data/images 下为国密密文
   （SDC2 魔数），报告生成（_read_gray）能解密读取 → PDF 缺陷图谱正常；
5. 密钥缺失降级：encrypt=True 但无 SCAN_CRYPTO_KEY → 明文落盘 + 日志告警（不崩）。

注：gmssl 纯 Python 实现较慢，测试数据均为小块。
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from backend.infra.crypto import (
    AesCrypto,
    CryptoIntegrityError,
    CryptoKeyError,
    SoftSmProvider,
    get_provider,
    sm3_hex,
)

_TEST_KEY = AesCrypto.generate_key()


def _png_bytes() -> bytes:
    img = np.full((60, 80), 128, dtype=np.uint8)
    cv2.circle(img, (30, 30), 10, 60, -1)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _legacy_sdc1(key: bytes, plaintext: bytes, aad: bytes | None = None) -> bytes:
    """按历史格式构造 SDC1（AES-256-GCM）信封，模拟国密化前的存量密文。"""
    nonce = os.urandom(12)
    return b"SDC1" + nonce + AESGCM(key).encrypt(nonce, plaintext, aad)


# ---------------------------------------------------------------------------
# 单元：SM3 / SM4 信封（默认 provider，SDC2）
# ---------------------------------------------------------------------------


def test_sm3_known_vector() -> None:
    """SM3 标准测试向量（GB/T 32905-2016："abc"）。"""
    assert sm3_hex(b"abc") == "66c7f0f462eeedd9d1f2d46bdc10e4e24167c4875cf2f7a2297da02b8f4ba8e0"


def test_encrypt_decrypt_roundtrip() -> None:
    cipher = AesCrypto(bytes.fromhex("00" * 32))
    plaintext = b"film image bytes"
    ct = cipher.encrypt(plaintext)
    assert ct.startswith(b"SDC2"), "新写入应为国密 SDC2 信封"
    assert ct != plaintext
    assert cipher.decrypt(ct) == plaintext


def test_sm_provider_roundtrip_with_aad() -> None:
    cipher = SoftSmProvider(bytes.fromhex("00" * 32))
    ct = cipher.encrypt(b"payload", aad=b"header-1")
    assert cipher.decrypt(ct, aad=b"header-1") == b"payload"
    with pytest.raises(CryptoIntegrityError):
        cipher.decrypt(ct, aad=b"header-2")  # aad 参与校验，不一致即拒


def test_sm_tamper_detected() -> None:
    cipher = SoftSmProvider(bytes.fromhex("11" * 32))
    ct = bytearray(cipher.encrypt(b"secret"))
    ct[25] ^= 0xFF  # 篡改密文体
    with pytest.raises(CryptoIntegrityError):
        cipher.decrypt(bytes(ct))
    ct2 = bytearray(cipher.encrypt(b"secret"))
    ct2[-1] ^= 0xFF  # 篡改 MAC 尾字节
    with pytest.raises(CryptoIntegrityError):
        cipher.decrypt(bytes(ct2))


def test_sm_wrong_key_raises() -> None:
    ct = SoftSmProvider(bytes.fromhex("22" * 32)).encrypt(b"secret")
    with pytest.raises(CryptoIntegrityError):
        SoftSmProvider(bytes.fromhex("33" * 32)).decrypt(ct)


def test_legacy_sdc1_envelope_still_decryptable() -> None:
    """旧 AES（SDC1）信封兼容：同一主密钥下历史密文仍可解，新写入已换国密。"""
    key = bytes(range(32))
    legacy = _legacy_sdc1(key, b"old aes data")
    cipher = SoftSmProvider(master_key=key)
    assert cipher.decrypt(legacy) == b"old aes data"
    new = cipher.encrypt(b"new data")
    assert new.startswith(b"SDC2") and cipher.decrypt(new) == b"new data"


def test_legacy_sdc1_wrong_key_raises() -> None:
    legacy = _legacy_sdc1(bytes.fromhex("44" * 32), b"x")
    with pytest.raises(CryptoIntegrityError):
        SoftSmProvider(bytes.fromhex("45" * 32)).decrypt(legacy)


# ---------------------------------------------------------------------------
# 单元：SM2 签名 / 验签
# ---------------------------------------------------------------------------


def test_sm2_sign_verify_roundtrip() -> None:
    provider = SoftSmProvider(bytes.fromhex("55" * 32))
    sig = provider.sign(b"report fingerprint")
    assert len(sig) == 128  # r||s 各 64 hex
    assert provider.verify(b"report fingerprint", sig) is True


def test_sm2_tamper_detected() -> None:
    provider = SoftSmProvider(bytes.fromhex("66" * 32))
    sig = provider.sign(b"report fingerprint")
    assert provider.verify(b"tampered fingerprint", sig) is False
    assert provider.verify(b"report fingerprint", "00" * 128) is False
    assert provider.verify(b"report fingerprint", "zz-not-hex") is False  # 格式非法不抛错


def test_sm2_key_derivation_deterministic() -> None:
    """同主密钥派生同一 SM2 密钥对（跨实例可互验）；不同主密钥互不相关。"""
    p1 = SoftSmProvider(bytes.fromhex("77" * 32))
    p2 = SoftSmProvider(bytes.fromhex("77" * 32))
    p3 = SoftSmProvider(bytes.fromhex("78" * 32))
    assert p1.public_key_hex == p2.public_key_hex and len(p1.public_key_hex) == 128
    assert p1.public_key_hex != p3.public_key_hex
    sig = p1.sign(b"msg")
    assert p2.verify(b"msg", sig) is True
    assert p3.verify(b"msg", sig) is False


def test_sm2_explicit_private_key_env(monkeypatch) -> None:
    """SCAN_SM2_PRIVATE_KEY 显式指定私钥（合规备份/轮换路径）。"""
    d = "3945208f7b2144b13f36e38ac6d39f95889393692860b51a42fb81ef4df7c5b8"  # 标准文档示例私钥
    monkeypatch.setenv("SCAN_SM2_PRIVATE_KEY", d)
    p1 = SoftSmProvider(bytes.fromhex("00" * 32))
    p2 = SoftSmProvider(bytes.fromhex("ff" * 32))
    assert p1.public_key_hex == p2.public_key_hex, "显式私钥优先于主密钥派生"


# ---------------------------------------------------------------------------
# 单元：provider 抽象与切换
# ---------------------------------------------------------------------------


def test_provider_default_is_soft_sm(monkeypatch) -> None:
    monkeypatch.setenv("SCAN_CRYPTO_KEY", _TEST_KEY)
    provider = get_provider()
    assert provider.hash_algo == "SM3"
    assert isinstance(provider, SoftSmProvider)


def test_provider_pkcs11_unconfigured_raises(monkeypatch) -> None:
    monkeypatch.setenv("SCAN_CRYPTO_PROVIDER", "pkcs11")
    monkeypatch.delenv("SCAN_PKCS11_LIBRARY", raising=False)
    with pytest.raises(CryptoKeyError, match="PKCS11|PKCS#11|SCAN_PKCS11_LIBRARY"):
        get_provider()


def test_provider_unknown_name_raises(monkeypatch) -> None:
    monkeypatch.setenv("SCAN_CRYPTO_PROVIDER", "bogus")
    with pytest.raises(CryptoKeyError):
        get_provider()


def test_env_key_roundtrip(monkeypatch) -> None:
    monkeypatch.setenv("SCAN_CRYPTO_KEY", _TEST_KEY)
    cipher = AesCrypto()
    assert cipher.decrypt(cipher.encrypt(b"x")) == b"x"


def test_missing_key_raises(monkeypatch, tmp_path: Path) -> None:
    """env 密钥缺失且本地密钥文件不可写（加固部署口径）→ 拒绝并抛 CryptoKeyError。"""
    monkeypatch.delenv("SCAN_CRYPTO_KEY", raising=False)
    # 父路径是一个文件 → mkdir 必败 → 密钥文件不可用
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir", encoding="utf-8")
    monkeypatch.setenv("SCAN_CRYPTO_KEY_FILE", str(blocker / ".crypto_key"))
    with pytest.raises(CryptoKeyError):
        AesCrypto()


def test_local_keyfile_created_and_reused(monkeypatch, tmp_path: Path) -> None:
    """默认部署模式：本地密钥文件首启生成、进程/重启后复用（密文跨实例可解）。"""
    monkeypatch.delenv("SCAN_CRYPTO_KEY", raising=False)
    key_file = tmp_path / "data" / ".crypto_key"
    monkeypatch.setenv("SCAN_CRYPTO_KEY_FILE", str(key_file))
    assert not key_file.exists()
    cipher1 = AesCrypto()
    assert key_file.is_file(), "首启应生成持久密钥文件"
    blob = cipher1.encrypt("跨实例解密".encode())
    # 模拟进程重启：全新实例从同一文件加载密钥
    cipher2 = AesCrypto()
    assert cipher2.decrypt(blob) == "跨实例解密".encode()


def test_local_keyfile_unwritable_raises(monkeypatch, tmp_path: Path) -> None:
    """密钥文件父目录不可创建 → CryptoKeyError（绝不静默降级）。"""
    monkeypatch.delenv("SCAN_CRYPTO_KEY", raising=False)
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir", encoding="utf-8")
    monkeypatch.setenv("SCAN_CRYPTO_KEY_FILE", str(blocker / "data" / ".crypto_key"))
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
    src = (
        Path(__file__).resolve().parents[2]
        / "backend"
        / "domain"
        / "standards"
        / "tables"
        / "nb47013.yaml"
    )
    text = src.read_text(encoding="utf-8").replace("authorized: false", "authorized: true")
    (tmp_path / "nb.yaml").write_text(text, encoding="utf-8")
    for sub in ("db", "images", "reports", "tmp"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    # 同步自建 Registry 并注入：经由环境变量 + 全局懒加载的间接链路会与
    # lifespan 的后台初始化线程竞态（CI Linux 实测踩中——上一个测试的装配线程
    # 可能在本测试重置之后才完成构建，把 conftest 配置的 registry 塞回全局）。
    # 自建实例单线程赋值，时序确定。
    deps._registry = None
    reg = deps.Registry()
    deps._registry = reg
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
        images_dir_env = os.environ.get("SCAN_PATHS__IMAGES_DIR", "<unset>")
        listed = sorted(p.name for p in Path(tmp_path).rglob("*") if p.is_file())[:20]
        copies = list(Path(tmp_path / "images").glob(f"{image_id}.*"))
        assert copies, (
            f"影像副本应已落盘：images_dir_env={images_dir_env} "
            f"cfg_images_dir={deps.get_registry().config.paths.images_dir} "
            f"tmp_files={listed}"
        )
        return copies[0], pdf_resp.content
    finally:
        reg.config.density.low = orig_low
        reg.config.quality.block_on_quality = orig_block
        deps._registry = None


def test_persist_encrypts_with_key(tmp_path: Path, monkeypatch) -> None:
    """encrypt=True + 密钥 → 副本为国密密文（SDC2），报告 PDF 正常生成（解密读取路径有效）。"""
    copy_path, pdf = _build_report_with_env(tmp_path, monkeypatch, key=_TEST_KEY)
    raw = copy_path.read_bytes()
    assert raw.startswith(b"SDC2"), "落盘副本应为国密 SM4 密文"
    assert pdf.startswith(b"%PDF"), "报告 PDF 应正常生成（解密读取路径打通）"
    # 用同一密钥可解密回 PNG
    from backend.infra.crypto import AesCrypto as AC

    monkeypatch.setenv("SCAN_CRYPTO_KEY", _TEST_KEY)
    plain = AC().decrypt(raw)
    assert plain.startswith(b"\x89PNG")


def test_persist_reads_legacy_sdc1_copy(tmp_path: Path, monkeypatch) -> None:
    """存量 AES（SDC1）影像副本：_read_gray 仍可解密读取（历史数据兼容）。"""
    from backend.infra.reporting.pdf_reporter import _read_gray

    monkeypatch.setenv("SCAN_CRYPTO_KEY", _TEST_KEY)
    legacy = _legacy_sdc1(base64.b64decode(_TEST_KEY), _png_bytes())
    p = tmp_path / "film.old.enc"
    p.write_bytes(legacy)
    img = _read_gray(str(p))
    assert img is not None and img.shape == (60, 80)


def test_persist_encrypts_via_local_keyfile(tmp_path: Path, monkeypatch) -> None:
    """encrypt=True 且未设 env 密钥 → 本地持久密钥文件兜底，副本仍为密文（默认部署开箱即加密）。"""
    copy_path, _pdf = _build_report_with_env(tmp_path, monkeypatch, key=None)
    raw = copy_path.read_bytes()
    assert raw.startswith(b"SDC2"), "无 env 密钥时应经本地密钥文件加密落盘"


def test_persist_refuses_plaintext_when_key_unavailable(tmp_path: Path, monkeypatch) -> None:
    """密钥完全不可用（env 缺失 + 密钥文件不可写）→ 拒绝明文落盘，入库报错而非降级。"""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir", encoding="utf-8")
    monkeypatch.setenv("SCAN_CRYPTO_KEY_FILE", str(blocker / ".crypto_key"))
    with pytest.raises(Exception):  # noqa: B017 - CryptoKeyError 经 FastAPI 转译，链路层只需"非明文落盘"
        _build_report_with_env(tmp_path, monkeypatch, key=None)
    copies = list((tmp_path / "images").glob("*")) if (tmp_path / "images").exists() else []
    assert not [p for p in copies if p.read_bytes().startswith(b"\x89PNG")], (
        "密钥不可用时绝不允许明文副本落盘"
    )


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
