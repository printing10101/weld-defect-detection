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
    IncrementalSm3,
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


def test_sdc2_ciphertext_byte_compatible_with_gmssl_reference(monkeypatch) -> None:
    """SDC2 密文与 gmssl 参考实现逐字节一致（原生 SM4-CTR/HMAC-SM3 迁移锚点）。

    加密原语已从 gmssl 纯 Python 迁到 cryptography 原生后端（性能：大底片
    归档单张加密数十秒 → 毫秒级）。信封格式、SM4-CTR 密钥流与 HMAC-SM3
    必须逐字节不变——否则存量密文（影像副本/备份）全部不可解。本测试用
    gmssl 原语独立重算密文体与 MAC 作对照（含非块对齐长度，覆盖 CTR 末块）。
    """
    import base64

    from gmssl import func
    from gmssl import sm3 as gm_sm3
    from gmssl.sm4 import SM4_ENCRYPT, CryptSM4

    from backend.infra.crypto import (
        _KDF_MAC,
        _KDF_SM4,
        _MAGIC_SM,
        _hmac_sm3,
        _kdf,
        default_crypto_provider,
    )

    def _ref_sm3_hex(data: bytes) -> str:
        return gm_sm3.sm3_hash(func.bytes_to_list(data))

    def _ref_hmac_sm3(key: bytes, msg: bytes) -> bytes:
        """RFC 2104 结构 HMAC-SM3（gmssl 参考实现，独立于被测模块）。"""
        block = 64
        k = key if len(key) <= block else bytes.fromhex(_ref_sm3_hex(key))
        k = k + b"\x00" * (block - len(k))
        inner = _ref_sm3_hex(bytes(b ^ 0x36 for b in k) + msg)
        outer = _ref_sm3_hex(bytes(b ^ 0x5C for b in k) + bytes.fromhex(inner))
        return bytes.fromhex(outer)

    master = bytes(range(32))
    monkeypatch.setenv("SCAN_CRYPTO_KEY", base64.b64encode(master).decode())
    # 固定 nonce：encrypt() 内部 os.urandom 只用于生成 16B 计数器
    monkeypatch.setattr("backend.infra.crypto.os.urandom", lambda n: bytes(range(n))[::-1])
    provider = default_crypto_provider()

    nonce = bytes(range(16))[::-1]
    plaintext = b"compat-payload" * 37  # 非块对齐（481B），覆盖 CTR 末块部分计数
    ciphertext = provider.encrypt(plaintext)

    sm4_key = _kdf(master, _KDF_SM4, 16)
    sm4 = CryptSM4()
    sm4.set_key(sm4_key, SM4_ENCRYPT)
    counter_base = int.from_bytes(nonce, "big")
    body = bytearray()
    for off in range(0, len(plaintext), 16):
        ctr = ((counter_base + off // 16) & ((1 << 128) - 1)).to_bytes(16, "big")
        ks = bytes(sm4.one_round(sm4.sk, ctr))
        body += bytes(a ^ b for a, b in zip(plaintext[off : off + 16], ks))

    mac_key = _kdf(master, _KDF_MAC, 32)
    expected_mac = _ref_hmac_sm3(mac_key, _MAGIC_SM + nonce + bytes(body))
    assert bytes(body) == ciphertext[len(_MAGIC_SM) + 16 : -32]
    assert ciphertext[-32:] == expected_mac
    # 模块自算 MAC 与参考实现一致（_hmac_sm3 迁移后仍逐字节等值）
    assert _hmac_sm3(mac_key, _MAGIC_SM + nonce + bytes(body)) == expected_mac


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


def test_local_keyfile_migrates_across_devices(monkeypatch, tmp_path: Path) -> None:
    """跨卷密钥迁移：os.replace 抛 EXDEV 时必须退化为复制，且沿用原密钥。

    历史缺陷：源与目标不同卷（安装盘 C: / 用户数据盘 D:）时 ``os.replace``
    抛 WinError 17 → CryptoKeyError → 静态加密 provider 整体不可用 →
    fail-closed **拒绝一切影像副本落盘**，用户看到的是"软件突然不能评片"。
    修复后迁移应成功，且必须是原密钥（重新生成会让既有密文全部不可解）。
    """
    monkeypatch.delenv("SCAN_CRYPTO_KEY", raising=False)
    monkeypatch.delenv("SCAN_CRYPTO_KEY_FILE", raising=False)
    udd = tmp_path / "udd"
    monkeypatch.setenv("SCANDETECTION_USER_DATA_DIR", str(udd))
    cwd = tmp_path / "cwd"
    legacy = cwd / "data" / ".crypto_key"
    legacy.parent.mkdir(parents=True)
    legacy_key = base64.b64encode(bytes.fromhex("11" * 32)) + b"\n"
    legacy.write_bytes(legacy_key)
    monkeypatch.chdir(cwd)

    def _exdev(src, dst):
        raise OSError(17, "系统无法将文件移到不同的磁盘驱动器")

    monkeypatch.setattr(os, "replace", _exdev)

    cipher = AesCrypto()
    target = udd / "data" / ".crypto_key"
    assert target.is_file(), "跨卷时应复制而非放弃迁移"
    assert target.read_bytes() == legacy_key, "必须沿用原密钥（新密钥会让既有密文不可解）"
    assert not legacy.exists(), "迁移完成后源文件应清理"
    blob = cipher.encrypt(b"cross-device")
    assert AesCrypto().decrypt(blob) == b"cross-device"


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
    # lifespan 的后台初始化线程竞态（上一个测试的装配线程
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


# ---------------------------------------------------------------------------
# 流式加密（encrypt_stream）：信封与一次性 encrypt 一致，大文件分块控内存
# ---------------------------------------------------------------------------


def test_incremental_sm3_matches_one_shot() -> None:
    """增量 SM3 与一次性 sm3_hex 等值（覆盖 64 字节块边界与不等长分块）。"""
    import random

    rng = random.Random(20260915)
    for n in (0, 1, 55, 63, 64, 65, 127, 128, 129, 1000, 4096):
        data = rng.randbytes(n)
        h = IncrementalSm3()
        step = max(1, n // 7)  # 不等长分块喂入
        for i in range(0, n, step):
            h.update(data[i : i + step])
        assert h.hexdigest() == sm3_hex(data), f"n={n}"


def test_encrypt_stream_byte_identical_to_encrypt(monkeypatch, tmp_path) -> None:
    """同 nonce 下流式信封与一次性 encrypt 逐字节一致（存量密文互解不变）。

    明文跨多个 1MiB 块，验证 CTR 块序号跨块连续与增量 HMAC 截断正确。
    """
    import io
    import random

    from backend.infra import crypto as crypto_mod

    rng = random.Random(7)
    plain = rng.randbytes((1 << 20) * 2 + 12345)
    cipher = SoftSmProvider(bytes.fromhex("11" * 32))
    fixed = bytes([0xAB] * 16)
    real_urandom = crypto_mod.os.urandom
    monkeypatch.setattr(crypto_mod.os, "urandom", lambda n: fixed if n == 16 else real_urandom(n))
    one_shot = cipher.encrypt(plain, aad=b"film")
    sink = io.BytesIO()
    cipher.encrypt_stream(io.BytesIO(plain), sink, aad=b"film")
    assert sink.getvalue() == one_shot


def test_encrypt_stream_roundtrip_tamper_and_aad() -> None:
    """流式密文解密往返；中部篡改与 aad 不匹配均拒绝（encrypt-then-MAC 语义）。"""
    import io
    import random

    rng = random.Random(3)
    plain = rng.randbytes(70000)  # 跨块
    cipher = SoftSmProvider(bytes.fromhex("22" * 32))
    sink = io.BytesIO()
    cipher.encrypt_stream(io.BytesIO(plain), sink, aad=b"ctx")
    ct = sink.getvalue()
    assert ct.startswith(b"SDC2")
    assert cipher.decrypt(ct, aad=b"ctx") == plain
    tampered = bytearray(ct)
    tampered[len(ct) // 2] ^= 0x01
    with pytest.raises(CryptoIntegrityError):
        cipher.decrypt(bytes(tampered), aad=b"ctx")
    with pytest.raises(CryptoIntegrityError):
        cipher.decrypt(ct, aad=b"other")
