"""国密算法静态加密与签名 provider 层（C-01~C-03 算法内核）。

用途：对落盘的敏感数据（影像副本、报告、导出包）做静态加密，并为审计链/
报告提供 SM3 哈希与 SM2 签名能力。通过 CryptoProvider 协议抽象算法实现，
默认 SoftSmProvider 为纯软件国密实现（基于 gmssl 3.2.2）：
- 静态加密：SM4-CTR + HMAC-SM3 组合认证加密（信封 SDC2）；
- 哈希：SM3（审计哈希链，见 repository.verify_chain）；
- 签名：SM2（SM3withSM2，报告防篡改签名，见 pdf_reporter）。

模式选型说明：gmssl 3.2.2 仅提供 SM4 的 ECB/CBC，无 GCM/CTR。GCM 需要
GF(2^128) 乘法表，gmssl 未实现；故以 SM4-CTR + HMAC-SM3（encrypt-then-MAC）
组合提供等效的 AEAD 属性（机密性 + 完整性 + aad 绑定）。MAC 先于解密校验，
密文被篡改在还原明文之前即被拒绝。

信封格式（首部魔数带版本号，向后兼容解密）：
- SDC2（新，国密）：b"SDC2" || nonce(16B) || ciphertext || mac(HMAC-SM3, 32B)；
  mac 覆盖 magic || nonce || ciphertext || aad；
- SDC1（旧，AES-256-GCM）：保留解密路径用于历史数据；新写入一律 SDC2。

密钥分层（软件 provider）：
- 主密钥来源（优先级从高到低）：
  1) 环境变量 SCAN_CRYPTO_KEY（base64/hex 编码 32 字节）——加固部署模式；
  2) 本地持久密钥文件（SCAN_CRYPTO_KEY_FILE 指定，默认 CWD 相对
     data/.crypto_key）——桌面单机默认模式：首启生成一次并复用，密文
     生命周期与该文件绑定（文件在则密文永久可解；文件丢失即不可解，
     部署须知见下）。与历史 AES 信封共用同一主密钥，保证存量 SDC1 密文仍可解；
- 数据密钥：主密钥经 SM3 域分离 KDF 派生——
    SM4 密钥 = SM3("sd-kdf-sm4" || master) 前 16 字节（SM4-128）；
    MAC 密钥 = SM3("sd-kdf-mac" || master)（32 字节）；
    SM2 私钥 = int(SM3("sd-kdf-sm2" || master)) mod n（确定性派生，公钥由
    曲线点乘计算）；亦可用 SCAN_SM2_PRIVATE_KEY（64 hex）显式指定以便
    独立备份/轮换；
- 对接商密硬件（Pkcs11Provider）后，SM4/SM2 密钥改由硬件模块托管生成并
  保存在硬件内（禁止导出），软件侧仅持句柄，主密钥不再参与数据密钥派生。

性能说明：gmssl 为纯 Python 实现，SM4-CTR 吞吐约 100~200 KB/s（本机实测
4KB 约 26ms），SM2 签名约 10ms/次。适合影像副本、报告等落盘数据的一次性
静态加密；大文件或高并发场景请对接商密密码卡/加速卡（Pkcs11Provider）。

**仍不提供"随机临时密钥"**：进程内随机、重启即丢的密钥会让密文永久不可解，
属于静默的数据丢失。默认模式的本地密钥文件是**持久**的（首启生成一次、
之后复用），静态加密开箱即生效；对保密性有更高要求的部署应设
SCAN_CRYPTO_KEY（或对接 Pkcs11Provider）。部署须知：data/.crypto_key 必须
随 data 目录一并备份、严禁入库/外传；该文件与密文二选一皆不可单独存活。
密钥完全不可用（env 缺失且密钥文件不可读/不可写）时一律抛 CryptoKeyError，
由调用方决定拒绝落盘——绝不静默降级明文。
"""

from __future__ import annotations

import base64
import binascii
import hmac as _hmac
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from gmssl import func, sm2, sm3
from gmssl.sm4 import SM4_ENCRYPT, CryptSM4

_LOG = logging.getLogger(__name__)

_KEY_BYTES = 32  # 主密钥长度（与历史 AES-256 信封共用）
_NONCE_BYTES = 16  # SDC2 CTR 计数器（128bit）
_MAC_BYTES = 32  # HMAC-SM3 输出长度
_MAGIC_SM = b"SDC2"  # ScanDetection Crypto v2：国密 SM4-CTR + HMAC-SM3
_MAGIC_AES = b"SDC1"  # ScanDetection Crypto v1：AES-256-GCM（历史信封）
_ENV_KEY = "SCAN_CRYPTO_KEY"
_ENV_KEY_FILE = "SCAN_CRYPTO_KEY_FILE"  # 本地持久密钥文件路径（可选覆盖）
_ENV_PROVIDER = "SCAN_CRYPTO_PROVIDER"  # soft-sm（默认）| pkcs11
_ENV_SM2_KEY = "SCAN_SM2_PRIVATE_KEY"  # 可选：显式 SM2 私钥（64 hex）
_LOCAL_KEY_FILE_REL = Path("data") / ".crypto_key"  # 默认密钥文件（data/ 相对路径）
_KDF_SM4 = b"sd-kdf-sm4"
_KDF_MAC = b"sd-kdf-mac"
_KDF_SM2 = b"sd-kdf-sm2"
_AES_NONCE_BYTES = 12  # SDC1 GCM nonce（历史格式常量，不可改）


class CryptoKeyError(ValueError):
    """密钥缺失或格式非法。"""


class CryptoIntegrityError(ValueError):
    """密文损坏、被篡改或密钥不匹配。"""


def _decode_key(raw: str) -> bytes:
    """解析 base64 / hex 编码的密钥字符串。"""
    text = raw.strip()
    for decoder in (base64.b64decode, binascii.unhexlify):
        try:
            key = decoder(text)
        except (binascii.Error, ValueError):
            continue
        if len(key) == _KEY_BYTES:
            return key
    raise CryptoKeyError(f"{_ENV_KEY} 必须是 base64 或 hex 编码的 {_KEY_BYTES} 字节密钥")


def local_key_file() -> Path:
    """本地持久主密钥文件路径：SCAN_CRYPTO_KEY_FILE 优先，否则 data/.crypto_key。

    相对路径锚定安装根（与 paths/resolve_config_path 同语义，与 CWD 解耦）；
    Tauri 打包版随 SCANDETECTION_USER_DATA_DIR 改锚到用户数据目录。此前按
    CWD 解析：CWD≠安装根启动会在新位置再生成一枚主密钥，既有密文在该进程内
    全部不可解——已随数据目录统一锚定修复。
    """
    env = os.environ.get(_ENV_KEY_FILE, "").strip()
    if env:
        return Path(env)
    from backend.infra.paths import resolve_data_path

    return resolve_data_path(_LOCAL_KEY_FILE_REL)


def _load_or_create_local_key() -> bytes:
    """读取本地持久密钥文件；缺失则生成一次并写入（0600 尽力而为）。

    与"随机临时密钥"的本质区别：文件持久，密文生命周期与其绑定——
    文件在，密文永久可解；文件丢失 = 数据不可解（须随 data 一并备份，
    模块 docstring 有部署须知）。生成/读取事件都留日志，部署方可见。
    并发首启用 O_EXCL 保证单次生成，竞争失败方回读已写入的文件。
    """
    path = local_key_file()
    if path.is_file():
        try:
            key = _decode_key(path.read_text(encoding="utf-8"))
        except (OSError, CryptoKeyError) as exc:
            raise CryptoKeyError(f"本地密钥文件不可读或内容非法 {path}: {exc}") from None
        _LOG.info("静态加密：已加载本地主密钥文件 %s", path)
        return key
    # 历史布局迁移：旧版本按 CWD 解析密钥文件（CWD≠安装根启动时落在
    # <CWD>/data/.crypto_key）。目标位置缺密钥而旧 CWD 位置有 → 迁移而非
    # 再生成一枚新密钥（新密钥会让既有密文全部不可解）。
    # 仅默认布局参与迁移：SCAN_CRYPTO_KEY_FILE 显式指定新位置时绝不动 CWD
    # 下的旧文件——那可能是真实主密钥，move 走即永久不可解（测试/运维
    # 显式覆盖是常见场景）。
    if not os.environ.get(_ENV_KEY_FILE, "").strip():
        legacy = Path.cwd() / _LOCAL_KEY_FILE_REL
        if legacy.is_file() and legacy.resolve() != path.resolve():
            # 复查目标仍不存在：并发首启可能刚用 O_EXCL 生成新密钥，
            # 此处若覆盖会让对方进程持有失配密钥。
            if path.exists():
                _LOG.warning("目标密钥文件已存在，跳过历史密钥迁移（防覆盖）: %s", path)
            else:
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    legacy.replace(path)
                    _LOG.warning("静态加密：已把历史 CWD 相对密钥文件迁移 %s → %s", legacy, path)
                    return _load_or_create_local_key()
                except OSError as exc:
                    raise CryptoKeyError(f"历史密钥文件迁移失败 {legacy} → {path}: {exc}") from None
    key = os.urandom(_KEY_BYTES)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(base64.b64encode(key) + b"\n")
    except FileExistsError:
        # 并发首启：另一进程已写入密钥文件，回读保证双方一致。
        # 注意 Windows 上父路径是文件时 mkdir 同样抛 FileExistsError——
        # 此时密钥文件并不存在，属路径不可用，须报错而非递归回读。
        if path.is_file():
            return _load_or_create_local_key()
        raise CryptoKeyError(f"本地密钥文件路径不可用 {path}") from None
    except OSError as exc:
        raise CryptoKeyError(f"本地密钥文件不可写 {path}: {exc}") from None
    try:
        os.chmod(path, 0o600)  # Windows ACL 不做强承诺（同 ipc_token 的尽力而为口径）
    except OSError:
        pass
    _LOG.warning(
        "静态加密：已首启生成主密钥文件 %s（须随 data 目录备份；丢失将导致既有密文不可解）",
        path,
    )
    return key


# ---------------------------------------------------------------------------
# SM3 原语（审计哈希链 / KDF / HMAC 共用）
# ---------------------------------------------------------------------------


def sm3_hex(data: bytes) -> str:
    """SM3 摘要（64 hex）。国密哈希统一入口（审计链、KDF）。"""
    return sm3.sm3_hash(func.bytes_to_list(data))


def _hmac_sm3(key: bytes, msg: bytes) -> bytes:
    """HMAC-SM3（RFC 2104 结构，SM3 分组 64 字节）。

    gmssl 未内置 HMAC，按标准结构组合 SM3：H(K'^opad || H(K'^ipad || m))。
    """
    block = 64
    k = key if len(key) <= block else bytes.fromhex(sm3_hex(key))
    k = k + b"\x00" * (block - len(k))
    inner = bytes(b ^ 0x36 for b in k)
    outer = bytes(b ^ 0x5C for b in k)
    return bytes.fromhex(sm3_hex(outer + bytes.fromhex(sm3_hex(inner + msg))))


def _kdf(master: bytes, label: bytes, length: int) -> bytes:
    """密钥派生：SM3(label || master) 截取 length 字节。

    域分离标签防止跨用途密钥重用（SM4/MAC/SM2 各一枚）。label+master
    合计 ≤ 55 字节，落在 SM3 单块内，无需计数器迭代。
    """
    return bytes.fromhex(sm3_hex(label + master))[:length]


# ---------------------------------------------------------------------------
# Provider 抽象
# ---------------------------------------------------------------------------


class CryptoProvider(Protocol):
    """算法抽象层：静态加密 + 哈希 + 签名的统一接口（C-01~C-03）。

    encrypt/decrypt 与历史 AesCrypto 完全同签名（positional 明文 +
    keyword aad），调用方（app/pipelines.py 影像副本落盘、pdf_reporter
    影像读取）无需感知具体算法实现。
    """

    def encrypt(self, plaintext: bytes, *, aad: bytes | None = None) -> bytes:
        """认证加密并封装信封（魔数带算法版本号）。"""
        ...

    def decrypt(self, ciphertext: bytes, *, aad: bytes | None = None) -> bytes:
        """解密并校验完整性；篡改/错钥抛 CryptoIntegrityError。"""
        ...

    @property
    def hash_algo(self) -> str:
        """哈希算法名（审计哈希链/指纹用途，如 "SM3"）。"""
        ...

    @property
    def public_key_hex(self) -> str:
        """SM2 公钥（128 hex = x||y），随签名落 sidecar 供验签方使用。"""
        ...

    def sign(self, data: bytes) -> str:
        """SM2 签名（SM3withSM2，128 hex r||s）。"""
        ...

    def verify(self, data: bytes, signature: str) -> bool:
        """SM2 验签；签名格式非法返回 False 而非抛错。"""
        ...


class SoftSmProvider:
    """纯软件国密 provider（默认）：SM4-CTR+HMAC-SM3 / SM3 / SM2。

    密钥分层见模块 docstring：主密钥来自 SCAN_CRYPTO_KEY（或构造参数），
    数据密钥经 SM3 域分离 KDF 派生。实例无可变状态（SM4 轮密钥固定、
    SM2 密钥只读），线程安全。
    """

    def __init__(self, master_key: bytes | None = None) -> None:
        if master_key is None:
            env = os.environ.get(_ENV_KEY)
            if env:
                master_key = _decode_key(env)
            else:
                # 默认部署模式：本地持久密钥文件（首启生成、之后复用）。
                # env 与密钥文件均不可用时仍抛 CryptoKeyError——由调用方
                # 决定拒绝落盘，绝不静默降级明文（见模块 docstring）。
                master_key = _load_or_create_local_key()
        if len(master_key) != _KEY_BYTES:
            raise CryptoKeyError(f"密钥长度必须为 {_KEY_BYTES} 字节，实得 {len(master_key)}")
        self._master = bytes(master_key)
        # 数据密钥：SM4-128 与 HMAC 密钥（域分离派生，互不相关）
        sm4_key = _kdf(self._master, _KDF_SM4, 16)
        self._mac_key = _kdf(self._master, _KDF_MAC, _MAC_BYTES)
        # SM4 单块加密器：CTR 只用其单块原语 one_round（gmssl 的 crypt_ecb
        # 自带 PKCS7 填充，不适合流式 CTR）
        self._sm4 = CryptSM4()
        self._sm4.set_key(sm4_key, SM4_ENCRYPT)
        # 历史 SDC1（AES-256-GCM）解密路径：主密钥即历史 AES 密钥
        self._aes = AESGCM(self._master)
        # SM2 签名密钥对：显式私钥（合规备份/轮换）优先，否则主密钥派生
        env_d = os.environ.get(_ENV_SM2_KEY, "").strip()
        if env_d:
            d_int = int.from_bytes(binascii.unhexlify(env_d), "big") % int(
                sm2.default_ecc_table["n"], 16
            )
        else:
            d_int = int.from_bytes(_kdf(self._master, _KDF_SM2, 32), "big") % int(
                sm2.default_ecc_table["n"], 16
            )
        if d_int < 1:
            raise CryptoKeyError("SM2 私钥派生结果非法（d≡0 mod n），请更换主密钥")
        d_hex = f"{d_int:064x}"
        # 公钥 Q = d·G。gmssl 未公开点乘 API，此处使用其内部 _kg（曲线表即
        # sm2p256v1 标准参数）；正确性由测试中 SM2 签名/验签往返锚定。
        pub = sm2.CryptSM2(private_key=d_hex, public_key="")._kg(d_int, sm2.default_ecc_table["g"])
        self._sm2_pub = pub
        self._sm2 = sm2.CryptSM2(private_key=d_hex, public_key=pub)

    @staticmethod
    def generate_key() -> str:
        """生成一枚新主密钥（base64 文本），供部署时写入 SCAN_CRYPTO_KEY。"""
        return base64.b64encode(os.urandom(_KEY_BYTES)).decode("ascii")

    @property
    def hash_algo(self) -> str:
        return "SM3"

    @property
    def public_key_hex(self) -> str:
        return self._sm2_pub  # type: ignore[return-value]  # __init__ 中恒已赋值

    # ---- 静态加密（SM4-CTR + HMAC-SM3）----

    def _ctr_xor(self, nonce: bytes, data: bytes) -> bytes:
        """SM4-CTR：128bit 计数器大端递增，密钥流 = SM4(counter)。

        XOR 对合，加解密同函数。gmssl 纯 Python 单块约 1.6μs/字节量级，
        大数据量耗时见模块 docstring 性能说明。
        """
        base = int.from_bytes(nonce, "big")
        out = bytearray()
        for off in range(0, len(data), 16):
            ctr_block = ((base + off // 16) & ((1 << 128) - 1)).to_bytes(16, "big")
            ks = bytes(self._sm4.one_round(self._sm4.sk, ctr_block))
            out += bytes(a ^ b for a, b in zip(data[off : off + 16], ks))
        return bytes(out)

    def encrypt(self, plaintext: bytes, *, aad: bytes | None = None) -> bytes:
        """SM4-CTR 加密 + HMAC-SM3（encrypt-then-MAC），信封 SDC2。

        aad 为可选附加认证数据（不加密但参与 MAC 校验）。
        """
        nonce = os.urandom(_NONCE_BYTES)
        ciphertext = self._ctr_xor(nonce, plaintext)
        mac = _hmac_sm3(self._mac_key, _MAGIC_SM + nonce + ciphertext + (aad or b""))
        return _MAGIC_SM + nonce + ciphertext + mac

    def decrypt(self, ciphertext: bytes, *, aad: bytes | None = None) -> bytes:
        """按信封魔数分流：SDC2 走国密，SDC1 走历史 AES-GCM（只读兼容）。

        完整性校验失败（篡改/错钥/截断）抛 CryptoIntegrityError。
        """
        if ciphertext.startswith(_MAGIC_AES):
            return self._decrypt_legacy_aes(ciphertext, aad=aad)
        head = len(_MAGIC_SM) + _NONCE_BYTES
        if len(ciphertext) < head + _MAC_BYTES or not ciphertext.startswith(_MAGIC_SM):
            raise CryptoIntegrityError("密文头部非法（非本模块产出或已损坏）")
        nonce = ciphertext[len(_MAGIC_SM) : head]
        body = ciphertext[head:-_MAC_BYTES]
        mac = ciphertext[-_MAC_BYTES:]
        expect = _hmac_sm3(self._mac_key, _MAGIC_SM + nonce + body + (aad or b""))
        if not _hmac.compare_digest(mac, expect):
            raise CryptoIntegrityError("完整性校验失败：密文被篡改或密钥不匹配")
        return self._ctr_xor(nonce, body)

    def _decrypt_legacy_aes(self, ciphertext: bytes, *, aad: bytes | None) -> bytes:
        """历史信封 SDC1（AES-256-GCM）解密，仅用于国密化前的存量数据。"""
        head = len(_MAGIC_AES) + _AES_NONCE_BYTES
        if len(ciphertext) <= head or not ciphertext.startswith(_MAGIC_AES):
            raise CryptoIntegrityError("密文头部非法（非本模块产出或已损坏）")
        nonce = ciphertext[len(_MAGIC_AES) : head]
        try:
            return self._aes.decrypt(nonce, ciphertext[head:], aad)
        except InvalidTag as exc:
            raise CryptoIntegrityError("完整性校验失败：密文被篡改或密钥不匹配") from exc

    # ---- SM2 签名 ----

    def sign(self, data: bytes) -> str:
        """SM2 签名（SM3withSM2，128 hex r||s）。

        每次签名的随机数 K 不同，签名值不可复现但均可验。
        """
        for _ in range(8):  # gmssl 在 r/s 越界（概率≈0）时返回 None，重试即可
            sig = self._sm2.sign_with_sm3(data)
            if sig:
                return sig
        raise CryptoIntegrityError("SM2 签名失败：随机数反复越界（环境异常）")

    def verify(self, data: bytes, signature: str) -> bool:
        """SM2 验签。签名格式非法/验签不通过一律 False，不抛错。"""
        try:
            return bool(self._sm2.verify_with_sm3(signature, data))
        except (ValueError, TypeError, KeyError):
            return False


class Pkcs11Provider:
    """PKCS#11 商密硬件 provider 骨架（预留，本环境无硬件、未经真机验证）。

    对接经商用密码认证的硬件（PCI-E 密码卡 / 加密机 / USB-Key，符合
    GM/T 0018 密码设备应用接口规范并向上提供 PKCS#11 动态库）时启用：
    SCAN_CRYPTO_PROVIDER=pkcs11 + SCAN_PKCS11_LIBRARY（厂商动态库路径），
    可选 SCAN_PKCS11_SLOT（槽位/令牌序列号）与 SCAN_PKCS11_PIN。

    密钥由硬件托管：SM4 数据密钥与 SM2 签名密钥对在硬件内生成并持久化
    （CKA_TOKEN），标记 CKA_SENSITIVE 禁止导出，软件侧仅持对象句柄；
    主密钥/明文密钥不落软件侧，SCAN_CRYPTO_KEY 对本 provider 不参与
    数据密钥派生。

    实现状态（诚实声明）：本骨架给出完整的接口与调用链（加载库 → 定位
    槽位令牌 → 登录会话 → 按标签定位密钥对象 → 调用 CK_ 原语），未配置
    即抛带指引的 CryptoKeyError；但具体机制号（如 CKM_SM4_GCM）随厂商
    固件能力而异，部署时需按厂商文档核对，未经过真机验证。
    """

    _LIB_ENV = "SCAN_PKCS11_LIBRARY"  # 厂商 PKCS#11 动态库路径
    _SLOT_ENV = "SCAN_PKCS11_SLOT"  # 槽位/令牌序列号或标签
    _PIN_ENV = "SCAN_PKCS11_PIN"  # 用户 PIN（定位私钥需登录）
    _KEY_LABEL = "scandetection"  # 硬件内密钥对象标签（CKA_LABEL）

    def __init__(
        self,
        library_path: str | None = None,
        slot: str | None = None,
        pin: str | None = None,
    ) -> None:
        self._library_path = library_path or os.environ.get(self._LIB_ENV, "").strip()
        self._slot = slot or os.environ.get(self._SLOT_ENV, "").strip() or None
        self._pin = pin or os.environ.get(self._PIN_ENV, "").strip() or None
        if not self._library_path:
            raise CryptoKeyError(
                "PKCS#11 provider 未配置：请设置 "
                f"{self._LIB_ENV}（商密硬件厂商 PKCS#11 动态库路径）后重试；"
                "无硬件环境请使用默认 soft-sm provider"
            )
        self._lib = None  # 延迟加载（进程内仅加载一次厂商动态库）

    def _session(self):
        """加载 PKCS#11 库 → 定位槽位令牌 → 打开并登录会话。

        SM4/SM2 各操作共用此调用链；密钥对象在硬件内，句柄按需查找。
        """
        if self._lib is None:
            try:
                import pkcs11  # type: ignore  # python-pkcs11 绑定，可选硬件依赖
            except ImportError as exc:
                raise CryptoKeyError(
                    "PKCS#11 运行时缺失：需安装 python-pkcs11 并配置厂商动态库"
                ) from exc
            try:
                self._lib = pkcs11.lib(self._library_path)
            except Exception as exc:
                raise CryptoKeyError(f"PKCS#11 库加载失败：{self._library_path}（{exc}）") from exc
        token = None
        for slot in self._lib.get_slots():
            info = slot.get_token()
            if self._slot in (getattr(info, "serial", None), getattr(info, "label", None)):
                token = info
                break
        if token is None:
            raise CryptoKeyError(f"未找到 PKCS#11 槽位/令牌：{self._slot or '<未指定>'}")
        session = token.open()
        if self._pin:
            session.login(self._pin)
        return session

    def _find_key(self, session, *, private: bool):
        """按标签定位硬件内密钥对象（私钥需已登录会话）。"""
        from pkcs11 import ObjectClass  # type: ignore  # 可选硬件依赖

        template = {
            "CKA_LABEL": self._KEY_LABEL,
            "CKA_CLASS": ObjectClass.PRIVATE_KEY if private else ObjectClass.SECRET_KEY,
        }
        for obj in session.get_objects(template):
            return obj
        kind = "SM2 私钥" if private else "SM4 密钥"
        raise CryptoKeyError(f"硬件内未找到 {kind} 对象（CKA_LABEL={self._KEY_LABEL}）")

    # ---- 接口实现（与 CryptoProvider 对齐）----

    @property
    def hash_algo(self) -> str:
        return "SM3"

    @property
    def public_key_hex(self) -> str:
        """SM2 公钥：从硬件读取 EC 点（CKA_EC_POINT，非压缩 x||y）。"""
        from pkcs11 import ObjectClass  # type: ignore  # 可选硬件依赖

        session = self._session()
        template = {"CKA_LABEL": self._KEY_LABEL, "CKA_CLASS": ObjectClass.PUBLIC_KEY}
        for obj in session.get_objects(template):
            return self._read_ec_point(obj)
        raise CryptoKeyError(f"硬件内未找到 SM2 公钥对象（CKA_LABEL={self._KEY_LABEL}）")

    def _read_ec_point(self, obj) -> str:
        """读取公钥 EC 点并转 128 hex（x||y）。具体属性访问随绑定库版本而定。"""
        point = obj["CKA_EC_POINT"]
        raw = bytes(point)
        if raw.startswith(b"\x04") and len(raw) == 65:  # 非压缩点：0x04 || x(32) || y(32)
            raw = raw[1:]
        if len(raw) != 64:
            raise CryptoKeyError(f"SM2 公钥点长度异常：{len(raw)} 字节")
        return raw.hex()

    def encrypt(self, plaintext: bytes, *, aad: bytes | None = None) -> bytes:
        """静态加密：优先 CKM_SM4_GCM；硬件不支持时回退 CKM_SM4_CTR +
        CKM_SM3-HMAC（与本模块软实现同构的信封），由厂商固件能力决定。
        """
        # 骨架：机制号选择、nonce/信封封装需按厂商文档定稿后启用（未真机验证）。
        self._find_key(self._session(), private=False)
        raise CryptoKeyError("PKCS#11 静态加密未在真机上验证启用：请先完成厂商对接联调")

    def decrypt(self, ciphertext: bytes, *, aad: bytes | None = None) -> bytes:
        """静态解密（含硬件侧完整性校验；机制选择同 encrypt）。"""
        self._find_key(self._session(), private=False)
        raise CryptoKeyError("PKCS#11 静态解密未在真机上验证启用：请先完成厂商对接联调")

    def sign(self, data: bytes) -> str:
        """SM2 签名：硬件内私钥执行（SM3withSM2），返回 128 hex r||s。"""
        session = self._session()
        key = self._find_key(session, private=True)
        sig = key.sign(data)  # 机制 CKM_SM2_SIGN_SM3 由密钥对象能力决定
        return bytes(sig).hex()

    def verify(self, data: bytes, signature: str) -> bool:
        """SM2 验签：硬件内公钥执行。格式非法/不通过一律 False。"""
        try:
            session = self._session()
            for obj in session.get_objects({"CKA_LABEL": self._KEY_LABEL}):
                obj.verify(bytes.fromhex(signature), data)
                return True
            return False
        except Exception:  # noqa: BLE001  # 验签失败（格式/硬件错误）统一 False，不抛错
            return False


def get_provider(master_key: bytes | None = None) -> CryptoProvider:
    """按 SCAN_CRYPTO_PROVIDER 装配算法实现（默认 soft-sm）。

    - soft-sm：纯软件国密（默认，主密钥来自 SCAN_CRYPTO_KEY）；
    - pkcs11：商密硬件（需 SCAN_PKCS11_LIBRARY，见 Pkcs11Provider）。
    """
    name = os.environ.get(_ENV_PROVIDER, "soft-sm").strip().lower()
    if name in ("", "soft-sm", "soft"):
        return SoftSmProvider(master_key)
    if name in ("pkcs11", "hsm"):
        return Pkcs11Provider()
    raise CryptoKeyError(
        f"未知 crypto provider：{name}（可选 soft-sm | pkcs11，经 {_ENV_PROVIDER} 切换）"
    )


class AesCrypto(SoftSmProvider):
    """向后兼容入口（保留历史类名，调用方 app/pipelines.py、pdf_reporter.py
    无需改动）。

    历史上本类为 AES-256-GCM（SDC1 信封）。国密化后 encrypt 委托
    SoftSmProvider 输出 SDC2（SM4）信封——新写入一律国密；decrypt 按信封
    魔数自动分流，存量 SDC1 AES 信封用同一主密钥（SCAN_CRYPTO_KEY）仍可
    解。构造参数与 generate_key 语义不变。
    """


@lru_cache(maxsize=1)
def _cached_default_provider(
    env_key: str | None, env_key_file: str | None, env_sm2_key: str | None, env_provider: str | None
) -> SoftSmProvider:
    """按密钥来源环境变量缓存 provider 实例（key 参数仅作缓存键使用）。

    provider 无可变状态且线程安全（SoftSmProvider docstring），进程内复用
    免去每张影像/每份报告重复执行 SM2 公钥点乘与 SM3 KDF（纯 Python，每次
    数十 ms）。以环境变量四元组为缓存键：测试/运维在同一进程内切换密钥
    来源时自动得到新实例，与"每次新建"行为等价。
    """
    return AesCrypto()


def default_crypto_provider() -> SoftSmProvider:
    """进程内共享的默认 provider（按 SCAN_CRYPTO_* 环境变量缓存）。

    所有"用环境默认密钥落盘/解密"的调用方（影像副本、报告、DICOM 解密）
    统一经此取实例；需要自定义主密钥的场景仍直接构造 AesCrypto(key)。
    """
    return _cached_default_provider(
        os.environ.get(_ENV_KEY),
        os.environ.get(_ENV_KEY_FILE),
        os.environ.get(_ENV_SM2_KEY),
        os.environ.get(_ENV_PROVIDER),
    )


# ---------------------------------------------------------------------------
# SM2 账号凭据工具（C-06/C-07 三员登录）：与 provider 实例无关的纯函数。
# 私钥格式统一为 64 hex（gmssl 约定）；公钥为 128 hex（x||y）。
# ---------------------------------------------------------------------------


def sm2_generate_keypair() -> tuple[str, str]:
    """生成一对 SM2 密钥（private_key 64 hex, public_key 128 hex）。

    供系统管理员为账号签发软证书：私钥一次性交予本人保存（不出系统日志/审计），
    公钥登记在账号上用于登录验签。
    """
    d = int.from_bytes(os.urandom(32), "big") % int(sm2.default_ecc_table["n"], 16)
    if d < 1:  # 概率≈0，防御性重试
        return sm2_generate_keypair()
    d_hex = f"{d:064x}"
    return d_hex, sm2_public_from_private(d_hex)


def sm2_public_from_private(d_hex: str) -> str:
    """由私钥计算 SM2 公钥（Q = d·G，128 hex x||y）。"""
    d_int = int(d_hex, 16)
    pub = sm2.CryptSM2(private_key=d_hex, public_key="")._kg(d_int, sm2.default_ecc_table["g"])
    if not pub:
        raise CryptoKeyError("SM2 公钥派生结果为空（私钥 d 非法）")
    return pub


def sm2_sign_with_private(d_hex: str, data: bytes) -> str:
    """用给定私钥做 SM2 签名（SM3withSM2，128 hex r||s）。

    用途：登录挑战-响应的服务端签名代理（前端不碰密码学）与 UKey 联调对照。
    注意：gmssl 的 sign_with_sm3 计算 Z 值时使用实例上的公钥，故必须先由
    私钥推出公钥一并注入（否则验签方 Z 不一致恒失败）。
    """
    pub = sm2_public_from_private(d_hex)
    signer = sm2.CryptSM2(private_key=d_hex, public_key=pub)
    for _ in range(8):
        sig = signer.sign_with_sm3(data)
        if sig:
            return sig
    raise CryptoIntegrityError("SM2 签名失败：随机数反复越界（环境异常）")


def sm2_verify_with_public(public_key_hex: str, data: bytes, signature: str) -> bool:
    """用账号登记的公钥验签。格式非法/验签不通过一律 False，不抛错。"""
    try:
        verifier = sm2.CryptSM2(private_key="", public_key=public_key_hex)
        return bool(verifier.verify_with_sm3(signature, data))
    except (ValueError, TypeError, KeyError):
        return False
