"""AES-256-GCM 静态加密。

用途：对落盘的敏感数据（影像副本、报告、导出包）做静态加密。选择 GCM 而非
CBC，因为它同时提供机密性与完整性（AEAD），可检出密文被篡改——这对需要
可追溯、防篡改的检测报告是必需属性。

密钥来源（按优先级）：
1. 构造参数 key（32 字节）；
2. 环境变量 SCAN_CRYPTO_KEY（base64 或 hex 编码的 32 字节）。

**不提供"找不到密钥就随机生成"的行为**：随机临时密钥会让密文在进程重启后
永久不可解，属于静默的数据丢失。密钥缺失一律抛 CryptoKeyError，由部署方
显式用 generate_key 生成并妥善保管。

密文格式：b"SDC1" || nonce(12B) || ciphertext||tag。首部魔数带版本号，便于
后续更换算法时保持向后兼容解密。
"""

from __future__ import annotations

import base64
import binascii
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_KEY_BYTES = 32  # AES-256
_NONCE_BYTES = 12  # GCM 推荐 96bit
_MAGIC = b"SDC1"  # ScanDetection Crypto v1
_ENV_KEY = "SCAN_CRYPTO_KEY"


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


class AesCrypto:
    """AES-256-GCM 加解密器（线程安全：AESGCM 实例无可变状态）。"""

    def __init__(self, key: bytes | None = None) -> None:
        if key is None:
            env = os.environ.get(_ENV_KEY)
            if not env:
                raise CryptoKeyError(
                    f"未提供密钥：请设置环境变量 {_ENV_KEY}（可用 AesCrypto.generate_key() 生成）"
                )
            key = _decode_key(env)
        if len(key) != _KEY_BYTES:
            raise CryptoKeyError(f"密钥长度必须为 {_KEY_BYTES} 字节，实得 {len(key)}")
        self._key = bytes(key)
        self._aead = AESGCM(self._key)

    @staticmethod
    def generate_key() -> str:
        """生成一枚新密钥（base64 文本），供部署时写入 SCAN_CRYPTO_KEY。"""
        return base64.b64encode(os.urandom(_KEY_BYTES)).decode("ascii")

    def encrypt(self, plaintext: bytes, *, aad: bytes | None = None) -> bytes:
        """加密并附完整性标签。aad 为可选的附加认证数据（不加密但参与校验）。"""
        nonce = os.urandom(_NONCE_BYTES)
        return _MAGIC + nonce + self._aead.encrypt(nonce, plaintext, aad)

    def decrypt(self, ciphertext: bytes, *, aad: bytes | None = None) -> bytes:
        """解密并校验完整性；密文被改动或密钥不符时抛 CryptoIntegrityError。"""
        head = len(_MAGIC) + _NONCE_BYTES
        if len(ciphertext) <= head or not ciphertext.startswith(_MAGIC):
            raise CryptoIntegrityError("密文头部非法（非本模块产出或已损坏）")
        nonce = ciphertext[len(_MAGIC) : head]
        try:
            return self._aead.decrypt(nonce, ciphertext[head:], aad)
        except InvalidTag as exc:
            raise CryptoIntegrityError("完整性校验失败：密文被篡改或密钥不匹配") from exc
