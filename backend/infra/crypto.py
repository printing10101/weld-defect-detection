"""AES 静态加密（§7.5 / §T4）。

M1 提供接口与密钥生成；完整加解密在 M5 安全模块实现。
密钥来源：环境变量 SCAN_CRYPTO_KEY 或首次启动生成并安全存储（M5）。
"""
from __future__ import annotations

import os


class AesCrypto:
    def __init__(self, key: bytes | None = None) -> None:
        self.key = key or os.urandom(32)

    def encrypt(self, plaintext: bytes) -> bytes:
        raise NotImplementedError("M5 安全模块实现")

    def decrypt(self, ciphertext: bytes) -> bytes:
        raise NotImplementedError("M5 安全模块实现")
