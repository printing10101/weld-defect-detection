"""报告二维码追溯（G01：二维码档案追溯）。

报告每页页脚嵌入追溯二维码，编码 `RT-TRACE|{report_id}|{指纹前16位}`：
- report_id 供扫码后在系统内定位档案（GET /report/trace/{code}）；
- 指纹前缀供人工与报告页脚指纹比对，快速识别调包/篡改。

内容刻意保持最小——不含工件号/级别等业务数据（报告自身有密级体系，
二维码可能脱离系统流转，避免泄露）。生成失败返回 None（fail-soft，
不阻断出片：二维码是增强能力，报告是主业务）。
"""

from __future__ import annotations

import io
import logging

_LOG = logging.getLogger(__name__)

_TRACE_PREFIX = "RT-TRACE"
_HASH_PREFIX_LEN = 16


def trace_code(report_id: str, fingerprint: str) -> str:
    """构造追溯码文本（report_id + 内容指纹前缀）。"""
    return f"{_TRACE_PREFIX}|{report_id}|{fingerprint[:_HASH_PREFIX_LEN]}"


def parse_trace_code(code: str) -> tuple[str, str] | None:
    """解析追溯码 → (report_id, hash_prefix)；格式非法返回 None。

    hash 前缀必须是 16 位 hex——防止把任意文本当追溯码注入查询。
    """
    parts = code.strip().split("|")
    if len(parts) != 3 or parts[0] != _TRACE_PREFIX:
        return None
    report_id, hash_prefix = parts[1], parts[2]
    if not report_id or len(hash_prefix) != _HASH_PREFIX_LEN:
        return None
    try:
        int(hash_prefix, 16)
    except ValueError:
        return None
    return report_id, hash_prefix


def qr_png_bytes(payload: str) -> bytes | None:
    """生成二维码 PNG bytes；qrcode 未安装/生成失败时返回 None（fail-soft）。"""
    try:
        import qrcode
        from qrcode.constants import ERROR_CORRECT_M

        qr = qrcode.QRCode(error_correction=ERROR_CORRECT_M, box_size=4, border=1)
        qr.add_data(payload)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as exc:  # noqa: BLE001 —— 二维码生成失败不阻断出片
        _LOG.warning("追溯二维码生成失败（报告继续生成）: %s", exc)
        return None
