"""时间戳统一口径：全库数据层时间戳一律 naive-UTC 字符串。

历史包袱：同一台账里混用三种时间——审计链 naive-UTC（repository）、
批次/备份/看门狗 time.strftime() 本地时间（Asia/Shanghai 相差 8 小时）、
评估产物 datetime.now() 本地时间。本模块提供唯一实现，业务侧一律取用。
（展示层如需本地时区，由前端/报表层自行转换。）
"""

from __future__ import annotations

from datetime import UTC, datetime


def naive_utc_now() -> datetime:
    """当前 UTC 时间的 naive datetime（与 DB created_at 同口径）。"""
    return datetime.now(UTC).replace(tzinfo=None)


def fmt_naive_utc(fmt: str = "%Y-%m-%dT%H:%M:%S") -> str:
    """按格式输出 naive-UTC 时间戳字符串。"""
    return naive_utc_now().strftime(fmt)
