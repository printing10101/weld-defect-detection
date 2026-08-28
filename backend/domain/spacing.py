"""像素标定解析。

``/detect`` 与 ``/report`` 两条链路此前各自解析像素标定（mm/px），语义分裂：
``/report`` 链路由 ``_resolve_spacing`` 返回 ``(spacing, known)``，未标定时
``known=False``、``context.pixel_spacing_mm`` 置 None 由 grader 熔断（不输出级别）；
而 ``/detect`` 端点静默 ``or 1.0`` 退化，未标定时仍输出伪物理 mm。

本模块把该逻辑提升为 domain 层共享工具，消除分裂：两链路共用同一判定，
未标定时 **不输出伪物理尺寸**（``/detect`` 物理字段置 None 并标 ``calibrated=False``，
``/report`` 仍由 grader 熔断）。
"""

from __future__ import annotations


def resolve_spacing(
    requested: float | None,
    from_meta: float | None,
) -> tuple[float, bool]:
    """确定像素标定（mm/px）并标明其是否可信。

    返回 ``(spacing, known)``：

    - 任一来源为有效正数（``> 0``）时采用之，``known=True``。
    - 两处均无效（``None`` 或 ``<= 0``）时退化为 ``(1.0, False)``：1.0 仅用于让
      几何换算继续跑通（供人工查看像素级形状），**不得**据此定级或输出物理尺寸——
      调用方须据 ``known`` 标记 ``calibrated=False``（/detect）或将
      ``pixel_spacing_mm`` 置 None 由 grader 熔断（/report）。
    """
    for value in (requested, from_meta):
        if value is not None and value > 0:
            return float(value), True
    return 1.0, False
