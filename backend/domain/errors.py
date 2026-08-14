"""领域异常族（冻结骨架，§T2 / §13.4）。

错误码与 §14 错误码表一一对应；由应用层全局处理器映射 HTTP 状态。
"""

from __future__ import annotations


class AppError(Exception):
    """领域异常基类。"""

    code: str = "UNKNOWN"
    http_status: int = 500


class ImageUnreadableError(AppError):
    """影像无法解析（400）。"""

    code = "IMG_UNREADABLE"
    http_status = 400


class IQIFailError(AppError):
    """IQI/底片质量校验不通过（409，阻断评片）。"""

    code = "IQI_FAIL"
    http_status = 409


class ModelUnavailableError(AppError):
    """模型不可用（503）。"""

    code = "MODEL_UNAVAILABLE"
    http_status = 503


class GradingAmbiguousError(AppError):
    """判定信息不足/边界歧义（422，应触发人工复核）。"""

    code = "GRADING_AMBIGUOUS"
    http_status = 422
