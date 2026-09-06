"""统一日志配置（可观测性基础设施，）。

提供 `configure_logging(obs: ObservabilityCfg | None)`：
- 固定 INFO 级别、统一格式（时间/级别/模块/消息）；
- log_format=json 时用 JSON 结构化输出（每行一个 JSON 记录），
  便于接入日志采集/ELK/云日志，是十年数据积累的可观测基础；
- log_format=text 时保持人类可读（本地开发默认）。

配置三处同步：default.yaml + schema.yaml + infra/config.py。
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import time
from typing import Any

# 日志文件轮转（2026：持久化到用户数据目录 + RotatingFileHandler，杜绝 100h
# 长跑下 %TEMP%/ScanDetection/backend.log 无界增长撑爆磁盘/拖累 I/O）。
_LOGFILE_MAX_BYTES = 5 * 1024 * 1024  # 单文件 5 MB
_LOGFILE_BACKUP_COUNT = 20  # 保留最近 20 个轮转文件（合计上限 ~105 MB）


class JsonFormatter(logging.Formatter):
    """将日志记录序列化为单行 JSON（结构化、可机器消费）。

    保留时间/级别/logger/消息；extra 自动并入顶层字段（如 model_uri），
    崩溃信息完整保留。序列化失败回退字符串。
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in {
                "args",
                "message",
                "asctime",
                "created",
                "exc_info",
                "exc_text",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "msg",
                "name",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "thread",
                "threadName",
                "taskName",
            }:
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                payload[key] = value
        try:
            return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            # extra 含不可序列化对象时降级：串成 str 再打包，保证日志不崩。
            payload["extra"] = repr({k: v for k, v in record.__dict__.items() if k not in set()})
            return super().format(record)


def _writable_log_base_dir() -> str | None:
    """返回可写的日志目录；无法创建/不可写返回 None。

    候选优先级：
    1. 打包版用户数据目录（SCANDETECTION_USER_DATA_DIR）/logs——随用户数据
       卸载保留。**不得**用 %LOCALAPPDATA%/ScanDetection：installMode=
       currentUser 下该路径即安装目录，硬编码会让日志进 $INSTDIR 被卸载
       清空（历史缺陷实测：11MB 日志随卸载丢失）。
    2. %LOCALAPPDATA%/ScanDetection/logs（开发布局，用户可写）。
    3. %TEMP%/ScanDetection/logs（LOCALAPPDATA 缺失时的兜底）。
    """
    candidates: list[str] = []
    from backend.infra.paths import data_dir_override

    override = data_dir_override()
    if override is not None:
        candidates.append(os.path.join(str(override), "logs"))
    la = os.environ.get("LOCALAPPDATA")
    if la:
        candidates.append(os.path.join(la, "ScanDetection", "logs"))
    tmp = os.environ.get("TEMP") or os.environ.get("TMP")
    if tmp:
        candidates.append(os.path.join(tmp, "ScanDetection", "logs"))
    for base in candidates:
        try:
            os.makedirs(base, exist_ok=True)
        except OSError:
            continue
        if os.access(base, os.W_OK):
            return base
    return None


def configure_logging(log_format: str = "text") -> None:
    """按配置装配根/应用级日志（INFO，统一格式）。

    force=True 保证本配置优先生效（避免第三方库早期 basicConfig 覆盖）。
    关键路径（模型加载/推理/判定/审计）据此可观测。

    输出两路，兼顾持久化与紧急兜底，均不构成无界增长：
    - 主路：RotatingFileHandler 写入日志目录（打包版=用户数据目录 logs/，
      开发版=%LOCALAPPDATA%/ScanDetection/logs，见 _writable_log_base_dir），
      超 5MB 轮转、保留 20 份（上限 ~105MB）；
    - 兜底：ERROR 级 stderr（由 Tauri 壳收入 %TEMP%/ScanDetection/backend.log），
      只承载严重异常，频次低，避免逐条业务日志无界写入临时文件。
    """
    level = logging.INFO
    root = logging.getLogger()
    root.setLevel(level)
    # 清理既有 handler，避免重复输出（force 之外再显式去重）。
    for h in list(root.handlers):
        root.removeHandler(h)

    if log_format == "json":
        fmt = JsonFormatter()
    else:
        fmt = logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

    base = _writable_log_base_dir()
    if base is not None:
        fh = logging.handlers.RotatingFileHandler(
            os.path.join(base, "backend.log"),
            maxBytes=_LOGFILE_MAX_BYTES,
            backupCount=_LOGFILE_BACKUP_COUNT,
            encoding="utf-8",
        )
        fh.setFormatter(fmt)
        fh.setLevel(level)
        root.addHandler(fh)
        logging.getLogger("scandetection.logging").info(
            "logs written to %s", os.path.join(base, "backend.log")
        )
    else:
        logging.getLogger("scandetection.logging").warning(
            "no writable log dir; fall back to stderr only"
        )

    # 紧急兜底：ERROR 级走 stderr（崩溃/严重异常仍可在临时日志查到）。
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    sh.setLevel(logging.ERROR)
    root.addHandler(sh)
    # 允许业务日志在 INFO 可见（子 logger 默认继承 root 的级别）。
    logging.addLevelName(logging.WARNING, "WARNING")
    # 削减 uvicorn 逐请求访问日志噪音（桌面场景意义小且会持续膨胀日志）。
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
