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
import sys
import time
from typing import Any


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


def configure_logging(log_format: str = "text") -> None:
    """按配置装配根/应用级日志（INFO，统一格式）。

    force=True 保证本配置优先生效（避免第三方库早期 basicConfig 覆盖）。
    关键路径（模型加载/推理/判定/审计）据此可观测。
    """
    level = logging.INFO
    root = logging.getLogger()
    root.setLevel(level)
    # 清理既有 handler，避免重复输出（force 之外再显式去重）。
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stderr)
    if log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s [%(name)s] %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
    handler.setLevel(level)
    root.addHandler(handler)
    # 允许业务日志在 INFO 可见（子 logger 默认继承 root 的级别）。
    logging.addLevelName(logging.WARNING, "WARNING")
