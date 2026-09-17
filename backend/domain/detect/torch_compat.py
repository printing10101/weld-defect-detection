"""torch>=2.6 weights_only 兼容垫片。

torch 2.6 起 ``torch.load`` 默认 ``weights_only=True``，会把旧版 ultralytics
（<8.3）加载整包 .pt（内含 DetectionModel 等序列化对象）的路径直接拦截。
本工程权重为本机分发、指纹受模型注册表管控的受信文件，加载自身权重时
在受控作用域内恢复旧行为（``weights_only=False``），不改全局默认值。

只用于加载自有受信权重；严禁用于加载外部来源文件（反序列化有代码执行风险）。
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


@contextmanager
def trusted_torch_load() -> Iterator[None]:
    """作用域内 torch.load 默认 weights_only=False（退出即恢复）。"""
    import torch  # type: ignore  # torch 为可选 ML 依赖（ml extra），类型环境不装

    orig_load = torch.load

    def _trusted_load(*args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        kwargs.setdefault("weights_only", False)
        return orig_load(*args, **kwargs)

    torch.load = _trusted_load  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.load = orig_load
