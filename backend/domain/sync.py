"""端边云同步 v1 适配器。

设计文档：v1 = LocalAdapter（空操作/本地）；v3 = CloudAdapter（联邦平均）。
v1 语义——数据**不出本机**：
- push：把待同步记录追加到本地待同步队列（经 QueuePort 注入，JSONL 落盘在 infra），
  不发起任何网络调用；
- pull：本地无远端，恒返回空列表；
- federate：无远端可联邦，空操作（v3 才实现联邦平均）。

IO 依赖倒置：JSONL 落盘走 QueuePort、HTTP 推送走 HttpPushPort，
均由 infra 实现并经构造注入；domain 不直接触碰文件系统/网络。调用方（dependencies）
装配 infra 实现即切换持久化/传输方案，不影响本适配器契约与上层调用。
"""

from __future__ import annotations

from typing import Any

from backend.domain.interfaces import HttpPushPort, QueuePort


class LocalAdapter:
    """本地同步适配器。"""

    name = "local"

    def __init__(self, queue: QueuePort | None = None) -> None:
        """queue 缺省时不做持久化（纯空操作）；提供时 push 追加到队列。"""
        self._queue = queue

    def push(self, record) -> None:
        """本地记录待同步项（不发起网络）。record 需可 JSON 序列化。"""
        if self._queue is not None:
            self._queue.append(record)

    def pull(self) -> list[dict[str, Any]]:
        """本地无远端：恒返回空列表。"""
        return []

    def federate(self, weights) -> None:
        """v3 才实现联邦平均；v1 空操作（无远端可联邦）。"""
        return

    @property
    def pending_count(self) -> int:
        """本地待同步队列条数（观测用）。"""
        return self._queue.count() if self._queue is not None else 0


class CloudAdapter:
    """云侧联邦适配器。

    占位语义：接口契约完整（push / pull / federate），但 v3 联邦平均算法
    **未实现**——三个方法均显式抛 ``NotImplementedError``，绝不静默假装已
    同步/已联邦（对"数据不出本机"合规而言，静默空操作比报错更危险：调用方
    会误以为数据已安全上云）。

    v3 实现契约（供后续落地）：
    - ``federate(weights)``：联邦平均（FedAvg）——聚合参与方模型权重，
      ``weights`` 为各端权重/增量（结构待 v3 定义），按轮次聚合后回传或落库；
    - ``push(record)``：把待同步记录推送到云侧（加密传输 + 鉴权由实现保证）；
    - ``pull``：拉取云侧联邦结果/模型更新（取代 v1 恒空列表）。
    配置 ``sync.kind=cloud`` 接入；未实现前启用会得到显式 NotImplementedError，
    而非静默无效果。
    """

    name = "cloud"

    def __init__(self, endpoint: str, token: str | None = None) -> None:
        if not endpoint:
            raise ValueError("CloudAdapter 需要配置 sync.http_endpoint（云侧地址）")
        self._endpoint = endpoint
        self._token = token

    def push(self, record) -> None:
        """v3 未接入：显式占位，禁止静默空操作。"""
        raise NotImplementedError("v3 CloudAdapter 未实现：云侧同步尚未接入")

    def pull(self) -> list[dict[str, Any]]:
        """v3 未接入：显式占位，禁止静默空列表。"""
        raise NotImplementedError("v3 CloudAdapter 未实现：云侧拉取尚未接入")

    def federate(self, weights) -> None:
        """v3 未接入：显式占位（联邦平均算法待实现）。"""
        raise NotImplementedError("v3 CloudAdapter 未实现：联邦平均（FedAvg）待实现")

    @property
    def pending_count(self) -> int:
        """云侧无本地待同步队列（占位语义）。"""
        return 0


class HttpSyncAdapter:
    """HTTP 同步适配器。

    与 LocalAdapter 同契约：
    - push：本地留档（经 QueuePort）+ POST 到 http_endpoint（经 HttpPushPort，
      Bearer token 可选）；网络失败仅告警、不抛（同步是尽力而为，不阻断主流程）；
    - pull：返回空列表（云侧拉取留待 v3 联邦）；
    - federate：空操作（v3 才实现联邦平均）。

    传输加密由调用方负责（建议 https + 端点鉴权），本适配器不内置 TLS 终止逻辑。
    """

    name = "http"

    def __init__(
        self,
        endpoint: str | None,
        token: str | None = None,
        queue: QueuePort | None = None,
        transport: HttpPushPort | None = None,
    ) -> None:
        if not endpoint:
            raise ValueError("HttpSyncAdapter 需要配置 sync.http_endpoint")
        self._endpoint = endpoint
        self._token = token
        self._queue = queue
        self._transport = transport

    def push(self, record) -> None:
        """本地留档 + 推送远端（尽力而为，失败仅告警）。"""
        if self._queue is not None:
            self._queue.append(record)
        if self._transport is not None:
            self._transport.post(self._endpoint, self._token, record)

    def pull(self) -> list[dict[str, Any]]:
        """云侧拉取留待 v3；当前返回空列表。"""
        return []

    def federate(self, weights) -> None:
        """v3 才实现联邦平均；当前空操作。"""
        return

    @property
    def pending_count(self) -> int:
        """本地待同步队列条数（观测用）。"""
        return self._queue.count() if self._queue is not None else 0
