"""双链审计留痕的唯一封装（C-19）。

口径：**主链全量记录**；``security=True`` 时另入独立安全链
（防单链被整体覆盖），安全链可用 sec_* 参数覆盖差异字段。
所有需要双链留痕的动作一律经由本函数，不得各自手写两条 append。
"""

from __future__ import annotations

from typing import Any


def dual_audit(
    repository: Any,
    security_store: Any,
    *,
    actor: str,
    action: str,
    object_type: str,
    object_id: str,
    before: dict[str, Any] | None = None,
    after: dict[str, Any] | None = None,
    note: str | None = None,
    security: bool = False,
    sec_object_id: str | None = None,
    sec_before: dict[str, Any] | None = None,
    sec_after: dict[str, Any] | None = None,
) -> None:
    """关键动作入主审计链；security=True 时同步入独立安全审计链。

    参数为两个 infra 仓储对象（app 层调用方从各自 registry 取用），
    不依赖 app 层类型——分层合约（infra 不得 import app）保持成立。
    """
    repository.append_audit(
        actor=actor,
        action=action,
        object_type=object_type,
        object_id=object_id,
        before=before,
        after=after,
        note=note,
    )
    if security:
        security_store.append_security_audit(
            actor=actor,
            action=action,
            object_type=object_type,
            object_id=sec_object_id or object_id,
            before=sec_before if sec_before is not None else before,
            after=sec_after if sec_after is not None else after,
            note=note,
        )
