"""标注治理（DB50/T 1807-2025 ）：三人标注一致性 + 数据集互斥/泄漏审计。"""

from backend.domain.labeling.consensus import (
    ConsensusResult,
    LabelBox,
    arbitrate,
    new_session_id,
    resolve_consensus,
)
from backend.domain.labeling.dataset_guard import (
    OverlapReport,
    assert_disjoint,
    find_overlaps,
)
from backend.domain.labeling.leakage import (
    LeakageAudit,
    PoolEntry,
    assert_no_leakage,
    assign_groups,
    audit_leakage,
    film_group,
    scan_split,
)

__all__ = [
    "ConsensusResult",
    "LabelBox",
    "LeakageAudit",
    "OverlapReport",
    "PoolEntry",
    "arbitrate",
    "assert_disjoint",
    "assert_no_leakage",
    "assign_groups",
    "audit_leakage",
    "film_group",
    "find_overlaps",
    "new_session_id",
    "resolve_consensus",
    "scan_split",
]
