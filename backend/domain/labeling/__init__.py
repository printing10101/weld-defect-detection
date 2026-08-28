"""标注治理（DB50/T 1807-2025 ）：三人标注一致性 + 数据集互斥校验。"""

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

__all__ = [
    "ConsensusResult",
    "LabelBox",
    "OverlapReport",
    "arbitrate",
    "assert_disjoint",
    "find_overlaps",
    "new_session_id",
    "resolve_consensus",
]
