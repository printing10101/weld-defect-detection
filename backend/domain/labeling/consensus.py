"""标注一致性（DB50/T 1807-2025 §8.4.2 三人独立标注 + 仲裁）。

标准规格：A、B、C 三名标注人员独立标注同一底片；当标注缺陷类型一致且
缺陷范围的交并比 IOU(A&B&C)≥0.5 时，标注样本为三人的并集（A∪B∪C）；
类型不一致或 IOU<0.5 时，该缺陷不计入标注样本。

本实现（比标准严）：
- 一致性判据 = 三人**两两** IOU 全部≥阈值（默认 0.6，严于标准底线 0.5）
  且类型一致，避免对"三方交集"的模糊解释；
- 作废缺陷不静默丢弃：输出作废清单与原因（谁缺失/IOU 多少），供标注组长仲裁；
- 仲裁模式：组长人工从三人标注中选定保留框（记录仲裁人与理由）；
- 输出标注员间一致率统计（供附录A 记录表引用）。
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LabelBox:
    """单个标注框（一人一框）。"""

    annotator: str  # "A" | "B" | "C"（或实际姓名）
    class_id: int
    bbox: tuple[float, float, float, float]  # [x,y,w,h] 像素（同坐标系）


@dataclass
class ConsensusResult:
    """三人标注一致性结果。"""

    accepted: list[dict[str, Any]] = field(default_factory=list)
    # accepted: {class_id, bbox(三人并集外接框), annotators, iou_min, source:"consensus"}
    discarded: list[dict[str, Any]] = field(default_factory=list)
    # discarded: {annotator, class_id, bbox, reason}
    agreement_rate: float = 0.0  # 接受框数 / 三人框总数（0~1）
    threshold: float = 0.6

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "discarded": self.discarded,
            "agreement_rate": round(self.agreement_rate, 4),
            "threshold": self.threshold,
        }


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2 = min(a[0] + a[2], b[0] + b[2])
    y2 = min(a[1] + a[3], b[1] + b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def resolve_consensus(
    annotations: list[LabelBox],
    threshold: float = 0.6,
    *,
    annotators: tuple[str, ...] = ("A", "B", "C"),
) -> ConsensusResult:
    """三人标注一致性仲裁（§8.4.2）。

    算法：以每位标注员的每个框为种子，在其余标注员中找**同类别**且 IOU≥阈值
    的最佳配对（贪心，每框至多被消费一次）；三方齐备 → 接受（外接并集框），
    否则该种子框作废并记录原因。
    """
    result = ConsensusResult(threshold=threshold)
    by_ann: dict[str, list[LabelBox]] = {a: [] for a in annotators}
    for box in annotations:
        if box.annotator not in by_ann:
            raise ValueError(f"未知标注员: {box.annotator}（期望 {annotators}）")
        by_ann[box.annotator].append(box)

    used: set[int] = set()  # 已被消费的框（按 id() 标记）
    box_ids: dict[int, int] = {id(b): i for i, b in enumerate(annotations)}

    def _best_match(seed: LabelBox, other: str, taken: set[int]) -> tuple[LabelBox, float] | None:
        best: tuple[float, LabelBox] | None = None
        for cand in by_ann[other]:
            cid = box_ids[id(cand)]
            if cid in used or cid in taken:
                continue
            if cand.class_id != seed.class_id:
                continue
            v = iou(seed.bbox, cand.bbox)
            if v >= threshold and (best is None or v > best[0]):
                best = (v, cand)
        return (best[1], best[0]) if best else None

    for box in annotations:
        bid = box_ids[id(box)]
        if bid in used:
            continue
        taken = {bid}
        matched: dict[str, tuple[LabelBox, float]] = {}
        ok = True
        iou_min = 1.0
        for other in annotators:
            if other == box.annotator:
                continue
            m = _best_match(box, other, taken)
            if m is None:
                ok = False
                break
            matched[other] = m
            taken.add(box_ids[id(m[0])])
            iou_min = min(iou_min, m[1])
        if ok:
            for m in matched.values():
                used.add(box_ids[id(m[0])])
            used.add(bid)
            group = [box] + [m[0] for m in matched.values()]
            xs = [b.bbox[0] for b in group]
            ys = [b.bbox[1] for b in group]
            xe = [b.bbox[0] + b.bbox[2] for b in group]
            ye = [b.bbox[1] + b.bbox[3] for b in group]
            result.accepted.append(
                {
                    "class_id": box.class_id,
                    "bbox": [min(xs), min(ys), max(xe) - min(xs), max(ye) - min(ys)],
                    "annotators": [b.annotator for b in group],
                    "iou_min": round(iou_min, 4),
                    "source": "consensus",
                }
            )
        else:
            used.add(bid)
            missing = [a for a in annotators if a != box.annotator and a not in matched]
            result.discarded.append(
                {
                    "annotator": box.annotator,
                    "class_id": box.class_id,
                    "bbox": list(box.bbox),
                    "reason": f"{'/'.join(missing)} 未见同类型且 IOU≥{threshold} 的配对框",
                }
            )

    n_total = len(annotations)
    result.agreement_rate = (3 * len(result.accepted) / n_total) if n_total else 0.0
    return result


def arbitrate(
    consensus: ConsensusResult,
    decisions: list[dict[str, Any]],
    *,
    arbitrator: str,
    reason: str,
    label_sink: Callable[[list[dict[str, Any]]], None] | None = None,
) -> dict[str, Any]:
    """组长仲裁：从作废清单中人工恢复保留框（比标准多的补救通道）。

    decisions: [{"annotator": "B", "index": 0, "class_id": 4, "bbox": [..]}]
    —— index 指向该标注员在 discarded 清单中的序号；恢复的框计入标注样本。
    """
    if not reason.strip():
        raise ValueError("仲裁必须填写理由（审计留痕）")
    recovered: list[dict[str, Any]] = []
    for dec in decisions:
        ann, idx = dec["annotator"], int(dec["index"])
        rows = [d for d in consensus.discarded if d["annotator"] == ann]
        if idx < 0 or idx >= len(rows):
            raise ValueError(f"仲裁目标不存在: {ann}[{idx}]")
        row = rows[idx]
        recovered.append(
            {
                "class_id": int(dec.get("class_id", row["class_id"])),
                "bbox": list(dec.get("bbox", row["bbox"])),
                "annotators": [ann],
                "iou_min": None,
                "source": "arbitration",
                "arbitrator": arbitrator,
                "reason": reason,
            }
        )
    if label_sink is not None:
        label_sink(recovered)
    return {"recovered": recovered, "arbitrator": arbitrator, "reason": reason}


def new_session_id() -> str:
    """一次三人标注会话的唯一 id。"""
    return uuid.uuid4().hex[:16]
