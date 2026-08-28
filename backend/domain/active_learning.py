"""主动学习闭环（§5.5/§5.6，M7 实现）。

规格书要求"检测→人工→回流→再训练"的持续学习闭环（数据 centric）：
1. **采样**：从检测结果中挑高价值样本优先人工标注（§5.6 主动学习）——
   uncertainty 高（近边界/低置信）、稀有或安全关键类优先；
2. **回流**：人工确认后的缺陷导出为 YOLO 训练标注（normalized txt），
   写入训练池目录，并更新数据版本 manifest（复用 §7.4 指纹语义）；
3. **再训练**：训练池 + 既有标注合并后触发重训（training 脚本/外部执行，
   本模块只负责数据层产出，不执行训练——分层铁律 §19.1）。

伪标签说明：本模块输出的 YOLO 标注即"人工复核结果写入训练池"的标准形态；
半监督伪标签（模型自预测低置信兜底）由调用方按 confidence 阈值决定是否
采纳，本模块不臆造标注类别。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.domain.dto import DefectClass, Detection
from backend.domain.interfaces import PoolStore

# 安全关键/稀有类（漏检代价高 → 主动学习优先采样）
_HIGH_VALUE_CLASSES = {
    DefectClass.CRACK,
    DefectClass.LACK_OF_FUSION,
    DefectClass.INCOMPLETE_PENETRATION,
    DefectClass.UNDERCUT,
    DefectClass.CONCAVITY,
}


@dataclass
class SampleCandidate:
    """高价值样本候选（供人工优先标注）。"""

    detection_id: str
    class_id: int
    score: float
    uncertainty: float
    value_score: float  # 综合采样价值（0~1，越高越优先）
    reasons: list[str] = field(default_factory=list)


def high_value_score(detection: Detection, *, safety_base: float = 0.5) -> float:
    """单条检测的主动学习采样价值（0~1）。

    三因子取 max（任一强烈信号即高价值）：
    - 不确定性 u：直接采用（M4 估计器语义，接近 1=极不可信 → 必须人工确认）；
    - 近边界信号：score 接近有效阈值（0.05~0.15 区间的低置信候选）；
    - 安全关键/稀有类：置 safety_base 基线，确保重大缺陷即便高置信也采样。
    """
    u = max(0.0, min(1.0, float(detection.uncertainty)))
    score = max(0.0, min(1.0, float(detection.score)))
    # 近边界：score 落在低置信区（<0.3）视为"差一点没检出"的高价值信号
    near_boundary = max(0.0, (0.3 - score) / 0.3) if score < 0.3 else 0.0
    class_base = safety_base if detection.class_id in _HIGH_VALUE_CLASSES else 0.0
    return round(max(u, near_boundary, class_base), 4)


def select_high_value(
    detections: list[Detection],
    *,
    top_k: int = 10,
    min_value: float = 0.0,
) -> list[SampleCandidate]:
    """从检测结果中选出高价值样本（按价值降序，取 top_k）。

    min_value=0 时仍按价值排序但全返回（调用方自行过滤）；>0 时仅返回
    价值达标的候选（避免把一堆低质量检测全部推给人工）。
    """
    scored = [
        SampleCandidate(
            detection_id=d.id,
            class_id=d.class_id.value,
            score=round(float(d.score), 4),
            uncertainty=round(float(d.uncertainty), 4),
            value_score=high_value_score(d),
            reasons=_reasons(d),
        )
        for d in detections
    ]
    scored.sort(key=lambda c: c.value_score, reverse=True)
    if min_value > 0:
        scored = [c for c in scored if c.value_score >= min_value]
    return scored[:top_k]


def _reasons(detection: Detection) -> list[str]:
    reasons: list[str] = []
    if detection.uncertainty >= 0.5:
        reasons.append("高不确定性")
    if detection.score < 0.3:
        reasons.append("低置信/近边界")
    if detection.class_id in _HIGH_VALUE_CLASSES:
        reasons.append("安全关键/稀有类")
    return reasons


# ---------------------------------------------------------------------------
# 标注回流：人工确认缺陷 → YOLO 训练标注
# ---------------------------------------------------------------------------


def to_yolo_label(
    detection: Detection,
    image_w: float,
    image_h: float,
    *,
    class_id_override: int | None = None,
) -> str:
    """单条检测 → YOLO normalized 标注行 "class cx cy w h"。

    坐标归一化到 [0,1]；bbox 越界时裁剪（不产出非法 >1 坐标）。
    class_id_override 用于人工复核修正类别（如把误检气孔改为裂纹）。
    """
    w = max(0.0, min(float(detection.bbox.w) / max(image_w, 1e-9), 1.0))
    h = max(0.0, min(float(detection.bbox.h) / max(image_h, 1e-9), 1.0))
    cx = max(0.0, min((detection.bbox.x + detection.bbox.w / 2) / max(image_w, 1e-9), 1.0))
    cy = max(0.0, min((detection.bbox.y + detection.bbox.h / 2) / max(image_h, 1e-9), 1.0))
    cid = class_id_override if class_id_override is not None else detection.class_id.value
    return f"{int(cid)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def export_training_labels(
    image_stem: str,
    detections: list[Detection],
    image_w: float,
    image_h: float,
    *,
    store: PoolStore,
    class_overrides: dict[str, int] | None = None,
) -> Path:
    """把人工确认的缺陷导出为 YOLO 标注文件（训练池回流，IO 经 PoolStore 注入）。

    store 写入 {image_stem}.txt（同 stem 覆盖，防重复导出旧标注残留）；
    class_overrides 按 detection.id 修正类别（人工复核改判）。
    返回标注文件路径。
    """
    lines = []
    for d in detections:
        cid = class_overrides.get(d.id) if class_overrides else None
        lines.append(to_yolo_label(d, image_w, image_h, class_id_override=cid))
    content = "\n".join(lines) + ("\n" if lines else "")
    return store.write_label(image_stem, content)


def training_pool_manifest(store: PoolStore) -> dict[str, Any]:
    """训练池数据版本 manifest（§7.4 指纹语义 + §5.6 划分记录）。

    返回 {sample_count, fingerprint, files, exported_at}；无标注时
    sample_count=0 / fingerprint=None（不臆造）。
    """
    files = store.list_labels()
    return {
        "sample_count": len(files),
        "fingerprint": store.fingerprint(),
        "files": files,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def load_pool_manifest(store: PoolStore) -> dict[str, Any] | None:
    """读取持久化 manifest（data/active/pool_manifest.json）；无则 None。"""
    return store.load_manifest()


def save_pool_manifest(store: PoolStore, manifest: dict[str, Any]) -> Path:
    """持久化 manifest 到 pool_dir 同级的 pool_manifest.json。"""
    return store.save_manifest(manifest)
