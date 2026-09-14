"""《射线检测报告》汇总表的补充信息字段（对齐正式 RT 报告样张）。

键名是 前端表单 / API 白名单 / PDF 汇总表填充 三方的约定；值一律为短文本
（用户从检测委托单/工艺卡抄录）。分组与样张汇总表的分区一一对应。

sanitize_report_meta 是唯一入口：API 层用它白名单校验 + 清洗，PDF 渲染
与前端展示只消费清洗后的 dict，保证三方看到同一份数据。
"""

from __future__ import annotations

from typing import Any

_META_VALUE_MAX = 64  # 单值长度上限（超长截断，表格单元格放不下更长的）

# (组名, ((键, 中文标签), ...))；顺序即前端表单与报告分区的展示顺序。
REPORT_META_GROUPS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "工程信息",
        (
            ("client_unit", "委托单位"),
            ("project_name", "工程名称"),
            ("project_category", "工程类别/检测时机"),
            ("test_address", "检测地址"),
        ),
    ),
    (
        "工件概况",
        (
            ("material", "材质"),
            ("part_no", "工件编号"),
            ("groove_type", "坡口形式"),
            ("surface_status", "表面状况"),
            ("weld_process", "焊接方式"),
            ("heat_treatment", "热处理状态"),
        ),
    ),
    (
        "技术要求",
        (
            ("tech_level", "检测技术等级"),
            ("accept_level", "合格级别"),
            ("record_no", "原始记录编号"),
            ("scatter_control", "散射线控制"),
        ),
    ),
    (
        "检测器材及工艺参数",
        (
            ("source_kind", "源种类"),
            ("device_no", "设备型号/编号"),
            ("focus_size", "焦点尺寸"),
            ("film_model", "胶片型号"),
            ("film_size", "胶片规格"),
            ("film_class", "胶片分类等级"),
            ("screen_way", "增感方式"),
            ("iqi_position", "像质计摆放"),
            ("screens", "前屏/后屏"),
            ("technique", "透照方式"),
            ("focus_distance", "F（焦距）"),
            ("source_distance", "f（源至工件）"),
            ("film_distance", "b（工件至胶片）"),
            ("tube_voltage", "管电压"),
            ("tube_current", "管电流"),
            ("exposure_time", "曝光时间"),
            ("develop_method", "冲洗条件"),
            ("developer", "显影液配方"),
            ("develop_temp", "洗片温度"),
        ),
    ),
)

REPORT_META_LABELS: dict[str, str] = {
    key: label for _, fields in REPORT_META_GROUPS for key, label in fields
}


def sanitize_report_meta(raw: Any) -> dict[str, str]:
    """白名单清洗报告补充信息：非 dict/空值丢弃、值截断、未知键丢弃。

    宽容策略（与出片降级哲学一致）：清洗后为空映射返回 {}，不抛错——
    报告补充信息缺失只影响汇总表留空，不应阻断评片出片。
    """
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        if key not in REPORT_META_LABELS or value is None:
            continue
        text = str(value).strip()
        if text:
            out[str(key)] = text[:_META_VALUE_MAX]
    return out
