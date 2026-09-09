/**
 * 底片印字（扫描日期/编号）展示工具：后端识别快照 → 徽标文案/样式。
 * 镜像 backend domain/stamp.py 的 status 语义；档案表格、底片观察、批量结果共用，
 * 避免"present/missing/镜像"各处各写一套文案漂移。
 */

/** 印字数据的两种形态（REST 记录字段 / 查看器 store 快照）的公共子集 */
export interface FilmStampLike {
  status?: string | null;
  text?: string | null;
  orientation?: string | null;
  /** REST 形态（RecordItem.stamp_need_review / BatchTaskOut.stamp_need_review） */
  need_review?: boolean | null;
  /** store 快照形态（FilmStampInfo.needReview） */
  needReview?: boolean | null;
}

export interface StampBadge {
  /** 徽标主文案（短） */
  label: string;
  /** 语义色：ok=有印字 | warn=缺印字已转复核 | muted=缺印字未触发复核 */
  cls: "ok" | "warn" | "muted";
  /** 悬浮完整解释（含识别内容） */
  title: string;
}

/** 表格/徽标里印字内容的截断展示（日期+编号并存时可能较长） */
export function stampShortText(text: string | null | undefined, max = 24): string {
  const t = (text ?? "").trim();
  if (!t) return "";
  return t.length > max ? `${t.slice(0, max)}…` : t;
}

/**
 * 印字徽标；无识别数据（历史记录/未启用/引擎不可用）返回 null，调用方显示 "—"。
 * 语义映射：
 * - present + normal  → 「正向 <内容>」（绿）
 * - present + mirrored→ 「镜像 <内容>」（绿，悬浮注明背面扫描）
 * - missing + 已转复核 → 「无印字·待复核」（红）
 * - missing + 未转复核 → 「无印字」（灰，批量豁免/批次未裁决——按用户规则不追责）
 */
export function stampBadge(f: FilmStampLike | null | undefined): StampBadge | null {
  if (!f || !f.status) return null;
  const text = stampShortText(f.text);
  const withText = (label: string): string => (text ? `${label}（${text}）` : label);
  if (f.status === "present") {
    if (f.orientation === "mirrored") {
      return {
        label: text ? `镜像 ${text}` : "印字·镜像",
        cls: "ok",
        title: withText("识别到印字（镜像：底片背面扫描，文字左右翻转）"),
      };
    }
    return {
      label: text ? `正向 ${text}` : "印字·正向",
      cls: "ok",
      title: withText("识别到印字（正向）"),
    };
  }
  if (f.status === "missing") {
    if (f.need_review || f.needReview) {
      return {
        label: "无印字·待复核",
        cls: "warn",
        title: "未识别到日期/编号印字，需人工复核确认底片身份",
      };
    }
    return {
      label: "无印字",
      cls: "muted",
      title: "未识别到日期/编号印字（未触发人工复核：小批量未裁决或该批已按印字占比豁免）",
    };
  }
  return null; // unavailable / off：无有效识别结论
}
