/**
 * 底片印字徽标工具单测：status/orientation/复核标记 → 文案与语义色。
 * 覆盖档案表格（RecordItem）与查看器 store（FilmStampInfo）两种入参形态。
 */
import { describe, expect, it } from "vitest";
import { stampBadge, stampShortText } from "./filmStamp";

describe("stampShortText", () => {
  it("空值返回空串，长文本截断加省略号", () => {
    expect(stampShortText(null)).toBe("");
    expect(stampShortText("  ")).toBe("");
    expect(stampShortText("2023-08-12 No.0421")).toBe("2023-08-12 No.0421");
    const long = stampShortText("2023-08-12 No.0421 W-12", 10);
    expect(long).toHaveLength(11);
    expect(long.endsWith("…")).toBe(true);
  });
});

describe("stampBadge", () => {
  it("无识别数据（null/历史记录/unavailable/off）不显示徽标", () => {
    expect(stampBadge(null)).toBeNull();
    expect(stampBadge(undefined)).toBeNull();
    expect(stampBadge({})).toBeNull();
    expect(stampBadge({ status: "unavailable" })).toBeNull();
    expect(stampBadge({ status: "off" })).toBeNull();
  });

  it("正向印字：ok 徽标带识别内容", () => {
    const b = stampBadge({
      status: "present",
      text: "2023-08-12",
      orientation: "normal",
      need_review: false,
    });
    expect(b).not.toBeNull();
    expect(b!.cls).toBe("ok");
    expect(b!.label).toContain("正向");
    expect(b!.label).toContain("2023-08-12");
  });

  it("镜像印字：ok 徽标注明镜像（背面扫描）", () => {
    const b = stampBadge({
      status: "present",
      text: "No.0421",
      orientation: "mirrored",
      needReview: false,
    });
    expect(b!.cls).toBe("ok");
    expect(b!.label).toContain("镜像");
    expect(b!.title).toContain("背面扫描");
  });

  it("缺印字已转人工复核：warn 红标", () => {
    const b = stampBadge({ status: "missing", need_review: true });
    expect(b!.cls).toBe("warn");
    expect(b!.label).toContain("待复核");
  });

  it("缺印字未触发复核（批量豁免/未裁决）：muted 灰标，两种形态均可识别", () => {
    for (const f of [
      { status: "missing", need_review: false },
      { status: "missing", needReview: false },
    ]) {
      const b = stampBadge(f);
      expect(b!.cls).toBe("muted");
      expect(b!.label).toBe("无印字");
    }
  });

  it("store 快照形态（needReview=true）同样判 warn", () => {
    expect(stampBadge({ status: "missing", needReview: true })!.cls).toBe("warn");
  });
});
