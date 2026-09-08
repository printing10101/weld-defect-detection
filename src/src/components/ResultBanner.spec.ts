// ResultBanner 组件测试：结论先行的三态语义（ok / review / fail）。
// 数据契约：props.result 为后端 ReportOut 镜像；文案全部来自真实字段。
import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import ResultBanner from "./ResultBanner.vue";
import type { ReportOut } from "../types/api";

function makeResult(overrides: Partial<ReportOut> = {}): ReportOut {
  return {
    report_id: "r1",
    image_id: "i1",
    joint_level: "II",
    need_review: false,
    evaluable: true,
    defect_count: 3,
    disclaimer: null,
    disposition: "accept",
    disposition_label: "可放行",
    disposition_actions: [],
    pdf_url: "/api/v1/report/r1/pdf",
    ...overrides,
  };
}

describe("ResultBanner", () => {
  it("可评片且无需复核：ok 态 + 级别 + 缺陷计数", () => {
    const w = mount(ResultBanner, { props: { result: makeResult() } });
    expect(w.text()).toContain("II");
    expect(w.text()).toContain("检出 3 处缺陷");
    expect(w.classes().join(" ")).not.toContain("review");
  });

  it("需复核：review 态 + 待复核文案", () => {
    const w = mount(ResultBanner, {
      props: { result: makeResult({ need_review: true }) },
    });
    expect(w.text()).toContain("复核");
    expect(w.text()).toContain("II"); // 有级别但需复核
  });

  it("需复核且无级别：显示待复核", () => {
    const w = mount(ResultBanner, {
      props: {
        result: makeResult({ need_review: true, joint_level: null, defect_count: 0 }),
      },
    });
    expect(w.text()).toContain("待复核");
    expect(w.text()).toContain("未检出缺陷");
  });

  it("不可评片：fail 态 + 保守原则文案（不构造级别）", () => {
    const w = mount(ResultBanner, {
      props: { result: makeResult({ evaluable: false, joint_level: null }) },
    });
    expect(w.text()).toContain("不可评片");
    expect(w.text()).toContain("未达标准");
  });

  it("零缺陷通过：ok 态 + 未检出缺陷文案", () => {
    const w = mount(ResultBanner, {
      props: { result: makeResult({ defect_count: 0 }) },
    });
    expect(w.text()).toContain("未检出缺陷");
  });
});
