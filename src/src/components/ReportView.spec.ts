// ReportView 的两条界面契约：
// 1) 预筛级别必须显著标识，且结论不得被读成正式合格/不合格判定；
// 2) 本地大模型评片结论：成功展示正文，不可用时展示**原因**而不是空白。
import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";

vi.mock("../services/api", async (importOriginal) => {
  const orig = await importOriginal<typeof import("../services/api")>();
  return {
    ...orig,
    getReportNarrative: vi.fn(),
    getReportDetections: vi.fn(),
    verifyReport: vi.fn(),
    activeExport: vi.fn(),
  };
});

// useControlledPdf 内部会发起导出审批轮询，此处整体替换为静态桩。
vi.mock("../composables/useControlledPdf", async () => {
  const { ref } = await import("vue");
  return {
    useControlledPdf: () => ({
      gateOpen: ref(false),
      gateBusy: ref(false),
      gateMsg: ref(""),
      gateErr: ref(""),
      gateReason: ref(""),
      reportId: ref(""),
      requestId: ref(""),
      approval: ref(null),
      openPdf: vi.fn(),
      applyExportRequest: vi.fn(),
      fetchTokenAndDownload: vi.fn(),
      setReason: vi.fn(),
      closeGate: vi.fn(),
    }),
  };
});

import ReportView from "./ReportView.vue";
import { getReportNarrative } from "../services/api";
import type { ReportNarrativeOut, ReportOut } from "../types/api";

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
    warnings: [],
    basis: [],
    pdf_url: "/api/v1/report/r1/pdf",
    ...overrides,
  };
}

function mountView(result: ReportOut) {
  return mount(ReportView, {
    props: { result, sourceUrl: null, fileName: null },
    global: {
      stubs: {
        ResultBanner: true,
        DispositionPanel: true,
        ReviewPanel: true,
        PdfGateModal: true,
      },
    },
  });
}

function narrative(overrides: Partial<ReportNarrativeOut> = {}): ReportNarrativeOut {
  return {
    report_id: "r1",
    status: "ok",
    text: "检测概况：检出气孔 12 个。",
    model: "Qwen3-4B-Q4_K_M.gguf",
    reason: "",
    elapsed_ms: 1200,
    disclaimer: "本段结论不构成缺陷等级判定。",
    ...overrides,
  };
}

describe("ReportView 预筛级别", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("grade_preliminary=true 时显示预筛标识，且结论不含合格/不合格判定", () => {
    const w = mountView(
      makeResult({ evaluable: false, joint_level: "IV", grade_preliminary: true }),
    );
    const text = w.text();
    expect(text).toContain("AI 预筛·非正式级别");
    expect(text).toContain("不构成正式评定依据");
    expect(text).toContain("不得作为验收结论");
    expect(text).not.toContain("结果不合格");
  });

  it("未请求预筛且底片不可评时不输出级别", () => {
    const w = mountView(makeResult({ evaluable: false, joint_level: null }));
    const text = w.text();
    expect(text).toContain("本片不可评片");
    expect(text).not.toContain("AI 预筛·非正式级别");
  });

  it("翻拍影像有级别：结论为翻拍降级强提示，质量判定不再写不可评片", () => {
    const w = mountView(
      makeResult({ photo_mode: true, evaluable: true, joint_level: "III", grade_preliminary: true }),
    );
    const text = w.text();
    expect(text).toContain("本片为翻拍影像");
    expect(text).toContain("AI 预筛级别为Ⅲ级");
    expect(text).toContain("翻拍影像（质量门禁降级，须人工复核）");
    expect(text).not.toContain("本片不可评片");
    expect(text).not.toContain("结果合格");
  });
});

describe("ReportView 本地大模型评片结论", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("点击后展示正文、模型与免责声明", async () => {
    vi.mocked(getReportNarrative).mockResolvedValue(narrative());
    const w = mountView(makeResult());
    await w.find(".narr-btn").trigger("click");
    await flushPromises();
    const text = w.text();
    expect(getReportNarrative).toHaveBeenCalledWith("r1");
    expect(text).toContain("检测概况：检出气孔 12 个。");
    expect(text).toContain("Qwen3-4B-Q4_K_M.gguf");
    expect(text).toContain("不构成缺陷等级判定");
  });

  it("大模型不可用时展示原因而非空白", async () => {
    vi.mocked(getReportNarrative).mockResolvedValue(
      narrative({ status: "unavailable", text: "", model: "", reason: "端点不可用：18780" }),
    );
    const w = mountView(makeResult());
    await w.find(".narr-btn").trigger("click");
    await flushPromises();
    const text = w.text();
    expect(text).toContain("AI 结论不可用");
    expect(text).toContain("端点不可用：18780");
  });

  it("传输层失败时给出重试入口", async () => {
    vi.mocked(getReportNarrative).mockRejectedValue(new Error("后端不可达"));
    const w = mountView(makeResult());
    await w.find(".narr-btn").trigger("click");
    await flushPromises();
    expect(w.text()).toContain("重试生成评片结论");
  });
});
