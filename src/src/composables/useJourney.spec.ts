/** useJourney composable 单元测试（T4-4）：上传→处理→报告 状态流转。 */
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../services/api", async (importOriginal) => {
  const orig = await importOriginal<typeof import("../services/api")>();
  return { ...orig, createReport: vi.fn() };
});

import { createReport } from "../services/api";
import { useJourney } from "./useJourney";

function okReport(): Record<string, unknown> {
  return { report_id: "r-1", image_id: "img-1", joint_level: "I" };
}

describe("useJourney", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("初始处于上传阶段且无数据", () => {
    const j = useJourney();
    expect(j.phase.value).toBe("upload");
    expect(j.file.value).toBeNull();
    expect(j.result.value).toBeNull();
    expect(j.error.value).toBeNull();
  });

  it("提交成功后进入 result 阶段并携带真实响应", async () => {
    vi.mocked(createReport).mockResolvedValue(okReport() as never);
    const j = useJourney();
    await j.submit(new FormData());
    expect(j.phase.value).toBe("result");
    expect(j.result.value?.report_id).toBe("r-1");
    expect(j.error.value).toBeNull();
  });

  it("提交失败记录错误信息", async () => {
    vi.mocked(createReport).mockRejectedValue(new Error("评级所需厚度缺失"));
    const j = useJourney();
    await j.submit(new FormData());
    expect(j.phase.value).toBe("result");
    expect(j.error.value).toBe("评级所需厚度缺失");
    expect(j.result.value).toBeNull();
  });

  it("reset 清空文件与结果回到上传阶段", () => {
    const j = useJourney();
    j.setFile(new File([""], "a.png"));
    j.reset();
    expect(j.phase.value).toBe("upload");
    expect(j.file.value).toBeNull();
    expect(j.sourceUrl.value).toBeNull();
  });
});