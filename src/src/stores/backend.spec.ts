/** backend store 单元测试（T4-2 / T4-4）：离线/在线全局信号与冷启动轮询。 */
import { createPinia, setActivePinia } from "pinia";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { BACKEND_DOWN_EVENT, BACKEND_UP_EVENT } from "../services/api";
import { useBackendStore } from "./backend";

vi.mock("../services/api", async (importOriginal) => {
  const orig = await importOriginal<typeof import("../services/api")>();
  return { ...orig, getHealth: vi.fn() };
});

describe("backend store", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    setActivePinia(createPinia());
  });

  it("BACKEND_DOWN 信号置离线态，BACKEND_UP 信号恢复", () => {
    const s = useBackendStore();
    s.bind();
    window.dispatchEvent(new CustomEvent(BACKEND_DOWN_EVENT));
    expect(s.backendDown).toBe(true);

    window.dispatchEvent(new CustomEvent(BACKEND_UP_EVENT));
    expect(s.backendDown).toBe(false);
    expect(s.modelLoading).toBe(false);
    s.unbind();
  });

  it("unbind 后不再响应离线信号", () => {
    const s = useBackendStore();
    s.bind();
    s.unbind();
    window.dispatchEvent(new CustomEvent(BACKEND_DOWN_EVENT));
    expect(s.backendDown).toBe(false);
  });

  it("轮询：starting → 模型加载中；就绪 → 清除加载提示", async () => {
    vi.useFakeTimers();
    try {
      const { getHealth } = await import("../services/api");
      const health = vi.mocked(getHealth);
      health.mockResolvedValueOnce({ status: "starting" } as never);

      const s = useBackendStore();
      s.start();
      await vi.advanceTimersByTimeAsync(1000);
      expect(s.modelLoading).toBe(true);

      health.mockReset();
      health.mockResolvedValueOnce({ status: "ok" } as never);
      await vi.advanceTimersByTimeAsync(1000);
      expect(s.modelLoading).toBe(false);
    } finally {
      vi.useRealTimers();
    }
  });
});