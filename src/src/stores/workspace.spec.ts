/** workspace store 单元测试（T4-1 / T4-4）：操作员状态的持久化与响应式一致性。 */
import { createPinia, setActivePinia } from "pinia";
import { beforeEach, describe, expect, it } from "vitest";
import { useWorkspaceStore } from "./workspace";

describe("workspace store", () => {
  beforeEach(() => {
    localStorage.clear();
    setActivePinia(createPinia());
  });

  it("未设置操作员时回退为 local", () => {
    const s = useWorkspaceStore();
    expect(s.operator).toBe("local");
  });

  it("setOperator 会 trim 并持久化到 localStorage", () => {
    const s = useWorkspaceStore();
    s.setOperator("  张三  ");
    expect(s.operator).toBe("张三");
    expect(localStorage.getItem("scan_operator_name")).toBe("张三");
  });

  it("空名回退 local 并清除存储", () => {
    const s = useWorkspaceStore();
    s.setOperator("李四");
    s.setOperator("   ");
    expect(s.operator).toBe("local");
    expect(localStorage.getItem("scan_operator_name")).toBeNull();
  });

  it("重启（新活跃 pinia）后从 localStorage 恢复操作员", () => {
    localStorage.setItem("scan_operator_name", "赵五");
    const s = useWorkspaceStore();
    expect(s.operator).toBe("赵五");
  });
});