/** 工作台路由单元测试（ / ）：四工作区注册、默认重定向、直接导航。 */
import { createMemoryHistory } from "vue-router";
import { describe, expect, it } from "vitest";
import { createAppRouter, routeNameToViewId, routes } from "./index";

const WORKSPACES = ["journey", "batch", "archive", "device"] as const;

describe("router", () => {
  it("注册四个工作区命名路由", () => {
    const names = routes.filter((r) => r.name).map((r) => r.name);
    expect(names).toEqual(expect.arrayContaining([...WORKSPACES]));
  });

  it("默认重定向到单张检测工作区", async () => {
    const r = createAppRouter(createMemoryHistory());
    await r.push("/");
    expect(r.currentRoute.value.name).toBe("journey");
  });

  it("可直达任意工作区", async () => {
    const r = createAppRouter(createMemoryHistory());
    await r.push("/device");
    expect(r.currentRoute.value.name).toBe("device");
    await r.push("/batch");
    expect(r.currentRoute.value.name).toBe("batch");
  });

  it("routeNameToViewId 兜底未知/空名为 journey", () => {
    expect(routeNameToViewId("archive")).toBe("archive");
    expect(routeNameToViewId(undefined)).toBe("journey");
    expect(routeNameToViewId("nope")).toBe("journey");
  });
});