/**
 * Vitest 全局装夹：
 * - afterEach 卸载挂载组件，避免跨用例泄漏；
 * - jsdom 未实现 URL.createObjectURL/revokeObjectURL，打桩以支持 useJourney 的预览流；
 * - localStorage 垫片：兜底 jsdom 文件型 localStorage（依赖本机路径，CI/离线不可靠）。
 */
import { enableAutoUnmount } from "@vue/test-utils";
import { afterEach, beforeEach, vi } from "vitest";

function installLocalStorageStub(): void {
  if (
    typeof globalThis.localStorage !== "undefined" &&
    typeof globalThis.localStorage.clear === "function"
  ) {
    return;
  }
  const store = new Map<string, string>();
  const ls: Storage = {
    getItem: (k: string) => (store.has(k) ? store.get(k)! : null),
    setItem: (k: string, v: string) => {
      store.set(k, String(v));
    },
    removeItem: (k: string) => {
      store.delete(k);
    },
    clear: () => {
      store.clear();
    },
    key: (i: number) => [...store.keys()][i] ?? null,
    get length() {
      return store.size;
    },
  };
  Object.defineProperty(globalThis, "localStorage", {
    value: ls,
    writable: true,
    configurable: true,
  });
}

beforeEach(() => {
  installLocalStorageStub();
  if (typeof URL.createObjectURL !== "function") {
    URL.createObjectURL = vi.fn(() => "blob:jest-mock");
    URL.revokeObjectURL = vi.fn();
  }
});

// @vue/test-utils 2.4.9 起 cleanup 导出移除，改用 enableAutoUnmount 在 afterEach 统一卸载挂载组件。
enableAutoUnmount(afterEach);