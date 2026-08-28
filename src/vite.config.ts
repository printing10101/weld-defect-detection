/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import { sharedVue } from "./vite.shared";

export default defineConfig({
  ...sharedVue,
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:18773",
        changeOrigin: false,
      },
    },
  },
  // T4-4 Vitest 测试基建：与 Vite/Rollup 共用同一解析链，组件/store 可被直接单测。
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    include: ["src/**/*.spec.ts"],
    css: false,
  },
});