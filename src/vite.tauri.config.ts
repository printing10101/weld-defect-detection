import { defineConfig } from "vite";
import { sharedVue } from "./vite.shared";

// 打包版前端（Tauri webview 内）无 vite dev 代理，必须直连本机后端绝对地址。
// VITE_API_BASE 可由外部环境变量覆盖（如 CI 自定义端口）。
process.env.VITE_API_BASE = process.env.VITE_API_BASE ?? "http://127.0.0.1:18773/api/v1";

export default defineConfig({
  ...sharedVue,
  // 生产构建不产出 sourcemap，避免通过桌面包泄露前端源码（§E6）。
  // emptyOutDir：两种构建共用 dist，打包前清空防陈旧 hash 资产混装入包。
  build: { emptyOutDir: true, sourcemap: false },
});
