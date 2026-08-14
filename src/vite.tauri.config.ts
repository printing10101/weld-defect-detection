import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

// 打包版前端（Tauri webview 内）无 vite dev 代理，必须直连本机后端绝对地址。
// VITE_API_BASE 可由外部环境变量覆盖（如 CI 自定义端口）。
process.env.VITE_API_BASE = process.env.VITE_API_BASE ?? "http://127.0.0.1:18773";

export default defineConfig({
  plugins: [vue()],
  build: { emptyOutDir: false },
});
