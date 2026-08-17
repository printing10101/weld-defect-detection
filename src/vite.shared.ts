import vue from "@vitejs/plugin-vue";

/**
 * 开发（vite.config.ts）与打包（vite.tauri.config.ts）共享的 Vite 基础配置，
 * 避免两份配置各自重复声明插件（§E6 去重）。
 */
export const sharedVue = {
  plugins: [vue()],
} as const;
