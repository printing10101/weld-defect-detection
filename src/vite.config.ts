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
});
