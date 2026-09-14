import { createPinia } from "pinia";
import { createApp } from "vue";
import App from "./App.vue";
import { router } from "./router";
import "./styles/pro.css";

const app = createApp(App);
app.use(createPinia());
app.use(router);
app.mount("#app");

// 全局兜底：未捕获的 Promise 拒绝/异常至少进控制台日志（此前个别 void promise
// 静默失败，现场排查无线索）；不弹窗打断操作员，业务错误仍由各视图自行展示。
window.addEventListener("unhandledrejection", (e) => {
  console.error("[unhandledrejection]", e.reason);
});
window.addEventListener("error", (e) => {
  console.error("[window.error]", e.message);
});
