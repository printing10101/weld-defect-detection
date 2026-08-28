/**
 * 后端连通性/模型加载状态（Pinia）——全局离线横幅 + 冷启动轮询的唯一状态源。
 * T4-2：把 App.vue 里的后端探测逻辑抽为应用级 store，避免全局信号处理的
 * 副作用散落在顶层组件，并让离线恢复行为可以被单元测试覆盖。
 *
 * 语义：
 * - backendDown：/health 不可达或请求超时（红色横幅，自动指数重试）；
 * - modelLoading：端口已绑定、registry 装配中（琥珀提示，模型加载可能需 1~2 分钟）。
 */
import { defineStore } from "pinia";
import { ref } from "vue";
import { BACKEND_DOWN_EVENT, BACKEND_UP_EVENT, getHealth } from "../services/api";

export const useBackendStore = defineStore("backend", () => {
  const backendDown = ref(false);
  const modelLoading = ref(false);

  /** 离线自动恢复轮询句柄：冷启动期间持续探测 /health 而非直接报错。 */
  let pollTimer: number | undefined;

  function stopPolling(): void {
    if (pollTimer !== undefined) {
      window.clearTimeout(pollTimer);
      pollTimer = undefined;
    }
  }

  function pollBackend(): void {
    stopPolling();
    pollTimer = window.setTimeout(async () => {
      try {
        const health = await getHealth();
        if (health.status === "starting") {
          // 端口已绑定、registry 装配中：继续轮询直到模型就绪。
          modelLoading.value = true;
          pollBackend();
        } else {
          modelLoading.value = false;
        }
      } catch {
        pollBackend(); // 仍不可达 → 继续等待（窗口不关闭就持续重试）。
      }
    }, 1000);
  }

  function onBackendDown(): void {
    backendDown.value = true;
    pollBackend();
  }
  function onBackendUp(): void {
    backendDown.value = false;
    modelLoading.value = false;
    stopPolling();
  }

  /** 订阅后端在线/离线全局信号；在 App 根组件挂载时调用。 */
  function bind(): void {
    window.addEventListener(BACKEND_DOWN_EVENT, onBackendDown);
    window.addEventListener(BACKEND_UP_EVENT, onBackendUp);
  }
  function unbind(): void {
    stopPolling();
    window.removeEventListener(BACKEND_DOWN_EVENT, onBackendDown);
    window.removeEventListener(BACKEND_UP_EVENT, onBackendUp);
  }
  /** 挂载即开始探测：应用刚打开时后端可能仍在冷启动。 */
  function start(): void {
    pollBackend();
  }

  return { backendDown, modelLoading, bind, unbind, start, stopPolling };
});