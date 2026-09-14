/**
 * 长任务在跑标记（模块级单例）：单幅评定提交中 / 批量批次执行中置位。
 * App.vue 据此注册 beforeunload 拦截，防止操作员误关窗口丢掉进行中的评定
 * （批量可从历史批次续接，单幅进行中关窗即丢）。
 */
import { ref } from "vue";

export const busyTask = ref<"batch" | "journey" | null>(null);
