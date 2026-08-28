/**
 * 工作台路由（Vue Router）——四个工作区由 URL 驱动。
 *：用真实路由取代 AppShell 里基于 ref 的 v-show 手工切换，
 * 使「当前工作区」成为可寻址/可回溯的导航状态，为后续深链与多文档范式留出扩展位。
 * 采用 hash 历史：Tauri 生产以本地文件运行，hash 路由免去服务端 rewrite 配置。
 * createAppRouter(history?) 工厂便于单元测试注入 memory 历史。
 */
import { createRouter, createWebHashHistory, type RouteRecordRaw } from "vue-router";
import type { ViewId } from "../types/api";
import ArchiveView from "../views/ArchiveView.vue";
import BatchView from "../views/BatchView.vue";
import DeviceView from "../views/DeviceView.vue";
import JourneyView from "../views/JourneyView.vue";
import StdEvalView from "../views/StdEvalView.vue";
import ViewerView from "../views/ViewerView.vue";

/** 路由名与 ViewId 一一对应，AppShell 用 route.name 直接得到当前工作区。 */
export const routes: RouteRecordRaw[] = [
  { path: "/", redirect: "/journey" },
  { path: "/journey", name: "journey", component: JourneyView },
  { path: "/batch", name: "batch", component: BatchView },
  { path: "/archive", name: "archive", component: ArchiveView },
  { path: "/device", name: "device", component: DeviceView },
  { path: "/viewer", name: "viewer", component: ViewerView },
  { path: "/std-eval", name: "std-eval", component: StdEvalView },
];

export function createAppRouter(history = createWebHashHistory()) {
  return createRouter({ history, routes });
}

/** 应用级单例（生产/开发用）；测试请走 createAppRouter(createMemoryHistory)。 */
export const router = createAppRouter();

/** 把路由名映射回操作层约定的 ViewId（兜底到单张检测）。 */
export function routeNameToViewId(name: unknown): ViewId {
  if (
    name === "batch" ||
    name === "archive" ||
    name === "device" ||
    name === "journey" ||
    name === "viewer" ||
    name === "std-eval"
  ) {
    return name;
  }
  return "journey";
}