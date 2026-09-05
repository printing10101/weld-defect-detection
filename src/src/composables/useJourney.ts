/**
 * 上传 → 处理 → 报告 的旅程状态机。
 * 所有状态与内容均来自真实数据流：
 * - file / sourceUrl：用户真实上传的文件（objectURL 预览原图）；
 * - submit(fd)：把 UploadPanel 构造的真实 FormData 交给后端 POST /api/v1/report；
 * - result：后端真实响应（ReportOut）；
 * - error：后端统一错误包的真实 message；
 * - elapsedMs：真实请求耗时计时。
 */
import { getCurrentScope, onScopeDispose, readonly, ref } from "vue";
import { toErrorMessage } from "../utils/errorMessage";
import { createReport } from "../services/api";
import type { ReportOut } from "../types/api";

export type JourneyPhase = "upload" | "processing" | "result";

export function useJourney() {
  const phase = ref<JourneyPhase>("upload");
  const file = ref<File | null>(null);
  const sourceUrl = ref<string | null>(null);
  const result = ref<ReportOut | null>(null);
  const error = ref<string | null>(null);
  const elapsedMs = ref(0);

  let timer: ReturnType<typeof setInterval> | undefined;

  // 兜底回收：处理中（上传最长 120s）切走视图时，卸载当前组件作用域——
  // 计时 interval 停掉、blob 预览 URL 释放，不再等到请求 settle。
  // getCurrentScope 守卫：测试等裸调用场景无作用域，注册会产生 Vue 警告。
  if (getCurrentScope()) {
    onScopeDispose(() => {
      if (timer) clearInterval(timer);
      timer = undefined;
      if (sourceUrl.value) URL.revokeObjectURL(sourceUrl.value);
    });
  }

  function setFile(f: File | null): void {
    if (sourceUrl.value) URL.revokeObjectURL(sourceUrl.value);
    sourceUrl.value = null;
    file.value = f;
    if (f) sourceUrl.value = URL.createObjectURL(f);
  }

  async function submit(fd: FormData): Promise<void> {
    phase.value = "processing";
    error.value = null;
    result.value = null;
    const start = performance.now();
    timer = setInterval(() => {
      elapsedMs.value = Math.round(performance.now() - start);
    }, 200);

    try {
      result.value = await createReport(fd);
      phase.value = "result";
    } catch (e) {
      error.value = toErrorMessage(e);
      phase.value = "result";
    } finally {
      if (timer) clearInterval(timer);
      timer = undefined;
    }
  }

  function reset(): void {
    if (timer) clearInterval(timer);
    timer = undefined;
    setFile(null);
    phase.value = "upload";
    error.value = null;
    result.value = null;
    elapsedMs.value = 0;
  }

  return {
    phase: readonly(phase),
    file: readonly(file),
    sourceUrl: readonly(sourceUrl),
    result: readonly(result),
    error: readonly(error),
    elapsedMs: readonly(elapsedMs),
    setFile,
    submit,
    reset,
  };
}
