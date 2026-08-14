/**
 * 上传 → 处理 → 报告 的旅程状态机。
 * 所有状态与内容均来自真实数据流：
 * - file / sourceUrl：用户真实上传的文件（objectURL 预览原图）；
 * - submit(fd)：把 UploadPanel 构造的真实 FormData 交给后端 POST /api/v1/report；
 * - result：后端真实响应（ReportOut）；
 * - error：后端统一错误包的真实 message；
 * - elapsedMs：真实请求耗时计时。
 */
import { readonly, ref } from "vue";
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
      error.value = e instanceof Error ? e.message : String(e);
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
