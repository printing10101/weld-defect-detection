<script setup lang="ts">
/** 检测旅程视图：编排 上传 → 处理中 → 报告解读（全部由 useJourney 的真实数据流驱动）。 */
import { computed } from "vue";
import { useJourney } from "../composables/useJourney";
import Stepper from "../components/Stepper.vue";
import UploadPanel from "../components/UploadPanel.vue";
import PipelineTrack from "../components/PipelineTrack.vue";
import ReportView from "../components/ReportView.vue";

const emit = defineEmits<{ archive: [] }>();

const { phase, sourceUrl, file, elapsedMs, error, result, setFile, submit, reset } = useJourney();

const step = computed<1 | 2 | 3>(() => (phase.value === "upload" ? 1 : phase.value === "processing" ? 2 : 3));

function onFileChanged(f: File | null): void {
  setFile(f);
}
function onSubmit(fd: FormData): void {
  void submit(fd);
}
</script>

<template>
  <div>
    <Stepper :current="step" />

    <!-- 阶段1：上传 -->
    <div v-if="phase === 'upload'">
      <h1 class="title-zine" data-t="开始一次检测">开始一次检测</h1>
      <div class="lede">UPLOAD · 三步完成 · 全程本地处理</div>
      <UploadPanel @file-changed="onFileChanged" @submit="onSubmit" />
    </div>

    <!-- 阶段2：处理中 -->
    <div v-else-if="phase === 'processing'">
      <h1 class="title-zine" data-t="正在处理">正在处理</h1>
      <div class="lede">PROCESSING · 影像已提交至本地流水线</div>
      <PipelineTrack status="running" :elapsed-ms="elapsedMs" :error-message="null" />
    </div>

    <!-- 阶段3：结果（失败分支） -->
    <div v-else-if="phase === 'result' && error !== null">
      <h1 class="title-zine" data-t="处理失败">处理失败</h1>
      <div class="lede">RESULT · 后端返回了真实错误信息</div>
      <PipelineTrack status="error" :elapsed-ms="elapsedMs" :error-message="error" @retry="reset()" />
    </div>

    <!-- 阶段3：结果（成功 / 需复核 / 不可评片，均为后端真实输出） -->
    <ReportView
      v-else-if="phase === 'result' && result !== null"
      :result="result"
      :source-url="sourceUrl"
      :file-name="file?.name ?? null"
      @archive="emit('archive')"
      @reset="reset()"
    />
  </div>
</template>
