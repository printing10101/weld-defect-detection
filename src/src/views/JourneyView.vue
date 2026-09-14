<script setup lang="ts">
/** 检测旅程视图：编排 上传 → 处理中 → 报告解读（全部由 useJourney 的真实数据流驱动）。
 *  上传面板用 v-show 常驻挂载：处理失败「返回修改」保留已选底片与全部参数，
 *  不再强制重选重填。 */
import { computed, ref } from "vue";
import { useJourney } from "../composables/useJourney";
import { useViewerFilmsStore } from "../stores/viewerFilms";
import Stepper from "../components/Stepper.vue";
import UploadPanel from "../components/UploadPanel.vue";
import PipelineTrack from "../components/PipelineTrack.vue";
import ReportView from "../components/ReportView.vue";

const emit = defineEmits<{ archive: [] }>();

const viewerFilms = useViewerFilmsStore();
const uploadPanel = ref<InstanceType<typeof UploadPanel> | null>(null);

const { phase, sourceUrl, file, elapsedMs, error, result, setFile, submit, backToUpload, reset } =
  useJourney();

const step = computed<1 | 2 | 3>(() => (phase.value === "upload" ? 1 : phase.value === "processing" ? 2 : 3));

function onFileChanged(f: File | null): void {
  setFile(f);
  // 选片即同步到底片观察（reset 走 null，不入库）
  if (f) viewerFilms.add([f]);
}
function onSubmit(fd: FormData): void {
  void submit(fd);
}
/** 失败后返回上传页：保留底片与参数，仅清结果态。 */
function onRetry(): void {
  backToUpload();
}
/** 「新建评定任务」：整单清空（含上传面板的表单与预览）。 */
function onNewTask(): void {
  reset();
  uploadPanel.value?.reset();
}
</script>

<template>
  <div>
    <Stepper :current="step" />

    <!-- 阶段1：上传（v-show 常驻：失败返回时保留底片与参数） -->
    <div v-show="phase === 'upload'">
      <h1
        class="title-zine"
        data-t="开始一次检测"
      >
        新建单幅评定任务
      </h1>
      <div class="lede">
        导入射线底片 → 本地推理流水线评定 → 签发评片报告（数据全程本地化处理）
      </div>
      <UploadPanel
        ref="uploadPanel"
        @file-changed="onFileChanged"
        @submit="onSubmit"
      />
    </div>

    <!-- 阶段2：处理中 -->
    <div v-if="phase === 'processing'">
      <h1
        class="title-zine"
        data-t="正在处理"
      >
        自动评定进行中
      </h1>
      <div class="lede">
        底片已提交至本地推理流水线，正在执行缺陷检出与标准符合性评定
      </div>
      <PipelineTrack
        status="running"
        :elapsed-ms="elapsedMs"
        :error-message="null"
      />
    </div>

    <!-- 阶段3：结果（失败分支） -->
    <div v-else-if="phase === 'result' && error !== null">
      <h1
        class="title-zine"
        data-t="处理失败"
      >
        评定任务失败
      </h1>
      <div class="lede">
        本次评定未完成；已选底片与参数仍保留，可修改后重新提交
      </div>
      <PipelineTrack
        status="error"
        :elapsed-ms="elapsedMs"
        :error-message="error"
        @retry="onRetry"
      />
    </div>

    <!-- 阶段3：结果（成功 / 需复核 / 不可评片，均为后端真实输出） -->
    <ReportView
      v-else-if="phase === 'result' && result !== null"
      :result="result"
      :source-url="sourceUrl"
      :file-name="file?.name ?? null"
      @archive="emit('archive')"
      @reset="onNewTask"
    />
  </div>
</template>
