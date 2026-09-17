<script setup lang="ts">
/**
 * 上传面板（设计稿：DropZone + 参数行内帮助 + 即时校验 + 预览）。
 * 展示口径：仅处理用户真实选择的文件；预览为该文件的真实 objectURL；
 * 校验消息针对真实文件（扩展名/大小）。不包含任何预设样例。
 *
 * 报告补充信息（report_meta）：按《射线检测报告》样张汇总表分组录入
 * （工程信息/工件概况/技术要求/检测器材及工艺参数），全部选填；
 * 字段模式 REPORT_META_GROUPS（types/api.ts）镜像后端 meta_fields 白名单，
 * 出片时填入 PDF 汇总表对应空格，未填栏留空供机构手工补填。
 */
import { computed, nextTick, onUnmounted, ref, watch } from "vue";
import { IMAGE_ACCEPT, IMAGE_EXTS as FILE_EXTS, isBrowserDecodable } from "../services/imageFormats";
import { REPORT_META_GROUPS } from "../types/api";
import type { ReportMeta } from "../types/api";
import { useWorkspaceStore } from "../stores/workspace";

const emit = defineEmits<{
  fileChanged: [file: File | null];
  submit: [form: FormData];
}>();

const MAX_BYTES = 50 * 1024 * 1024;

const file = ref<File | null>(null);
const previewUrl = ref<string | null>(null);
const fileErr = ref<string | null>(null);
const thicknessErr = ref<string | null>(null);
const pixelSpacingMm = ref("0.1000");
const baseMetalThicknessMm = ref("");
const workpieceNo = ref("");
const weldNo = ref("");
/**
 * AI 预筛级别（默认开启）：底片质量门禁未通过时，仍计算并给出级别，但结果被
 * 强制标记为"非正式级别"（界面显示 AI 预筛标签、结论带强提示）。
 * 关闭时维持从严语义——不合格底片不输出任何级别。
 */
const allowPreliminaryGrade = ref(true);

/* ── 报告补充信息：组内 2 列紧凑栅格；默认收起，展开后按样张分区填写 ── */
const metaOpen = ref(false);
const meta = ref<ReportMeta>({});
const metaFilledCount = computed(
  () => Object.values(meta.value).filter((v) => v.trim()).length,
);

function toggleMeta(): void {
  metaOpen.value = !metaOpen.value;
}

/** WebView 能否直接解码该文件（DICOM/TIFF/HEIC/PGM 等合法上传格式不解码）。
 *  不可解码时不渲染 <img>（此前显示破图，操作员误以为文件坏了），改为占位说明。 */
const previewDecodable = computed(() => !!file.value && isBrowserDecodable(file.value.name));
const fileExtUpper = computed(() => (file.value?.name.split(".").pop() ?? "?").toUpperCase());
/** <img> 解码失败（文件损坏/扩展名与内容不符）：给出明确解释而非静默破图。 */
const previewBroken = ref(false);

function onPick(picked: File): void {
  const ext = picked.name.split(".").pop()?.toLowerCase() ?? "";
  if (!(FILE_EXTS as readonly string[]).includes(ext)) {
    fileErr.value = `不支持的格式${ext ? ` .${ext}` : ""}：请提供 DICOM(.dcm) 或常见图像格式（JPG/PNG/BMP/GIF/WebP/TIFF/HEIC 等）。`;
    return;
  }
  if (picked.size > MAX_BYTES) {
    fileErr.value = `文件 ${(picked.size / 1024 / 1024).toFixed(1)}MB 超过 50MB 上限，请压缩后重新导入。`;
    return;
  }
  fileErr.value = null;
  previewBroken.value = false;
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value);
  file.value = picked;
  previewUrl.value = URL.createObjectURL(picked);
  emit("fileChanged", picked);
}

function onDrop(e: DragEvent): void {
  e.preventDefault();
  const f = e.dataTransfer?.files?.[0];
  if (f) onPick(f);
}

function onClick(): void {
  inputEl.value?.click();
}

function onInput(e: Event): void {
  const f = (e.target as HTMLInputElement).files?.[0];
  if (f) onPick(f);
  (e.target as HTMLInputElement).value = "";
}

const inputEl = ref<HTMLInputElement | null>(null);

/** 菜单/快捷键「打开射线影像…」直达文件选择器：
 *  AppShell 经 workspace 下发意图，这里无论先挂载还是后挂载都会消费一次。 */
const workspace = useWorkspaceStore();
watch(
  () => workspace.pendingFileOpen,
  (v) => {
    if (v !== "image") return;
    workspace.pendingFileOpen = null;
    void nextTick(() => inputEl.value?.click());
  },
  { immediate: true },
);

function onSubmit(): void {
  thicknessErr.value = null;
  if (!file.value) {
    fileErr.value = "请先导入射线底片影像。";
    return;
  }
  if (!baseMetalThicknessMm.value.trim()) {
    thicknessErr.value = "母材公称厚度 T 为必填项（分级评定依据）；缺省时级别评定将被锁定。";
    return;
  }
  const fd = new FormData();
  fd.append("image", file.value);
  fd.append("pixel_spacing_mm", pixelSpacingMm.value || "");
  fd.append("base_metal_thickness_mm", baseMetalThicknessMm.value.trim());
  if (workpieceNo.value.trim()) fd.append("workpiece_no", workpieceNo.value.trim());
  if (weldNo.value.trim()) fd.append("weld_no", weldNo.value.trim());
  // AI 预筛级别开关：显式传值（后端默认 false=从严），界面默认开启以便评片员
  // 在底片质量不达标时仍能看到 AI 预筛级别（带"非正式级别"强标记）。
  fd.append("allow_preliminary_grade", allowPreliminaryGrade.value ? "true" : "false");
  // 报告补充信息：仅提交非空项（JSON 字符串，后端白名单清洗后落库/出片）
  const filled: ReportMeta = {};
  for (const g of REPORT_META_GROUPS) {
    for (const f of g.fields) {
      const v = (meta.value[f.key] ?? "").trim();
      if (v) filled[f.key] = v;
    }
  }
  if (Object.keys(filled).length > 0) {
    fd.append("report_meta", JSON.stringify(filled));
  }
  emit("submit", fd);
}

const fileMeta = computed(() => {
  if (!file.value) return "";
  const kb = (file.value.size / 1024).toFixed(0);
  return `${file.value.name} · ${kb} KB`;
});

//：objectURL 必须在释放/卸载时回收，否则浏览器内存泄漏。
function revokePreview(): void {
  if (previewUrl.value) {
    URL.revokeObjectURL(previewUrl.value);
    previewUrl.value = null;
  }
}

// 复位表单并回收预览 URL（供父组件在无重挂载场景下原地复用）。
function reset(): void {
  revokePreview();
  file.value = null;
  fileErr.value = null;
  thicknessErr.value = null;
  previewBroken.value = false;
  pixelSpacingMm.value = "0.1000";
  baseMetalThicknessMm.value = "";
  workpieceNo.value = "";
  weldNo.value = "";
  meta.value = {};
  metaOpen.value = false;
  emit("fileChanged", null);
}

onUnmounted(() => {
  revokePreview();
});

defineExpose({ reset });
</script>

<template>
  <div>
    <div class="guide">
      <div class="g">
        <div class="n">
          一 · 底片导入
        </div>
        <div class="t">
          拖入或选择射线底片。支持 DICOM(.dcm) 及常见图像格式（JPG/PNG/BMP/GIF/WebP/TIFF/HEIC 等）。
        </div>
      </div>
      <div class="g">
        <div class="n">
          二 · 工艺参数录入
        </div>
        <div class="t">
          空间像素标定有默认值；母材公称厚度 T 为分级评定必备参数。可展开填写《射线检测报告》汇总表补充信息。
        </div>
      </div>
      <div class="g">
        <div class="n">
          三 · 评定报告
        </div>
        <div class="t">
          提交后由本地推理流水线评定（约 15–30 秒），输出《射线检测报告》与底片评定表。
        </div>
      </div>
    </div>

    <div class="row">
      <div class="grow">
        <div class="chips">
          <span class="chip on">DICOM .dcm</span>
          <span class="chip on">JPG/PNG/BMP</span>
          <span class="chip on">TIFF/GIF/WebP</span>
          <span class="chip on">HEIC/AVIF</span>
        </div>
        <div class="hint">
          常见图像与 DICOM 均可导入，系统自动转换为灰度影像处理。单文件 ≤ 50MB。
        </div>
        <div
          class="drop"
          role="button"
          tabindex="0"
          aria-label="拖入或选择射线底片"
          @click="onClick"
          @keydown.enter.prevent="onClick"
          @keydown.space.prevent="onClick"
          @dragover.prevent
          @drop="onDrop"
        >
          <div class="big">
            拖入射线底片至此处，或点击选择文件
          </div>
          <div class="hint">
            影像全程本机处理，不经任何外部网络传输
          </div>
        </div>
        <input
          ref="inputEl"
          type="file"
          :accept="IMAGE_ACCEPT"
          style="display: none"
          @change="onInput"
        >
        <div
          v-if="file"
          class="preview show"
        >
          <!-- WebView 不可解码的合法上传格式（DICOM/TIFF/HEIC/PGM…）显示占位说明 -->
          <span
            v-if="!previewDecodable"
            class="thumb noimg"
            title="该格式由后端转档处理，本机不做预览；可正常提交评定"
          >{{ fileExtUpper }}</span>
          <img
            v-else-if="!previewBroken"
            :src="previewUrl ?? undefined"
            alt="影像预览"
            class="thumb"
            @error="previewBroken = true"
          >
          <span
            v-else
            class="thumb noimg"
            title="预览解码失败：文件可能损坏或扩展名与实际格式不符；仍可尝试提交评定"
          >预览失败</span>
          <div class="meta">
            {{ fileMeta }}
          </div>
          <div
            v-if="!previewDecodable || previewBroken"
            class="hint"
          >
            该格式本机不做缩略图预览（不影响评定）；评定结果页可查看完整报告。
          </div>
        </div>
        <div
          v-if="fileErr"
          class="err show"
        >
          注意：{{ fileErr }}
        </div>
      </div>

      <div class="grow">
        <div class="field">
          <label for="spacing">空间像素标定（mm/px）</label>
          <input
            id="spacing"
            v-model="pixelSpacingMm"
          >
          <div class="why">
            默认 0.1000 mm/px；底片带影像标尺时可覆盖。用于将像素尺寸换算为缺陷实际当量。
          </div>
        </div>
        <div class="field">
          <label for="thick">母材公称厚度 T（mm）<span class="req">*</span></label>
          <input
            id="thick"
            v-model="baseMetalThicknessMm"
            placeholder="如 20"
          >
          <div class="why">
            分级评定必备（NB/T 47013.2 依 T 划定评定区与各级限值）。
          </div>
          <div
            v-if="thicknessErr"
            class="err show"
          >
            注意：{{ thicknessErr }}
          </div>
        </div>
        <div class="field">
          <label for="wp">工件名称（选填）</label>
          <input
            id="wp"
            v-model="workpieceNo"
            placeholder="如 前水冷壁"
          >
        </div>
        <div class="field">
          <label for="wn">焊缝/管口编号（选填）</label>
          <input
            id="wn"
            v-model="weldNo"
            placeholder="如 QSB"
          >
        </div>
        <div class="field">
          <label
            class="check"
            for="prelim"
          >
            <input
              id="prelim"
              v-model="allowPreliminaryGrade"
              type="checkbox"
            >
            <span>
              底片质量不达标时仍输出 AI 预筛级别
              <em>级别会标注为「AI 预筛·非正式级别」，仅供参考，不得作为验收结论；取消勾选则不合格底片不输出级别。</em>
            </span>
          </label>
        </div>
        <button
          class="btn"
          type="button"
          :disabled="!file"
          @click="onSubmit"
        >
          提交评定 →
        </button>
      </div>
    </div>

    <!-- 报告补充信息：对齐《射线检测报告》样张汇总表分区，全部选填 -->
    <div class="section-h">
      <span class="no">报告</span>《射线检测报告》补充信息（选填）
    </div>
    <div class="hint">
      按正式报告首页汇总表分区录入，出片时自动填入对应栏位；留空栏打印后可手工补填。
    </div>
    <div class="meta-head">
      <button
        class="btn link"
        type="button"
        @click="toggleMeta"
      >
        {{ metaOpen ? "▾ 收起补充信息" : "▸ 展开补充信息" }}
        <template v-if="metaFilledCount > 0">
          　· 已填 {{ metaFilledCount }} 项
        </template>
      </button>
    </div>
    <div
      v-if="metaOpen"
      class="fgroups"
    >
      <fieldset
        v-for="g in REPORT_META_GROUPS"
        :key="g.title"
        class="fset"
      >
        <legend>{{ g.title }}</legend>
        <div class="fgrid">
          <div
            v-for="f in g.fields"
            :key="f.key"
            class="fitem"
          >
            <label :for="`meta-${f.key}`">{{ f.label }}</label>
            <input
              :id="`meta-${f.key}`"
              v-model="meta[f.key]"
              :placeholder="f.ph ?? ''"
            >
          </div>
        </div>
      </fieldset>
    </div>
  </div>
</template>
