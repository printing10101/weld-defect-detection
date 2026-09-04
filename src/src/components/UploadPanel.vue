<script setup lang="ts">
/**
 * 上传面板（设计稿：DropZone + 参数行内帮助 + 即时校验 + 预览）。
 * 数据诚实性：仅处理用户真实选择的文件；预览为该文件的真实 objectURL；
 * 校验消息针对真实文件（扩展名/大小）。不包含任何预设样例。
 */
import { computed, onUnmounted, ref } from "vue";
import { IMAGE_EXTS as FILE_EXTS } from "../services/imageFormats";

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

const isDicom = computed(() => {
  const ext = file.value?.name.split(".").pop()?.toLowerCase();
  return ext === "dcm";
});

function onPick(picked: File): void {
  const ext = picked.name.split(".").pop()?.toLowerCase() ?? "";
  if (!(FILE_EXTS as readonly string[]).includes(ext)) {
    fileErr.value = `不支持 .${ext}：请提供 DICOM(.dcm) 或常见图像格式（JPG/PNG/BMP/GIF/WebP/TIFF/HEIC 等）。`;
    return;
  }
  if (picked.size > MAX_BYTES) {
    fileErr.value = `文件 ${(picked.size / 1024 / 1024).toFixed(1)}MB 超过 50MB 上限，请压缩后重试。`;
    return;
  }
  fileErr.value = null;
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

function onSubmit(): void {
  thicknessErr.value = null;
  if (!file.value) {
    fileErr.value = "请先选择影像文件。";
    return;
  }
  if (!baseMetalThicknessMm.value.trim()) {
    thicknessErr.value = "母材厚度 T 必填（评级依据）；不填则评级将被锁定。";
    return;
  }
  const fd = new FormData();
  fd.append("image", file.value);
  fd.append("pixel_spacing_mm", pixelSpacingMm.value || "");
  fd.append("base_metal_thickness_mm", baseMetalThicknessMm.value.trim());
  if (workpieceNo.value.trim()) fd.append("workpiece_no", workpieceNo.value.trim());
  if (weldNo.value.trim()) fd.append("weld_no", weldNo.value.trim());
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
  pixelSpacingMm.value = "0.1000";
  baseMetalThicknessMm.value = "";
  workpieceNo.value = "";
  weldNo.value = "";
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
          一 · 准备底片
        </div>
        <div class="t">
          拖入或选择射线底片。支持 DICOM(.dcm) 及常见图像格式（JPG/PNG/BMP/GIF/WebP/TIFF/HEIC 等）。
        </div>
      </div>
      <div class="g">
        <div class="n">
          二 · 填写参数
        </div>
        <div class="t">
          像素标定有默认值；母材厚度 T 是评级依据，必填。缺省时仅出图谱、评级锁定。
        </div>
      </div>
      <div class="g">
        <div class="n">
          三 · 查看报告
        </div>
        <div class="t">
          提交后等待真实处理（15–30 秒），得到级别结论、缺陷解读与操作建议。
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
          常见图像与 DICOM 均可导入，自动转为灰度处理。文件 ≤ 50MB。
        </div>
        <div
          class="drop"
          @click="onClick"
          @dragover.prevent
          @drop="onDrop"
        >
          <div class="big">
            拖入底片，或点击选择
          </div>
          <div class="hint">
            影像只在本机处理，不上传任何外部服务器
          </div>
        </div>
        <input
          ref="inputEl"
          type="file"
          accept=".dcm,.dicom,.ima,.png,.jpg,.jpeg,.jfif,.bmp,.gif,.webp,.tif,.tiff,.avif,.heic,.heif,.pgm,.ppm,.pnm,.ico"
          style="display: none"
          @change="onInput"
        >
        <div
          v-if="file"
          class="preview show"
        >
          <img
            v-if="!isDicom"
            :src="previewUrl ?? undefined"
            alt="影像预览"
            class="thumb"
          >
          <span
            v-else
            class="thumb"
          >DICOM</span>
          <div class="meta">
            {{ fileMeta }}
          </div>
        </div>
        <div
          v-if="fileErr"
          class="err show"
        >
          ⚠ {{ fileErr }}
        </div>
      </div>

      <div class="grow">
        <div class="field">
          <label for="spacing">像素标定（mm/px）</label>
          <input
            id="spacing"
            v-model="pixelSpacingMm"
          >
          <div class="why">
            默认 0.1000 mm/px；若底片带标尺可覆盖。用于把像素尺寸换算为真实当量。
          </div>
        </div>
        <div class="field">
          <label for="thick">母材厚度 T（mm）<span class="req">*</span></label>
          <input
            id="thick"
            v-model="baseMetalThicknessMm"
            placeholder="如 20"
          >
          <div class="why">
            评级必须（NB/T47013.2 按 T 分档评定区与限值）。
          </div>
          <div
            v-if="thicknessErr"
            class="err show"
          >
            ⚠ {{ thicknessErr }}
          </div>
        </div>
        <div class="field">
          <label for="wp">工件号（可选）</label>
          <input
            id="wp"
            v-model="workpieceNo"
            placeholder="如 WP-7781"
          >
        </div>
        <div class="field">
          <label for="wn">焊口编号（可选）</label>
          <input
            id="wn"
            v-model="weldNo"
            placeholder="如 W-12"
          >
        </div>
        <button
          class="btn"
          type="button"
          :disabled="!file"
          @click="onSubmit"
        >
          开始检测 →
        </button>
      </div>
    </div>
  </div>
</template>
