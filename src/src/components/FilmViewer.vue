<script setup lang="ts">
/**
 * 底片查看器（DB50/T 1807-2025  图像操作功能清单）。
 *
 * 功能：放大/缩小/适应屏幕/1:1/还原、平移、旋转、水平/垂直镜像、
 * 正反片转换（反相）、亮度、对比度、窗位窗宽（灰度映射）、锐化、浮雕。
 * 快捷键：+/- 缩放、方向键平移、r/R 旋转、i 反相、f 适应、1 1:1、0 还原
 * （仅当鼠标悬停在本查看器且焦点不在输入控件时生效，多实例互不干扰）。
 *
 * 渲染管线：源图 → 旋转/镜像 → 灰度 LUT（反相/亮度/对比度/窗宽窗位）→
 * 卷积（锐化/浮雕）→ 结果缓存（滤波/姿态/分辨率档任一变化才重算，缩放/平移
 * 零开销；重建分辨率按显示所需降档，≥1:1 放大保持全分辨率）。
 * 说明：浏览器 webview 不解码 TIFF/DICOM，此类影像请先经后端预处理接口转为
 * PNG/JPG 预览格式（或使用已入库影像的预览 URL）。
 */
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import type { Transform } from "../types/api";

const props = withDefaults(
  defineProps<{
    src: string | null;
    /** 双片对比模式下，由父组件下发的同步变换（null=不同步） */
    syncTransform?: Transform | null;
    height?: string;
    label?: string;
  }>(),
  { syncTransform: null, height: "62vh", label: "" },
);

const emit = defineEmits<{
  /** 变换状态变化（供双片对比同步） */
  transformChanged: [t: Transform];
}>();


const canvas = ref<HTMLCanvasElement | null>(null);
const wrap = ref<HTMLDivElement | null>(null);

// ---- 变换与滤波状态 ----
const scale = ref(1);
const tx = ref(0);
const ty = ref(0);
const rotation = ref(0);
const flipH = ref(false);
const flipV = ref(false);
const invert = ref(false);
const brightness = ref(100); // %
const contrast = ref(100); // %
const winEnabled = ref(false);
const winWidth = ref(255); // 窗宽（0-255 尺度）
const winLevel = ref(128); // 窗位
const sharpen = ref(0); // 0-100
const emboss = ref(false);

const img = ref<HTMLImageElement | null>(null);
const imgErr = ref<string | null>(null);
/** 鼠标是否悬停在本查看器内：window 级快捷键只作用于悬停窗（双片对比时两
 * 实例都在 window 上监听，不限定归属会让同一按键同时作用于两窗）。 */
const hovered = ref(false);

// 处理结果缓存：仅滤波/姿态/分辨率档变化时重算
const processed = document.createElement("canvas");
// 虚拟（原图尺寸）维度：重建画布可按显示分辨率降档，变换/适应数学始终用原图尺寸
let processedVW = 0;
let processedVH = 0;
let processedKey = "";

/** 重建分辨率档（半倍频程分档，取 ≥ 所需因子的最小档）：
 * 缩小/适应态按显示所需设备像素重建——此前 8K 底片 fit 态每次滑滤波都全图
 * 卷积（单帧 ImageData ≈256MB，数百 ms 卡顿），降档后重建代价降约两个数量级；
 * ≥1:1 态档位为 1（全分辨率），放大判读精度不受影响。 */
function resBucket(factor: number): number {
  return 2 ** (Math.ceil(Math.log2(Math.max(factor, 1e-3)) * 2) / 2);
}

function expectedKey(): string {
  const dpr = window.devicePixelRatio || 1;
  const f = resBucket(Math.min(1, Math.max(scale.value * dpr, 0.02)));
  return `${filterKey.value}|res${f}`;
}

const transformState = computed<Transform>(() => ({
  scale: scale.value,
  tx: tx.value,
  ty: ty.value,
  rotation: rotation.value,
  flipH: flipH.value,
  flipV: flipV.value,
}));

const statusText = computed(
  () =>
    `${Math.round(scale.value * 100)}% · ${rotation.value}°` +
    (flipH.value ? " · 水平镜像" : "") +
    (flipV.value ? " · 垂直镜像" : "") +
    (invert.value ? " · 反相" : "") +
    (winEnabled.value ? ` · 窗宽${winWidth.value}/窗位${winLevel.value}` : ""),
);

// ---- 源图加载 ----
watch(
  () => props.src,
  (src) => {
    imgErr.value = null;
    if (!src) {
      img.value = null;
      return;
    }
    // 世代标记：快速切换 src 时旧图的 onload/onerror 可能晚到，没有守卫
    // 会把旧图画进 canvas / 用旧图的失败覆盖新图的成功。
    const mySrc = src;
    const el = new Image();
    el.onload = () => {
      if (props.src !== mySrc) return; // 已切到别的图，丢弃迟到回调
      img.value = el;
      fit();
    };
    el.onerror = () => {
      if (props.src !== mySrc) return;
      imgErr.value = "影像加载失败（TIFF/DICOM 请先用后端转换为 PNG/JPG 预览格式）";
      img.value = null;
    };
    el.src = src;
  },
  { immediate: true },
);

// ---- 滤波参数变化 → 失效缓存重绘 ----
const filterKey = computed(
  () =>
    `${img.value?.src ?? ""}|${rotation.value}|${flipH.value}|${flipV.value}|` +
    `${invert.value}|${brightness.value}|${contrast.value}|` +
    `${winEnabled.value}|${winWidth.value}|${winLevel.value}|${sharpen.value}|${emboss.value}`,
);
watch(filterKey, () => {
  processedKey = ""; // 失效
  scheduleRender();
});

// 外部同步变换下发（双片对比）
watch(
  () => props.syncTransform,
  (t) => {
    if (!t) return;
    scale.value = t.scale;
    tx.value = t.tx;
    ty.value = t.ty;
    rotation.value = t.rotation;
    flipH.value = t.flipH;
    flipV.value = t.flipV;
    // 渲染由各交互函数显式调度，watch 只改 refs 不触发重绘——补一次调度，
    // 否则 B 窗在 A 窗缩放/平移后保持旧画面（旋转/镜像恰经 filterKey 触发，
    // 行为不一致更隐蔽）。
    scheduleRender();
  },
);

function emitTransform(): void {
  emit("transformChanged", { ...transformState.value });
}

// ---- 卷积（锐化/浮雕，3×3，仅滤波变化时执行一次） ----
function applyConvolution(data: ImageData, kernel: number[], divisor: number, offset = 0): void {
  const src = new Uint8ClampedArray(data.data);
  const out = data.data;
  const w = data.width;
  const h = data.height;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      for (let c = 0; c < 3; c++) {
        let acc = 0;
        let ki = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++, ki++) {
            acc += src[((y + dy) * w + (x + dx)) * 4 + c] * kernel[ki];
          }
        }
        out[(y * w + x) * 4 + c] = acc / divisor + offset;
      }
    }
  }
}

function rebuildProcessed(): void {
  const source = img.value;
  if (!source) return;
  const rot = ((rotation.value % 360) + 360) % 360;
  const swap = rot === 90 || rot === 270;
  const natW = source.naturalWidth;
  const natH = source.naturalHeight;
  processedVW = swap ? natH : natW;
  processedVH = swap ? natW : natH;
  // 显示分辨率降档：fit/缩小态只在显示所需分辨率上做 LUT/卷积（见 expectedKey），
  // ≥1:1 时 factor=1 保持全分辨率。
  const dpr = window.devicePixelRatio || 1;
  const factor = resBucket(Math.min(1, Math.max(scale.value * dpr, 0.02)));
  const w = Math.max(1, Math.round(processedVW * factor));
  const h = Math.max(1, Math.round(processedVH * factor));
  processed.width = w;
  processed.height = h;
  const ctx = processed.getContext("2d");
  if (!ctx) return;
  ctx.save();
  ctx.translate(w / 2, h / 2);
  ctx.rotate((rot * Math.PI) / 180);
  ctx.scale(factor * (flipH.value ? -1 : 1), factor * (flipV.value ? -1 : 1));
  ctx.drawImage(source, -natW / 2, -natH / 2);
  ctx.restore();

  // 灰度 LUT：反相 / 亮度 / 对比度 / 窗宽窗位（0-255 尺度；16bit 影像已被浏览器缩到 8bit）
  const imageData = ctx.getImageData(0, 0, w, h);
  const lut = new Uint8ClampedArray(256);
  const b = brightness.value / 100;
  const c = contrast.value / 100;
  for (let v = 0; v < 256; v++) {
    let x = v;
    if (winEnabled.value) {
      const lo = winLevel.value - winWidth.value / 2;
      x = ((x - lo) / Math.max(winWidth.value, 1)) * 255; // 窗外截断
    }
    x = (x - 128) * c + 128 * b;
    if (invert.value) x = 255 - x;
    lut[v] = Math.min(255, Math.max(0, x));
  }
  const px = imageData.data;
  for (let i = 0; i < px.length; i += 4) {
    px[i] = lut[px[i]];
    px[i + 1] = lut[px[i + 1]];
    px[i + 2] = lut[px[i + 2]];
  }
  if (sharpen.value > 0) {
    const k = sharpen.value / 100;
    applyConvolution(imageData, [0, -k, 0, -k, 1 + 4 * k, -k, 0, -k, 0], 1);
  }
  if (emboss.value) {
    applyConvolution(imageData, [-2, -1, 0, -1, 1, 1, 0, 1, 2], 1, 128);
  }
  ctx.putImageData(imageData, 0, 0);
  processedKey = expectedKey();
}

// ---- 绘制（缩放/平移路径零重算） ----
let raf = 0;
function scheduleRender(): void {
  if (raf) return;
  raf = requestAnimationFrame(() => {
    raf = 0;
    render();
  });
}

function render(): void {
  const el = canvas.value;
  if (!el) return;
  const rect = el.parentElement?.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const cssW = Math.max(1, Math.floor(rect?.width ?? 640));
  const cssH = Math.max(1, Math.floor(rect?.height ?? 480));
  if (el.width !== cssW * dpr || el.height !== cssH * dpr) {
    el.width = cssW * dpr;
    el.height = cssH * dpr;
  }
  const ctx = el.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);
  if (!img.value) return;
  if (processedKey !== expectedKey()) rebuildProcessed();
  ctx.save();
  ctx.translate(cssW / 2 + tx.value, cssH / 2 + ty.value);
  ctx.scale(scale.value, scale.value);
  // 重建画布可能已按显示分辨率降档，绘制时以虚拟（原图）尺寸为目标大小
  ctx.drawImage(processed, -processedVW / 2, -processedVH / 2, processedVW, processedVH);
  ctx.restore();
}

// ---- 视图操作 ----
function fit(): void {
  const el = canvas.value;
  if (!el || !img.value) return;
  const rect = el.parentElement?.getBoundingClientRect();
  const cw = rect?.width ?? 640;
  const ch = rect?.height ?? 480;
  const pad = 0.94;
  // 以原图自然尺寸计算适应比例（此前用处理画布尺寸，换图瞬间会读到上一张的陈旧尺寸）
  const rot = ((rotation.value % 360) + 360) % 360;
  const swap = rot % 180 !== 0;
  const iw = swap ? img.value.naturalHeight : img.value.naturalWidth;
  const ih = swap ? img.value.naturalWidth : img.value.naturalHeight;
  scale.value = Math.min(cw / iw, ch / ih) * pad;
  tx.value = 0;
  ty.value = 0;
  scheduleRender();
  emitTransform();
}

function zoom1to1(): void {
  scale.value = 1;
  tx.value = 0;
  ty.value = 0;
  scheduleRender();
  emitTransform();
}

function reset(): void {
  rotation.value = 0;
  flipH.value = false;
  flipV.value = false;
  invert.value = false;
  brightness.value = 100;
  contrast.value = 100;
  winEnabled.value = false;
  winWidth.value = 255;
  winLevel.value = 128;
  sharpen.value = 0;
  emboss.value = false;
  fit();
}

function zoomBy(factor: number, cx = 0, cy = 0): void {
  const next = Math.min(32, Math.max(0.02, scale.value * factor));
  const ratio = next / scale.value;
  tx.value = cx + (tx.value - cx) * ratio;
  ty.value = cy + (ty.value - cy) * ratio;
  scale.value = next;
  scheduleRender();
  emitTransform();
}

function rotate(dir: 1 | -1): void {
  rotation.value = (rotation.value + dir * 90 + 360) % 360;
  scheduleRender();
  emitTransform();
}

function toggleWin(): void {
  winEnabled.value = !winEnabled.value;
  if (winEnabled.value) {
    winWidth.value = 255;
    winLevel.value = 128;
  }
}

// ---- 交互：滚轮缩放 + 拖拽平移 ----
let dragging = false;
let lastX = 0;
let lastY = 0;

function onWheel(e: WheelEvent): void {
  e.preventDefault();
  const rect = canvas.value?.getBoundingClientRect();
  const cx = rect ? e.clientX - rect.left - rect.width / 2 : 0;
  const cy = rect ? e.clientY - rect.top - rect.height / 2 : 0;
  zoomBy(e.deltaY < 0 ? 1.1 : 1 / 1.1, cx, cy);
}

function onDown(e: MouseEvent): void {
  dragging = true;
  lastX = e.clientX;
  lastY = e.clientY;
}

function onMove(e: MouseEvent): void {
  if (!dragging) return;
  tx.value += e.clientX - lastX;
  ty.value += e.clientY - lastY;
  lastX = e.clientX;
  lastY = e.clientY;
  scheduleRender();
}

function onUp(): void {
  if (dragging) {
    dragging = false;
    emitTransform();
  }
}

function isFormControl(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  if (!el || !el.tagName) return false;
  return (
    el.tagName === "INPUT" ||
    el.tagName === "TEXTAREA" ||
    el.tagName === "SELECT" ||
    el.isContentEditable
  );
}

function onKey(e: KeyboardEvent): void {
  // 输入控件内不抢占按键：此前在"档案影像编号"等输入框打字，
  // r/i/f/0/方向键会被查看器拦截并 preventDefault，输入被吞。
  if (isFormControl(e.target)) return;
  // 修饰键组合让给全局/浏览器快捷键（AppShell 的 Ctrl+1..6 切工作区、
  // Ctrl+R 等）：不排除时两边监听都会执行，切工作区的同时查看器被偷改缩放。
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  // 快捷键只作用于鼠标悬停的本查看器（双片对比时两实例都在 window 监听）。
  if (!hovered.value || !canvas.value || !img.value) return;
  const step = 24;
  switch (e.key) {
    case "+":
    case "=":
      zoomBy(1.15);
      break;
    case "-":
      zoomBy(1 / 1.15);
      break;
    case "ArrowLeft":
      tx.value -= step;
      scheduleRender();
      emitTransform();
      break;
    case "ArrowRight":
      tx.value += step;
      scheduleRender();
      emitTransform();
      break;
    case "ArrowUp":
      ty.value -= step;
      scheduleRender();
      emitTransform();
      break;
    case "ArrowDown":
      ty.value += step;
      scheduleRender();
      emitTransform();
      break;
    case "r":
      rotate(1);
      break;
    case "R":
      rotate(-1);
      break;
    case "i":
      invert.value = !invert.value;
      break;
    case "f":
      fit();
      break;
    case "1":
      zoom1to1();
      break;
    case "0":
      reset();
      break;
    default:
      return;
  }
  e.preventDefault();
}

onMounted(() => {
  window.addEventListener("keydown", onKey);
  window.addEventListener("resize", scheduleRender);
  scheduleRender();
});

onBeforeUnmount(() => {
  window.removeEventListener("keydown", onKey);
  window.removeEventListener("resize", scheduleRender);
  if (raf) cancelAnimationFrame(raf);
});
</script>

<template>
  <div
    class="film-viewer"
    :style="{ height }"
    @mouseenter="hovered = true"
    @mouseleave="hovered = false"
  >
    <div class="fv-toolbar">
      <button title="放大（+）" @click="zoomBy(1.2)">＋</button>
      <button title="缩小（-）" @click="zoomBy(1 / 1.2)">－</button>
      <button title="适应屏幕（f）" @click="fit()">适应</button>
      <button title="1:1 尺寸（1）" @click="zoom1to1()">1:1</button>
      <button title="逆时针旋转（R）" @click="rotate(-1)">↺</button>
      <button title="顺时针旋转（r）" @click="rotate(1)">↻</button>
      <button title="水平镜像" :class="{ on: flipH }" @click="flipH = !flipH">⇋</button>
      <button title="垂直镜像" :class="{ on: flipV }" @click="flipV = !flipV">⇅</button>
      <button title="正反片转换（i）" :class="{ on: invert }" @click="invert = !invert">◐</button>
      <button title="窗位窗宽" :class="{ on: winEnabled }" @click="toggleWin">窗</button>
      <button title="还原（0）" @click="reset()">还原</button>
    </div>
    <div ref="wrap" class="fv-stage">
      <canvas
        ref="canvas"
        @wheel="onWheel"
        @mousedown="onDown"
        @mousemove="onMove"
        @mouseup="onUp"
        @mouseleave="onUp"
      />
      <div v-if="imgErr" class="fv-error">{{ imgErr }}</div>
      <div v-else-if="!img" class="fv-hint">{{ props.label || "未加载影像" }}</div>
      <div v-else-if="props.label" class="fv-label">{{ props.label }}</div>
    </div>
    <div class="fv-filters">
      <label>亮度<input v-model.number="brightness" type="range" min="20" max="300" step="5" /></label>
      <label>对比度<input v-model.number="contrast" type="range" min="20" max="300" step="5" /></label>
      <template v-if="winEnabled">
        <label>窗宽<input v-model.number="winWidth" type="range" min="1" max="512" step="1" /></label>
        <label>窗位<input v-model.number="winLevel" type="range" min="0" max="255" step="1" /></label>
      </template>
      <label>锐化<input v-model.number="sharpen" type="range" min="0" max="100" step="5" /></label>
      <label class="chk"><input v-model="emboss" type="checkbox" />浮雕</label>
    </div>
    <div class="fv-status">{{ statusText }}</div>
  </div>
</template>

<style scoped>
.film-viewer {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}
.fv-toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}
.fv-toolbar button {
  padding: 4px 10px;
  cursor: pointer;
}
.fv-toolbar button.on {
  background: #2c5aa0;
  color: #fff;
}
.fv-stage {
  position: relative;
  flex: 1;
  min-height: 0;
  background: #111;
  overflow: hidden;
  border-radius: 4px;
}
.fv-stage canvas {
  width: 100%;
  height: 100%;
  display: block;
  cursor: grab;
}
.fv-hint,
.fv-error,
.fv-label {
  position: absolute;
  top: 8px;
  left: 10px;
  color: #ddd;
  font-size: 12px;
  pointer-events: none;
}
.fv-error {
  color: #ff8f8f;
}
.fv-filters {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  font-size: 12px;
  color: #555;
}
.fv-filters label {
  display: flex;
  align-items: center;
  gap: 4px;
}
.fv-filters input[type="range"] {
  width: 90px;
}
.fv-status {
  font-size: 11px;
  color: #888;
}
</style>
