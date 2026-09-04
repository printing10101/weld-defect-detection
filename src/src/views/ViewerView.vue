<script setup lang="ts">
/**
 * 底片查看工作区（DB50/T 1807-2025  图像操作 + 双片对比）。
 * 单片：全部查看/滤波操作；双片：左右两窗并排，缩放/平移/旋转联动（可关）。
 * 数据诚实性：仅显示用户真实选择的本地文件，不预置任何样例。
 */
import { ref } from "vue";
import FilmViewer from "../components/FilmViewer.vue";
import { imagePreviewUrl } from "../services/api";
import type { Transform } from "../types/api";

const urlA = ref<string | null>(null);
const urlB = ref<string | null>(null);
const nameA = ref("");
const nameB = ref("");
const dualMode = ref(false);
const synced = ref(true);
const lastTransform = ref<Transform | null>(null);
const archiveId = ref("");
const archiveErr = ref<string | null>(null);

/** 从检测档案加载：输入影像编号，走后端 PNG 预览接口（支持 TIFF/DICOM 密文副本）。 */
function loadFromArchive(): void {
  const id = archiveId.value.trim();
  archiveErr.value = null;
  if (!id) {
    archiveErr.value = "请输入影像编号（可在档案检索中复制）。";
    return;
  }
  if (urlA.value) URL.revokeObjectURL(urlA.value);
  urlA.value = imagePreviewUrl(id);
  nameA.value = `档案影像 ${id}`;
}

function pick(which: "A" | "B"): void {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = ".dcm,.png,.jpg,.jpeg,.jfif,.bmp,.gif,.webp,.tif,.tiff,.avif,.heic,.heif";
  input.onchange = () => {
    const f = input.files?.[0];
    if (!f) return;
    const url = URL.createObjectURL(f);
    if (which === "A") {
      if (urlA.value) URL.revokeObjectURL(urlA.value);
      urlA.value = url;
      nameA.value = f.name;
    } else {
      if (urlB.value) URL.revokeObjectURL(urlB.value);
      urlB.value = url;
      nameB.value = f.name;
      dualMode.value = true;
    }
  };
  input.click();
}

function onTransform(t: Transform): void {
  lastTransform.value = t;
}

function clear(which: "A" | "B"): void {
  if (which === "A") {
    if (urlA.value) URL.revokeObjectURL(urlA.value);
    urlA.value = null;
    nameA.value = "";
  } else {
    if (urlB.value) URL.revokeObjectURL(urlB.value);
    urlB.value = null;
    nameB.value = "";
    dualMode.value = false;
  }
}
</script>

<template>
  <div>
    <h1 class="title-zine">底片查看</h1>
    <div class="lede">
      缩放 / 平移 / 旋转 / 镜像 / 正反片转换 / 窗位窗宽 / 锐化 / 浮雕（DB50/T 1807 §6.1.5）；
      快捷键：+ − r R i f 1 0 方向键
    </div>
    <div class="viewer-controls">
      <button @click="pick('A')">{{ urlA ? "更换" : "选择" }}主片…</button>
      <span v-if="nameA" class="fname">{{ nameA }} <a @click.prevent="clear('A')" href="#">移除</a></span>
      <input
        v-model="archiveId"
        class="aid"
        placeholder="档案影像编号…"
        @keyup.enter="loadFromArchive"
      >
      <button @click="loadFromArchive">从档案加载</button>
      <span
        v-if="archiveErr"
        class="aerr"
      >{{ archiveErr }}</span>
      <button :class="{ on: dualMode }" @click="dualMode = !dualMode">双片对比</button>
      <template v-if="dualMode">
        <button @click="pick('B')">{{ urlB ? "更换" : "选择" }}对比片…</button>
        <label class="chk"><input v-model="synced" type="checkbox" />联动</label>
      </template>
    </div>
    <div :class="dualMode ? 'dual' : 'single'">
      <FilmViewer
        :src="urlA"
        :label="nameA || '未加载影像'"
        :sync-transform="synced ? lastTransform : null"
        @transform-changed="onTransform"
      />
      <FilmViewer
        v-if="dualMode"
        :src="urlB"
        :label="nameB || '未加载对比片'"
        :sync-transform="synced ? lastTransform : null"
        @transform-changed="onTransform"
      />
    </div>
  </div>
</template>

<style scoped>
.viewer-controls {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  margin: 10px 0;
}
.viewer-controls button.on {
  background: #2c5aa0;
  color: #fff;
}
.fname {
  font-size: 12px;
  color: #666;
}
.fname a {
  color: #2c5aa0;
  margin-left: 4px;
}
.chk {
  font-size: 12px;
  color: #555;
  display: flex;
  align-items: center;
  gap: 4px;
}
.single,
.dual {
  display: grid;
  gap: 8px;
  height: 66vh;
}
.single {
  grid-template-columns: 1fr;
}
.dual {
  grid-template-columns: 1fr 1fr;
}
.aid {
  font-size: 13px;
  padding: 4px 8px;
  width: 220px;
}
.aerr {
  color: #b03030;
  font-size: 12px;
}
</style>
