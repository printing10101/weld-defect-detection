<script setup lang="ts">
/**
 * 底片查看工作区（DB50/T 1807-2025  图像操作 + 双片对比）。
 * 单片：全部查看/滤波操作；双片：左右两窗并排，缩放/平移/旋转联动（可关）。
 * 数据诚实性：仅显示用户真实选择的本地文件，不预置任何样例。
 */
import { onUnmounted, ref } from "vue";
import FilmViewer from "../components/FilmViewer.vue";
import { imagePreviewUrl } from "../services/api";
import { IMAGE_ACCEPT } from "../services/imageFormats";
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

// 当前 URL 是否为 blob:（档案预览是后端 http URL——revokeObjectURL 对它
// 无意义；只 revoke 真正的 blob，避免生命周期语义混乱）。
const blobA = ref(false);
const blobB = ref(false);

function setUrl(which: "A" | "B", url: string | null, isBlob: boolean): void {
  const old = which === "A" ? urlA.value : urlB.value;
  const oldIsBlob = which === "A" ? blobA.value : blobB.value;
  if (old && oldIsBlob) URL.revokeObjectURL(old);
  if (which === "A") {
    urlA.value = url;
    blobA.value = isBlob;
  } else {
    urlB.value = url;
    blobB.value = isBlob;
  }
}

// SPA 会话内组件卸载（路由切走）也回收，避免反复"选择主片"累积泄漏 Blob。
onUnmounted(() => {
  if (urlA.value && blobA.value) URL.revokeObjectURL(urlA.value);
  if (urlB.value && blobB.value) URL.revokeObjectURL(urlB.value);
});

/** 从检测档案加载：输入影像编号，走后端 PNG 预览接口（支持 TIFF/DICOM 密文副本）。 */
function loadFromArchive(): void {
  const id = archiveId.value.trim();
  archiveErr.value = null;
  if (!id) {
    archiveErr.value = "请输入影像编号（可在档案检索中复制）。";
    return;
  }
  setUrl("A", imagePreviewUrl(id), false);
  nameA.value = `档案影像 ${id}`;
}

function pick(which: "A" | "B"): void {
  const input = document.createElement("input");
  input.type = "file";
  // 白名单取唯一事实源（此前手写清单漏 .dicom/.ima/.pgm/.ppm/.pnm/.ico）
  input.accept = IMAGE_ACCEPT;
  input.onchange = () => {
    const f = input.files?.[0];
    if (!f) return;
    const url = URL.createObjectURL(f);
    setUrl(which, url, true);
    if (which === "A") {
      nameA.value = f.name;
    } else {
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
  setUrl(which, null, false);
  if (which === "A") {
    nameA.value = "";
  } else {
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
