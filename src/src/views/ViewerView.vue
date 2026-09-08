<script setup lang="ts">
/**
 * 底片查看工作区（DB50/T 1807-2025  图像操作 + 双片对比）。
 * 单片：全部查看/滤波操作；双片：左右两窗并排，缩放/平移/旋转联动（可关）。
 * 即时同步：单张/批量上传的影像经 viewerFilms store 汇入此处——缩略图条点选
 * 即看，停留本页时新到影像自动上屏，无需重新选文件。
 * 数据诚实性：仅显示用户真实上传/选择的影像，不预置任何样例。
 */
import { onMounted, onUnmounted, ref, watch } from "vue";
import FilmViewer from "../components/FilmViewer.vue";
import { imagePreviewUrl } from "../services/api";
import { IMAGE_ACCEPT } from "../services/imageFormats";
import { useViewerFilmsStore, type ViewerFilm } from "../stores/viewerFilms";
import type { Transform } from "../types/api";

const store = useViewerFilmsStore();

/** 面板内容来源：owned=本页 pick 创建的 Blob（替换/卸载时须回收）；
 * store 影像与档案 http URL 的生命周期不归本页，绝不在此 revoke。 */
interface PaneSource {
  url: string;
  name: string;
  owned: boolean;
  filmId?: number;
}

const paneA = ref<PaneSource | null>(null);
const paneB = ref<PaneSource | null>(null);
const dualMode = ref(false);
const synced = ref(true);
const lastTransform = ref<Transform | null>(null);
const archiveId = ref("");
const archiveErr = ref<string | null>(null);

function setPane(which: "A" | "B", src: PaneSource | null): void {
  const old = which === "A" ? paneA.value : paneB.value;
  if (old?.owned) URL.revokeObjectURL(old.url);
  if (which === "A") paneA.value = src;
  else paneB.value = src;
}

// SPA 会话内组件卸载（路由切走）也回收本页自建的 Blob；store/档案 URL 不动。
onUnmounted(() => {
  if (paneA.value?.owned) URL.revokeObjectURL(paneA.value.url);
  if (paneB.value?.owned) URL.revokeObjectURL(paneB.value.url);
});

function openFilm(film: ViewerFilm, which: "A" | "B" = "A"): void {
  setPane(which, { url: film.url, name: film.name, owned: false, filmId: film.id });
  if (which === "B") dualMode.value = true;
}

// 挂载即同步：面板为空且已有上传影像 → 自动展示最新一张可预览的。
onMounted(() => {
  if (paneA.value) return;
  const latest = store.films.find((f) => f.decodable);
  if (latest) openFilm(latest);
});

// 停留本页时新影像到达 → 自动上屏：双片且 B 空则进对比位，否则进主位。
// 不可解码格式（DICOM/TIFF…）只入缩略图条不打断当前画面，点选时给出转档提示。
watch(
  () => store.lastAddedId,
  (id) => {
    if (id < 0) return;
    const film = store.films.find((f) => f.id === id);
    if (!film || !film.decodable) return;
    if (dualMode.value && !paneB.value) openFilm(film, "B");
    else openFilm(film);
  },
);

// 影像被移除/超限驱逐（Blob 已由 store 回收）→ 对应面板同步清空，避免破图。
watch(
  () => store.films.map((f) => f.id),
  (ids) => {
    if (paneA.value?.filmId && !ids.includes(paneA.value.filmId)) setPane("A", null);
    if (paneB.value?.filmId && !ids.includes(paneB.value.filmId)) setPane("B", null);
  },
);

/** 从检测档案加载：输入影像编号，走后端 PNG 预览接口（支持 TIFF/DICOM 密文副本）。 */
function loadFromArchive(): void {
  const id = archiveId.value.trim();
  archiveErr.value = null;
  if (!id) {
    archiveErr.value = "请输入影像编号（可在档案检索中复制）。";
    return;
  }
  setPane("A", { url: imagePreviewUrl(id), name: `档案影像 ${id}`, owned: false });
}

function pick(which: "A" | "B"): void {
  const input = document.createElement("input");
  input.type = "file";
  // 白名单取唯一事实源（此前手写清单漏 .dicom/.ima/.pgm/.ppm/.pnm/.ico）
  input.accept = IMAGE_ACCEPT;
  input.onchange = () => {
    const f = input.files?.[0];
    if (!f) return;
    setPane(which, { url: URL.createObjectURL(f), name: f.name, owned: true });
    if (which === "B") dualMode.value = true;
  };
  input.click();
}

function onTransform(t: Transform): void {
  lastTransform.value = t;
}

function clear(which: "A" | "B"): void {
  setPane(which, null);
  if (which === "B") dualMode.value = false;
}

function extOf(name: string): string {
  return (name.split(".").pop() ?? "?").toUpperCase();
}
</script>

<template>
  <div>
    <h1 class="title-zine">
      底片查看
    </h1>
    <div class="lede">
      单张/批量上传的影像自动同步到这里；缩放 / 平移 / 旋转 / 镜像 / 正反片转换 / 窗位窗宽 / 锐化 / 浮雕
      （DB50/T 1807 §6.1.5）；快捷键：+ − r R i f 1 0 方向键
    </div>
    <div class="viewer-controls">
      <button @click="pick('A')">
        {{ paneA ? "更换" : "选择" }}主片…
      </button>
      <span
        v-if="paneA"
        class="fname"
      >{{ paneA.name }} <a
        href="#"
        @click.prevent="clear('A')"
      >移除</a></span>
      <input
        v-model="archiveId"
        class="aid"
        placeholder="档案影像编号…"
        @keyup.enter="loadFromArchive"
      >
      <button @click="loadFromArchive">
        从档案加载
      </button>
      <span
        v-if="archiveErr"
        class="aerr"
      >{{ archiveErr }}</span>
      <button
        :class="{ on: dualMode }"
        @click="dualMode = !dualMode"
      >
        双片对比
      </button>
      <template v-if="dualMode">
        <button @click="pick('B')">
          {{ paneB ? "更换" : "选择" }}对比片…
        </button>
        <label class="chk"><input
          v-model="synced"
          type="checkbox"
        >联动</label>
      </template>
    </div>

    <!-- 上传同步影像条：最新在前；点选上屏，双片模式下可指派为对比片 -->
    <div
      v-if="store.count"
      class="strip-wrap"
    >
      <div class="strip-head">
        已同步影像 {{ store.count }} 张<span class="faint">（来自单张/批量上传，最新在前）</span>
        <a
          href="#"
          class="clear"
          @click.prevent="store.clear()"
        >清空</a>
      </div>
      <div class="strip">
        <div
          v-for="f in store.films"
          :key="f.id"
          class="thumb"
          :class="{ active: paneA?.filmId === f.id || paneB?.filmId === f.id }"
          :title="f.name"
          @click="openFilm(f)"
        >
          <span class="timg">
            <img
              v-if="f.decodable"
              :src="f.url"
              loading="lazy"
              decoding="async"
              alt=""
            >
            <span
              v-else
              class="noimg"
              title="WebView 无法直接解码该格式，点击加载会提示先经后端转档"
            >{{ extOf(f.name) }}</span>
          </span>
          <span class="tname">{{ f.name }}</span>
          <button
            v-if="dualMode"
            class="tb"
            title="设为对比片（B）"
            @click.stop="openFilm(f, 'B')"
          >
            对比
          </button>
          <button
            class="tx"
            title="从列表移除"
            @click.stop="store.remove(f.id)"
          >
            ✕
          </button>
        </div>
      </div>
    </div>

    <div :class="dualMode ? 'dual' : 'single'">
      <FilmViewer
        :src="paneA?.url ?? null"
        :label="paneA?.name || '未加载影像'"
        :sync-transform="synced ? lastTransform : null"
        @transform-changed="onTransform"
      />
      <FilmViewer
        v-if="dualMode"
        :src="paneB?.url ?? null"
        :label="paneB?.name || '未加载对比片'"
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
  height: 62vh;
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
.strip-wrap {
  margin: 4px 0 10px;
}
.strip-head {
  font-size: 12px;
  color: #44577a;
  margin-bottom: 6px;
}
.strip-head .faint {
  color: #8a99b5;
}
.strip-head .clear {
  color: #b03030;
  margin-left: 8px;
}
.strip {
  display: flex;
  gap: 8px;
  overflow-x: auto;
  padding: 6px 2px;
  border: 1px solid rgba(120, 140, 180, 0.25);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.6);
}
.thumb {
  position: relative;
  flex: 0 0 auto;
  width: 96px;
  cursor: pointer;
  border: 2px solid transparent;
  border-radius: 6px;
  padding: 3px;
  text-align: center;
}
.thumb:hover {
  border-color: rgba(47, 107, 255, 0.5);
}
.thumb.active {
  border-color: #2f6bff;
  background: rgba(47, 107, 255, 0.08);
}
.timg {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 90px;
  height: 60px;
  border-radius: 4px;
  background: #111;
  overflow: hidden;
}
.timg img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.noimg {
  font-size: 11px;
  color: #9fb0cc;
  letter-spacing: 0.5px;
}
.tname {
  display: block;
  font-size: 11px;
  color: #44577a;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  margin-top: 3px;
}
.tb,
.tx {
  position: absolute;
  top: 4px;
  font-size: 10px;
  padding: 1px 5px;
  border-radius: 4px;
  border: 1px solid rgba(120, 140, 180, 0.4);
  background: rgba(255, 255, 255, 0.92);
  cursor: pointer;
}
.tb {
  left: 4px;
  color: #2f6bff;
}
.tx {
  right: 4px;
  color: #b03030;
}
.tb:hover {
  background: #2f6bff;
  color: #fff;
}
.tx:hover {
  background: #b03030;
  color: #fff;
}
</style>
