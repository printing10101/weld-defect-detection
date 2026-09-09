/**
 * 底片查看同步仓库（Pinia）——单张/批量上传的影像即时汇入「底片查看」工作区。
 *：上传动作发生在检测工作区（单张 UploadPanel / 批量 BatchView），而查看在
 * 独立路由；用应用级 store 把两处上传汇成一份有序列表，用户切到「底片查看」
 * 时无需重新选文件即可查看，且停留在查看页时新到影像自动上屏。
 * Blob URL 生命周期归 store 所有：去重、容量上限驱逐、移除/清空时统一回收，
 * 查看页只引用不回收（避免路由切换 revoke 掉缩略图仍在用的 URL）。
 */
import { defineStore } from "pinia";
import { computed, ref } from "vue";
import { isBrowserDecodable } from "../services/imageFormats";
import type { FilmAnnotations } from "../types/api";

export interface ViewerFilm {
  /** 稳定标识（自增），供查看页建立"面板 ↔ 影像"归属关系 */
  id: number;
  name: string;
  /** blob: URL，store 负责回收 */
  url: string;
  size: number;
  /** WebView 能否直接解码预览（DICOM/TIFF/HEIC 等需后端转档） */
  decodable: boolean;
  addedAt: number;
}

/** 底片印字（扫描日期/编号）性质快照：批量完成后按文件名匹配回填，供查看页展示 */
export interface FilmStampInfo {
  status: string; // present | missing | unavailable | off
  text: string | null;
  orientation: "normal" | "mirrored" | null;
  needReview: boolean;
}

/** 容量上限：与单批 100 张对齐并留余量；超出按先到先驱逐（回收 Blob）。 */
export const MAX_FILMS = 120;

let nextId = 1;

export const useViewerFilmsStore = defineStore("viewerFilms", () => {
  /** 新影像在前（最新上传排最前），查看页默认展示队头。 */
  const films = ref<ViewerFilm[]>([]);

  /** 最近一次新增的影像 id（查看页据此自动上屏；-1 表示无新增）。 */
  const lastAddedId = ref(-1);

  /** 批量检测结果回填的缺陷标注：filmId → 标注集合（仅"有问题"的底片有条目；
   * 无缺陷底片不入表，即"没有问题就不标注"）。影像移除/驱逐时同步清除。 */
  const annotations = ref<Record<number, FilmAnnotations>>({});

  /** 批量检测结果回填的印字性质：filmId → 快照（识别到才展示内容） */
  const stamps = ref<Record<number, FilmStampInfo>>({});

  const count = computed(() => films.value.length);

  function findDuplicate(name: string, size: number): ViewerFilm | undefined {
    return films.value.find((f) => f.name === name && f.size === size);
  }

  /**
   * 汇入上传文件（单张传一个、批量传一组）。
   * 同名同大小的重复选择不重复入库（仍会刷新 lastAddedId 供自动上屏）；
   * WebView 解码不了的格式（DICOM/TIFF/HEIC…）同样入库，查看页给出转档提示。
   */
  function add(files: File[]): void {
    for (const file of files) {
      const dup = findDuplicate(file.name, file.size);
      if (dup) {
        lastAddedId.value = dup.id;
        continue;
      }
      const film: ViewerFilm = {
        id: nextId++,
        name: file.name,
        url: URL.createObjectURL(file),
        size: file.size,
        decodable: isBrowserDecodable(file.name),
        addedAt: Date.now(),
      };
      films.value.unshift(film);
      lastAddedId.value = film.id;
    }
    evictOverflow();
  }

  /** 超容量时从队尾（最早）驱逐并回收其 Blob。 */
  function evictOverflow(): void {
    while (films.value.length > MAX_FILMS) {
      const oldest = films.value.pop();
      if (oldest) {
        URL.revokeObjectURL(oldest.url);
        forgetAnnotations(oldest.id);
        forgetStamp(oldest.id);
      }
    }
  }

  /** 回填一幅底片的缺陷标注（整组覆盖，供批量完成后按文件名匹配写入）。 */
  function setAnnotations(filmId: number, data: FilmAnnotations): void {
    annotations.value = { ...annotations.value, [filmId]: data };
  }

  function annotationsOf(filmId: number | null | undefined): FilmAnnotations | null {
    if (filmId == null) return null;
    return annotations.value[filmId] ?? null;
  }

  function forgetAnnotations(filmId: number): void {
    if (filmId in annotations.value) {
      const next = { ...annotations.value };
      delete next[filmId];
      annotations.value = next;
    }
  }

  /** 回填一幅底片的印字性质（覆盖写，供批量完成后按文件名匹配写入）。 */
  function setStamp(filmId: number, info: FilmStampInfo): void {
    stamps.value = { ...stamps.value, [filmId]: info };
  }

  function stampOf(filmId: number | null | undefined): FilmStampInfo | null {
    if (filmId == null) return null;
    return stamps.value[filmId] ?? null;
  }

  function forgetStamp(filmId: number): void {
    if (filmId in stamps.value) {
      const next = { ...stamps.value };
      delete next[filmId];
      stamps.value = next;
    }
  }

  function remove(id: number): void {
    const idx = films.value.findIndex((f) => f.id === id);
    if (idx === -1) return;
    const [film] = films.value.splice(idx, 1);
    URL.revokeObjectURL(film.url);
    forgetAnnotations(id);
    forgetStamp(id);
    if (lastAddedId.value === id) lastAddedId.value = -1;
  }

  function clear(): void {
    for (const f of films.value) URL.revokeObjectURL(f.url);
    films.value = [];
    annotations.value = {};
    stamps.value = {};
    lastAddedId.value = -1;
  }

  return {
    films,
    lastAddedId,
    count,
    annotations,
    stamps,
    add,
    setAnnotations,
    annotationsOf,
    setStamp,
    stampOf,
    remove,
    clear,
  };
});
