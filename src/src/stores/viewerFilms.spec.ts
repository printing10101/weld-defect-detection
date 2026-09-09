/**
 * viewerFilms store 单元测试：上传 → 底片查看即时同步的数据面。
 * 覆盖：单张/批量汇入（最新在前）、同名同大小去重、可解码判定、
 * 移除/清空的 Blob 回收、超容量驱逐最早条目。
 */
import { createPinia, setActivePinia } from "pinia";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { MAX_FILMS, useViewerFilmsStore } from "./viewerFilms";

function makeFile(name: string, size = 10): File {
  return new File([new Uint8Array(size)], name, { type: "image/png" });
}

describe("viewerFilms store", () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    const revoked: string[] = [];
    let seq = 0;
    vi.stubGlobal("URL", {
      ...URL,
      createObjectURL: vi.fn(() => `blob:test-${++seq}`),
      revokeObjectURL: vi.fn((u: string) => revoked.push(u)),
    });
  });

  it("单张上传入库：最新在前，PNG 判为可直接预览", () => {
    const s = useViewerFilmsStore();
    s.add([makeFile("a.png")]);
    expect(s.count).toBe(1);
    expect(s.films[0]).toMatchObject({ name: "a.png", decodable: true });
    expect(s.films[0].url).toMatch(/^blob:/);
    expect(s.lastAddedId).toBe(s.films[0].id);
  });

  it("批量上传按选择顺序倒序排列（最新在前）且 DICOM 标记为需转档", () => {
    const s = useViewerFilmsStore();
    s.add([makeFile("1.png"), makeFile("2.dcm"), makeFile("3.jpg")]);
    expect(s.films.map((f) => f.name)).toEqual(["3.jpg", "2.dcm", "1.png"]);
    expect(s.films[1].decodable).toBe(false);
    expect(s.count).toBe(3);
  });

  it("同名同大小重复选择不重复入库，仅刷新最近上屏标记", () => {
    const s = useViewerFilmsStore();
    s.add([makeFile("a.png", 42)]);
    const first = s.films[0];
    s.add([makeFile("b.png"), makeFile("a.png", 42)]);
    expect(s.count).toBe(2);
    expect(s.lastAddedId).toBe(first.id);
    // 新文件 b.png 入队头，重复的 a.png 保留原条目不重复建 Blob
    expect(s.films.map((f) => f.name)).toEqual(["b.png", "a.png"]);
    expect(vi.mocked(URL.createObjectURL)).toHaveBeenCalledTimes(2);
  });

  it("remove 回收对应 Blob 并清空最近上屏标记", () => {
    const s = useViewerFilmsStore();
    s.add([makeFile("a.png")]);
    const film = s.films[0];
    s.remove(film.id);
    expect(s.count).toBe(0);
    expect(vi.mocked(URL.revokeObjectURL).mock.calls[0][0]).toBe(film.url);
    expect(s.lastAddedId).toBe(-1);
  });

  it("clear 回收全部 Blob", () => {
    const s = useViewerFilmsStore();
    s.add([makeFile("a.png"), makeFile("b.png")]);
    const urls = s.films.map((f) => f.url);
    s.clear();
    expect(s.count).toBe(0);
    const revoked = vi.mocked(URL.revokeObjectURL).mock.calls.map((c) => c[0]);
    for (const u of urls) expect(revoked).toContain(u);
  });

  it(`超过 ${MAX_FILMS} 张时按先到先驱逐并回收`, () => {
    const s = useViewerFilmsStore();
    const files = Array.from({ length: MAX_FILMS + 1 }, (_, i) => makeFile(`f${i}.png`));
    s.add(files);
    expect(s.count).toBe(MAX_FILMS);
    expect(s.films[0].name).toBe(`f${MAX_FILMS}.png`); // 最新保留
    expect(s.films.at(-1)!.name).toBe("f1.png"); // f0 最早被驱逐
    const revoked = vi.mocked(URL.revokeObjectURL).mock.calls.map((c) => c[0]);
    expect(revoked.length).toBe(1);
  });

  // 批量标注约定：有问题的底片回填缺陷框，无问题底片无条目（不标注）
  it("setAnnotations 回填缺陷标注，annotationsOf 按 id 取用", () => {
    const s = useViewerFilmsStore();
    s.add([makeFile("a.png"), makeFile("b.png")]);
    const [a, b] = [s.films[0], s.films[1]];
    expect(s.annotationsOf(a.id)).toBeNull(); // 未检出的底片无标注
    s.setAnnotations(a.id, {
      imageW: 100,
      imageH: 80,
      reportId: "r1",
      boxes: [{ id: "d1", classId: 4, bbox: [10, 10, 20, 20], confidence: 0.9, needReview: false }],
    });
    expect(s.annotationsOf(a.id)?.boxes).toHaveLength(1);
    expect(s.annotationsOf(b.id)).toBeNull(); // 无问题底片保持无标注
    expect(s.annotationsOf(null)).toBeNull(); // 档案影像等无 filmId 的面板安全返回
  });

  it("remove/clear/驱逐时同步清除对应标注", () => {
    const s = useViewerFilmsStore();
    s.add([makeFile("a.png")]);
    const film = s.films[0];
    s.setAnnotations(film.id, {
      imageW: 100,
      imageH: 80,
      reportId: "r1",
      boxes: [{ id: "d1", classId: 0, bbox: [0, 0, 5, 5], confidence: 0.5, needReview: false }],
    });
    s.remove(film.id);
    expect(s.annotationsOf(film.id)).toBeNull();

    s.add([makeFile("b.png")]);
    const b = s.films[0];
    s.setAnnotations(b.id, { imageW: 1, imageH: 1, reportId: "r2", boxes: [] });
    s.clear();
    expect(s.annotationsOf(b.id)).toBeNull();
  });

  // 批量印字性质回填：日期/编号（正/镜像）作为底片性质供查看页查阅
  it("setStamp 回填印字性质，stampOf 按 id 取用", () => {
    const s = useViewerFilmsStore();
    s.add([makeFile("a.png"), makeFile("b.png")]);
    const [a, b] = [s.films[0], s.films[1]];
    expect(s.stampOf(a.id)).toBeNull(); // 未回填（单张上传未识别）安全返回
    s.setStamp(a.id, {
      status: "present",
      text: "2023-08-12 No.0421",
      orientation: "mirrored",
      needReview: false,
    });
    expect(s.stampOf(a.id)).toMatchObject({ orientation: "mirrored", needReview: false });
    expect(s.stampOf(b.id)).toBeNull();
    expect(s.stampOf(null)).toBeNull();
  });

  it("remove/clear 时同步清除印字性质", () => {
    const s = useViewerFilmsStore();
    s.add([makeFile("a.png")]);
    const film = s.films[0];
    s.setStamp(film.id, { status: "missing", text: null, orientation: null, needReview: true });
    s.remove(film.id);
    expect(s.stampOf(film.id)).toBeNull();

    s.add([makeFile("b.png")]);
    const b = s.films[0];
    s.setStamp(b.id, { status: "present", text: "0421", orientation: "normal", needReview: false });
    s.clear();
    expect(s.stampOf(b.id)).toBeNull();
  });
});
