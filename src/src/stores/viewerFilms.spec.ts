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
});
