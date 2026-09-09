/**
 * 标注叠加坐标换算单测：原图像素 → 处理画布（FilmViewer 烧录缺陷框用）。
 * 期望值按 canvas 变换管线（translate(中心)→rotate→scale(flip·factor)→
 * drawImage(-natW/2,-natH/2)）手工推导，保证框与影像姿态逐像素一致。
 */
import { describe, expect, it } from "vitest";
import { filmBBoxToProcessedRect, filmPointToProcessed } from "./annotationOverlay";

describe("filmPointToProcessed", () => {
  it("rot=0 无镜像：画布中心即影像中心，坐标平移对应", () => {
    const pose = { natW: 200, natH: 100, rot: 0, w: 200, h: 100, factor: 1, flipH: false, flipV: false };
    expect(filmPointToProcessed(0, 0, pose)).toEqual([0, 0]); // 左上角
    expect(filmPointToProcessed(200, 100, pose)).toEqual([200, 100]); // 右下角
    expect(filmPointToProcessed(100, 50, pose)).toEqual([100, 50]); // 中心不动
  });

  it("rot=90（顺时针）：左上角→右上角，底边→左边", () => {
    // 处理画布尺寸随旋转互换：w=100, h=200
    const pose = { natW: 200, natH: 100, rot: 90, w: 100, h: 200, factor: 1, flipH: false, flipV: false };
    expect(filmPointToProcessed(0, 0, pose)).toEqual([100, 0]); // 左上→右上
    expect(filmPointToProcessed(0, 100, pose)).toEqual([0, 0]); // 左下→左上
    expect(filmPointToProcessed(200, 100, pose)).toEqual([0, 200]); // 右下→左下
  });

  it("rot=180：对角镜像", () => {
    const pose = { natW: 200, natH: 100, rot: 180, w: 200, h: 100, factor: 1, flipH: false, flipV: false };
    expect(filmPointToProcessed(0, 0, pose)).toEqual([200, 100]);
    expect(filmPointToProcessed(200, 100, pose)).toEqual([0, 0]);
  });

  it("rot=270（逆时针）：左上角→左下角", () => {
    const pose = { natW: 200, natH: 100, rot: 270, w: 100, h: 200, factor: 1, flipH: false, flipV: false };
    expect(filmPointToProcessed(0, 0, pose)).toEqual([0, 200]);
    expect(filmPointToProcessed(200, 0, pose)).toEqual([0, 0]); // 右上→左上
  });

  it("水平镜像（rot=0）：左右翻转", () => {
    const pose = { natW: 200, natH: 100, rot: 0, w: 200, h: 100, factor: 1, flipH: true, flipV: false };
    expect(filmPointToProcessed(0, 0, pose)).toEqual([200, 0]);
    expect(filmPointToProcessed(200, 0, pose)).toEqual([0, 0]);
  });

  it("分辨率降档系数按比例缩放坐标", () => {
    const pose = { natW: 200, natH: 100, rot: 0, w: 100, h: 50, factor: 0.5, flipH: false, flipV: false };
    expect(filmPointToProcessed(100, 50, pose)).toEqual([50, 25]); // 中心
    expect(filmPointToProcessed(200, 100, pose)).toEqual([100, 50]); // 右下角
  });
});

describe("filmBBoxToProcessedRect", () => {
  it("rot=0：bbox 平移不变（含尺寸回正系数）", () => {
    const pose = { natW: 100, natH: 100, rot: 0, w: 100, h: 100, factor: 1, flipH: false, flipV: false };
    const r = filmBBoxToProcessedRect([10, 20, 30, 40], 1, 1, pose);
    expect(r).toEqual({ x: 10, y: 20, w: 30, h: 40 });
  });

  it("后端记录尺寸与解码尺寸不一致时按系数回正", () => {
    const pose = { natW: 100, natH: 100, rot: 0, w: 100, h: 100, factor: 1, flipH: false, flipV: false };
    // bbox 以 50×50 记录，实际解码 100×100 → 全部翻倍
    const r = filmBBoxToProcessedRect([5, 5, 10, 10], 2, 2, pose);
    expect(r).toEqual({ x: 10, y: 10, w: 20, h: 20 });
  });

  it("rot=90：框随影像旋转，仍为轴对齐矩形", () => {
    const pose = { natW: 200, natH: 100, rot: 90, w: 100, h: 200, factor: 1, flipH: false, flipV: false };
    // 原图贴左缘竖条 (0,0,10,100) → 顺时针转 90° 后贴顶缘横条（左缘变顶缘）
    const r = filmBBoxToProcessedRect([0, 0, 10, 100], 1, 1, pose);
    expect(r).toEqual({ x: 0, y: 0, w: 100, h: 10 });
  });

  it("镜像后框位置随翻转", () => {
    const pose = { natW: 100, natH: 100, rot: 0, w: 100, h: 100, factor: 1, flipH: true, flipV: false };
    const r = filmBBoxToProcessedRect([10, 20, 30, 40], 1, 1, pose);
    expect(r).toEqual({ x: 60, y: 20, w: 30, h: 40 }); // 100-10-30=60
  });
});
