/**
 * 底片标注叠加的坐标换算（纯函数）。
 *
 * FilmViewer 把缺陷框烧录进"处理画布"（旋转/镜像/LUT 之后的中间画布），
 * 本模块负责原图像素坐标 → 处理画布像素坐标的映射，数学与
 * FilmViewer.rebuildProcessed 的绘制管线逐项对应：
 *   居中（以影像中心为原点）→ 镜像 → 分辨率档缩放 → 旋转（90° 倍数）→ 平移到画布中心
 * 旋转只允许 0/90/180/270，框在画布空间仍为轴对齐矩形。
 */

/** 处理画布的位姿：natW/H=影像自然尺寸，w/h=画布尺寸，factor=分辨率降档系数 */
export interface OverlayPose {
  natW: number;
  natH: number;
  rot: number;
  w: number;
  h: number;
  factor: number;
  flipH: boolean;
  flipV: boolean;
}

/** 原图像素坐标 → 处理画布像素坐标 */
export function filmPointToProcessed(
  px: number,
  py: number,
  pose: OverlayPose,
): [number, number] {
  let x = (px - pose.natW / 2) * pose.factor;
  let y = (py - pose.natH / 2) * pose.factor;
  if (pose.flipH) x = -x;
  if (pose.flipV) y = -y;
  const rot = ((pose.rot % 360) + 360) % 360;
  let rx = x;
  let ry = y;
  if (rot === 90) {
    rx = -y;
    ry = x;
  } else if (rot === 180) {
    rx = -x;
    ry = -y;
  } else if (rot === 270) {
    rx = y;
    ry = -x;
  }
  return [rx + pose.w / 2, ry + pose.h / 2];
}

/** 原图像素 bbox [x,y,w,h]（先按 scaleX/Y 回正到自然尺寸）→ 画布空间轴对齐矩形 */
export function filmBBoxToProcessedRect(
  bbox: readonly [number, number, number, number],
  scaleX: number,
  scaleY: number,
  pose: OverlayPose,
): { x: number; y: number; w: number; h: number } {
  const [x1, y1] = filmPointToProcessed(bbox[0] * scaleX, bbox[1] * scaleY, pose);
  const [x2, y2] = filmPointToProcessed(
    (bbox[0] + bbox[2]) * scaleX,
    (bbox[1] + bbox[3]) * scaleY,
    pose,
  );
  return { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1) };
}
