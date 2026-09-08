/**
 * 影像扩展名白名单（唯一事实源，镜像 backend/configs/default.yaml upload.allowed_suffixes）。
 * 上传面板与批量导入共用；新增格式须与后端白名单同步。
 */
export const IMAGE_EXTS = [
  "dcm", "dicom", "ima", "png", "jpg", "jpeg", "jfif", "bmp", "gif", "webp",
  "tif", "tiff", "avif", "heic", "heif", "pgm", "ppm", "pnm", "ico",
] as const;

export type ImageExt = (typeof IMAGE_EXTS)[number];

export function isImageExt(ext: string): boolean {
  return (IMAGE_EXTS as readonly string[]).includes(ext);
}

/** <input accept> 属性值（含点前缀）。 */
export const IMAGE_ACCEPT = IMAGE_EXTS.map((e) => `.${e}`).join(",");

/**
 * WebView（Chromium）能直接解码为 <img>/<canvas> 的子集。
 * DICOM/TIFF/HEIC/PGM 等合法上传格式但浏览器不解码：底片查看中这些影像
 * 须经后端预览接口转档（或由 FilmViewer 明确提示加载失败），不可假装可预览。
 */
export const BROWSER_DECODABLE_EXTS = [
  "png", "jpg", "jpeg", "jfif", "bmp", "gif", "webp", "avif", "ico",
] as const;

export function isBrowserDecodable(name: string): boolean {
  const ext = name.split(".").pop()?.toLowerCase() ?? "";
  return (BROWSER_DECODABLE_EXTS as readonly string[]).includes(ext);
}
