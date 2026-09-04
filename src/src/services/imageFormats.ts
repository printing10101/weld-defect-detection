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
