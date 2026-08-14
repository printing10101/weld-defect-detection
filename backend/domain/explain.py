"""可解释性热力图（§12.3，M6 实现）。

规格书要求"类激活热力图（Grad-CAM / 注意力）叠加原图"，帮助评片员秒懂
模型关注区。本环境部署推理为 ONNX 路径（无梯度回传，且 default venv 无
torch/onnxruntime 训练栈），真 Grad-CAM 不可行——与 M4「SAM→Contour 精修」
同思路，这里提供**模型无关的注意力近似**：

1. 在目标缺陷 bbox ROI 内计算局部显著性（Sobel 梯度幅值 + 高斯平滑）——
   缺陷边缘/纹理强处即"模型关注区"的合理代理；
2. 叠加原图（JET 伪彩 + alpha 混合），供人工复核视图查看。

诚实的替代声明：真 Grad-CAM 需要分类头梯度回传（torch 后端 + 保留梯度），
属未来工作（检测器换成 torch 推理后可升级）。本近似零新权重、可离线、可单测。
"""

from __future__ import annotations

import cv2
import numpy as np

from backend.domain.dto import Detection


def attention_heatmap(
    image: np.ndarray,
    detection: Detection,
    *,
    sigma: float = 4.0,
    alpha: float = 0.55,
) -> np.ndarray:
    """对单个缺陷生成注意力热力图叠加图（BGR uint8）。

    image     : 单通道灰度（原图，与检测一致）；会先归一化到 8bit。
    detection : 目标缺陷（取其 bbox 为 ROI）。
    sigma     : 显著性高斯平滑核（越大越扩散）。
    alpha     : 热力图叠加不透明度。

    返回与 image 同尺寸的 BGR 叠加图（热区=红，冷区≈原图）。
    """
    gray = _to_uint8(image)
    x, y, w, h = _clip_roi(detection, gray.shape)
    if w <= 2 or h <= 2:  # 退化框：无可视区域
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 1) ROI 内局部显著性：Sobel 梯度幅值（缺陷边缘/纹理=关注区代理）
    roi = gray[y : y + h, x : x + w].astype(np.float32)
    gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
    saliency = cv2.magnitude(gx, gy)
    saliency = cv2.GaussianBlur(saliency, (0, 0), sigmaX=sigma)
    # 2) 归一化到 [0,1]（ROI 内相对强度）
    hi = float(saliency.max()) if saliency.size else 0.0
    saliency = saliency / hi if hi > 1e-6 else np.zeros_like(saliency)

    # 3) JET 伪彩 + 叠加原图（alpha 混合），热区保留色彩、冷区接近原图
    heat_full = np.zeros(gray.shape, dtype=np.float32)
    heat_full[y : y + h, x : x + w] = saliency
    heat_color = cv2.applyColorMap(_to_uint8(heat_full), cv2.COLORMAP_JET)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(base, 1.0 - alpha, heat_color, alpha, 0.0)
    return overlay


def _clip_roi(detection: Detection, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    """把检测框裁剪到图像范围内（越界坐标会导致切片错误/黑边）。"""
    h_img, w_img = shape[:2]
    x = int(max(0, min(detection.bbox.x, w_img - 1)))
    y = int(max(0, min(detection.bbox.y, h_img - 1)))
    x1 = int(max(0, min(detection.bbox.x + detection.bbox.w, w_img)))
    y1 = int(max(0, min(detection.bbox.y + detection.bbox.h, h_img)))
    return x, y, x1 - x, y1 - y


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """统一到 uint8 灰阶（16bit 底片直接做色彩映射会溢出/淡到不可见）。"""
    if image.dtype == np.uint8:
        return image
    arr = image.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return ((arr - lo) * (255.0 / (hi - lo))).astype(np.uint8)
