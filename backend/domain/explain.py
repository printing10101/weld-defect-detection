"""可解释性热力图。

设计文档要求"类激活热力图（Grad-CAM / 注意力）叠加原图"，帮助评片员秒懂
模型关注区。实现按可用信息分两级：

1. **真 Grad-CAM**（``domain/detect/gradcam.py``）：检测器以 torch 后端加载
   （训练/验证工作站）时自动启用——对目标缺陷类别回传梯度生成类激活图。
2. **Sobel 显著性近似**（模型无关回退）：部署推理为 ONNX 路径（无梯度回传），
   在目标缺陷 bbox ROI 内计算局部显著性（Sobel 梯度幅值 + 高斯平滑）——
   缺陷边缘/纹理强处即"模型关注区"的合理代理。

任何 Grad-CAM 失败（缺 torch/权重结构变化）自动回退近似路径，并叠加原图
（JET 伪彩 + alpha 混合），供人工复核视图查看。
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
    cam_model: object | None = None,
    model_image: np.ndarray | None = None,
) -> np.ndarray:
    """对单个缺陷生成注意力热力图叠加图（BGR uint8）。

    image     : 单通道灰度（原图，与检测一致）；会先归一化到 8bit，叠加基底；
    detection : 目标缺陷（取其 bbox 为 ROI）；
    sigma     : 显著性高斯平滑核（越大越扩散，仅近似路径）；
    alpha     : 热力图叠加不透明度；
    cam_model : torch 检测模型（YoloDetector.cam_model）。提供时先尝试真
                Grad-CAM，失败/None 回退 Sobel 显著性近似；
    model_image : 模型实际看到的图（预处理后）。缺省用 image——CAM 输入应与
                推理输入一致，调用方传入 enhanced 更准确。

    返回与 image 同尺寸的 BGR 叠加图（热区=红，冷区≈原图）。
    """
    gray = _to_uint8(image)
    x, y, w, h = _clip_roi(detection, gray.shape)
    if w <= 2 or h <= 2:  # 退化框：无可视区域
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if cam_model is not None:
        from backend.domain.detect.gradcam import grad_cam_map

        cam_src = model_image if model_image is not None else image
        cam = grad_cam_map(cam_model, cam_src, (x, y, w, h), class_id=int(detection.class_id.value))
        if cam is not None:
            return _blend(gray, cam, alpha)
    return _sobel_overlay(gray, (x, y, w, h), sigma=sigma, alpha=alpha)


def _blend(gray: np.ndarray, heat: np.ndarray, alpha: float) -> np.ndarray:
    """热力图（∈[0,1]，全图尺寸）JET 伪彩叠加灰度基底。"""
    heat_color = cv2.applyColorMap(_to_uint8(heat), cv2.COLORMAP_JET)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(base, 1.0 - alpha, heat_color, alpha, 0.0)


def _sobel_overlay(
    gray: np.ndarray,
    roi: tuple[int, int, int, int],
    *,
    sigma: float,
    alpha: float,
) -> np.ndarray:
    """Sobel 梯度显著性近似（模型无关回退路径，原实现保留）。"""
    x, y, w, h = roi
    roi_img = gray[y : y + h, x : x + w].astype(np.float32)
    gx = cv2.Sobel(roi_img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi_img, cv2.CV_32F, 0, 1, ksize=3)
    saliency = cv2.magnitude(gx, gy)
    saliency = cv2.GaussianBlur(saliency, (0, 0), sigmaX=sigma)
    hi = float(saliency.max()) if saliency.size else 0.0
    saliency = saliency / hi if hi > 1e-6 else np.zeros_like(saliency)
    heat_full = np.zeros(gray.shape, dtype=np.float32)
    heat_full[y : y + h, x : x + w] = saliency
    return _blend(gray, heat_full, alpha)


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
