"""真 Grad-CAM 类激活热力图（torch/Ultralytics 路径）。

domain/explain.py「诚实替代声明」的兑现：当检测器以 torch 后端加载
（训练/验证工作站，ml 可选依赖已装）时，对目标缺陷生成真 Grad-CAM——
对目标类在目标框内的最高锚框得分回传梯度，GAP 池化得通道权重，ReLU 加权
检测头之前的最后一层卷积特征图（YOLOv8/v11 惯例取 SPPF/Neck 深层特征）。

ONNX Runtime 部署路径无梯度回传，仍走 explain.py 的 Sobel 显著性近似；
本模块任何失败（缺 torch/权重结构变化/梯度断裂）一律返回 None 由调用方
回退——可解释性是辅助功能，绝不阻断主推理链路。
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

_LOG = logging.getLogger("scandetection.gradcam")


def letterbox_rgb(
    image_rgb: np.ndarray, size: int = 640
) -> tuple[np.ndarray, float, int, int, int, int]:
    """与部署推理一致的 letterbox（等比缩放 + 灰 114 填充）。

    2D 灰度图自动转 3 通道。返回 (方形图, 缩放比 r, 顶部填充 top, 左侧填充 left,
    缩放后高 nh, 缩放后宽 nw)。
    """
    if image_rgb.ndim == 2:
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
    h, w = image_rgb.shape[:2]
    r = min(size / w, size / h)
    nw, nh = max(1, int(w * r)), max(1, int(h * r))
    resized = cv2.resize(image_rgb, (nw, nh))
    top = (size - nh) // 2
    left = (size - nw) // 2
    padded = cv2.copyMakeBorder(
        resized,
        top,
        size - nh - top,
        left,
        size - nw - left,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return padded, r, top, left, nh, nw


def _candidate_layers(det_model: object) -> list[object]:
    """Grad-CAM 候选层：检测头（Detect）之前的全部 Conv2d（按注册顺序）。

    YOLO 多尺度预测：不同缺陷的得分可能走 P3/P4/P5 不同路径，单层 hook 会
    选中"梯度恰好为零"的分支（如缺陷走 P3 而目标层在 P5 分支）。因此对
    检测头之前的**全部**卷积层挂钩，backward 后再选梯度非零的最深层
    （见 grad_cam_map）。检测头分支（cv2/cv3/dfl）的末端 1×1 卷积输出已是
    逐锚框回归/分类语义，空间结构过粗，不作为候选。
    """
    from torch import nn

    root = getattr(det_model, "model", None)
    if root is None or len(root) == 0:
        return []
    layers: list[object] = []
    for child in list(root.children())[:-1]:  # 末位为 Detect/分类头，排除
        for module in child.modules():
            if isinstance(module, nn.Conv2d):
                layers.append(module)
    return layers


def grad_cam_map(
    model: object,
    image_rgb: np.ndarray,
    bbox_xywh: tuple[float, float, float, float],
    *,
    class_id: int | None = None,
    input_size: int = 640,
) -> np.ndarray | None:
    """计算目标框的 Grad-CAM 灰度图（与原图同尺寸，float32 ∈ [0,1]）。

    model     : Ultralytics YOLO 实例（torch 后端已加载）；
    image_rgb : 与模型输入通道序一致的图（RGB；灰度图传 2D 自动转 3 通道）；
    bbox_xywh : 目标框 (x, y, w, h)，原图坐标系；
    class_id  : 目标类通道；None 时取框内得分最高的类（与推理 argmax 语义一致）。

    任何失败返回 None（调用方回退近似路径）。
    """
    try:
        import torch

        det_model = getattr(model, "model", None)
        if det_model is None:
            return None
        src = image_rgb
        if src.ndim == 2:
            src = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)
        h, w = src.shape[:2]
        padded, r, top, left, nh, nw = letterbox_rgb(src, input_size)
        blob = torch.from_numpy(padded.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
        # 输入参与计算图：即使权重被冻结（推理导出场景），梯度仍可回传到激活
        blob.requires_grad_(True)

        # 多层 hook：缺陷得分可能走 P3/P4/P5 任一分支，先对全部候选层挂钩，
        # backward 后再选"梯度非零的最深层"（最高层语义特征）计算 CAM。
        candidates = _candidate_layers(det_model)
        if not candidates:
            return None
        acts: dict[int, torch.Tensor] = {}
        grads: dict[int, torch.Tensor] = {}
        handles: list[object] = []

        def _make(layer_id: int):
            def _fwd(_m: object, _inp: object, out: object) -> None:
                acts[layer_id] = out  # type: ignore[assignment]

            def _bwd(_m: object, _gin: object, gout: object) -> None:
                grads[layer_id] = gout[0]  # type: ignore[assignment]

            return _fwd, _bwd

        for i, layer in enumerate(candidates):
            fwd, bwd = _make(i)
            handles.append(layer.register_forward_hook(fwd))  # type: ignore[attr-defined]
            handles.append(layer.register_full_backward_hook(bwd))  # type: ignore[attr-defined]
        try:
            det_model.eval()  # type: ignore[attr-defined]
            with torch.enable_grad():
                preds = det_model(blob)  # type: ignore[attr-defined,operator]
            out = preds[0] if isinstance(preds, tuple) else preds
            arr = out[0] if isinstance(out, (list, tuple)) else out
            arr = arr[0]  # 去 batch 维 → [4+nc, N] 或 [N, 4+nc]
            if arr.dim() == 2 and arr.shape[0] > arr.shape[1]:
                arr = arr.T
            if arr.dim() != 2 or arr.shape[0] < 5:
                return None
            scores_all = arr[4:]  # [nc, N]，锚框中心在 arr[0]/arr[1]（input 坐标）
            cx, cy = arr[0], arr[1]
            # 目标框映射到 letterbox 坐标系，取框内锚框得分；框内无锚框
            # （极小目标/坐标漂移）时退化为全图最高分。
            bx1, by1 = bbox_xywh[0] * r + left, bbox_xywh[1] * r + top
            bx2, by2 = bx1 + bbox_xywh[2] * r, by1 + bbox_xywh[3] * r
            in_box = (cx >= min(bx1, bx2)) & (cx <= max(bx1, bx2))
            in_box &= (cy >= min(by1, by2)) & (cy <= max(by1, by2))
            if class_id is not None and 0 <= class_id < scores_all.shape[0]:
                scores = scores_all[class_id]
            else:
                # 类别未知：取框内锚框上得分最高的类（与推理 argmax 语义一致）
                per_class = (
                    scores_all[:, in_box].max(dim=1).values
                    if bool(in_box.any())
                    else scores_all.max(dim=1).values
                )
                scores = scores_all[int(per_class.argmax().item())]
            scalar = scores[in_box].max() if bool(in_box.any()) else scores.max()
            det_model.zero_grad()  # type: ignore[attr-defined]
            scalar.backward()
        finally:
            for hd in handles:
                hd.remove()  # type: ignore[attr-defined]

        # 选层：梯度非零的最深层（注册顺序靠后 = 语义层级越高）。
        # 骨干层在所有路径上梯度必非零；走 P3 的缺陷会延伸到 P3 颈部分支。
        chosen: tuple[torch.Tensor, torch.Tensor] | None = None
        for i in range(len(candidates) - 1, -1, -1):
            if i not in acts or i not in grads:
                continue
            a, g = acts[i].detach(), grads[i].detach()
            if float(g.abs().max()) <= 0.0:
                continue
            chosen = (a, g)
            break
        if chosen is None:
            return None
        a, g = chosen
        weights = g.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * a).sum(dim=1, keepdim=True))[0, 0].cpu().numpy()
        hi = float(cam.max())
        if hi <= 1e-6:
            return None
        cam = cam / hi
        # letterbox 坐标 → 原图坐标：裁掉填充区再放大回原尺寸
        cam_cropped = cam[top : top + nh, left : left + nw]
        if cam_cropped.size == 0:
            return None
        cam_full = cv2.resize(cam_cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(cam_full, 0.0, 1.0).astype(np.float32)
    except Exception:  # noqa: BLE001 — 可解释性辅助功能，任何失败静默回退
        _LOG.warning("Grad-CAM 失败，回退显著性近似", exc_info=True)
        return None
