"""底片区域检测：从翻拍照片/扫描件中分割胶片有效区。纯算法，无 I/O。

背景：现场底片常以"相机拍灯箱"方式数字化——四周是过曝亮背景与边框，
胶片只占画面一部分。黑度/IQI/质量门禁若在整图上计算，亮背景会把平均
灰阶大幅拉高（黑度被低估数倍），IQI 自动定位也易锁到边框/焊缝亮线。
先分割胶片区，再在其上做门禁与检测，是翻拍影像可评的前置条件。

方法：高斯去噪 → Otsu 反相阈值（暗=胶片）→ 闭/开运算（核随图幅自适应，
弥合胶片内亮焊缝造成的割裂）→ 最大连通域 → 掩膜；外接框取腐蚀后的
"核区"（剥离翻拍边框/扫描白边，避免框被边框撑满整幅）。

is_photo 判据（翻拍影像特征）：
- 胶片占画面比例 ≤ max_photo_area_frac（满幅 = 扫描件，非翻拍）；且
- 亮背景（灰度 ≥ surround_bright_gray）占整幅比例 ≥ surround_min_frac
  （灯箱过曝环绕）。阈值取得较低：误判翻拍的代价只是门禁降级+强制人工
  复核（安全方向），漏判翻拍则会让影像被误阻断。

已知边界：is_photo=True 仅代表"绝对光学黑度不可测"（8bit 容器黑度上限
2.41，相机曝光/伽马又破坏线性映射），不代表底片合格；门禁是否降级由
density.photo_policy 配置决定，本模块不输出任何合格性结论。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FilmRegionCfg:
    """底片区域分割配置（infra/config.FilmRegionCfg 的领域镜像）。"""

    min_area_frac: float = 0.08  # 胶片区最小占画面比例（低于视为分割失败）
    max_photo_area_frac: float = 0.88  # 胶片占比低于此值才可能判翻拍（满幅=扫描件）
    surround_bright_gray: float = 200.0  # 环绕背景"亮"的灰度下限（灯箱过曝特征）
    surround_min_frac: float = 0.05  # 亮背景占**整幅**最小占比（低阈偏安全：误判翻拍仅多人工复核）


@dataclass(frozen=True)
class FilmRegion:
    """分割出的胶片区：外接框 + 像素级掩膜 + 翻拍判定。"""

    x: int
    y: int
    w: int
    h: int
    area_frac: float  # 胶片面积 / 整幅面积
    is_photo: bool  # True=翻拍影像（绝对黑度不可测）
    mask: np.ndarray  # bool (H, W)，胶片区=True（未腐蚀，完整胶片域）


def detect_film_region(gray: np.ndarray, cfg: FilmRegionCfg | None = None) -> FilmRegion | None:
    """检测胶片有效区。分割失败（无显著暗色连通域）返回 None，调用方按整图处理。"""
    cfg = cfg or FilmRegionCfg()
    if gray is None or gray.size == 0:
        return None
    arr = np.asarray(gray)
    h, w = arr.shape[:2]
    img8 = _to_uint8(arr)
    blur = cv2.GaussianBlur(img8, (5, 5), 0)
    # 反相二值化：暗区=胶片，亮区=灯箱/扫描白边。Otsu 对双峰（亮背景/暗胶片）
    # 稳健；均匀图无峰时阈值落在中位，最大暗连通域往往过小而被 min_area_frac 拒绝。
    _, dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = _refine_dark_core(blur, dark, min_frac=cfg.min_area_frac)
    k = max(3, int(min(h, w) * 0.02) | 1)
    kernel = np.ones((k, k), np.uint8)
    # 闭运算弥合胶片内亮焊缝/亮线的割裂；开运算去掉边框毛刺与孤立暗斑。
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark)
    if n < 2:
        return None
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[idx, cv2.CC_STAT_AREA])
    area_frac = area / float(h * w)
    if area_frac < cfg.min_area_frac:
        return None
    mask = labels == idx
    x, y, bw, bh = _core_bbox(mask, margin_frac=0.03, fallback=stats[idx])
    # 亮背景占比按**整幅**计：扫描件白边（线性边距 ~2-3%）占整幅比例很小，
    # 不触发翻拍判定；灯箱翻拍的亮背景通常占整幅 20% 以上。
    bright_frac = float((img8.reshape(-1) >= cfg.surround_bright_gray).mean())
    is_photo = area_frac <= cfg.max_photo_area_frac and bright_frac >= cfg.surround_min_frac
    return FilmRegion(x=x, y=y, w=bw, h=bh, area_frac=area_frac, is_photo=is_photo, mask=mask)


def _refine_dark_core(blur: np.ndarray, dark: np.ndarray, min_frac: float) -> np.ndarray:
    """两级 Otsu：暗类内部再分一次，取更暗子类为胶片本体。

    翻拍照片的橙色边框/画面黑条灰度介于灯箱与胶片本体之间，单级 Otsu 会把
    边框并入胶片连通域（外接框被撑满整幅，腐蚀也剥不掉）。在暗类灰度分布
    内部再做一次 Otsu，可把胶片本体（更暗）与边框/亮度渐变（中等暗度）分开。

    防塌缩护栏：仅当更暗子类面积占比 ∈ [min_frac, 80%×暗类] 时采用——
    低于下限说明二次分割锁到了孤立小特征（如全幅扫描件里的单个缺陷），
    接近暗类说明没分出有意义的子类，两种情况都保留单级结果。
    """
    vals = blur[dark > 0]
    if vals.size < 100:
        return dark
    t2, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if t2 <= 0:
        return dark
    core = ((blur <= t2) & (dark > 0)).astype(np.uint8) * 255
    frac = float(core.mean()) / 255.0
    dark_frac = float((dark > 0).mean())
    if min_frac <= frac <= 0.8 * dark_frac:
        return core
    return dark


def _core_bbox(
    mask: np.ndarray, margin_frac: float, fallback: np.ndarray
) -> tuple[int, int, int, int]:
    """腐蚀后掩膜**最大连通域**的外接框（剥离翻拍边框/扫描白边/画面黑条）。

    翻拍照片的橙色边框、画面上下黑条常与胶片连成单一暗色连通域，仅腐蚀
    不足以收缩外接框（框仍被撑满整幅）。故腐蚀后重取最大连通域：胶片本体
    面积远大于残余边框条带，其核区即胶片有效区。核区为空（胶片本身细窄）
    时回退连通域统计外接框。
    """
    h, w = mask.shape[:2]
    m = max(3, int(min(h, w) * margin_frac) | 1)
    core = cv2.erode(mask.astype(np.uint8), np.ones((m, m), np.uint8))
    if core.any():
        _n, _labels, stats, _ = cv2.connectedComponentsWithStats(core)
        idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return (
            int(stats[idx, cv2.CC_STAT_LEFT]),
            int(stats[idx, cv2.CC_STAT_TOP]),
            int(stats[idx, cv2.CC_STAT_WIDTH]),
            int(stats[idx, cv2.CC_STAT_HEIGHT]),
        )
    return (
        int(fallback[cv2.CC_STAT_LEFT]),
        int(fallback[cv2.CC_STAT_TOP]),
        int(fallback[cv2.CC_STAT_WIDTH]),
        int(fallback[cv2.CC_STAT_HEIGHT]),
    )


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """分割专用归一化（min-max 拉伸仅服务阈值分割，黑度测量仍用原数组）。"""
    if arr.dtype == np.uint8:
        return arr
    a = arr.astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return np.zeros_like(a, dtype=np.uint8)
    return ((a - lo) / (hi - lo) * 255.0).astype(np.uint8)
