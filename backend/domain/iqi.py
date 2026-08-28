"""线型/孔型像质计（wire/hole IQI）识别。纯算法。

基线假设（标准 IQI 通用几何）：
- 丝/孔在 ROI 内沿水平方向平行排列、垂直方向等距分布；
- 金属丝吸收射线 → 底片黑度低 → 透射数字化影像上呈"亮线"；
  孔型反之呈"暗斑"。
算法：对每根丝取水平行带 → 行均值剖面 → 峰对比度 vs 局部噪声 → 判定可见；
achieved = 最细可见丝/孔号（号越大越细）；passed = achieved ≥ required。

自动定位（模板匹配/小目标检测）：roi 为 None 且 auto_locate=True 时，
用 Sobolev 垂直梯度逐行"边缘能量"定位 IQI 占据的连续垂直带，再经 verify
校验（≥2 个可见单元）确认，失败回退全图。该准则尺度无关、对 pitch 误差免疫。
已知边界：当丝间距 ≤ 丝宽（无间隙、合并为实心块）时，块内无内部边缘，边缘
能量法无法分辨且易与平滑亮区混淆 → 返回 None，需人工/边框 ROI。真实 IQI 丝
间必有可分辨间隙，故该边界不影响工业底片。丝号-直径表为公开参考值，正式使用
前须按 IQI 标准复核。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from backend.domain.dto import IQIResult


@dataclass(frozen=True)
class IqiConfig:
    # 线型（wire）像质计：丝号 1..N 对应直径（mm），递增
    wire_diameters_mm: tuple[float, ...] = (
        3.2,
        2.5,
        2.0,
        1.6,
        1.25,
        1.0,
        0.8,
        0.63,
        0.5,
        0.4,
        0.32,
        0.25,
        0.2,
        0.16,
        0.125,
        0.1,
        0.08,
        0.063,
        0.05,
    )
    required_wire_no: int = 10  # 工艺要求的最细丝号（越大越难）
    # 孔型（hole）像质计：孔径 1..N（mm），递增（公开参考，待官方复核）
    type: str = "wire"  # wire | hole
    hole_diameters_mm: tuple[float, ...] = (
        1.0,
        0.8,
        0.63,
        0.5,
        0.4,
        0.32,
        0.25,
        0.2,
        0.16,
        0.125,
    )
    required_hole_no: int = 6  # 工艺要求的可见最小孔号（越大越难）
    min_contrast_ratio: float = 3.0  # 峰对比度 / 噪声 阈值
    band_radius_px: int = 2  # 每根丝/孔行带半宽
    auto_locate: bool = True  # 自动定位像质计（模板匹配）；False 则必须由前端/人工给 ROI
    locate_threshold: float = 0.35  # 归一化互相关匹配得分下限（低于视为未找到）
    # A/AB/B 影像质量等级 → 线型像质计要求丝号（公开参考，正式使用前须按标准复核）。
    # 条目：[透照厚度上限 mm, A级要求丝号, AB级要求丝号, B级要求丝号]；
    # 等级越严（A）要求丝号越大（越细）；achieved 丝号 ≥ 要求即满足该等级。
    sensitivity: tuple[tuple[float, int, int, int], ...] = (
        (2.0, 14, 13, 12),
        (4.0, 13, 12, 11),
        (8.0, 12, 11, 10),
        (12.0, 11, 10, 9),
        (18.0, 10, 9, 8),
        (30.0, 9, 8, 7),
        (50.0, 8, 7, 6),
        (80.0, 7, 6, 5),
        (120.0, 6, 5, 4),
        (200.0, 5, 4, 3),
        (350.0, 4, 3, 2),
        (9999.0, 3, 2, 1),
    )


def verify_iqi(
    image: np.ndarray,
    cfg: IqiConfig,
    roi: tuple[int, int, int, int] | None = None,
    iqi_type: str | None = None,
) -> IQIResult:
    """按类型路由到线型/孔型识别。

    iqi_type 优先于 cfg.type；二者皆缺省为 wire。保持与 verify_wire_iqi
    同签名（image, cfg, roi），便于端点/管线统一调用，不破坏既有调用点。

    roi 为 None 且 cfg.auto_locate=True 时，先尝试自动定位像质计（模板匹配）；
    定位失败则回退全图（等价于旧行为），确保端点"免 ROI"也能跑。
    """
    t = (iqi_type or cfg.type or "wire").lower()
    if roi is None and cfg.auto_locate:
        located = locate_iqi(image, cfg, iqi_type=t, threshold=cfg.locate_threshold)
        if located is not None:
            roi = located
    if t == "hole":
        return verify_hole_iqi(image, cfg, roi=roi)
    return verify_wire_iqi(image, cfg, roi=roi)


def locate_iqi(
    image: np.ndarray,
    cfg: IqiConfig,
    iqi_type: str | None = None,
    threshold: float = 0.3,
) -> tuple[int, int, int, int] | None:
    """自动定位像质计。

    返回 ROI=(x, y, w, h) 垂直带。线型像质计丝为水平亮线 → 模板取全宽水平线，
    只需搜 y；孔型像质计为暗点列 → 模板取窄列，搜 x,y。二者均以"行/列均值剖面"
    判定可见性，故线型 ROI 的 x 取 0、w 取全宽（验证时跨整行统计，与 x 无关）。

    选择准则为带内垂直周期性强度（对 pitch 误差免疫）；低于 threshold 视为未找到，
    返回 None（调用方回退全图或人工 ROI）。
    """
    t = (iqi_type or cfg.type or "wire").lower()
    gray = _to_float01(image)
    if gray is None:
        return None
    return _locate_template(gray, cfg, t, threshold)


def _locate_template(
    gray: np.ndarray,
    cfg: IqiConfig,
    kind: str,
    threshold: float,
) -> tuple[int, int, int, int] | None:
    """基于边缘能量剖面的像质计垂直带定位。

    像质计的丝（亮线）/孔（暗点）在底片上都是强边缘特征：用 Sobel 垂直梯度
    得到逐行"线/孔边缘能量"，IQI 占据的连续行能量显著高于背景。对能量剖面
    阈值化 + 形态学闭合（连接丝/孔之间的低能间隙）→ 取最大连通垂直段的
    行范围即像质计带。该准则尺度无关、对 pitch 误差免疫，且直接给出正确
    带高，使后续 verify 的逐丝对齐天然成立。

    孔型像质计为暗点列，还需用水平梯度定位所在列 x，给出窄带；线型丝跨全宽，
    带 x 取 0、w 取全宽。定位带经 verify 校验（≥2 个可见单元）才算数，
    否则返回 None（调用方回退全图或人工 ROI）。
    """
    h_img, w_img = gray.shape[:2]
    n = len(cfg.hole_diameters_mm) if kind == "hole" else len(cfg.wire_diameters_mm)
    if h_img < 8 or n < 2:
        return None
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)  # 垂直梯度 → 水平线/孔边缘
    row_energy = np.mean(np.abs(gy), axis=1)
    if row_energy.max() < 1e-6:
        return None
    thr = threshold * row_energy.max()
    mask = (row_energy > thr).astype(np.uint8)
    # 闭合丝/孔之间的低能间隙（核宽随图高自适应）。强制 1D 以便按行索引。
    k = max(3, int(h_img * 0.02) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones(k, np.uint8)).ravel()
    rows = np.where(mask > 0)[0]
    if rows.size == 0:
        return None
    y0, y1 = int(rows.min()), int(rows.max())  # numpy ints; 下游切片/算术均兼容
    if (y1 - y0) < 2:
        return None
    if kind == "hole":
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)  # 水平梯度 → 定位暗点所在列
        col_energy = np.mean(np.abs(gx), axis=0)
        cx = int(np.argmax(col_energy))
        half = max(6, round((y1 - y0) / n * 0.6)) if n > 0 else 10
        x0 = max(0, cx - half)
        x1 = min(w_img, cx + half)
        band = (x0, y0, x1 - x0, y1 - y0)
    else:
        band = (0, y0, w_img, y1 - y0)
    # 校验：定位带必须真含 ≥2 个可见单元，否则判为误匹配（如单条划痕）。
    sub = gray[band[1] : band[1] + band[3], band[0] : band[0] + band[2]]
    v = (
        verify_hole_iqi(sub, cfg, roi=None)
        if kind == "hole"
        else verify_wire_iqi(sub, cfg, roi=None)
    )
    if v.achieved is None or int(v.achieved) < 2:
        return None
    return band


def _to_float01(image: np.ndarray) -> np.ndarray | None:
    if image is None or image.size == 0:
        return None
    arr = image.astype(np.float32)
    if arr.max() > 0:
        arr = arr / float(arr.max())
    return arr


def map_sensitivity_grade(
    achieved_no: int | None,
    thickness_mm: float | None,
    table: tuple[tuple[float, int, int, int], ...],
) -> str | None:
    """由可达丝号 + 透照厚度映射影像质量等级。

    返回满足条件的**最高**等级；连 B 级都达不到返回 None。厚度/丝号缺失或表为空
    返回 None。参考表（待官方复核）见 IqiConfig.sensitivity。
    """
    if achieved_no is None or thickness_mm is None or not table:
        return None
    req: tuple[int, int, int] | None = None
    for tmax, a, ab, b in table:
        if thickness_mm <= tmax:
            req = (a, ab, b)
            break
    if req is None:
        return None
    a_req, ab_req, b_req = req
    if achieved_no >= a_req:
        return "A"
    if achieved_no >= ab_req:
        return "AB"
    if achieved_no >= b_req:
        return "B"
    return None


def enrich_grade(
    iqi: IQIResult,
    thickness_mm: float | None,
    table: tuple[tuple[float, int, int, int], ...],
) -> IQIResult:
    """用厚度表为已有 IQIResult 补全 grade 字段（不破坏原有 achieved/required）。"""
    if iqi.achieved is None:
        grade: str | None = None
    else:
        grade = map_sensitivity_grade(int(iqi.achieved), thickness_mm, table)
    return IQIResult(
        iqi_type=iqi.iqi_type,
        achieved=iqi.achieved,
        required=iqi.required,
        passed=iqi.passed,
        grade=grade,
    )


def verify_wire_iqi(
    image: np.ndarray,
    cfg: IqiConfig,
    roi: tuple[int, int, int, int] | None = None,
) -> IQIResult:
    """在 ROI（默认全图）内逐丝测量可见性，返回可达最细丝号。"""
    h_img, w_img = image.shape[:2]
    x, y, w, h = roi or (0, 0, w_img, h_img)
    # 负坐标会让 numpy 切片回绕到末尾行/列，产生错误 patch；夹到 >=0 即可，
    # 越界上界由 numpy 自然处理（返回空/部分 patch，下面判空降级）。
    x, y = max(0, x), max(0, y)
    patch = image[y : y + h, x : x + w]

    achieved: int | None = None
    if patch.size > 0:
        n = len(cfg.wire_diameters_mm)
        for i in range(n):
            y_center = round((i + 0.5) / n * h)
            contrast, noise = _row_profile_contrast(patch, y_center, cfg.band_radius_px)
            # 丝号由粗到细排列，可见性应单调下降：一旦某丝不可见，更细的丝
            # 必然不可见。立即终止，避免噪声导致的非单调误判把 achieved 推到
            # 实际不可见的更细丝号（错误高估像质计可达丝号 → 误判评片合格）。
            if contrast > cfg.min_contrast_ratio * noise:
                achieved = i + 1  # 粗→细遍历，最后一个通过值即最细可见丝号
            else:
                break

    passed = achieved is not None and achieved >= cfg.required_wire_no
    return IQIResult(
        iqi_type="wire",
        achieved=str(achieved) if achieved is not None else None,
        required=str(cfg.required_wire_no),
        passed=bool(passed),
    )


def verify_hole_iqi(
    image: np.ndarray,
    cfg: IqiConfig,
    roi: tuple[int, int, int, int] | None = None,
) -> IQIResult:
    """孔型像质计（hole IQI）自动识别。纯算法。

    与线型对称：孔在底片上呈暗点/暗斑，可见性取行剖面对比度的绝对值
    （中心暗于背景 → contrast 为负，取 abs 度量亮度差）。丝号/孔号由粗到细
    排列，可见性单调下降：一旦某孔不可见即终止，避免噪声把 achieved 推高到
    实际不可见的最细孔（错误高估像质计可达等级 → 误判评片合格）。
    自动 ROI 搜索同线型，留作后续增强；当前 ROI 由前端/人工提供。
    """
    h_img, w_img = image.shape[:2]
    x, y, w, h = roi or (0, 0, w_img, h_img)
    x, y = max(0, x), max(0, y)
    patch = image[y : y + h, x : x + w]

    achieved: int | None = None
    if patch.size > 0:
        n = len(cfg.hole_diameters_mm)
        for i in range(n):
            y_center = round((i + 0.5) / n * h)
            contrast, noise = _hole_row_contrast(patch, y_center, cfg.band_radius_px)
            # 孔为暗斑：背景高于孔中心，contrast 为正（亮度差大小）。
            if contrast > cfg.min_contrast_ratio * noise:
                achieved = i + 1  # 粗→细遍历，最后一个通过值即最小可见孔号
            else:
                break

    passed = achieved is not None and achieved >= cfg.required_hole_no
    return IQIResult(
        iqi_type="hole",
        achieved=str(achieved) if achieved is not None else None,
        required=str(cfg.required_hole_no),
        passed=bool(passed),
    )


def _hole_row_contrast(patch: np.ndarray, y_center: int, radius: int) -> tuple[float, float]:
    """孔型对比度（暗孔于亮背景）。返回 (正值对比度=背景−中心, 噪声σ)。

    孔为暗斑，落在行均值剖面的低百分位；背景为亮区，取高百分位（≥70%）行估计，
    避免暗孔污染背景估计（线型 bright-feature 逻辑反之，见 _row_profile_contrast）。
    """
    row_mean = patch.mean(axis=1).astype(np.float64)
    if row_mean.size < 2 * radius + 1:
        return 0.0, 0.0
    lo, hi = max(0, y_center - radius), min(row_mean.size, y_center + radius + 1)
    center = float(row_mean[lo:hi].mean())
    q70 = float(np.percentile(row_mean, 70))
    bg_mask = row_mean >= q70
    if int(bg_mask.sum()) >= 2:
        bg_level = float(np.median(row_mean[bg_mask]))
        noise = float(np.std(patch[bg_mask, :].astype(np.float64)))
    else:
        bg_level = float(np.median(row_mean))
        noise = float(patch.std())
    return bg_level - center, max(noise, 1e-6)


def _row_profile_contrast(patch: np.ndarray, y_center: int, radius: int) -> tuple[float, float]:
    """返回 (峰对比度, 噪声σ)。

    丝为水平亮线：对比度体现在**逐行均值剖面**（垂直方向）上——
    丝带行均值显著高于背景中位数。噪声仅在**背景行（≤中位数）**上以
    MAD 稳健估计，使信号占比较高（甚至 ≥50% 行）时噪声估计不被丝峰污染。
    """
    row_mean = patch.mean(axis=1).astype(np.float64)
    if row_mean.size < 2 * radius + 1:
        return 0.0, 0.0
    lo, hi = max(0, y_center - radius), min(row_mean.size, y_center + radius + 1)
    center = float(row_mean[lo:hi].mean())
    # 背景簇：30 分位以下的行（信号占比 <70% 时均落在背景区）。
    q30 = float(np.percentile(row_mean, 30))
    bg_mask = row_mean <= q30
    if int(bg_mask.sum()) >= 2:
        bg_level = float(np.median(row_mean[bg_mask]))
        # 可见性对比对象是**像素级噪声**（丝对比度 vs σ_pix），
        # 而非行均值噪声（列平均会淹没像素噪声，导致过度敏感）。
        noise = float(np.std(patch[bg_mask, :].astype(np.float64)))
    else:
        bg_level = float(np.median(row_mean))
        noise = float(patch.std())
    return center - bg_level, max(noise, 1e-6)
