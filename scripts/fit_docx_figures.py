"""把 docx 内的插图宽度统一到版心宽度（默认 16.2 cm，A4 减左右各 2.2 cm 页边距）。

为什么需要这一步：``html_to_docx`` 转换引擎会**忽略** HTML 里 ``<img>`` 的
``style="width: ..."`` 与 CSS 的 ``width/max-width``，直接按图片自身像素尺寸
嵌入。实测结果是 18.2–18.7 cm 宽，超出 16.6 cm 的版心并溢进页边距。
本脚本在转换后按比例重设内联图形的宽高，是当前工具链下唯一可靠的定宽办法。

用法::

    python scripts/fit_docx_figures.py <docx 路径> [宽度 cm]

默认宽度 16.2 cm；高度按原始宽高比等比缩放，不改变图片像素内容。
"""

from __future__ import annotations

import sys
from pathlib import Path

from docx import Document
from docx.shared import Cm


def fit(path: Path, width_cm: float = 16.2) -> None:
    doc = Document(str(path))
    shapes = doc.inline_shapes
    if not shapes:
        print("未找到内联图形，未作修改。")
        return
    changed = 0
    for sh in shapes:
        native_w = sh.width.cm
        native_h = sh.height.cm
        ratio = native_h / native_w if native_w else 1.0
        new_w = Cm(width_cm)
        new_h = Cm(width_cm * ratio)
        if abs(native_w - width_cm) > 0.01:
            changed += 1
        sh.width = new_w
        sh.height = new_h
    doc.save(str(path))
    print(f"已处理 {len(shapes)} 张图，其中 {changed} 张宽度被调整到 {width_cm} cm")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    p = Path(sys.argv[1])
    if not p.exists():
        raise SystemExit(f"文件不存在：{p}")
    w = float(sys.argv[2]) if len(sys.argv) > 2 else 16.2
    fit(p, w)


if __name__ == "__main__":
    main()
