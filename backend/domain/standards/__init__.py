"""标准适配与数值表（§6 / §T8）。

- 每种标准一个实现模块（如 nb47013.py），实现 StandardGrader。
- 数值表统一放 tables/*.yaml，经 loader 加载并校验。
- 未授权数值（authorized=false）熔断：不得输出级别（§T8）。
"""
from __future__ import annotations
