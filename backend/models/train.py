"""训练入口（§17，M4 里程碑实现）。

M1 仅占位：训练脚本、数据版本（DVC）、实验追踪（MLflow）、
模型卡（models/*.md）在 M4 落地。约束：
- 训练与推理必须共用 domain/preprocess/transform.py（ADR-007）；
- 数据版本与评估结果须可复现（固定种子 + Golden Set 隔离，§15.6）。
"""
from __future__ import annotations


def main() -> None:
    raise NotImplementedError("M4: 训练流水线（见规格书 §17）")


if __name__ == "__main__":
    main()
