"""pytest 根配置：保证 backend 包可从仓库根导入。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
