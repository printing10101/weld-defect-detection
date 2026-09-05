#!/usr/bin/env python3
"""重新生成 docs/api/openapi.json（当前代码全量路由快照）。

历史上靠手工导出、长期滞后于实际路由数。路由变更后在仓库根目录
执行一次本脚本即可同步：

    python scripts/gen_openapi.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from backend.app.main import app


def main() -> None:
    spec = app.openapi()
    out_dir = _ROOT / "docs" / "api"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "openapi.json"
    path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {path} ({len(spec['paths'])} paths)")


if __name__ == "__main__":
    main()
