#!/usr/bin/env python3
"""重新生成 docs/api 下的 OpenAPI 快照。

docs/api/openapi.json 与 openapi-live.json 均为当前代码 create_app() 的
完整路由快照（历史上靠手工导出、长期滞后于实际路由数）。路由变更后
在仓库根目录执行一次本脚本即可同步：

    python scripts/gen_openapi.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from backend.app.main import app  # noqa: E402


def main() -> None:
    spec = app.openapi()
    out_dir = _ROOT / "docs" / "api"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("openapi.json", "openapi-live.json"):
        path = out_dir / name
        path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {path} ({len(spec['paths'])} paths)")


if __name__ == "__main__":
    main()
