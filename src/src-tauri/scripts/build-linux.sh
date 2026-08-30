#!/usr/bin/env bash
# S-02 国产 OS（麒麟 V10 / UOS 等 Debian 系）Linux 构建脚本（最小骨架）。
#
# 诚实声明：本脚本未在麒麟 V10 / UOS 真机上验证，仅给出预期可复现的构建
# 步骤（deb + appimage 目标已在 tauri.conf.json bundle.targets 登记）。
# 真机验证后请回填"验证记录"（见 docs/国产化适配矩阵.md）。
#
# 前置（目标机/构建机）：
#   - Rust toolchain（rustup）、Node.js 20+、pnpm；
#   - Tauri Linux 依赖（Debian 系）：
#       sudo apt install libwebkit2gtk-4.1-dev build-essential curl wget file \
#            libxdo-dev libssl-dev libayatana-appindicator3-dev librsvg2-dev
#   - Python 3.10+（后端侧车目录 <项目根>/python_embed/bin/python3 由部署流程
#     准备，可用 pyenv: `pyenv install 3.10 && pyenv shell 3.10 && python -m venv
#     python_embed`，或系统 venv 拷贝；路径解析见 src/main.rs 的
#     cfg(target_os = "linux") pick_python）。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT/src"

# 1. 前端构建（tauri beforeBuildCommand 亦会执行，此处显式跑以便失败早暴露）
pnpm install --frozen-lockfile
pnpm build:pkg

# 2. 后端依赖（目标 python 侧车环境内安装；示例用 python_embed，按实际调整）
#    python_embed/bin/pip install -r "$ROOT/backend/requirements.lock"

# 3. Tauri 打包（deb + appimage）
cargo tauri build --bundles deb,appimage

echo "构建产物见 src/src-tauri/target/release/bundle/{deb,appimage}"
echo "注意：deb/appimage 未真机验证，请在麒麟 V10 / UOS 上安装验证后回填适配矩阵。"
