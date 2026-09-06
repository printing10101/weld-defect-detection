# ScanDetection 一键打包脚本（Windows）
# 用法：在仓库根目录执行  powershell -ExecutionPolicy Bypass -File scripts\build_installer.ps1
# 产物：src\src-tauri\target\release\bundle\nsis\ScanDetection_0.1.0_x64-setup.exe
#
# 前置（打包机一次性准备）：
#   1. Rust toolchain（rustup，MSVC target）+ Node.js 20+/pnpm 10+
#   2. 后端开发环境 backend\.venv（仅用于执行裁剪脚本，不随包分发）
#   3. 模型权重：把训练产物 best.onnx 放到 backend\models\weights\best.onnx
#      （缺失时安装包仍可构建，但应用启动后自动退化为基线检测器，
#        界面会显示"降级"横幅——正式交付必须放权重）

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

Write-Host "==> [1/4] 裁剪嵌入 Python 运行时（剔除 pip/pytest/测试目录，省 ~17MB）" -ForegroundColor Cyan
& (Join-Path $root "backend\.venv\Scripts\python.exe") (Join-Path $root "scripts\slim_python_embed.py")
if ($LASTEXITCODE -ne 0) { throw "slim_python_embed 失败" }

$weights = Join-Path $root "backend\models\weights\best.onnx"
if (Test-Path $weights) {
    $mb = [math]::Round((Get-Item $weights).Length / 1MB, 1)
    Write-Host "==> 模型权重就绪: best.onnx (${mb} MB)" -ForegroundColor Green
} else {
    Write-Warning "未找到 backend\models\weights\best.onnx —— 安装包将不含 AI 权重（运行时退化为基线检测器）"
}

Write-Host "==> [2/4] 前端构建（vue-tsc + vite，tauri beforeBuildCommand 亦会执行，此处提前失败早暴露）" -ForegroundColor Cyan
Push-Location (Join-Path $root "src")
try {
    pnpm install --frozen-lockfile
    if ($LASTEXITCODE -ne 0) { throw "pnpm install 失败" }

    Write-Host "==> [3/4] Tauri 打包（Rust release 编译 + 资源收集 + NSIS 安装器）" -ForegroundColor Cyan
    pnpm exec tauri build
    if ($LASTEXITCODE -ne 0) { throw "tauri build 失败" }
} finally {
    Pop-Location
}

$installer = Get-ChildItem (Join-Path $root "src\src-tauri\target\release\bundle\nsis\*-setup.exe") |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $installer) { throw "未找到安装包产物" }

Write-Host "==> [4/4] 完成" -ForegroundColor Green
Write-Host ("安装包: " + $installer.FullName)
Write-Host ("大小:   " + [math]::Round($installer.Length / 1MB, 1) + " MB")
Write-Host ""
Write-Host "分发说明：安装包离线自足（内嵌 Python 运行时 + 全部后端依赖 + WebView2" 
Write-Host "离线安装器），目标机无需联网/无需管理员权限（当前用户级安装）。"
Write-Host "未签名：首次运行可能触发 Windows SmartScreen 提示，点'仍要运行'即可。"
