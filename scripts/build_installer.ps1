# ScanDetection 一键打包脚本（Windows）
# 用法：在仓库根目录执行
#   powershell -ExecutionPolicy Bypass -File scripts\build_installer.ps1            # 无权重也可构建（警告）
#   powershell -ExecutionPolicy Bypass -File scripts\build_installer.ps1 -RequireWeights   # 无权重即中止（正式交付用）
# 产物：src\src-tauri\target\release\bundle\nsis\ScanDetection_0.1.0_x64-setup.exe
#
# 前置（打包机一次性准备）：
#   1. Rust toolchain（rustup，MSVC target）+ Node.js 20+/pnpm 10+
#   2. 后端开发环境 backend\.venv（仅用于执行裁剪脚本，不随包分发）
#   3. 模型权重：把训练产物 best.onnx 放到 backend\models\weights\best.onnx
param(
    [switch]$RequireWeights  # 正式交付模式：缺权重直接中止，防止静默降级版外发
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $root "backend\.venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "未找到 backend\.venv\Scripts\python.exe —— 请先按 README 创建后端开发环境"
}

Write-Host "==> [1/5] 裁剪嵌入 Python 运行时（剔除 pip/pytest/包内测试目录，省 ~35MB）" -ForegroundColor Cyan
& $venvPython (Join-Path $root "scripts\slim_python_embed.py")
if ($LASTEXITCODE -ne 0) { throw "slim_python_embed 失败" }

Write-Host "==> [2/5] 裁剪后导入冒烟（防止误删运行时依赖进包）" -ForegroundColor Cyan
& (Join-Path $root "src\python_embed\python.exe") -B -c "import scipy.optimize, scipy.signal, scipy.special, scipy.ndimage, skimage.metrics, cv2, onnxruntime, fastapi, uvicorn, pydicom, gmssl, Cryptodome, reportlab, sqlalchemy, alembic; print('embed imports OK')"
if ($LASTEXITCODE -ne 0) { throw "嵌入运行时导入冒烟失败——裁剪规则误删了运行时依赖" }

$weights = Join-Path $root "backend\models\weights\best.onnx"
if (Test-Path $weights) {
    $mb = [math]::Round((Get-Item $weights).Length / 1MB, 1)
    Write-Host "==> 模型权重就绪: best.onnx (${mb} MB)" -ForegroundColor Green
} elseif ($RequireWeights) {
    throw "未找到 backend\models\weights\best.onnx —— RequireWeights 模式禁止构建无 AI 权重的降级版"
} else {
    Write-Warning "未找到 backend\models\weights\best.onnx —— 本次安装包为基线降级版（界面将显示降级横幅），不得作为正式交付物外发！"
}

Write-Host "==> [3/5] 前端依赖安装（锁定锁文件）" -ForegroundColor Cyan
Push-Location (Join-Path $root "src")
try {
    pnpm install --frozen-lockfile
    if ($LASTEXITCODE -ne 0) { throw "pnpm install 失败" }

    Write-Host "==> [4/5] Tauri 打包（前端构建 + Rust release 编译 + 资源收集 + NSIS 安装器）" -ForegroundColor Cyan
    pnpm exec tauri build
    if ($LASTEXITCODE -ne 0) { throw "tauri build 失败" }
} finally {
    Pop-Location
}

$installer = Get-ChildItem (Join-Path $root "src\src-tauri\target\release\bundle\nsis\*-setup.exe") |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $installer) { throw "未找到安装包产物" }

Write-Host "==> [5/5] 完成" -ForegroundColor Green
Write-Host ("安装包: " + $installer.FullName)
Write-Host ("大小:   " + [math]::Round($installer.Length / 1MB, 1) + " MB")
Write-Host ""
Write-Host "分发说明：安装包离线自足（内嵌 Python 运行时 + 全部后端依赖 + WebView2"
Write-Host "离线安装器），目标机无需联网；当前用户级安装，WebView2 缺失时需允许"
Write-Host "一次 UAC 提权完成其机器级安装。未签名：SmartScreen 提示点'仍要运行'。"
if (-not (Test-Path $weights)) {
    Write-Host "!!!!! 再次提醒：本安装包不含 AI 权重（基线降级版），禁止正式交付 !!!!!" -ForegroundColor Yellow
}
