# 射线焊缝缺陷智能检测系统（ScanDetection）一键打包脚本（Windows）
# 用法：在仓库根目录执行
#   powershell -ExecutionPolicy Bypass -File scripts\build_installer.ps1            # 无权重也可构建（警告）
#   powershell -ExecutionPolicy Bypass -File scripts\build_installer.ps1 -RequireWeights   # 无权重即中止（正式交付用）
# 产物：src\release\射线焊缝缺陷智能检测系统_0.1.0_x64-setup.exe（electron-builder NSIS）
#
# 交付口径：对外只交付本安装包。scripts\launch_app.vbs / stop_app.vbs 为开发
# 调试启动器（会打开系统默认浏览器），不随安装包分发，禁止作为交付物外发。
#
# 前置（打包机一次性准备）：
#   1. Node.js 20+/pnpm 10+（桌面壳已从 Tauri/Rust 迁移到 Electron，无需 Rust 工具链；
#      网络受限时先设 $env:ELECTRON_MIRROR = "https://npmmirror.com/mirrors/electron/"）
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

Write-Host "==> [0/5] 供给嵌入式 Python 运行时（缺失时自动从 python.org 嵌入包 + 锁定依赖构建，CI 可复现）" -ForegroundColor Cyan
& (Join-Path $PSScriptRoot "provision_python_embed.ps1")

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

# ---- 安装版必须走 external 连接模式 ----
# 安装包不随分发模型运行时（CUDA 版 llama.cpp 约 1.3GB + GGUF 2.4GB 起，超 NSIS
# 2GB 上限），故安装版必须 llm.mode=external（只连接本机已有服务）。managed 会让
# 安装版永远停在 unavailable。构建期临时改写后端配置，构建结束（含失败）恢复原值。
$cfgPath = Join-Path $root "backend\configs\default.yaml"
$cfgOriginal = Get-Content -Raw -LiteralPath $cfgPath
$cfgPatched = $cfgOriginal -replace "(?m)^(\s*mode:\s*)managed", '${1}external'
$cfgChanged = $cfgPatched -ne $cfgOriginal
if ($cfgChanged) {
    Set-Content -LiteralPath $cfgPath -Value $cfgPatched -NoNewline -Encoding utf8
    Write-Host "==> 安装版配置：llm.mode managed → external（连接本机已有服务）" -ForegroundColor Green
} elseif ($cfgOriginal -match "(?m)^\s*mode:\s*external") {
    Write-Host "==> 安装版配置：llm.mode 已是 external" -ForegroundColor Green
} else {
    throw "backend\configs\default.yaml 未找到 llm.mode 字段——配置结构已变更，请人工确认后再打包（禁止静默产出配置错误的安装包）"
}

try {
    Write-Host "==> [3/5] 前端依赖安装（锁定锁文件）" -ForegroundColor Cyan
    Push-Location (Join-Path $root "src")
    try {
        pnpm install --frozen-lockfile
        if ($LASTEXITCODE -ne 0) { throw "pnpm install 失败" }

        Write-Host "==> [4/5] Electron 打包（前端构建 + asar 收集 + 资源收集 + NSIS 安装器）" -ForegroundColor Cyan
        pnpm run dist
        if ($LASTEXITCODE -ne 0) { throw "electron-builder 打包失败" }
    } finally {
        Pop-Location
    }
} finally {
    if ($cfgChanged) {
        Set-Content -LiteralPath $cfgPath -Value $cfgOriginal -NoNewline -Encoding utf8
        Write-Host "==> 已恢复开发配置（llm.mode=managed）" -ForegroundColor DarkGray
    }
}

$installer = Get-ChildItem (Join-Path $root "src\release\*-setup.exe") |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $installer) { throw "未找到安装包产物" }

Write-Host "==> [5/5] 完成" -ForegroundColor Green
Write-Host ("安装包: " + $installer.FullName)
Write-Host ("大小:   " + [math]::Round($installer.Length / 1MB, 1) + " MB")

# 交付前校验：训练模型权重必须真正进入打包产物。
# 历史事故：Tauri 时代的 tauri.windows.conf.json 的 bundle.resources 覆盖了主
# 配置且漏配 models/weights，安装包的 models\weights 成了空目录——当时靠
# resolve_model_uri 的双锚点回落到 backend\models\weights 才没崩，但"配置指向
# 空目录"本身是隐患（任何人不动双锚点就会静默降级到连通域基线）。此处把
# "权重入包"变成硬门禁（electron-builder 走 extraResources，同样必须校验）。
$bundledWeights = @(
    (Join-Path $root "src\release\win-unpacked\resources\models\weights\best.onnx"),
    (Join-Path $root "src\release\win-unpacked\resources\backend\models\weights\best.onnx")
) | Where-Object { Test-Path $_ }
if ($bundledWeights.Count -eq 0) {
    throw "打包产物未包含 best.onnx —— 安装版将无法加载训练模型（回退连通域基线）。请检查 electron-builder.yml 的 extraResources。"
}
Write-Host (">>> 权重已入包: " + $bundledWeights[0]) -ForegroundColor Green
Write-Host ""
Write-Host "分发说明：安装包离线自足（内嵌 Chromium + Python 运行时 + 全部后端依赖），"
Write-Host "目标机无需联网、无需预装 WebView2 或任何浏览器运行时；当前用户级安装，"
Write-Host "免管理员权限。未签名：SmartScreen 提示点'仍要运行'。"
if (-not (Test-Path $weights)) {
    Write-Host "!!!!! 再次提醒：本安装包不含 AI 权重（基线降级版），禁止正式交付 !!!!!" -ForegroundColor Yellow
}
