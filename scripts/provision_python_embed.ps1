# 供给可复现的嵌入式 Python 运行时（默认 src\python_embed）——打包机/CI 通用。
# 此前该目录"每台打包机手工准备"（不入 git），CI 无法复现打包；本脚本把
# "python.org 嵌入包 + 锁定依赖 requirements.txt"固化为一条可重复执行的命令。
#
# 用法（仓库根目录）：
#   powershell -ExecutionPolicy Bypass -File scripts\provision_python_embed.ps1            # 缺失才构建
#   powershell -ExecutionPolicy Bypass -File scripts\provision_python_embed.ps1 -Force     # 强制重建
#   powershell -ExecutionPolicy Bypass -File scripts\provision_python_embed.ps1 -Target <dir>  # 自定义目标（测试用）
#
# 说明：
# - 嵌入发行版默认禁用 import site（python312._pth），不解开则 site-packages 不可见，
#   依赖装了也找不到——本脚本显式解开；
# - 依赖用 dev venv 的 pip 以 --python 跨解释器安装（目标解释器无需自带 pip），
#   来源 requirements.txt（== 全锁定，与运行时验证过的版本一致）；
# - 构建完成后跑一次导入冒烟，防止"装好但 import 就炸"的假成功。
param(
    [string]$PythonVersion = "3.12.10",
    [string]$Target = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $Target) { $Target = Join-Path $root "src\python_embed" }

if ((Test-Path (Join-Path $Target "python.exe")) -and -not $Force) {
    Write-Host "==> 嵌入运行时已存在（$Target），跳过供给（-Force 重建）"
    exit 0
}
if (Test-Path $Target) { Remove-Item -Recurse -Force $Target }
New-Item -ItemType Directory -Force -Path $Target | Out-Null

$zip = Join-Path $env:TEMP "python-$PythonVersion-embed-amd64.zip"
$url = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip"
if (-not (Test-Path $zip)) {
    Write-Host "==> 下载 $url"
    # Windows PowerShell 5.1 无 Invoke-WebRequest -MaximumRetryCount，手动重试
    $downloaded = $false
    for ($i = 1; $i -le 3; $i++) {
        try {
            Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
            $downloaded = $true
            break
        } catch {
            Write-Warning "下载失败（第 $i/3 次）：$_"
            Start-Sleep -Seconds 5
        }
    }
    if (-not $downloaded) { throw "python.org 嵌入包下载失败（网络不可达？可手动放置 $zip）" }
}
Expand-Archive -Path $zip -DestinationPath $Target -Force

$pth = Get-ChildItem $Target -Filter "python*._pth" | Select-Object -First 1
if (-not $pth) { throw "嵌入包中未找到 python*._pth（版本 $PythonVersion 布局异常？）" }
(Get-Content $pth.FullName) -replace '^#import site', 'import site' | Set-Content $pth.FullName
Write-Host "==> 已启用 import site（$($pth.Name)）"

$venvPython = Join-Path $root "backend\.venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) { throw "未找到 dev venv：$venvPython" }
Write-Host "==> 安装运行时依赖（requirements.txt 全锁定）"
& $venvPython -m pip --python (Join-Path $Target "python.exe") install -r (Join-Path $root "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "嵌入运行时依赖安装失败" }

Write-Host "==> 导入冒烟"
& (Join-Path $Target "python.exe") -B -c "import scipy.optimize, scipy.signal, scipy.special, scipy.ndimage, skimage.metrics, cv2, onnxruntime, fastapi, uvicorn, pydicom, gmssl, Cryptodome, reportlab, sqlalchemy, alembic; print('embed imports OK')"
if ($LASTEXITCODE -ne 0) { throw "嵌入运行时导入冒烟失败" }
Write-Host "==> 嵌入运行时就绪：$Target"
