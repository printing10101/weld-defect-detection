# 安装包端到端冒烟验证（打包后运行）
# 用法：powershell -ExecutionPolicy Bypass -File scripts\smoke_test_installer.ps1 <安装包路径>
# 流程：静默安装 → 验证文件布局 → 启动应用 → /health 探活 → 关闭 → 静默卸载 → 验证数据保留
param(
    [Parameter(Mandatory = $true)]
    [string]$InstallerPath
)

$ErrorActionPreference = "Stop"
$pkgDir = "$env:LOCALAPPDATA\ScanDetection"   # Tauri NSIS installMode=currentUser 默认目录
$dataDir = "$env:APPDATA\com.scandetection.sd"

Write-Host "==> [0/6] 预检：18773 端口必须空闲（残留孤儿后端会造成假 PASS）" -ForegroundColor Cyan
$stale = Get-NetTCPConnection -LocalPort 18773 -State Listen -ErrorAction SilentlyContinue
if ($stale) {
    $stale | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 2
    if (Get-NetTCPConnection -LocalPort 18773 -State Listen -ErrorAction SilentlyContinue) {
        throw "端口 18773 被占用且无法清理，请手工处理后重试"
    }
}
Write-Host "    端口空闲"
# 自清理：上一轮冒烟/卸载的残留目录（.pyc 等）不得影响本轮断言
if (Test-Path $pkgDir) { Remove-Item -Recurse -Force $pkgDir }

Write-Host "==> [1/6] 静默安装（当前用户级，无需管理员）" -ForegroundColor Cyan
$proc = Start-Process -FilePath $InstallerPath -ArgumentList "/S" -PassThru -Wait
if ($proc.ExitCode -ne 0) { throw "静默安装失败 exit=$($proc.ExitCode)" }
if (-not (Test-Path "$pkgDir\ScanDetection.exe")) { throw "安装后未找到主程序" }
Write-Host "    已安装到 $pkgDir"

Write-Host "==> [2/6] 验证文件布局（壳/后端/嵌入Python/模型目录）" -ForegroundColor Cyan
foreach ($p in @(
    "$pkgDir\ScanDetection.exe",
    "$pkgDir\backend\app\main.py",
    "$pkgDir\backend\configs\default.yaml",
    "$pkgDir\backend\infra",
    "$pkgDir\backend\models\weights",
    "$pkgDir\python_embed\python.exe",
    "$pkgDir\python_embed\Lib\site-packages\uvicorn",
    "$pkgDir\python_embed\Lib\site-packages\onnxruntime"
)) {
    if (-not (Test-Path $p)) { throw "布局缺失: $p" }
}
Write-Host "    布局完整"

Write-Host "==> [3/6] 启动应用（后台）" -ForegroundColor Cyan
$app = Start-Process -FilePath "$pkgDir\ScanDetection.exe" -PassThru
try {
    $ready = $false
    foreach ($i in 1..120) {   # 最多等 240s（首启含模型加载/杀软扫描）
        Start-Sleep -Seconds 2
        try {
            $r = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:18773/api/v1/health" -TimeoutSec 3
            if ($r.StatusCode -eq 200) { $ready = $true; break }
        } catch { }
        if ($app.HasExited) { throw "应用进程提前退出 exit=$($app.ExitCode)" }
    }
    if (-not $ready) { throw "240s 内后端未就绪（查 %TEMP%\ScanDetection\backend.log）" }
    $health = (Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:18773/api/v1/health" -TimeoutSec 5).Content
    Write-Host "    后端就绪: $($health.Substring(0, [Math]::Min(160, $health.Length)))"

    Write-Host "==> [4/6] 验证数据目录重定向（业务数据落 %APPDATA%，卸载保留）" -ForegroundColor Cyan
    if (-not (Test-Path "$dataDir\data")) { throw "未找到用户数据目录 $dataDir\data" }
    Write-Host "    数据目录: $dataDir\data"

    Write-Host "==> [5/6] 造一条数据 → 卸载 → 验证保留" -ForegroundColor Cyan
    "smoke-marker" | Out-File -FilePath "$dataDir\data\smoke_marker.txt" -Encoding utf8
} finally {
    if (-not $app.HasExited) { Stop-Process -Id $app.Id -Force -ErrorAction SilentlyContinue }
    Get-Process -Name "python_embed","python" -ErrorAction SilentlyContinue |
        Where-Object { $_.Path -like "$pkgDir*" } | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

$uninstaller = "$pkgDir\uninstall.exe"
if (-not (Test-Path $uninstaller)) { throw "卸载器缺失（$uninstaller）——卸载/数据保留断言不可跳过" }
Write-Host "==> 静默卸载" -ForegroundColor Cyan
$un = Start-Process -FilePath $uninstaller -ArgumentList "/S" -PassThru -Wait
if ($un.ExitCode -ne 0) { throw "静默卸载失败 exit=$($un.ExitCode)" }
Start-Sleep -Seconds 3
if (Test-Path $pkgDir) {
    # 卸载器会删除其安装清单内文件；断言口径 = 无任何文件残留
    # （运行期可能留下个别空目录，无害）。
    $left = @(Get-ChildItem $pkgDir -Recurse -File -ErrorAction SilentlyContinue)
    if ($left.Count -gt 0) {
        throw "卸载后有文件残留: $(($left | ForEach-Object FullName) -join '; ')"
    }
    Remove-Item -Recurse -Force $pkgDir -ErrorAction SilentlyContinue
}
if (-not (Test-Path "$dataDir\data\smoke_marker.txt")) {
    throw "卸载后业务数据丢失（数据保护语义被破坏）"
}
Write-Host "==> [6/6] 通过：卸载后数据保留，布局正确，后端可一键拉起" -ForegroundColor Green
