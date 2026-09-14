# INSTINCTS

## 公开仓本机路径（2026-09-09）

- **触发**：写入校准表、评估 JSON、model card、DATA_LICENSE 等会提交的产物。
- **正确做法**：路径用相对路径或 `<user>` / `<project>` 占位符；提交前 `git grep 'C:\\Users'`。
- **证据**：`best.calibration.json`、`real_baseline.json` 等曾含 `D:\扫描检测软件\...`，清洗后 grep 为空。

## 权重与数据

- **触发**：训练产出 `best.onnx` / 真实底片。
- **正确做法**：权重与真实工业数据不入库；合成评估图可入库。
- **证据**：`backend/models/weights/` 仅有 README 与 calibration 元数据。

## electron-builder Windows 打包（2026-09-15）

- **触发**：`electron-builder` 在 Windows 上构建失败，报
  `Cannot create symbolic link : 客户端没有所需的特权`（winCodeSign 缓存解压）。
- **正确做法**：无管理员/开发者模式时的解法——首次失败的临时解压目录里内容
  其实已 99% 完整（只有 2 个 darwin dylib 符号链接建不出来，对 Windows 打包
  无用），把该目录复制为 `%LOCALAPPDATA%\electron-builder\Cache\winCodeSign\winCodeSign-2.6.0`
  即跳过下载重试。日志里 7z 的 Sub items Errors 只影响 macOS 侧文件。
- **证据**：迁移 Electron 后首次 `build_installer.ps1` 即栽在此；手工补缓存后
  一次通过并产出 225MB 安装包。

## Windows 下 spawn(pnpm, shell:true) 的进程树回收（2026-09-15）

- **触发**：Node 脚本用 `spawn("pnpm", [...], { shell: true })` 后 `child.kill()`。
- **正确做法**：`kill()` 只杀 cmd 外壳，底下的 node/子进程残留（vite 残留=
  5173 继续占用）；Windows 一律 `taskkill /PID <pid> /T /F` 连树回收。
  Electron 壳退出同理（`main.cjs` stopBackend 已内置）。
- **证据**：开发冒烟关闭窗口后 electron 全退、vite 却残留占着 5173。
