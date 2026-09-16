# INSTINCTS

## reportlab PDF 的生成器痕迹要重写文件才能除名（2026-09-17）

- **触发**：需要"无生成工具痕迹"的对外 PDF（软著鉴别材料等）；或打印源码
  到 PDF 时出现豆腐块。
- **正确做法**：`setProducer/setCreator` 只覆盖信息字典，文件头尾仍有
  `ReportLab Generated PDF document` 字节注释——用 pypdf
  （`PdfWriter.append(reader)` + 写回 + 回写 metadata）克隆重写才除得掉，
  验收必须做**字节级**扫描而非只看信息字典。SimHei 缺 ²/⇒/⚠/−(U+2212)
  等字形，打印前按 `getFont(face).charToGlyph` 逐字符守卫，缺字替换为 `?`。
- **证据**：软著材料包三轮自查均漏掉字节层，红队审查在 @15/@285932 偏移
  抓到；第 17/25 页豆腐块来自 batch_queue 注释里的 O(N²) 与 ⇒。


## 前端 UI 自动化截图：整页 goto 会丢会话，用应用内导航（2026-09-17）

- **触发**：用 puppeteer/浏览器自动化对 Electron 应用（Vue Router + sessionStorage 会话）
  截各工作区页面，`page.goto('/std-eval')` 后页面跳回工作台/登录态丢失。
- **正确做法**：会话标志存 sessionStorage，整页加载会重建内存态——从登录页点
  "访客模式"进入后，改用 `document.querySelectorAll('a,button')` 按文字找标签页
  click 导航（SPA 内不丢状态）；错误提示、文件上传用
  `input[type=file]` 的 `uploadFile()` 可绕过原生文件对话框。另外"评定进行中"
  状态转瞬即逝（样例底片推理仅 3～4 秒），要 submit 后 500ms 内抓拍；查看器
  底片是"会话内同步"，须先完成一次评定再看。软著材料生成器在
  `scripts/make_copyright_materials.py`（program/manual 两子命令，重跑即可换名/换版重出）。
- **证据**：截图三轮才齐；first round 06/07/08 三张因 goto 全部截错页面。


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

## Electron 壳注入的自定义请求头必须进 CORS 白名单（2026-09-16）

- **触发**：给 webview 页面注入新的自定义请求头（如 `X-IPC-Token`）时。
- **正确做法**：同步把该头加进 `main.py` 的 `CORSMiddleware.allow_headers`；
  且 `IpcTokenMiddleware` 必须豁免 OPTIONS 预检（预检按规范不带自定义头）。
  漏掉任一处，渲染进程所有请求死在预检层，症状是"后端活着、UI 永远
  显示未连接"，而主进程探测（不走 CORS）与浏览器 dev（不注入令牌）全部
  正常——三层观测互相矛盾时要先想到 CORS 预检。
- **证据**：安装版"一直打不开"即此因；`allow_headers` 有 Tauri 时代的
  `X-Export-Token` 却漏了 Electron 新增的 `X-IPC-Token`，回归测试
  `test_cors_preflight_allows_ipc_token_header` 已补。修复后症状降级为
  "正在连接推理服务"永不变绿——那是第二层 bug：UP 事件是一次性的，
  状态栏在登录页之后才挂载必然错过，组件须从 store 响应式状态派生
  （backendUp 标志），不能只订阅事件。

## 遗留库 stamp head 会静默跳过列迁移（2026-09-16）

- **触发**：给已有列变更的新迁移（如 0014_film_no）排查"表不存在某列"的
  OperationalError；或给老安装版/开发库跑评定归档必炸而全新库正常。
- **正确做法**：`migrate.py` 对无 alembic_version 的遗留库 `stamp head`
  只推进版本号、不执行 DDL，而 `create_all` 只建缺失表、**永不补列**——
  新增列必须靠列级自愈（`_reconcile_missing_columns`，ORM 元数据为真源，
  幂等 ADD COLUMN）或在 finally 里兜底。排查时先对照
  `alembic_version` 与 `PRAGMA table_info`：版本到头而物理列缺失 = 此坑。
- **证据**：安装版 0014 stamp 后 images 无 film_no，评定归档 INSERT 即炸
  （用户截图"评定任务失败"）；开发库同病（版本卡 0012 + 0013 撞表回退，
  film_no 同样缺失）。

## 服务端错误响应不带 CORS 头，前端误报"无法连接"（2026-09-16）

- **触发**：后端日志明明有请求进来且抛了异常（或应返回 401/429），前端却
  显示"无法连接本地推理服务"（BACKEND_UNREACHABLE/TypeError），三层观测
  互相矛盾时。
- **正确做法**：CORS 中间件只装饰**流经它**的响应——它必须挂最外层
  （create_app 中最后 add），否则任何在外层直接返回的错误响应（IPC 令牌
  401、限流 429、ServerErrorMiddleware 的裸 500）都没有
  `Access-Control-Allow-*`，跨源页面一律拦成 TypeError：500 误报成断连、
  401（令牌刷新窗口）也误报成断连。固定顺序（内→外）：UnhandledException
  → SecurityHeaders → RateLimit → IpcToken → Metrics → CORS。与"CORS 预检
  漏头"同族不同层（预检漏头=全部请求死；本坑=只有错误响应死）。
- **证据**：film_no 事故的真实表象正是"无法连接"而非"服务内部错误"；
  IPC 401 被误报断连是同构隐患（后端重启换令牌必现窗口）。
  `test_unhandled_exception_cors.py` + `test_ipc_401_response_carries_cors_headers` 锚定。

## 纯 Python 国密是大底片归档的性能天花板（2026-09-16）

- **触发**：单张评定数十秒到数分钟、且耗时与影像文件大小成正比；分段
  画像时"落库/落盘"一档独大。
- **正确做法**：gmssl 纯 Python SM4-CTR 吞吐 ~100-200 KB/s，大底片
  （10-20MB PNG）单张加密数十秒——SDC2 信封的 SM4-CTR/HMAC-SM3 用
  `cryptography`（OpenSSL 后端）原语替换，信封与密钥流逐字节不变
  （同标准算法），SM2 签名留 gmssl。迁移时必须留"gmssl 参考实现逐字节
  对照"的兼容锚点测试。性能问题先跑
  `scripts/profile_single_eval.py` 分段画像再动手——本次最大头是加密
  （157s→0.35s），不是直觉上的检测器（~2.6s）。
- **证据**：3000×8000 底片端到端 258s → 10.4s；1200 万像素照片
  78.8s → 6.8s。

## 安装版 LLM external 端点别用常见端口（2026-09-16）

- **触发**：安装版 `llm.mode=external` 指向 `127.0.0.1:8080` 这类常被占用端口。
- **正确做法**：优先自托管（`tools/llama` + GGUF 拷入安装目录后改
  `mode: managed`，独占冷门端口 18780）；机器 RAM/显存吃紧退而 external 时，
  端点指向专用服务，发现"已连接已有服务（模型 N 个）"日志要核对是不是
  自己的服务在应答。自托管 GPU 层（`n_gpu_layers`）按当期显存余量定，
  显存被大模型占满时改 0 走 CPU。
- **证据**：安装版曾静默"借宿"用户自己的 model-proxy(8080)；本机 30B
  llama-server 常驻吃 11.4GB RAM + 14.9GB 显存，自托管 4B 须等资源空闲。

## 降级路径只豁免"阻断"不重算结果字段，特性等于没做（2026-09-16）

- **触发**：排查"某类影像永远显示不可评片/不合格"、且单测全绿实机必现时。
- **正确做法**：降级逻辑若只跳过 raise/拦截而不重算 `evaluable` 这类结果
  字段，字段仍会被最严门禁一票否决（照片必为 8bit，位深硬门禁
  `allow_8bit=false` 恒假）——界面永远不可评片、评级熔断，翻拍可评形同
  虚设。修法：按策略口径重算结果字段（`evaluable = pd.passed if
  photo_advisory else gate_evaluable`），并与姊妹端点对齐（/verify 早已是
  `evaluable=pd.passed`，两端口径分叉本身就是信号）；前端展示勿只映射单一
  布尔，横幅/报告页用 `photo_mode`/`need_review` 出降级文案，PDF 结论对
  "有级别+需复核"补限定语。**测试全绿 ≠ 实机行为**：conftest 注入的
  `SCAN_GATE__ALLOW_8BIT=true` 让位深门禁在测试里恒放行，排查时先对照
  conftest env 与 default.yaml 的差异。
- **证据**：定检照片全部"不可评片"（photo_mode=True 但 evaluable 恒
  false）；修复后真实照片输出 Ⅳ 级预筛 + basis 首条 ⚠ 声明 + 强制人工
  复核，PDF 结论带复核限定语。
