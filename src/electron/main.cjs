// ScanDetection Electron 入口（桌面壳，替代 Tauri/Rust 壳）。
// 启动流程与原 main.rs 一致：
//   1. 单实例锁：第二个实例把焦点交还已运行实例后退出（双开会 spawn 两个后端，
//      后到者绑 18773 失败 → 监督循环秒级重生风暴，故锁必须是第一件事）；
//   2. 用户数据目录固定为 %APPDATA%/com.scandetection.sd（与 Tauri 版
//      app_data_dir 同路径，升级后 scan.db/主密钥/审计链无缝延续）；
//   3. 后台拉起后端（uvicorn，监听 127.0.0.1:18773）并持续探测 /api/v1/health，
//      就绪即把 ipc_token 注入页面（window.__IPC_TOKEN__）；冷启动（重依赖导入
//      + 模型加载）可能数分钟，注入不设截止时间；
//   4. 监督循环：崩溃自愈（8s 退避）、看门狗优雅重启标记消费、令牌补注入；
//   5. 页面经自定义标准协议 app://scandetection 伺服 dist（origin 明确，
//      CORS 可精确放行；file:// 的 null origin 无法进 CORS 白名单）。
//
// 后端 stdout/stderr 写入 %TEMP%/ScanDetection/backend.log，壳自身日志写
// shell.log（原 Rust 版 println! 随 GUI 子系统丢失，此处落盘便于排查）。
"use strict";

const { app, BrowserWindow, Menu, ipcMain, protocol, session } = require("electron");
const { pathToFileURL } = require("url");
const path = require("path");
const fs = require("fs");
const http = require("http");
const { spawn, spawnSync } = require("child_process");

const BACKEND_PORT = 18773;
// 单轮就绪等待上限：仅约束一次轮询窗口（监督循环会继续探测并补注入令牌），
// 不是"超时即放弃"的硬截止——后端多晚就绪都能补上（杜绝"令牌永远缺失 →
// 前端全部 401 → 用户以为后端没启动"的静默故障）。
const BACKEND_STARTUP_TIMEOUT_SECS = 180;
// 崩溃循环退避阈值：子进程拉起后存活不足该时长即死亡（端口被占/缺依赖秒退）
// 时，重启前先退避一轮，避免秒级重生风暴。
const BACKEND_CRASH_BACKOFF_SECS = 8;
const SUPERVISOR_CHECK_INTERVAL_MS = 2000;
// 单代日志上限：每次拉起后端前滚动一次（.log → .log.1），长期运行不无限增长。
const BACKEND_LOG_MAX_BYTES = 10 * 1024 * 1024;
// 与 tauri.conf.json 的 app.security.csp 同源：生产构建的前端 API 地址是
// 绝对路径（VITE_API_BASE=http://127.0.0.1:18773/api/v1），connect/img 须放行。
const PROD_CSP =
  "default-src 'self'; connect-src 'self' http://127.0.0.1:18773 http://localhost:18773; " +
  "img-src 'self' data: blob: http://127.0.0.1:18773 http://localhost:18773; " +
  "style-src 'self' 'unsafe-inline'; font-src 'self' data:; script-src 'self'";
const APP_ORIGIN = "app://scandetection";

// 运行期状态（单窗口应用，全部收拢为模块级变量）。
let mainWindow = null;
let backendChild = null;
let backendExitAt = 0; // 子进程最近一次退出时刻（0=存活中/未拉起）
let lastSpawnAt = 0; // 最近一次拉起时刻（崩溃退避判定用）
let tokenInjected = false;
let stopping = false; // 应用退出中：监督循环不得再 spawn
let supervisorTimer = null;
let busyTaskActive = false; // 渲染层有长任务（批量/单幅评定）进行中
let closeConfirmed = false; // 用户经确认框同意关闭后放行 close

// ---------------------------------------------------------------------------
// 日志：壳自身事件落盘 %TEMP%/ScanDetection/shell.log（追加）。
// ---------------------------------------------------------------------------
function log(message) {
  const line = `[${new Date().toISOString()}] [ScanDetection] ${message}`;
  console.log(line);
  try {
    const dir = path.join(app.getPath("temp"), "ScanDetection");
    fs.mkdirSync(dir, { recursive: true });
    fs.appendFileSync(path.join(dir, "shell.log"), line + "\n");
  } catch {
    /* 日志写不进去不阻断主流程 */
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ---------------------------------------------------------------------------
// 路径解析（对应 main.rs 的 exe_dir/resolve_app_root/resolve_data_dir/pick_python）。
// ---------------------------------------------------------------------------
// 解析应用根目录：开发布局从 src/electron 向上回溯找含 backend/ 的祖先（仓库根）；
// 打包布局 = resources 目录（extraResources 已把 backend/、python_embed/ 落在其下）。
function resolveAppRoot() {
  if (app.isPackaged) return process.resourcesPath;
  return resolveAppRootDev();
}

// 开发布局的仓库根定位：不依赖 app 就绪状态（userData 解析在 ready 前就要用）。
function resolveAppRootDev() {
  let cur = __dirname;
  while (true) {
    if (fs.existsSync(path.join(cur, "backend"))) return cur;
    const parent = path.dirname(cur);
    if (parent === cur) break;
    cur = parent;
  }
  return path.join(__dirname, "..");
}

// 按优先级选择 Python 解释器（S-02 国产 OS 适配，与 main.rs pick_python 同序）：
// <根>/python_embed → <根>/src/python_embed（开发布局）→ venv → 系统 PATH。
function pickPython(appRoot) {
  const candidates =
    process.platform === "win32"
      ? [
          path.join(appRoot, "python_embed", "python.exe"),
          path.join(appRoot, "src", "python_embed", "python.exe"),
          path.join(appRoot, "venv", "Scripts", "python.exe"),
        ]
      : [
          path.join(appRoot, "python_embed", "bin", "python3"),
          path.join(appRoot, "src", "python_embed", "bin", "python3"),
          path.join(appRoot, "venv", "bin", "python3"),
        ];
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) return candidate;
  }
  return process.platform === "win32" ? "python.exe" : "python3";
}

// ---------------------------------------------------------------------------
// 后端进程管理（对应 build_uvicorn_command/try_spawn_backend/backend_log_stdio）。
// ---------------------------------------------------------------------------
function openBackendLogStreams() {
  const dir = path.join(app.getPath("temp"), "ScanDetection");
  fs.mkdirSync(dir, { recursive: true });
  const logPath = path.join(dir, "backend.log");
  try {
    if (fs.statSync(logPath).size >= BACKEND_LOG_MAX_BYTES) {
      const rolled = path.join(dir, "backend.log.1");
      fs.rmSync(rolled, { force: true });
      fs.renameSync(logPath, rolled);
    }
  } catch {
    /* 首次无日志文件属正常 */
  }
  // 两个独立句柄分别接 stdout 与 stderr（追加写入，保留历史记录）。
  return [
    fs.openSync(logPath, "a"),
    fs.openSync(logPath, "a"),
  ];
}

// 构造 uvicorn 启动参数。环境变量与 main.rs 逐一对应：
//   SCANDETECTION_PARENT_PID —— 孤儿兜底：壳被强杀时后端监控父进程消失即自杀；
//   SCANDETECTION_USER_DATA_DIR —— data/ 前缀路径落到用户数据目录（打包版），
//     与壳侧 ipc_token / restart_required 同源；
//   PYTHONPYCACHEPREFIX —— 字节码缓存集中到 TEMP：嵌入态 Python 首启编译一次、
//     后续秒级导入；缓存不进安装目录，卸载器清单保持干净。
function buildUvicornCommand(appRoot, python, userDataDir) {
  const pycacheDir = path.join(app.getPath("temp"), "ScanDetection", "pycache");
  fs.mkdirSync(pycacheDir, { recursive: true });
  return {
    file: python,
    args: [
      "-m", "uvicorn", "backend.app.main:app",
      "--host", "127.0.0.1",
      "--port", String(BACKEND_PORT),
      // 关闭逐请求访问日志：直链下载经 ?access_token= 携带会话凭据，访问日志会
      // 把完整 URL 写进 %TEMP%（本机其他进程可读）；桌面场景亦无观测价值。
      "--no-access-log",
    ],
    cwd: appRoot,
    env: {
      ...process.env,
      SCANDETECTION_PARENT_PID: String(process.pid),
      SCANDETECTION_USER_DATA_DIR: userDataDir,
      PYTHONPYCACHEPREFIX: pycacheDir,
    },
  };
}

// 拉起后端子进程；spawn 失败打日志（不抛异常，交监督循环下轮重试）。
function trySpawnBackend(appRoot, python, userDataDir) {
  const cmd = buildUvicornCommand(appRoot, python, userDataDir);
  let stdio;
  try {
    stdio = openBackendLogStreams();
  } catch {
    stdio = null;
  }
  let child;
  try {
    child = spawn(cmd.file, cmd.args, {
      cwd: cmd.cwd,
      env: cmd.env,
      windowsHide: true, // 对应 CREATE_NO_WINDOW：不闪黑色控制台
      stdio: ["ignore", ...(stdio ?? ["ignore", "ignore"])],
    });
  } catch (e) {
    log(`backend spawn failed: ${e.message}`);
    if (stdio) for (const fd of stdio) fs.closeSync(fd);
    return false;
  }
  if (stdio) for (const fd of stdio) fs.closeSync(fd); // 句柄已移交子进程，壳侧即关
  backendChild = child;
  backendExitAt = 0;
  lastSpawnAt = Date.now();
  child.on("exit", () => {
    backendExitAt = Date.now();
    if (!stopping) log("backend process exited; supervisor will handle it");
  });
  log(`backend launched via ${python} (cwd=${appRoot})`);
  return true;
}

// 单次健康探测：GET /api/v1/health 收到 200 即视为自家后端就绪。
// 不用裸 TCP 连通判定——那会把"任意占用 18773 的进程"误判为就绪。
function probeBackendHealth() {
  return new Promise((resolve) => {
    const req = http.get(
      { host: "127.0.0.1", port: BACKEND_PORT, path: "/api/v1/health", timeout: 2000 },
      (res) => {
        res.resume();
        resolve(res.statusCode === 200);
      },
    );
    req.on("timeout", () => {
      req.destroy();
      resolve(false);
    });
    req.on("error", () => resolve(false));
  });
}

// 停止后端并回收整棵进程树（与监督循环回收双保险，杜绝孤儿/端口占用）。
function stopBackend() {
  stopping = true;
  if (supervisorTimer) clearTimeout(supervisorTimer);
  const child = backendChild;
  backendChild = null;
  if (!child) return;
  try {
    child.kill();
  } catch {
    /* 已退出则忽略 */
  }
  if (process.platform === "win32" && child.pid) {
    // 兜底：uvicorn 正常为单进程，kill 已足够；taskkill /T 覆盖意外派生场景。
    try {
      spawnSync("taskkill", ["/PID", String(child.pid), "/T", "/F"], {
        windowsHide: true,
        timeout: 5000,
      });
    } catch {
      /* 进程已死时 taskkill 报错属正常 */
    }
  }
  log("backend process stopped");
}

// ---------------------------------------------------------------------------
// IPC 令牌注入（对应 inject_ipc_token，C-17）。
// 后端 lifespan 启动时生成令牌写 <数据目录>/data/ipc_token（进程生命周期有效，
// 重启即刷新）；注入后前端 services/api.ts 统一携带 X-IPC-Token 头。
// 诚实边界：令牌防"其他本机进程误调/网页 CSRF 式调用"，回环明文传输，
// 不解决传输加密。
// ---------------------------------------------------------------------------
async function injectIpcToken() {
  const tokenPath = path.join(userDataDir, "data", "ipc_token");
  let token = null;
  // 令牌落盘与端口就绪存在毫秒级竞态：短重试兜底（正常一次即中）。
  for (let i = 0; i < 5 && !token; i += 1) {
    try {
      const content = fs.readFileSync(tokenPath, "utf8").trim();
      if (content) token = content;
    } catch {
      /* 尚未落盘，稍后重试 */
    }
    if (!token) await sleep(300);
  }
  if (!token) {
    // 高危告警：ipc.enforce=true 时前端将持续 401、全部业务请求不可用。
    log(`高危：读取 IPC 令牌失败: ${tokenPath}（若 ipc.enforce=true，前端将持续 401；请查 backend.log）`);
    return false;
  }
  const win = mainWindow;
  if (!win || win.isDestroyed()) return false;
  try {
    // JSON.stringify 保证任意字符集令牌安全内插为 JS 字面量。
    await win.webContents.executeJavaScript(
      `window.__IPC_TOKEN__ = ${JSON.stringify(token)};`,
    );
    log("IPC token injected into page");
    return true;
  } catch (e) {
    log(`IPC 令牌注入失败: ${e.message}`);
    return false;
  }
}

// 等待后端就绪（单轮窗口，对应 wait_for_backend_ready_stoppable）：就绪即注入
// 令牌；超时/子进程早死返回 false 交监督循环接手（持续探测、就绪补注入）。
async function waitForBackendReadyAndInject() {
  const deadline = Date.now() + BACKEND_STARTUP_TIMEOUT_SECS * 1000;
  while (Date.now() < deadline) {
    if (stopping) return false;
    if (backendExitAt) return false; // 启动期即退出：交监督循环快速重启
    if (await probeBackendHealth()) {
      log(`backend ready on 127.0.0.1:${BACKEND_PORT}`);
      tokenInjected = await injectIpcToken();
      return tokenInjected;
    }
    await sleep(500);
  }
  log(`backend not ready after ${BACKEND_STARTUP_TIMEOUT_SECS}s (still importing?); keep probing in supervisor loop`);
  return false;
}

// ---------------------------------------------------------------------------
// 监督循环（对应 run_supervisor）：存活监控 + 崩溃自愈 + 看门狗重启 + 补注入。
// ---------------------------------------------------------------------------
async function supervisorTick(appRoot, python) {
  // 1) 消费看门狗优雅重启标记（后端内存超阈值时写入；存在即重启一次，
  //    先删标记防重复重启）。
  let needRestart = false;
  const markerPath = path.join(userDataDir, "data", "restart_required");
  try {
    if (fs.existsSync(markerPath)) {
      fs.rmSync(markerPath, { force: true });
      log("watchdog restart marker present; restarting backend");
      needRestart = true;
    }
  } catch {
    /* 标记读取失败按无标记处理 */
  }

  // 2) 子进程已退出（崩溃/被杀/人为终止）→ 需要重启。
  if (!needRestart && backendExitAt) needRestart = true;

  if (needRestart) {
    if (stopping) return;
    // 崩溃退避：拉起后极短时间即死亡（端口冲突/缺依赖秒退）时先等一轮，
    // 避免秒级重生风暴；正常长存后的崩溃不受影响。
    if (!backendExitAt || Date.now() - lastSpawnAt < BACKEND_CRASH_BACKOFF_SECS * 1000) {
      const backoffDeadline = Date.now() + BACKEND_CRASH_BACKOFF_SECS * 1000;
      while (Date.now() < backoffDeadline && !stopping) await sleep(200);
      if (stopping) return;
    }
    trySpawnBackend(appRoot, python, userDataDir);
    tokenInjected = false;
    await waitForBackendReadyAndInject();
    return;
  }

  // 3) 令牌补注入：子进程存活但尚未注入（冷启动超过单轮窗口/首次注入失败）
  //    时持续探测，就绪即注入——后端多晚就绪前端都能恢复。
  if (!tokenInjected && (await probeBackendHealth())) {
    tokenInjected = await injectIpcToken();
  }
}

function scheduleSupervisor(appRoot, python) {
  const tick = async () => {
    if (stopping) return;
    try {
      await supervisorTick(appRoot, python);
    } catch (e) {
      log(`supervisor tick error: ${e.message}`);
    }
    if (!stopping) supervisorTimer = setTimeout(tick, SUPERVISOR_CHECK_INTERVAL_MS);
  };
  supervisorTimer = setTimeout(tick, SUPERVISOR_CHECK_INTERVAL_MS);
}

// ---------------------------------------------------------------------------
// 窗口与生命周期。
// ---------------------------------------------------------------------------
function createWindow(devUrl) {
  const win = new BrowserWindow({
    width: 1280,
    height: 800,
    minWidth: 1024,
    minHeight: 640,
    title: "射线焊缝缺陷智能检测系统",
    icon: resolveWindowIcon(),
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      spellcheck: false,
    },
  });
  mainWindow = win;

  // 点 × 关闭：有长任务时先拦下，交渲染层确认框（AppShell 的关闭确认对话框）。
  win.on("close", (event) => {
    if (closeConfirmed || !busyTaskActive) return; // 无长任务/已确认：放行
    event.preventDefault();
    if (!win.isDestroyed()) win.webContents.send("desktop:close-requested");
  });

  win.webContents.setWindowOpenHandler(() => ({ action: "deny" })); // 禁弹新窗
  win.webContents.on("will-navigate", (event, url) => {
    const allowed = devUrl ? url.startsWith(devUrl) : url.startsWith(APP_ORIGIN);
    if (!allowed) event.preventDefault(); // 禁导航到外部源
  });
  win.webContents.on("render-process-gone", (_event, details) => {
    log(`renderer gone: ${details.reason}`);
  });
  win.on("closed", () => {
    if (mainWindow === win) mainWindow = null;
  });

  if (devUrl) {
    win.loadURL(devUrl);
  } else {
    win.loadURL(`${APP_ORIGIN}/index.html`);
  }
}

function resolveWindowIcon() {
  // 打包版任务栏/窗口图标来自 exe 资源；此处主要服务开发态。
  const iconPath = path.join(__dirname, "icon.ico");
  return fs.existsSync(iconPath) ? iconPath : undefined;
}

// 生产态去菜单栏（界面自带 MenuBar 组件）；开发态保留默认菜单（DevTools 可达）。
function setupMenu() {
  if (app.isPackaged) Menu.setApplicationMenu(null);
}

// ---------------------------------------------------------------------------
// 渲染层桥接的 IPC（preload.cjs 暴露的 desktop API 的主进程侧）。
// ---------------------------------------------------------------------------
ipcMain.on("desktop:set-busy", (_event, busy) => {
  busyTaskActive = Boolean(busy);
});

ipcMain.on("desktop:confirm-quit", () => {
  // 用户已确认放弃长任务：置放行标志后重走 close（close 监听据此不再拦截）。
  closeConfirmed = true;
  busyTaskActive = false;
  const win = mainWindow;
  if (win && !win.isDestroyed()) win.close();
});

// ---------------------------------------------------------------------------
// 启动编排。
// ---------------------------------------------------------------------------
let userDataDir = null;

const gotSingleInstanceLock = app.requestSingleInstanceLock();
if (!gotSingleInstanceLock) {
  // 第二实例：把参数交给已运行实例后立即退出（避免后端双 spawn 风暴）。
  app.quit();
} else {
  app.on("second-instance", () => {
    const win = mainWindow;
    if (!win) return;
    if (win.isMinimized()) win.restore();
    win.focus();
  });

  // 数据目录（对应 main.rs resolve_data_dir）：打包版 = %APPDATA%/<identifier>
  // （NSIS 卸载清空安装目录，业务数据不能随之删除；且与 Tauri 版同路径，老
  // 用户升级后 scan.db/影像副本/主密钥/审计链原地延续）；开发版 = 仓库根
  // （沿用仓库 data/ 的既有开发数据，与 launch_app.vbs 调试流程同源）。
  userDataDir = app.isPackaged
    ? path.join(app.getPath("appData"), "com.scandetection.sd")
    : resolveAppRootDev();
  app.setPath("userData", userDataDir);

  // app:// 须在 ready 前注册为标准协议（页面 origin 才是 app://scandetection）。
  protocol.registerSchemesAsPrivileged([
    { scheme: "app", privileges: { standard: true, secure: true, supportFetchAPI: true, stream: true } },
  ]);

  app.whenReady().then(async () => {
    setupMenu();

    const appRoot = resolveAppRoot();
    const python = pickPython(appRoot);
    const distDir = path.join(__dirname, "..", "dist");
    const devUrl = process.env.VITE_DEV_SERVER_URL || null;

    // 生产：自定义协议伺服 dist + CSP 响应头（与 tauri.conf.json 同源）。
    if (!devUrl) {
      const { net } = require("electron");
      protocol.handle("app", (request) => {
        const url = new URL(request.url);
        let pathname = decodeURIComponent(url.pathname);
        if (pathname === "/" || pathname === "") pathname = "/index.html";
        const target = path.join(distDir, pathname);
        if (!target.startsWith(distDir)) return new Response("forbidden", { status: 403 });
        return net.fetch(pathToFileURL(target).toString()).catch(() =>
          net.fetch(pathToFileURL(path.join(distDir, "index.html")).toString()),
        );
      });
      session.defaultSession.webRequest.onHeadersReceived((details, callback) => {
        if (details.url.startsWith(APP_ORIGIN)) {
          callback({
            responseHeaders: {
              ...details.responseHeaders,
              "Content-Security-Policy": [PROD_CSP],
            },
          });
        } else {
          callback({});
        }
      });
    }

    createWindow(devUrl);

    // 数据目录先落定再拉后端（业务数据/令牌/看门狗标记都在其 data/ 下）。
    try {
      fs.mkdirSync(path.join(userDataDir, "data"), { recursive: true });
    } catch (e) {
      log(`cannot create data dir: ${e.message}`);
    }

    // 首启后端并等就绪注入令牌；之后进入监督循环（存活监控/自愈/补注入）。
    trySpawnBackend(appRoot, python, userDataDir);
    await waitForBackendReadyAndInject();
    scheduleSupervisor(appRoot, python);
  });

  // 所有窗口关闭 = 应用退出：回收后端进程树（与 stopBackend 兜底互为双保险）。
  app.on("window-all-closed", () => {
    stopBackend();
    app.quit();
  });

  app.on("before-quit", () => {
    stopping = true;
    if (supervisorTimer) clearTimeout(supervisorTimer);
  });
}
