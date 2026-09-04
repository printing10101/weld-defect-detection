// ScanDetection Tauri 入口（打包版，Tauri v2）。
// 启动流程：
//   1. 在 setup 阶段于后台启动后端（uvicorn，监听 127.0.0.1:18773）；
//   2. 后端冷启动（首次运行需导入重依赖并加载 ONNX 模型，叠加杀毒扫描）可能超过
//      60s，等待上限放宽到 180s；窗口不阻塞显示，前端会自动轮询 /health 直到就绪；
//   3. 前端（已构建的 Vue SPA）通过绝对地址 http://127.0.0.1:18773/api/v1 调用后端。
//
// 目录布局约定：
//   安装布局（NSIS 安装 / 直接运行 target/release/ScanDetection.exe）：
//     <exe 目录>/python_embed/python.exe、<exe 目录>/backend
//   开发布局（tauri dev，exe 位于 target/debug）：
//     向上回溯查找包含 backend/ 的祖先目录（即项目根），并用其 src/python_embed。
//   找不到时回退到系统 PATH 的 python.exe。
//
// 后端 stdout/stderr 写入 %TEMP%/ScanDetection/backend.log，便于启动失败时排查。
use std::io;
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use tauri::Manager;

const BACKEND_PORT: u16 = 18773;
/// 后端冷启动耗时上限：实测首次运行（重依赖导入 + 模型加载 + Defender 扫描）
/// 可超过 60s，放宽到 180s 避免误报“后端未响应”。
const BACKEND_STARTUP_TIMEOUT_SECS: u64 = 180;

const SUPERVISOR_CHECK_INTERVAL_MS: u64 = 2000;

fn main() {
    // 供后台监督线程与窗口事件钩子各自持有一份 Arc 克隆（引用计数，零额外开销）。
    let launch_slot: Arc<Mutex<Option<Child>>> = Arc::new(Mutex::new(None));
    let event_slot = launch_slot.clone();
    // 停止标志：窗口销毁即置位，监督线程据此退出并回收后端子进程。
    let stop = Arc::new(AtomicBool::new(false));
    let supervisor_stop = stop.clone();
    let event_stop = stop.clone();

    tauri::Builder::default()
        .setup(move |app| {
            // 单一后台监督线程：负责首次启动 + 存活监控 + 崩溃自愈 + 优雅退出回收；
            // 窗口不阻塞显示，前端通过轮询 /health 自动恢复。
            let handle = app.handle().clone();
            thread::spawn(move || {
                let dir = match exe_dir() {
                    Ok(d) => d,
                    Err(e) => {
                        eprintln!("[ScanDetection] cannot resolve exe dir: {e}");
                        return;
                    }
                };
                let app_root = resolve_app_root(&dir);
                let python = pick_python(&app_root);

                // 首启后端并等待就绪。
                try_spawn_backend(&launch_slot, &app_root, &python);
                let ready = wait_for_port_stoppable(
                    "127.0.0.1",
                    BACKEND_PORT,
                    BACKEND_STARTUP_TIMEOUT_SECS,
                    &supervisor_stop,
                );
                if ready {
                    // C-17：后端就绪（端口可连 = lifespan 完成）后注入 IPC 一次性令牌。
                    inject_ipc_token(&handle, &app_root);
                }

                // 进入存活监控/自愈循环（含看门狗重启标记消费）。
                let marker = app_root.join("data").join("restart_required");
                run_supervisor(
                    launch_slot,
                    supervisor_stop,
                    marker,
                    SUPERVISOR_CHECK_INTERVAL_MS,
                    handle,
                    app_root,
                    python,
                );
            });
            Ok(())
        })
        .on_window_event(move |_window, event| {
            // 主窗口销毁即代表应用退出：置位停止标志并回收后端子进程，
            // 杜绝孤儿进程/端口占用（与监督线程回收双保险）。
            if let tauri::WindowEvent::Destroyed = event {
                event_stop.store(true, Ordering::SeqCst);
                if let Ok(mut guard) = event_slot.lock() {
                    if let Some(mut child) = guard.take() {
                        let _ = child.kill();
                        println!("[ScanDetection] backend process stopped");
                    }
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

/// 返回当前可执行文件所在目录（安装布局下即“安装目录”）。
fn exe_dir() -> io::Result<PathBuf> {
    let exe = std::env::current_exe()?;
    exe.parent()
        .map(|p| p.to_path_buf())
        .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "cannot resolve exe dir"))
}

/// 解析应用根目录：优先 exe 所在目录；若不含 backend/（开发布局，exe 在
/// target/debug），则向上回溯找包含 backend/ 的祖先目录。
fn resolve_app_root(dir: &std::path::Path) -> PathBuf {
    let mut cur = Some(dir);
    while let Some(d) = cur {
        if d.join("backend").is_dir() {
            return d.to_path_buf();
        }
        cur = d.parent();
    }
    dir.to_path_buf()
}

/// 按优先级选择 Python 解释器（S-02 国产 OS 打包，按目标平台条件编译）：
/// Windows：<根>/python_embed/python.exe → <根>/src/python_embed/python.exe（开发布局）
///        → <根>/venv/Scripts/python.exe → 系统 PATH 的 python.exe。
#[cfg(not(target_os = "linux"))]
fn pick_python(app_root: &std::path::Path) -> PathBuf {
    for base in [app_root.to_path_buf(), app_root.join("src")] {
        let embed = base.join("python_embed").join("python.exe");
        if embed.exists() {
            return embed;
        }
    }
    let venv = app_root.join("venv").join("Scripts").join("python.exe");
    if venv.exists() {
        return venv;
    }
    PathBuf::from("python.exe")
}

/// Linux（麒麟 V10 / UOS 适配，未真机验证）：pyenv/venv 布局为 bin/python3，
/// 侧车目录约定为 <根>/python_embed/bin/python3 → <根>/venv/bin/python3 → 系统 PATH python3。
#[cfg(target_os = "linux")]
fn pick_python(app_root: &std::path::Path) -> PathBuf {
    for base in [app_root.to_path_buf(), app_root.join("src")] {
        let embed = base.join("python_embed").join("bin").join("python3");
        if embed.exists() {
            return embed;
        }
    }
    let venv = app_root.join("venv").join("bin").join("python3");
    if venv.exists() {
        return venv;
    }
    PathBuf::from("python3")
}

/// 后端日志输出对（stdout/stderr），写入 %TEMP%/ScanDetection/backend.log。
/// 用 TEMP 而非安装目录：安装到 Program Files 时安装目录不可写。
/// 返回两个独立句柄分别接子进程的 stdout 与 stderr（追加写入，保留历史记录）。
fn backend_log_stdio() -> Option<(Stdio, Stdio)> {
    let dir = std::env::temp_dir().join("ScanDetection");
    std::fs::create_dir_all(&dir).ok()?;
    let path = dir.join("backend.log");
    let out = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .ok()?;
    let err = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .ok()?;
    Some((Stdio::from(out), Stdio::from(err)))
}

/// 构造 uvicorn 启动命令（stdout/stderr 归并到同一天日志文件）。
fn build_uvicorn_command(app_root: &std::path::Path, python: &std::path::Path) -> Command {
    let mut cmd = Command::new(python);
    cmd.arg("-m")
        .arg("uvicorn")
        .arg("backend.app.main:app")
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(BACKEND_PORT.to_string())
        .current_dir(app_root);
    // 孤儿兜底：把壳自身 PID 传给后端，后端据此监控父进程消失即自杀退出
    //（覆盖壳被强杀/崩溃、窗口 Destroyed 事件不触发而遗留孤儿后端的场景）。
    cmd.env("SCANDETECTION_PARENT_PID", std::process::id().to_string());
    match backend_log_stdio() {
        Some((out, err)) => {
            cmd.stdout(out).stderr(err);
        }
        None => {
            cmd.stdout(Stdio::null()).stderr(Stdio::null());
        }
    }
    #[cfg(windows)]
    {
        // CREATE_NO_WINDOW：避免后端启动时闪出黑色控制台窗口。
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000);
    }
    cmd
}

/// 组装命令并 spawn 后端子进程，句柄写入 `child_slot`。
/// 成功返回 true；spawn 失败打日志返回 false（不 panic，交由监控循环下轮重试）。
fn try_spawn_backend(
    child_slot: &Arc<Mutex<Option<Child>>>,
    app_root: &std::path::Path,
    python: &std::path::Path,
) -> bool {
    match build_uvicorn_command(app_root, python).spawn() {
        Ok(mut child) => {
            match child_slot.lock() {
                Ok(mut slot) => *slot = Some(child),
                Err(_) => {
                    // 锁已毒化（理论上不应发生）：放弃托管，先杀掉避免成为孤儿进程。
                    let _ = child.kill();
                    return false;
                }
            }
            println!(
                "[ScanDetection] backend launched via {:?} (cwd={:?})",
                python, app_root
            );
            true
        }
        Err(e) => {
            eprintln!("[ScanDetection] backend spawn failed: {e}");
            false
        }
    }
}

/// 轮询 TCP 端口，直到可连接或超时（stop 置位时提前返回中断等待）。
fn wait_for_port_stoppable(
    host: &str,
    port: u16,
    timeout_secs: u64,
    stop: &AtomicBool,
) -> bool {
    let addr = format!("{}:{}", host, port);
    let deadline = Duration::from_secs(timeout_secs);
    let start = std::time::Instant::now();
    loop {
        if stop.load(Ordering::SeqCst) {
            return false; // 应用退出中，中止等待
        }
        if TcpStream::connect(&addr).is_ok() {
            println!("[ScanDetection] backend ready on {}", addr);
            return true;
        }
        if start.elapsed() > deadline {
            eprintln!(
                "[ScanDetection] 等待后端超时（{}s），前端可能暂时无法连接 API。",
                timeout_secs
            );
            return false;
        }
        thread::sleep(Duration::from_millis(500));
    }
}

/// 后端存活监控与自愈（S-09 配套：Tauri 壳侧崩溃重启）。
///
/// 在应用生命周期内循环：
/// 1. 消费看门狗标记文件 `data/restart_required`（后端内存超阈值写的优雅重启请求）；
/// 2. 用 `try_wait` 轮询后端子进程——若已退出（崩溃/被杀/人为终止）则自动重启，
///    并重新注入新生成的一次性 IPC 令牌（后端重启会刷新令牌，需重注入避免 401）；
/// 3. 按 `check_interval_ms` 周期循环，`stop` 置位（窗口销毁）时退出并回收子进程。
///
/// 说明：Tauri 进程在 main() 返回时随宿主退出，本线程会在该时机被打断；正常
/// 关机路径由 on_window_event 直接 kill + 置位 stop 兜底，避免后端子进程残留。
#[allow(clippy::too_many_arguments)]
fn run_supervisor(
    child_slot: Arc<Mutex<Option<Child>>>,
    stop: Arc<AtomicBool>,
    marker_path: PathBuf,
    check_interval_ms: u64,
    handle: tauri::AppHandle,
    app_root: PathBuf,
    python: PathBuf,
) {
    let interval = Duration::from_millis(check_interval_ms.max(200));
    while !stop.load(Ordering::SeqCst) {
        // 1) 消费看门狗优雅重启标记（存在即触发一次重启）。
        let marker_restart = if marker_path.exists() {
            let _ = std::fs::remove_file(&marker_path); // 先删标记，防重复重启
            eprintln!("[ScanDetection] watchdog restart marker present; restarting backend");
            true
        } else {
            false
        };

        // 2) 检测后端子进程是否已退出。
        let exited = {
            match child_slot.lock() {
                Ok(mut guard) => match guard.as_mut() {
                    Some(c) => match c.try_wait() {
                        Ok(Some(_)) => true, // 已退出（优雅或被收割）
                        Ok(None) => false,   // 仍在运行
                        Err(_) => true,      // 句柄异常，视作死亡
                    },
                    None => true, // 无子进程，需首启或重启
                },
                Err(_) => {
                    eprintln!("[ScanDetection] supervisor lock poisoned; exiting");
                    return;
                }
            }
        };

        if exited || marker_restart {
            // 应用退出中：不得再 spawn（与 on_window_event 的 kill 存在竞态窗口，
            // 此前置位检查杜绝"窗口已销毁又拉起新后端"的双 spawn）。
            if stop.load(Ordering::SeqCst) {
                break;
            }
            // 3) 回收旧句柄 → 重新拉起 → 等端口就绪 → 重注入令牌。
            {
                if let Ok(mut guard) = child_slot.lock() {
                    if let Some(mut c) = guard.take() {
                        let _ = c.kill();
                    }
                }
            }
            try_spawn_backend(&child_slot, &app_root, &python);
            let ready = wait_for_port_stoppable(
                "127.0.0.1",
                BACKEND_PORT,
                BACKEND_STARTUP_TIMEOUT_SECS,
                &stop,
            );
            if ready {
                inject_ipc_token(&handle, &app_root); // 后端重启用新令牌，重注入 WebView
            }
        }

        // 4) 分片睡眠：周期内也能及时响应 stop（应用退出）。
        let mut remaining = interval;
        while remaining > Duration::ZERO && !stop.load(Ordering::SeqCst) {
            let step = remaining.min(Duration::from_millis(200));
            thread::sleep(step);
            remaining -= step;
        }
    }
    // stop 置位：回收后端子进程（与 on_window_event 双保险）。
    if let Ok(mut guard) = child_slot.lock() {
        if let Some(mut c) = guard.take() {
            let _ = c.kill();
            println!("[ScanDetection] backend process stopped (supervisor)");
        }
    }
}

/// C-17：读取后端落盘的一次性 IPC 令牌并注入 WebView（window.__IPC_TOKEN__）。
///
/// 后端在 lifespan 启动时生成令牌写入 <应用根>/data/ipc_token（进程生命周期
/// 有效）；端口可连即 lifespan 已完成，令牌已就绪。注入后前端 services/api.ts
/// 统一携带 X-IPC-Token 头。
///
/// 诚实边界：令牌防"其他本机进程误调 / 网页 CSRF 式调用"，本机回环为明文
/// 传输，令牌不解决传输加密（需 TLS 时后续挂本机证书，不在本次范围）。
/// 文件权限为尽力而为（Windows 下依赖用户数据目录继承的 ACL）。
fn inject_ipc_token(handle: &tauri::AppHandle, app_root: &std::path::Path) {
    let path = app_root.join("data").join("ipc_token");
    // 令牌落盘与端口就绪存在毫秒级竞态：短重试兜底（正常一次即中）。
    let mut token: Option<String> = None;
    for _ in 0..5 {
        match std::fs::read_to_string(&path) {
            Ok(t) if !t.trim().is_empty() => {
                token = Some(t.trim().to_string());
                break;
            }
            _ => thread::sleep(Duration::from_millis(300)),
        }
    }
    let token = match token {
        Some(t) => t,
        None => {
            // 高危告警：后端 ipc.enforce=true 时前端将因缺令牌持续 401、
            // 全部业务请求不可用——运维须立即查看后端日志。
            eprintln!(
                "[ScanDetection] 高危：读取 IPC 令牌失败: {:?}（若 ipc.enforce=true，前端将持续 401；请查后端日志）",
                path
            );
            return;
        }
    };
    if let Some(win) = handle.get_webview_window("main") {
        // 令牌为 token_urlsafe 字符集（[A-Za-z0-9_-]），可安全内插 JS 字符串。
        let js = format!("window.__IPC_TOKEN__ = '{}';", token);
        match win.eval(&js) {
            Ok(()) => println!("[ScanDetection] IPC token injected into webview"),
            Err(e) => eprintln!("[ScanDetection] IPC 令牌注入 WebView 失败: {e}"),
        }
    } else {
        eprintln!("[ScanDetection] 未找到主窗口，IPC 令牌未注入");
    }
}
