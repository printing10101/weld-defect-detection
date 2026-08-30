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
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use tauri::Manager;

const BACKEND_PORT: u16 = 18773;
/// 后端冷启动耗时上限：实测首次运行（重依赖导入 + 模型加载 + Defender 扫描）
/// 可超过 60s，放宽到 180s 避免误报“后端未响应”。
const BACKEND_STARTUP_TIMEOUT_SECS: u64 = 180;

/// 后端子进程句柄的共享槽位：随应用生命周期持有，窗口销毁时显式回收，
/// 避免 `mem::forget` 造成的孤儿进程（应用退出后 uvicorn 仍常驻占用端口）。
struct BackendState {
    child: Arc<Mutex<Option<Child>>>,
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

/// 在后台启动后端；返回前先阻塞等待端口就绪（最多 180s）。
/// 子进程句柄写入 `child_slot`，由窗口关闭钩子负责回收。
fn launch_backend(child_slot: Arc<Mutex<Option<Child>>>) -> io::Result<()> {
    let dir = exe_dir()?;
    let app_root = resolve_app_root(&dir);
    let python = pick_python(&app_root);

    let mut cmd = Command::new(&python);
    cmd.arg("-m")
        .arg("uvicorn")
        .arg("backend.app.main:app")
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(BACKEND_PORT.to_string())
        .current_dir(&app_root);
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
    let mut child = cmd.spawn()?;

    // 记录句柄供退出时回收；不再 mem::forget（那会泄漏后端进程）。
    if let Ok(mut slot) = child_slot.lock() {
        *slot = Some(child);
    } else {
        // 锁已毒化（理论上不应发生）：放弃托管，先杀掉避免成为无人管理的孤儿进程。
        let _ = child.kill();
    }

    println!(
        "[ScanDetection] backend launched via {:?} (cwd={:?})",
        python, app_root
    );

    // 等待端口就绪（uvicorn 在应用 lifespan 完成后才绑定端口，端口可连即模型已加载）。
    wait_for_port("127.0.0.1", BACKEND_PORT, BACKEND_STARTUP_TIMEOUT_SECS);
    Ok(())
}

/// 轮询 TCP 端口，直到可连接或超时。
fn wait_for_port(host: &str, port: u16, timeout_secs: u64) {
    let addr = format!("{}:{}", host, port);
    let deadline = Duration::from_secs(timeout_secs);
    let start = std::time::Instant::now();
    loop {
        if TcpStream::connect(&addr).is_ok() {
            println!("[ScanDetection] backend ready on {}", addr);
            return;
        }
        if start.elapsed() > deadline {
            eprintln!(
                "[ScanDetection] 等待后端超时（{}s），前端可能暂时无法连接 API。",
                timeout_secs
            );
            return;
        }
        thread::sleep(Duration::from_millis(500));
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
fn inject_ipc_token(handle: &tauri::AppHandle) {
    let dir = match exe_dir() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[ScanDetection] IPC 令牌注入失败（无法定位安装目录）: {e}");
            return;
        }
    };
    let path = resolve_app_root(&dir).join("data").join("ipc_token");
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
            eprintln!(
                "[ScanDetection] 读取 IPC 令牌失败: {:?}（后端 ipc.enforce=false 时无需令牌）",
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

fn main() {
    let backend_state = BackendState {
        child: Arc::new(Mutex::new(None)),
    };
    // 供后台启动线程与窗口事件钩子各自持有一份 Arc 克隆（引用计数，零额外开销）。
    let launch_slot = backend_state.child.clone();
    let event_slot = backend_state.child.clone();

    tauri::Builder::default()
        .manage(backend_state)
        .setup(move |app| {
            // 后台启动后端并等待就绪；窗口不阻塞，前端通过轮询 /health 自动恢复。
            let handle = app.handle().clone();
            thread::spawn(move || {
                if let Err(e) = launch_backend(launch_slot) {
                    eprintln!("[ScanDetection] launch_backend error: {e}");
                }
                // C-17：后端就绪（端口可连 = lifespan 完成）后注入 IPC 一次性令牌。
                inject_ipc_token(&handle);
            });
            Ok(())
        })
        .on_window_event(move |_window, event| {
            // 主窗口销毁即代表应用退出，回收后端子进程，杜绝孤儿进程/端口占用。
            if let tauri::WindowEvent::Destroyed = event {
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
