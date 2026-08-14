// ScanDetection Tauri 入口（打包版，Tauri v2）。
// 启动流程：
//   1. 在 setup 阶段于后台启动后端（uvicorn，监听 127.0.0.1:18773）；
//   2. 等待后端端口就绪后再显示窗口（确保前端首次 API 调用不落空）；
//   3. 前端（已构建的 Vue SPA）通过绝对地址 http://127.0.0.1:18773/api/v1 调用后端。
//
// 后端路径约定（由安装程序 setup.ps1 在“安装目录”内布置）：
//   <安装目录>/venv/Scripts/python.exe   —— 虚拟环境 Python（打包时随安装脚本创建）
//   <安装目录>/backend                   —— 后端源码
// Tauri 二进制自身位于 <安装目录>/ScanDetection.exe。
use std::io;
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

const BACKEND_PORT: u16 = 18773;

/// 后端子进程句柄的共享槽位：随应用生命周期持有，窗口销毁时显式回收，
/// 避免 `mem::forget` 造成的孤儿进程（应用退出后 uvicorn 仍常驻占用端口）。
struct BackendState {
    child: Arc<Mutex<Option<Child>>>,
}

/// 返回当前可执行文件所在目录（即“安装目录”）。
fn exe_dir() -> io::Result<PathBuf> {
    let exe = std::env::current_exe()?;
    exe.parent()
        .map(|p| p.to_path_buf())
        .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "cannot resolve exe dir"))
}

/// 按优先级选择 Python 解释器：内置可嵌入 Python → 安装脚本创建的 venv → 系统 PATH。
fn pick_python(dir: &std::path::Path) -> PathBuf {
    let embed_python = dir.join("python_embed").join("python.exe");
    let venv_python = dir.join("venv").join("Scripts").join("python.exe");
    if embed_python.exists() {
        embed_python
    } else if venv_python.exists() {
        venv_python
    } else {
        PathBuf::from("python.exe")
    }
}

/// 在后台启动后端；返回前先阻塞等待端口就绪（最多 ~60s）。
/// 子进程句柄写入 `child_slot`，由窗口关闭钩子负责回收。
fn launch_backend(child_slot: Arc<Mutex<Option<Child>>>) -> io::Result<()> {
    let dir = exe_dir()?;
    let python = pick_python(&dir);
    let backend_dir = dir.join("backend");

    let mut child = Command::new(&python)
        .arg("-m")
        .arg("uvicorn")
        .arg("backend.app.main:app")
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(BACKEND_PORT.to_string())
        .current_dir(&dir)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;

    // 记录句柄供退出时回收；不再 mem::forget（那会泄漏后端进程）。
    if let Ok(mut slot) = child_slot.lock() {
        *slot = Some(child);
    } else {
        // 锁已毒化（理论上不应发生）：放弃托管，按原语义让其随线程结束被回收前先杀掉，
        // 避免成为无人管理的孤儿进程。
        let _ = child.kill();
    }

    println!("[ScanDetection] backend launched via {:?}", python);

    // 等待端口就绪（前端首屏依赖后端）。
    wait_for_port("127.0.0.1", BACKEND_PORT, 60);
    let _ = backend_dir;
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

fn main() {
    let backend_state = BackendState {
        child: Arc::new(Mutex::new(None)),
    };
    // 供后台启动线程与窗口事件钩子各自持有一份 Arc 克隆（引用计数，零额外开销）。
    let launch_slot = backend_state.child.clone();
    let event_slot = backend_state.child.clone();

    tauri::Builder::default()
        .manage(backend_state)
        .setup(move |_app| {
            // 后台启动后端，并等待其就绪（窗口随后由 Tauri 创建并显示）。
            thread::spawn(move || {
                if let Err(e) = launch_backend(launch_slot) {
                    eprintln!("[ScanDetection] launch_backend error: {e}");
                }
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
