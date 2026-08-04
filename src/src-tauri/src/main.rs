// ScanDetection Tauri 入口（M1 骨架占位）。
// 构建需 Rust 工具链；Python sidecar（backend）在 M6 以 externalBin 注入。
fn main() {
    tauri::Builder::default()
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
