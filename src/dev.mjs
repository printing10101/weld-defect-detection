// Electron 开发启动器：同时拉起 vite dev server 与 Electron 壳，任一退出即全退。
// 用法：pnpm app:dev（等价 vite dev + electron，无需分别开两个终端）。
// 后端（uvicorn:18773）由 Electron 主进程在开发布局下自行拉起，无需在此管理。
"use strict";

import { spawn, spawnSync } from "node:child_process";
import http from "node:http";
import process from "node:process";

const VITE_URL = "http://127.0.0.1:5173";
const VITE_PORT = 5173;

let shuttingDown = false;

const vite = spawn("pnpm", ["dev"], {
  shell: process.platform === "win32", // Windows 下 pnpm 是 .cmd，需经 shell 解析
  stdio: ["ignore", "inherit", "inherit"],
});

// 等 vite 端口就绪再起 Electron：过早 loadURL 会拿到 ERR_CONNECTION_REFUSED。
// 就绪判定是端口可连即可（首个请求触发编译属 vite 正常行为）。
async function waitViteReady(timeoutMs = 30000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (vite.exitCode !== null) return false;
    if (await probePort()) return true;
    await new Promise((resolve) => setTimeout(resolve, 300));
  }
  return false;
}

function probePort() {
  return new Promise((resolve) => {
    const req = http.get({ host: "127.0.0.1", port: VITE_PORT, timeout: 1000 }, (res) => {
      res.resume();
      resolve(true);
    });
    req.on("timeout", () => {
      req.destroy();
      resolve(false);
    });
    req.on("error", () => resolve(false));
  });
}

const ready = await waitViteReady();
if (!ready) {
  console.error("[dev] vite dev server 未能在 30s 内就绪，退出");
  killTree(vite, "SIGTERM");
  process.exit(1);
}

const electron = spawn("pnpm", ["exec", "electron", "."], {
  shell: process.platform === "win32",
  stdio: ["ignore", "inherit", "inherit"],
  env: { ...process.env, VITE_DEV_SERVER_URL: VITE_URL },
});

// Windows 下 child.kill() 只能杀掉 shell:true 的 cmd 外壳，底下的
// node(vite)/electron 进程会残留（vite 残留即 5173 端口继续占用），
// 须用 taskkill /T 连进程树一起回收。
function killTree(child, signal) {
  if (child.killed || child.exitCode !== null) return;
  if (process.platform === "win32" && child.pid) {
    spawnSync("taskkill", ["/PID", String(child.pid), "/T", "/F"], {
      windowsHide: true,
      timeout: 5000,
    });
  } else {
    child.kill(signal);
  }
}

function shutdown(signal) {
  if (shuttingDown) return;
  shuttingDown = true;
  killTree(electron, signal);
  killTree(vite, signal);
  process.exit(0);
}

// 任一子进程先死则收掉另一个，避免残留半死进程。
electron.on("exit", (code) => {
  if (!shuttingDown) {
    shuttingDown = true;
    killTree(vite, "SIGTERM");
    process.exitCode = code ?? 0;
  }
});
vite.on("exit", (code) => {
  if (!shuttingDown) {
    shuttingDown = true;
    killTree(electron, "SIGTERM");
    process.exitCode = code ?? 0;
  }
});

process.on("SIGINT", () => shutdown("SIGINT"));
process.on("SIGTERM", () => shutdown("SIGTERM"));
