// 渲染层桥（contextIsolation 下唯一的受控通道）。
// 只暴露窗口关闭守卫所需的最小面：长任务上报、关闭请求回调、确认退出。
"use strict";

const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("desktop", {
  /** 渲染层长任务（批量/单幅评定）状态上报：主进程据此决定是否拦截点 × 关闭。 */
  setBusy: (busy) => ipcRenderer.send("desktop:set-busy", Boolean(busy)),

  /** 订阅"点 × 被主进程拦截"事件（渲染层弹自建确认框）。返回取消订阅函数。 */
  onCloseRequested: (callback) => {
    const handler = () => callback();
    ipcRenderer.on("desktop:close-requested", handler);
    return () => ipcRenderer.removeListener("desktop:close-requested", handler);
  },

  /** 用户确认放弃任务并关闭窗口：主进程置放行标志后真正执行 close。 */
  confirmQuit: () => ipcRenderer.send("desktop:confirm-quit"),
});
