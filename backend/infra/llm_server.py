"""llama.cpp 本地大模型服务随软件启停（自拉起 / 连接已有服务两种模式）。

两种加载方式（``llm.mode``）
---------------------------
- ``managed``：由本软件拉起 ``tools/llama`` 下的 llama-server 加载 ``model_file``
  （dev 布局可用；模型权重 2.4GB 起，随 NSIS 分发会突破 2GB 打包上限）；
- ``external``：**只连接一个已运行的** OpenAI 兼容端点（用户自建的 llama.cpp
  router / Ollama / LM Studio），不重复占用显存、秒级就绪，退出时**不回收**该
  服务（归外部所有）。安装版分发走这条路径。

生命周期（managed 模式）
------------------------
- 启动：lifespan 装配期后台拉起（独立线程，不阻塞端口绑定与 registry 装配），
  就绪判定为 ``GET /health`` 返回 200（模型加载中 llama-server 返回 503）；
- 停止：应用退出时优雅回收（terminate → kill 兜底）。Windows 上辅以
  **Job Object（KILL_ON_JOB_CLOSE）**：壳（Tauri）对后端是硬杀（child.kill），
  Python 退出钩子不会执行，由 OS 保证"后端进程死 → llama-server 同死"，
  杜绝孤儿进程长期占用显存/端口；Linux（麒麟/UOS）用 PR_SET_PDEATHSIG 同语义；
- 降级：一律**不阻断主应用启动**（与 registry 装配同一失败哲学），状态经
  /health 的 ``llm`` 字段暴露；
- 复活：看门狗线程监控进程存活，意外退出按上限重启（max_restart），
  防止崩溃循环空转；启动前先探测端口——已有健康 llama-server（外部手工
  启动/极端竞态遗留）则收编复用，不重复拉起、退出时不回收非本进程拉起的实例。

状态机与语义（``status()["state"]``）
------------------------------------
``disabled``     配置关闭 / 启动前未启用。
``starting``     managed：进程已拉起、模型加载中。
``ready``        可用（自拉起成功、收编已有实例、或 external 端点可达）。
``unavailable``  **环境不具备**：托管运行时/权重缺失，或 external 端点未运行。
                 这不是故障，是"本机没有可用的模型运行时"这一预期事实；
                 ``advice`` 给出可执行的下一步（连接已有服务 / 指定模型目录）。
``auth_required`` external 端点活着但要求鉴权，且未配置 API Key 环境变量。
``error``        **尝试过但失败**：启动超时、进程提前退出、端点 URL 非法。
``stopped``      退出流程已回收。

区分 ``unavailable`` 与 ``error`` 是有意为之：此前把"安装包未随包分发模型运行时"
一律报成 ``error``，让正常的降级看起来像故障（用户据此认为"模型一直启动失败"）；
现在只有**真正尝试过并失败**的情况才叫 ``error``。

安全边界（单机纯离线产品的服务面收敛）
--------------------------------------
- host 配置**仅接受回环地址**（127.0.0.0/8、::1、localhost 字面量），
  external 端点同样仅接受回环——非回环一律拒绝拉起/探测，本模块只服务于
  "本机前后端"拓扑，不接受把托管端点指向远端；
- 健康探测先 getaddrinfo 解析并校验**全部解析结果**均为回环，再用解析出的
  IP 字面量建连（无二次解析 = DNS rebinding 失效）；采用 http.client 裸
  连接，协议/路径固定（http + /health），3xx 原样返回不跟随重定向；
- 端点鉴权 Key 只从**环境变量**读取（``llm.api_key_env`` 只是变量名），
  不落配置、不落盘、不写日志；
- 子进程以 **argv 列表 + shell=False** 拉起（不经 shell 拼接），
  二进制/模型路径仅来自受控配置文件与项目内置目录，非请求期用户输入。

诚实边界：stop() 只回收"本管理器拉起"的进程；收编实例（adopted，含 external
模式的端点）归外部所有，仅报告状态、不做生命周期干预。
"""

from __future__ import annotations

import http.client
import ipaddress
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from backend.infra.config import LlmCfg, resolve_config_path

_LOG = logging.getLogger("scandetection.llm")

_INSTALL_ROOT = Path(__file__).resolve().parents[2]

# 进程退出码轮询粒度 / 停止宽限期
_STOP_GRACE_SEC = 8.0
_LOG_MAX_BYTES = 10 * 1024 * 1024  # llama-server 日志超限即截断重开（进程输出，无审计价值）

# llama.cpp 会读取这些环境变量并**自动开启鉴权**；托管实例按设计无鉴权，
# 故拉起时显式剔除（详见 _child_env 的说明）。
_CHILD_ENV_STRIP = frozenset({"LLAMA_API_KEY", "LLAMA_SERVER_API_KEY"})


def is_loopback_host(host: str) -> bool:
    """仅允许回环目的：本模块的探测/绑定拓扑限定为"本机前后端"。"""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _resolve_loopback_ip(host: str) -> str | None:
    """把 host 解析为 IP 字面量并校验**全部解析结果**均为回环地址。

    仅接受回环是本模块的既定策略；"解析后再校验、用解析结果建连"同时封死
    DNS rebinding（后续连接不再二次解析）。解析失败或出现任一非回环结果 → None。
    """
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return None
    addrs = {str(info[4][0]) for info in infos}
    if not addrs:
        return None
    try:
        if all(ipaddress.ip_address(a).is_loopback for a in addrs):
            return addrs.pop()
    except ValueError:
        return None
    return None


def _probe_health(host: str, port: int, timeout: float = 1.5) -> bool:
    """探测 llama-server 就绪：/health 200 = 就绪（模型加载中为 503）。

    host 须为回环（字面量或 localhost，经解析校验）；http.client 裸连接
    不跟随重定向（3xx 原样返回，status!=200 即视为未就绪）。回环地址在
    egress_guard 中恒放行（C-16 代码级保证），本探测不构成外联。
    """
    if not is_loopback_host(host):
        return False
    ip = _resolve_loopback_ip(host)
    if ip is None:
        return False
    conn: http.client.HTTPConnection | None = None
    try:
        conn = http.client.HTTPConnection(ip, port, timeout=timeout)
        conn.request("GET", "/health")
        resp = conn.getresponse()
        return resp.status == 200
    except OSError:
        # 连接拒绝/超时 = 服务不在或未就绪
        return False
    finally:
        if conn is not None:
            conn.close()


def _probe_api_status(host: str, port: int, timeout: float = 1.5) -> int | None:
    """探测 OpenAI 兼容面 ``GET /v1/models`` 的状态码；连不上返回 None。

    为什么不能只看 ``/health``：llama-server 的 ``/health`` **免鉴权**，而
    ``/v1/*`` 在启用密钥时会 401。只探 /health 会把"进程活着但客户端用不了"
    误判为就绪——正是「状态显示 ready，实际一个请求都发不出去」那种假绿。
    """
    if not is_loopback_host(host):
        return None
    ip = _resolve_loopback_ip(host)
    if ip is None:
        return None
    conn: http.client.HTTPConnection | None = None
    try:
        conn = http.client.HTTPConnection(ip, port, timeout=timeout)
        conn.request("GET", "/v1/models")
        return conn.getresponse().status
    except OSError:
        return None
    finally:
        if conn is not None:
            conn.close()


def _child_env() -> dict[str, str]:
    """托管实例的子进程环境：剔除会让 llama-server **静默开启鉴权**的密钥变量。

    llama.cpp 会读取 ``LLAMA_API_KEY`` 环境变量并自动要求 Bearer 鉴权。本模块
    拉起的托管实例按设计是「无鉴权」（argv 里不含 ``--api-key``），且后端自身不
    会带 key 去访问它。若原样继承用户环境里那份变量，就会出现：进程正常加载模型、
    ``/health`` 返回 200、状态机报 ``ready``，但**所有** ``/v1/*`` 请求一律 401
    ——表现为"本地大模型启动了却完全用不了"，而且只在配置过该变量的机器上复现
    （典型的按机器环境分叉的隐形故障）。显式剔除后行为由代码决定、与环境解耦。
    """
    return {k: v for k, v in os.environ.items() if k not in _CHILD_ENV_STRIP}


def _default_server_exe() -> Path:
    """项目内置 llama.cpp 运行时（tools/llama/，不入库，随安装包分发）。"""
    exe = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    return _INSTALL_ROOT / "tools" / "llama" / exe


def parse_endpoint(endpoint: str) -> tuple[str, int]:
    """解析 external 端点 ``http://host:port`` → ``(host, port)``。

    仅接受 http + **回环** host：本模块不引入 TLS/远端语义，"把托管端点指向
    远端"是被明确排除的拓扑。非法输入抛 ``ValueError``（由调用方转状态）。
    """
    from urllib.parse import urlsplit

    parts = urlsplit((endpoint or "").strip())
    if parts.scheme != "http":
        raise ValueError(f"端点仅支持 http://，收到: {endpoint!r}")
    host = parts.hostname
    if not host:
        raise ValueError(f"端点缺少主机名: {endpoint!r}")
    if not is_loopback_host(host):
        raise ValueError(f"端点仅允许回环地址，收到: {host}")
    try:
        port = int(parts.port or 80)
    except ValueError as exc:
        raise ValueError(f"端点端口非法: {endpoint!r}") from exc
    return host, port


def _assign_job_object(proc: subprocess.Popen) -> object | None:
    """Windows：把子进程绑入 Job Object（KILL_ON_JOB_CLOSE）。

    返回 job 句柄（保持打开直到 stop()；后端进程无论以何种方式退出，
    OS 关闭句柄即回收 llama-server）。失败返回 None（降级为仅看门狗兜底）。
    """
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                (name, ctypes.c_uint64)
                for name in (
                    "ReadOperationCount",
                    "WriteOperationCount",
                    "OtherOperationCount",
                    "ReadTransferCount",
                    "WriteTransferCount",
                    "OtherTransferCount",
                )
            ]

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        JobObjectExtendedLimitInformation = 9
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            raise OSError("CreateJobObjectW failed")
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            job, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info)
        ):
            raise OSError("SetInformationJobObject failed")
        # Popen._handle 是 CPython 私有属性，无公开等价物（故豁免类型检查）
        handle = int(proc._handle)  # pyright: ignore[reportAttributeAccessIssue]
        if not kernel32.AssignProcessToJobObject(job, handle):
            raise OSError("AssignProcessToJobObject failed")
        return job
    except Exception as exc:  # noqa: BLE001 - 兜底武装失败不阻断启动
        _LOG.warning("llama-server Job Object 绑定失败（孤儿兜底降级为看门狗）: %s", exc)
        return None


def _set_parent_death_signal() -> None:  # pragma: no cover - POSIX 专属，Windows 不执行
    """Linux（麒麟/UOS）：父进程死亡即内核直发 SIGKILL（PR_SET_PDEATHSIG）。"""
    if sys.platform == "win32":
        return
    try:
        import ctypes
        import signal

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("PR_SET_PDEATHSIG 设置失败（孤儿兜底降级为看门狗）: %s", exc)


class LlamaServerManager:
    """llama-server 进程生命周期管理（managed 自拉起 / external 连接已有服务）。

    状态机：disabled（配置关）→ starting → ready / error / unavailable / auth_required。
    ``status()`` 输出经 /health 暴露，前端可据此展示"本地大模型"运行状态。

    ``ready`` 的判据是 ``/health`` 与 ``/v1/models`` **双双**可用，不是只看前者：
    llama-server 的 /health 免鉴权，单看它会把"要密钥、客户端根本用不了"的实例
    报成就绪（假绿）。详见 ``_probe_api_status``。
    """

    def __init__(self, cfg: LlmCfg, log_dir: str | Path) -> None:
        if not is_loopback_host(cfg.host):
            raise ValueError(
                f"llm.host 仅允许回环地址（127.0.0.0/8、::1、localhost），收到: {cfg.host}"
            )
        self._cfg = cfg
        self._log_dir = Path(log_dir)
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._job_handle: object | None = None
        self._adopted = False
        # enabled 但尚未 start 的真实语义是"启动在途"，不是"被关掉"。
        self._state = "starting" if cfg.enabled else "disabled"
        self._error: str | None = None
        self._advice: str | None = None
        # external 端点解析（非回环/非法 URL 直接拒绝：这是安全边界，非运行期可选项）
        self._external_host = cfg.host
        self._external_port = cfg.port
        if cfg.mode == "external":
            self._external_host, self._external_port = parse_endpoint(cfg.external_endpoint)
            self._endpoint = cfg.external_endpoint.rstrip("/")
        else:
            self._endpoint = f"http://{cfg.host}:{cfg.port}"
        self._stop_requested = threading.Event()
        self._watch_thread: threading.Thread | None = None
        self._start_thread: threading.Thread | None = None

    # ---------- 状态 ----------

    def status(self) -> dict:
        with self._lock:
            state = self._state
            info: dict = {
                "enabled": self._cfg.enabled,
                "mode": self._cfg.mode,
                "state": state,
                "endpoint": self._endpoint,
                "adopted": self._adopted,
                "error": self._error,
                "advice": self._advice,
            }
            if self._proc is not None and self._proc.poll() is not None:
                info["exit_code"] = self._proc.returncode
                if state == "ready":
                    # 进程已死、看门狗尚未巡检到：如实降格为"重启在途"，
                    # 避免 /health 在最长一个巡检周期内虚报 ready。
                    info["state"] = "starting"
            return info

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def _set_state(self, state: str, error: str | None = None, advice: str | None = None) -> None:
        with self._lock:
            self._state = state
            self._error = error
            self._advice = advice
        if error:
            _LOG.warning("llama-server 状态=%s：%s", state, error)
        else:
            _LOG.info("llama-server 状态=%s", state)

    # ---------- 启动 ----------

    def start_async(self) -> None:
        """后台线程拉起（不阻塞调用方：模型加载可达数十秒）。"""
        if not self._cfg.enabled:
            self._set_state("disabled")
            return
        # 先同步置 starting 再开线程：否则从构造到工作线程首次写状态之间的窗口里，
        # status() 会返回 __init__ 的初值，让"正在启动"被误读成"被配置关掉了"。
        self._set_state("starting")
        t = threading.Thread(target=self.start, name="llm-server-start", daemon=True)
        self._start_thread = t
        t.start()

    def start(self) -> None:
        """就绪判定（阻塞；正常经 start_async 调用）。按 ``cfg.mode`` 分派。"""
        cfg = self._cfg
        if not cfg.enabled:
            self._set_state("disabled")
            return
        if cfg.mode == "external":
            self._start_external()
            return
        self._start_managed()

    # ---- external：连接已有服务（不重复拉起、不占额外显存）----

    def _start_external(self) -> None:
        """探测已运行端点并收编；端点未运行时报 ``unavailable``（非故障）。"""
        self._set_state("starting")
        probe = self._probe_external()
        self._apply_external_probe(probe)
        with self._lock:
            ready = self._state == "ready"
        if ready:
            self._ensure_watchdog()

    def _probe_external(self):
        """探测 external 端点（局部导入避免与 llm_discovery 成环）。"""
        from backend.infra.llm_discovery import probe_service

        return probe_service(
            self._external_host,
            self._external_port,
            timeout=self._cfg.probe_timeout_sec,
            api_key_env=self._cfg.api_key_env,
        )

    def _apply_external_probe(self, probe) -> None:
        """按探测结果落状态（启动与看门狗巡检共用，保证两条路径语义一致）。"""
        if probe.reachable and not probe.needs_auth:
            with self._lock:
                self._adopted = True
            self._set_state("ready")
            _LOG.info(
                "external 模式：已连接已有服务 %s（模型 %d 个，退出时不回收）",
                self._endpoint,
                len(probe.models),
            )
            return
        if probe.needs_auth:
            with self._lock:
                self._adopted = True
            self._set_state(
                "auth_required",
                f"端点 {self._endpoint} 要求鉴权，但未取到 API Key",
                advice=(
                    f"设置环境变量 {self._cfg.api_key_env or '<API_KEY_ENV>'} 后重启应用；"
                    "Key 只从环境变量读取，不落配置、不写日志。"
                ),
            )
            return
        self._set_state(
            "unavailable",
            f"external 端点未运行: {self._endpoint}",
            advice=(
                "本软件不随包分发模型运行时（权重 ≥2.4GB，超出安装包体积上限）。"
                f"请先启动本机 llama 服务（如 {self._endpoint}），"
                "或在「本地大模型」面板中选择本机已有的 GGUF 模型 / 添加模型目录。"
            ),
        )

    # ---- managed：自拉起 llama-server ----

    def _start_managed(self) -> None:
        cfg = self._cfg
        if not is_loopback_host(cfg.host):
            self._set_state("error", f"llm.host 仅允许回环地址，收到: {cfg.host}")
            return

        exe = Path(cfg.server_exe) if cfg.server_exe else _default_server_exe()
        if not exe.is_file():
            # 环境不具备（安装版未随包分发 llama 运行时）→ unavailable，不是故障
            self._set_state(
                "unavailable",
                f"llama-server 可执行文件不存在: {exe}",
                advice="安装包未包含本地模型运行时；可改用 external 模式连接已有服务。",
            )
            return
        try:
            model = resolve_config_path(cfg.model_file)
        except Exception as exc:  # noqa: BLE001
            self._set_state("error", f"模型路径解析失败: {exc}")
            return
        if not model.is_file():
            self._set_state(
                "unavailable",
                f"模型文件不存在: {model}",
                advice="权重未随包分发；可在「本地大模型」面板中选择本机已有模型。",
            )
            return

        # 端口已有实例（外部手工启动 / 上轮进程遗留）→ 收编复用。
        # 收编条件必须含 /v1/models 可用：只探 /health 会收编一个"要密钥"的实例，
        # 于是状态报 ready、调用全 401。
        if _probe_health(cfg.host, cfg.port) and _probe_api_status(cfg.host, cfg.port) == 200:
            with self._lock:
                self._adopted = True
            self._set_state("ready")
            _LOG.warning(
                "llama-server 端口 %d 已有健康实例，收编复用（非本进程拉起，退出时不回收）",
                cfg.port,
            )
            self._ensure_watchdog()
            return

        self._set_state("starting")
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            log_path = self._log_dir / "llama-server.log"
            self._rotate_log_if_large(log_path)
            # append 模式句柄交子进程继承；父进程写完即关，不长期占用
            with log_path.open("a", encoding="utf-8", errors="replace") as log_fh:
                # argv 列表 + shell=False：不经 shell 解释；路径来自受控配置
                cmd = self._build_command(exe, model)
                log_fh.write(
                    f"\n[{time.strftime('%Y-%m-%dT%H:%M:%S')}] $ {subprocess.list2cmdline(cmd)}\n"
                )
                log_fh.flush()
                if sys.platform == "win32":
                    proc = subprocess.Popen(
                        cmd,
                        stdout=log_fh,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                        shell=False,
                        env=_child_env(),
                    )
                else:
                    # PLW1509：PDEATHSIG 必须在 exec 前由子进程自身注册，
                    # preexec_fn 是唯一入口；fork→exec 间隙极短，可接受。
                    proc = subprocess.Popen(
                        cmd,
                        stdout=log_fh,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        preexec_fn=_set_parent_death_signal,  # noqa: PLW1509
                        shell=False,
                        env=_child_env(),
                    )
            with self._lock:
                self._proc = proc
                self._adopted = False
            self._job_handle = _assign_job_object(proc)
            _LOG.info(
                "llama-server 已拉起 (pid=%s, model=%s, endpoint=%s)",
                proc.pid,
                model.name,
                self._endpoint,
            )
        except Exception as exc:  # noqa: BLE001 - 拉起失败降级，不阻断主应用
            self._set_state("error", f"llama-server 拉起失败: {exc}")
            return

        if self._await_ready():
            self._set_state("ready")
            self._ensure_watchdog()
            return
        self._terminate_process()
        if self._stop_requested.is_set():
            return  # 软件退出流程已接管：状态由 stop() 置 stopped
        if self._state not in ("error", "auth_required"):
            # 进程提前退出/鉴权异常的分支已在 _await_ready 内记了精确原因，
            # 这里只兜底"真的只是超时"的情况，不覆盖更具体的信息。
            self._set_state(
                "error",
                f"llama-server 启动超时（>{cfg.startup_timeout_sec:.0f}s 未就绪），已回收；"
                f"详见 {self._log_dir / 'llama-server.log'}",
            )

    def _build_command(self, exe: Path, model: Path) -> list[str]:
        cfg = self._cfg
        cmd = [
            str(exe),
            "-m",
            str(model),
            "--host",
            cfg.host,
            "--port",
            str(cfg.port),
            "-c",
            str(cfg.n_ctx),
            "-ngl",
            str(cfg.n_gpu_layers),
            # 关闭内置 Web UI：单机交付面收敛，仅保留 OpenAI 兼容 API
            "--no-webui",
        ]
        if cfg.threads > 0:
            cmd += ["-t", str(cfg.threads)]
        return cmd

    def _await_ready(self) -> bool:
        deadline = time.monotonic() + self._cfg.startup_timeout_sec
        while time.monotonic() < deadline:
            if self._stop_requested.is_set():
                return False
            proc = self._proc
            if proc is not None and proc.poll() is not None:
                self._set_state(
                    "error",
                    f"llama-server 进程提前退出（exit={proc.returncode}），"
                    f"详见 {self._log_dir / 'llama-server.log'}",
                )
                return False
            if _probe_health(self._cfg.host, self._cfg.port):
                code = _probe_api_status(self._cfg.host, self._cfg.port)
                if code is None or code == 200:
                    return True
                if code in (401, 403):
                    # /health 免鉴权而 /v1/* 要 key：进程"活着"但客户端用不了。
                    # 不报 ready，否则就是典型假绿；也不静默重试（重试无用）。
                    self._set_state(
                        "auth_required",
                        f"llama-server 已加载模型，但 /v1/models 返回 {code}（要求鉴权）",
                        advice=(
                            "这是 llama.cpp 从环境变量 LLAMA_API_KEY 自动开启鉴权所致，"
                            "本模块已显式剔除该变量；若仍复现，请检查是否有包装脚本重新注入。"
                        ),
                    )
                    return False
            time.sleep(0.5)
        return False

    # ---------- 看门狗 ----------

    def _ensure_watchdog(self) -> None:
        if self._watch_thread is not None and self._watch_thread.is_alive():
            return
        # stop() 已请求后不得重新武装看门狗（原无条件 clear 会撤销已发出的
        # stop：退出竞态窗口内收编/就绪路径恰通过时，已停的管理器被拉回
        # 运行态，状态与实际进程脱节）。Event 在 __init__ 即为清零态，无需
        # 此处重置。
        if self._stop_requested.is_set():
            return
        t = threading.Thread(target=self._watch_loop, name="llm-server-watch", daemon=True)
        self._watch_thread = t
        t.start()

    def _watch_loop(self) -> None:
        """managed：意外退出按上限重启；external：巡检端点存活。stop() 请求即退出。"""
        restarts = 0
        while not self._stop_requested.wait(timeout=self._cfg.watch_interval_sec):
            if self._cfg.mode == "external":
                # 外部端点进出状态都可能变化（用户手动起停服务），持续巡检并如实
                # 反映——不复用 managed 的"进程死即重启"语义，因为不归我们所有。
                self._apply_external_probe(self._probe_external())
                continue
            with self._lock:
                proc = self._proc
                adopted = self._adopted
                state = self._state
            if adopted or state != "ready":
                continue
            if proc is None or proc.poll() is None:
                continue
            exit_code = proc.returncode
            if restarts >= self._cfg.max_restart:
                self._set_state(
                    "error",
                    f"llama-server 意外退出（exit={exit_code}）且已达重启上限 ({self._cfg.max_restart})",
                )
                return
            restarts += 1
            _LOG.warning("llama-server 意外退出（exit=%s），第 %d 次重启", exit_code, restarts)
            with self._lock:
                self._proc = None
            self.start()
            with self._lock:
                if self._state == "ready":
                    restarts = 0  # 恢复健康则重置计数

    # ---------- 停止 ----------

    def stop(self, timeout: float = _STOP_GRACE_SEC) -> None:
        """随软件退出回收本管理器拉起的 llama-server（幂等）。"""
        self._stop_requested.set()
        with self._lock:
            adopted = self._adopted
        if adopted:
            _LOG.info("llama-server 为收编实例，退出时不回收（归外部所有）")
            self._set_state("stopped")
            return
        self._terminate_process(timeout)
        self._set_state("stopped")

    def _terminate_process(self, timeout: float = _STOP_GRACE_SEC) -> None:
        with self._lock:
            proc = self._proc
            self._proc = None
        if proc is None or proc.poll() is not None:
            self._close_job_handle()
            return
        try:
            proc.terminate()
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                _LOG.warning("llama-server 未在 %.0fs 内退出，强制 kill", timeout)
                proc.kill()
                proc.wait(timeout=5)
        except Exception as exc:  # noqa: BLE001 - 回收失败不掩盖其它退出逻辑
            _LOG.warning("llama-server 回收异常: %s", exc)
        finally:
            self._close_job_handle()

    def _close_job_handle(self) -> None:
        # KILL_ON_JOB_CLOSE：句柄关闭即回收残留进程（正常路径进程已退出，幂等）
        if self._job_handle is not None:
            try:
                import ctypes

                ctypes.windll.kernel32.CloseHandle(self._job_handle)
            except Exception as exc:  # noqa: BLE001 - 句柄清理失败不影响退出语义
                _LOG.debug("Job 句柄关闭失败（进程回收不受影响）: %s", exc)
            self._job_handle = None

    @staticmethod
    def _rotate_log_if_large(log_path: Path) -> None:
        try:
            if log_path.is_file() and log_path.stat().st_size > _LOG_MAX_BYTES:
                rotated = log_path.with_suffix(".log.1")
                log_path.replace(rotated)
        except OSError as exc:
            # 尽力而为操作，但失败留痕：否则日志无限增长且无人知晓
            _LOG.warning("llama-server 日志轮转失败 %s: %s", log_path, exc)
