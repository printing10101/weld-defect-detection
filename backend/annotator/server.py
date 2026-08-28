"""
轻量焊缝缺陷标注器（零依赖，纯标准库）

主动学习闭环的一部分：
  - 用户在浏览器画框标注 data/real_label/images 下真实 X 光图
  - 标注以 YOLO txt 落盘到 data/real_label/labels/{stem}.txt
  - need_review 标记存 sidecar data/real_label/labels/{stem}.review（"1"/"0"）
  - 支持预标注：若存在 data/real_label/prelabels/{stem}.txt，打开即作为可编辑起始框

运行：python -m backend.annotator.server  →  http://localhost:8899
不需要任何第三方包，离线可用，避开沙箱 safe-delete 对 pip 的限制。
"""

import json
import os
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# 安装根目录锚点：backend/annotator/server.py -> parents[2] = 安装根目录
# （§部署硬化 #6：原 Path("data/real_label") 相对 CWD，Tauri/安装包场景 CWD≠项目根会失效）
ROOT = Path(__file__).resolve().parents[2] / "data" / "real_label"
IMG = ROOT / "images"
LBL = ROOT / "labels"
PRE = ROOT / "prelabels"
MANIFEST = ROOT / "manifest.json"
LBL.mkdir(parents=True, exist_ok=True)
PRE.mkdir(parents=True, exist_ok=True)


def _safe_image_path(name: str) -> Path | None:
    """把 URL 中的影像名解析到 IMG 目录内，越界即返回 None（防路径穿越）。

    原实现直接 `IMG / name` 后仅用 .exists() 判断，未校验解析后是否仍落在 IMG
    之内：name 形如 `../../etc/passwd` 会越界读任意文件。这里与主 API 的
    backend.infra.fs.safe_resolve 保持同语义——空名/'.'/'..' 拒绝，解析结果
    必须仍相对 IMG，否则返回 None（调用方按 404 处理）。
    """
    if not name or name in (".", ".."):
        return None
    try:
        candidate = (IMG / name).resolve()
    except (OSError, ValueError):
        return None
    if not candidate.is_relative_to(IMG.resolve()):
        return None
    return candidate if candidate.is_file() else None


# 7 类，与 class_map / DefectClass 一致（2026-08 内凹拆分为独立类，索引 6）
CLASS_NAMES = ["气孔", "夹渣", "未焊透", "未熔合", "裂纹", "咬边", "内凹"]

HERE = Path(__file__).resolve().parent
INDEX_HTML = HERE / "index.html"


def _load_manifest():
    """读取 data/real_label/manifest.json（去重 + 跨试板族分层抽样的种子集）。
    若不存在则退回：全量 glob 且前 25 张为种子（兼容旧行为）。"""
    if not MANIFEST.exists():
        files = sorted(p.name for p in IMG.glob("*.jpg"))
        return [{"name": n, "family": "?", "is_seed": i < 25} for i, n in enumerate(files)]
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return data["images"]


def _list_images():
    manifest = _load_manifest()
    out = []
    for i, item in enumerate(manifest):
        name = item["name"]
        stem = Path(name).stem
        lbl = LBL / (stem + ".txt")
        rev = LBL / (stem + ".review")
        pre = PRE / (stem + ".txt")
        out.append(
            {
                "name": name,
                "index": i,
                "family": item.get("family", "?"),
                "labeled": lbl.exists() and lbl.read_text(encoding="utf-8").strip() != "",
                "has_prelabel": pre.exists(),
                "need_review": rev.exists() and rev.read_text(encoding="utf-8").strip() == "1",
                "is_seed": bool(item.get("is_seed", False)),
            }
        )
    return out


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, ctype):
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        p = u.path
        if p in ("/", "/index.html"):
            if INDEX_HTML.exists():
                self._send_file(INDEX_HTML, "text/html; charset=utf-8")
            else:
                self._send(500, "index.html 缺失")
            return
        if p == "/api/images":
            self._send(200, json.dumps(_list_images(), ensure_ascii=False))
            return
        if p.startswith("/img/"):
            name = urllib.parse.unquote(p[len("/img/") :])
            fp = _safe_image_path(name)
            if fp is not None:
                self._send_file(fp, "image/jpeg")
            else:
                self._send(404, "not found")
            return
        if p.startswith("/api/label/"):
            name = urllib.parse.unquote(p[len("/api/label/") :])
            stem = Path(name).stem
            lbl = LBL / (stem + ".txt")
            pre = PRE / (stem + ".txt")
            rev = LBL / (stem + ".review")
            # 优先返回已保存标注；否则返回预标注（若有）
            if lbl.exists():
                content = lbl.read_text(encoding="utf-8")
            elif pre.exists():
                content = pre.read_text(encoding="utf-8")
            else:
                content = ""
            payload = {
                "label": content,
                "need_review": rev.exists() and rev.read_text(encoding="utf-8").strip() == "1",
                "from_prelabel": (not lbl.exists()) and pre.exists(),
            }
            self._send(200, json.dumps(payload, ensure_ascii=False))
            return
        self._send(404, "not found")

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        p = u.path
        if p.startswith("/api/label/"):
            name = urllib.parse.unquote(p[len("/api/label/") :])
            stem = Path(name).stem
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length).decode("utf-8")
            try:
                obj = json.loads(raw)
            except Exception:  # noqa: BLE001 - 任意解析失败一律回 400
                self._send(400, json.dumps({"ok": False, "err": "bad json"}))
                return
            label_txt = obj.get("label", "")
            need_review = bool(obj.get("need_review", False))
            (LBL / (stem + ".txt")).write_text(label_txt, encoding="utf-8")
            (LBL / (stem + ".review")).write_text("1" if need_review else "0", encoding="utf-8")
            self._send(200, json.dumps({"ok": True}))
            return
        self._send(404, "not found")

    def log_message(self, *args):
        pass  # 静默


def main():
    port = int(os.environ.get("ANNO_PORT", "8899"))
    host = os.environ.get("ANNO_HOST", "127.0.0.1")
    # 仅监听本机回环：该标注服务无任何鉴权，绑定 0.0.0.0 会让同网段任意主机
    # 既能读取标注目录、又能利用（已修复的）路径穿越之外的其它隐患访问本机文件。
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"[annotator] 服务已启动: http://{host}:{port}  (Ctrl+C 停止)")
    print(f"[annotator] 图像目录: {IMG}  标注目录: {LBL}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
