#!/usr/bin/env python3
"""Python 包裝器: 一鍵啟動即時中文轉譯或 HTTP/WS 伺服器。

為什麼需要 app bundle?
    macOS TCC 要求語音辨識程式必須是 .app bundle（內含用途說明），
    直接 python 執行會被系統 SIGABRT。本包裝器會:
      1. 自動建立 MacSTT.app（最小 bundle，3 個檔案，可随时手動刪除，
         下次執行會自動重建）
      2. open MacSTT.app 啟動 live_zh.py 或 uvicorn server
      3. 把後端輸出即時 tail 到終端
      4. 把終端的 Ctrl+C 轉發給後端，乾淨收尾

用法（live 模式，預設）:
    uv run python start.py                 # 預設 zh-TW，引擎自斷句（連續收音）
    uv run python start.py --lang zh-CN    # 簡體中文
    uv run python start.py --no-engine     # 改回舊版「靜音 --pause 秒斷句」
    uv run python start.py --pause 2.0     # 靜音 2 秒才斷句（--no-engine 時）
    uv run python start.py --device 1      # 指定輸入裝置
    uv run python start.py --cloud         # 允許雲端辨識（中英混合較穩，需網路）
    uv run python start.py --hints 詞表.txt # 自訂提示詞（VS Code 等專有名詞）
    uv run python start.py --no-hints      # 停用內建提示詞

用法（serve 模式，HTTP/WS 伺服器，供各專案串流 STT 接入）:
    uv run python start.py serve                        # 127.0.0.1:8765, zh-TW
    uv run python start.py serve --port 9000            # 換 port
    uv run python start.py serve --vocab 詞表.txt        # 自訂詞庫（取代內建）
    uv run python start.py serve --max-sessions 3       # 最大並發串流 session
    uv run python start.py serve --allow-cloud          # 允許 on_device=false
    # 端點: REST /transcribe /transcribe/raw /vocabulary /sessions /health
    #        /languages；串流 WS /stream（PCM 16k mono int16 推上來）
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time

PROJECT = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(PROJECT, "MacSTT.app")
APP_MACOS = os.path.join(APP, "Contents", "MacOS")
APP_EXEC = os.path.join(APP_MACOS, "macstt")
INFO_PLIST = os.path.join(APP, "Contents", "Info.plist")
CFG = "/tmp/macstt_args.json"
LOG = "/tmp/macstt_live.log"
PIDF = "/tmp/macstt_pid"
END_MARKER = "👋 已停止"
SERVE_LOG = "/tmp/macstt_server.log"
SERVE_READY_MARKER = "Application startup complete."

INFO_PLIST_XML = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
\t<key>CFBundleName</key>
\t<string>MacSTT</string>
\t<key>CFBundleDisplayName</key>
\t<string>MacSTT 即時轉譯</string>
\t<key>CFBundleIdentifier</key>
\t<string>local.birc.macstt</string>
\t<key>CFBundleVersion</key>
\t<string>1.0</string>
\t<key>CFBundleShortVersionString</key>
\t<string>1.0</string>
\t<key>CFBundlePackageType</key>
\t<string>APPL</string>
\t<key>CFBundleExecutable</key>
\t<string>macstt</string>
\t<key>NSSpeechRecognitionUsageDescription</key>
\t<string>本程式使用系統內建語音辨識將您的語音轉為文字，處理全程在本機完成，不會上傳。</string>
\t<key>NSMicrophoneUsageDescription</key>
\t<string>本程式需要麥克風訊號以進行即時語音轉文字。</string>
</dict>
</plist>
"""

# app bundle 內的執行檔（Python 腳本，shebang 指向 venv）
# 依 /tmp/macstt_args.json 的 "mode" 分流: "live"（預設）或 "serve"
APP_EXEC_SRC = '''{shebang}
"""MacSTT app launcher — 由 start.py 自動建立並透過 open 啟動。"""
import json
import os
import runpy
import sys

PROJECT = {project!r}
with open("/tmp/macstt_pid", "w") as _f:
    _f.write(str(os.getpid()))
try:
    a = json.load(open("/tmp/macstt_args.json"))
except Exception:
    a = {{"lang": "zh-TW", "pause": 1.5, "device": None, "mode": "live"}}

os.chdir(PROJECT)
sys.path.insert(0, os.path.join(PROJECT, "src"))

if a.get("mode") == "serve":
    # open .app 啟動的 stdout 不會到終端 → uvicorn 輸出導向 log 檔
    # （start.py 會 tail 這個檔顯示「Application startup complete.」等訊息）
    _log = open("/tmp/macstt_server.log", "w", buffering=1)
    sys.stdout = _log
    sys.stderr = _log
    from macos_stt import engine as _engine
    try:
        # 首次授權在 main thread + 空閒 RunLoop 上完成（TCC 對話框需要）；
        # 之後 uvicorn 佔住 main thread，請求路徑只需讀快取的授權狀態
        _engine._ensure_authorized()
    except Exception as _e:
        print("授權檢查失敗: " + repr(_e), flush=True)
    import uvicorn
    from macos_stt.server import create_app
    from macos_stt.vocab import VocabularyStore
    store = VocabularyStore.load_file(a["vocab"]) if a.get("vocab") else VocabularyStore.load_default()
    app = create_app(
        default_lang=a.get("lang", "zh-TW"),
        vocab=store,
        max_sessions=int(a.get("max_sessions", 3)),
        allow_cloud=bool(a.get("allow_cloud", False)),
    )
    uvicorn.run(app, host="127.0.0.1", port=int(a.get("port", 8765)), log_level="info")
    sys.exit(0)

sys.argv = [
    "live_zh.py",
    "--lang", str(a.get("lang", "zh-TW")),
    "--pause", str(a.get("pause", 1.5)),
    "--log", "/tmp/macstt_live.log",
]
if a.get("device") is not None:
    sys.argv += ["--device", str(a["device"])]
if a.get("no_hints"):
    sys.argv += ["--no-hints"]
else:
    sys.argv += ["--hints", str(a.get("hints") or "")]
if a.get("cloud"):
    sys.argv += ["--cloud"]
if a.get("no_engine"):
    sys.argv += ["--no-engine"]

runpy.run_path(os.path.join(PROJECT, "live_zh.py"), run_name="__main__")
'''


def ensure_app() -> None:
    """建立或更新最小 app bundle（每次執行都重寫，保持同步）。"""
    venv_python = os.path.join(PROJECT, ".venv", "bin", "python")
    os.makedirs(APP_MACOS, exist_ok=True)
    with open(INFO_PLIST, "w") as f:
        f.write(INFO_PLIST_XML)
    with open(APP_EXEC, "w") as f:
        f.write(APP_EXEC_SRC.format(
            shebang=f"#!{venv_python}\n", project=PROJECT))
    os.chmod(APP_EXEC, 0o755)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, TypeError, ValueError):
        return False


def find_child() -> int | None:
    """讀 PID 檔拿後端 PID。"""
    try:
        return int(open(PIDF).read().strip())
    except (OSError, ValueError):
        return None


def _launch(cfg: dict, log_path: str, ready_marker: bytes, note: str,
            exit_on_marker: bool) -> None:
    """通用啟動流程：清舊程序 → 寫 cfg → open app → 等後端 → tail log。

    exit_on_marker=True（live）：ready_marker（「👋 已停止」）出現即 break；
    False（serve）：ready_marker（uvicorn startup 訊息）出現不 break，
    伺服器常駐直到後端程序結束。
    """
    # 清掉舊程序（open 不會重啟已執行的 app）
    old = find_child()
    if old and _alive(old):
        try:
            os.kill(old, signal.SIGTERM)
        except (ProcessLookupError, TypeError):
            pass
    subprocess.run(["pkill", "-f", "MacSTT.app/Contents/MacOS/macstt"],
                   capture_output=True)
    time.sleep(0.5)
    try:
        os.unlink(PIDF)
    except FileNotFoundError:
        pass

    with open(CFG, "w") as f:
        json.dump(cfg, f)
    try:
        os.unlink(log_path)
    except FileNotFoundError:
        pass

    subprocess.run(["open", APP])
    print(note, flush=True)

    # 等後端活起來
    pid = None
    deadline = time.time() + 20
    while time.time() < deadline:
        pid = find_child()
        if pid and _alive(pid):
            break
        time.sleep(0.3)
    if not pid or not _alive(pid):
        print("錯誤: 後端未啟動。最近 crash 記錄在 ~/Library/Logs/DiagnosticReports/")
        sys.exit(1)

    def _forward(*_a):
        try:
            os.kill(pid, signal.SIGINT)
        except (ProcessLookupError, TypeError):
            pass

    signal.signal(signal.SIGINT, _forward)
    print("（Ctrl+C 結束）\n", flush=True)

    # 等後端建立 log 檔再 tail（PID 檔先寫，log 晚一點）
    deadline = time.time() + 20
    while not os.path.exists(log_path) and time.time() < deadline:
        time.sleep(0.1)
    if not os.path.exists(log_path):
        print("錯誤: 後端未產生輸出（可能 crash），檢查 ~/Library/Logs/DiagnosticReports/")
        sys.exit(1)

    # tail 輸出到終端
    with open(log_path, "rb") as f:
        gone = 0
        while True:
            chunk = f.read(4096)
            if chunk:
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                if exit_on_marker and ready_marker in chunk:
                    break
                gone = 0
            elif _alive(pid):
                gone = 0
                time.sleep(0.1)
            else:
                gone += 1
                if gone >= 10:  # 後端消失 ~1 秒，確認結束
                    break
                time.sleep(0.1)

    # 收尾
    if _alive(pid):
        try:
            os.kill(pid, signal.SIGINT)
        except (ProcessLookupError, TypeError):
            pass
    try:
        os.unlink(log_path)
    except FileNotFoundError:
        pass


def main():
    p = argparse.ArgumentParser(description="即時中文轉譯 / HTTP-WS 伺服器（終端即時顯示）")
    sub = p.add_subparsers(dest="mode")

    # --- serve 子命令 ---
    ps = sub.add_parser("serve", help="啟動 HTTP/WS 識別伺服器（127.0.0.1）")
    ps.add_argument("--port", type=int, default=8765, help="port（預設 8765）")
    ps.add_argument("--lang", default="zh-TW",
                    help="全域預設語言（zh-TW/zh-CN/yue-CN/...）")
    ps.add_argument("--vocab", default=None,
                    help="詞表檔路徑（每行/逗號分隔）；不指定用內建 vocab.default.txt")
    ps.add_argument("--max-sessions", type=int, default=3,
                    help="最大同時串流 session（預設 3）")
    ps.add_argument("--allow-cloud", action="store_true",
                    help="允許 on_device=false（雲端辨識，需網路）")

    # --- live 子命令 ---
    pl = sub.add_parser("live", help="麥克風即時轉譯（預設）")
    pl.add_argument("--lang", default="zh-TW",
                    help="zh-TW(繁中) / zh-CN(簡中) / yue-CN(粵語)")
    pl.add_argument("--pause", type=float, default=1.5,
                    help="靜音幾秒斷句（預設 1.5；中英混合說術語時建議 1.5-2.0）")
    pl.add_argument("--hints", default=None,
                    help="提示詞: 檔案路徑（每行/逗號分隔）或逗號分隔清單")
    pl.add_argument("--no-hints", action="store_true",
                    help="停用提示詞（contextualStrings）")
    pl.add_argument("--cloud", action="store_true",
                    help="允許雲端辨識（預設強制離線）；中英混合辨識較穩，需網路")
    pl.add_argument("--no-engine", action="store_true",
                    help="停用引擎自斷句，改回舊版「靜音 --pause 秒斷句」")
    pl.add_argument("--device", type=int, default=None,
                    help="輸入裝置編號（預設系統預設麥克風）")

    # 相容舊用法: python start.py --lang zh-CN（無子命令 → live）
    p.add_argument("--lang", dest="t_lang", default=None,
                    help=argparse.SUPPRESS)
    p.add_argument("--pause", dest="t_pause", default=None, type=float,
                    help=argparse.SUPPRESS)
    p.add_argument("--hints", dest="t_hints", default=None,
                    help=argparse.SUPPRESS)
    p.add_argument("--no-hints", dest="t_no_hints", action="store_true",
                    default=None, help=argparse.SUPPRESS)
    p.add_argument("--cloud", dest="t_cloud", action="store_true",
                    default=None, help=argparse.SUPPRESS)
    p.add_argument("--no-engine", dest="t_no_engine", action="store_true",
                    default=None, help=argparse.SUPPRESS)
    p.add_argument("--device", dest="t_device", default=None, type=int,
                    help=argparse.SUPPRESS)

    args = p.parse_args()

    ensure_app()

    if args.mode == "serve":
        _launch(
            {"mode": "serve", "port": args.port, "lang": args.lang,
             "vocab": args.vocab, "max_sessions": args.max_sessions,
             "allow_cloud": args.allow_cloud},
            SERVE_LOG, SERVE_READY_MARKER.encode(),
            f"啟動中，等待 HTTP/WS 伺服器（127.0.0.1:{args.port}）...",
            exit_on_marker=False,
        )
        sys.stdout.write("\n")
        sys.exit(0)

    # live 模式：無子命令時合併頂層（舊用法）參數
    if args.mode is None:
        args = argparse.Namespace(
            mode="live",
            lang=args.t_lang or "zh-TW",
            pause=args.t_pause if args.t_pause is not None else 1.5,
            hints=args.t_hints,
            no_hints=bool(args.t_no_hints),
            cloud=bool(args.t_cloud),
            no_engine=bool(args.t_no_engine),
            device=args.t_device,
        )
    _launch(
        vars(args) | {"mode": "live"},
        LOG, END_MARKER.encode(),
        "啟動中，等待辨識引擎...",
        exit_on_marker=True,
    )


if __name__ == "__main__":
    main()
