#!/usr/bin/env python3
"""Python 包裝器: 一鍵啟動即時中文轉譯，終端即時顯示辨識與斷句。

為什麼需要 app bundle?
    macOS TCC 要求語音辨識程式必須是 .app bundle（內含用途說明），
    直接 python live_zh.py 會被系統 SIGABRT。本包裝器會:
      1. 自動建立 MacSTT.app（最小 bundle，3 個檔案，可随时手動刪除，
         下次執行會自動重建）
      2. open MacSTT.app 啟動 live_zh.py
      3. 把後端輸出即時 tail 到終端
      4. 把終端的 Ctrl+C 轉發給後端，乾淨收尾

用法:
    uv run python start.py                 # 預設 zh-TW，引擎自斷句（連續收音）
    uv run python start.py --lang zh-CN    # 簡體中文
    uv run python start.py --no-engine     # 改回舊版「靜音 --pause 秒斷句」
    uv run python start.py --pause 2.0     # 靜音 2 秒才斷句（--no-engine 時）
    uv run python start.py --device 1      # 指定輸入裝置
    uv run python start.py --cloud         # 允許雲端辨識（中英混合較穩，需網路）
    uv run python start.py --hints 詞表.txt # 自訂提示詞（VS Code 等專有名詞）
    uv run python start.py --no-hints      # 停用內建提示詞
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
    a = {{"lang": "zh-TW", "pause": 1.5, "device": None}}

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

os.chdir(PROJECT)
sys.path.insert(0, os.path.join(PROJECT, "src"))
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


def main():
    p = argparse.ArgumentParser(description="即時中文轉譯（終端即時顯示）")
    p.add_argument("--lang", default="zh-TW",
                   help="zh-TW(繁中) / zh-CN(簡中) / yue-CN(粵語)")
    p.add_argument("--pause", type=float, default=1.5,
                   help="靜音幾秒斷句（預設 1.5；中英混合說術語時建議 1.5-2.0）")
    p.add_argument("--hints", default=None,
                   help="提示詞: 檔案路徑（每行/逗號分隔）或逗號分隔清單；"
                        "不指定時用內建開發術語清單")
    p.add_argument("--no-hints", action="store_true",
                   help="停用提示詞（contextualStrings）")
    p.add_argument("--cloud", action="store_true",
                   help="允許雲端辨識（預設強制離線）；中英混合辨識較穩，需網路")
    p.add_argument("--no-engine", action="store_true",
                   help="停用引擎自斷句，改回舊版「靜音 --pause 秒斷句」")
    p.add_argument("--device", type=int, default=None,
                   help="輸入裝置編號（預設系統預設麥克風）")
    args = p.parse_args()

    ensure_app()

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
        json.dump(vars(args), f)
    try:
        os.unlink(LOG)
    except FileNotFoundError:
        pass

    subprocess.run(["open", APP])
    print("啟動中，等待辨識引擎...", flush=True)

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
    while not os.path.exists(LOG) and time.time() < deadline:
        time.sleep(0.1)
    if not os.path.exists(LOG):
        print("錯誤: 後端未產生輸出（可能 crash），檢查 ~/Library/Logs/DiagnosticReports/")
        sys.exit(1)

    # tail 輸出到終端
    with open(LOG, "rb") as f:
        gone = 0
        while True:
            chunk = f.read(4096)
            if chunk:
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                if END_MARKER.encode() in chunk:
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
        os.unlink(LOG)
    except FileNotFoundError:
        pass
    sys.stdout.write("\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
