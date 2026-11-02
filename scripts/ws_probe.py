#!/usr/bin/env python3
"""WS STT 手動探測腳本（docs/fastapi-server-plan.md §5.4）。

用法:
    uv run python scripts/ws_probe.py --audio /path/to/any_audio \
        [--url ws://127.0.0.1:8765/stream] [--lang zh-TW] [--hints 商智中心,BIRC]

行為:
    - .pcm 視為 raw int16 mono 16kHz；其他格式先用 afconvert 轉成 16k mono int16
    - 每 0.2s 送 6400 samples（12800 bytes），模擬即時音訊節奏
    - 並行印出伺服器事件（partial / utterance / final / session_restarted / error）
    - 全部送完後再送 1s 靜音，等 final 收尾，送 {"type":"close"} 結束
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time

import websockets.sync.client as wsclient

FRAME = 6400  # 16kHz × 0.2s int16 mono samples


def main() -> int:
    p = argparse.ArgumentParser(description="macos-stt WS /stream 探測腳本")
    p.add_argument("--audio", required=True, help="音訊檔（.pcm=16k mono int16，其他自動 afconvert）")
    p.add_argument("--url", default="ws://127.0.0.1:8765/stream")
    p.add_argument("--lang", default=None, help="省略 → 用伺服器預設")
    p.add_argument("--hints", default=None, help="逗號分隔的附加提示詞")
    args = p.parse_args()

    if not os.path.isfile(args.audio):
        print(f"錯誤: 找不到音訊檔 {args.audio}", file=sys.stderr)
        return 1

    # ---- 準備 PCM（16k mono int16 raw） ----
    tmp = None
    if args.audio.lower().endswith(".pcm"):
        pcm = args.audio
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=".raw", delete=False)
        tmp.close()
        print("afconvert 轉 16k mono int16 ...")
        r = subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
             args.audio, tmp.name]
        )
        if r.returncode != 0:
            print("錯誤: afconvert 轉換失敗", file=sys.stderr)
            return 1
        with open(tmp.name, "rb") as f:
            blob = f.read()
        # WAVE header（RIFF....data 偏移）
        if blob[:4] == b"RIFF":
            off = blob.index(b"data") + 8
            blob = blob[off:]
        with open(tmp.name, "wb") as f:
            f.write(blob)
        pcm = tmp.name

    with open(pcm, "rb") as f:
        data = f.read()
    if tmp:
        os.unlink(tmp.name)
    n_frames = (len(data) + (FRAME * 2 - 1)) // (FRAME * 2)
    print(f"音訊 {len(data) // 2 / 16000:.1f}s（{n_frames} 幀），連線 {args.url}")

    # ---- 事件接收 thread ----
    done = threading.Event()
    stop_receiver = threading.Event()

    def receiver(ws):
        try:
            while not stop_receiver.is_set():
                try:
                    msg = ws.recv(timeout=0.2)
                except TimeoutError:
                    continue
                except Exception:
                    return
                try:
                    ev = json.loads(msg)
                except json.JSONDecodeError:
                    print(f"[?] {msg!r}")
                    continue
                t = ev.get("type")
                if t == "partial":
                    print(f"  … {ev['text']}")
                else:
                    print(json.dumps(ev, ensure_ascii=False))
        except Exception:
            pass
        finally:
            done.set()

    try:
        ws = wsclient.connect(args.url, open_timeout=5)
    except Exception as e:
        print(f"錯誤: 無法連線 {args.url}: {e}", file=sys.stderr)
        return 1

    t = threading.Thread(target=receiver, args=(ws,), daemon=True)
    t.start()

    # ---- config ----
    cfg = {"type": "config", "sample_rate": 16000}
    if args.lang:
        cfg["lang"] = args.lang
    if args.hints:
        cfg["hints"] = [h.strip() for h in args.hints.split(",") if h.strip()]
    ws.send(json.dumps(cfg))

    # ---- 送音 ----
    for i in range(n_frames):
        frame = data[i * FRAME * 2:(i + 1) * FRAME * 2]
        ws.send(frame)
        time.sleep(0.2)
    # 1s 靜音收尾（讓引擎出 final）
    for _ in range(5):
        ws.send(b"\x00" * (FRAME * 2))
        time.sleep(0.2)
    time.sleep(1.5)
    ws.send(json.dumps({"type": "close"}))
    time.sleep(0.5)
    stop_receiver.set()
    try:
        ws.close()
    except Exception:
        pass
    print("👋 結束。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
