"""live 測試（需已授權的 macOS 語音辨識環境 + 真實 SFSpeechRecognizer）。

預設全部 skip（conftest.py 處理）；跑法:
    MACSTT_LIVE=1 uv run pytest tests/test_live.py -v

內容（對應 docs/fastapi-server-plan.md §5.2）：
1. TTS 回灌：say 生成繁中 wav → 餵 WS session → 斷言辨識出關鍵詞
2. 詞庫熱更新：POST /vocabulary 後新 session 用新詞庫
3. 3 並發：3 條 WS 同時串流，期間 /health < 200ms
4. session 加速重開：max_session_seconds=5 → 收到 session_restarted
"""

import json
import math
import os
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time

import httpx
import pytest
import websockets.sync.client as wsclient

pytestmark = pytest.mark.live


def _tone_pcm(sr=16000, freq=440, duration=0.5) -> bytes:
    n = int(sr * duration)
    return b"".join(
        struct.pack("<h", int(16000 * math.sin(2 * math.pi * freq * i / sr)))
        for i in range(n)
    )


def _tts_pcm(text: str) -> bytes:
    """TTS 生成語音 → 16k mono int16 raw PCM bytes。

    注意：afconvert 的重取樣要寫在 -d 的 data format（`LEI16@16000`）；
    單獨 `-r 16000` 會被靜默忽略（輸出維持 22050Hz），餵進引擎會
    因音調偏移辨識不出內容。WAVE 輸出有 44B header，需剝掉。
    """
    aiff = tempfile.NamedTemporaryFile(suffix=".aiff", delete=False).name
    wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav.close()
    try:
        r = subprocess.run(["say", "-o", aiff, "-v", "Ting-Ting", text],
                           capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(f"say failed: {r.stderr.decode(errors='replace')}")
        r = subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
             aiff, wav.name],
            capture_output=True,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"afconvert failed: {r.stderr.decode(errors='replace')}")
        with open(wav.name, "rb") as f:
            blob = f.read()
        assert blob[:4] == b"RIFF", "afconvert 輸出不是 WAVE"
        return blob[blob.index(b"data") + 8:]
    finally:
        os.unlink(aiff)
        os.unlink(wav.name)


@pytest.fixture(scope="module")
def live_server():
    """真引擎 server（無 mock）。

    **必須經 MacSTT.app bundle 啟動**：非 bundle process 呼叫
    SFSpeechRecognizer 會 TCC SIGABRT（crash 報告證實）。
    若首次執行時還沒授權過，此 fixture 會提示先手動跑一次
    `python start.py serve` 同意授權對話框。
    """
    import sys as _sys

    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import start as start_mod  # 專案根目錄的 start.py

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()

    start_mod.ensure_app()
    # 清舊程序
    old = start_mod.find_child()
    if old and start_mod._alive(old):
        try:
            os.kill(old, signal.SIGTERM)
        except (ProcessLookupError, TypeError):
            pass
    subprocess.run(["pkill", "-f", "MacSTT.app/Contents/MacOS/macstt"],
                   capture_output=True)
    time.sleep(0.5)
    for f in (start_mod.PIDF, start_mod.SERVE_LOG):
        try:
            os.unlink(f)
        except FileNotFoundError:
            pass

    with open(start_mod.CFG, "w") as f:
        json.dump({"mode": "serve", "port": port, "lang": "zh-TW",
                   "vocab": None, "max_sessions": 3, "allow_cloud": False}, f)
    subprocess.run(["open", start_mod.APP])

    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 45  # 首次授權對話框可能需要人工點擊
    up = False
    while time.time() < deadline:
        try:
            if httpx.get(base + "/health", timeout=0.5).status_code == 200:
                up = True
                break
        except Exception:
            time.sleep(0.3)
    if not up:
        try:
            status = httpx.get(base + "/health", timeout=1).json().get("auth", "?")
        except Exception:
            status = "unreachable"
        pytest.fail(
            "bundle 伺服器未上線（auth status=%s）。"
            "若這是首次執行：請先跑 `python start.py serve` 並同意"
            "「語音辨識」授權對話框，再重跑本測試。" % status
        )

    yield base

    # 收尾：SIGTERM 結束 bundle 內的 server
    pid = start_mod.find_child()
    if pid and start_mod._alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, TypeError):
            pass
    time.sleep(1)
    subprocess.run(["pkill", "-f", "MacSTT.app/Contents/MacOS/macstt"],
                   capture_output=True)
    try:
        os.unlink(start_mod.SERVE_LOG)
    except FileNotFoundError:
        pass


def _require_authorized(base: str) -> None:
    """未授權時 skip（而非 fail）——授權需人工點對話框。"""
    h = httpx.get(base + "/health", timeout=2).json()
    if h["auth"] != "authorized":
        pytest.skip(f"語音辨識未授權（auth={h['auth']}）：先跑 python start.py serve 授權")


def _ws(base: str):
    return wsclient.connect(base.replace("http://", "ws://") + "/stream")


def _stream_pcm(ws, pcm: bytes, frame=3200, silence_seconds=3.2):
    """0.2s/幀（3200 samples@16kHz）**真實時**餵完 pcm，再送 3.2s 靜音。

    注意 pacing 必須等於每幀實際秒數（3200 samples @16k = 0.2s），
    否則變成 2× 速率餵入，引擎處理跟不上、VAD 時鐘與引擎 backlog
    錯位，短句（<1.5s）會完全拿不到 final。
    final 只在「靜音 ≥ vad_flush_seconds 觸發安全網 flush」時送出，
    收尾靜音要夠長（config 用 vad_flush_seconds=1.5 → 3.2s 有餘裕）。
    """
    for i in range(0, len(pcm) - len(pcm) % (frame * 2), frame * 2):
        ws.send(pcm[i:i + frame * 2])
        time.sleep(0.2)
    for _ in range(int(silence_seconds / 0.2)):
        ws.send(b"\x00" * frame * 2)
        time.sleep(0.2)


def test_live_tts_roundtrip(live_server):
    """TTS「你好，商智中心」→ WS 串流 → final 文字含關鍵詞。

    「商智中心」不在預設詞庫（A7 專測詞庫），故本測試以 per-request
    hints 注入——引擎 contextualStrings 命中後 final 精準回「商智中心」。
    """
    assert os.environ.get("MACSTT_LIVE") == "1"
    _require_authorized(live_server)
    pcm = _tts_pcm("你好，商智中心")
    with _ws(live_server) as ws:
        ws.send(json.dumps({"type": "config", "lang": "zh-TW",
                            "sample_rate": 16000, "max_session_seconds": 60,
                            "vad_flush_seconds": 1.5,
                            "hints": ["商智中心"]}))
        _stream_pcm(ws, pcm)
        text = ""
        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                ev = json.loads(ws.recv(timeout=5))
            except Exception:
                break
            if ev.get("type") in ("final", "utterance"):
                text += ev.get("text", "")
                if "商智" in text:
                    break
        ws.send(json.dumps({"type": "close"}))
    print(f"\n[live] 辨識結果: {text!r}")
    assert "商智" in text, f"TTS 回灌辨識失敗: {text!r}"


def test_live_vocab_hot_reload(live_server):
    """熱更新詞庫後，新 session 的 contextualStrings 含新詞（store 層驗證
    + live 辨識對照）。"""
    _require_authorized(live_server)
    r = httpx.post(live_server + "/vocabulary",
                   json={"mode": "merge", "words": ["商智中心", "BIRC"]})
    assert r.status_code == 200
    words = r.json()["words"]
    assert "商智中心" in words and "BIRC" in words

    # 新 session 用新詞庫辨識（句稍長，讓引擎來得及出 partial/flush）
    pcm = _tts_pcm("歡迎來到商智中心")
    with _ws(live_server) as ws:
        ws.send(json.dumps({"type": "config", "lang": "zh-TW",
                            "sample_rate": 16000, "vad_flush_seconds": 1.5}))
        _stream_pcm(ws, pcm)
        text = ""
        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                ev = json.loads(ws.recv(timeout=5))
            except Exception:
                break
            if ev.get("type") in ("final", "utterance"):
                text += ev.get("text", "")
                if "商智" in text:
                    break
        ws.send(json.dumps({"type": "close"}))
    print(f"\n[live] 詞庫熱更新辨識: {text!r}")
    assert "商智" in text, f"詞庫熱更新辨識失敗: {text!r}"
    # 還原（replace 回內建）
    from macos_stt.vocab import VocabularyStore
    default = VocabularyStore.load_default().get()
    r = httpx.post(live_server + "/vocabulary",
                   json={"mode": "replace", "words": default})
    assert r.status_code == 200


def test_live_three_parallel(live_server):
    """3 條 WS 同時串流（各 2s TTS），期間 /health < 200ms。"""
    _require_authorized(live_server)
    pcm = _tts_pcm("這是並發測試，第一句話")
    results = {}

    def client(i: int):
        with _ws(live_server) as ws:
            ws.send(json.dumps({"type": "config", "lang": "zh-TW",
                                "sample_rate": 16000,
                                "vad_flush_seconds": 1.5}))
            _stream_pcm(ws, pcm)
            text = ""
            deadline = time.time() + 20
            while time.time() < deadline:
                try:
                    ev = json.loads(ws.recv(timeout=5))
                except Exception:
                    break
                if ev.get("type") in ("final", "utterance"):
                    text += ev.get("text", "")
                    break  # 收到任一句結果即達標（並發測試不看內容）
            ws.send(json.dumps({"type": "close"}))
        results[i] = text

    threads = [threading.Thread(target=client, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    time.sleep(3)  # 讓 3 個 session 都建立後測 health

    t0 = time.monotonic()
    r = httpx.get(live_server + "/health", timeout=2)
    elapsed_ms = (time.monotonic() - t0) * 1000
    assert r.status_code == 200
    assert r.json()["sessions_active"] == 3
    assert elapsed_ms < 200, f"/health 在 3 並發時 took {elapsed_ms:.0f}ms"

    for t in threads:
        t.join(timeout=60)
    for i, text in results.items():
        print(f"\n[live] 並發#{i}: {text!r}")
        assert text, f"並發#{i} 沒有收到任何辨識結果"


def test_live_session_restart(live_server):
    """max_session_seconds=5 加速重開：持續餵 20s → 收到 session_restarted。

    時序說明：重開點 = 5s（上限）+ 0.4s（句尾等待）+ finish 等 final
    （pure tone 無語音時引擎 final 可能慢，最長 10s）≈ 5.4~15.4s，
    故餵 20s 覆蓋兩種情形；pump 的 recv 超時不中斷（持續收到 restart）。
    """
    _require_authorized(live_server)
    with _ws(live_server) as ws:
        ws.send(json.dumps({"type": "config", "lang": "zh-TW",
                            "sample_rate": 16000, "max_session_seconds": 5}))
        events = []
        restarted = threading.Event()

        def pump():
            try:
                while not restarted.is_set():
                    try:
                        msg = ws.recv(timeout=5)
                    except TimeoutError:
                        continue  # 靜音期沒事件是正常
                    ev = json.loads(msg)
                    events.append(ev)
                    if ev.get("type") == "session_restarted":
                        restarted.set()
                        return
            except Exception:
                pass

        pt = threading.Thread(target=pump, daemon=True)
        pt.start()
        tone = _tone_pcm(duration=0.2)
        # 餵 20s（0.2s/幀）
        for _ in range(100):
            if restarted.is_set() and not pt.is_alive():
                break
            ws.send(tone)
            time.sleep(0.2)
        # 再給 5s 讓 pump 收尾
        deadline = time.time() + 5
        while not restarted.is_set() and time.time() < deadline:
            time.sleep(0.2)
        ws.send(json.dumps({"type": "close"}))
    assert any(e.get("type") == "session_restarted" for e in events), \
        f"20s 串流未觸發重開，events={events}"
