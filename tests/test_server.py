"""server.py 測試：REST（TestClient）+ WS 整合（真 uvicorn + websockets client）。

授權 stub：engine_mod._auth_cached = 1 避開 TCC（與 test_engine.py 同模式）。
WS/Session 層用 MockRecognizer（不跑真引擎）；REST 音訊端點會觸發真
SFSpeechRecognizer——無語音授權環境下預期回 4xx/5xx JSON 錯誤（不 crash
不 hang 即通過）；已授權環境則回 200（無語音）或 422（no speech）。
"""

import base64
import json
import math
import socket
import struct
import threading
import time

import httpx
import pytest
import uvicorn
import websockets.sync.client as wsclient
from fastapi.testclient import TestClient
from websockets.exceptions import ConnectionClosed

import macos_stt.engine as engine_mod
from macos_stt.server import create_app
from macos_stt.session import SessionConfig


@pytest.fixture(autouse=True)
def _stub_auth():
    """stub 授權狀態 = authorized，避免 TCC 彈窗/abort。

    注意：常數值必須取 Speech 框架的真實值（Authorized=3，不是 1——
    1 是 Restricted，會誤判成未授權）。
    """
    from Speech import SFSpeechRecognizerAuthorizationStatusAuthorized

    orig = engine_mod.__dict__.get("_auth_cached")
    engine_mod._auth_cached = int(SFSpeechRecognizerAuthorizationStatusAuthorized)
    yield
    if orig is None:
        if "_auth_cached" in engine_mod.__dict__:
            del engine_mod._auth_cached
    else:
        engine_mod._auth_cached = orig


class MockRecognizer:
    """與 test_session.py 同款：餵滿 32000B 模擬出 partial。"""

    def __init__(self, contextual_strings=None, on_partial=None, on_utterance=None,
                 on_final=None, **_kw):
        self.contextual_strings = contextual_strings
        self._on_partial = on_partial
        self.feed_bytes = 0
        self._last_partial = ""

    def start(self):
        pass

    def feed(self, chunk):
        self.feed_bytes += len(chunk)
        if self.feed_bytes >= 32000 and not self._last_partial:
            self._last_partial = "mock-partial"
            if self._on_partial:
                self._on_partial(self._last_partial)

    def finish(self, timeout=15.0):
        from macos_stt.engine import RecognitionResult

        return RecognitionResult(
            text=self._last_partial, confidence=1.0, language="zh-TW"
        )


def _tone_pcm(sr=16000, freq=440, duration=0.5) -> bytes:
    n = int(sr * duration)
    return b"".join(
        struct.pack("<h", int(16000 * math.sin(2 * math.pi * freq * i / sr)))
        for i in range(n)
    )


def _client():
    return TestClient(create_app(recognizer_factory=MockRecognizer))


# ---------------- REST（TestClient） ----------------


def test_health():
    with _client() as c:
        r = c.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["auth"] in ("authorized", "denied", "not_determined")
        assert body["sessions_active"] == 0
        assert body["max_sessions"] == 3
        assert body["default_lang"] == "zh-TW"
        assert body["vocab_size"] == 7


def test_languages():
    with _client() as c:
        r = c.get("/languages")
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == len(body["languages"]) == 63
        assert "zh-TW" in body["languages"]


def test_sessions_endpoint():
    with _client() as c:
        r = c.get("/sessions")
        assert r.status_code == 200
        assert r.json() == {"active": 0, "max": 3}


def test_vocab_get_post():
    with _client() as c:
        r = c.get("/vocabulary")
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 7
        assert "VS Code" in body["words"]

        r = c.post("/vocabulary", json={"mode": "merge", "words": ["商智中心"]})
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 8
        assert "商智中心" in body["words"]

        r = c.post("/vocabulary", json={"mode": "replace", "words": ["A"]})
        assert r.status_code == 200
        assert r.json()["count"] == 1

        r = c.post("/vocabulary", json={"mode": "bogus"})
        assert r.status_code == 400


def test_transcribe_raw_bad_lang():
    with _client() as c:
        r = c.post(
            "/transcribe/raw",
            json={"bytes": base64.b64encode(_tone_pcm()).decode(), "sr": 16000,
                  "lang": "xx-XX"},
        )
        assert r.status_code == 400


def test_transcribe_raw_bad_sr():
    with _client() as c:
        r = c.post(
            "/transcribe/raw",
            json={"bytes": base64.b64encode(_tone_pcm()).decode(), "sr": 44100},
        )
        assert r.status_code == 400


def test_transcribe_raw_tone():
    """440Hz 純音：有語音授權環境 → 200（空文字）或 422（no speech）；
    無授權環境 → 408/500 JSON 錯誤。重點：回 JSON、不 crash、不 hang。"""
    with _client() as c:
        r = c.post(
            "/transcribe/raw",
            json={"bytes": base64.b64encode(_tone_pcm()).decode(), "sr": 16000,
                  "timeout": 3},
            timeout=20,
        )
        assert r.status_code in (200, 408, 422, 500)
        body = r.json()
        if r.status_code == 200:
            assert "text" in body
        else:
            assert "detail" in body


def test_transcribe_upload_tone():
    with _client() as c:
        r = c.post(
            "/transcribe",
            files={"file": ("tone.wav", _tone_pcm(), "application/octet-stream")},
            data={"timeout": "3"},
            timeout=20,
        )
        # 501 = 無語音授權（本環境預期）；其餘為授權環境的正常結果
        assert r.status_code in (200, 408, 422, 500, 501)
        assert r.json()


# ---------------- WS 整合（真 uvicorn thread） ----------------


@pytest.fixture(scope="module")
def ws_base_url():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()

    app = create_app(recognizer_factory=MockRecognizer)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    )
    t = threading.Thread(target=server.run, daemon=True)
    t.start()

    http_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            if httpx.get(http_url + "/health", timeout=0.5).status_code == 200:
                break
        except Exception:
            time.sleep(0.1)
    else:
        pytest.fail("uvicorn did not start in 10s")

    yield f"ws://127.0.0.1:{port}"
    server.should_exit = True
    t.join(timeout=5)


def _send_config(c, **kw):
    cfg = {"type": "config", "sample_rate": 16000}
    cfg.update(kw)
    c.send(json.dumps(cfg))


def _recv_until_close(c, timeout=5.0):
    events = []
    with pytest.raises(ConnectionClosed) as ei:
        while True:
            events.append(json.loads(c.recv(timeout=timeout)))
    return events, ei.value.rcvd


def test_ws_session_events(ws_base_url):
    with wsclient.connect(ws_base_url + "/stream") as c:
        _send_config(c, lang="zh-TW")
        frame = b"\x00\x01" * 6400  # 12800 bytes
        for _ in range(3):
            c.send(frame)
            time.sleep(0.05)
        ev = json.loads(c.recv(timeout=10))
        assert ev["type"] in ("partial", "utterance", "final")
        assert ev.get("text") == "mock-partial"
        c.send(json.dumps({"type": "close"}))


def test_ws_no_config_close(ws_base_url):
    c = wsclient.connect(ws_base_url + "/stream")
    try:
        c.send(b"\x00" * 100)  # 第一則不是 config text
        events, rcvd = _recv_until_close(c)
        assert any(e.get("type") == "error" and e.get("code") == "protocol"
                   for e in events)
        assert rcvd.code == 1002
    finally:
        c.close()


def test_ws_bad_lang(ws_base_url):
    c = wsclient.connect(ws_base_url + "/stream")
    try:
        _send_config(c, lang="xx-XX")
        events, rcvd = _recv_until_close(c)
        assert any(e.get("code") == "protocol" for e in events)
        assert rcvd.code == 1002
    finally:
        c.close()


def test_ws_session_limit(ws_base_url):
    holders = []
    try:
        for _ in range(3):
            c = wsclient.connect(ws_base_url + "/stream")
            _send_config(c)
            c.send(b"\x00\x01" * 6400)
            holders.append(c)
        time.sleep(0.5)  # 讓伺服器完成 3 個 session 建立

        c4 = wsclient.connect(ws_base_url + "/stream")
        _send_config(c4)
        events, rcvd = _recv_until_close(c4)
        assert any(e.get("code") == "too_many_sessions" for e in events)
        assert rcvd.code == 1013
    finally:
        for c in holders:
            try:
                c.close()
            except Exception:
                pass


def test_ws_health_not_blocked(ws_base_url):
    """3 個 session 串流時 /health 仍 < 200ms（event loop 未被鎖）。"""
    http_url = ws_base_url.replace("ws://", "http://")
    holders = []
    try:
        for _ in range(3):
            c = wsclient.connect(ws_base_url + "/stream")
            _send_config(c)
            c.send(b"\x00\x01" * 6400)
            holders.append(c)
        time.sleep(0.5)

        t0 = time.monotonic()
        r = httpx.get(http_url + "/health", timeout=2)
        elapsed_ms = (time.monotonic() - t0) * 1000
        assert r.status_code == 200
        assert r.json()["sessions_active"] == 3
        assert elapsed_ms < 200, f"/health took {elapsed_ms:.0f}ms"
    finally:
        for c in holders:
            try:
                c.close()
            except Exception:
                pass


def test_transcribe_busy_when_stream_holds_engine():
    """串流 session 持槽時，REST 辨識回 503（不與串流重疊）。"""
    app = create_app(recognizer_factory=MockRecognizer)
    mgr = app.state.manager
    mgr._slot = True  # mock 模擬真引擎的單槽
    with TestClient(app) as c:
        s = mgr.acquire(SessionConfig())
        r = c.post(
            "/transcribe/raw",
            json={"bytes": base64.b64encode(_tone_pcm()).decode(), "sr": 16000, "timeout": 3},
            timeout=20,
        )
        assert r.status_code == 503
        assert r.headers.get("retry-after") == "2"
        mgr.release(s)
        assert mgr._rest_busy is False


def test_ws_flush_message_returns_final_with_timing(ws_base_url):
    with wsclient.connect(ws_base_url + "/stream") as c:
        _send_config(c, lang="zh-TW", flush_pre_delay_seconds=0.05)
        frame = b"\x00\x20" * 6400
        for _ in range(6):
            c.send(frame)
        assert json.loads(c.recv(timeout=10))["type"] == "partial"
        c.send(json.dumps({"type": "flush"}))
        events = []
        while not any(e["type"] == "final" for e in events):
            events.append(json.loads(c.recv(timeout=10)))
        final = next(e for e in events if e["type"] == "final")
        assert final["text"] == "mock-partial" and final["timing"]["reason"] == "client_flush"
        assert final["timing"]["pre_delay_ms"] < 200
        c.send(json.dumps({"type": "close"}))
