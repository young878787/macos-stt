"""HTTP + WebSocket server for macos-stt（FastAPI + uvicorn）。

架構：所有 SFSpeechRecognizer 的 ObjC 操作都序列化到 EngineThread
（常駐 daemon thread）執行，不留在 event loop；recognition callback 實測
固定遞交在主執行緒 RunLoop，因此 lifespan 啟動 _pump_main_runloop 定期
pump 主 RunLoop 作接收端。引擎同時只能跑 1 個識別 task，SessionManager
以單槽輪替讓多條 WS 連線共用。REST 端點用 asyncio.to_thread(fut.result)
等非阻塞 Future；WS /stream 的 on_event 用 loop.call_soon_threadsafe
橋接回 event loop。

只綁 127.0.0.1（決策 D3）；預設語言 zh-TW（D4 詞庫見 vocab.default.txt）。
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import tempfile
import threading
import wave
from contextlib import asynccontextmanager
from typing import Any, Callable

import Foundation
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .engine import (
    MacOSSpeechError,
    RecognitionError,
    RecognitionTimeout,
    UnsupportedLanguage,
    list_languages,
    transcribe_file_engine,
)
from .engine_thread import EngineThread
from .session import FLUSH_PRE_DELAY, SessionConfig, SessionLimitExceeded, SessionManager
from .vocab import VocabularyStore

_NO_SPEECH_RE = re.compile(r"no[_ ]?speech|1110", re.I)
_WS_CONFIG_TIMEOUT = 10.0
_WS_MAX_FRAME = 256 * 1024
_VALID_SR = 16000


class RawRequest(BaseModel):
    """POST /transcribe/raw：base64 int16 mono PCM。"""

    bytes: str  # base64
    sr: int = _VALID_SR
    lang: str | None = None
    hints: list[str] = []
    timeout: float = 15.0
    on_device: bool = True


class VocabRequest(BaseModel):
    """POST /vocabulary：replace | merge | load。"""

    mode: str
    words: list[str] = []
    file: str | None = None


class _ProtocolError(Exception):
    """WS 協定錯誤（close 1002）。"""


def _auth_status_str() -> str:
    """讀語音辨識授權狀態（**只讀**，不呼叫 requestAuthorization_，
    避免在伺服器背景彈 TCC 授權窗）。"""
    from Speech import (
        SFSpeechRecognizer,
        SFSpeechRecognizerAuthorizationStatusAuthorized,
        SFSpeechRecognizerAuthorizationStatusDenied,
    )

    status = int(SFSpeechRecognizer.authorizationStatus())
    if status == int(SFSpeechRecognizerAuthorizationStatusAuthorized):
        return "authorized"
    if status == int(SFSpeechRecognizerAuthorizationStatusDenied):
        return "denied"
    return "not_determined"


def _speech_http_error(e: MacOSSpeechError) -> HTTPException:
    if isinstance(e, UnsupportedLanguage):
        return HTTPException(status_code=400, detail=str(e))
    if isinstance(e, MacOSSpeechError) and type(e).__name__ == "AuthorizationError":
        return HTTPException(
            status_code=501,
            detail="Speech recognition not authorized. Grant permission in "
            "System Settings → Privacy & Security → Speech Recognition, "
            "then restart the server.",
        )
    if isinstance(e, RecognitionTimeout):
        return HTTPException(status_code=408, detail="Recognition timed out")
    if isinstance(e, RecognitionError) and _NO_SPEECH_RE.search(str(e)):
        return HTTPException(status_code=422, detail="No speech detected")
    return HTTPException(status_code=500, detail=str(e))


async def _pump_main_runloop() -> None:
    """uvicorn 的 event loop 用 kqueue，不 pump 主 CFRunLoop；但
    SFSpeechRecognizer 的 callback 遞交在主執行緒的 RunLoop /
    main dispatch queue 上（engine.py 原始實測：子執行緒 pump 自己的
    RunLoop 收不到 callback）。因此主執行緒必須定期切短片（≤4ms）
    pump 主 RunLoop，callback 與 main queue 的 dispatch_sync 才會被
    處理。此 coroutine 只在 uvicorn loop 確實在主執行緒時啟動。"""
    rl = Foundation.NSRunLoop.mainRunLoop()
    while True:
        rl.runMode_beforeDate_(
            Foundation.NSDefaultRunLoopMode,
            Foundation.NSDate.dateWithTimeIntervalSinceNow_(0.004),
        )
        await asyncio.sleep(0.01)


def create_app(
    default_lang: str = "zh-TW",
    vocab: VocabularyStore | None = None,
    max_sessions: int = 3,
    allow_cloud: bool = False,
    recognizer_factory: Callable[..., Any] | None = None,
) -> FastAPI:
    """建立 FastAPI app。

    Args:
        default_lang: 全域預設語言（請求省略 lang 時使用）。
        vocab: 詞庫 store；None → 內建 vocab.default.txt。
        max_sessions: 最大同時串流 session（預設 3）。
        allow_cloud: True 時允許 on_device=false（雲端辨識）。
        recognizer_factory: 測試注入（簽名 factory(contextual_strings=...,
            on_partial=..., on_utterance=..., on_final=..., **kw)）。
    """
    vocab = vocab if vocab is not None else VocabularyStore.load_default()
    manager = SessionManager(
        max_sessions=max_sessions, vocabulary=vocab, recognizer_factory=recognizer_factory
    )
    languages = list_languages()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        et = EngineThread.shared()  # 啟動 daemon engine thread
        pump_task = None
        if threading.current_thread() is threading.main_thread():
            # 主執行緒 = bundle 的 main thread：啟動 RunLoop pump
            # （測試環境 uvicorn 跑在背景 thread → 不啟動，CFRunLoop
            #   不可跨執行緒 run）
            pump_task = asyncio.get_running_loop().create_task(_pump_main_runloop())
        yield
        if pump_task is not None:
            pump_task.cancel()
            try:
                await pump_task
            except (asyncio.CancelledError, Exception):
                pass
        et.stop()

    app = FastAPI(
        title="macos-stt",
        description="macOS 系統語音辨識 HTTP/WS 服務 — 全離線、串流、63 語言",
        version="0.3.0",
        lifespan=lifespan,
    )
    app.state.default_lang = default_lang
    app.state.vocab = vocab
    app.state.manager = manager
    app.state.allow_cloud = allow_cloud

    # ---------- REST ----------

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "auth": _auth_status_str(),
            "sessions_active": manager.stats["active"],
            "max_sessions": manager.stats["max"],
            "default_lang": default_lang,
            "vocab_size": len(vocab),
        }

    @app.get("/languages")
    async def get_languages():
        return {"languages": languages, "count": len(languages)}

    @app.get("/sessions")
    async def get_sessions():
        return manager.stats

    def _merged_hints(request_hints: list[str]) -> list[str]:
        """詞庫 store ⊕ 請求 hints（保序去重，vocab 先）。"""
        return list(dict.fromkeys(list(vocab.get()) + list(request_hints)))

    def _run_transcribe(
        path: str, lang: str, timeout: float, on_device: bool, hints: list[str]
    ) -> dict:
        """同步執行 transcribe_file_engine（由 to_thread 呼叫）。

        引擎同時只能跑 1 個 task：有串流 session 持槽或另一個 REST 進行中
        時回 503，不與串流重疊（重疊會讓串流收不到 final）。"""
        if not manager.try_acquire_engine():
            raise HTTPException(
                status_code=503,
                detail="Engine busy (streaming session or another request in progress)",
                headers={"Retry-After": "2"},
            )
        try:
            fut = transcribe_file_engine(
                path,
                language=lang,
                timeout=timeout,
                on_device=on_device,
                contextual_strings=_merged_hints(hints) or None,
            )
            return fut.result(timeout + 5)
        finally:
            manager.release_engine()

    @app.post("/transcribe")
    async def transcribe(
        file: UploadFile = File(...),
        lang: str | None = Form(None),
        timeout: float = Form(15.0),
        on_device: bool = Form(True),
        hints: str = Form(""),
    ):
        """整檔上傳辨識（WAV/MP3/M4A 等 macOS 可解格式）。"""
        lang = lang or default_lang
        if lang not in languages:
            raise HTTPException(status_code=400, detail=f"Unsupported language '{lang}'.")
        if not on_device and not allow_cloud:
            raise HTTPException(status_code=400, detail="Cloud recognition not allowed on this server.")
        request_hints = [h.strip() for h in re.split(r"[,，\n;；]+", hints) if h.strip()]

        suffix = ".wav"
        if file.filename and "." in file.filename:
            suffix = "." + file.filename.rsplit(".", 1)[-1]
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        try:
            tmp.write(await file.read())
            tmp.close()
            try:
                result = await asyncio.to_thread(
                    _run_transcribe, tmp.name, lang, timeout, on_device, request_hints
                )
            except MacOSSpeechError as e:
                raise _speech_http_error(e)
            return {"text": result.text, "confidence": result.confidence, "language": result.language}
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    @app.post("/transcribe/raw")
    async def transcribe_raw(req: RawRequest):
        """base64 PCM 辨識（int16 mono，v1 只接受 16kHz）。"""
        lang = req.lang or default_lang
        if lang not in languages:
            raise HTTPException(status_code=400, detail=f"Unsupported language '{lang}'.")
        if req.sr != _VALID_SR:
            raise HTTPException(status_code=400, detail=f"Only {_VALID_SR} Hz PCM supported in v1.")
        if not req.on_device and not allow_cloud:
            raise HTTPException(status_code=400, detail="Cloud recognition not allowed on this server.")
        try:
            data = base64.b64decode(req.bytes)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 payload")

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            with wave.open(tmp, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(req.sr)
                wf.writeframes(data)
            tmp.close()
            try:
                result = await asyncio.to_thread(
                    _run_transcribe, tmp.name, lang, req.timeout, req.on_device, list(req.hints)
                )
            except MacOSSpeechError as e:
                raise _speech_http_error(e)
            return {"text": result.text, "confidence": result.confidence, "language": result.language}
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    @app.get("/vocabulary")
    async def get_vocabulary():
        words = vocab.get()
        return {"words": words, "count": len(words), "warning": vocab.warning}

    @app.post("/vocabulary")
    async def post_vocabulary(req: VocabRequest):
        """詞庫熱更新（僅對新 session 生效）。"""
        try:
            if req.mode == "replace":
                vocab.replace(req.words)
            elif req.mode == "merge":
                vocab.merge(req.words)
            elif req.mode == "load":
                if not req.file:
                    raise HTTPException(status_code=400, detail="mode=load requires 'file'")
                vocab.load_file_merged(req.file)
            else:
                raise HTTPException(status_code=400, detail=f"Invalid mode '{req.mode}' (replace|merge|load)")
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail=f"Vocab file not found: {req.file}")
        words = vocab.get()
        return {"words": words, "count": len(words), "warning": vocab.warning}

    # ---------- WebSocket /stream ----------

    @app.websocket("/stream")
    async def ws_stream(ws: WebSocket):
        """一 connection = 一 session（佔 max_sessions 名額）。

        協定 v1：首則 text frame 為 config；之後 binary frame = int16 mono
        16kHz PCM（建議 200ms≈6400 samples）；text {"type":"flush"} 要求立刻
        收尾這一句（client 自己判斷說完了，final 帶 timing）；text {"type":"close"} 收尾。
        config 可選 flush_pre_delay_seconds（斷句後 endAudio 前的等待，預設 0.4）。
        伺服器回 text JSON：partial / utterance / final / session_restarted /
        error。
        """
        await ws.accept()
        loop = asyncio.get_running_loop()
        out_q: asyncio.Queue = asyncio.Queue()
        session = None
        try:
            # 1) config（10s 超時）
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=_WS_CONFIG_TIMEOUT)
            except asyncio.TimeoutError:
                raise _ProtocolError("timed out waiting for config")
            except Exception as e:
                raise _ProtocolError(f"config must be a text JSON frame ({type(e).__name__})")

            try:
                cfg = json.loads(raw)
            except json.JSONDecodeError:
                raise _ProtocolError("config is not valid JSON")
            if not isinstance(cfg, dict) or cfg.get("type") != "config":
                raise _ProtocolError("first message must be {\"type\":\"config\",...}")

            lang = cfg.get("lang") or default_lang
            if lang not in languages:
                raise _ProtocolError(f"unsupported language '{lang}'")
            sr = cfg.get("sample_rate", _VALID_SR)
            if sr != _VALID_SR:
                raise _ProtocolError(f"only sample_rate {_VALID_SR} supported in v1")
            on_device = bool(cfg.get("on_device", True))
            if not on_device and not allow_cloud:
                raise _ProtocolError("on_device=false requires server started with --allow-cloud")
            scfg = SessionConfig(
                language=lang,
                on_device=on_device,
                multi_utterance=bool(cfg.get("multi_utterance", True)),
                vad_flush_seconds=float(cfg.get("vad_flush_seconds", 3.0)),
                max_session_seconds=float(cfg.get("max_session_seconds", 300.0)),
                flush_pre_delay=float(cfg.get("flush_pre_delay_seconds", FLUSH_PRE_DELAY)),
                hints=list(cfg.get("hints") or []),
            )

            # on_event 在 engine thread / worker thread 被呼叫 → 橋接回 event loop
            def on_event(ev: dict) -> None:
                loop.call_soon_threadsafe(out_q.put_nowait, ev)

            try:
                session = await asyncio.to_thread(manager.acquire, scfg, on_event)
            except SessionLimitExceeded:
                await ws.send_json(
                    {"type": "error", "code": "too_many_sessions",
                     "message": f"session limit reached (max={manager.stats['max']})"}
                )
                await ws.close(1013)
                return

            # 2) 雙任務：收音 / 發文字
            async def feed() -> None:
                while True:
                    msg = await ws.receive()
                    if msg.get("type") == "websocket.disconnect":
                        return
                    if msg.get("bytes") is not None:
                        data = msg["bytes"]
                        if len(data) > _WS_MAX_FRAME:
                            raise _ProtocolError("audio frame too large (max 256KB)")
                        await asyncio.to_thread(session.feed, data)
                    elif msg.get("text") is not None:
                        try:
                            t = json.loads(msg["text"]).get("type")
                        except json.JSONDecodeError:
                            continue
                        if t == "close":
                            return
                        if t == "flush":
                            # 與音訊幀同一條序列處理：flush 一定在先前的幀都餵進去之後才執行
                            await asyncio.to_thread(session.flush)

            async def pump() -> None:
                while True:
                    ev = await out_q.get()
                    if ev is None:
                        return
                    await ws.send_text(json.dumps(ev, ensure_ascii=False))

            feed_task = asyncio.create_task(feed())
            pump_task = asyncio.create_task(pump())
            try:
                done, pending = await asyncio.wait(
                    {feed_task, pump_task}, return_when=asyncio.FIRST_COMPLETED
                )
                for t in pending:
                    t.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                for t in done:
                    exc = t.exception()
                    if exc is not None:
                        raise exc
            finally:
                # 停止 pump，避免後續事件送進已關閉的 ws
                out_q.put_nowait(None)

        except _ProtocolError as e:
            try:
                await ws.send_json({"type": "error", "code": "protocol", "message": str(e)})
            except Exception:
                pass
            try:
                await ws.close(1002)
            except Exception:
                pass
        except WebSocketDisconnect:
            pass
        except Exception as e:  # 未預期錯誤
            try:
                await ws.send_json({"type": "error", "code": "internal", "message": str(e)})
            except Exception:
                pass
            try:
                await ws.close(1011)
            except Exception:
                pass
        finally:
            if session is not None:
                await asyncio.to_thread(manager.release, session)

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=8765)
