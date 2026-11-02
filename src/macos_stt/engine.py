"""macOS SFSpeechRecognizer 核心引擎。

SFSpeechRecognizer 的 recognition task callback 固定綁定在「主執行緒」的
RunLoop 上，因此所有辨識邏輯必須在主執行緒執行、並持續 pump
NSRunLoop 才能收到 callback。

（舊版把 recognizer 建在子執行緒、pump 子執行緒的 RunLoop，
永遠收不到 callback，導致 transcribe 一律超時，現已修正。）

注意: 本模組的 transcribe_* 函式只能從主執行緒呼叫。
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
import wave
from concurrent.futures import Future, InvalidStateError
from dataclasses import dataclass

import Foundation
from Speech import (
    SFSpeechRecognizer,
    SFSpeechRecognizerAuthorizationStatusAuthorized,
    SFSpeechRecognizerAuthorizationStatusDenied,
    SFSpeechRecognizerAuthorizationStatusNotDetermined,
    SFSpeechURLRecognitionRequest,
)

SUPPORTED_LANGUAGES = [
    "ar-SA", "ca-ES", "cs-CZ", "da-DK", "de-AT", "de-CH", "de-DE",
    "el-GR", "en-AE", "en-AU", "en-CA", "en-GB", "en-ID", "en-IE",
    "en-IN", "en-NZ", "en-PH", "en-SA", "en-SG", "en-US", "en-ZA",
    "es-419", "es-CL", "es-CO", "es-ES", "es-MX", "es-US", "fi-FI",
    "fr-BE", "fr-CA", "fr-CH", "fr-FR", "he-IL", "hi-IN",
    "hi-IN-translit", "hi-Latn", "hr-HR", "hu-HU", "id-ID", "it-CH",
    "it-IT", "ja-JP", "ko-KR", "ms-MY", "nb-NO", "nl-BE", "nl-NL",
    "pl-PL", "pt-BR", "pt-PT", "ro-RO", "ru-RU", "sk-SK", "sv-SE",
    "th-TH", "tr-TR", "uk-UA", "vi-VN", "wuu-CN", "yue-CN",
    "zh-CN", "zh-HK", "zh-TW",
]

_auth_lock = threading.Lock()
_auth_cached: int | None = None


def _has_speech_usage_description() -> bool:
    """本 process 是否跑在宣告 NSSpeechRecognitionUsageDescription 的 .app bundle 內。

    非 bundle process（直接 CLI / pytest / CI）呼叫 requestAuthorization_
    會被 TCC 直接 SIGABRT（__TCC_CRASHING_DUE_TO_PRIVACY_VIOLATION__），
    整個 process 暴死——所以授權「請求」只能在 bundle 內做；
    只讀 authorizationStatus() 則是安全的。
    """
    try:
        info = Foundation.NSBundle.mainBundle().infoDictionary() or {}
        return bool(info.get("NSSpeechRecognitionUsageDescription"))
    except Exception:
        return False


def _ensure_authorized() -> None:
    global _auth_cached
    if _auth_cached is not None:
        return
    with _auth_lock:
        if _auth_cached is not None:
            return
        status = SFSpeechRecognizer.authorizationStatus()
        if status == SFSpeechRecognizerAuthorizationStatusAuthorized:
            _auth_cached = int(status)
            return
        if status == SFSpeechRecognizerAuthorizationStatusNotDetermined:
            if not _has_speech_usage_description():
                # 非 bundle：無法彈授權窗（會 TCC abort），只記錄現狀，
                # 後續 transcribe 會以 AuthorizationError → 501 明確回報
                _auth_cached = int(status)
                return
            done = threading.Event()
            result = []
            def cb(s):
                result.append(int(s))
                done.set()
            SFSpeechRecognizer.requestAuthorization_(cb)
            deadline = time.time() + 30.0
            run_loop = Foundation.NSRunLoop.currentRunLoop()
            while time.time() < deadline and not done.is_set():
                run_loop.runMode_beforeDate_(
                    Foundation.NSDefaultRunLoopMode,
                    Foundation.NSDate.dateWithTimeIntervalSinceNow_(0.1),
                )
            _auth_cached = result[0] if result else int(SFSpeechRecognizerAuthorizationStatusDenied)
        else:
            _auth_cached = int(status)


class MacOSSpeechError(Exception):
    pass


class AuthorizationError(MacOSSpeechError):
    pass


class RecognitionError(MacOSSpeechError):
    pass


class RecognitionTimeout(MacOSSpeechError):
    pass


class UnsupportedLanguage(MacOSSpeechError):
    pass


@dataclass
class RecognitionResult:
    text: str
    confidence: float = 1.0
    language: str = ""


def list_languages() -> list[str]:
    return list(SUPPORTED_LANGUAGES)


def transcribe_file(
    path: str,
    *,
    language: str | None = None,
    timeout: float = 15.0,
    on_device: bool = True,
    contextual_strings: list[str] | None = None,
) -> RecognitionResult:
    """辨識音訊檔案。

    Args:
        on_device: True（預設）強制本機辨識；False 允許雲端（中英混合
            語音辨識較穩，但需要網路）。
        contextual_strings: 提示詞清單（如 ["VS Code", "Python"]），
            提升專有名詞/術語的辨識命中率。
    """
    _ensure_authorized()
    if _auth_cached != int(SFSpeechRecognizerAuthorizationStatusAuthorized):
        raise AuthorizationError(f"Speech recognition not authorized (status={_auth_cached})")

    lang = language or "zh-CN"
    if lang not in SUPPORTED_LANGUAGES:
        raise UnsupportedLanguage(f"Language '{lang}' is not supported. Use list_languages() to see all {len(SUPPORTED_LANGUAGES)} options.")

    locale = Foundation.NSLocale.alloc().initWithLocaleIdentifier_(lang)
    url = Foundation.NSURL.fileURLWithPath_(os.path.abspath(path))

    if not os.path.exists(path):
        raise MacOSSpeechError(f"File not found: {path}")

    # SFSpeechRecognizer 的 callback 只會在「主執行緒」觸發，
    # 因此辨識與 RunLoop 都必須在主執行緒執行。
    recognizer = SFSpeechRecognizer.alloc().initWithLocale_(locale)
    if recognizer is None:
        raise MacOSSpeechError("Failed to create SFSpeechRecognizer")

    request = SFSpeechURLRecognitionRequest.alloc().initWithURL_(url)
    request.setRequiresOnDeviceRecognition_(on_device)
    if contextual_strings:
        request.setContextualStrings_(list(contextual_strings))
    request.setShouldReportPartialResults_(False)

    result_holder: list[RecognitionResult] = []
    error_holder: list[str] = []
    done = threading.Event()

    def handler(result, error):
        if error:
            error_holder.append(str(error))
            done.set()
            return
        if result is not None and result.isFinal():
            text = result.bestTranscription().formattedString()
            result_holder.append(RecognitionResult(text=text, confidence=1.0, language=lang))
            done.set()

    task = recognizer.recognitionTaskWithRequest_resultHandler_(request, handler)

    run_loop = Foundation.NSRunLoop.currentRunLoop()
    deadline = time.time() + timeout
    while time.time() < deadline and not done.is_set():
        run_loop.runMode_beforeDate_(
            Foundation.NSDefaultRunLoopMode,
            Foundation.NSDate.dateWithTimeIntervalSinceNow_(0.1),
        )
    task.cancel()

    if error_holder:
        raise RecognitionError(error_holder[0])
    if not result_holder:
        raise RecognitionTimeout(f"Recognition timed out after {timeout:.0f}s")
    return result_holder[0]


def transcribe_bytes(
    data: bytes,
    sample_rate: int,
    *,
    language: str | None = None,
    timeout: float = 15.0,
    on_device: bool = True,
    contextual_strings: list[str] | None = None,
) -> RecognitionResult:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        with wave.open(tmp, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(data)
        tmp.close()
        return transcribe_file(
            tmp.name,
            language=language,
            timeout=timeout,
            on_device=on_device,
            contextual_strings=contextual_strings,
        )
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def transcribe_file_engine(
    path: str,
    *,
    language: str,
    timeout: float = 15.0,
    on_device: bool = True,
    contextual_strings: list[str] | None = None,
) -> Future:
    """transcribe_file 的非阻塞版（供 HTTP server，走 EngineThread）。

    為什麼：uvicorn 的 event loop 在 main thread、阻塞等待會卡住所有請求，
    所以 ObjC 操作（建 task）序列化到 EngineThread 執行；callback 實際由
    server.py 的 _pump_main_runloop 在主執行緒 RunLoop 遞交。長工作
    （等 final）用「登記 EngineJob 輪詢 done event」模式，不占住 engine
    thread 的 tick，其他 session 的音訊不積壓。

    同步檢查（授權、語言、檔案存在）立即拋既有例外類別；
    成功時回 Future：
      - 結果 RecognitionResult
      - RecognitionError（引擎錯誤，含 no-speech）
      - RecognitionTimeout（逾時）
    """
    _ensure_authorized()
    if _auth_cached != int(SFSpeechRecognizerAuthorizationStatusAuthorized):
        raise AuthorizationError(f"Speech recognition not authorized (status={_auth_cached})")

    lang = language or "zh-CN"
    if lang not in SUPPORTED_LANGUAGES:
        raise UnsupportedLanguage(f"Language '{lang}' is not supported. Use list_languages() to see all {len(SUPPORTED_LANGUAGES)} options.")

    if not os.path.exists(path):
        raise MacOSSpeechError(f"File not found: {path}")

    from .engine_thread import EngineThread

    et = EngineThread.shared()
    locale = Foundation.NSLocale.alloc().initWithLocaleIdentifier_(lang)
    url = Foundation.NSURL.fileURLWithPath_(os.path.abspath(path))

    state: dict = {"done": threading.Event(), "result": None, "error": None}

    def _create():
        # 在 engine thread 建立 recognizer + task；callback 由主執行緒 RunLoop 遞交
        recognizer = SFSpeechRecognizer.alloc().initWithLocale_(locale)
        if recognizer is None:
            raise MacOSSpeechError("Failed to create SFSpeechRecognizer")
        request = SFSpeechURLRecognitionRequest.alloc().initWithURL_(url)
        request.setRequiresOnDeviceRecognition_(on_device)
        if contextual_strings:
            request.setContextualStrings_(list(contextual_strings))
        request.setShouldReportPartialResults_(False)

        def handler(result, error):
            if error:
                state["error"] = str(error)
                state["done"].set()
                return
            if result is not None and result.isFinal():
                text = result.bestTranscription().formattedString()
                state["result"] = RecognitionResult(text=text, confidence=1.0, language=lang)
                state["done"].set()

        state["task"] = recognizer.recognitionTaskWithRequest_resultHandler_(request, handler)

    et.run(_create, timeout=12.0).result(timeout=15.0)

    fut: Future = Future()

    class _WaitJob:
        """長工作：每 tick（engine thread）檢查 done event，完成即回 Future。"""

        def advance(self) -> bool:
            if not state["done"].is_set():
                return False
            try:
                state["task"].cancel()
            except Exception:
                pass
            try:
                if state["error"] is not None:
                    fut.set_exception(RecognitionError(state["error"]))
                elif state["result"] is not None:
                    fut.set_result(state["result"])
                else:
                    fut.set_exception(RecognitionError("No result available"))
            except InvalidStateError:
                pass  # 看門狗已設 RecognitionTimeout
            return True

    et.register(_WaitJob())

    def _expire():
        try:
            if not fut.done():
                fut.set_exception(RecognitionTimeout(f"Recognition timed out after {timeout:.0f}s"))
        except InvalidStateError:
            pass
        # 讓 _WaitJob 下一 tick 取消 task，逾時後不留殘存 task 佔引擎
        state["done"].set()

    timer = threading.Timer(timeout, _expire)
    timer.daemon = True
    timer.start()
    return fut
