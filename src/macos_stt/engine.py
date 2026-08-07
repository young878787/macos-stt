"""macOS SFSpeechRecognizer 核心引擎。

SFSpeechRecognizer 的回调必须由主线程 RunLoop 派发。
用子线程 + CFRunLoop pump + threading.Event 同步，保证线程安全。
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
import wave
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


def transcribe_file(path: str, *, language: str | None = None, timeout: float = 15.0) -> RecognitionResult:
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

    result_holder: list[RecognitionResult] = []
    error_holder: list[str] = []
    done = threading.Event()

    def worker():
        recognizer = SFSpeechRecognizer.alloc().initWithLocale_(locale)
        if recognizer is None:
            error_holder.append("Failed to create SFSpeechRecognizer")
            done.set()
            return

        request = SFSpeechURLRecognitionRequest.alloc().initWithURL_(url)
        request.setRequiresOnDeviceRecognition_(True)
        request.setShouldReportPartialResults_(False)

        def handler(result, error):
            if error:
                error_holder.append(str(error))
                done.set()
                return
            if result is not None and result.isFinal():
                text = result.bestTranscription().formattedString()
                result_holder.append(RecognitionResult(text=text, confidence=1.0, language=lang))
                done.set()

        recognizer.recognitionTaskWithRequest_resultHandler_(request, handler)

        run_loop = Foundation.NSRunLoop.currentRunLoop()
        deadline = time.time() + timeout
        while time.time() < deadline and not done.is_set():
            run_loop.runMode_beforeDate_(
                Foundation.NSDefaultRunLoopMode,
                Foundation.NSDate.dateWithTimeIntervalSinceNow_(0.1),
            )

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    done.wait(timeout=timeout + 1.0)
    thread.join(timeout=1.0)

    if error_holder:
        raise RecognitionError(error_holder[0])
    if not result_holder:
        raise RecognitionTimeout(f"Recognition timed out after {timeout:.0f}s")
    return result_holder[0]


def transcribe_bytes(data: bytes, sample_rate: int, *, language: str | None = None, timeout: float = 15.0) -> RecognitionResult:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        with wave.open(tmp, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(data)
        tmp.close()
        return transcribe_file(tmp.name, language=language, timeout=timeout)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
