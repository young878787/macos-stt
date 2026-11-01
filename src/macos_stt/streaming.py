"""Streaming speech recognition via SFSpeechAudioBufferRecognitionRequest."""

from __future__ import annotations

import queue
import threading
import time
from typing import Callable

import Foundation
from Speech import (
    SFSpeechAudioBufferRecognitionRequest,
    SFSpeechRecognizer,
    SFSpeechRecognizerAuthorizationStatusAuthorized,
)

from .engine import (
    AuthorizationError,
    MacOSSpeechError,
    RecognitionResult,
    RecognitionTimeout,
    _ensure_authorized,
)


def _on_main_thread() -> bool:
    import dispatch
    cur = dispatch.dispatch_get_current_queue()
    if cur is not None:
        label = dispatch.dispatch_queue_get_label(cur)
        if isinstance(label, bytes):
            label = label.decode("utf-8", "replace")
        if (label or "").endswith("com.apple.main-thread"):
            return True
    # 主執行緒通常不在任何 dispatch 佇列，用 NSThread 判斷
    import Foundation
    return bool(Foundation.NSThread.isMainThread())


def _main_sync(block):
    """在 main 執行緒的執行脈絡同步執行 block。

    pyobjc 的 dispatch 模組只提供 C API（沒有 ObjC 風格 DispatchQueue）。
    注意：dispatch_sync 從主執行緒呼叫 main queue 會死鎖，
    因此若已在主執行緒就直接執行，否則 sync 過去。
    """
    import dispatch
    if _on_main_thread():
        block()
    else:
        dispatch.dispatch_sync(dispatch.dispatch_get_main_queue(), block)


class StreamingRecognizer:
    def __init__(
        self,
        *,
        language: str = "zh-CN",
        on_partial: Callable[[str], None] | None = None,
        on_final: Callable[[RecognitionResult], None] | None = None,
        sample_rate: int = 16000,
        on_device: bool = True,
        contextual_strings: list[str] | None = None,
        detect_multiple_utterances: bool = False,
        on_utterance: Callable[[str], None] | None = None,
    ):
        """建立串流辨識器。

        Args:
            on_device: True（預設）強制本機辨識；False 允許雲端
                （中英混合語音辨識較穩，但需要網路）。
            contextual_strings: 提示詞清單（如 ["VS Code", "Python"]），
                提升專有名詞/術語的辨識命中率。
            detect_multiple_utterances: 讓引擎自己做端點偵測（句子斷點
                由語言模型+韵律判斷，而非外部靜音計時器）。引擎判定一句
                結束時會把 partial 重設；此時 on_utterance 收到該句。
            on_utterance: 引擎自斷句回呼（仅在 detect_multiple_utterances
                開啟時有意義）。注意: 開多句模式下 finish() 回傳的
                final 只覆蓋最後一句，之前每句以 on_utterance 送出。
        """
        _ensure_authorized()
        if int(SFSpeechRecognizer.authorizationStatus()) != int(SFSpeechRecognizerAuthorizationStatusAuthorized):
            raise AuthorizationError("Speech recognition not authorized")

        self._language = language
        self._on_partial = on_partial
        self._on_final = on_final
        self._sample_rate = sample_rate
        self._on_device = on_device
        self._contextual_strings = list(contextual_strings) if contextual_strings else None
        self._detect_multiple = detect_multiple_utterances
        self._on_utterance = on_utterance

        locale = Foundation.NSLocale.alloc().initWithLocaleIdentifier_(language)
        self._recognizer = SFSpeechRecognizer.alloc().initWithLocale_(locale)
        if self._recognizer is None:
            raise MacOSSpeechError(f"Failed to create SFSpeechRecognizer for '{language}'")
        if on_device and not self._recognizer.supportsOnDeviceRecognition():
            raise MacOSSpeechError(
                f"On-device recognition not supported for '{language}'. "
                "Use on_device=False to allow server-based recognition."
            )

        self._request: SFSpeechAudioBufferRecognitionRequest | None = None
        self._started = False
        self._finished = False
        self._partials: list[str] = []
        self._last_partial: str | None = None
        self._final_result: RecognitionResult | None = None
        self._final_error: str | None = None
        self._done = threading.Event()
        self._feed_queue = queue.Queue()

    def start(self):
        if self._started:
            return
        self._request = SFSpeechAudioBufferRecognitionRequest.alloc().init()
        self._request.setRequiresOnDeviceRecognition_(self._on_device)
        if self._contextual_strings:
            self._request.setContextualStrings_(self._contextual_strings)
        self._request.setShouldReportPartialResults_(self._on_partial is not None)
        if self._detect_multiple:
            self._request.setDetectMultipleUtterances_(True)

        def handler(result, error):
            if error:
                self._final_error = str(error)
                self._done.set()
                return
            if result is None:
                return
            text = result.bestTranscription().formattedString()
            if result.isFinal():
                # 多句模式: final 只覆蓋「最後一句」；前面每句已在
                # partial 被引擎重設時透過 on_utterance 送出。
                self._final_result = RecognitionResult(text=text, confidence=1.0, language=self._language)
                if self._on_final:
                    self._on_final(self._final_result)
                self._last_partial = None
                self._done.set()
            else:
                self._partials.append(text)
                # 引擎自斷句偵測: partial 被引擎「重設」（新句從頭開始）。
                # 判據: 新 partial 顯著比上一個短（< 60%）且上一個夠長
                # （>=6 字）—— 排除引擎小幅「更正」partial 的誤觸發。
                if (self._detect_multiple and self._on_utterance
                        and self._last_partial
                        and len(self._last_partial) >= 6
                        and len(text) * 5 < len(self._last_partial) * 3):
                    self._on_utterance(self._last_partial)
                self._last_partial = text
                if self._on_partial:
                    self._on_partial(text)

        def _start():
            self._task = self._recognizer.recognitionTaskWithRequest_resultHandler_(self._request, handler)

        _main_sync(_start)

        self._feed_thread = threading.Thread(target=self._feed_loop, daemon=True)
        self._feed_thread.start()
        self._started = True

    def _feed_loop(self):
        while not self._finished:
            try:
                chunk = self._feed_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if chunk is None:
                break
            self._append_audio_chunk(chunk)

    def _append_audio_chunk(self, chunk: bytes):
        import ctypes
        from AVFoundation import AVAudioPCMBuffer, AVAudioFormat

        data = chunk

        def _do_append():
            # common format 3 = kAudioFormatLinearPCM 16-bit int
            fmt = AVAudioFormat.alloc().initWithCommonFormat_sampleRate_channels_interleaved_(
                3, float(self._sample_rate), 1, True
            )
            if fmt is None:
                return
            frame_count = len(data) // 2
            buf = AVAudioPCMBuffer.alloc().initWithPCMFormat_frameCapacity_(fmt, frame_count)
            if buf is None:
                return
            buf.setFrameLength_(frame_count)
            # int16ChannelData()[0] 是 objc.varlist，as_buffer(count) 取 memoryview
            view = buf.int16ChannelData()[0].as_buffer(frame_count)
            target = (ctypes.c_char * len(view)).from_buffer(view)
            ctypes.memmove(ctypes.addressof(target), data, len(data))
            self._request.appendAudioPCMBuffer_(buf)

        _main_sync(_do_append)

    def feed(self, chunk: bytes | memoryview):
        if not self._started:
            raise MacOSSpeechError("call start() before feed()")
        if self._finished:
            raise MacOSSpeechError("already finished")
        self._feed_queue.put(bytes(chunk))

    def finish(self, timeout: float = 15.0) -> RecognitionResult:
        if self._finished:
            if self._final_result:
                return self._final_result
            raise MacOSSpeechError("No result available")

        self._feed_queue.put(None)
        self._finished = True

        def _end():
            if self._request:
                self._request.endAudio()
        _main_sync(_end)

        # 最終結果的 callback 綁在主執行緒: 若在執行主執行緒就要 pump
        # NSRunLoop 才能收到；在子執行緒則直接 wait 即可。
        if _on_main_thread():
            import Foundation
            rl = Foundation.NSRunLoop.currentRunLoop()
            deadline = time.time() + timeout
            while time.time() < deadline and not self._done.is_set():
                rl.runMode_beforeDate_(
                    Foundation.NSDefaultRunLoopMode,
                    Foundation.NSDate.dateWithTimeIntervalSinceNow_(0.1),
                )
        else:
            if not self._done.wait(timeout=timeout):
                pass

        if not self._done.is_set():
            raise RecognitionTimeout(f"Streaming recognition timed out after {timeout:.0f}s")

        if self._final_error:
            raise MacOSSpeechError(self._final_error)
        if self._final_result:
            return self._final_result
        text = "".join(self._partials)
        return RecognitionResult(text=text, confidence=0.5, language=self._language)

    @property
    def partials(self) -> list[str]:
        return list(self._partials)
