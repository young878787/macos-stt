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


class StreamingRecognizer:
    def __init__(
        self,
        *,
        language: str = "zh-CN",
        on_partial: Callable[[str], None] | None = None,
        on_final: Callable[[RecognitionResult], None] | None = None,
        sample_rate: int = 16000,
    ):
        _ensure_authorized()
        if int(SFSpeechRecognizer.authorizationStatus()) != int(SFSpeechRecognizerAuthorizationStatusAuthorized):
            raise AuthorizationError("Speech recognition not authorized")

        self._language = language
        self._on_partial = on_partial
        self._on_final = on_final
        self._sample_rate = sample_rate

        locale = Foundation.NSLocale.alloc().initWithLocaleIdentifier_(language)
        self._recognizer = SFSpeechRecognizer.alloc().initWithLocale_(locale)
        if self._recognizer is None:
            raise MacOSSpeechError(f"Failed to create SFSpeechRecognizer for '{language}'")

        self._request: SFSpeechAudioBufferRecognitionRequest | None = None
        self._started = False
        self._finished = False
        self._partials: list[str] = []
        self._final_result: RecognitionResult | None = None
        self._final_error: str | None = None
        self._done = threading.Event()
        self._feed_queue = queue.Queue()

    def start(self):
        if self._started:
            return
        self._request = SFSpeechAudioBufferRecognitionRequest.alloc().init()
        self._request.setRequiresOnDeviceRecognition_(True)
        self._request.setShouldReportPartialResults_(self._on_partial is not None)

        def handler(result, error):
            if error:
                self._final_error = str(error)
                self._done.set()
                return
            if result is None:
                return
            text = result.bestTranscription().formattedString()
            if result.isFinal():
                self._final_result = RecognitionResult(text=text, confidence=1.0, language=self._language)
                if self._on_final:
                    self._on_final(self._final_result)
                self._done.set()
            else:
                self._partials.append(text)
                if self._on_partial:
                    self._on_partial(text)

        def _start():
            self._task = self._recognizer.recognitionTaskWithRequest_resultHandler_(self._request, handler)

        import dispatch
        dispatch.DispatchQueue.main.sync_(_start)

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
        import dispatch
        from AVFoundation import AVAudioPCMBuffer, AVAudioFormat

        data = chunk

        def _do_append():
            fmt = AVAudioFormat.alloc().initWithCommonFormatSampleRate_channels_interleaved_(
                1, float(self._sample_rate), 1, True
            )
            if fmt is None:
                return
            frame_count = len(data) // 2
            buf = AVAudioPCMBuffer.alloc().initWithPCMFormat_frameCapacity_(fmt, frame_count)
            if buf is None:
                return
            buf.setFrameLength_(frame_count)
            ptr = buf.int16ChannelData()[0]
            ctypes.memmove(ptr, data, len(data))
            self._request.appendAudioPCMBuffer_(buf)

        dispatch.DispatchQueue.main.sync_(_do_append)

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

        import dispatch
        def _end():
            if self._request:
                self._request.endAudio()
        dispatch.DispatchQueue.main.sync_(_end)

        if not self._done.wait(timeout=timeout):
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
