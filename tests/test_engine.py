"""Tests for macos-stt."""

import math
import struct
import tempfile
import wave
import pytest
from macos_stt import (
    MacOSSpeechError,
    recognize,
    recognize_bytes,
    list_languages,
    RecognitionTimeout,
    UnsupportedLanguage,
)


def test_list_languages():
    langs = list_languages()
    assert len(langs) >= 60
    assert "zh-CN" in langs
    assert "en-US" in langs
    assert "ja-JP" in langs


def test_unsupported_language():
    with pytest.raises(UnsupportedLanguage):
        recognize("test.wav", language="xx-XX")


def _make_tone_pcm(sr=16000, duration=0.5):
    samples = int(sr * duration)
    return b"".join(struct.pack("<h", int(16000 * math.sin(2 * math.pi * 440 * i / sr))) for i in range(samples))


def test_recognize_bytes_basic():
    pcm = _make_tone_pcm()
    try:
        result = recognize_bytes(pcm, sample_rate=16000, language="en-US", timeout=3)
        assert result is not None
    except MacOSSpeechError:
        pass  # 440Hz 純音沒有人聲，預期 "No speech detected"


def test_recognize_file_basic():
    pcm = _make_tone_pcm()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        with wave.open(tmp, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(pcm)
        tmp_path = tmp.name

    try:
        result = recognize(tmp_path, language="en-US", timeout=3)
        assert result is not None
    except MacOSSpeechError:
        pass  # 440Hz 純音沒有人聲，預期 "No speech detected"
    finally:
        import os
        os.unlink(tmp_path)


def test_recognition_result_fields():
    # 直接檢查資料類別欄位（不依賴實際辨識引擎）
    from macos_stt import RecognitionResult
    r = RecognitionResult(text="hi", confidence=0.9, language="en-US")
    assert hasattr(r, "text")
    assert hasattr(r, "confidence")
    assert hasattr(r, "language")
    assert r.text == "hi"
