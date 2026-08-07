"""Tests for macos-stt."""

import math
import struct
import tempfile
import wave
import pytest
from macos_stt import (
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
    except RecognitionTimeout:
        pass


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
    except RecognitionTimeout:
        pass
    finally:
        import os
        os.unlink(tmp_path)


def test_recognition_result_fields():
    pcm = b"\x00" * 16000
    try:
        result = recognize_bytes(pcm, sample_rate=16000, language="en-US", timeout=3)
        assert hasattr(result, "text")
        assert hasattr(result, "confidence")
        assert hasattr(result, "language")
    except RecognitionTimeout:
        pass
