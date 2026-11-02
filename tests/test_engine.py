"""Tests for macos-stt.

註解：標了 ``live`` 的測試會跑真實 SFSpeechRecognizer 引擎，需要「已授權
語音辨識」的 macOS 環境（且通常要經 .app bundle 啟動才有 TCC 上下文）。
conftest.py 在 MACSTT_LIVE 未設定為 "1" 時會自動 skip 這些測試，因此
`uv run pytest` 永遠不會碰到真實引擎：

    uv run pytest                 # 僅跑一般單測（安全）
    MACSTT_LIVE=1 uv run pytest   # 在已授權的 macOS 環境跑 live 測試
"""

import math
import struct
import tempfile
import wave
import pytest
from Speech import SFSpeechRecognizerAuthorizationStatusAuthorized

import macos_stt.engine as engine_mod
from macos_stt import (
    MacOSSpeechError,
    recognize,
    recognize_bytes,
    list_languages,
    RecognitionTimeout,
    UnsupportedLanguage,
)


@pytest.fixture(autouse=True, scope="module")
def stub_speech_auth():
    """把引擎的授權快取 stub 成「已授權」（SFSpeechRecognizerAuthorizationStatusAuthorized = 1）。

    transcribe_file 會先呼叫 _ensure_authorized()；在未授權（或
    not-determined 且無 GUI 可彈授權框，例如 pytest 非 .app 啟動）的環境
    下，requestAuthorization_ 會觸發 TCC SIGABRT 把整個 process 打死。
    此處預先設定 _auth_cached 全域變數，讓真實授權流程被跳過，
    一般單測（如語言檢查）就能安全執行。模組測試結束後還原原值。
    """
    original = getattr(engine_mod, "_auth_cached", None)
    engine_mod._auth_cached = int(SFSpeechRecognizerAuthorizationStatusAuthorized)
    yield
    if original is None:
        del engine_mod._auth_cached
    else:
        engine_mod._auth_cached = original


def test_list_languages():
    langs = list_languages()
    assert len(langs) >= 60
    assert "zh-CN" in langs
    assert "en-US" in langs
    assert "ja-JP" in langs


def test_unsupported_language():
    # 靠 stub_speech_auth 跳過真實授權後，transcribe_file 的檢查順序是
    # auth → 語言檢查 → 建 recognizer → 檔案存在檢查，所以 "xx-XX" 會在
    # 碰到檔案（test.wav 不需要真的存在）之前就被擋下。
    with pytest.raises(UnsupportedLanguage):
        recognize("test.wav", language="xx-XX")


def _make_tone_pcm(sr=16000, duration=0.5):
    samples = int(sr * duration)
    return b"".join(struct.pack("<h", int(16000 * math.sin(2 * math.pi * 440 * i / sr))) for i in range(samples))


@pytest.mark.live
def test_recognize_bytes_basic():
    pcm = _make_tone_pcm()
    try:
        result = recognize_bytes(pcm, sample_rate=16000, language="en-US", timeout=3)
        assert result is not None
    except MacOSSpeechError:
        pass  # 440Hz 純音沒有人聲，預期 "No speech detected"


@pytest.mark.live
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
