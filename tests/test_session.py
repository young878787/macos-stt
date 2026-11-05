"""session.py 自動測試（mock recognizer，不觸發真實 SFSpeechRecognizer / TCC）。

跑法：uv run pytest tests/test_session.py -q
"""

import struct
import time

import pytest

from macos_stt import session as session_mod
from macos_stt.engine import MacOSSpeechError
from macos_stt.session import (
    Session,
    SessionConfig,
    SessionLimitExceeded,
    SessionManager,
)
from macos_stt.vocab import VocabularyStore

FRAME_SAMPLES = 6400  # 16kHz × 0.2s


def pcm16(amplitude: int, n: int = FRAME_SAMPLES) -> bytes:
    return struct.pack(f"<{n}h", *([amplitude] * n))


class MockRecognizer:
    """模仿 StreamingRecognizer 介面（純 Python，不碰引擎）。

    餵滿 32000 bytes（~1s 音訊）後模擬引擎出 partial，呼叫 on_partial。
    """

    created: list["MockRecognizer"] = []

    def __init__(self, contextual_strings=None, on_partial=None, on_utterance=None,
                 on_final=None, **_kw):
        self.contextual_strings = contextual_strings
        self._on_partial = on_partial
        self.feed_count = 0
        self.feed_bytes = 0
        self._last_partial = ""
        self._started = False
        self._closed = False
        MockRecognizer.created.append(self)

    def start(self):
        self._started = True

    def feed(self, chunk):
        self.feed_count += 1
        self.feed_bytes += len(chunk)
        if self.feed_bytes >= 32000 and not self._last_partial:
            self._last_partial = "mock-partial"
            if self._on_partial:
                self._on_partial(self._last_partial)

    def finish(self, timeout=15.0):
        from macos_stt.engine import RecognitionResult

        self._closed = True
        return RecognitionResult(
            text=self._last_partial, confidence=1.0, language="zh-TW"
        )


@pytest.fixture(autouse=True)
def _reset_mock():
    MockRecognizer.created = []
    yield


def test_feed_and_close():
    events: list[dict] = []
    s = Session(SessionConfig(), on_event=events.append,
                recognizer_factory=MockRecognizer)
    for _ in range(10):
        s.feed(pcm16(8000))
        time.sleep(0.005)
    assert s._rec.feed_count == 10
    assert s._rec.feed_bytes == 10 * FRAME_SAMPLES * 2
    s.close()
    with pytest.raises(MacOSSpeechError):
        s.feed(pcm16(100))
    # close 是 idempotent
    s.close()


def test_silence_flush_safety_net(monkeypatch):
    # 加速：靜音 0.1s 即判定「不在說話」，安全網 0.3s 觸發
    monkeypatch.setattr(session_mod, "SILENCE_PAUSE", 0.1)
    events: list[dict] = []
    cfg = SessionConfig(vad_flush_seconds=0.3)
    s = Session(cfg, on_event=events.append, recognizer_factory=MockRecognizer)
    rec = s._rec

    # 「說話」：loud/quiet 交錯（模擬有字間空隙的語音——VAD 的底噪估計
    # 需要低 RMS 塊才能保持低門檻；連續常幅值音訊會污染 10 分位數）
    for i in range(12):
        s.feed(pcm16(8000 if i % 2 == 0 else 10))
        time.sleep(0.05)
    assert s.speaking or s._session_had_speech
    # mock 餵滿 32000B 後自動出 partial
    assert rec._last_partial == "mock-partial"

    # 2s 靜音：安全網應觸發 flush + 重開
    for _ in range(24):
        s.feed(pcm16(0))
        time.sleep(0.05)

    types = [e["type"] for e in events]
    assert "session_restarted" in types
    finals = [e for e in events if e["type"] == "final"]
    assert any(e["text"] == "mock-partial" for e in finals)
    assert len(MockRecognizer.created) == 2  # flush 後重開
    # 新 session 的 partial 應為空
    assert MockRecognizer.created[-1].feed_count > 0
    s.close()


def test_session_limit_restart(monkeypatch):
    monkeypatch.setattr(session_mod, "SILENCE_PAUSE", 0.1)
    events: list[dict] = []
    cfg = SessionConfig(max_session_seconds=0.4)
    s = Session(cfg, on_event=events.append, recognizer_factory=MockRecognizer)
    for _ in range(24):  # 1.2s > 0.4s 上限
        s.feed(pcm16(8000))
        time.sleep(0.05)
    assert len(MockRecognizer.created) >= 2
    assert "session_restarted" in [e["type"] for e in events]
    s.close()


def test_manager_limit():
    mgr = SessionManager(max_sessions=3, recognizer_factory=MockRecognizer)
    s1 = mgr.acquire(SessionConfig())
    s2 = mgr.acquire(SessionConfig())
    s3 = mgr.acquire(SessionConfig())
    with pytest.raises(SessionLimitExceeded):
        mgr.acquire(SessionConfig())
    assert mgr.stats == {"active": 3, "max": 3}
    mgr.release(s2)
    assert mgr.stats == {"active": 2, "max": 3}
    s4 = mgr.acquire(SessionConfig())  # 釋放後可再建
    assert mgr.stats == {"active": 3, "max": 3}
    for s in (s1, s3, s4):
        mgr.release(s)
    # release 是 idempotent
    mgr.release(s2)


def test_manager_vocab_merge():
    v = VocabularyStore(words=["Python", "Git"])
    mgr = SessionManager(max_sessions=3, vocabulary=v,
                         recognizer_factory=MockRecognizer)
    s = mgr.acquire(SessionConfig(hints=["商智中心"]))
    assert s._rec.contextual_strings == ["Python", "Git", "商智中心"]
    mgr.release(s)


def test_rms_int16():
    from macos_stt.session import _rms_int16

    # 全 0 → 0
    assert _rms_int16(pcm16(0)) == 0.0
    # 常數幅值 8000 → 8000/32768
    rms = _rms_int16(pcm16(8000))
    assert abs(rms - 8000 / 32768) < 1e-6


def test_slot_rotation_chain_no_deadlock(monkeypatch):
    """單槽輪替鏈：s0 flush → 輪給 s1（重放後即 flush）→ 輪給 s2（重放後
    即 flush，無人可再輪替 → 自己重開）。曾因 _become_active 持非重入鎖
    又在內部重入而自我死鎖（final 已送出，但 feed thread 永久卡住）。"""
    import threading

    monkeypatch.setattr(session_mod, "FLUSH_PRE_DELAY", 0)
    monkeypatch.setattr(session_mod, "RESTART_DELAY", 0)
    monkeypatch.setattr(session_mod, "SILENCE_PAUSE", 0.1)
    mgr = SessionManager(max_sessions=3, recognizer_factory=MockRecognizer)
    mgr._slot = True  # mock 模擬真引擎的單槽行為
    evs: list[list[dict]] = [[], [], []]
    cfg = SessionConfig(vad_flush_seconds=0.2)
    ss = [mgr.acquire(cfg, evs[i].append) for i in range(3)]
    assert [s._waiting for s in ss] == [False, True, True]

    for s in (ss[1], ss[2], ss[0]):  # s1/s2 為等待者：幀進緩衝
        for _ in range(3):
            s.feed(pcm16(8000))
    time.sleep(0.25)

    t = threading.Thread(target=ss[0].feed, args=(pcm16(0),), daemon=True)
    t.start()
    t.join(timeout=5)
    assert not t.is_alive(), "輪替鏈死鎖"
    for i in range(3):
        assert any(e["type"] == "final" for e in evs[i]), f"s{i} 沒收到 final"
    # 單槽不變式：同一時間只有一個 session 持有 recognizer
    assert sum(1 for s in ss if s._rec is not None) == 1


def test_acquire_slow_start_does_not_block_stats():
    """recognizer 啟動很慢時，acquire 不得持 manager 鎖（會擋住 /health）。"""
    import threading

    class SlowRec(MockRecognizer):
        def start(self):
            time.sleep(0.6)
            super().start()

    mgr = SessionManager(max_sessions=3, recognizer_factory=SlowRec)
    t = threading.Thread(target=mgr.acquire, args=(SessionConfig(),), daemon=True)
    t.start()
    time.sleep(0.1)
    t0 = time.monotonic()
    assert mgr.stats["active"] == 1  # 名額已佔
    assert time.monotonic() - t0 < 0.2
    t.join(timeout=3)


def test_acquire_start_failure_releases_quota():
    class BadRec(MockRecognizer):
        def start(self):
            raise MacOSSpeechError("boom")

    mgr = SessionManager(max_sessions=1, recognizer_factory=BadRec)
    with pytest.raises(MacOSSpeechError):
        mgr.acquire(SessionConfig())
    assert mgr.stats["active"] == 0
    mgr._slot = True
    with pytest.raises(MacOSSpeechError):
        mgr.acquire(SessionConfig())
    assert mgr.stats["active"] == 0 and mgr._active_id is None


def test_rest_engine_lease_excludes_streaming():
    """REST 借用引擎與串流 session 互斥（單槽）。"""
    mgr = SessionManager(max_sessions=3, recognizer_factory=MockRecognizer)
    mgr._slot = True
    # REST 先借：第二個 REST 被拒；此時新 session 為等待者，且 feed 搶不到槽
    assert mgr.try_acquire_engine() is True
    assert mgr.try_acquire_engine() is False
    s = mgr.acquire(SessionConfig())
    assert s._waiting and s._rec is None
    s.feed(pcm16(8000))
    assert s._rec is None
    mgr.release_engine()
    s.feed(pcm16(8000))  # REST 結束後下一幀即拿到槽
    assert s._rec is not None
    # 串流持槽時 REST 被拒
    assert mgr.try_acquire_engine() is False
    mgr.release(s)
    assert mgr.try_acquire_engine() is True
    mgr.release_engine()


def _talk(s, frames=6):
    for _ in range(frames):
        s.feed(pcm16(8000))
    assert s._rec._last_partial == "mock-partial"  # 6 幀 × 12800B 已超過 32000B


def test_flush_emits_final_with_timing_and_restarts():
    events: list[dict] = []
    s = Session(SessionConfig(flush_pre_delay=0.05), on_event=events.append, recognizer_factory=MockRecognizer)
    _talk(s)
    s.flush()

    finals = [e for e in events if e["type"] == "final"]
    assert len(finals) == 1 and finals[0]["text"] == "mock-partial"
    t = finals[0]["timing"]
    assert t["reason"] == "client_flush"
    assert 40 <= t["pre_delay_ms"] < 150 and t["finish_ms"] >= 0
    assert "session_restarted" in [e["type"] for e in events]
    assert len(MockRecognizer.created) == 2  # flush 後重開，之後還能繼續 feed
    s.feed(pcm16(8000))
    assert MockRecognizer.created[-1].feed_count == 1
    s.close()


def test_flush_without_speech_or_text_does_nothing():
    events: list[dict] = []
    s = Session(SessionConfig(flush_pre_delay=0.05), on_event=events.append, recognizer_factory=MockRecognizer)
    s.feed(pcm16(0))
    s.flush()
    assert events == [] and len(MockRecognizer.created) == 1
    s.close()
    s.flush()  # 關閉後不拋錯


def test_flush_pre_delay_defaults_to_legacy_value_and_is_configurable():
    assert SessionConfig().flush_pre_delay == 0.4
    events: list[dict] = []
    s = Session(SessionConfig(flush_pre_delay=0.0), on_event=events.append, recognizer_factory=MockRecognizer)
    _talk(s)
    s.flush()
    assert [e for e in events if e["type"] == "final"][0]["timing"]["pre_delay_ms"] < 30
    s.close()


def test_server_vad_flush_final_reports_its_reason(monkeypatch):
    monkeypatch.setattr(session_mod, "SILENCE_PAUSE", 0.1)
    events: list[dict] = []
    s = Session(SessionConfig(vad_flush_seconds=0.3, flush_pre_delay=0.05), on_event=events.append,
                recognizer_factory=MockRecognizer)
    for i in range(12):
        s.feed(pcm16(8000 if i % 2 == 0 else 10))
        time.sleep(0.05)
    for _ in range(24):
        s.feed(pcm16(0))
        time.sleep(0.05)
    finals = [e for e in events if e["type"] == "final"]
    assert finals and finals[0]["timing"]["reason"] == "server_vad"
    s.close()
