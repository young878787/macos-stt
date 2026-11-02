"""EngineThread 測試。

全部不呼叫 SFSpeechRecognizer、不需要語音授權即可安全執行：
只驗證「工作確實跑在 engine thread 上、Future 語義、timeout、單例、
短工作不被長工作阻塞」這些機制行為。
"""

import threading
import time

import pytest

from macos_stt.engine_thread import EngineThread


@pytest.fixture(autouse=True)
def _clean_engine_thread():
    """每個測試從乾淨狀態開始；測試結束（含失敗）清除 singleton，
    避免 daemon thread 殘留影響後續測試。"""
    yield
    EngineThread._reset()


def test_runs_on_engine_thread():
    et = EngineThread.shared()
    names = et.run(lambda: threading.current_thread().name).result(timeout=5)
    assert names == "macstt-engine"
    assert names != threading.main_thread().name


def test_future_result_and_exception():
    et = EngineThread.shared()
    assert et.run(lambda: 41 + 1).result(timeout=5) == 42

    def boom():
        raise ValueError("boom")

    fut = et.run(boom)
    with pytest.raises(ValueError, match="boom"):
        fut.result(timeout=5)


def test_timeout():
    et = EngineThread.shared()
    start = time.monotonic()
    fut = et.run(lambda: time.sleep(0.3), timeout=0.05)
    with pytest.raises(TimeoutError):
        fut.result(timeout=1)
    # 逾時應在 timeout 附近（+ 餘量）就拋出，而非等工作跑完
    assert time.monotonic() - start < 0.1
    time.sleep(0.4)  # 確認逾時後 engine thread 沒有卡死


def test_singleton():
    a = EngineThread.shared()
    b = EngineThread.shared()
    assert a is b
    assert a._thread.is_alive()


def test_stop_clears_singleton():
    et = EngineThread.shared()
    et.stop()
    EngineThread._reset()
    et2 = EngineThread.shared()
    assert et2 is not et
    assert et2._thread.is_alive()


class _FakeLongJob:
    """前 20 次 advance 回 False、每次睡 0.01s；之後回 True。"""

    def __init__(self):
        self.calls = 0

    def advance(self) -> bool:
        self.calls += 1
        time.sleep(0.01)
        return self.calls >= 20


def test_short_jobs_while_long_job_running():
    et = EngineThread.shared()
    job = _FakeLongJob()
    et.register(job)

    futures = [et.run(lambda i=i: i) for i in range(10)]
    results = []
    start = time.monotonic()
    for f in futures:
        results.append(f.result(timeout=5))
    elapsed = time.monotonic() - start

    assert results == list(range(10))
    # 短工作不能被長工作阻塞（長工作每圈 ~10ms，10 個短工作應 << 長工作總時長）
    assert elapsed < 0.1

    # 長工作最終完成並被移除（tick loop 每圈 ~20ms + 20 次 × 10ms advance）
    deadline = time.monotonic() + 5
    while et.active_jobs and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not et.active_jobs
    assert job.calls >= 20


def test_idle_does_not_spin():
    """閒置時 engine thread 不可空轉（RunLoop 無 source 時 runMode 立即返回，
    曾因此 100% CPU 餓死 uvicorn）。"""
    EngineThread.shared()
    time.sleep(0.2)  # 讓 thread 進入穩態
    cpu0 = time.process_time()
    time.sleep(1.0)
    assert time.process_time() - cpu0 < 0.3
