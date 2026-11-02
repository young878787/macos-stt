"""streaming.py 純邏輯單元測試（不碰真實語音辨識引擎）。

- 不 SFSpeechRecognizer 實例化、不觸發 TCC 授權：import
  macos_stt.streaming 只載入 Foundation/Speech framework 的參照，
  不會查授權狀態；授權檢查在 __init__ 內，本測試一律不建 instance。
- 測 `_dispatch(use_engine, on_main, block)` 的三種分派路徑。
  唯一的真實 ObjC 呼叫是「本地模式」路的 dispatch.dispatch_sync 到
  main dispatch queue（與 live_zh.py 主迴圈同一機制）與 EngineThread
  （純 thread/RunLoop 機制，非語音辨識引擎）。
- on_main 偵測以 monkeypatch 成 fake True/False：on_main=True 路徑用
  fake 驗證（不需真的在主執行緒跑 dispatch）；on_main=False 路徑由
  子執行緒呼叫，主執行緒同時 pump RunLoop 等待（與 live_zh.py 主迴圈
  一致——不 pump 時 main queue 可能不被服務，dispatch_sync 會死鎖）。
"""

import inspect
import queue as queue_mod
import threading
import time

import Foundation
import pytest

import macos_stt.streaming as streaming
from macos_stt.engine import RecognitionResult
from macos_stt.engine_thread import EngineThread
from macos_stt.streaming import StreamingRecognizer


@pytest.fixture(autouse=True)
def _clean_engine_thread():
    """前後皆清除 EngineThread singleton，避免 daemon thread 殘留影響
    後續測試（同 test_engine_thread.py 的清理策略）。"""
    EngineThread._reset()
    yield
    EngineThread._reset()


# ---- _dispatch：本地模式（use_engine=False，現行行為） ----

def test_dispatch_local_on_main_runs_in_caller(monkeypatch):
    """use_engine=False、on_main=True（fake）→ block 直接在呼叫端執行，
    不走任何 dispatch（main thread 上 dispatch_sync 到 main queue 會死鎖）。"""
    monkeypatch.setattr(streaming, "_on_main_thread", lambda: True)
    main_name = threading.main_thread().name
    ran = []
    streaming._dispatch(False, streaming._on_main_thread(),
                        lambda: ran.append(threading.current_thread().name))
    assert ran == [main_name]


def test_dispatch_local_off_main_goes_through_main_queue(monkeypatch):
    """use_engine=False、on_main=False（fake）→ block 經 dispatch_sync 送
    main dispatch queue、在 main thread 執行。

    由子執行緒呼叫；主執行緒（pytest）以 pump RunLoop 的方式等待
    （與 live_zh.py 主迴圈一致）——不 pump 時 main queue 可能不被服務，
    子執行緒的 dispatch_sync 會死鎖。
    """
    monkeypatch.setattr(streaming, "_on_main_thread", lambda: False)
    ran = []
    done = threading.Event()

    def worker():
        try:
            streaming._dispatch(False, streaming._on_main_thread(),
                                lambda: ran.append(threading.current_thread().name))
        finally:
            done.set()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    rl = Foundation.NSRunLoop.currentRunLoop()
    deadline = time.monotonic() + 5.0
    while not done.is_set() and time.monotonic() < deadline:
        rl.runMode_beforeDate_(
            Foundation.NSDefaultRunLoopMode,
            Foundation.NSDate.dateWithTimeIntervalSinceNow_(0.05),
        )
    t.join(timeout=5.0)

    assert done.is_set(), "子執行緒 dispatch_sync 到 main queue 未完成（死鎖？）"
    assert ran == [threading.main_thread().name]


# ---- _dispatch：engine 模式（use_engine=True，伺服器環境） ----

def test_dispatch_engine_runs_on_engine_thread():
    """use_engine=True → block 在 'macstt-engine' thread 執行（非呼叫端）。

    這是伺服器模式的關鍵前提：recognition task 在 engine thread 建立，
    callback 即落在 engine thread 的 RunLoop（uvicorn 的 main thread
    被 event loop 佔住不 pump RunLoop，舊法 dispatch_sync 到 main
    queue 會死鎖）。
    """
    ran = []
    streaming._dispatch(True, False, lambda: ran.append(threading.current_thread().name))
    assert ran == ["macstt-engine"]
    assert "macstt-engine" != threading.current_thread().name


def test_dispatch_engine_ignores_on_main():
    """engine 路徑不受 on_main 影響：callback 一律由 engine thread 遞交，
    與呼叫端是否在 main thread 無關。"""
    ran = []
    streaming._dispatch(True, True, lambda: ran.append(threading.current_thread().name))
    assert ran == ["macstt-engine"]


def test_dispatch_engine_propagates_block_exception():
    """block 內的異常經 Future 傳播回呼叫端（不被吞掉）。"""
    def boom():
        raise ValueError("engine-boom")

    with pytest.raises(ValueError, match="engine-boom"):
        streaming._dispatch(True, False, boom)


# ---- finish() 的 engine 模式等待邏輯 ----

def test_finish_engine_mode_waits_on_event(monkeypatch):
    """use_engine=True 時 finish() 一律以 _done.wait() 等待（callback 由
    engine thread 遞交），不 pump main RunLoop——由子執行緒呼叫驗證
    （uvicorn 環境下 finish 正是從 worker thread 叫的）。

    用 object.__new__ 造佔位 instance（不跑 __init__ → 不碰
    SFSpeechRecognizer、不觸發 TCC）。
    """
    rec = object.__new__(StreamingRecognizer)
    rec._use_engine = True
    rec._finished = False
    rec._feed_queue = queue_mod.Queue()
    rec._request = None          # endAudio 被 if self._request: 保護，安全跳過
    rec._final_result = None
    rec._final_error = None
    rec._partials = ["你好"]
    rec._language = "zh-TW"
    rec._done = threading.Event()

    results = []

    def worker():
        results.append(rec.finish(timeout=5.0))

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    # 模擬 recognition callback：由 engine thread 的 tick loop 遞交
    EngineThread.shared().run(rec._done.set).result(timeout=5.0)
    t.join(timeout=5.0)

    assert not t.is_alive(), "finish() 未在 5 秒內完成（等待邏輯有誤？）"
    assert len(results) == 1
    r = results[0]
    assert isinstance(r, RecognitionResult)
    assert r.text == "你好"


# ---- 建構子簽名（live_zh.py 不相容檢查） ----

def test_init_signature_use_engine_thread_default_false():
    """use_engine_thread 是最後一個 keyword-only 參數、預設 False：
    live_zh.py 現行呼叫法（不傳此參數）維持本地模式、行為不變。"""
    sig = inspect.signature(StreamingRecognizer.__init__)
    p = sig.parameters["use_engine_thread"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY
    assert p.default is False
    assert list(sig.parameters.values())[-1].name == "use_engine_thread"
