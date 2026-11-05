"""SFSpeechRecognizer 專用引擎執行緒（全程序唯一的 daemon thread）。

為什麼需要：SFSpeechRecognizer 的 recognition callback 實測固定遞交在
「主執行緒」的 RunLoop（由 server.py 的 _pump_main_runloop 定期 pump 接收）。
但 uvicorn 的 asyncio event loop 佔住 main thread，ObjC 操作（建 recognizer、
start task、appendAudioPCMBuffer、endAudio 等）不能在 event loop 內同步
等待；streaming.py 舊法 dispatch_sync 到 main dispatch queue 在伺服器環境
會死鎖。

解法：本模組提供一個常駐 daemon 執行緒，作為「ObjC 操作序列化執行緒」——
所有 SFSpeechRecognizer 操作排到它上面依序執行（單一呼叫點、避免跨執行緒
同時碰 ObjC 物件）；callback 仍由主執行緒遞交，不經本執行緒。
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import Future, InvalidStateError
from typing import Callable, Protocol, TypeVar

import Foundation

logger = logging.getLogger("macos_stt.engine_thread")

T = TypeVar("T")

# 每圈 pump RunLoop ~20ms：讓 callback / 音訊 chunk 不積壓，同時控制開銷
_TICK_SECONDS = 0.02


class EngineJob(Protocol):
    """長工作介面（如進行中的串流辨識 session）。"""

    def advance(self) -> bool:
        """每 tick 在 engine thread 呼叫；回 True 表示完成（該 job 被移除）。"""
        ...


class EngineThread:
    """daemon thread，全程序唯一，序列化所有 SFSpeechRecognizer 的 ObjC 操作。

    為什麼操作要送來這裡：task 建立、音訊餵入、endAudio 需在同一條執行緒
    依序進行，不能跨執行緒碰同一批 ObjC 物件。tick 內仍 pump 自己的
    RunLoop（無 source 時會立即返回，故 tick 需補等，見 _loop）。
    """

    _instance: "EngineThread | None" = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        # 短工作佇列：(fn, future, submitted_at)。音訊 chunk 不能積壓，
        # 主迴圈每圈全部排空。
        self._short_jobs: "queue.Queue[tuple[Callable[[], object], Future, float]]" = queue.Queue()
        # 長工作清單：每 tick 逐一 advance()，回 True 移除
        self._long_jobs: list[EngineJob] = []
        self._long_jobs_lock = threading.Lock()
        self._running = threading.Event()
        self._running.set()
        # 有新短工作時喚醒 tick 的等待（見 _loop）
        self._wake = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="macstt-engine", daemon=True)

    # ---- 單例 ----

    @classmethod
    def shared(cls) -> "EngineThread":
        """惰性單例（threading.Lock 保護），首次呼叫時啟動 thread。"""
        inst = cls._instance
        if inst is not None and inst._running.is_set():
            return inst
        with cls._instance_lock:
            inst = cls._instance
            if inst is None or not inst._running.is_set():
                inst = cls()
                inst._thread.start()
                cls._instance = inst
            return inst

    @classmethod
    def _reset(cls) -> None:
        """測試用：停止並清除 singleton。"""
        with cls._instance_lock:
            inst, cls._instance = cls._instance, None
        if inst is not None:
            inst.stop()

    # ---- 公開 API ----

    def run(self, fn: Callable[[], T], timeout: float = 30.0) -> Future[T]:
        """把「短」工作（建 recognizer、start task、appendAudioPCMBuffer、
        endAudio 等）排到 engine thread 執行，回傳 concurrent.futures.Future。

        為什麼必须在 engine thread：task 的建立與音訊餵入要發生在 callback
        所綁定的那個執行緒上，否則 callback 不會觸發、objC 物件會在錯誤
        執行緒被碰。

        timeout 秒內未完成 → Future 以 TimeoutError 結束；工作本身仍留在
        engine thread 排定執行（絕不在呼叫端執行）。
        """
        fut: Future[T] = Future()
        self._short_jobs.put((fn, fut, time.monotonic()))
        self._wake.set()

        def _expire() -> None:
            # 工作可能正在 engine thread 執行中（fn 慢或佇列積壓），
            # 看門狗只能結束 Future，不能去碰工作本身。
            try:
                if not fut.done():
                    fut.set_exception(
                        TimeoutError(f"engine thread job timed out after {timeout:.1f}s")
                    )
            except InvalidStateError:
                pass  # 與 engine thread 競態：future 剛被正常完成

        timer = threading.Timer(timeout, _expire)
        timer.daemon = True
        timer.start()
        # 工作完成就收掉看門狗：音訊幀每個都走這裡，不取消的話每幀留一條 30 s 的 Timer 執行緒
        fut.add_done_callback(lambda _f: timer.cancel())
        return fut

    def register(self, job: EngineJob) -> None:
        """掛一個長工作：tick loop 每圈在 engine thread 呼叫 job.advance()，
        回 True 時移除。

        為什麼用「每 tick advance」而非佔住佇列：長工作（等辨識結果）不能
        阻塞音訊 chunk 的短工作佇列。
        """
        with self._long_jobs_lock:
            self._long_jobs.append(job)

    @property
    def active_jobs(self) -> int:
        """目前掛著的長工作數（供 server 統計與測試）。"""
        with self._long_jobs_lock:
            return len(self._long_jobs)

    def stop(self) -> None:
        """停止 thread（idempotent），join 等待結束。"""
        if not self._running.is_set():
            return
        self._running.clear()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)

    # ---- thread 主迴圈 ----

    def _loop(self) -> None:
        run_loop = Foundation.NSRunLoop.currentRunLoop()
        while self._running.is_set():  # 4) 每圈檢查 stop event
            try:
                # 1) pump 自己的 RunLoop（recognition callback 實測由主執行緒
                #    遞交，這裡通常沒有 source）
                tick_start = time.monotonic()
                run_loop.runMode_beforeDate_(
                    Foundation.NSDefaultRunLoopMode,
                    Foundation.NSDate.dateWithTimeIntervalSinceNow_(_TICK_SECONDS),
                )
                # 本執行緒 RunLoop 沒有 source 時 runMode 會立即返回（不阻塞），
                # 若不補等，整圈變 100% CPU 空轉並以 pyobjc 呼叫霸佔 GIL，
                # 餓死 uvicorn。補等到 tick 結束；有新短工作即被 _wake 喚醒。
                remaining = _TICK_SECONDS - (time.monotonic() - tick_start)
                if remaining > 0:
                    self._wake.wait(remaining)
                self._wake.clear()
                # 2) 排空短工作佇列（音訊 chunk 不能積壓）
                self._drain_short_jobs()
                # 3) 推進長工作
                self._advance_long_jobs()
            except Exception:
                # 單圈異常不殺死 thread：記 log 到 stderr（未配置 handler 時
                # logging 的 lastResort handler 寫 stderr）繼續
                logger.warning("macstt-engine loop error, continuing", exc_info=True)
                time.sleep(0.01)

    def _drain_short_jobs(self) -> None:
        items: list[tuple[Callable[[], object], Future, float]] = []
        while True:
            try:
                items.append(self._short_jobs.get_nowait())
            except queue.Empty:
                break
        for fn, fut, _submitted_at in items:
            try:
                result = fn()
            except Exception as exc:
                try:
                    if not fut.done():  # 已逾時結束 → 丟棄
                        fut.set_exception(exc)
                except InvalidStateError:
                    pass
                continue
            try:
                if not fut.done():  # 已逾時（fut 被設 TimeoutError）→ 丟棄結果
                    fut.set_result(result)
            except InvalidStateError:
                # 看門狗與本執行緒競態：future 剛被設 TimeoutError。
                # 不抛異常，避免中斷同批剩餘工作。
                pass

    def _advance_long_jobs(self) -> None:
        with self._long_jobs_lock:
            jobs = list(self._long_jobs)
        if not jobs:
            return
        finished: list[EngineJob] = []
        for job in jobs:
            try:
                if job.advance():
                    finished.append(job)
            except Exception:
                logger.warning("engine job advance() error", exc_info=True)
        if finished:
            with self._long_jobs_lock:
                for job in finished:
                    try:
                        self._long_jobs.remove(job)
                    except ValueError:
                        pass
