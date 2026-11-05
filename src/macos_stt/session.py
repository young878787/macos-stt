"""伺服器端串流 session 生命週期管理。

移植 live_zh.py 已驗證的「引擎自斷句 + VAD 安全網 + session 上限重開」
邏輯，重構成 server 可複用的 Session / SessionManager：

- live_zh.LiveSTT.process_block（引擎模式分支）→ Session._process_block
- live_zh.LiveSTT._flush_tail                  → Session._flush_tail
- live_zh.LiveSTT._finish_session              → Session._finish_session
- live_zh.LiveSTT._start_session               → Session._start_session
- live_zh.VAD                                  → VAD（一處差異見 class docstring：
  疑似語音幀不入底噪視窗，開頭即語音的 clip 也能觸發安全網 flush）

差異：
1. 回呼不再印螢幕，改為 on_event(dict)（partial / utterance / final /
   session_restarted），實現必須 thread-safe（callback 由 server.py 的
   主執行緒 RunLoop pump 遞交，事件在主執行緒或 worker thread 發出）。
2. recognizer 走 use_engine_thread=True 路徑（ObjC 操作序列化在
   engine thread；callback 由主執行緒 pump 遞交），等待端一律 event.wait，
   不 pump main RunLoop。
3. 呼叫端（uvicorn worker thread / asyncio.to_thread）不得是 engine
   thread 本身 —— 會在 _dispatch 的 run().result() 自己等自己而死鎖。
4. 引擎單槽：實證 SFSpeechRecognizer on-device 同時只能跑 1 個識別
   任務（多 task 時其餘 0 事件）。真引擎模式下同時僅 1 個 session 持槽
   識別，其餘 session 成為等待者（照收音、幀進有界緩衝）；持槽者
   flush / close 時輪替給最早等待者並重放緩衝。mock 模式不啟用槽。
"""

from __future__ import annotations

import math
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Callable

from .engine import MacOSSpeechError

# ---- 常數（與 live_zh.py 對應）----

SILENCE_PAUSE = 1.5       # 靜音幾秒判定「不在說話」（live_zh 的 --pause 預設值）
FLUSH_PRE_DELAY = 0.4     # 斷句後等引擎處理完句尾音訊再 endAudio（live_zh 的 0.4s pump）
RESTART_DELAY = 0.3       # flush 後重開 session 前的間隔（live_zh 的 _wait(0.3)）
FINISH_TIMEOUT = 10.0     # 單次 finish 等 final 的秒數

# 引擎單槽（實證：SFSpeechRecognizer on-device 同時只能跑 1 個識別任務，
# 硬開多 task 時僅先建立的收得到結果，其餘 0 事件）。等待者把音幫緩衝
# 到拿回槽時重放。
SLOT_PREBUF_FRAMES = 80   # 等待者保留最近 80 幀（0.2s/幀 = 16s）
SLOT_REPLAY_STEP = 50     # 重放每 50 幀稍歇，避免一口氣積壓引擎佇列

# VAD 常數（來源 live_zh.py）
NOISE_WINDOW = 50
NOISE_FLOOR = 0.005
NOISE_FACTOR = 4.0


class VAD:
    """簡單的自適應背景雜音音量偵測。

    與 live_zh.py 原版有一處關鍵差異：原版先把當前幀 RMS 納入視窗、
    再取 10 分位數當底噪——若音訊**開頭就是連續語音、無前置靜音**（
    TTS 回灌、WS 串流 clip），前 10 幀全是語音值 → p10 跟著語音水位 →
    threshold = 4×語音音量 → 全程「偵測不到語音」→ `_session_had_speech`
    恒 False → 安全網 flush 永不觸發（離線模擬 + DBG 實證）。live_zh 的
    mic 開頭有環境底噪自體修正，踩不到。

    本版改為「先比較、疑似語音幀不入視窗」：threshold 永遠只由
    （疑似）非語音幀構成，開口即語音也在第一幀命中；語音高值不會
    污染底噪（原版註解的本意）。
    """

    def __init__(self, factor=NOISE_FACTOR, floor=NOISE_FLOOR):
        self.factor = factor
        self.floor = floor
        self._window: list[float] = []
        self._noise = floor

    @property
    def noise(self) -> float:
        return self._noise

    def update(self, rms: float) -> None:
        if rms <= self.threshold:
            # 疑似非語音幀：納入視窗、重算底噪
            w = self._window
            w.append(rms)
            if len(w) > NOISE_WINDOW:
                w.pop(0)
            # 用 10 分位數當底噪
            s = sorted(w)
            self._noise = max(self.floor, s[max(0, len(s) // 10)])
        # else 疑似語音幀：不納入，底噪不變

    @property
    def threshold(self) -> float:
        return self.noise * self.factor


class SessionLimitExceeded(Exception):
    """session 數達到 max_sessions 時由 SessionManager.acquire 拋出。"""


@dataclass
class SessionConfig:
    """一個串流 session 的參數（對應 WS config 消息）。"""

    language: str = "zh-TW"
    on_device: bool = True
    multi_utterance: bool = True        # 引擎自斷句（detect_multiple_utterances）
    vad_flush_seconds: float = 3.0      # 安全網：靜音 N 秒仍無新句 → flush 尾句重開
    max_session_seconds: float = 300.0  # 單一 session 最長秒數（到點自動重開）
    flush_pre_delay: float = FLUSH_PRE_DELAY  # 斷句後、endAudio 前等引擎處理句尾音訊的秒數
    hints: list[str] = field(default_factory=list)


def _rms_int16(data: bytes) -> float:
    """int16 mono PCM 的 RMS（0.0~1.0）。6400 samples 規模純 python 就夠快。"""
    n = len(data) // 2
    if n == 0:
        return 0.0
    samples = struct.unpack(f"<{n}h", data[: n * 2])
    acc = sum(s * s for s in samples)
    return math.sqrt(acc / n) / 32768.0


class Session:
    """一個串流辨識 session。

    生命週期：__init__ 即建立第一個 recognizer session；靜音安全網或
    session 上限觸發時自動 flush 尾句 + 重開（發 session_restarted）；
    close() 後 feed 拋錯、名額由 SessionManager 釋放。

    注意：feed()/close() 必須從「非 engine thread」執行（worker thread
    或 asyncio.to_thread），否則 _dispatch 會自己等自己死鎖。
    """

    def __init__(
        self,
        config: SessionConfig,
        on_event: Callable[[dict], None] | None = None,
        recognizer_factory: Callable[..., object] | None = None,
        manager: "SessionManager | None" = None,
        waiting: bool = False,
    ):
        """Args:
            on_event: 事件回呼 {"type": "partial"|"utterance"|"final"|
                "session_restarted", ...}。可能在主執行緒（callback 經
                server 的 main RunLoop pump 遞交）或 worker thread 發出，
                實現必須 thread-safe。
            recognizer_factory: 測試注入用；簽名 factory(contextual_strings=...)。
                None → 建真 StreamingRecognizer（use_engine_thread=True）。
            manager: 所屬 SessionManager（engine 單槽的 claim/rotate 用）。
            waiting: True → 引擎單槽已被佔，先不啟 recognizer（成為等待者；
                音幀進 _prebuf，輪到時由 _become_active 重放）。
        """
        self._config = config
        self._on_event = on_event
        self._factory = recognizer_factory
        self._manager = manager

        self._vad = VAD()
        self.speaking = False
        self._session_had_speech = False
        self._pending_flush = False
        self._last_speech = 0.0
        self._rec = None
        self._final_via_on_final = False
        self._timing: dict | None = None  # 收尾中的計時：{reason, pre_delay_ms, t_end}，final 事件附上 timing
        self._session_start = 0.0
        self._closed = False

        self._waiting = waiting
        # RLock：_become_active 內的 flush 在無人可輪替時會重入 _become_active 重開
        self._promote_lock = threading.RLock()
        self._prebuf: deque[bytes] = deque(maxlen=SLOT_PREBUF_FRAMES)

        if waiting:
            self._session_start = time.monotonic()
        else:
            self._start_session()

    # ---- 公開 API ----

    def feed(self, pcm16: bytes | memoryview) -> None:
        """餵一個音訊 frame（int16 mono，建議 16kHz × 0.2s ≈ 3200 samples）。

        步驟：算 RMS → VAD/斷句決策 → 餵進 recognizer。
        必須在「非 engine thread」執行。

        引擎單槽：未持槽（等待者或 flush 間隙）時，先嘗試 `_claim_slot`；
        拿不到就把幀放進有界緩衝（保留最近 `SLOT_PREBUF_FRAMES` 幀），
        拿到槽之後由 _become_active 重放。
        """
        if self._closed:
            raise MacOSSpeechError("session closed")
        data = bytes(pcm16)
        self._process_block(_rms_int16(data))
        rec = self._rec
        if rec is None:
            m = self._manager
            got = m._claim_slot(self) if m is not None else True
            if not got:
                self._prebuf.append(data)
                return
            rec = self._rec
            if rec is None:  # 拿到槽但啟動失敗（異常往上拋前先不丟音）
                self._prebuf.append(data)
                return
        rec.feed(data)

    def flush(self) -> None:
        """client 判斷這句說完了：立刻收尾（final 立即出）並重開。必須從「非 engine thread」、
        且與 feed() 同一條序列呼叫（WS 的 feed 迴圈依序處理，flush 一定排在先前的音訊幀之後）。
        沒在識別（等待者）、或這一句還沒有任何語音／文字時什麼都不做。"""
        if self._closed:
            return
        rec = self._rec
        if rec is None or not (self._session_had_speech or (rec._last_partial or "").strip()):
            return
        self._flush_tail("client_flush")

    # ---- 引擎單槽：槽生涯 ----

    def _become_active(self) -> None:
        """拿回 engine 槽：啟 recognizer，把等待期間緩衝的幀重放（先重算
        VAD 時鐘——_start_session 會重置狀態，重放得把「講過話」的事實
        重新帶回，否則安全網 flush 判不出語音）。

        重放完若確實有語音，**立即提交一次 flush**：等待者多半「說完了
        才輪到」，若不馬上收尾，要再等 1.5s 靜音 + 引擎 partial 延遲，
        早收線的 client 會完全拿不到 final（實測踩雷）。"""
        with self._promote_lock:
            if self._rec is not None:
                return  # 已在持槽（rotate 與 feed claim 競態保護）
            m = self._manager
            if m is not None and m._slot and m._active_id != id(self):
                return  # 排隊搶鎖期間槽已輪給別人：再啟動會疊出第二個 task
            self._waiting = False
            snapshot = list(self._prebuf)
            self._prebuf.clear()
            self._start_session()
            rec = self._rec
            if rec is not None:
                for i, frame in enumerate(snapshot):
                    self._process_block(_rms_int16(frame))
                    rec.feed(frame)
                    if i % SLOT_REPLAY_STEP == SLOT_REPLAY_STEP - 1:
                        time.sleep(0.01)
            # 重放的音含語音 → 直接收尾結算（final 立即出，不等新靜音期）
            if snapshot and self._session_had_speech and not self._closed:
                self._flush_tail("replay")

    def _become_waiting(self) -> None:
        """讓出 engine 槽（flush 已把當前 recognizer 收尾完成）。"""
        self._rec = None
        self._waiting = True
        self._prebuf.clear()

    def close(self) -> None:
        """連線中斷/收尾：停止接受 feed，尽力收尾（background endAudio，
        不阻塞呼叫端、不等 final）。"""
        if self._closed:
            return
        self._closed = True
        rec = self._rec
        self._rec = None
        if rec is not None:

            def _fin():
                try:
                    rec.finish(timeout=5.0)
                except Exception:
                    pass  # 收尾失败不影響資源釋放

            threading.Thread(target=_fin, daemon=True).start()

    # ---- 內部（移植 live_zh.py 引擎模式）----

    def _emit(self, event: dict) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(event)
        except Exception:
            # 事件通道（如已斷開的 WS queue）出錯不該打斷 session
            pass

    def _start_session(self):
        self.speaking = False
        self._session_had_speech = False
        self._pending_flush = False
        self._last_speech = 0.0
        self._session_start = time.monotonic()

        cfg = self._config
        hints = cfg.hints or None
        self._final_via_on_final = False

        def _on_final(r) -> None:
            # 引擎 final callback；標記已送，避免 _finish_session 再送一次
            # 重複 final
            self._final_via_on_final = True
            self._emit({"type": "final", "text": r.text, "confidence": r.confidence, **self._timing_field()})

        callbacks = dict(
            on_partial=lambda t: self._emit({"type": "partial", "text": t}),
            on_utterance=(
                lambda t: self._emit({"type": "utterance", "text": t})
                if cfg.multi_utterance
                else None
            ),
            on_final=_on_final,
        )
        if self._factory is not None:
            # mock factory 也收到回呼（可忽略），行為與真 recognizer 一致
            rec = self._factory(contextual_strings=hints, **callbacks)
        else:
            from .streaming import StreamingRecognizer

            rec = StreamingRecognizer(
                language=cfg.language,
                on_device=cfg.on_device,
                contextual_strings=hints,
                detect_multiple_utterances=cfg.multi_utterance,
                use_engine_thread=True,
                **callbacks,
            )
        rec.start()
        self._rec = rec

    def _process_block(self, rms: float) -> None:
        """VAD + 斷句決策（live_zh.LiveSTT.process_block 引擎模式分支）。"""
        now = time.monotonic()
        vad = self._vad
        vad.update(rms)
        if rms > vad.threshold:
            self._last_speech = now
            self.speaking = True
            self._session_had_speech = True
            self._pending_flush = False
        elif self.speaking and (now - self._last_speech) >= SILENCE_PAUSE:
            self.speaking = False

        rec = self._rec
        if rec is None:
            return
        # 安全網：靜音超過 vad_flush_seconds 仍無新句 → 提交尾句重開
        if (
            not self.speaking
            and self._session_had_speech
            and (now - self._last_speech) >= self._config.vad_flush_seconds
            and not self._pending_flush
            and (rec._last_partial or "").strip()
        ):
            self._flush_tail("server_vad")
        # 保險：session 超過上限就收掉重開
        elif (now - self._session_start) > self._config.max_session_seconds:
            self._flush_tail("max_session")

    def _flush_tail(self, reason: str = "server_vad"):
        if self._pending_flush or self._closed:
            return
        self._pending_flush = True
        try:
            if self._finish_session(reason):
                rotated = False
                m = self._manager
                if m is not None:
                    # 引擎單槽：flush 點是最佳輪替時機——把槽交給最早的
                    # 等待者，並重放它的緩衝；沒有等待者才自己重開。
                    rotated = m._rotate_slot(self)
                if not rotated:
                    # 呼叫方在 worker thread：直接 sleep（engine thread 持續
                    # 遞交 callback，不依賴 main RunLoop）
                    time.sleep(RESTART_DELAY)
                    if not self._closed:
                        self._become_active()
                        self._emit({"type": "session_restarted"})
        finally:
            self._pending_flush = False

    def _timing_field(self) -> dict:
        """final 事件的 timing 欄位：收尾原因、endAudio 前等了多久、endAudio 到 final 花多久（引擎辨識）"""
        t = self._timing
        if t is None:
            return {}
        return {"timing": {"reason": t["reason"], "pre_delay_ms": t["pre_delay_ms"],
                           "finish_ms": round((time.monotonic() - t["t_end"]) * 1000, 1)}}

    def _finish_session(self, reason: str = "server_vad") -> bool:
        """結束當前 recognizer session，final 有文字則發 final 事件。
        回傳是否真的有 session 被結束。"""
        rec = self._rec
        self._rec = None
        if rec is None:
            return False
        # 尾部處理：靜音確認後等引擎處理完句尾音訊再 endAudio
        t0 = time.monotonic()
        time.sleep(self._config.flush_pre_delay)
        t_end = time.monotonic()
        self._timing = {"reason": reason, "pre_delay_ms": round((t_end - t0) * 1000, 1), "t_end": t_end}
        try:
            r = rec.finish(timeout=FINISH_TIMEOUT)
        except MacOSSpeechError:
            r = None
        # on_final 已送過（同一結果）就不重複送
        if not self._final_via_on_final and r and (r.text or "").strip():
            self._emit(
                {"type": "final", "text": r.text, "confidence": r.confidence, **self._timing_field()}
            )
        t = self._timing
        self._timing = None
        print(f"[flush] reason={reason} pre_delay_ms={t['pre_delay_ms']} "
              f"finish_ms={round((time.monotonic() - t['t_end']) * 1000, 1)} "
              f"text_len={len((r.text or '').strip()) if r else 0}", flush=True)
        return True


class SessionManager:
    """管理同時活躍的 session 數（預設 max_sessions=3，並發上限）。

    **engine 單槽**（實證：SFSpeechRecognizer on-device 同時只能跑 1 個
    識別任務，多 task 時其餘全部 0 事件）：

    - 真引擎模式（recognizer_factory=None）：同時只有 1 個 session 持
      槽在識別；等待者照收音、把幀緩衝，持槽者 flush / close 時輪替給
      最早的等待者並重放其緩衝。
    - mock 模式（測試注入 factory）：不啟用槽（mock 沒有引擎限制），
      行為與多 session 並行相同，維持既有測試語意。
    """

    def __init__(
        self,
        max_sessions: int = 3,
        vocabulary=None,
        recognizer_factory: Callable[..., object] | None = None,
    ):
        """Args:
            vocabulary: macos_stt.vocab.VocabularyStore，可選。acquire 時
                與 config.hints 合併（vocab 先、request 後，保序去重）。
            recognizer_factory: 測試注入用，透傳給 Session。
        """
        self._max = max_sessions
        self._vocabulary = vocabulary
        self._factory = recognizer_factory
        self._slot = recognizer_factory is None  # mock 不啟用單槽
        self._active_id: int | None = None
        # REST 辨識借用引擎中（同樣只能有 1 個 task，不得與串流 session 重疊）
        self._rest_busy = False
        self._rot_lock = threading.Lock()
        self._sessions: dict[int, Session] = {}
        self._lock = threading.Lock()

    def acquire(self, config: SessionConfig, on_event: Callable | None = None) -> Session:
        """建立並註冊一個 session；滿額拋 SessionLimitExceeded。"""
        with self._lock:
            if len(self._sessions) >= self._max:
                raise SessionLimitExceeded(
                    f"session limit reached (max={self._max})"
                )
            if self._vocabulary is not None:
                merged = list(
                    dict.fromkeys(list(self._vocabulary.get()) + list(config.hints))
                )
                config = replace(config, hints=merged)
            # 先以等待者狀態建構（不啟 recognizer）並登記名額／槽；慢的
            # recognizer 啟動放到鎖外，避免擋住 /health 的 stats。
            session = Session(
                config,
                on_event=on_event,
                recognizer_factory=self._factory,
                manager=self,
                waiting=True,
            )
            waiting = False
            if self._slot:
                with self._rot_lock:  # 判斷與佔槽同一臨界區，避免與 REST 借用競態
                    waiting = self._active_id is not None or self._rest_busy
                    if not waiting:
                        self._active_id = id(session)
            self._sessions[id(session)] = session
        if not waiting:
            try:
                session._become_active()
            except BaseException:
                self.release(session)
                raise
        return session

    # ---- engine 單槽 ----

    def _claim_slot(self, sess: Session) -> bool:
        """session 在 feed 時未持槽 → 嘗試拿槽。拿到回 True（並已啟動、
        重放緩衝），拿不到回 False（呼叫端把幀放進緩衝）。

        flush 進行中（_pending_flush）禁止提前拿槽：此時舊 recognizer
        正在 finish，提前 start 會疊出第二個 task。
        """
        if not self._slot:
            return True
        ok = False
        with self._rot_lock:
            if sess._pending_flush or self._rest_busy:
                return False
            if self._active_id is None or self._active_id == id(sess):
                self._active_id = id(sess)
                ok = True
        if ok:
            sess._become_active()
        return ok

    def _rotate_slot(self, sess: Session) -> bool:
        """持槽者 flush 收尾後的輪替：把槽交給最早的等待者（重放其緩衝），
        原持有者轉為等待者。沒有等待者 → False（由呼叫端自行重開）。"""
        if not self._slot:
            return False
        cand = None
        with self._rot_lock:
            if len(self._sessions) == 0:
                return False
            for s in self._sessions.values():  # dict 依序 = acquire 順序（最舊優先）
                if s is not sess and s._waiting and not s._closed:
                    cand = s
                    break
            if cand is None:
                return False
            self._active_id = id(cand)
        cand._become_active()
        sess._become_waiting()
        return True

    def try_acquire_engine(self) -> bool:
        """REST 辨識借用引擎：引擎同時只能跑 1 個 task，若有串流 session
        持槽或另一個 REST 進行中就回 False（呼叫端回 503），避免擠壞串流。
        mock 模式不啟用槽 → 恆 True。借用成功必須配對 release_engine()。"""
        if not self._slot:
            return True
        with self._rot_lock:
            if self._active_id is not None or self._rest_busy:
                return False
            self._rest_busy = True
            return True

    def release_engine(self) -> None:
        if not self._slot:
            return
        with self._rot_lock:
            self._rest_busy = False

    def release(self, session: Session) -> None:
        """關閉並釋放一個 session 的名額（idempotent）。持槽者關閉時釋槽，
        等待者會在其後 feed 時靠 _claim_slot 拿回。"""
        with self._lock:
            if self._sessions.pop(id(session), None) is None:
                return
        if self._slot:
            with self._rot_lock:
                if self._active_id == id(session):
                    self._active_id = None
        session.close()

    @property
    def stats(self) -> dict:
        with self._lock:
            return {"active": len(self._sessions), "max": self._max}
