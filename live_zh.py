#!/usr/bin/env python3
"""麥克風即時中文轉譯 — 自動斷句版。

執行方式（macOS 隱私限制：SFSpeechRecognizer 只能由 .app bundle 啟動）:
    python start.py                  # 繁體中文，引擎自斷句（預設）
    python start.py --lang zh-CN     # 簡體中文
    python start.py --no-engine      # 改回舊版「靜音 1.5s 斷句」
    python start.py --cloud          # 允許雲端辨識（中英混合較穩，需網路）
    python start.py --hints 詞表.txt # 自訂提示詞（VS Code 等專有名詞）

斷句邏輯:
    - 預設「引擎自斷句」: 開啟 SFSpeechRecognizer 的多句端點偵測，由引擎
      依語言模型+韵律自行判斷句子結束，連續收音、無斷句空窗；
      音量偵測 (VAD) 降級為安全網（靜音 3 秒仍無新句 → 提交尾句）。
    - --no-engine 回退舊版: 音量偵測 (VAD): 超過 10 秒背景雜音底噪 4 倍
      才視為「有說話」；連續靜音超過 --pause 秒 → 結束當前辨識 → 重開。

主執行緒需持續驅動 NSRunLoop（SFSpeechRecognizer 回调綁定主執行緒）。
"""

import argparse
import os
import re
import signal
import sys
import threading
import time

import Foundation
import numpy as np
import sounddevice as sd

from macos_stt.streaming import StreamingRecognizer
from macos_stt.engine import MacOSSpeechError

SAMPLE_RATE = 16000
BLOCK_DURATION = 0.2          # 每 0.2 秒送一次音訊 / 一次 VAD 判斷
NOISE_WINDOW = 50             # 底噪統計: 最近 50 個 block (10 秒)
NOISE_FLOOR = 0.005           # 底噪最小值（避免完全靜音時門檻=0）
NOISE_FACTOR = 4.0            # 門檻 = 底噪 × 4
RESTART_DELAY = 0.8           # 斷句後停多久再開新 session（僅靜音計時器模式）
SESSION_LIMIT = 60.0          # 單一 session 最長秒數（靜音計時器模式）
ENGINE_SESSION_LIMIT = 300.0  # 引擎自斷句模式下單一 session 最長秒數（保險重開）
ENGINE_SILENCE_FLUSH = 3.0    # 引擎自斷句模式安全網: 靜音 3 秒仍無新句 → 提交尾句

# 內建提示詞: 開發者高頻英文術語（中英混合辨識時容易被中文模型誤听）
DEFAULT_HINTS = [
    "VS Code", "Python", "GitHub", "Git", "Mac", "Terminal", "Markdown",
]


def _load_hints(spec: str | None) -> list[str]:
    """解析 --hints: 檔案路徑 或 逗號/分號/換行分隔清單。

    None 或空字串 → 回傳內建清單。
    """
    if spec is None or not spec.strip():
        return list(DEFAULT_HINTS)
    if os.path.isfile(spec):
        with open(spec, encoding="utf-8") as f:
            text = f.read()
    else:
        text = spec
    return [t.strip() for t in re.split(r"[,\n;]+", text) if t.strip()]


_stop = threading.Event()


class VAD:
    """簡單的自適應背景雜音音量偵測。"""

    def __init__(self, factor=NOISE_FACTOR, floor=NOISE_FLOOR):
        self.factor = factor
        self.floor = floor
        self._window: list[float] = []
        self._noise = floor

    @property
    def noise(self) -> float:
        return self._noise

    def update(self, rms: float) -> None:
        w = self._window
        w.append(rms)
        if len(w) > NOISE_WINDOW:
            w.pop(0)
        # 用 10 分位數當底噪（說話時的高值不會污染它）
        s = sorted(w)
        self._noise = max(self.floor, s[max(0, len(s) // 10)])

    @property
    def threshold(self) -> float:
        return self.noise * self.factor


class LiveSTT:
    def __init__(self, lang: str, pause: float, hints: list[str], on_device: bool,
                 engine: bool = True):
        self.lang = lang
        self.pause = pause
        self.hints = hints
        self.on_device = on_device
        self.engine = engine          # True: 引擎自斷句 / False: 靜音計時器
        self.speaking = False
        self._session_had_speech = False
        self._pending_flush = False
        self._last_speech = 0.0
        self._rec = None
        self._session_start = 0.0
        self.sentence_no = 0

    def _start_session(self):
        self.speaking = False
        self._session_had_speech = False
        self._pending_flush = False
        self._last_speech = 0.0
        self._session_start = time.monotonic()
        self._rec = StreamingRecognizer(
            language=self.lang,
            on_partial=self._on_partial,
            on_device=self.on_device,
            contextual_strings=self.hints or None,
            detect_multiple_utterances=self.engine,
            on_utterance=self._on_utterance if self.engine else None,
        )
        self._rec.start()

    def _on_partial(self, text: str):
        if text:
            print(f"\r  … {text}", end="", flush=True)

    def _print_sentence(self, text: str):
        text = text.strip()
        if not text:
            return
        self.sentence_no += 1
        print("\r" + " " * 60 + "\r", end="", flush=True)  # 清掉 partial 行
        print(f"📝 第{self.sentence_no}句: {text}", flush=True)

    def _on_utterance(self, text: str):
        """引擎自斷句回呼: 引擎判定一句話結束時觸發。

        不重建 session —— 音訊流持續餵進同一個 request，
        沒有斷句空窗，接下來講的話不會被吃掉。
        """
        self._print_sentence(text)

    def _finish_session(self) -> bool:
        """結束當前 session，印出整句。回傳是否真的有 session 被結束。"""
        rec = self._rec
        self._rec = None
        if rec is None:
            return False
        # 尾部處理: 靜音確認後等 0.4s 再 endAudio，讓引擎處理完句尾音訊
        # （pump RunLoop 而非 sleep，避免 feed 執行緒的 dispatch 卡住）
        rl = Foundation.NSRunLoop.currentRunLoop()
        deadline = time.monotonic() + 0.4
        while time.monotonic() < deadline:
            rl.runMode_beforeDate_(
                Foundation.NSDefaultRunLoopMode,
                Foundation.NSDate.dateWithTimeIntervalSinceNow_(0.05),
            )
        print("\r" + " " * 60 + "\r", end="", flush=True)  # 清掉 partial 行
        try:
            r = rec.finish(timeout=10.0)
        except MacOSSpeechError:
            r = None
        if r:
            # 引擎自斷句模式: final 只覆蓋「最後一句」（前面的句已由
            # on_utterance 逐句印出）；靜音計時器模式: final 就是整句。
            self._print_sentence(r.text)
        return True

    def _wait(self, seconds: float):
        """pump RunLoop 等待（不可用 time.sleep：主執行緒要驅動
        feed 執行緒的 dispatch_sync 才能送音訊）。"""
        rl = Foundation.NSRunLoop.currentRunLoop()
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rl.runMode_beforeDate_(
                Foundation.NSDefaultRunLoopMode,
                Foundation.NSDate.dateWithTimeIntervalSinceNow_(0.05),
            )

    def process_block(self, rms: float):
        """主執行緒每 0.2 秒呼叫一次，做 VAD 與 session 生命週期管理。"""
        now = time.monotonic()
        vad.update(rms)
        if rms > vad.threshold:
            self._last_speech = now
            self.speaking = True
            self._session_had_speech = True
            self._pending_flush = False
        elif self.speaking and (now - self._last_speech) >= self.pause:
            self.speaking = False

        if self.engine:
            # 引擎自斷句: 斷句由引擎端點偵測負責（on_utterance 印句）。
            # VAD 只當安全網: 靜音超過 ENGINE_SILENCE_FLUSH 仍無新句
            # （例如尾句沒觸發端點）→ 提交當前尾句後重開。
            if (not self.speaking and self._session_had_speech
                    and (now - self._last_speech) >= ENGINE_SILENCE_FLUSH
                    and not self._pending_flush
                    and self._rec is not None
                    and (self._rec._last_partial or "").strip()):
                self._flush_tail()
            elif self._rec is not None \
                    and (now - self._session_start) > ENGINE_SESSION_LIMIT:
                self._flush_tail()
            return

        # 傳統靜音計時器模式: 說過話 + 靜音超過 pause 秒 → 斷句重開
        if self._session_had_speech and not self.speaking \
                and (now - self._last_speech) >= self.pause:
            if self._finish_session():
                self._wait(RESTART_DELAY)
                self._start_session()
        # 保險: session 超過上限就收掉重開
        elif not self.speaking and self._rec is not None \
                and (now - self._session_start) > SESSION_LIMIT:
            if self._finish_session():
                self._wait(RESTART_DELAY)
                self._start_session()

    def _flush_tail(self):
        """提交引擎自斷句模式下還沒印出的尾句（安全網/收尾共用）。

        多句模式下 final 只覆蓋「最後一句」，前面的句已由 on_utterance
        逐句印出，所以 flush 後不會重複。
        """
        if self._pending_flush:
            return
        self._pending_flush = True
        if self._finish_session():
            self._wait(0.3)
            self._start_session()
        self._pending_flush = False


def main():
    parser = argparse.ArgumentParser(description="即時中文語音轉文字（自動斷句）")
    parser.add_argument("--lang", default="zh-TW",
                        help="語言代碼: zh-TW(繁中) / zh-CN(簡中) / yue-CN(粵語)")
    parser.add_argument("--pause", type=float, default=1.5,
                        help="靜音幾秒算斷句（預設 1.5；中英混合說術語時建議 1.5-2.0）")
    parser.add_argument("--hints", default=None,
                        help="提示詞: 檔案路徑（每行/逗號分隔）或逗號分隔清單；"
                             "不指定時用內建開發術語清單")
    parser.add_argument("--no-hints", action="store_true",
                        help="停用提示詞（contextualStrings）")
    parser.add_argument("--cloud", action="store_true",
                        help="允許雲端辨識（預設強制離線）；中英混合辨識較穩，需網路")
    parser.add_argument("--no-engine", action="store_true",
                        help="停用引擎自斷句，改回舊版「靜音 --pause 秒斷句」")
    parser.add_argument("--device", type=int, default=None,
                        help="輸入裝置編號（預設系統預設麥克風）")
    parser.add_argument("--log", default=None,
                        help="額外輸出到檔案（供 start.py tail 顯示）")
    args = parser.parse_args()

    hints = [] if args.no_hints else _load_hints(args.hints)
    on_device = not args.cloud

    if args.log:
        # 雙重輸出: terminal + log（供 start.py 顯示）
        class Tee:
            def __init__(self, *fs):
                self.fs = fs
            def write(self, s):
                for f in self.fs:
                    f.write(s)
            def flush(self):
                for f in self.fs:
                    f.flush()
        sys.stdout = Tee(sys.__stdout__, open(args.log, "w", buffering=1))

    global vad
    vad = VAD()

    mode = "雲端" if not on_device else "離線"
    seg = "引擎自斷句" if not args.no_engine else f"靜音 {args.pause:.1f}s 斷句"
    hint_info = f" | 提示詞 {len(hints)} 個" if hints else " | 提示詞 停用"
    print(f"🎤 即時轉譯（{args.lang} | {mode} | {seg}）{hint_info} | Ctrl+C 結束",
          flush=True)

    stt = LiveSTT(args.lang, args.pause, hints, on_device,
                  engine=not args.no_engine)
    stt._start_session()

    def _shutdown(*_a):
        _stop.set()

    signal.signal(signal.SIGINT, _shutdown)

    run_loop = Foundation.NSRunLoop.currentRunLoop()

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=int(SAMPLE_RATE * BLOCK_DURATION),
        device=args.device,
        callback=lambda indata, f, t, s: _feed(indata),
    ):
        while not _stop.is_set():
            run_loop.runMode_beforeDate_(
                Foundation.NSDefaultRunLoopMode,
                Foundation.NSDate.dateWithTimeIntervalSinceNow_(0.1),
            )
            stt.process_block(_last_rms)

    # 收尾
    stt._finish_session()
    print("👋 已停止。", flush=True)


_last_rms = 0.0
_holder = {}


def _feed(indata):
    """Core Audio callback: 轉 int16 送進 recognizer，並記錄 RMS 供 VAD。"""
    global _last_rms
    arr = indata.flatten()
    _last_rms = float(np.sqrt(np.mean(arr.astype(np.float32) ** 2)))
    pcm16 = (arr * 32767).astype("int16").tobytes()
    rec = _holder.get("rec")
    if rec is not None:
        try:
            rec.feed(pcm16)
        except MacOSSpeechError:
            pass


# LiveSTT 重建 session 時同步 _holder，讓 callback 永遠喂給最新的 recognizer
_orig_start_session = LiveSTT._start_session


def _start_session_sync(self):
    _orig_start_session(self)
    _holder["rec"] = self._rec


LiveSTT._start_session = _start_session_sync


if __name__ == "__main__":
    # 注意: 直接 python live_zh.py 會被 macOS TCC 拒絕（SIGABRT），
    # 請改用 python start.py 啟動。
    main()
