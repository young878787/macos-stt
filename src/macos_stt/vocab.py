"""詞彙庫（VocabularyStore）— SFSpeechRecognizer contextualStrings 的資料源。

純 Python 模組，不 import pyobjc，可在任何環境（含 CI）直接測試。
解析格式沿用 live_zh.py 的 `_load_hints`：中英文逗號、頓號、分號與換行
都算分隔，strip 後丟掉空項。
"""

from __future__ import annotations

import re
import threading
from pathlib import Path

# 建議上限：超過時回 warning（不拒絕）。
# Apple 未公開 contextualStrings 條數限制，實測過長會稀釋效果。
WARN_LIMIT = 200

# 分隔符：中逗號、英逗號、頓號、中分號、英分號、換行
_DELIM_RE = re.compile(r"[，,、;；\n]+")

# 套件內預設詞庫（相對於本檔，D4 定案）
_DEFAULT_VOCAB = Path(__file__).with_name("vocab.default.txt")


def _split_text(text: str) -> list[str]:
    """依分隔符切分文字，strip 後丟掉空項（保序）。"""
    return [t.strip() for t in _DELIM_RE.split(text) if t.strip()]


def _clean_words(words: list[str]) -> list[str]:
    """逐詞 strip、丟掉空項、保序去重（保留第一次出現位置）。"""
    seen: set[str] = set()
    out: list[str] = []
    for w in words:
        t = w.strip()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _read_words(path: str) -> list[str]:
    """讀取並解析詞表檔；檔案不存在 → FileNotFoundError。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"詞表檔不存在: {path}")
    return _split_text(p.read_text(encoding="utf-8"))


class VocabularyStore:
    """Thread-safe 的詞庫（SFSpeechRecognizer contextualStrings 資料源）。

    所有讀寫都以 RLock 保護；`get()` 回傳副本，呼叫端改動不影響內部狀態。
    超過 `WARN_LIMIT` 條時 `warning` 回警示字串（只警示，不拒絕）。
    """

    def __init__(self, words: list[str] | None = None):
        self._lock = threading.RLock()
        self._words: list[str] = []
        if words is not None:
            # 初始詞表（去空白、去空項、保序去重）
            with self._lock:
                self._words = _clean_words(words)

    @classmethod
    def load_default(cls) -> "VocabularyStore":
        """載入套件內 vocab.default.txt（pathlib 相對 `__file__` 定位）。"""
        return cls.load_file(str(_DEFAULT_VOCAB))

    @classmethod
    def load_file(cls, path: str) -> "VocabularyStore":
        """載入使用者詞表檔，建立全新 store。

        解析格式與 live_zh.py `_load_hints` 相同：用
        `re.split(r"[，,、;；\\n]+", ...)` 切分（中英文逗號、頓號、
        分號與換行都算分隔），strip 後丟掉空項。
        檔案不存在 → FileNotFoundError。
        """
        return cls(_read_words(path))

    def replace(self, words: list[str]) -> None:
        """整批覆寫詞表（去空白、去空項、保序去重）。"""
        with self._lock:
            self._words = _clean_words(words)

    def merge(self, words: list[str]) -> None:
        """合併進詞表：保序去重，重複詞保留第一次出現的位置。"""
        with self._lock:
            existing = set(self._words)
            for t in _clean_words(words):
                if t not in existing:
                    existing.add(t)
                    self._words.append(t)

    def load_file_merged(self, path: str) -> None:
        """讀取詞表檔並合併進現有詞表（不清空原有內容）。"""
        # 先讀檔再 merge，避免持鎖讀 I/O 太久
        parsed = _read_words(path)
        self.merge(parsed)

    def get(self) -> list[str]:
        """回傳詞表的副本（呼叫端改動不會影響 store）。"""
        with self._lock:
            return list(self._words)

    def __len__(self) -> int:
        with self._lock:
            return len(self._words)

    @property
    def warning(self) -> str | None:
        """詞表超過建議上限（`WARN_LIMIT`）時回警示字串，否則 None。"""
        with self._lock:
            n = len(self._words)
        if n > WARN_LIMIT:
            return (
                f"詞表共 {n} 條，超過建議上限 {WARN_LIMIT} 條；"
                "contextualStrings 過長可能稀釋辨識效果。"
            )
        return None
