"""vocab.VocabularyStore 單元測試。

純 Python、純記憶體操作，不碰 SFSpeechRecognizer，不需 macOS 語音授權。
"""

import threading

import pytest

from macos_stt.vocab import WARN_LIMIT, VocabularyStore


def test_load_default():
    """套件內預設詞庫（D4）應含 live_zh.py DEFAULT_HINTS 的 7 詞。"""
    store = VocabularyStore.load_default()
    assert store.get() == [
        "VS Code", "Python", "GitHub", "Git", "Mac", "Terminal", "Markdown",
    ]
    assert len(store) == 7
    assert store.warning is None


def test_load_file_formats(tmp_path):
    """混用每行、英逗號、中逗號、分號（中/英）與空行 → 解析正確。"""
    p = tmp_path / "vocab.txt"
    p.write_text(
        "商智中心\n"
        "BIRC,Python\n"
        "   \n"
        "台北大學，終端機、快訊;GitHub；Uvicorn\n"
        "\n\n"
        "   FastAPI   \n",
        encoding="utf-8",
    )
    store = VocabularyStore.load_file(str(p))
    assert store.get() == [
        "商智中心", "BIRC", "Python", "台北大學",
        "終端機", "快訊", "GitHub", "Uvicorn", "FastAPI",
    ]


def test_load_file_missing(tmp_path):
    """不存在的檔 → FileNotFoundError（load_file 與 load_file_merged 皆同）。"""
    missing = str(tmp_path / "nope.txt")
    with pytest.raises(FileNotFoundError):
        VocabularyStore.load_file(missing)
    with pytest.raises(FileNotFoundError):
        VocabularyStore().load_file_merged(missing)


def test_replace_and_merge():
    """replace 整批覆寫；merge 保序去重（重複詞只留第一次出現位置）。"""
    store = VocabularyStore(["Python", "Git"])

    # replace: 舊詞全被清掉
    store.replace(["Mac", "Python"])
    assert store.get() == ["Mac", "Python"]

    # merge: "Python" 已存在 → 保留原位置；新詞依序附加到後面
    store.merge(["Python", "VS Code", "Mac", "GitHub"])
    assert store.get() == ["Mac", "Python", "VS Code", "GitHub"]

    # 空白 / 空項被丟掉
    store.merge(["  ", "", "  Terminal  "])
    assert store.get() == ["Mac", "Python", "VS Code", "GitHub", "Terminal"]


def test_get_returns_copy():
    """get() 回傳副本：改回傳 list 不影響 store。"""
    store = VocabularyStore(["Python", "Git"])
    got = store.get()
    got.append("Hacked")
    got[0] = "改過的詞"
    assert store.get() == ["Python", "Git"]
    assert len(store) == 2


def test_warning_threshold():
    """len > WARN_LIMIT(200) → warning 非 None；<= 200 → None。"""
    store = VocabularyStore()
    assert store.warning is None

    store.replace([f"w{i:03d}" for i in range(WARN_LIMIT)])
    assert len(store) == WARN_LIMIT
    assert store.warning is None

    store.merge(["overflow"])  # 第 201 條
    assert len(store) == WARN_LIMIT + 1
    assert isinstance(store.warning, str)
    assert store.warning


def test_thread_safety():
    """8 個 thread 各 merge 100 個互不重複詞 → 最終恰好 800 條。"""
    store = VocabularyStore()
    n_threads, per_thread = 8, 100

    def worker(idx: int):
        store.merge([f"w{idx}-{i}" for i in range(per_thread)])

    threads = [
        threading.Thread(target=worker, args=(i,)) for i in range(n_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    words = store.get()
    assert len(words) == n_threads * per_thread
    assert len(set(words)) == n_threads * per_thread  # 全數保留、無重複
