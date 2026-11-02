"""pytest 全域設定（live 測試的自動 skip）。

帶 ``live`` mark 的測試需要「已授權語音辨識」的 macOS 環境，會跑真實的
SFSpeechRecognizer 引擎。在沒有授權（或狀態為 not-determined、且無 GUI 可
彈授權框，例如 pytest 非 .app 啟動）的環境下，呼叫
``SFSpeechRecognizer.requestAuthorization_`` 會觸發 TCC SIGABRT，直接把
整個 process 打死。

因此：MACSTT_LIVE 未設定為 "1" 時，於收集階段就把 live 測試標記為 skip，
確保 `uv run pytest` 絕不碰真實語音辨識引擎。

    uv run pytest                 # 不跑 live 測試（安全）
    MACSTT_LIVE=1 uv run pytest   # 跑 live 測試（需在已授權的 macOS 環境）
"""

import os

import pytest


def pytest_collection_modifyitems(config, items):
    """MACSTT_LIVE != "1" 時，自動 skip 所有 live 測試。"""
    if os.environ.get("MACSTT_LIVE") == "1":
        return
    skip_live = pytest.mark.skip(reason="需要 MACSTT_LIVE=1（已授權的 macOS 環境）")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
