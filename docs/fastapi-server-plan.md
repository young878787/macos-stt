# macos-stt FastAPI 串流伺服器改造計畫

> 狀態：**已實施（2026-07，M1–M5 完成）**｜ 並發上限：**3 sessions**（`--max-sessions` 預設 3）
> 目標讀者：實作者 ｜ 日期：2026-07

---

## 1. 背景與目標

### 1.1 現況

專案已有 FastAPI HTTP 伺服器（`server.py` + `macos-stt serve`），但只有 4 個端點：

| 端點 | 形式 | 限制 |
|---|---|---|
| `POST /transcribe` | 整檔上傳 | 要先錄完、封成檔案 |
| `POST /transcribe/raw` | base64 PCM JSON | 整段送、整段回；base64 膨脹 33% |
| `GET /languages` | 查詢 | — |
| `GET /health` | 查詢 | — |

**關鍵缺陷：**

1. **沒有串流辨識**：`streaming.py` 的 `StreamingRecognizer`（WebSocket 串流的基石）完全沒有暴露到 HTTP 層。
2. **event loop 阻塞**：`server.py` 的 `async def` 端點直接呼叫 `transcribe_file()`，辨識期間（最長 15s）在 uvicorn 的 event loop（主執行緒）上 pump RunLoop 阻塞。辨識中 health check 與其他請求全卡住，等於單工串行。
3. **`streaming.py` 無法直接搬上 server**：它靠 `_main_sync()` 把 ObjC 操作 `dispatch_sync` 到 **main dispatch queue**。uvicorn 的 asyncio 用 kqueue、**不 pump CFRunLoop**，從 worker thread sync 過去 main queue 會死鎖。
4. **詞庫（contextualStrings）沒接到 HTTP 層**：`engine.py` / `streaming.py` 底層都支援，但 API 無參數可傳、無熱更新。
5. **預設語言寫死 `zh-CN`**，應可配置（需求：預設 `zh-TW`）。

### 1.2 目標

1. **串流 STT**：各專案（終端、瀏覽器、App、CI 錄音…）用 WebSocket 把 PCM 推上來，即時收 partial / 斷句 / final 文字。
2. **自訂詞彙庫外部輸入**：啟動載入 + 執行中熱更新，提升專有名詞命中率。
3. **自訂語言**：全域預設 `zh-TW`，個別請求可覆寫。
4. **event loop 不阻塞**：辨識工作與 FastAPI 排程徹底解耦，`/health` 在满载時仍 < 200ms。
5. **並發 3 個串流 session**（可配），超限明確拒絕。
6. 保留現有 CLI / 本地麥克風模式（`live_zh.py`、`start.py`）不受影響。

### 1.3 非目標（本次不做）

- 語音（TTS）輸出、音訊格式轉碼（伺服器只收 int16 mono 16kHz PCM，客戶端自責重取樣）
- 使用者鑑權 / TLS / LAN 存取（伺服器只綁本機 127.0.0.1:8765，已定案 D3）
- SSE 版串流（只出 WebSocket，已定案 D2）
- launchd 背景 daemon 模式（一律經 MacSTT.app bundle 前台啟動，已定案 D1）
- 雲端辨識預設開啟（保留 `--allow-cloud` 旗標）

---

## 2. 架構設計

### 2.1 核心問題：SFSpeechRecognizer 綁定執行緒 RunLoop

`SFSpeechRecognizer` 的 recognition callback 只會在**建立 task 的那個執行緒**的 CFRunLoop 觸發，且該執行緒必須持續 pump RunLoop。現有 `engine.py`（本地模式：main thread）與 `streaming.py`（dispatch_sync 到 main queue）都是围绕「主執行緒」設計。

uvicorn 的 event loop 在 main thread 但不 pump CFRunLoop，因此：

- 辨識操作不能放 event loop（會鎖死 + 收不到 callback）；
- 也不能照 `streaming.py` 舊法 dispatch_sync 到 main queue（會死鎖）。

### 2.2 解法：獨立 engine thread

```
┌──────────────────────────────────────────────────────┐
│ FastAPI + uvicorn（event loop @ main thread）          │
│   REST + WebSocket：純排程、I/O、session 註冊          │
│   不直接碰任何 ObjC API                                │
└───────────────┬──────────────────────────────────────┘
                │  job queue（短工作）+ 長工作 advance()
                │  結果以 concurrent.futures.Future 回傳
┌───────────────▼──────────────────────────────────────┐
│ engine thread（daemon，全程序唯一）                    │
│   tick loop：                                          │
│     1. 進程自己的 NSRunLoop.runMode_beforeDate_(20ms)  │
│        ← 所有 recognition callback 在這裡被遞交        │
│     2. 處理 job queue 的短工作（建 recognizer、        │
│        start task、appendAudioPCMBuffer、endAudio）    │
│     3. 對每個進行中的長工作呼叫 advance()              │
│        （檢查 done event → 回 Future 結果）            │
└──────────────────────────────────────────────────────┘
```

要點：

- **所有** SFSpeechRecognizer 的建立、task 啟動、音訊餵入都必須在 engine thread 執行（`run_on_engine`），callback 因而落在 engine thread 的 RunLoop 上。
- engine thread 的 RunLoop 常駐 → 多個 session 的 callback 都能被遞交，**天生並行**（各 session 各自建 recognizer）。
- 長工作（等辨識結果）不占住 queue：以「state + advance()」方式掛在 engine thread，每 tick 推進；短工作（音訊 chunk 約 5/s/session）每 tick 排空，餵音不受長工作影響。

### 2.3 並發控制（max_sessions = 3）

- `SessionManager` 維護 `id → Session`，`--max-sessions`（預設 **3**）。
- 滿額時：REST `/transcribe*` 與 WS 新 session → HTTP 503 / WS `{"type":"error","code":"too_many_sessions"}` 後 close(1013)。
- 選擇理由：on-device 辨識同時跑會搶 ANE，Apple 未公開上限；3 是安全起點，之後可依實測調。

---

## 3. API 設計

### 3.1 REST 端點

| 端點 | 說明 |
|---|---|
| `GET /health` | `{status, auth: authorized\|denied\|not_determined, sessions_active, max_sessions, default_lang, vocab_size}`。`auth` 讓部署方知道 macOS 授權狀態（見 §6.1） |
| `GET /languages` | 63 語言清單（保留現狀） |
| `POST /transcribe` | 整檔上傳（保留相容）；query 參數 `lang`（預設全域 `zh-TW`）、`timeout`、`on_device`；body 可附 `hints[]`（multipart form） |
| `POST /transcribe/raw` | `{bytes: base64, sr, lang?, hints?, timeout?, on_device?}` |
| `GET /vocabulary` | `{words[], count, source}` |
| `POST /vocabulary` | `{mode: "replace"\|"merge", words?[] }` 或 `{mode:"load", file:"/path/vocab.txt"}`；**僅對新 session 生效**（contextualStrings 是 per-request 屬性）。回 `{words, count}`；超過建議上限（200 條，警告不拒絕）回 200 + `warning` |
| `GET /sessions` | `{active, max}` |

錯誤映射（沿用現狀）：400 語言/參數錯誤、408 逾時、500 引擎錯誤、503 session 滿額、501 授權未給。

### 3.2 WebSocket 串流端點（核心）

`WS /stream` —— 一 connection = 一 session，佔用一個 max_sessions 名額，關閉即釋放。

**協定（v1）：**

```
客戶端 → 伺服器
  text  frame: 連線後第一則必須是 config（10 秒內未送 → close(1002)）
    {"type":"config",
     "lang":"zh-TW",            # 省略 = 伺服器全域預設
     "sample_rate":16000,       # v1 只接受 16000
     "hints":["商智中心","BIRC"],  # 與伺服器詞庫 merge
     "on_device":true,          # false 需伺服器以 --allow-cloud 啟動
     "multi_utterance":true,    # 引擎自斷句（預設 true）
     "vad_flush_seconds":3.0,   # 安全網：靜音 N 秒 flush 尾句
     "max_session_seconds":300, # session 上限，到點自動重開
     "flush_pre_delay_seconds":0.4} # 可選。收尾時 endAudio 前等引擎處理句尾音訊的秒數（預設 0.4）
  binary frame: raw int16 mono 16kHz PCM（建議 200ms ≈ 6.4KB/幀，上限 256KB/幀）
  text  frame: {"type":"flush"}   # 客戶端自己判斷這句說完了：立刻收尾（final 立即出）並重開。
                                  # 與音訊幀同一條序列處理，一定排在先前的幀之後；這句還沒有語音／文字時不做事
  text  frame: {"type":"close"}

伺服器 → 客戶端
  {"type":"partial",  "text":"…"}                      # 引擎 partial
  {"type":"utterance","text":"第一句"}                 # 引擎自斷句
  {"type":"final",    "text":"…", "confidence":1.0,    # session 收尾
   "timing":{"reason":"client_flush|server_vad|max_session|replay",  # 收尾原因
             "pre_delay_ms":100, "finish_ms":55}}     # endAudio 前等了多久／endAudio 到 final 花多久（舊 client 可忽略）
  {"type":"session_restarted"}                        # 自動重開（透明）
  {"type":"error",    "code":"…", "message":"…"}       # 致命錯誤後 close(1011)
```

**伺服器端 session 生命週期**（移植 `live_zh.py` 已驗證的邏輯到 `session.py`）：

- VAD：伺服器端對每個 binary frame 算 RMS（int16 → float），用 `live_zh.py` 的 `VAD` 類（底噪 10 分位 × 4 門檻）。
- 引擎自斷句模式（預設）：`detect_multiple_utterances=True`，斷句由引擎端點偵測負責（`on_utterance` → WS `utterance`）；VAD 只當安全網（靜音 3s 仍無新句 → flush 尾句重開）。
- 300s session 上限到點 → flush + 重開（發 `session_restarted`），避免 on-device 長音訊不穩。
- 連線中斷：`endAudio` + cancel task，釋放名額。

**為什麼用 WebSocket（而非 SSE）：** 雙向（音訊上、文字下）；音訊走 binary frame 免 base64；瀏覽器原生支援；uvicorn 原生支援，不需額外套件。
（若日後需要 SSE 版本可加 `POST /stream-sse`，本次不實作。）

### 3.3 詞彙庫（contextualStrings）

- **詞庫來源與合併策略**（合併順序：詞庫 store ⊕ 請求自帶 `hints`，去重、保持順序）：
  1. **詞庫 store（server 級）**：啟動時由 `--vocab 詞表.txt` 載入（格式沿用 `live_zh.py`：每行/逗號/分號分隔）；**不指定時預設載入套件內的 `vocab.default.txt`**（已定案 D4，內容 = `live_zh.py` 的 `DEFAULT_HINTS`：VS Code、Python、GitHub、Git、Mac、Terminal、Markdown，每行一詞）。`--vocab` 指定使用者檔時**取代**內建檔。
  2. **`POST /vocabulary` 熱更新**：replace / merge / load 檔案，執行中改，**僅新 session 生效**（contextualStrings 是 per-request 屬性）。
  3. **每請求 / 每 WS session 的 `hints[]`**：與詞庫 store 合併，只對該 session 生效。
- 上限：> 200 條回 warning（Apple 未公開條數限制，實測過長會稀釋效果）。

### 3.4 語言預設

- `--lang zh-TW`（server 全域預設，**預設值即 zh-TW**）；REST 與 WS 的 `lang` 省略時取全域值，也可逐請求覆寫。
- 合法值以 `SUPPORTED_LANGUAGES` 為準（含 `zh-TW`、`zh-HK`、`yue-CN`）；非法 → 400 / WS error。

---

## 4. 實作細節（檔案級）

| 檔案 | 動作 | 說明 |
|---|---|---|
| `src/macos_stt/engine_thread.py` | **新增** ~100 行 | daemon thread + tick loop；`run(fn)->Future`（短工作）；`register(job)`/長工作 advance 機制；`EngineThread.shared()` 單例；`stop()` 供測試 |
| `src/macos_stt/streaming.py` | 重構 | `_main_sync()` 改為：engine thread 活躍時 → `EngineThread.shared().run(...)`（並 wait future）；否則保留原 main-thread/dispatch 路徑（`live_zh.py` 本地模式不回歸） |
| `src/macos_stt/session.py` | **新增** ~150 行 | `Session`（搬 `live_zh.py` 的 LiveSTT 生命週期：VAD、安全網 flush、300s 重開、tail flush）＋ `SessionManager`（max 3、acquire/release、統計）；`on_partial/on_utterance/on_final` 回呼改為「push 到 async queue」（server 端）或可注入 |
| `src/macos_stt/vocab.py` | **新增** ~60 行 | `VocabularyStore`：load_file / replace / merge / get；thread-safe（lock）；上限 warning |
| `src/macos_stt/server.py` | **重寫** ~250 行 | `create_app(default_lang, vocab, max_sessions, allow_cloud)`；lifespan 啟動/關閉 EngineThread；所有端點；WS 端點；錯誤映射 |
| `src/macos_stt/cli.py` | 擴充 | `serve` 加 `--lang`（預設 zh-TW）、`--vocab`（預設 `vocab.default.txt`）、`--max-sessions`（預設 3）、`--allow-cloud`；host 固定 127.0.0.1、port 固定 8765（無 `--host`，D3） |
| `start.py` | 擴充 | 加 `serve` 模式：app bundle 內啟動 uvicorn（D1，TCC 需要 .app，見 §6.1）；`live` 模式（預設）維持原狀 |
| `src/macos_stt/vocab.default.txt` | **新增** | 預設詞庫（D4）：VS Code、Python、GitHub、Git、Mac、Terminal、Markdown，每行一詞；需加進 `pyproject.toml` 的 package data |
| `pyproject.toml` | 修改 | `[server]` extra 加 `websockets>=12`（uvicorn WS 實作）；`[dev]` 加 `websockets>=12`（測試用 WS 客戶端） |
| `tests/test_engine_thread.py` | **新增** | 見 §5 |
| `tests/test_session.py` | **新增** | 見 §5 |
| `tests/test_vocab.py` | **新增** | 見 §5 |
| `tests/test_server.py` | **新增** | 見 §5 |

### 4.1 `engine_thread.py` 草圖

```python
class EngineThread:
    _instance: "EngineThread | None" = None

    @classmethod
    def shared(cls) -> "EngineThread":
        # 惰性單例 + lock
        ...

    def run(self, fn, timeout: float = 30.0):
        """把「短」工作送到 engine thread 執行，回 concurrent.futures.Future。
        短 = 建 recognizer / start task / appendAudioPCMBuffer / endAudio。"""
        fut = Future()
        self._jobs.put((fn, fut))
        return fut

    def register(self, job: "EngineJob"):
        """掛長工作（辨識中）：每 tick 呼叫 job.advance()，
        完成時 job 自己回傳結果到它的 Future。"""
        ...

    def _loop(self):
        rl = Foundation.NSRunLoop.currentRunLoop()
        while self._running:
            # 1) pump 自己的 RunLoop —— 所有 recognition callback 在此遞交
            rl.runMode_beforeDate_(
                Foundation.NSDefaultRunLoopMode,
                Foundation.NSDate.dateWithTimeIntervalSinceNow_(0.02))
            # 2) 排空短工作（音訊 chunk 不能堆）
            self._drain_jobs()
            # 3) 推進長工作
            for job in list(self._jobs_long):
                job.advance()
```

`transcribe_file` / `StreamingRecognizer` 的等待迴圈（原本 pump main RunLoop 那段）改為：engine 模式下「等 `done` event」即可（callback 由 engine thread 的 tick 遞交）；本地模式維持原樣。

### 4.2 WS 端點草圖

```python
@app.websocket("/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    cfg = await _read_config(ws)          # 10s 超時 → close(1002)
    try:
        session = await asyncio.to_thread(manager.acquire, cfg)  # 滿額 → 503 等價
    except SessionLimitExceeded:
        await ws.send_json({"type": "error", "code": "too_many_sessions"})
        await ws.close(1013); return
    try:
        out_q: asyncio.Queue = session.bind(async_queue=out_q)
        # 雙任務：收音 / 發文字
        async def feed():
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect": break
                if "bytes" in msg and msg["bytes"]:
                    session.feed(msg["bytes"])      # 內部走 engine thread
        async def pump():
            while True:
                ev = await out_q.get()
                if ev is None: break
                await ws.send_text(json.dumps(ev, ensure_ascii=False))
        await asyncio.gather(feed(), pump())
    finally:
        await asyncio.to_thread(manager.release, session)
```

### 4.3 `session.py` 要點

- `Session.process_block(rms)` 原樣搬 `live_zh.LiveSTT.process_block`（引擎自斷句分支 + 安全網 flush + 300s 上限）。
- `Session.feed(chunk)`：算 RMS（`int16 → float / 32768` 的 RMS）→ `process_block` → `recognizer.feed(chunk)`。
- 回呼 `on_partial / on_utterance / on_final` 由 server 注入：`out_q.put_nowait({...})`（engine thread → asyncio queue 是 thread-safe 的）。
- `_flush_tail` 的重開邏輯（`_finish_session` + 0.3s 等待 + `_start_session`）在 engine 模式下：所有 ObjC 步驟走 `EngineThread.run()`；「0.3s 等待」用 event wait（engine thread tick 會持續遞交 callback，不需要 pump main）。

---

## 5. 測試計畫

### 5.1 自動化測試（`uv run pytest`，不需麥克風/授權）

| 檔 | 測試 | 對應驗收 |
|---|---|---|
| `test_engine_thread.py` | `run()` 的 fn 確實在 engine thread 執行（thread name 驗證）；Future 結果正確；逾時抛 `TimeoutError`；`shared()` 單例；`stop()` 後 queue 清空 | A1 |
| `test_engine_thread.py` | 長工作與短工作並行：掛一個 2s 長工作，期間 10 個短工作全部 < 100ms 完成（queue 不堆積） | A1、A3 |
| `test_session.py` | `SessionManager(max=3)`：第 4 個 `acquire` 抛 `SessionLimitExceeded`；release 後可再 acquire | A2 |
| `test_session.py` | `Session` 狀態機（mock recognizer）：partial → utterance → final 順序正確；mock 靜音 3s 觸發 flush；mock 300s 觸發重開並發 `session_restarted` | A6 |
| `test_vocab.py` | load_file（行/逗號/分號格式）；replace / merge；重複詞去重；> 200 條回 warning | A4 |
| `test_server.py`（TestClient） | `/health` 回 200 且含 `auth`、`sessions_active`、`max_sessions=3`、`default_lang=zh-TW` | A1、A5 |
| `test_server.py` | `/languages` 63 條 | A5 |
| `test_server.py` | `/vocabulary` GET/POST replace+merge | A4 |
| `test_server.py` | `/transcribe/raw`：合法 PCM 440Hz 音調 → 200 或 408（無語音時引擎報錯映射正確）；非法語言 → 400 | A5 |
| `test_server.py`（WS，`websockets`） | 連線 → 送 config → 送 10 幀 binary → 收到 `partial` 或 `final` 事件（440Hz 音調預期空文字 final，**驗證協定與回傳路徑**，不驗辨識內容） | A2、A3 |
| `test_server.py`（WS） | 第 4 條 WS 連線 → 收到 `too_many_sessions` + close(1013) | A2 |
| `test_server.py`（WS） | 不送 config 直接送 binary → close(1002) | 協定 |
| `test_server.py` | **不阻塞驗證**：開 3 個 mock session 各餵 2s 假音訊同時，`/health` 響應 < 200ms | A3 |

> 所有需要真实 SFSpeechRecognizer 的測試標 `@pytest.mark.live`，預設 skip；`uv run pytest -m live` 單獨跑。

### 5.2 live 測試（真機、需已授權語音辨識）

- **TTS 回灌**：`say -o /tmp/t1.aiff -v Ting-Ting "你好，商智中心"`（macOS 內建 TTS；注意 `say -o` 不支援 .wav 副檔名，用 .aiff）→ 餵進 WS session → 斷言 final 文字包含「商智中心」（模糊 match，容許標點）。
- **詞庫熱更新**：先 `POST /vocabulary {"words":["商智中心","BIRC"]}` → 新 session 播 TTS「商智中心 BIRC」→ 斷言辨識出兩個詞（未更新前對照組允許失敗）。
- **3 並發**：3 條 WS 同時餵不同 TTS 音頻 → 全部收到 final，且期間 `/health` < 200ms。
- **300s 重開**（可加速：config 設 `max_session_seconds=5`）→ 收到 `session_restarted` 且後續辨識正常。

### 5.3 驗收條件（Acceptance Criteria）

| # | 條件 | 驗證方式 |
|---|---|---|
| A1 | `macos-stt serve` 以 FastAPI+uvicorn 啟動；`GET /health` 200 且回傳 auth/sessions/default_lang/vocab_size | 自動（test_server）+ 手動 curl |
| A2 | 同時 3 個 WS session 各自收 partial/utterance/final；第 4 個被拒（503 等價 + 明確 error code）；釋放後可再建 | 自動 + live |
| A3 | 3 session 满载串流時 `/health` 仍 < 200ms（event loop 未被 RunLoop 鎖死） | 自動（mock 並發） |
| A4 | `POST /vocabulary` 熱更新後，**新** session 用新詞庫；live 對照驗證「商智中心」命中率提升 | 自動（store 行為）+ live（辨識內容） |
| A5 | 全域預設語言 `zh-TW`：不帶 `lang` 的請求按 zh-TW 處理；`/languages` 63 條；非法語言 400 | 自動 |
| A6 | WS session 300s 自動重開（`session_restarted`）；靜音 3s flush 尾句不卡死 | 自動（mock 加速）+ 手動長線 |
| A7 | 伺服器程序**只**要求「語音辨識」授權，不彈麥克風權限（WS 架構不收本地音訊） | 手動（首次執行觀察） |
| A8 | `live_zh.py` / `start.py` 本地麥克風模式行為不回歸（手動跑一次中文轉譯） | 手動 |
| A9 | `uv run pytest` 全部通過（live 標記另跑） | CI 本地 |

### 5.4 手動驗收腳本（部署後 10 分鐘）

```bash
# 1. 啟動（建議：經 app bundle，TCC 授權可用）
python start.py serve --lang zh-TW --vocab 詞表.txt --max-sessions 3
# 或（需已授權過；無 bundle 時首次會彈不出授權框）
macos-stt serve --lang zh-TW --vocab 詞表.txt --max-sessions 3

# 2. 健康
curl -s localhost:8765/health | python3 -m json.tool

# 3. 整段轉譯
curl -s -X POST localhost:8765/transcribe/raw -H 'content-type: application/json' \
  -d "{\"bytes\":\"$(base64 -i /tmp/t1.wav | tr -d '\n')\",\"sr\":16000}"

# 4. 詞庫熱更新
curl -s -X POST localhost:8765/vocabulary -H 'content-type: application/json' \
  -d '{"mode":"merge","words":["商智中心","BIRC"]}'

# 5. WS 串流（python 一頁腳本，吃 /tmp 的 PCM 或即時錄音）
python3 scripts/ws_probe.py --url ws://localhost:8765/stream --lang zh-TW
```

---

## 6. 風險與已識別問題

### 6.1 TCC 授權（已定案：MacSTT.app bundle 模式，D1）

macOS 要求語音辨識授權（`requestAuthorization`），**背景 daemon（launchd）彈不出授權對話框**，因此 server 一律走 `start.py` 的 MacSTT.app bundle 模式：

- `start.py serve [參數]`：建立/更新 bundle（`Info.plist` 加 `NSSpeechRecognitionUsageDescription`，**不需要** `NSMicrophoneUsageDescription`，WS 架構不收本地麥克風）→ `open` 前台啟動 uvicorn server。
- 首次前台運行會彈授權對話框；同意一次後系統持久記憶，之後同一 bundle 隨時可再啟動。
- 授權未給時：`GET /health` 回 `auth: "denied"/"not_determined"`，辨識端點回 501；部署方引導使用者到「系統設定 → 隱私權與安全性 → 語音辨識」檢查。
- 不提供 launchd daemon 模式；bundle 重建邏輯沿用 `start.py` 現有的 `ensure_app()`。

### 6.2 on-device 並發上限未公開

3 session 是保守值；若實測 3 路 on-device 已搶 ANE 導致延遲上升，`--max-sessions` 可調 2，或部分 session 開 `on_device=false`（需 `--allow-cloud`）。

### 6.3 音訊格式單一

v1 只收 16kHz int16 mono PCM。客戶端 44.1k/48k 必須先重取樣（瀏覽器端 `AudioWorklet`、Python 端 `sounddevice` 都已驗證可行）。不做伺服器端重取樣（AVAudioConverter 會再綁一個執行緒問題）。

### 6.4 WS 協定版本化

首版協定即 `v1`；config 加 `"protocol":"v1"` 欄位（可省略），日後破壞性變更靠版本號分流。

### 6.5 `websockets` 依賴

目前 `uv.lock` 無 `websockets`/`wsproto`——uvicorn 的 WS 支援是 optional。`[server]` 與 `[dev]` extra 需加 `websockets>=12` 並 `uv lock`。

### 6.6 安全（已定案：僅本機，D3）

**只監聽 127.0.0.1:8765**，無 `--host` 參數、無 LAN 存取，因此不需要 TLS/鑑權。
日後若需 LAN 接入（手機/其他專案跨機），另案處理：需防火牆、明文傳輸風險評估與最低限度 token 鑑權。

---

## 7. 實施順序（milestones）

| # | 里程碑 | 內容 | 出口標準 |
|---|---|---|---|
| M1 | engine thread 解耦 | `engine_thread.py`；`server.py` 重寫走 engine thread；`streaming.py` 適配 | A1、A3（mock 版）；現有 4 端點不回歸 |
| M2 | session 基礎 | `session.py`（Session + SessionManager，mock recognizer 可測）；`vocab.py` | A2（manager 層）、A4（store 層） |
| M3 | WS /stream | WS 端點 + 協定 + 錯誤路徑 | A2、A3（WS mock 版）、協定測試 |
| M4 | 詞庫 + 語言 + CLI | `--vocab`/`--lang`/`--max-sessions`/`--allow-cloud`；`start.py` serve 模式；`pyproject` 依賴 | A4、A5、A7 |
| M5 | live 驗證 + 文件 | live 測試全跑；README 更新（新端點、WS 協定、部署步驟）；`scripts/ws_probe.py` | A6、A8、A9 + §5.4 手動腳本全過 |

每個 milestone 完成即跑 `uv run pytest`（+ 該里程碑的 live 項），不累積債務。

---

## 8. 決定事項（已確認，2026-07）

| # | 項目 | 決定 | 影響章節 |
|---|---|---|---|
| D1 | 執行方式 | 沿用 `start.py` 的 MacSTT.app bundle 模式：`start.py serve` 在 bundle 內前台啟動 uvicorn；不做 launchd daemon | §3.4、§4、§6.1、§5.4 |
| D2 | 串流協定 | 只用 WebSocket，不做 SSE | §1.3、§3.2 |
| D3 | 監聽位址 | 僅本機 127.0.0.1:8765；無 `--host` 參數、無 LAN/TLS/鑑權 | §1.3、§4、§6.6 |
| D4 | 預設詞庫 | `vocab.default.txt` = `live_zh.py` 的 `DEFAULT_HINTS`（VS Code、Python、GitHub、Git、Mac、Terminal、Markdown） | §3.3、§4 |
| D5 | 並發 | `--max-sessions` 預設 3 | §2.3、§5、A2 |

---

## 9. 實施記錄（2026-07，M1–M5 完成）

### 9.1 交付物

| 檔案 | 狀態 |
|---|---|
| `src/macos_stt/engine_thread.py` | ✅ EngineThread 單例 daemon thread；`run()→Future`（看門狗 TimeoutError）＋ `register(EngineJob)` advance 模式 |
| `src/macos_stt/streaming.py` | ✅ `_dispatch(use_engine, on_main, block)`；`use_engine_thread` 參數（預設 False，`live_zh.py` 不回歸） |
| `src/macos_stt/session.py` | ✅ Session/SessionManager（VAD 安全網、300s 重開、max_sessions 門控） |
| `src/macos_stt/vocab.py` + `vocab.default.txt` | ✅ 詞庫 store（thread-safe、>200 warning）；7 詞內建詞庫 |
| `src/macos_stt/engine.py` | ✅ `transcribe_file_engine()` 非阻塞版（短工作建 task + EngineJob 輪詢 + 看門狗）；`_ensure_authorized()` 加 TCC 防護（見 9.2） |
| `src/macos_stt/server.py` | ✅ 完整重寫：8 REST 端點 + `WS /stream`；lifespan 管理 EngineThread；on_event 經 `loop.call_soon_threadsafe` 橋接 |
| `src/macos_stt/cli.py` | ✅ `serve` 加 `--lang/--vocab/--max-sessions/--allow-cloud`；未授權時印警告 |
| `start.py` | ✅ `serve` 子命令（共用 MacSTT.app bundle，launcher 讀 mode 分流）；live 模式向下相容（無子命令 = live）；serve 模式 uvicorn 輸出導向 `/tmp/macstt_server.log` |
| `scripts/ws_probe.py` | ✅ WS 手動探測腳本（任意音訊自動 afconvert） |
| `tests/` | ✅ 44 passed / 6 skipped（mock）＋ live 4/4（含 5 個真 uvicorn+WS 整合測試、4 個 live 測試；見 9.5） |

### 9.2 與原規劃的差異（實作中發現的真實問題）

1. **TCC abort 擴大到「任何 SFSpeechRecognizer API」**：原以為只有 `requestAuthorization_` 在無 Info.plist usage description 的 process 會 SIGABRT；實測 **`SFSpeechRecognizer.authorizationStatus()` 在 not-determined 狀態的非 bundle process 也會觸發 `__TCC_CRASHING_DUE_TO_PRIVACY_VIOLATION__`**（crash 報告證實）。→ `engine._ensure_authorized()` 加 `_has_speech_usage_description()` 防護：非 bundle 時**不 request**（只記錄現狀，後續以 501 回報）；bundle 內才 request。
   推論：**部署 server 必須走 `python start.py serve`（bundle 模式）**，直接 `macos-stt serve` 在非授權機上第一次碰引擎會暴死（已加啟動警告）。
2. **授權常數值**：`SFSpeechRecognizerAuthorizationStatusAuthorized` 實際是 **3**（1=Denied、2=Restricted、0=NotDetermined）——原測試 stub 用 1 會誤判，已修正為取框架常數。
3. **REST 同步辨識也走 EngineThread**（`transcribe_file_engine`），原規劃只提 WS 路徑——REST 不這樣會鎖 event loop 且非 bundle 環境卡 30s。
4. **Pydantic 模型放模組級**：`RawRequest`/`VocabRequest` 在 `create_app()` 內定義時 FastAPI 會誤解析成 query 參數（422 Field required），移到模組級解決。
5. **VAD 底噪污染**：連續常幅值合成音訊會污染 10 分位數底噪估計 → 測試改用 loud/quiet 交錯波形（真實語音特性）。

### 9.3 驗收狀態

| # | 條件 | 狀態 |
|---|---|---|
| A1 | FastAPI+uvicorn 啟動、/health 回 auth/sessions/default_lang/vocab_size | ✅ 自動 + smoke（curl 驗證） |
| A2 | 3 並發 WS + 第 4 個拒（1013）+ 釋放可再建 | ✅ 自動（真 uvicorn+WS 整合）；真引擎 3 並發見 9.5 |
| A3 | 3 session 串流時 /health < 200ms | ✅ 自動（test_ws_health_not_blocked） |
| A4 | 詞庫熱更新 → 新 session 用新詞庫 | ✅ 自動（store 層）；live 對照待授權環境跑 `MACSTT_LIVE=1 uv run pytest -m live` |
| A5 | 預設 zh-TW、63 語言、非法 400 | ✅ 自動 |
| A6 | 300s 重開（session_restarted）+ 靜音 flush | ✅ 自動（mock 加速）；live 版在 test_live.py |
| A7 | 只需 Speech 權限、不需麥克風權限 | ✅ 設計上成立（server 不收本地音訊）；首次授權對話框需真機確認 |
| A8 | live_zh.py 本地模式不回歸 | ✅ `use_engine_thread` 預設 False，簽名相容（import 驗證）；真機麥克風測試待做 |
| A9 | uv run pytest 全綠 | ✅ 44 passed / 6 skipped；`MACSTT_LIVE=1` live 4/4 |

### 9.5 live 收斂實證（2026-11-02）

1. **callback 遞交模型**：SFSpeechRecognizer 的 callback 固定遞交在**主執行緒** RunLoop；`server.py::_pump_main_runloop()` 每 ~14ms pump 4ms 作接收端。EngineThread 的角色是「ObjC 操作序列化執行緒」，不負責接 callback。
2. **EngineThread 空轉（啟動僵死根因）**：engine thread 自己的 RunLoop 沒有 source 時 `runMode` 立即返回，`_loop` 原本只在 `except` 才 sleep → 100% CPU 空轉並以 pyobjc 呼叫霸佔 GIL，uvicorn startup 餓死、`/health` 不上線。修正：每圈補等至 `_TICK_SECONDS`（以 `_wake` Event 等待，新短工作即喚醒）。回歸測試 `test_idle_does_not_spin`（修正前閒置 1s 耗 1.39s CPU）。
3. **引擎單槽**：zh-TW on-device 同時建 2–3 個 recognition task 時只有最先建立的收得到事件。真引擎模式下 `SessionManager` 單槽輪替（等待者緩衝最多 80 幀，持槽者 flush/close 時輪替並重放）；mock 模式不啟用。
4. **輪替鏈自我死鎖**：`_become_active` 持 `_promote_lock`（非重入）並在內部 `_flush_tail`；重放後無人可輪替時 `_flush_tail` 重入 `_become_active` → 同 thread 重複取鎖永久卡死（final 已送出但 feed thread 卡住、`sessions_active` 殘留、uvicorn 關閉卡「Waiting for background tasks」）。修正：`RLock`；另加 slot ownership guard（排隊搶鎖期間槽被輪走則不再啟動，避免疊出第二個 task）。回歸測試 `test_slot_rotation_chain_no_deadlock`（`_slot=True` 的 mock，修正前死鎖）。
5. **VAD 修正**：疑似語音幀不加入底噪視窗（開頭即語音的 clip 原本永遠偵測不到語音）。
6. **測試音檔**：`say -o *.aiff`、`afconvert -d LEI16@16000`（`-r` 被忽略）；final 只在 VAD flush 或重開時保證，`_stream_pcm` 收尾 3.2s 靜音。
7. **REST 與單槽**：REST `/transcribe*` 納入單槽，串流持槽或另一個 REST 進行中回 503（實測重疊會讓串流收不到 final）；`acquire` 不再持鎖啟動 recognizer。
8. **驗證結果**：live 4/4；手動 3 並發 ×3 輪共 9/9 條皆收到 final 且內容正確、`/sessions` 歸 0、SIGTERM 1s 內退出、server log 無錯誤、閒置 CPU 約 3%。

### 9.4 剩餘待真機確認（需已授權的 macOS 環境）

```bash
# 1. 首次啟動（會彈語音辨識授權對話框 → 同意）
python start.py serve

# 2. TTS 回灌串流測試（say -o 不支援 .wav，用 .aiff）
say -o /tmp/t1.aiff -v Ting-Ting "你好，商智中心"
python scripts/ws_probe.py --audio /tmp/t1.aiff

# 3. live 測試全套
MACSTT_LIVE=1 uv run pytest -m live -v
```
