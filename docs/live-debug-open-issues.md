# Live 測試問題清單（2026-11-02 已收斂）

> 前次暫停點的兩個未解問題皆已修正並實證；完整實證記錄在
> `docs/fastapi-server-plan.md` §9.5。本檔只保留結論與重現備忘。

## 結論

| 問題 | 根因 | 修正 | 回歸測試 |
|---|---|---|---|
| B：伺服器啟動僵死（CPU 100%、`/health` 不上線） | `engine_thread._loop` 在自己的 RunLoop 無 source 時 `runMode` 立即返回，而 `sleep` 只在 `except` 分支 → 空轉霸佔 GIL，餓死 uvicorn | 每圈補等至 `_TICK_SECONDS`（`_wake` Event，新工作即喚醒） | `test_engine_thread.py::test_idle_does_not_spin` |
| A：3 並發輪替未收斂 | `Session._become_active` 持非重入 `_promote_lock`，內部 flush 在無人可輪替時重入 `_become_active` → 自我死鎖（feed thread 卡死、session 殘留、關閉卡住）；另有排隊搶鎖後槽已輪走仍再啟動的競態 | `RLock` ＋ slot ownership guard | `test_session.py::test_slot_rotation_chain_no_deadlock` |

**不需降為 1 並發**：單槽輪替在真引擎下穩定。

## 驗證結果

- mock：44 passed / 6 skipped。
- live（`MACSTT_LIVE=1 uv run pytest tests/test_live.py`）：4/4。
- 手動 3 並發 × 3 輪：9/9 條皆收到正確 final；`/sessions` 歸 0；SIGTERM 1s 內退出；log 無錯誤。

## 重現／除錯備忘

- 非 bundle process 碰授權相關 API 會 TCC SIGABRT → 一律經 `open MacSTT.app`（`start.py serve` 已包）。
- 後端 stdout → `/tmp/macstt_server.log`；PID 檔 `/tmp/macstt_pid`。
- 懷疑空轉：`sample <pid> 2`；看 CPU：`ps -o %cpu= -p <pid>`（閒置約 3%，主要是 14ms 的主 RunLoop pump）。
- live fixture teardown 會 `pkill` 所有 macstt process；若 server 不理 SIGTERM，下一輪 `open` 不會起新 process（先檢查 `/sessions` 是否有殘留 session）。

## 後續收斂（同日）

- **REST × 串流互斥**：實測 REST `/transcribe*` 與串流同時跑，REST 回 200 但串流只收到 2 個 partial、無 final。現在 `SessionManager.try_acquire_engine()/release_engine()` 把 REST 納入單槽：串流 session 持槽（含閒置連線）或另一個 REST 進行中 → REST 回 **503**（`Retry-After: 2`）；REST 進行中新建的 session 先當等待者，REST 結束後下一幀拿槽。REST 逾時時一併取消 task，不留殘存 task。
- **`acquire` 不再持 `_lock` 啟動 recognizer**：先以等待者狀態登記名額／槽，慢的 `rec.start()` 移到鎖外，`/health` 的 stats 不被擋；啟動失敗會釋放名額與槽。
- 回歸測試：`test_session.py`（慢啟動不擋 stats、啟動失敗釋放名額、REST 借用與串流互斥）、`test_server.py::test_transcribe_busy_when_stream_holds_engine`。
- 驗證：mock 48 passed / 6 skipped；live 4/4；真引擎 REST × WS 衝突情境 REST 回 503、串流收到 partial＋final。

## 取捨

- 串流連線只要開著（即使閒置）就持有引擎槽，REST 會一直 503，直到所有串流連線關閉。若需要「閒置時允許 REST」，要另做槽的讓出機制（閒置逾時釋槽），尚未實作。
