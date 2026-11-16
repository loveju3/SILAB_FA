# Part 1 量測證據

由 `scripts/collect_evidence.py` 產生，2026-11-16T08:27:53+00:00。每列為新 Python 程序、無預熱、不清 OS 快取；總時間由外部 `perf_counter()` 量整個程序（含 Python 啟動、模型載入、推論、寫檔）。逐筆時間的定義見各 `runs/*.json` 的 `latency_definition`。

- 環境：Linux-6.8.0-138-generic-aarch64-with-glibc2.35，4 CPU，RAM 3.8 GiB，Python 3.13.16
- 程式版本：`0e879a7f3ee15a8257ebe05ae200e064e8f7b571`（乾淨）
- 輸入：data/tickets.jsonl，SHA-256 `1cc95e29378e5143…`

| 方法 | 記憶體限制 | 通過 | 總時間（秒） | 平均每筆（ms） | P50（ms） | P95（ms） | 最大（ms） | 峰值 RSS（MiB） | 三次輸出相同 | 與 repo 交付檔相同 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| bge-finetuned | 無 | 3/3 | 23.32～24.86 | 11.7～12.4 | 40.8～43.6 | 124.8～135.4 | 155.0～194.6 | 570.0～598.0 | 是 | 是 |
| bge-finetuned | RLIMIT_AS 2 GiB | 3/3 | 23.10～23.64 | 11.6～11.8 | 40.2～41.1 | 123.9～128.3 | 154.5～189.3 | 567.5～578.2 | 是 | — |
| rules | 無 | 3/3 | 1.89～1.96 | 0.9～1.0 | 0.9 | 1.1～1.4 | 6.4～7.2 | 26.6 | 是 | — |
| rules | RLIMIT_AS 2 GiB | 3/3 | 1.88～1.94 | 0.9～1.0 | 0.9 | 1.1～1.2 | 6.5～27.6 | 26.6 | 是 | 是 |

題目門檻：2,000 筆總時間 ≤ 480 秒（平均每筆 ≤ 240 ms）；2 GB 版本須在 2 GB 上限下完成。

## 2 GiB 限制確實生效

在相同 RLIMIT_AS 下嘗試配置 2.5 GiB：被拒絕（exit 1，`MemoryError`）；子程序讀回的限制為 `(2147483648, 2147483648)`。

RLIMIT_AS 限制的是位址空間，永遠 ≥ 實際佔用的記憶體，所以在它之下跑完代表實際記憶體從未超過 2 GiB；這比 Docker 的 cgroup 限制更嚴格。Docker cgroup v2 的方案 A 驗證見 [Solution/budget.md](../../Solution/budget.md)。

## 檔案

- `summary.json`：環境、各組彙總與限制探測。
- `runs/<方法>.json`：每次執行的完整紀錄，含 2,000 筆逐筆秒數（只有 ticket_id，無工單內容）。
