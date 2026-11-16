# C-FT 在 Docker 2 GiB 容器的量測證據

由 `scripts/docker_2gb_check.sh` 產生（2026-11-16）。環境：Mac（Apple Silicon）上的 Docker Desktop 29.6.2，容器 linux/arm64；映像 `sha256:fb1edd4cd00e…`，由 `docs/testing/Dockerfile.finetuned` 建置。限制：`--memory=2g --memory-swap=2g --cpus=2 --network=none`。

| 項目 | 結果 |
|---|---|
| 超量探測（配置 2.5 GiB） | exit 137，OOMKilled=true，限制生效 |
| tickets 3 次 | 全部 exit 0；總時間 25.7～26.9 秒；容器峰值 428.0～517.9 MiB；P95 136.2～138.8 ms |
| 輸出 | 3 次逐位元相同，與 `tickets_predictions.jsonl`、`tickets_predictions_2gb.jsonl` 相同 |
| dev 1 次 | exit 0，容器峰值 392.9 MiB；category 0.898、device 0.990、urgency 1.000、duplicate 0.934、總分 0.944 |
| OOM 事件 | 所有正式執行的 `memory.events` oom／oom_kill 都是 0 |

## 檔案

- `summary.json`：環境、映像、限制讀回值、探測結果、每次執行的彙總與 dev 分數。
- `runs.json`：3 次 tickets 執行的完整紀錄，含 2,000 筆逐筆秒數（只有 ticket_id，沒有工單內容）。

原始預測與 log 留在本機的 `reports/docker-2gb/`，不進 Git。
