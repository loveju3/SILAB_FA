# 方案 A：2 GB 演練

2026-11-13 已完成 Linux ARM64 Docker 容器實測。算法沒有縮減，也沒有放棄欄位；A 只有標準函式庫與小型規則資產，直接保留完整能力。

## 限制與證據

- Docker Engine 29.6.2，cgroup v2，`--memory=2g --memory-swap=2g --cpus=2 --network=none`。
- 實際讀取 `memory.max=2147483648`、`memory.swap.max=0`、`cpu.max=200000 100000`。
- `docker inspect` 同時確認 Memory、MemorySwap 均為 2147483648、NanoCpus=2000000000、NetworkMode=none。
- 獨立 probe 逐步配置並觸及最多 2.5 GiB，得到退出碼 **137**、**OOMKilled=true**，確認硬限制生效。probe 與正式推論為不同容器。
- dev 與三次完整 tickets 均退出 0、OOMKilled=false、`memory.events` 中 oom/oom_kill 均為 0。

這是 2 GiB 容器上限，並非整台主機只有 2 GB；Docker VM 與宿主成本不在容器上限內。沒有使用 GPU、MPS 或網路。

## 標準映像重試結果

2026-11-13 重新執行 `docker build -f Solution/Dockerfile -t silab-fa:scheme-a .` 已成功下載標準 `python:3.13-slim` 並建置完成，容器內 Python 為 3.13.18。本節分數、限制證據與效能表均已改為標準映像重跑結果。

標準基礎映像 digest：`sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90`。

本次建置映像 ID：`sha256:5650654008c82d978ba028087e598039ddd89cd977c0d9e577586ac3be1b15a0`。

限制 runner 只讀掛載專案至 /app，僅實驗結果目錄可寫；另外已執行不掛載專案的映像預設 CMD，確認 Dockerfile 包入的程式、規則與資料本身也能離線完成 2,000 筆預測，且輸出與原生版逐位元一致。

首次 registry metadata 逾時後，曾以本機 `llm_project_cloud-backend:latest` 作替代驗證；舊結果仍保留在 `experiments/runs/budget-bd361ccaf608/` 及摘要的 previous_budget_runs，供追溯。現在重現不需該映像。

## 重現

本次已成功執行：

```bash
docker build -f Solution/Dockerfile -t silab-fa:scheme-a .
python3 Solution/scripts/run_budget.py --probe
```

runner 為每次執行建立唯一容器，保存 inspect 選定欄位後刪除自己建立的容器；不操作既有容器。probe 預期為 OOM，正式推論任何非零退出都視為失敗。

## 分數差異

| 指標 | 原生 A | 2 GB A | 差值 |
|---|---:|---:|---:|
| category macro-F1 | 0.848614 | 0.848614 | 0 |
| device accuracy | 0.993333 | 0.993333 | 0 |
| urgency macro-F1 | 1.000000 | 1.000000 | 0 |
| duplicate ID F1 | 0.853333 | 0.853333 | 0 |
| 總分 | 0.908779 | 0.908779 | 0 |

dev 及 tickets 的受限／原生預測檔逐位元相同。沒有刻意降低能力來製造取捨；目前真正取捨是保留低成本規則，接受口語分類召回不足，未加入昂貴 encoder。

## 2,000 筆效能

| 執行 | Python 子程序總秒數 | P50 ms | P95 ms | 最大 ms | 容器 memory.peak MiB |
|---|---:|---:|---:|---:|---:|
| 1 | 2.139700 | 0.923708 | 1.375336 | 9.091748 | 29.890625 |
| 2 | 2.161065 | 0.932001 | 1.457416 | 6.281583 | 29.988281 |
| 3 | 2.128644 | 0.921208 | 1.288040 | 10.526290 | 29.992188 |

每次新容器、無顯式預熱，OS 快取未清除。容器內 wrapper 用 perf_counter 包住預測子程序，包含 Python 啟動、載入、讀寫與量測輸出；不含 Docker 啟動。逐筆範圍與 nearest-rank 方法同 bench.md。峰值來自同容器 cgroup 的 memory.peak，包含監測 wrapper、子程序與計入該 cgroup 的快取。符合 480 秒預算；原生與容器 CPU 配額不同，不用兩組時間判斷限制對速度的影響。

原始證據：`experiments/runs/budget-9d1979fec3ba/`，包含 docker_runs.json、每次 limits/metrics JSON 與預測。不掛載原始碼的 smoke test 產物位於 `reports/standard_image_smoke/`。整理輸出為 `predictions/tickets_predictions_2gb.jsonl`。本次完成 A 的資源驗證；之後 B/C 是否保留完整模型，仍須各自重測。
