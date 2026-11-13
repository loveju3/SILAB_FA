# 方案 A：2 GB 演練

**分數定位：** 本文沿用的 A 曾在開發前期參考 dev，dev 分數僅為歷史開發驗證紀錄，不是獨立測試成績。2 GB 上限與效能實測仍保留其原有量測意義；新算法的資料使用遵循 [AGENTS.md](../AGENTS.md)。

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

## 方案 B：LR 與 LinearSVC 的 2 GB 追加驗證

2026-11-13 已建置 `Solution/Dockerfile.ml` 為 `silab-fa:scheme-b`，固定 requirements-ml.txt 依賴並使用本機 history 訓練資產。使用同樣 2 GiB、無 swap、2 CPU、無網路的容器限制；`--pull=never` 禁止推論時拉取映像。

LR 測試先執行獨立 OOM probe，結果為退出 137、OOMKilled=true；兩種方法的每次正式執行皆驗證 memory.max=2147483648、memory.swap.max=0、退出 0 且未 OOM。

| 方法 | 三次完整 tickets 總時間 s | P50 ms 範圍 | P95 ms 範圍 | 最大 ms | 容器 memory.peak MiB |
|---|---|---|---|---:|---|
| B-LR | 10.471～10.501 | 5.081～5.153 | 7.310～7.473 | 15.435 | 86.81～87.38 |
| B-SVC | 10.356～10.733 | 5.016～5.186 | 7.323～7.759 | 18.397 | 86.61～86.84 |

時間範圍包含 Python 子程序冷啟動、模型與依賴載入、推論及輸出，不含 Docker 啟動；逐筆範圍同 bench.md。每次新容器，無顯式預熱，OS 快取未清除。兩方法均保留完整模型與四欄能力；各自三次預測與原生 macOS 的三次預測逐位元相同。不同作業系統的 RSS 與 cgroup 峰值口徑不同，不直接比較數字高低。

本次 runner 預設以 history 檢查可執行性，再跑完整 tickets；完整 history 已用於重訓，故不把這次 history 推論當成泛化評分。dev 未提供且未執行，**尚無 B 的 dev 分數與 2 GB 掉分數據**，不能用 tickets 的一致性代替 dev 掉分量測。

```bash
docker build -f Solution/Dockerfile.ml -t silab-fa:scheme-b .
.venv/bin/python Solution/scripts/run_budget.py --method tfidf-lr --probe
.venv/bin/python Solution/scripts/run_budget.py --method tfidf-svc
```

原始紀錄：`experiments/runs/budget-tfidf-lr-287e1140ed0c/`、`experiments/runs/budget-tfidf-svc-80aec98fef41/`。目前 runner 預設 history；若要在版本固定後重現 A 舊紀錄的 dev 步驟，需明確指定 `--validation-dataset dev`，不在開發過程預設讀取 dev。

## 方案 C：完整 BGE encoder 與分類頭的 2 GB 驗證

2026-11-13 已建置 `Solution/Dockerfile.embedding` 為 `silab-fa:scheme-c`。保留完整 float32 BGE-small-zh-v1.5、512 維 CLS＋L2、batch size=16、兩個 CPU 執行緒及原有四欄管線，沒有量化、刪除模組或退回規則版。PyTorch 使用 CPU wheel；推論時禁止連網及拉取映像。

LR 的獨立超限 probe 退出 137 且 OOMKilled=true，證明限制可觸發。兩方法各執行一次 history 可執行性檢查及三次完整 2,000 筆 tickets：全部退出 0、OOMKilled=false；每次查驗 memory.max=2147483648、memory.swap.max=0、cpu.max=200000 100000，正式推論的 cgroup OOM 計數為 0。

| 方法 | 三次 tickets 完整程序時間 s | 每筆服務 P50 ms 範圍 | P95 ms 範圍 | 最大 ms | 容器 memory.peak MiB |
|---|---|---|---|---:|---|
| C-LR 保留名稱 | 29.625～30.027 | 104.860～107.889 | 288.484～299.787 | 384.960 | 499.88～508.50 |
| C-SVC 保留名稱 | 29.219～29.780 | 97.658～101.843 | 281.605～293.884 | 340.022 | 497.04～521.42 |

總時間包含子程序冷啟動、依賴與模型載入、真正 encoder 運算、規則及輸出，不含 Docker 啟動。每次使用新容器，無顯式預熱，不清 OS 快取。**每筆服務時間包含所在分類批次的完整共享時間，不能相加，也不能直接與 B 的單筆獨立分類 P50／P95 比較。** 定義見 [bench.md](bench.md)。容器 memory.peak 包含 wrapper、子程序及計入 cgroup 的快取，與 macOS 程序 RSS 口徑不同。

兩方法均在 480 秒預算內完成。各自三次容器輸出與三次原生輸出逐位元相同；整理預測留在 `predictions/bge-lr_tickets.jsonl`、`predictions/bge-svc_tickets.jsonl`。這證明本次 tickets 輸出一致，不能替代有標籤資料的掉分驗證。完整 history 已用於重訓，此處不把 history 回測當成泛化分數；本次沒有 dev 資料，沒有 C 的 dev 分數或受限環境掉分數據。

```bash
docker build -f Solution/Dockerfile.embedding -t silab-fa:scheme-c .
.venv/bin/python Solution/scripts/run_budget.py --method bge-lr --probe
.venv/bin/python Solution/scripts/run_budget.py --method bge-svc
```

正式原始紀錄：`experiments/runs/budget-bge-lr-818edeb29820/`、`experiments/runs/budget-bge-svc-b273098b153f/`，含 limits、metrics、預測與 docker inspect 選定欄位。之後 CLI 說明文字微調並重新建置，推論邏輯與權重沒有改動；各次實際映像 ID 與程式雜湊仍以其原始紀錄為準。

另以最終映像預設 CMD 執行不掛載原始碼的離線 smoke test，只掛載唯讀 tickets 輸入及結果資料夾，仍在 2 GB／禁止 swap／2 CPU 下成功處理 2,000 筆，輸出與原生 C-LR 逐位元一致。結果在 `reports/c-image-smoke/`。最終映像 manifest-list digest 為 `sha256:7ab33c3acc456d9a0f4a9725171e63554ed4047514f49e771cf9e65c757ca37e`。

UI 另以兩筆自行撰寫的合成資料驗證 C-LR 本機及 C-SVC 2 GB 的 API：輸出檔名、分類 macro-F1、完整／錯題結果及總／預測時間均可回傳；紀錄在 `reports/c-ui-smoke/`，不把合成資料的分數當成模型成績。開發測試 52 項中 50 項通過、2 項 dev 驗證預設略過；前端 JavaScript 語法及 diff whitespace 檢查通過。
