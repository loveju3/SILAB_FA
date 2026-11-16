# 2 GB 記憶體預算演練（摘要）

完整證據、Docker 指令與方案 B／C 的 2 GB 追加驗證見 [Solution/budget.md](Solution/budget.md)。本頁依 HW_RULES 要求的五項整理。

## 怎麼跑

> **2026-11-16 可重現證據：** `scripts/collect_evidence.py` 在 2 GiB 硬上限下重跑了 A 與 C-FT 各三次完整 2,000 筆，全部通過，並證明限制會擋下 2.5 GiB 配置；每次的時間、逐筆秒數與輸出雜湊在 [evidence/2026-11-16-linux-vm](evidence/2026-11-16-linux-vm/README.md)。

```bash
python3 predict.py data/tickets.jsonl tickets_predictions_2gb.jsonl --mem-limit 2g
```

`--mem-limit 2g` 固定使用**方案 A（完整規則）**；兩參數的預設行為不變（預設仍是 BGE 聯合微調）。在 Linux 上這個模式另以 `RLIMIT_AS` 把程序位址空間限制在 2 GiB 作為保護；**正式的 2 GB 驗證是 Docker cgroup v2 容器**（`--memory=2g --memory-swap=2g --cpus=2 --network=none`，讀回 `memory.max=2147483648`、`memory.swap.max=0`，並以超限 probe 觸發 OOMKilled 證明限制生效）。重現：

```bash
docker build -f Solution/Dockerfile -t silab-fa:scheme-a .
python3 Solution/scripts/run_budget.py --probe
```

`python3 run_demo.py` 會同時產生 `tickets_predictions_2gb.jsonl` 與 2 GB 模式的 dev 評分。

## 1. 改了什麼

8 GB 預設用 BGE 聯合微調（encoder＋分類頭，峰值 RSS 約 570–590 MiB）；2 GB 版本換成方案 A：只有 Python 標準函式庫與小型規則資產，容器 `memory.peak` 約 30 MiB。設備、urgency、重複報修的規則管線兩者共用，差別只在**故障分類**改由關鍵詞規則決定。

## 2. dev 四欄掉多少分（相對 8 GB 預設）

| 欄位 | 8 GB 預設（C-FT） | 2 GB（A） | 差值 |
|---|---:|---:|---:|
| category macro-F1 | 0.898 | 0.849 | **−0.049** |
| device_id accuracy | 0.990 | 0.993 | +0.003 |
| urgency macro-F1 | 1.000 | 1.000 | 0 |
| duplicate_of F1 | 0.934 | 0.853 | **−0.081** |
| 加權總分 | 0.944 | 0.909 | −0.035 |

A 在 2 GiB 容器內與原生執行的 dev／tickets 預測逐位元相同，所以掉分全部來自「換方案」，不是記憶體壓力造成的。（A 的 dev 分數有曾觀察 dev 的來源限制，見 [AGENTS.md](AGENTS.md)。）

## 3. 2 GB 版本跑 tickets 的時間

| 執行 | 總時間（秒） | P50（ms） | P95（ms） | 最大（ms） | memory.peak（MiB） |
|---|---:|---:|---:|---:|---:|
| 1 | 2.140 | 0.924 | 1.375 | 9.092 | 29.9 |
| 2 | 2.161 | 0.932 | 1.457 | 6.282 | 30.0 |
| 3 | 2.129 | 0.921 | 1.288 | 10.526 | 30.0 |

2,000 筆約 2 秒，遠低於 480 秒；0.24 秒／筆的預算有守住。量測口徑：每次新容器、無預熱、perf_counter 包住預測子程序（含 Python 啟動與載入，不含 Docker 啟動）；逐筆 P50／P95 為 nearest-rank。

## 4. 為什麼這樣取捨

- **先砍分類模型，因為它是唯一吃記憶體的部分**；設備、urgency、重複報修本來就是規則，搬到 2 GB 不必犧牲。
- 掉最多的是 duplicate（−0.081）：重複報修的比對鍵包含設備與故障類別，分類一錯就找不到前一張；A 的口語分類召回不足會連鎖拉低 duplicate。
- C-FT 的峰值約 570–600 MiB。2026-11-16 已在 2 GiB 位址空間硬上限（RLIMIT_AS）下跑完三次完整 2,000 筆（約 23 秒，見 [evidence](evidence/2026-11-16-linux-vm/README.md)），但還沒有 Docker cgroup 容器的驗證；A 則有完整容器驗證且峰值約 30 MiB。2 GB 版本仍選 A：證據最完整、餘裕最大，也是「最壞情況」的保底。固定 BGE＋LR／SVC 已在 2 GiB 容器跑完 2,000 筆（峰值約 500 MiB，見 [Solution/budget.md](Solution/budget.md#方案-c完整-bge-encoder-與分類頭的-2-gb-驗證)），但它們沒有 dev 分數，因此不拿來當 2 GB 版本。

## 5. 有沒有放棄某一欄

沒有。四欄都保留，沒有欄位固定回 `null`；`urgency` 仍三選一。

## 下一步

在 Docker cgroup v2 容器（`--memory=2g --memory-swap=2g`）跑 C-FT 的完整 2,000 筆與 dev；若通過，2 GB 版本可直接沿用 C-FT，不必為記憶體換方案、也就不必掉分。
