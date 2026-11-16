# 2 GB 記憶體預算演練（budget.md）

**一句話結論：** 2 GB 版本**沒有換方案**，沿用 8 GB 的預設 C-FT（BGE 聯合微調）。實測在 2 GiB 記憶體上限、不使用 swap 的 Docker 容器裡，記憶體峰值最高 518 MiB，3 次完整 2,000 筆全部通過，約 26 秒；輸出和 8 GB 版本逐位元相同，**dev 四欄都沒有掉分**。如果現場可用的記憶體再更少，備案是改用方案 A（約 30 MiB，總分少 0.035）。

方案 A、B、C 各是什麼，見 [bench.md](bench.md) 開頭的「先認識這 7 個方案」。

## 怎麼跑

```bash
python3 predict.py data/tickets.jsonl tickets_predictions_2gb.jsonl --mem-limit 2g
```

`--mem-limit 2g` 不換模型，方法選擇和兩參數預設相同（C-FT）；在 Linux 上會另外把程序的位址空間限制在 2 GiB。正式驗證用 Docker，一支指令重現（需要已安裝 Docker Desktop，並先執行過 `python3 install.py`）：

```bash
bash scripts/docker_2gb_check.sh
```

這支腳本會 build 映像，在 `--memory=2g --memory-swap=2g --cpus=2 --network=none` 的容器裡跑：一次超量探測、3 次完整 tickets、1 次 dev，結果寫到 `evidence/<日期>-docker-2gb/`。

---

## HW_RULES 要求的五項

### 1. 改了什麼

**沒有換方案。** 先量再決定：唯一吃記憶體的是分類模型，C-FT 在容器內的峰值是 428～518 MiB，只用了 2 GiB 的四分之一左右，不需要為了記憶體犧牲分數。`--mem-limit 2g` 只加上記憶體上限，不改模型、不改流程。

### 2. dev 四欄掉多少分

| 欄位 | 8 GB 預設（C-FT） | 2 GB 版本（C-FT，Docker 2 GiB） | 差值 | 備案：方案 A |
|---|---:|---:|---:|---:|
| category（macro-F1） | 0.898 | 0.898 | 0 | 0.849（−0.049） |
| device_id（accuracy） | 0.990 | 0.990 | 0 | 0.993（+0.003） |
| urgency（macro-F1） | 1.000 | 1.000 | 0 | 1.000（0） |
| duplicate_of（F1） | 0.934 | 0.934 | 0 | 0.853（−0.081） |
| 總分 | 0.944 | 0.944 | **0** | 0.909（−0.035） |

2 GB 版本的 dev 分數是在 Docker 2 GiB 容器裡實際預測、再用官方 `baseline/evaluate.py` 評分得到的（容器峰值 393 MiB）。

### 3. 2 GB 版本跑 tickets 的時間

Docker 2 GiB 容器，每次都是新容器、無預熱；總時間是容器內包住整個 `predict.py` 程序的時間（含 Python 啟動、模型載入、推論、寫檔，不含 Docker 啟動）。

| 次 | 結束碼 | 總時間（秒） | P50（ms） | P95（ms） | 最大（ms） | 容器記憶體峰值（MiB） |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 26.9 | 43.1 | 137.2 | 210.7 | 517.9 |
| 2 | 0 | 25.7 | 44.3 | 136.2 | 213.2 | 434.8 |
| 3 | 0 | 25.8 | 44.5 | 138.8 | 200.3 | 428.0 |

平均每筆約 13 ms，**0.24 秒／筆的預算有守住**（2,000 筆上限 480 秒，實際約 26 秒）。P95 約是 P50 的 3 倍，原因和 8 GB 版本相同：C-FT 以批次推論，每筆記的是整批的共用時間，是上界（見 bench.md）。

### 4. 為什麼這樣取捨

- **先量記憶體，再決定要不要砍。** 題目提醒「很可能要換模型」，但實測後 C-FT 在 2 GiB 內還有約 1.5 GiB 餘裕，換掉只會白白掉分。
- **如果真的要砍，第一個砍分類模型。** 設備、急迫性、重複報修都是規則，幾乎不佔記憶體；佔記憶體的只有分類模型。換成方案 A 的關鍵詞規則後，峰值降到約 30 MiB，代價是 category −0.049、duplicate −0.081（重複報修的比對鍵包含類別，分類錯了就找不到前一張），總分 −0.035。
- **什麼時候切到備案：** 這次量的是單一程序在 2 GiB 容器內的上限。如果現場的 2 GB 還要扣掉作業系統與其他服務，留給推論的記憶體低於約 1 GB 時，改用 `python3 predict.py <輸入> <輸出> --method rules --mem-limit 2g`。
- **修正紀錄：** 這份文件最早的版本把 2 GB 版本換成方案 A，因為當時 C-FT 還沒有 2 GB 的實測證據。補完 Docker 實測後，確認不需要換，所以改回 C-FT。

### 5. 有沒有放棄某一欄

沒有。四欄都保留，沒有欄位固定回 `null`；`urgency` 仍是三選一。

---

## 怎麼確認真的在 2 GB 上限下跑

證據在 [evidence/2026-11-16-docker-2gb](evidence/2026-11-16-docker-2gb/README.md)。

1. **容器設定**：`docker inspect` 讀回 Memory＝MemorySwap＝2147483648、NanoCpus＝2000000000、NetworkMode＝none；容器內 cgroup v2 讀到 `memory.max=2147483648`、`memory.swap.max=0`、`cpu.max=200000 100000`。
2. **限制確實生效**：同樣限制下，一支故意使用 2.5 GiB 的程式被系統強制終止（exit 137、OOMKilled=true）。
3. **正式執行沒有碰到上限**：3 次 tickets 與 1 次 dev 都是 exit 0，`memory.events` 的 oom／oom_kill 都是 0。
4. **輸出就是交付檔**：3 次輸出逐位元相同，SHA-256 與 repo 的 `tickets_predictions_2gb.jsonl`、`tickets_predictions.jsonl` 都相同。
5. **另一種限制方式也通過**：在 Linux VM 用 RLIMIT_AS（程序位址空間硬上限）2 GiB 再跑 3 次，同樣全部通過，見 [evidence/2026-11-16-linux-vm](evidence/2026-11-16-linux-vm/README.md)。

**測試環境：** Mac（Apple Silicon）上的 Docker Desktop 29.6.2，容器為 linux/arm64；映像由 `docs/testing/Dockerfile.finetuned` 建置。

## 限制

- 2 GiB 是**容器**的上限，不是整台機器只有 2 GB；Docker 自己的 VM 與作業系統不算在內。
- 容器記憶體峰值（cgroup `memory.peak`）會把檔案快取也算進去，口徑和程序 RSS 不同，所以和 bench.md 的「峰值記憶體」數字不能直接比較。
- 掉分只能用 dev 量；tickets 沒有答案，只交輸出檔。
- 備案方案 A 的 Docker 2 GiB 驗證（3 次約 2.1 秒、峰值約 30 MiB、輸出與原生逐位元相同）是 2026-11-13 的紀錄，見 [Solution/budget.md](Solution/budget.md)。
