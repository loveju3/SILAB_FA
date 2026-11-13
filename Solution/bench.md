# 方案 A：CPU 量測紀錄

**歷史結果限制：** A 在新增「dev 僅供驗證」規範之前曾觀察 dev；下列分數是既有開發驗證紀錄，不是未觀察 dev 的獨立測試成績。後續算法遵循 [AGENTS.md](../AGENTS.md)，不以本報告的 dev 錯題或分數調整設計。

實測日期：2026-11-13。主機 Apple M4 Max、14 核、36 GiB RAM，macOS ARM64、Python 3.13。純 Python、無 GPU/MPS、無模型載入及預熱。**本節為原生 macOS 未施加記憶體上限測試，未在實體 8 GB 機器驗證。** 受限 Linux ARM64 結果另見 budget.md，不跨環境推論速度差異。

## 正確性

原樣呼叫官方評分器；完整 dev 300 筆。

| 方案 | category macro-F1 | device accuracy | urgency macro-F1 | duplicate ID F1 | 加權總分 | tickets 總時間 | 程序峰值 RSS |
|---|---:|---:|---:|---:|---:|---|---|
| 原始 baseline | 0.481807 | 0.726667 | 0.517587 | 0.000000 | 0.441574 | 未量測 | 未量測 |
| A：完整規則 | 0.848614 | 0.993333 | 1.000000 | 0.853333 | 0.908779 | 1.8942～1.9486 s | 27.83～28.11 MiB |

雙 null 正確 17/17；300/300 筆有效預測、無缺少或多餘 ID，suggestion 全部存在。duplicate TP=64、FP=1、FN=21。相對 baseline 修正／新增錯題：device 80/0、category 101/26、urgency 66/0、duplicate 64/0；完整 ID 留在 reports/comparison.json。

## 延遲

```bash
python3 Solution/scripts/benchmark.py --runs 3
```

每次啟動新 Python 程序，用 host `perf_counter()` 包住 subprocess，包含解譯器啟動、讀檔、排序、推論、驗證、序列化、寫預測及量測檔，三次退出碼均為 0。OS 檔案快取未清除，不能宣稱是冷磁碟快取測試。

| 執行 | 程序總秒數 | 逐筆 P50 ms | P95 ms | 最大 ms | 峰值 RSS MiB |
|---|---:|---:|---:|---:|---:|
| 1 | 1.902992 | 0.897832 | 1.228126 | 7.136500 | 27.828125 |
| 2 | 1.894241 | 0.899708 | 1.163209 | 6.521335 | 27.843750 |
| 3 | 1.948626 | 0.897460 | 1.312334 | 7.710250 | 28.109375 |

每次量 2,000 筆，單筆範圍包含正規化、設備候選、分類、上下文、urgency、duplicate、建議與組裝；排除共用載入、排序、輸出驗證及序列化，這些成本仍包含在程序總時間。分位數採 nearest-rank，不用總耗時除以筆數冒充延遲分布。各階段與逐筆原始值另存 metrics JSON。

RSS 使用 `resource.getrusage(RUSAGE_SELF).ru_maxrss`，macOS 為 bytes、Linux 轉 KiB→bytes。這是程序 RSS，不等於容器記憶體總量，也不是強制記憶體限制證據。

P95 約為 P50 的 1.3～1.5 倍。可能原因包含文本長度、缺字模式比對次數及排程；目前未以 profiler 分離各因素，不能斷言尾延遲來源。全部三次遠低於 480 秒預算。

## 可追溯性與限制

正式留存原始紀錄：`experiments/runs/20261113T015620Z-5ae2e4d8/`。每次保留 command、git HEAD/工作目錄狀態、Python/OS、輸入 SHA-256、推論程式與規則 SHA-256、預測及逐筆量測。程式尚未 commit，不能只靠 HEAD 重現，須搭配產物雜湊與本次原始碼。較早的初測紀錄仍保留，但本表使用修訂後的最後一組。

已確認輸入輸出數量與 ID 相符、同設定三次輸出一致、原生與 2 GB 輸出一致；單筆延遲總和不大於管線總時間。測試另涵蓋標籤污染、跨檔狀態重設、時間平手、打亂輸入、無效資料與 CLI 失敗不覆寫輸出。

尚未對 B/C、實體 8 GB 或 x86 環境量測。這是方案 A 的基準，不是完整作業最終 bench。

## 方案 B 追加量測（2026-11-13）

B 已完成 LR／LinearSVC 及保留／遮罩四種 history 內部比較；詳細分割與分數見 [TF-IDF 實作與比較](docs/TF-IDF實作與比較.md)。本節正式效能量測使用選定的遮罩版本、以完整 history 重訓的固定資產。**B 未觀察 dev 內容，也未執行 dev 驗證；沿用 A 模組的來源限制仍存在。** 上述 A 章節的「尚未量測 B」為當時狀態。

使用專案 .venv、requirements-ml.txt 固定依賴、OPENBLAS_NUM_THREADS=2、OMP_NUM_THREADS=2、原生 macOS ARM64；未施加 8 GB 上限。每方法三個新 Python 程序、每次 2,000 筆，無顯式預熱、不清 OS 快取。程序總時間包含 sklearn 與模型冷載入、讀取、推論、驗證及寫檔。

| 實測方法 | history 內部 category F1 | dev 分數 | tickets 程序總時間 s | 逐筆 P50 ms | P95 ms | 最大 ms | 程序峰值 RSS MiB |
|---|---:|---|---|---|---|---:|---|
| B-LR 遮罩 | 0.746499 | 未驗證 | 10.183～10.498 | 4.983～5.201 | 7.123～7.176 | 32.577 | 118.09～123.73 |
| B-SVC 遮罩 | 0.752058 | 未驗證 | 10.252～10.488 | 4.966～5.077 | 7.106～7.151 | 46.491 | 121.11～121.77 |

history 分數來自 200 筆 fit／90 筆驗證、10 筆近似模板排除的模型；效能則量完整 history 重訓後的資產，兩者不可混稱同一份權重。單筆定義與 nearest-rank 方法同 A，各筆包含一次 TF-IDF transform、資訊判斷及適用的分類頭；載入等共用成本仍計入程序總時間。P95 與最大值可能受文字長度、逐筆稀疏特徵轉換及排程影響，未以 profiler 確認各因素比例。

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python Solution/scripts/benchmark.py --method tfidf-lr --runs 3
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python Solution/scripts/benchmark.py --method tfidf-svc --runs 3
```

原始紀錄分別為 `experiments/runs/20261113T032128Z-f32a357b/`、`experiments/runs/20261113T032237Z-1c68ecde/`。模型、訓練來源與版本在 metrics 的 model metadata；每次保存輸入、程式與資產雜湊、逐筆耗時及完整預測。已確認同方法三次結果一致，並與 2 GB 容器三次結果逐位元一致，總時間均小於 480 秒。BGE、實體 8 GB／x86 尚未量測。
