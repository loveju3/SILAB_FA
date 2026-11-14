# 第一部分：CPU 效能與評分量測

本文件是第一部分效能量測的統一交付入口，整合原先 `Solution/bench.md` 中分散的方案 A／B／C 紀錄。**這次只整理既有彙總，沒有重跑模型、開啟 dev 或讀取逐筆錯題。** 數字屬於下列日期與環境，不代表目前工作目錄中所有未提交修改已重新量測。

## 環境與範圍

- 既有量測日期：2026-11-13。
- 主機：Apple M4 Max、14 核、36 GiB RAM，macOS ARM64、Python 3.13。
- 資料：每次完整處理 `data/tickets.jsonl` 的 2,000 筆工單。
- 原生 CPU 執行，不使用 GPU／MPS；A 為純 Python，B／C 使用固定本機依賴與模型，B／C 的數值運算執行緒設為 2。
- 每方法三次新 Python 程序，無顯式預熱、不清除 OS 檔案快取；不是冷磁碟快取測試。
- 未施加 8 GB 記憶體上限，也未在實體 8 GB 機器驗證。2 GB 容器結果另見 [budget.md](Solution/budget.md)，不與本表混用。

**資料來源限制：** 方案 A 在現行規範建立前曾觀察 dev，因此其分數僅為歷史開發驗證紀錄，不是未觀察資料的獨立測試成績。B／C 的分類器開發使用 history，但沿用 A 的設備、正規化及上下文模組，仍保留來源限制。後續開發與驗證依 [AGENTS.md](AGENTS.md)。

## 統一效能比較表

範圍表示三次執行的最小～最大值；「逐筆最大值」為三次中最大的單筆值。總時間包含程序啟動、依賴／模型載入、讀檔、推論、驗證及輸出。延遲使用毫秒，記憶體使用 MiB。

| 方案 | dev 加權總分 | 2,000 筆程序總時間（秒） | 單筆 P50（ms） | 單筆 P95（ms） | 逐筆最大值（ms） | 程序峰值 RSS（MiB） | 單筆計時口徑 |
|---|---:|---:|---:|---:|---:|---:|---|
| 原始 baseline | 0.441574 | 未量測 | 未量測 | 未量測 | 未量測 | 未量測 | 官方評分對照 |
| A：完整規則 | 0.908779 | 1.894241～1.948626 | 0.897460～0.899708 | 1.163209～1.312334 | 7.710250 | 27.828125～28.109375 | 單筆獨立處理 |
| B-LR：TF-IDF＋LR，遮罩名稱 | 未驗證 | 10.183～10.498 | 4.983～5.201 | 7.123～7.176 | 32.577 | 118.09～123.73 | 單筆獨立處理 |
| B-SVC：TF-IDF＋LinearSVC，遮罩名稱 | 未驗證 | 10.252～10.488 | 4.966～5.077 | 7.106～7.151 | 46.491 | 121.11～121.77 | 單筆獨立處理 |
| C-LR：固定 BGE＋LR，保留名稱 | 未驗證 | 19.615～21.021 | 48.829～50.382 | 107.579～122.273 | 175.816 | 901.97～984.36 | 完整共享批次服務時間＋單筆其餘處理 |
| C-SVC：固定 BGE＋LinearSVC，保留名稱 | 未驗證 | 19.099～19.524 | 46.309～48.913 | 108.283～115.486 | 132.636 | 829.78～949.70 | 完整共享批次服務時間＋單筆其餘處理 |

已量測方法的三次總時間均低於 480 秒。**C 的 P50／P95／最大值不是 B 的單筆獨立分類延遲，不能直接以這幾欄比較 B／C 快慢，也不能將 C 的逐筆時間相加。** 相同批次的成員會各自記錄該批次的完整共享服務時間；跨方法處理 2,000 筆的總時間可以在揭露設定後比較。

本表僅列已有同口徑完整 tickets 效能紀錄的方案。BGE 聯合微調與其他設計若只有訓練／history 驗證結果，不將它們當作已完成此處的效能量測。採用／暫不採用的依據另見 [選型對照表](技術選型對照表.md) 及 [README 第六段](README.md#6-放棄或暫不採用的做法以及原因)。

## dev 四欄分數

以下轉錄既有文件記載的官方 `baseline/evaluate.py` 彙總，並非本次重新評分。B／C 尚未完成固定版本的 dev 驗證，維持「未驗證」，不以 history 分數補入。

| 方案 | category macro-F1 | device accuracy | urgency macro-F1 | duplicate ID F1 | 加權總分 |
|---|---:|---:|---:|---:|---:|
| 原始 baseline | 0.481807 | 0.726667 | 0.517587 | 0.000000 | 0.441574 |
| A：完整規則（歷史開發驗證） | 0.848614 | 0.993333 | 1.000000 | 0.853333 | 0.908779 |
| B-LR | 未驗證 | 未驗證 | 未驗證 | 未驗證 | 未驗證 |
| B-SVC | 未驗證 | 未驗證 | 未驗證 | 未驗證 | 未驗證 |
| C-LR | 未驗證 | 未驗證 | 未驗證 | 未驗證 | 未驗證 |
| C-SVC | 未驗證 | 未驗證 | 未驗證 | 未驗證 | 未驗證 |

供區分資料用途，既有 history 內部 category F1 分別為 B-LR 0.746499、B-SVC 0.752058、C-LR 0.793935、C-SVC 0.897928。它們來自 200 筆 fit／90 筆驗證、另排除 10 筆跨界近似模板的選型流程；正式 tickets 效能使用完整 history 重訓後的資產，兩者不是同一份權重，也都不能冒充 dev 成績。

詳細分割與選型說明：[TF-IDF 實作與比較](Solution/docs/TF-IDF實作與比較.md)、[BGE 方法與實驗](Solution/docs/BGE方案C方法與實驗.md)。

## 量測指令

環境準備依 [docs/setup/](docs/setup/README.md)。以下從專案根目錄執行；每個方法處理同一份完整 tickets，runner 保存預測與量測。**這些效能命令不需要 dev。**

方案 A：

```bash
python3 Solution/scripts/benchmark.py --method rules --runs 3
```

方案 B：

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python Solution/scripts/benchmark.py --method tfidf-lr --runs 3
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python Solution/scripts/benchmark.py --method tfidf-svc --runs 3
```

方案 C：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python Solution/scripts/benchmark.py --method bge-lr --runs 3
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python Solution/scripts/benchmark.py --method bge-svc --runs 3
```

上方是既有 macOS／Unix shell 的命令形式，不將它們視為 Windows PowerShell 命令。現有 Windows 相容性修改與單次產物生成檢查，也不等於在 Windows 重新完成這張三次量測表。

## 怎麼計時與量記憶體

**程序總時間：** host 端以 `perf_counter()` 包住新 Python subprocess，包含解譯器啟動、sklearn／Torch／Transformers 與模型載入、讀檔、排序、推論、輸出驗證、序列化、寫預測及量測檔；不包含事前安裝與模型下載。不能用單筆核心計時代替完整程序時間。

**A 的逐筆時間：** 正規化、設備候選、分類、上下文、urgency、duplicate、建議及結果組裝。共用載入、排序、輸出驗證與序列化不計入單筆，但仍包含在程序總時間。

**B 的逐筆時間：** 在 A 的共用管線之外，每筆還包含一次 TF-IDF transform、資訊判斷及適用的分類頭執行。分類頭與套件載入仍計入總時間。

**C 的逐筆時間：** 固定 BGE-small-zh-v1.5、CPU threads=2、batch size=16、max length=512、float32、CLS＋L2。每筆計入所在分類批次的完整共享服務時間，加上該筆其餘處理；逐筆量測另有 `batch_classification_seconds` 可辨認共享成本。共同文字準備、排序、驗證及序列化仍計入總時間。正式推論不使用訓練向量快取隱藏 encoder 成本。

**分位數：** 對每次的 2,000 筆紀錄使用 nearest-rank 計算 P50／P95，最大值直接取該次最高值；不以總時間除以筆數冒充延遲分布。

**峰值記憶體：** 原 macOS 量測使用 `resource.getrusage(RUSAGE_SELF).ru_maxrss`，macOS 原值為 bytes，Linux 原值須由 KiB 轉成 bytes，再以 1024² 換算為 MiB。這是單一程序的峰值 RSS，不是模型檔案大小、容器總記憶體或硬限制證據；2 GB 容器的 `memory.peak` 使用另一種帳務口徑，另收在 budget 文件。

## 尾延遲與可信度

A 的 P95 約為 P50 的 1.3～1.5 倍。文本長度、缺字模式匹配與作業系統排程可能造成差異；B 另有稀疏特徵轉換，C 另有分詞、批次補齊及矩陣運算。沒有 profiler 分離各因素，這些是可能解釋，不當作已證明的原因。

原量測文件記錄的檢查包含：每次輸入與輸出數量／ID 相符、同方法三次輸出一致、與各自 2 GB 結果逐位元一致，以及模型／輸入／程式雜湊保存。A／B 的獨立逐筆時間可與管線總時間核對；C 因共享批次計時不能套用逐筆相加檢查。這次文件整理沒有重新執行上述檢查。

原 runner 的追溯資料包含 command、Git HEAD／工作目錄狀態、Python／OS、輸入 SHA256、程式與資產 SHA256、預測及逐筆量測。原文件提到當時含未提交程式，不能只靠 HEAD 重現，還要比對實際原始碼與資產雜湊。

| 方案 | 原文件記錄的目錄（專案根目錄起算） | 這份工作目錄的狀態 |
|---|---|---|
| A | `Solution/experiments/runs/20261113T015620Z-5ae2e4d8/` | 未提供 |
| B-LR | `Solution/experiments/runs/20261113T032128Z-f32a357b/` | 未提供 |
| B-SVC | `Solution/experiments/runs/20261113T032237Z-1c68ecde/` | 未提供 |
| C-LR | `Solution/experiments/runs/20261113T074552Z-19afab94/` | 未提供 |
| C-SVC | `Solution/experiments/runs/20261113T074714Z-a04be153/` | 未提供 |

以上目錄在本次整理時皆不在目前工作目錄內，因此這些數字的依據是既有彙總文件，不能聲稱本次已重新核驗原始量測。原始資料、預測及逐筆量測仍依規範只留本機，不為補證據而加入 Git。

## A 的三次原始彙總

保留原表精度，供核對統一比較表的範圍與最大值：

| 執行 | 程序總時間（秒） | P50（ms） | P95（ms） | 最大值（ms） | 峰值 RSS（MiB） |
|---|---:|---:|---:|---:|---:|
| 1 | 1.902992 | 0.897832 | 1.228126 | 7.136500 | 27.828125 |
| 2 | 1.894241 | 0.899708 | 1.163209 | 6.521335 | 27.843750 |
| 3 | 1.948626 | 0.897460 | 1.312334 | 7.710250 | 28.109375 |

## 尚未完成的驗證

- B／C 的固定版本 dev 四欄成績與總分尚未提供，不能以 history 或 tickets 的結果替代；需要後續固定版本驗證，不在此次文件整理中執行。
- 尚未在實體 8 GB 機器重做這組原生效能測試，已如實揭露。
- 本次未將不同主機、日期、量測口徑或未提交版本的數字合併成一次實驗。
- 第二部分問答／巡檢的時間條件不同，其既有結果見 [Agent 驗證紀錄](agent/VALIDATION.md)，不併入 2,000 筆批次效能表。
