# SILAB FA：工單分流程式

這個 repo 分兩部分：

- **第一部分：批次分流。** `predict.py` 讀工單，輸出設備、故障類別、急迫性、重複報修與處置建議。預設模型是 **C-FT**，dev 總分 0.944，處理 2,000 筆約 24 秒。2 GB 版本沿用同一個模型，在 2 GiB 容器內峰值 518 MiB，分數不變。
- **第二部分：離線問答 Agent。** `ask.py` 回答自然語言問題，`patrol.py` 挑出最急的 10 張工單。使用本機 Qwen3-1.7B（Q4 量化）與 llama.cpp，只用 CPU。

**方案代號**（全文共用）：

| 代號 | 故障類別怎麼判斷 | dev 總分 |
|---|---|---:|
| baseline | 題目附的關鍵詞比對 | 0.442 |
| A | 人工整理的症狀規則 | 0.909 |
| B-LR／B-SVC | 字元 TF-IDF＋線性分類器，用 history 訓練 | 0.896／0.916 |
| C-LR／C-SVC | 中文語意模型 BGE 把句子轉成向量（模型不訓練）＋線性分類器 | 0.865／0.918 |
| **C-FT（預設）** | 同 C，但 BGE 本身也用 history 一起微調 | **0.944** |

除了 baseline，各方案的設備、急迫性與重複報修都用 A 的同一套規則，差別只在故障類別。

## 快速開始

支援 macOS Apple Silicon、Windows x64、Linux（Python 3.12／3.13）。Windows 請把 `python3` 換成 `python`：Windows 的 `python3` 通常是 Microsoft Store 捷徑，執行後不會有任何輸出。其他問題見 [docs/setup](docs/setup/README.md)。

```bash
git clone https://github.com/loveju3/SILAB_FA.git && cd SILAB_FA
# 把題目附的 devices.json、error_codes.json、tickets.jsonl、history.jsonl、dev.jsonl 放進 data/
python3 install.py --with-agent     # 建 .venv、裝 CPU 套件、下載並校驗模型、推論自測；看到 PASS 即完成
python3 run_demo.py                 # dev 評分、兩份 tickets 預測、ask.py --trace 與 patrol.py 示範
```

- 模型權重不放 Git。`install.py` 會從 [GitHub Release](https://github.com/loveju3/SILAB_FA/releases/tag/models-20261115) 下載微調模型（76 MB），校驗 SHA-256 後放到 `transfer/finetuned/`。`--with-agent` 會另外下載 Qwen3-1.7B 與 llama.cpp；只跑第一部分可以省略。
- **不需要手動啟用虛擬環境**：`predict.py`、`ask.py` 等入口會自動改用 `.venv`，題目規格的指令可以直接照打。
- 推論全程離線、只用 CPU。

### 交付項目對照

| HW_RULES 交付項目 | 在哪裡 |
|---|---|
| `predict.py <輸入> <輸出>` | [predict.py](predict.py)；預設 C-FT，找不到模型時退回方案 A 並在 stderr 警告 |
| `tickets_predictions.jsonl` | [tickets_predictions.jsonl](tickets_predictions.jsonl)（`run_demo.py` 會重新產生） |
| 選型對照表 | [技術選型對照表](技術選型對照表.md)、本文[第 6 段](#6-放棄或暫不採用的做法) |
| `bench.md` | [bench.md](bench.md)；原始量測在 [evidence/](evidence/)，用 `scripts/collect_evidence.py` 產生 |
| 2 GB 版本 | `predict.py <輸入> <輸出> --mem-limit 2g`（同一個 C-FT，加上 2 GiB 上限；Docker 驗證：`bash scripts/docker_2gb_check.sh`） |
| `tickets_predictions_2gb.jsonl` | [tickets_predictions_2gb.jsonl](tickets_predictions_2gb.jsonl) |
| `budget.md` | [budget.md](budget.md) |
| `ask.py`（`--trace`）、`patrol.py` | [ask.py](ask.py)、[patrol.py](patrol.py)；設計見[第 4 段](#4-agent-與評測設計)與 [agent/README.md](agent/README.md) |
| `evals/` | [evals/run.py](evals/run.py)、[驗證報告](agent/VALIDATION.md) |
| `ai-log/` | [ai-log/](ai-log/) |

**資料使用規則：** dev 不拿來設計算法或調參，只在版本固定後評分；開發一律用 history。完整規範見 [AGENTS.md](AGENTS.md)。方案 A 是在這條規則訂立前做的，當時參考過 dev，所以 A 以及沿用 A 規則的方案，dev 分數都不算獨立測試成績。

## 1. 我怎麼發現問題的

我把 baseline 程式逐行對照題目規格（[schema.md](schema.md)），再看官方 `evaluate.py` 拆開的四欄分數，而不是只看總分。baseline 總分 0.442，拆開後是 category 0.482、device 0.727、urgency 0.518、duplicate 0.000，四欄各有不同的問題：

| 欄位 | baseline 怎麼做 | 錯在哪 |
|---|---|---|
| `duplicate_of` | 永遠填 `null` | 整欄放棄，直接少掉總分的 20% |
| `urgency` | 一組危險詞＋一組猶豫詞 | 詞表跟規格對不上：缺積水、漏油、短路等詞，又多了規格沒有的單字（「燒」「爆」）；也沒做「關鍵設備＋停機」與「重複報修用語」這兩條規則 |
| `device_id` | 文字出現設備名稱或位置，就選第一台符合的 | 不認別名、型號、錯誤碼；同一位置有多台時會選錯；「那台」這類指涉無法處理 |
| `category` | 依固定順序比對關鍵詞，第一個命中就決定 | 錯字、簡體打不到；否定句或一次描述多個症狀時，會被清單排在前面的類別搶走 |

因此依問題性質分兩步改：

1. **規格寫清楚的部分，直接寫成規則。** urgency、duplicate、設備對照與文字正規化（錯字、簡體、全半形）照規格實作，這就是方案 A。
2. **規格列舉不完的部分，改用學習。** 故障類別的說法太多，一直加關鍵詞只會越補越多，所以改從 history（300 筆有答案的舊工單）學，依序試了 B、C、C-FT。

## 2. 我的分工策略

判斷依據：**這件事規格有沒有寫清楚？需不需要理解各種說法？**

| 工作 | 交給誰 | 原因 |
|---|---|---|
| 輸入驗證、排序、急迫性、重複報修 | Python 規則 | 規格明確，應該可重現、可測試，不需要模型猜 |
| 設備辨識、錯誤碼 | 公開對照表＋規則 | 名稱、型號、代碼就是直接證據；判斷不出就填未知，不自己編 |
| 第一部分的故障分類 | 統計模型（預設 C-FT；A 作備援） | 說法太多，規則列不完，從 history 學比較穩 |
| 第二部分理解問題、決定下一步 | 本機 LLM | 使用者不會照固定語法提問，需要模型選工具、填參數、決定補查或停止 |
| 篩選、計數、排序、引用原文 | Python 工具 | 對完整資料集合計算，避免模型心算或只看前幾筆 |

第一部分沒有用 LLM：批次 2,000 筆的時間與記憶體預算緊，規則加分類模型已經夠用。推論只讀工單的四個輸入欄位，不讀答案。第二部分只以 tickets 原文和公開對照表作為事實依據，不把第一部分的預測當作答案。

## 3. AI 協作故事

**分工：** 需求、資料使用規則（dev 只做驗證），以及最後採用哪個方案，由我決定。Codex 與 Claude 負責查規格、寫程式與測試、整理文件。AI 說「可行」不算數，一律看執行結果。

**AI 給過不可靠答案、我怎麼發現的：**

| 狀況 | 怎麼發現 | 怎麼修正 |
|---|---|---|
| 架構說明只講工具，沒講 LLM 在做什麼 | 我追問：「這個過程不需要有 llm 模型才能做到嗎？」 | 釐清為「LLM 負責決策，Python 負責執行」 |
| 想直接把預設改成微調模型 | 對照 HW_RULES，發現它當時還沒有 dev 分數和計時 | 先補量測再決定 |
| 安裝腳本只在 Linux 測過 | 我在 Mac 實跑，python.org 版 Python 缺 SSL 根憑證，下載失敗 | 改用系統 curl 下載 |
| 2 GB 版本被換成分數較低的方案 A | 我追問：「峰值沒超過 2 GB，為什麼要換？」 | 補做 Docker 2 GiB 實測，改回 C-FT，不白白掉分 |
| 小模型把位置填進設備欄、加入沒要求的篩選、漏掉條件 | 合成題、history 評測與 `--trace` | 把設備查詢拆成用途單一的工具，並回傳 `applied_filters` 讓模型核對 |
| 巡檢時模型把原文改寫成「看似合理但不是原文」的引句 | 檢查引用是否能對回原文；要求模型重寫一次仍失敗 | 改由程式從原文擷取引文，模型只負責挑選與排序 |

[ai-log/](ai-log/) 目前放的是協作截圖，不是完整對話匯出。工單原文與逐筆 trace 不放進 Git。

## 4. Agent 與評測設計

**工具架構：** 模型每一步輸出一個結構化動作（呼叫哪個工具、帶什麼參數）。Python 檢查參數後執行，再把結果交回模型；模型決定補查，或用 `finish(result_id)` 指定採用哪一次的結果。工具有：查單一設備、查某位置的設備、查錯誤碼、查工單、讀工單原文。**最終的數字、清單與引用都直接取自工具結果，不讓模型自己寫。**

**停止條件：** 模型主動 finish；最多 4 次決策；整體 29 秒時限（含模型載入與關閉）。重複查詢會收到錯誤提示。超時、超過步數或引用無效結果時，程式明確失敗，不輸出假答案。`--trace` 把每一步寫到 stderr，stdout 只輸出 JSON。

**巡檢（`patrol.py`）：**

1. 候選：原文提到關鍵設備（`critical: true`）的工單。
2. 含安全詞（漏電、冒煙、短路等，取自規格）的工單，由程式直接列為 high、排在最前面。
3. 剩下的名額，由模型從沒有安全詞的候選中挑選並排序。
4. 引文由程式從原文擷取。

改成這樣，是因為舊版全部交給模型挑選時，5 張漏電工單全部落選。目前仍可能漏掉用其他說法描述的急件。

**評測：**

- **目前題庫（v3，41 題）：** 開發組 36 題、保留組 5 題（版本凍結後才跑一次）。比起 v2，新增口語問法、相對時間（「今天」）、多步查詢，以及含簡體與錯字的題目。
- **最新結果：** 開發組 21/36，每題 8.5～15.3 秒。寫明規格的查表題 11/11 全對；口語、相對時間、多步題大多答錯。例如「今天有幾張」回了全部工單，模型沒把「今天」換算成快照日期。前一版 v2 是合成題 20/30、history 題 12/12。
- **巡檢：** 完整 2,000 筆實跑 31.4 秒（上限 120 秒），格式與引用檢查通過；排序是否正確仍需人工審查。
- 每題重新啟動程序並載入模型，檢查答案值、型別、引用格式與耗時。題目都是我自己出的，不能當作隱藏題的成績；也沒有依分數回頭調整 Agent。

**心得：** 1.7B 小模型在問題寫得明確時，選工具大致可靠；但遇到口語、相對時間或需要串兩步的問題，就容易填錯參數或提早停下。所以凡是能驗證的事，包括計數、引用與安全詞判斷，都交給程式做；下一步要補的是把「今天」這類說法先換成明確條件。

詳細設計見 [agent/README.md](agent/README.md)，逐次結果見 [agent/VALIDATION.md](agent/VALIDATION.md)。

## 5. 已知限制與下一步

**限制：**

- **第一部分：**
  - 7 個方案在同一台 Linux VM 上量測。C-FT 另在 Mac 的 Docker 2 GiB 容器，以及 Windows 筆電（i7-13620H，約 33～36 秒）跑過。**沒有在限制 8 GB 記憶體的環境驗證。**
  - 預設選 C-FT 時，用 dev 比較了一次，這個選擇本身用到了 dev。
  - B／C 的原始模型檔已不在本機，表中分數來自同配方重訓的版本。
- **第二部分：**
  - 查詢以字面比對為主，還不會處理錯字、代稱、跨工單指涉，也不支援自由摘要。
  - 巡檢只在候選子集裡挑選，不能宣稱已找出全部工單中最急的 10 張。
  - Agent 已在 Windows 與 macOS（問答 9.1 秒、巡檢 38.2 秒）實跑。Linux 需要 glibc 2.38 以上，WSL 未驗證。

**最想吐槽的地方：** 一張工單常常同時描述供電、溫度、畫面等多個異常，但 category 只能選一類；設備判錯又會連帶讓 urgency 和 duplicate 一起錯。分數高不等於現場處置正確，所以我同時看各欄錯誤，而不只追總分。另外，反覆用同一份 300 筆 dev 選方案，很容易過度擬合，所以我把 dev 限定為版本固定後才評分。

**下一步：**

1. 在限制 8 GB 記憶體的實體筆電上重做量測。
2. `suggestion` 依錯誤碼或設備型號，給出更具體的處置。
3. 問答先補相對時間換算與多步查詢，再在 history 上改善姓名與設備確認、語意查詢；Agent 版本凍結後跑一次保留組。
4. 巡檢補上「全資料是否漏掉候選」的檢查，以及人工排序審查。

## 6. 放棄或暫不採用的做法

| 做法 | 為什麼沒採用 | 驗證程度 |
|---|---|---|
| 第一部分每筆都交給 LLM | 2,000 筆的批次預算緊；規則加分類模型已經夠用。LLM 留給需要理解問句的第二部分 | 設計取捨，沒有實測對照，不能宣稱 LLM 一定較差 |
| 只靠一直加症狀關鍵詞（方案 A） | 每種新說法都要補一條規則。A 保留作備援與對照，另外加入 B／C／C-FT | 已有規則、TF-IDF、BGE 的實測比較 |
| 選 history 內部分數最高的 C-SVC | 它在 history 內部最好，但在 dev 上輸給 C-FT（0.918 對 0.944）。history 只有 300 筆，切出的驗證段太小，不夠可靠 | 預設是用 dev 比較一次後選定；沒有用 dev 調參，但這個選擇本身用到了 dev |
| 只看五類 F1 選模型 | 「資訊不足、無法判斷」也要能正確回答，所以事先定好用含 unknown 的六類指標，不事後換指標 | 資訊不足的樣本很少，結論不穩 |
| 第一版問答就加向量檢索 | 先做好可驗證的查表、文字查詢與計數工具；分類效果好，不代表適合拿來檢索 | 只評估過，沒做對照實驗 |
| 讓 LLM 自己寫巡檢引文 | 模型會改寫原文，修正一次仍失敗 | 改由程式擷取後，通過合成 smoke 測試 |

完整候選與模型大小見 [技術選型對照表](技術選型對照表.md)，時間與記憶體統一見 [bench.md](bench.md)。

## 其他文件

- [Solution 操作與架構](Solution/README.md)、[演算法流程圖](Solution/docs/演算法流程.md)
- [TF-IDF 實作與比較](Solution/docs/TF-IDF實作與比較.md)、[BGE 方案 C 實驗](Solution/docs/BGE方案C方法與實驗.md)、[BGE 微調與 UI 訓練](Solution/docs/BGE聯合微調與UI訓練.md)
- [建立環境](docs/setup/README.md)、[進階／歷史文件](docs/advanced/README.md)、[輔助工具](helper_tool/README.md)

題目 [HW_RULES.md](HW_RULES.md) 與 [schema.md](schema.md) 放在 repo 方便核對；資料使用仍以 [AGENTS.md](AGENTS.md) 為準。工單原始資料、逐筆錯誤報告與含工單內容的分析檔只留本機（`data/` 已設為忽略）；Git 只收題目要求的兩份預測檔。舊提交中仍可能留有資料。
