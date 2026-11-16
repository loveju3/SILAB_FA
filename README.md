# SILAB FA：工單分流程式

第一部分預設使用 **BGE encoder＋分類頭聯合微調**（dev 總分 0.944，見 [bench.md](bench.md)），2 GB 模式使用方案 A 規則版（0.909，見 [budget.md](budget.md)）；另保留方案 B 字元 TF-IDF、方案 C 固定 BGE encoder（B/C 均提供 LR 與 LinearSVC）、評分／查詢 UI、測試、流程圖及彙總效能說明。

第二部分新增獨立的 [離線智慧問答 Agent 初版](agent/README.md)：`ask.py` 模型選擇工具與停止時機、`patrol.py` 候選巡檢及 `evals/` 開發評測。使用本機 Qwen3-1.7B Q4 與 CPU llama.cpp；支援範圍、限制及實測須與第一部分分類成績分開看待。

工作台另提供訓練 modal：上傳 history、指定模型保存路徑、執行 BGE encoder＋分類頭聯合微調及凍結對照，並保存逐輪紀錄與時間戳模型。操作與限制見 [UI 訓練文件](Solution/docs/BGE聯合微調與UI訓練.md)。

## 快速開始

支援 macOS Apple Silicon、Windows x64、Linux（Python 3.12／3.13；Windows 把 `python3` 換成 `python`）。詳細說明與常見問題見 [docs/setup](docs/setup/README.md)。

```bash
git clone https://github.com/loveju3/SILAB_FA.git && cd SILAB_FA
# 把題目附的 devices.json、error_codes.json、tickets.jsonl、history.jsonl、dev.jsonl 放進 data/
python3 install.py --with-agent     # 建 .venv、裝 CPU 套件、下載並校驗模型、推論自測；看到 PASS 即完成
python3 run_demo.py                 # dev 評分、兩份 tickets 預測、ask.py --trace 與 patrol.py 示範
```

- 模型權重不放 Git：`install.py` 會從本 repo 的 [GitHub Release](https://github.com/loveju3/SILAB_FA/releases/tag/models-20261115) 下載微調模型 ZIP（76 MB），校驗 SHA-256 後放到 `transfer/finetuned/`；Agent 的 Qwen3-1.7B 與 llama.cpp 由 `--with-agent` 下載。只做第一部分可省略 `--with-agent`。
- **不需要啟用虛擬環境**：`python3 predict.py …`、`python3 ask.py …` 等入口會自動改用 `.venv` 執行，題目規格的指令可以直接照打。
- 推論全程離線、只用 CPU。

### 交付項目對照

| HW_RULES 交付項目 | 在哪裡 |
|---|---|
| `predict.py`（`python3 predict.py <輸入> <輸出>`） | [predict.py](predict.py)，預設使用 BGE 聯合微調；模型不存在時退回方案 A 並在 stderr 警告 |
| `tickets_predictions.jsonl` | [tickets_predictions.jsonl](tickets_predictions.jsonl)（`run_demo.py` 會重新產生） |
| 選型對照表 | [技術選型對照表](技術選型對照表.md)，以及本文[第 6 段](#6-放棄或暫不採用的做法以及原因) |
| `bench.md`（dev 四欄、總時間、P50／P95／最大、方案對照） | [bench.md](bench.md)；可重現的原始量測（含逐筆秒數、2 GiB 限制驗證）在 [evidence/](evidence/2026-11-16-linux-vm/README.md)，用 `scripts/collect_evidence.py` 產生 |
| 2 GB 版本 | `python3 predict.py <輸入> <輸出> --mem-limit 2g`（方案 A） |
| `tickets_predictions_2gb.jsonl` | [tickets_predictions_2gb.jsonl](tickets_predictions_2gb.jsonl) |
| `budget.md` | [budget.md](budget.md)（完整證據：[Solution/budget.md](Solution/budget.md)） |
| `ask.py`（支援 `--trace`）、`patrol.py` | [ask.py](ask.py)、[patrol.py](patrol.py)，設計見[第 4 段](#4-agent-與評測設計)與 [agent/README.md](agent/README.md) |
| `evals/`（≥10 題、一鍵跑分、報告） | [evals/run.py](evals/run.py)、[驗證報告](agent/VALIDATION.md) |
| `ai-log/` | [ai-log/](ai-log/2026-11-14-agent.md) |

**重要開發規則：不得觀察 `dev.jsonl` 的資料來設計算法；dev 僅供固定版本驗證。** 開發與調參使用 history，完整規範見 [AGENTS.md](AGENTS.md)。既有方案 A 曾參考 dev，其成績只作歷史開發驗證紀錄。

## 1. 我怎麼發現問題的

這份專案先從 baseline 的程式行為與既有彙總評分找問題，再把「設備辨識、故障分類、急迫性、重複報修」分開檢查。baseline 的限制不只在關鍵詞數量：它沒有完整處理文字正規化、同檔上下文與急迫性規則，`duplicate_of` 也固定回傳 `null`。如果只看加權總分，很容易忽略某個欄位其實完全沒有能力，因此需要同時看四欄指標、格式檢查與執行成本。

方案 A 補齊可明確定義的規則與時序處理，但症狀分類仍受措辭、否定與多症狀描述影響。後續 B／C 因而將分類改為從 history 學習，分別比較字元 TF-IDF 與固定 BGE encoder，而不是把所有改善都變成新增症狀關鍵詞。history 內部分割考慮時間順序與近似模板洩漏；例如 B 的既有比較使用 200 筆訓練、90 筆驗證，另排除 10 筆跨界近似模板，不能把這些分數當作未知資料的成績。

**資料來源限制必須一起保留：方案 A 在現行規範建立前曾觀察 dev，既有分數只代表歷史開發驗證；B／C 沿用的設備、正規化與上下文模組也保留這項限制。** 後續開發與錯誤分析使用 history，dev 只供版本與設定固定後驗證，不讀逐筆錯題來回頭調整算法。完整量測與來源說明見 [bench.md](bench.md) 及 [TF-IDF 實作與比較](Solution/docs/TF-IDF實作與比較.md)。

## 2. 我的分工策略

分工依據是「這個決策是否已有明確規格，以及是否需要理解不同的自然語言表達」，而不是每個步驟都使用同一種模型。

| 工作 | 交給誰 | 原因 |
|---|---|---|
| 輸入驗證、排序、急迫性與重複報修索引 | Python 與確定性規則 | 型別、時間順序及判準明確，應可重現、可測試，不需要生成式模型猜測。 |
| 設備辨識與公共錯誤碼 | 公共對照表與規則 | 已知名稱、型號與代碼可提供直接證據；無法判斷時保留未知，不自行編造對照關係。 |
| 第一部分的故障分類 | A 規則作對照；B／C 使用統計分類模型 | 用 history 學習措辭差異，並比較模型複雜度是否帶來可驗證的收益。第一部分目前不呼叫生成式 LLM。 |
| 第二部分的問題理解與下一步決策 | 本機 LLM | 使用者不會照固定查詢語法提問，模型負責選工具、產生參數、看結果並決定補查或停止。 |
| 工單篩選、計數、排序、答案值與引用輸出 | Python 工具 | 對完整符合條件的集合計算，避免模型自行心算，或把檢索前幾筆誤當成全部。 |

第一部分的核心推論只接收工單的四個輸入欄位，不取得當筆答案；第二部分預設以 tickets 原文與公共對照表作為事實依據，不把第一部分預測視為標準答案。正式推論使用本機 CPU，不依賴外部雲端 API。各方法的時間與記憶體統一收在 [效能量測](bench.md)，2 GB 限制與取捨另見 [budget.md](Solution/budget.md)。

## 3. AI 協作故事

協作中，使用者負責提出產品需求、追問設計依據及確認交付方向，Codex 協助查閱規格、設計工具介面、撰寫程式與測試。程式碼、執行結果與模型回覆都需要證據核對，不能因為 AI 說「可行」就當成已完成。

本次對話有一個直接影響設計說明的追問：

> 「這個過程不需要有 llm 模型才能做到嗎」

當時的架構說明側重工具，沒有把模型的角色講清楚。補正後的分工是：

> 「LLM 負責決策，Python 負責實際執行。」

真正接上模型後，問題也不只出在程式是否能跑。小模型曾把位置填進設備識別欄、加入未要求的篩選旗標，或漏掉問題中的文字條件。這些錯誤透過合成題、history 評測與工具 trace 發現，後續把設備查詢拆成用途較單一的工具，補上參數語意與 `applied_filters` 回查資訊。仍未解決的完整姓名抄錄問題也保留在報告，沒有為了通過單一題目加入姓名特例。

另一個教訓來自巡檢：模型會把多段症狀改寫成看似合理、卻不是逐字原文的引句；要求模型再修正一次仍不足。因此最後由模型選單與排序，程式從選中的工單擷取原文證據。這讓事實引用的責任落在可驗證的資料操作上。

上述需求節錄與開發摘要可搭配 [ai-log](ai-log/2026-11-14-agent.md) 閱讀；它們不等於完整原始對話匯出，去識別化的原始協作紀錄仍待整理。實際工單、答案及逐筆 trace 不加入 Git。

## 4. Agent 與評測設計

`ask.py` 使用本機 Qwen3-1.7B Q4 與 CPU llama.cpp。模型每次輸出結構化動作，程式檢查工具名稱與參數後執行，再把 observation 交回模型。工具涵蓋單一設備屬性、位置設備清單、錯誤碼、工單查詢與原文讀取；模型可以繼續補查，也可以用 `finish(result_id)` 選擇已成功取得的答案。最終數值、清單與 evidence 由程式保存的工具結果輸出。

停止條件同時包含模型主動完成、預設最多四次決策，以及含載入與關閉模型的整體 deadline。重複查詢會收到錯誤回饋；超時、超過步數或引用無效結果時明確失敗，不輸出假成功答案。`--trace` 將工具決策、觀察結果與量測寫到 stderr，stdout 保持 JSON。巡檢則先以安全詞與時間產生候選，再交由模型選出優先工單；這項候選策略仍可能漏掉其他措辭的急件。

評測包含 12 題人工合成問答，以及由 history 四欄原文獨立計算預期答案的 12 題開發問答，涵蓋公共表查詢、日期邊界、計數、排名與空結果。兩套各含六題相同的公共表問題，不能視為 24 題互相獨立的測試。每題重新啟動程序並載入模型，同時檢查答案值、型別、引用格式與耗時。

已保存的 Agent 初版開發報告為合成題 11/12、history 12/12；這不是隱藏題成績，也不是本段文字編輯時重新執行的結果。巡檢 smoke 只檢查格式、真實引用與唯一性，不能代替優先排序的人審。操作與設計見 [Agent 文件](agent/README.md)，結果與錯誤見 [驗證紀錄](agent/VALIDATION.md)，評測入口為 [evals/run.py](evals/run.py)。

## 5. 已知限制與下一步

第一部分的兩參數入口預設使用 BGE 聯合微調，`--mem-limit 2g` 使用方案 A，兩份正式預測輸出已放在 repo 根目錄；B／C 其餘方法可透過 `--method` 選擇。預設方法是以 dev 對兩個已固定版本各評分一次後選定，這個選擇本身用到了 dev，已在 [bench.md](bench.md) 揭露。微調版目前只在 Linux VM 完成三次量測，2 GiB 容器只做過單筆 smoke。Windows 原生規則版已補齊記憶體量測相容性並通過開發測試；各模型的環境與歷史效能仍須分開看待。既有 B／C dev 四欄成績與 2 GB 掉分資料尚未完成，不能以 history 回測或 tickets 輸出一致性取代。8 GB 實機驗證未完成的部分已揭露，2 GB 演練則另保存限制與執行證據。

第二部分目前以字面條件查詢為主，仍可能漏抄姓名或條件，尚不具備完整的錯字、代稱、跨工單指涉與一般自由摘要能力。巡檢只讓模型評估候選子集，不能宣稱已找到全部工單中真正最急的十張。Windows 原生環境已有模型準備流程，macOS／WSL 的 Agent 安裝與預設命令仍需進一步驗證。

從工程角度，這份題目最值得討論的是：**單一分類標籤與現場處置價值並不完全一致。** 一張工單可能同時描述供電、溫度與畫面異常，但輸出只能選一類；設備判斷也會連帶影響 urgency 與 duplicate。因此仍依官方規格評分，同時分開看各欄錯誤、證據與處置建議，避免只追求總分。另一個評測風險是反覆使用同一個小型驗證集；這也是目前將 dev 限定為固定版本驗證、保留歷史來源限制的原因。

下一步會先固定交付方案、執行環境與可重現入口，整理正式輸出及量測文件；算法改善則優先在 history 上驗證姓名／設備候選確認與語意查詢，再擴充較具變化的問答題型。巡檢需增加完整資料範圍的候選漏失檢查與人工排序審查。版本與設定固定後，才進行符合資料規範的 dev 彙總驗證。

## 6. 放棄或暫不採用的做法，以及原因

| 做法 | 目前決定與依據 | 驗證程度 |
|---|---|---|
| 第一部分每筆都交給生成式 LLM | 先以規則與分類模型處理批次分流，把 LLM 用在需要自然語言決策的問答；批次任務的資源預算較緊。 | 屬設計取捨，未完成逐筆生成式 LLM 的完整效能對照，不能宣稱已實測較差。 |
| 只靠持續增加症狀關鍵詞改善分類 | A 仍保留作預設與對照，但後續加入 B／C，避免把每個新措辭都變成一條規則。 | 已有規則、TF-IDF 與 BGE 的比較；各版本的資料來源限制仍適用。 |
| 以「有微調就一定較好」作為選型依據 | 初輪聯合微調在 history 內部未優於固定 encoder＋SVC，當時沒有據此替換預設模型。後來固定版本在 dev 上與方案 A 比較一次（0.944 對 0.909）後才改為預設；固定 BGE＋SVC 沒有 dev 分數，未納入比較。 | history 內部比較的驗證段同時參與選 epoch；dev 比較只做一次且未據此調參，但仍是用 dev 做的選擇。 |
| 只看五類分類 F1 挑模型 | 選型另考慮 unknown，使用預先定義的六類指標；不因某個版本單一分數較高就事後改規則。 | 已有 history 分割與候選比較，但資訊不足樣本很少，結論仍不穩定。 |
| 第一版問答就加入額外向量檢索模型 | 先完成可驗證的表格、文字查詢與聚合工具；BGE 分類效果不能直接證明它適合工單檢索。 | 僅評估，尚未完成檢索對照實驗。 |
| 讓 LLM 自由撰寫巡檢引用 | 改由程式擷取原文，保留模型在選單、排序與優先程度上的決策。 | 真實小模型曾改寫引句，修正一次仍失敗；改為程式擷取後通過合成契約 smoke。 |

完整的第一部分候選、模型／產物大小與 dev 驗證狀態見 [技術選型對照表](技術選型對照表.md)。延遲與記憶體數字統一放在 [bench.md](bench.md)，不把未實測方案包裝成已量測結果。

## 操作與技術文件

- [Solution 操作與架構](Solution/README.md)
- [演算法流程圖](Solution/docs/演算法流程.md)
- [TF-IDF 實作、history 比較與操作](Solution/docs/TF-IDF實作與比較.md)
- [BGE 方案 C、批次流程與實驗](Solution/docs/BGE方案C方法與實驗.md)
- [建立環境與執行](docs/setup/README.md)、[進階／歷史文件](docs/advanced/README.md)
- [輔助工具](helper_tool/README.md)
- [CPU 量測](bench.md)
- [2 GB 驗證](budget.md)（完整證據：[Solution/budget.md](Solution/budget.md)）

正式題目 [HW_RULES.md](HW_RULES.md) 與 [欄位規格](schema.md) 納入 repo，供核對交付需求；其中的規格示例不代表允許以 dev 開發算法，資料使用仍以 [AGENTS.md](AGENTS.md) 為準。原始工單 JSONL、逐筆錯誤報告及含真實工單摘錄的分析產物維持只留本機；預測結果只有題目要求的 `tickets_predictions.jsonl` 與 `tickets_predictions_2gb.jsonl` 納入 Git。測試程式中的合成範例用於驗證契約，真實資料評測仍需自行在本機提供資料。

題目資料請放在本機 data/（見[快速開始](#快速開始)）。data/、其他 JSONL 與報告已設為忽略；不應將任何工單資料加入 Git。

本次合併保留 main 既有歷史；舊提交中的資料仍可能存在。
