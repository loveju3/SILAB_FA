# 離線智慧問答 Agent（第一版）

使用 Qwen3-1.7B Q4_K_M、llama.cpp CPU runtime，以及模型自行選擇工具／停止時機的迴圈。模型每次產生結構化動作，Python 負責驗證參數、執行工具；模型讀取 observation 後選擇補查或 `finish(result_id)`。最終答案從被選中的成功 observation 取得，不由模型另外編寫數量或工單編號。

## 準備與執行

環境建立統一用 `python3 install.py --with-agent`（見 [docs/setup](../docs/setup/README.md)），同一個 `.venv` 準備兩階段所需套件、模型與 CPU runtime。以下為完成後的操作範例；根目錄入口會自動改用 `.venv`，直接用 `python3` 執行即可：

```bash
python3 ask.py "B1 動力室有哪些設備？" --trace
python3 ask.py "錯誤碼 E-01 屬於哪一類？"
python3 ask.py "哪位回報人的工單最多？"
python3 patrol.py --trace
```

`setup.py` 固定量化儲存庫 revision、模型 SHA256 與 llama.cpp b12366。Q4 檔由 bartowski 發布、原始模型為 Qwen；不是 Qwen 官方量化檔。設定及實際 runtime 壓縮檔 SHA256 保存於本機 `agent_assets/manifest.json`。自動 setup 支援 Windows x64、Mac Apple Silicon 與 Linux x64／arm64 的官方 CPU runtime（Linux 版需要 glibc 2.38 以上）；已在 Windows 原生與 macOS（Apple Silicon，`install.py --with-agent` 後問答 9.1 秒、巡檢 38.2 秒）實跑，Linux 與 WSL 尚未完成 Agent 實機端到端驗證。也可自行安裝同版本 llama.cpp，使用 `--server /absolute/path/llama-server --model /absolute/path/model.gguf`。CLI 預設 server 路徑隨平台選擇 `.exe` 或原生執行檔。

推論不下載資源，也不連外：每次 CLI 啟動自己的 CPU `llama-server`，只綁定 `127.0.0.1`，HTTP client 禁用 proxy，模型在同題內重用，結束時關閉。Windows Job Object 確保 CLI 被終止時一併結束模型子程序。這使用本機 loopback 通訊；若驗收完全禁止本機 socket，需改為進程內 llama.cpp binding。

`ask.py` 預設總 deadline 29 秒，含載入及關閉，最多 4 次模型決策，每次最多 180 個生成 tokens、4096 context、6 CPU threads、0 GPU layers、關閉思考。上限不是達標證明，須用新程序評測實際速度。`patrol.py` deadline 為 118 秒。成功 stdout 為 JSON、exit 0；錯誤只寫 stderr、exit 2，不輸出假成功答案。`--trace` 記錄工具選擇、observation、時間、token usage，以及 Windows 各程序峰值工作集；不輸出模型內部思考。完整 trace 含原始工單內容，只留本機。

## 工具與資料邊界

| 工具 | 功能 |
|---|---|
| `get_device` | 依 ID／名稱／型號查單一設備屬性，回單值 |
| `devices_at_location` | 依位置查設備編號清單，不混用單值或 critical 篩選旗標 |
| `list_devices` | 查全部／關鍵／非關鍵設備的編號清單 |
| `get_error_code` | 正規化大小寫、全半形、分隔符，查錯誤碼類別／意思／型號；未知回 null |
| `query_tickets` | UTC 區間、回報人、公開設備字面及任一文字條件；完整集合計數、最多回報人、回報人清單、工單 ID、最近原文 |
| `get_ticket_details` | 查最多 20 筆已知工單的原始四欄 |

日期 `since` 含邊界、`until` 不含邊界，必須帶時區；今天固定為需求快照 `2026-10-14T15:10:00Z`。排名同分回所有同分回報人的排序清單。計數先掃完整符合集合，再截取最多 20 個 evidence；讀原文最多 20 筆，明列 `total/truncated`。

送入模型的 observation 另有 context 預算：長答案清單最多展示前 20 值，records 最多展示 5 筆、每筆原文最多 160 字，明列預覽／摘錄旗標。完整 answer 保留在 Python，模型選擇 finish 後仍輸出完整清單；不把預覽長度當成完整計數。

預設只從 `data/tickets.jsonl` 取得工單證據，公開知識讀 `devices.json`、`error_codes.json`。入口僅投影 ticket_id／created_at／reporter／raw_text，丟棄其他標籤。不讀第一部分預測，不載入 A/B/C 規則、別名、模型。開發可以 `--tickets data/history.jsonl`；明確拒絕檔名含 dev 的輸入路徑，這是防呆而非完整的資料來源稽核，仍須遵守 AGENTS.md、不使用 dev 副本。

來源限制：本對話先前檢閱的正式 `schema.md` 內含 dev 邊界示例；本 Agent 不使用那些示例設計分類規則、提示或評測題，也沒有讀 dev 檔或逐筆結果，但不將整段對話宣稱為完全未接觸任何 dev 相關示例的獨立開發。工具與巡檢詞的依據為公開對照表及正式規格，不是方案 A 資產。

## 目前功能限制

- 查詢是 NFKC、大小寫、空白正規化後的字面匹配。設備僅匹配公共表的 ID／名稱／型號，未處理簡繁、錯字、代稱、跨工單指涉；不等於語意設備判定。
- 沒有原始 category／urgency 標籤可直接查；不能拿第一部分預測冒充事實。模型可選文字篩選條件，但其計數僅代表這些字面條件，不保證涵蓋語意上的所有同類工單。
- 工具 schema 約束格式，不保證模型選對工具或參數。模型最後只能選擇既有 observation 的答案；尚不支援自由文字摘要、跨 observation 組合新答案或針對原文逐筆語意分類。
- 工單內容不當成指令，但 prompt 本身不是完全的 prompt injection 防護。工具僅提供唯讀固定操作，無 shell／任意路徑／任意 SQL。
- 大量工單 ID、很長原文或多輪補查可能超過 context／deadline，此時明確失敗，不暗自截斷計數。此初版不是所有隱藏問答均已支援的宣稱。

## 主動巡檢

`patrol.py` 掃描快照前的原始工單，以 `devices.json` 的 `critical: true` 作候選資格。原文以設備編號、名稱或型號字面比對，涉及任一關鍵設備即納入候選池，沒有安全詞也納入；未辨識出關鍵設備的工單不補入。接著在候選池內依完整原文命中的安全詞種類數降冪、回報時間由新到舊縮選最多 30 張，交給模型根據人身安全、停機與時間挑選最多 10 張並排序、分配 high／medium／low；critical 不等於所有工單均為 high。

為保留 120 秒測試上限與 8192 context，採規則縮選後再呼叫模型，不讓全部關鍵設備工單逐批進 LLM。118 秒整體 deadline 保留，無效選單最多修正一次。候選不足 10 張只輸出實際數量，不用非關鍵設備湊數；要求恰好 10 張的契約測試此時仍判失敗。報告保存候選政策版本、devices.json SHA256，以及候選池數量／模型候選數量／模型候選 ID，供追查；不代表已通過記憶體硬限制驗證。

程式從選中工單原文擷取安全詞附近的連續片段（無安全詞取開頭）作為理由，不讓模型改寫引文，工單不可重複。每候選原文最多 90 字，UI 原文仍顯示完整來源快照。設備錯字、代稱、否定敘述、片段遺失上下文及重複事件尚未完整處理；未辨識設備或在敘述規則縮選時落選的急件可能被排除。未經人審不能宣稱選出了全資料真正最急的十張。設備比對沿用 Agent 公開表字面查詢；安全詞取自正式規格，未沿用方案 A。

## 評測

```powershell
python -m unittest discover -s agent/tests -v
python evals/run.py --suite synthetic
python evals/run.py --suite history
python evals/run_patrol.py
python evals/run_patrol.py --dataset tickets
```

合成 suite 共 30 題，保留原始 12 題，另加入 critical／非 critical、指定工單欄位、UTC／UTC+8 單日邊界、多條件交集、同分排名、未知設備／錯誤碼／工單、無關任務與捏造答案指令。使用人工撰寫的合成工單與公開對照表。每題在 `evals/fixtures.json` 保存 `rationale`（設計理由）與 `category`；理由會隨逐題事件與 JSON 報告保存，CLI 及 UI 在預期答案下方顯示。history suite 另產生 12 題，以 history 四欄建立獨立計算的預期答案，不用答案標籤；生成的題目、回報人及逐筆 trace 僅保存在忽略的 `evals/results/`。開發、錯誤分析、模型選型只用這些來源；不讀 dev、不依 dev 分數反覆挑選版本。

每題新啟動 `ask.py` 並載入模型，要求答案值與型別正確、evidence 格式正確、外部計時不超過 30 秒。不清除 OS 檔案快取，因此是新程序啟動測試，不是每次冷磁碟讀取。報告保存 git revision、實際 Agent／評測原始碼雜湊（包含未提交修改）、資料雜湊、模型雜湊、設定、時間與完整 trace。合成單元測試採用 scripted model 只驗證控制流程，與真實 LLM 評測分開計算。這些是開發評測，不能當作未觀察資料的泛化成績。

本次實測數字與已知問題見 [驗證紀錄](VALIDATION.md)。巡檢 smoke 只檢查格式、引用、唯一性與時間，沒有排序正確率的標準答案。

### 擴充題庫的評分契約

- `answer` 嚴格比對值與最外層 JSON 型別；`evidence_required` 指定是否需要工單引用。
- `expected_evidence` 若存在，要求引用集合完全相同（順序不影響）；公開查表、查無資料、拒答題要求空引用。
- 無關任務約定回答 `{"status":"out_of_scope"}`，且 `allowed_tools: []` 要求不執行資料查詢；程序錯誤、逾時或裸 null 不算拒答成功。
- 未知工單約定回答 `{"status":"not_found"}`。指定工單欄位、未知工單與拒答題包含目前工具未支援的能力，仍正常計入分母，不跳過、不當成預期通過。本次題庫擴充沒有更動 Agent 工具或提示詞。
- history 仍為 12 題，透過固定 ID 共用原始 6 題公開查表題，不因合成題庫增加而混入合成工單題。新增題目仍是開發評測，不能解讀成獨立泛化成績。
