# 開發輔助工具

第一次使用請先執行根目錄的 `python3 install.py`（見 [docs/setup](../docs/setup/README.md)），之後用 `python3 helper_tool/ui_server.py` 啟動，會自動使用 `.venv`。本頁以 UI 功能與操作為主。

**資料使用規則：** dev 僅供固定版本驗證，不得觀察 dev 原文、標籤或逐筆錯題來設計／調整算法。工具保留 dev 評分與報告功能，開發時的錯誤分析請使用 history。完整規範見 [AGENTS.md](../AGENTS.md)。

評分、查詢與方案 A 只需 Python 3，沒有第三方套件。方案 B 的 TF-IDF 與方案 C 的 BGE 方法另需各自依賴與 history 訓練模型。以下指令從專案根目錄執行；工具也能從其他目錄以絕對路徑執行。使用者提供的命令列檔案路徑相對於執行目錄，UI 輸出與預設資料路徑則固定相對於專案根目錄。

## 網頁操作版（建議）

在 macOS Finder 打開 `helper_tool` 資料夾，**雙擊「啟動工具.command」**。它會自動啟動本機服務並開啟瀏覽器，不需要每次輸入指令。使用期間保留啟動視窗；結束時在該視窗按 Control+C。

介面有三個分頁：

- **評分分析**：可以直接選測試資料集、方法、執行環境及輸出資料夾，一鍵執行預測與評分；也保留上傳已有預測檔與「載入 Baseline 範例」。畫面顯示總分、四項指標、錯誤長條圖及分類混淆矩陣。自動預測後可切換「全部判斷／只看錯題」，依錯誤欄位、ID、原文或回報人篩選，逐筆對照正解、預測與建議。可下載預測 JSONL、錯題 CSV／JSONL 及指標 JSON。
- **工單查詢**：輸入一個或多個 ticket_id，按查詢或 Enter。直接呈現原文、回報人、時間、標籤、來源檔與行號；重複工單編號可點擊追查。
- **問答 Agent**：第二階段本機 LLM 評測，可選 history（12 題）／合成問答集（30 題）與巡檢契約測試。背景執行，逐題顯示問題、預期答案、設計理由、實際答案、通過狀態、含模型載入的時間與執行紀錄；同時顯示總耗時、逐題耗時合計及最長單題，支援選取既有工作與下載完整 JSON 報告。

評分檔只在本機讀取，原始資料不會被修改。每次評分報告保留在 `helper_tool/reports/ui/<時間與唯一編號>/`；重啟後檔案仍保留，網頁上的分析狀態則不保留。評分規則及檔案檢查與下方命令列工具相同。選檔每檔上限 8 MB，請求總量上限 16 MB。

預設網址為 `http://127.0.0.1:8765`，埠號被占用時自動嘗試後續 19 個埠，實際網址顯示在啟動視窗。只監聽本機，不使用外部字型、CDN 或雲端服務。若瀏覽器未自動打開，點擊或複製啟動視窗內的網址。

其他系統或偏好自行啟動時可使用 `python3 helper_tool/ui_server.py`。UI 檔案位於 `helper_tool/web/`，後端入口為 `ui_server.py`。

### 一鍵執行預測

1. 選擇 dev、history、tickets，或「自行選擇 JSONL」上傳本機資料。
2. 選方法：方案 A、Baseline、TF-IDF＋Logistic Regression 、TF-IDF＋LinearSVC，或方案 C 的 BGE＋LR／SVC。
3. 選環境：本機（未限制記憶體）或 2 GB Docker 容器。
4. 填入輸出資料夾，預設 `Solution/predictions`。接受絕對路徑；相對路徑固定依專案根目錄解析，資料夾不存在會建立。
5. 按「執行預測與評分」，完成後下方顯示結果、實際儲存位置與下載按鈕。

### 一次跑全部方法並比較分數

想在同一份資料上比較所有方法時，選好「測試資料集」「執行環境」「輸出資料夾」後，按「一次跑全部方法」（上方「方法」與「模型路徑」會被忽略）。

- 依序執行 Baseline、方案 A、TF-IDF＋LR／SVC、BGE＋LR／SVC、BGE 聯合微調。不平行執行，耗時才可互相比較。
- 表單下方「各方法使用的模型」可為每個需要模型的方法選擇：
  - **最新一次訓練**（預設）：「模型訓練」頁最近一次成功訓練中，該方法自動選中的候選（★ 選型結果）。沒有訓練紀錄時改用預設位置。
  - **某一筆訓練紀錄的某個候選**：例如「11/16 15:39 · 凍結 BGE 訓練 · bge-svc-masked · 驗證總分 0.9143」。自動選型依六類 macro-F1，不一定是驗證總分最高的那個，所以兩個候選都列出。
  - **預設位置**：`Solution/models/<方法>.joblib`（命令列訓練的輸出）；聯合微調為 `transfer/finetuned`。
  - 清單只列出模型檔仍存在的紀錄；選了之後模型被刪掉，開始前就會報錯，不會跑到一半才失敗。比較表每列會註明實際用了哪個模型。
- 表單下方出現「方法分數比較」表，每完成一個方法就更新；結束後依加權總分排名，各欄最佳值以粗體標示，並列出錯題數與預測耗時。
- 某個方法失敗（例如模型未訓練、Docker 映像不存在）只會在該列顯示原因，其他方法照常執行。
- 每個方法的預測檔與報告和單次執行完全相同，也會出現在「實驗比較」；按「查看詳情」可載入該方法的指標、混淆矩陣與逐筆錯題。
- 背景執行，切換分頁或重新整理後會接回最近一次的進度；重新啟動服務後，表格不保留，但已保存的報告仍可在「實驗比較」查看。

結果區另有「本次實驗耗時」卡片，以秒顯示三位小數：總耗時從後端進入實驗函式起，量到準備資料、預測、評分及輸出檔／報告儲存完成；不含檔案傳輸、頁面繪製及最後的計時 metadata 寫入。預測執行時間另列，2 GB 模式包含容器啟動與清理。單純上傳預測檔評分時顯示「未執行預測」，不把評分時間當推論時間。計時使用 perf_counter，並保存至每次報告的 run.json。

有標準答案時，「分類 macro-F1」卡片與完成訊息會顯示六位小數，直接取官方評分結果的 category 指標；「加權總分」則是四欄加權後的另一個指標。此功能同時適用自動預測與上傳預測檔評分。完全未提供標籤的資料只顯示「未評分」，不顯示 F1；即使指標相同，訓練資料回測與獨立驗證的分數仍不能直接比較。

檔名自動組成，不需要手填：

```text
20261113-105241-169925_dev_solution-a_2gb_0.908779.jsonl
日期時間戳_測試資料集名稱_方法名稱_限制名稱_得分.jsonl
```

時間為台北時間，精確到微秒；得分為官方加權總分 0～1、六位小數。資料集名稱取輸入檔名（去副檔名並清除不適合檔名的符號）。重跑另存新檔；若名稱碰撞則報錯，避免覆寫既有預測。

含標準答案的資料每筆須有 `device_id/category/urgency/duplicate_of` 四欄，系統自動評分。完全沒有標籤時，檔名最後為 `未評分`，只呈現預測、不計分也不標示對錯；混合缺漏的標籤會報錯，不把部分資料當成完整評測。算法本身仍只接收四個輸入欄位。

方案 A／Baseline 的 2 GB 映像，以及 TF-IDF 兩種方法的套件、history 訓練模型與 Docker 映像，統一依 [CPU 模型與 Docker 環境](../docs/advanced/cpu-models.md) 準備。

UI 執行本機 TF-IDF 時優先選專案 .venv，A／Baseline 使用原本的 Python。方法名稱會進入檔名（tfidf-lr／tfidf-svc）；不會自動切換分類器或偷偷訓練模型。模型、.venv、預測與報告不加入 Git。詳見 [TF-IDF 設計與比較](../Solution/docs/TF-IDF實作與比較.md)。

方案 C 的 BGE 套件、公開 encoder、分類頭與 Docker 映像也依同一份 [CPU 模型與 Docker 環境](../docs/advanced/cpu-models.md) 準備；下載只發生於事前安裝，推論不連網。

UI 提供 bge-lr／bge-svc，2 GB 會選擇 scheme-c，依舊驗證實際記憶體與 swap 限制。模型以完整 history 重訓；UI 選 history 是訓練資料回測，不等於 history 後段 90 筆的內部驗證。C 的每筆延遲包含完整共享批次服務時間，不與 B 的單筆獨立推論延遲直接比較。詳見 [BGE 方案 C](../Solution/docs/BGE方案C方法與實驗.md)。

UI 預設 history；不存在的資料集選項會停用。開啟頁面不掃描 dev 的原文、標籤或筆數，dev 僅標示供固定版本驗證。

推論期間禁止拉取映像、關閉容器網路，設定 2 GiB RAM、禁止 swap、2 CPU，並讀取 cgroup 驗證實際限制。沒有 Docker、映像未建置或預測失敗時會顯示錯誤，不會改成未限制模式卻標示為 2 GB。每次預測上限 480 秒，逾時會清理本次容器。

新流程由 `experiments.py` 負責執行與儲存，`ui_server.py` 處理 HTTP，前端沿用既有指標與錯題檢視。每次執行在 `helper_tool/reports/ui/<run_id>/` 保存報告、預測副本、執行設定 `run.json`，以及適用的逐筆量測與容器限制；重新啟動後檔案仍在，但頁面不會自動載入先前狀態。所有新產生的 JSONL、預測資料夾與 reports 仍由 Git 忽略。

## 1. 自動評分與錯誤分析

先產生預測，再評分。每次調整算法後，對新預測檔執行第二行即可。

```bash
python3 baseline/predict.py data/dev.jsonl /tmp/baseline_dev.jsonl
python3 helper_tool/evaluate.py /tmp/baseline_dev.jsonl

# 自選答案檔及報告位置：不同實驗使用不同目錄即可保留結果
python3 helper_tool/evaluate.py /tmp/baseline_dev.jsonl data/dev.jsonl --output-dir helper_tool/reports/baseline
```

終端顯示官方各欄分數、總分、較精確的六位小數總分、百分制換算與各欄問題筆數。總分直接呼叫原本的 `baseline/evaluate.py`，不修改官方程式：

`total = 0.4 × category macro-F1 + 0.2 × device accuracy + 0.2 × urgency macro-F1 + 0.2 × duplicate F1`

預設輸出到 `helper_tool/reports/<預測檔名去副檔名>/`，同一目錄重跑會覆寫。自動產生的 reports 不加入 Git。

| 報告 | 用法 |
|---|---|
| `report.md` | 快速看總分、各欄錯誤數、前十種正解→預測混淆 |
| `summary.json` | 完整精度指標、各類 precision/recall/F1/support、duplicate TP/FP/FN、缺失/多餘 ID、無效預測及建議缺漏 |
| `errors.csv` | 用試算表篩選 `wrong_fields`；同列對照原文、各欄正解、預測及原因，UTF-8 BOM 支援 Excel |
| `errors.jsonl` | 每張有問題的工單一列，含原始資料、預測、差異、來源行號與格式問題 |
| `official.txt` | 原始官方評分輸出，方便核對 |

錯誤原因：

- `wrong_value`：欄位值錯誤。
- `abstention_requires_both_null`：標準答案 device/category 同為 null，預測未同時給 null，device 不得分。
- `false_positive` / `false_negative` / `wrong_target`：duplicate 誤報、漏報或指錯單。指錯單同時計入 FP 和 FN。
- `missing_prediction` / `invalid_prediction`：漏掉整筆，或缺必要欄位／llm_used 型別錯誤。這些工單的四欄都標示問題；分數仍完全依官方規則計算。

category 的 gold=null 不列入分類 F1，也不列為分類錯題；雙 null 棄答錯誤列入 device。duplicate 的 null/null 不是 TP。錯題筆數不能直接當作 F1 的扣分，也不能把各欄錯誤數相加當作有錯工單數。`suggestion` 不計分，缺漏記在摘要。多餘 ID 由摘要列出，不計入 gold 錯題清單或總分。

只接受有標籤的 dev/history 作為答案檔；tickets 沒有答案，不能拿來算分。重複 ticket_id、壞 JSON、非 object、空答案檔及計分欄位非字串/null 會明確報錯並退出，避免產生含糊結果。這比官方對重複 ID「後一筆覆蓋」更嚴格；必要欄位缺漏及 llm_used 非 boolean 則依官方方式評分並標記無效。這是開發診斷工具，不是完整輸出 schema validator。

成功評分退出碼為 0，即使有答錯或預測格式問題；無法讀取／解析／評分時為 2。查看 `summary.json` 的檢查結果判斷完整性。

## 2. 快速查詢工單

```bash
# 預設查詢 tickets、dev、history，保留原文與所有存在的欄位
python3 helper_tool/lookup.py T02306

# 同時查多筆，方便比較 duplicate 前後單
python3 helper_tool/lookup.py T02306 T02304

# 不帶 ID：載入一次，連續輸入編號查詢，輸入 q 離開
python3 helper_tool/lookup.py

# 輸出 JSON，可供其他程式使用
python3 helper_tool/lookup.py T02306 --json

# 查指定檔案，也可以同時查原始資料及自己的預測
python3 helper_tool/lookup.py T02306 --files data/dev.jsonl /tmp/baseline_dev.jsonl
```

每筆結果包含來源檔案、原始行號及完整 JSON。精確比對 ticket_id，不改寫缺字符號或原文。跨檔同 ID 的資料會全部顯示；這只是查詢，不會跨檔推論 duplicate 或上下文。同檔重複 ID 會報錯。

找到全部 ID 退出碼為 0；任一 ID 查無資料為 1（JSON 模式對應空陣列）；讀檔或格式錯誤為 2。`--files` 接受多個路徑，請放在 ID 後面，如上例。

## 驗證

```bash
python3 -m unittest discover -s helper_tool -p 'test_*.py' -v
```

開發新算法時，建議使用 `.venv/bin/python Solution/scripts/test_development.py`，會一併執行核心與工具測試、預設略過兩項 dev 驗證測試，並阻止測試程序直接開啟 dev。舊有 dev 驗證測試只有在版本固定後明確設定 `SILAB_VALIDATE_DEV=1` 才會執行；此輪開發沒有啟用。

測試會比較 helper 的指標加權結果與官方評分，涵蓋真實 baseline、雙 null、缺失／無效預測、duplicate 指錯對象、重複 ID、壞 JSON 及跨來源查詢。測試產物放在暫存目錄。

## 從網頁訓練模型

Modal 初次使用請依 [Modal 微調設定與操作](../Solution/docs/Modal微調設定與操作.md) 完成帳號、登入與 `Solution/config/modal_finetune.json` 設定。

從側欄「模型訓練」進入獨立訓練頁，或按評分頁右上角「訓練模型」，可選用專案 history 或上傳 history JSONL、設定保存根目錄與有限範圍的微調參數。預設比較凍結 BGE＋LR／SVC、凍結神經頭、以及 **encoder＋分類頭聯合微調**。既有 TF-IDF 與凍結 BGE 四候選也可從同一頁觸發。

訓練使用背景程序，和 UI 請求分開；切換頁面／重新整理不會停止訓練。Linux 使用 `.venv/bin/python`；Windows 優先使用 `.venv/Scripts/python.exe`，其次使用 `helper_tool/reports/training-runtime/Scripts/python.exe`，否則使用啟動 UI 的 Python。依跨程序鎖，同專案一次只允許一個訓練。來源快照、設定、程式快照、log、逐 epoch 指標及模型保存於「指定根目錄／台北日期時間戳＋唯一編號」，索引保存於 `helper_tool/reports/training/`。UI 顯示最近 50 筆，舊資料不自動刪除。自訂的工作資料夾也會有 `.gitignore`，避免模型與原始資料誤入 Git。

預設 encoder lr=1e-5、head lr=1e-3、最多 5 epochs、dropout=0.2、label smoothing=0.1、weight decay=0.01，驗證六類 F1 連續兩輪未改善即停止。這些只是降低小資料過擬合風險的起點，不能保證泛化。訓練與驗證指標並列保留；模型不自動取代預設版本。

成功後按「帶入預測」，填入該次模型路徑；也可手動填入。神經模型使用 `bge-finetuned` 方法及模型資料夾，LR／SVC 使用對應方法與 `.joblib`。訓練可選 CPU、本機 CUDA GPU 或 Modal T4 GPU，**沒有 2 GB 上限**；推論可另外選 2 GB 並讀取指定模型。仍須事先安裝依賴、下載公開 BGE 權重及建置所需 Docker 映像。

訓練 API 為 `POST /api/training`，查詢為 `GET /api/training` 及 `GET /api/training?id=工作ID`，皆受原有 localhost／Origin 檢查保護。API 不接受 dev 作為訓練來源；上傳前須確認資料是 history。檔名檢查不能識別改名的 dev，資料使用仍須遵守專案規範。

CPU／GPU／Modal 設定、進度與指標說明見 [訓練頁與運算環境](../Solution/docs/訓練頁與運算環境.md)。完整設定、比較分數、模型格式與保存結構見 [BGE 聯合微調與 UI 訓練](../Solution/docs/BGE聯合微調與UI訓練.md)。

## 第二階段 Agent 評測 UI

先依 [Agent 操作文件](../agent/README.md) 準備本機模型與 CPU runtime，再進入左側「問答 Agent」。Windows 原生 Python 可直接執行 `python helper_tool/ui_server.py --port 8785`；既有模型訓練仍需 Linux/WSL 與原訓練依賴，現在不會因 Windows 缺少 fcntl 而阻止整個 UI 啟動。Linux/WSL 使用 Agent UI 時需在 `agent_assets/runtime/llama-server` 安裝原生 CPU 執行檔，不以 Windows exe 冒充 Linux runtime。

選擇測試集與 CPU threads 後按「執行 Agent 測試」。history／synthetic 問答每題重新啟動並載入模型，30 秒內完成才可通過；巡檢契約測試要求 120 秒。兩套問答共用原始 6 題公開對照表問題，屬於開發評測；巡檢只驗證格式、真實引用與工單唯一性，不把它當作排序準確率。UI 不提供 dev 選項，也不讀 dev 作為 Agent 語料或證據。

總耗時由背景 worker 的 perf_counter 計量，包含評測準備、逐題程序、模型載入與報告保存，不含 HTTP 傳輸、等待排程及最後的計時狀態寫入。每題時間直接使用原評測程式的 monotonic 外部程序計時；逐題合計不等於含準備／存檔的總耗時。執行中約每兩秒更新狀態，畫面上的即時總時間是進行中估計，完成後以保存的精確計時為準。

背景工作與網頁請求分離，重新整理或離開頁面不會終止測試；UI 服務重啟後可讀回紀錄。問答答錯會顯示「完成」與未通過題目，不混同為程序失敗；崩潰、逾時或異常中止另列狀態並保留已完成題目。一次只允許一個 Agent 測試，避免互相影響 CPU 計時。尚未套用 8 GB 硬限制。

每次紀錄存放於 Git 忽略的 `helper_tool/reports/agent/<台北時間與唯一編號>/`：`status.json` 保存逐題進度與總耗時、`report.json` 保存原始評測與版本／來源資料、`runner.log`／`runner.stderr.log`／`worker.log` 保存本機執行紀錄。完整 trace 可能包含 history 內容，只留本機，不加入 Git。

巡檢可選「完整 tickets 工單」或原有合成契約測試。目前以 `devices.json` 中 critical=true 設備的編號／名稱／型號匹配工單作候選，再依完整原文的安全詞種類數與回報時間縮選最多 30 張，由 LLM 排出前 10 張，以保留 120 秒測試上限。報告記錄候選池與模型候選數量，超時會如實判失敗。完成後在原結果表格上方逐張查看選中的工單，可點清單或使用上一張／下一張，顯示優先程度、理由、回報人、時間、完整原文與原始四欄 JSON。來源位置顯示原始檔第幾筆／總筆數，以及 JSONL 實際行號，與模型排序分開；新報告保存來源 SHA256 與選中工單快照，避免之後檔案變動造成對應錯誤。合成 SYN 工單不對應正式 2000 筆；舊報告未保存的原文與行號不事後猜補。原本結果表格、執行紀錄與 JSON 下載繼續保留。

API 為 `POST /api/agent/jobs`（kind=history／synthetic／patrol／patrol-tickets、threads=1..12），其中 patrol 為合成巡檢、patrol-tickets 使用 `data/tickets.jsonl`。另提供 `GET /api/agent/jobs`、`GET /api/agent/jobs?id=工作ID` 與 `GET /api/agent/report?id=工作ID`。沿用 localhost／Origin 保護，不接受自訂命令、任意路徑或 dev。CLI 仍可單獨執行；`evals/run.py --events` 額外提供 JSONL 進度事件給 UI，不改變模型決策或評分。

驗證：`python -X utf8 -m unittest helper_tool.test_agent_jobs helper_tool.test_ui -v`、`node helper_tool/web/test_agent_render.cjs`。本次另經正式 UI API 跑完真實 Qwen 合成問答 12 題：11/12，總耗時 169.043 秒、逐題合計 167.396 秒；不以過往 CLI 耗時冒充 UI 本次耗時。前端單元測試涵蓋時間、答錯狀態、HTML 跳脫、展開紀錄在輪詢時保留。瀏覽器鍵鼠驗證曾因工具無法確認網址而被停止，因此未宣稱已完成目視或跨螢幕驗證。

## 多筆實驗比較

評分分析頁新增「實驗比較」，列出 `helper_tool/reports/ui/` 已保存的實驗。勾選多筆後，即時並排顯示加權總分、分類 macro-F1、設備正確率、urgency macro-F1、duplicate F1、總耗時與預測耗時。支援搜尋實驗名稱／方法／資料／模型路徑、全選搜尋結果、清除勾選及重新整理。

比較表同時保留方法、執行環境、資料名稱與筆數；「模型與資料識別」可展開模型路徑與資料 SHA-256。資料雜湊不同會提示；舊紀錄沒有資料雜湊，不能僅憑檔名與筆數斷言是同一份資料。新建評分／預測工作會在 `run.json` 保存資料 SHA-256 與建立時間，不重算舊紀錄的分數或補讀舊資料。未評分、缺少的指標／耗時分別顯示「未評分」或「—」，不當成零分。不同環境的耗時包含各自程序／容器啟動範圍，不直接視為純模型延遲。

API 為 `GET /api/experiments`，只讀取各次 `run.json` 與 `summary.json`，只回傳白名單中的彙總指標及來源設定，不開啟預測 JSONL、錯題、原始 dev 或逐筆輸出。比較功能不執行推論、訓練或付費雲端工作；不改變 dev 僅供固定版本驗證的規範。

2026-11-15 驗證：26 項相關 Python 測試中 25 項通過、1 項 dev opt-in 驗證略過；比較與既有 Agent 前端渲染測試通過。實際瀏覽器已驗證兩筆 history 實驗勾選、搜尋、清除與全選搜尋結果，並顯示比較表。
