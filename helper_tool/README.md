# 開發輔助工具

**資料使用規則：** dev 僅供固定版本驗證，不得觀察 dev 原文、標籤或逐筆錯題來設計／調整算法。工具保留 dev 評分與報告功能，開發時的錯誤分析請使用 history。完整規範見 [AGENTS.md](../AGENTS.md)。

評分、查詢與方案 A 只需 Python 3，沒有第三方套件。方案 B 的 TF-IDF 與方案 C 的 BGE 方法另需各自依賴與 history 訓練模型。以下指令從專案根目錄執行；工具也能從其他目錄以絕對路徑執行。使用者提供的命令列檔案路徑相對於執行目錄，UI 輸出與預設資料路徑則固定相對於專案根目錄。

## 網頁操作版（建議）

在 macOS Finder 打開 `helper_tool` 資料夾，**雙擊「啟動工具.command」**。它會自動啟動本機服務並開啟瀏覽器，不需要每次輸入指令。使用期間保留啟動視窗；結束時在該視窗按 Control+C。

介面有三個分頁：

- **評分分析**：可以直接選測試資料集、方法、執行環境及輸出資料夾，一鍵執行預測與評分；也保留上傳已有預測檔與「載入 Baseline 範例」。畫面顯示總分、四項指標、錯誤長條圖及分類混淆矩陣。自動預測後可切換「全部判斷／只看錯題」，依錯誤欄位、ID、原文或回報人篩選，逐筆對照正解、預測與建議。可下載預測 JSONL、錯題 CSV／JSONL 及指標 JSON。
- **工單查詢**：輸入一個或多個 ticket_id，按查詢或 Enter。直接呈現原文、回報人、時間、標籤、來源檔與行號；重複工單編號可點擊追查。
- **問答 Agent**：第二階段本機 LLM 評測，可選 history／合成問答集（各 12 題）與巡檢契約測試。背景執行，逐題顯示問題、預期答案、實際答案、通過狀態、含模型載入的時間與執行紀錄；同時顯示總耗時、逐題耗時合計及最長單題，支援選取既有工作與下載完整 JSON 報告。

評分檔只在本機讀取，原始資料不會被修改。每次評分報告保留在 `helper_tool/reports/ui/<時間與唯一編號>/`；重啟後檔案仍保留，網頁上的分析狀態則不保留。評分規則及檔案檢查與下方命令列工具相同。選檔每檔上限 8 MB，請求總量上限 16 MB。

預設網址為 `http://127.0.0.1:8765`，埠號被占用時自動嘗試後續 19 個埠，實際網址顯示在啟動視窗。只監聽本機，不使用外部字型、CDN 或雲端服務。若瀏覽器未自動打開，點擊或複製啟動視窗內的網址。

其他系統或偏好自行啟動時可使用 `python3 helper_tool/ui_server.py`。UI 檔案位於 `helper_tool/web/`，後端入口為 `ui_server.py`。

### 一鍵執行預測

1. 選擇 dev、history、tickets，或「自行選擇 JSONL」上傳本機資料。
2. 選方法：方案 A、Baseline、TF-IDF＋Logistic Regression 、TF-IDF＋LinearSVC，或方案 C 的 BGE＋LR／SVC。
3. 選環境：本機（未限制記憶體）或 2 GB Docker 容器。
4. 填入輸出資料夾，預設 `Solution/predictions`。接受絕對路徑；相對路徑固定依專案根目錄解析，資料夾不存在會建立。
5. 按「執行預測與評分」，完成後下方顯示結果、實際儲存位置與下載按鈕。

結果區另有「本次實驗耗時」卡片，以秒顯示三位小數：總耗時從後端進入實驗函式起，量到準備資料、預測、評分及輸出檔／報告儲存完成；不含檔案傳輸、頁面繪製及最後的計時 metadata 寫入。預測執行時間另列，2 GB 模式包含容器啟動與清理。單純上傳預測檔評分時顯示「未執行預測」，不把評分時間當推論時間。計時使用 perf_counter，並保存至每次報告的 run.json。

有標準答案時，「分類 macro-F1」卡片與完成訊息會顯示六位小數，直接取官方評分結果的 category 指標；「加權總分」則是四欄加權後的另一個指標。此功能同時適用自動預測與上傳預測檔評分。完全未提供標籤的資料只顯示「未評分」，不顯示 F1；即使指標相同，訓練資料回測與獨立驗證的分數仍不能直接比較。

檔名自動組成，不需要手填：

```text
20261113-105241-169925_dev_solution-a_2gb_0.908779.jsonl
日期時間戳_測試資料集名稱_方法名稱_限制名稱_得分.jsonl
```

時間為台北時間，精確到微秒；得分為官方加權總分 0～1、六位小數。資料集名稱取輸入檔名（去副檔名並清除不適合檔名的符號）。重跑另存新檔；若名稱碰撞則報錯，避免覆寫既有預測。

含標準答案的資料每筆須有 `device_id/category/urgency/duplicate_of` 四欄，系統自動評分。完全沒有標籤時，檔名最後為 `未評分`，只呈現預測、不計分也不標示對錯；混合缺漏的標籤會報錯，不把部分資料當成完整評測。算法本身仍只接收四個輸入欄位。

方案 A／Baseline 的 2 GB 模式需預先啟動 Docker，並從專案根目錄執行一次：

```bash
docker build -f Solution/Dockerfile -t silab-fa:scheme-a .
```

TF-IDF 兩種方法需先在本機準備：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r Solution/requirements-ml.txt
.venv/bin/python Solution/train.py
# 要跑 TF-IDF 的 2 GB 模式才需要建置：
docker build -f Solution/Dockerfile.ml -t silab-fa:scheme-b .
```

UI 執行本機 TF-IDF 時優先選專案 .venv，A／Baseline 使用原本的 Python。方法名稱會進入檔名（tfidf-lr／tfidf-svc）；不會自動切換分類器或偷偷訓練模型。模型、.venv、預測與報告不加入 Git。詳見 [TF-IDF 設計與比較](../Solution/docs/TF-IDF實作與比較.md)。

方案 C 另外執行以下步驟，下載只發生於事前安裝，推論不連網：

```bash
.venv/bin/python -m pip install -r Solution/requirements-embedding.txt
.venv/bin/python Solution/scripts/download_encoder.py
.venv/bin/python Solution/train_embedding.py
docker build -f Solution/Dockerfile.embedding -t silab-fa:scheme-c .
```

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

按工作台右上角「訓練模型」開啟 modal，可選用專案 history 或上傳 history JSONL、設定保存根目錄與有限範圍的微調參數。預設比較凍結 BGE＋LR／SVC、凍結神經頭、以及 **encoder＋分類頭聯合微調**。既有 TF-IDF 與凍結 BGE 四候選也可從同一個視窗觸發。

訓練使用 `.venv` 的背景程序，和 UI 請求分開；關閉 modal／重新整理頁面不會停止訓練。依跨程序鎖，同專案一次只允許一個訓練。來源快照、設定、程式快照、log、逐 epoch 指標及模型保存於「指定根目錄／台北日期時間戳＋唯一編號」，索引保存於 `helper_tool/reports/training/`。UI 顯示最近 50 筆，舊資料不自動刪除。自訂的工作資料夾也會有 `.gitignore`，避免模型與原始資料誤入 Git。

預設 encoder lr=1e-5、head lr=1e-3、最多 5 epochs、dropout=0.2、label smoothing=0.1、weight decay=0.01，驗證六類 F1 連續兩輪未改善即停止。這些只是降低小資料過擬合風險的起點，不能保證泛化。訓練與驗證指標並列保留；模型不自動取代預設版本。

成功後按「帶入預測」，填入該次模型路徑；也可手動填入。神經模型使用 `bge-finetuned` 方法及模型資料夾，LR／SVC 使用對應方法與 `.joblib`。訓練本身在本機 CPU 執行，**沒有 2 GB 上限**；推論可另外選 2 GB 並讀取指定模型。仍須事先安裝依賴、下載公開 BGE 權重及建置所需 Docker 映像。

訓練 API 為 `POST /api/training`，查詢為 `GET /api/training` 及 `GET /api/training?id=工作ID`，皆受原有 localhost／Origin 檢查保護。API 不接受 dev 作為訓練來源；上傳前須確認資料是 history。檔名檢查不能識別改名的 dev，資料使用仍須遵守專案規範。

完整設定、比較分數、模型格式與保存結構見 [BGE 聯合微調與 UI 訓練](../Solution/docs/BGE聯合微調與UI訓練.md)。

## 第二階段 Agent 評測 UI

先依 [Agent 操作文件](../agent/README.md) 準備本機模型與 CPU runtime，再進入左側「問答 Agent」。Windows 原生 Python 可直接執行 `python helper_tool/ui_server.py --port 8785`；既有模型訓練仍需 Linux/WSL 與原訓練依賴，現在不會因 Windows 缺少 fcntl 而阻止整個 UI 啟動。Linux/WSL 使用 Agent UI 時需在 `agent_assets/runtime/llama-server` 安裝原生 CPU 執行檔，不以 Windows exe 冒充 Linux runtime。

選擇測試集與 CPU threads 後按「執行 Agent 測試」。history／synthetic 問答每題重新啟動並載入模型，30 秒內完成才可通過；巡檢契約測試要求 120 秒。兩套問答各含 6 題相同的公開對照表問題，屬於開發評測；巡檢只驗證格式、真實引用與工單唯一性，不把它當作排序準確率。UI 不提供 dev 選項，也不讀 dev 作為 Agent 語料或證據。

總耗時由背景 worker 的 perf_counter 計量，包含評測準備、逐題程序、模型載入與報告保存，不含 HTTP 傳輸、等待排程及最後的計時狀態寫入。每題時間直接使用原評測程式的 monotonic 外部程序計時；逐題合計不等於含準備／存檔的總耗時。執行中約每兩秒更新狀態，畫面上的即時總時間是進行中估計，完成後以保存的精確計時為準。

背景工作與網頁請求分離，重新整理或離開頁面不會終止測試；UI 服務重啟後可讀回紀錄。問答答錯會顯示「完成」與未通過題目，不混同為程序失敗；崩潰、逾時或異常中止另列狀態並保留已完成題目。一次只允許一個 Agent 測試，避免互相影響 CPU 計時。尚未套用 8 GB 硬限制。

每次紀錄存放於 Git 忽略的 `helper_tool/reports/agent/<台北時間與唯一編號>/`：`status.json` 保存逐題進度與總耗時、`report.json` 保存原始評測與版本／來源資料、`runner.log`／`runner.stderr.log`／`worker.log` 保存本機執行紀錄。完整 trace 可能包含 history 內容，只留本機，不加入 Git。

API 為 `POST /api/agent/jobs`（kind=history／synthetic／patrol、threads=1..12），`GET /api/agent/jobs`、`GET /api/agent/jobs?id=工作ID` 與 `GET /api/agent/report?id=工作ID`。沿用 localhost／Origin 保護，不接受自訂命令、任意路徑或 dev。CLI 仍可單獨執行；`evals/run.py --events` 額外提供 JSONL 進度事件給 UI，不改變模型決策或評分。

驗證：`python -X utf8 -m unittest helper_tool.test_agent_jobs helper_tool.test_ui -v`、`node helper_tool/web/test_agent_render.cjs`。本次另經正式 UI API 跑完真實 Qwen 合成問答 12 題：11/12，總耗時 169.043 秒、逐題合計 167.396 秒；不以過往 CLI 耗時冒充 UI 本次耗時。前端單元測試涵蓋時間、答錯狀態、HTML 跳脫、展開紀錄在輪詢時保留。瀏覽器鍵鼠驗證曾因工具無法確認網址而被停止，因此未宣稱已完成目視或跨螢幕驗證。
