# Solution：維修工單分流

目前完成 **方案 A：完整規則版**、**方案 B：字元 TF-IDF＋線性分類**，以及 **方案 C：固定 BGE encoder＋線性分類**。A 僅需 Python 3.10+ 標準函式庫；B/C 使用各自固定依賴與本機資產，推論皆不需連網。第二部分新增獨立的 [Agent 初版](../agent/README.md)，不載入這裡的 A/B/C 模組或規則資產。

方案 B 可先閱讀 [字元 TF-IDF 分類方法說明與流程圖](docs/字元TF-IDF分類方法說明.md)，再看 [實作、四種對照與操作](docs/TF-IDF實作與比較.md)。全部文件見 [docs 索引](docs/README.md)。

方案 C 的 CLS 向量、批次流程、history 取捨與重現指令見 [BGE 方法與實驗](docs/BGE方案C方法與實驗.md)。

新增 [BGE 聯合微調與 UI 訓練](docs/BGE聯合微調與UI訓練.md)：在工作台按「訓練模型」，可上傳 history、指定保存根目錄，比較凍結 LR／SVC、凍結神經頭及 encoder＋分類頭聯合微調。設定、來源、逐 epoch 指標與 checkpoint 保存在各自時間戳資料夾；既有 B/C 四候選也能從 UI 觸發。新 checkpoint 可直接帶入預測，不覆寫原預設模型。

**開發算法不得觀察 dev 資料；dev 只用於固定版本驗證。** 開發、錯誤分析與調參使用 history 及其內部分割，遵循 [專案規範](../AGENTS.md)。現有 A 曾參考 dev，不能當作符合新規範的獨立開發結果。

想先了解判斷方式，可閱讀 [演算法流程與流程圖](docs/演算法流程.md)，用白話說明設備、分類、上下文、急迫性與重複報修的判斷順序。

## 資料夾與責任

```text
Solution/
├── predict.py                命令列、資產載入、計時與輸出協調
├── triage/
│   ├── contracts.py          純輸入/輸出型別與輸出驗證
│   ├── io.py                 JSONL、四欄投影、原子寫檔
│   ├── text.py               正規化、缺字匹配、錯誤碼解析
│   ├── devices.py            設備候選與證據、名稱遮罩
│   ├── classification.py     正向症狀規則分類；無資訊回 null
│   ├── rules.py              urgency、上下文與具體建議
│   ├── pipeline.py           時序處理、單檔狀態與 duplicate 索引
│   └── metrics.py            延遲分位數與程序 RSS
├── config/                   設備別名及分類規則資料
├── training/                 B/C 共用 history 分割、分類頭與選型指標
├── train.py                  LR／SVC × 名稱保留／遮罩四種實驗與重訓
├── train_embedding.py        C 固定 BGE 向量、四候選比較與重訓
├── models/                   本機模型與 metadata，不加入 Git
├── requirements-ml.txt       B 的固定依賴；A 不需要安裝
├── requirements-embedding.txt C 的固定依賴（含 B 的依賴）
├── docs/                     白話演算法說明與 Mermaid 流程圖
├── tests/                    行為與契約測試
├── scripts/                  本機量測、容器限制查驗與 2 GB 演練
├── predictions/              dev、baseline、正式 2,000 筆與 2 GB 輸出
├── reports/                  本機產生的評分、錯題及量測，不加入 Git
├── experiments/              實驗摘要；runs/ 原始結果留在本機
├── Dockerfile                標準 Python 容器建置定義
├── Dockerfile.ml             B 的 ML 依賴與本機模型映像
├── Dockerfile.embedding      C 的 CPU PyTorch、encoder 與分類頭映像
├── bench.md                  根目錄效能量測文件的導引頁
└── budget.md                 2 GB 限制證據及實測
```

技術選型表位於專案根目錄：[技術選型對照表](../技術選型對照表.md)。

根目錄 `predict.py` 僅轉接到 Solution，維持題目兩參數入口。**未指定 `--method` 時，已安裝 `transfer/finetuned/` 就使用 BGE 聯合微調（`bge-finetuned`），否則退回方案 A（`rules`）並在 stderr 警告；`--mem-limit 2g` 固定使用方案 A。**資料仍使用根目錄 `data/`，不複製另一份訓練/驗證資料。

資料隱私：`data/`、所有 JSONL、預測、錯題報告及含工單 ID 的實驗摘要只留在本機，已加入 Git 忽略規則。其他環境須自行提供資料；`.gitignore` 不會移除既有 Git 歷史中的檔案，因此清理歷史前不可直接推送包含資料的舊提交。

## 執行與評分

從專案根目錄執行：

```bash
python3 predict.py data/dev.jsonl Solution/predictions/dev_A.jsonl --method rules --metrics Solution/reports/dev_A_metrics.json
python3 helper_tool/evaluate.py Solution/predictions/dev_A.jsonl --output-dir Solution/reports/dev_A
python3 predict.py data/tickets.jsonl Solution/predictions/tickets_A.jsonl --method rules
python3 -m unittest discover -s Solution/tests -p 'test_*.py' -v
python3 Solution/scripts/benchmark.py --runs 3
```

也可直接執行 `python3 Solution/predict.py INPUT OUTPUT`；可從其他工作目錄使用絕對路徑，使用者指定的相對路徑依執行目錄解析。stdout 保持空白，錯誤到 stderr 並回傳 2；成功為 0。空輸入產生空輸出；壞 JSON、重複 ID、缺欄、錯誤型別與無時區日期明確拒絕。完整驗證後才原子替換輸出，失敗不留下半份預測。

現有 UI 的「評分分析」可上傳 `Solution/predictions/dev_A.jsonl`，標準答案選 dev。完整錯題與 baseline 對照另存 `reports/dev_A/` 和 `reports/comparison.json`。

也可直接在 UI 選擇測試資料、方案 A／B／C／Baseline、本機／2 GB 環境及輸出資料夾，一鍵產生含台北時間、資料集、方法、限制與分數的預測檔；下方切換全部判斷與錯題。操作與命名格式見 [輔助工具說明](../helper_tool/README.md)。

## 推論規則與資料邊界

1. JSONL 入口只投影 `ticket_id / created_at / reporter / raw_text` 為 Ticket，核心函式沒有標準答案欄位。測試確認加入錯誤標籤不改變預測。
2. 日期轉 UTC 後穩定排序，同時間保持原行序；輸出恢復輸入順序。上下文及 duplicate 狀態每次呼叫重新建立。
3. 原文保留；比對文字處理全半形、明列繁簡變體、連續重字及零寬字。分類視圖移除簽名檔；urgency 另從原文建立視圖，不加入前文或錯誤碼釋義。
4. 缺字匹配限長度至少三字的詞，允許一個字由 `□`、`�` 或 1～2 個問號取代；不容許任意刪字或語意同義詞。此策略可能誤匹配，需在新資料驗證。
5. 設備名稱、別名、型號與 ID 優先；其次有效錯誤碼與位置。共用位置保留候選，多個明確設備仍可能回 null。明確設備不被矛盾錯誤碼覆寫。
6. 分類先用已知錯誤碼；否則遮罩設備名稱/位置/型號，以設定權重與詞長選擇正向症狀。無正向證據回 null，不強迫 other；other 有自己的故障證據。設備未知不清空可判斷的類別。
7. 同回報人前單只有在當筆出現指涉且候選無衝突時提供設備；分類不整筆沿用。只有明確「同一個問題／老問題／上次那件」且設備一致、當筆無新症狀時才參考前單類別。
8. urgency 嚴格依 schema 的六條規則順序；critical 使用自判設備。duplicate 使用 `(reporter, 預測設備, 預測類別)` 的最近前單，null 不建索引。
9. 建議引用當筆原文片段與類別處置模板，最多 40 字；high 先確認安全與通知維修。全部 `llm_used=false`。

規則詞彙參考 schema、設備表、錯誤碼及 history 的代表案例；開發前期也曾觀察 dev。此版本 dev 首輪評估後未以錯題新增分類詞，但這無法消除前期影響，分數不代表無偏泛化能力，也不能追溯視為符合新規範。分類不是語意理解，主症狀、否定、未知措辭與錯誤碼衝突仍可能判錯。

## 目前結果與下一步

完整 dev 總分 **0.908779**，原始 baseline **0.441574**；27 項新舊測試全部通過。方案 A 有 70 張錯題，其中 category 62 筆、device 2 筆、duplicate 21 筆，欄位錯誤可能重疊。

分類相對 baseline 修正 101 筆、引入 26 筆錯誤。資訊不足策略也會漏掉有真實類別的口語故障；這是接方案 B 要處理的主要問題，不再靠持續堆 dev 關鍵詞消除。

已完成 B 與 C 的 history 內部比較：B 固定各方法的遮罩版，C 依相同的六類指標選到保留名稱版。兩者只用 history 訓練與選型，本機目前未提供 dev，尚未執行其驗證。沿用 A 的正規化、設備及上下文仍保留來源限制，不能宣稱整條流程全新未觀察 dev。C 的 masked 替代版也保留，未因其單一分類分數更高就事後改選型規則。

目前開發測試建議執行 `.venv/bin/python Solution/scripts/test_development.py`：核心與工具測試一併執行，2 項 dev 驗證測試預設略過，並攔截測試程序直接開啟 dev。這不是對所有外部程序的檔案存取沙箱，仍須遵守資料使用規範。

標準 Python 映像已於重試後建置成功，並完成 2 GB 演練與不掛載原始碼的獨立執行，詳見 [budget.md](budget.md)。本機測試為 Apple M4 Max、36 GiB RAM，沒有宣稱已在實體 8 GB／2 GB 筆電或 x86 上驗證。
