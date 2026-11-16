# Windows：一套環境跑第一、第二階段

> **進階／歷史文件。** 一般安裝請用 [docs/setup](../setup/README.md) 的 `python3 install.py`，不需要照本頁操作。本頁保留先前的手動流程與實測紀錄：文中的 `.venv-windows`、`.venv-macos` 現已統一為 `.venv`；`SILAB_FA_test_models_20261115.zip` 完整模型包只有要跑 `--method tfidf-lr／tfidf-svc／bge-lr／bge-svc` 時才需要，預設的微調模型由 `install.py` 自動下載。

適用 Windows x64、Python 3.12／3.13。整份流程都在 Windows PowerShell 執行，不需要 WSL 或 Docker；2 GB 演練留給開發者。Windows ARM 原生未驗證。

## 1. 取得工具與 repo

安裝 Git、Windows x64 Python 3.12 或 3.13，取得最新 repo，切到根目錄：

~~~powershell
Set-Location 'C:\path\to\SILAB_FA'
python --version
git --version
$env:PYTHONUTF8 = "1"
~~~

請替換專案路徑。UTF-8 設定會讓目前終端啟動的 Python 及其測試子程序使用 UTF-8，避免繁體中文 Windows 的 CP950 讀檔錯誤。重新開 PowerShell 時再設定一次。

## 2. 放好資料與完整測試模型包

1. 在本機放入 data/devices.json、data/error_codes.json、data/tickets.jsonl、data/history.jsonl。原始資料不隨 Git 提供，也不放雲端。
2. 向維護者取得 SILAB_FA_test_models_20261115.zip，**解壓到 repo 根目錄**。不是只有 finetuned 的舊 ZIP；完整包同時包含 TF-IDF、固定 BGE 與微調 BGE，不需重新訓練。
3. 核對以下位置，不要多包一層同名資料夾：

~~~text
SILAB_FA/
├── Solution/models/
│   ├── tfidf-lr.joblib、tfidf-lr.json
│   ├── tfidf-svc.joblib、tfidf-svc.json
│   ├── bge-lr.joblib、bge-lr.json
│   ├── bge-svc.joblib、bge-svc.json
│   └── bge-small-zh-v1.5/（完整 encoder 與 manifest）
├── transfer/finetuned/
│   ├── model.json
│   ├── heads.safetensors
│   └── encoder/（完整微調 encoder 與 manifest）
└── TEST_MODELS_MANIFEST.json
~~~

ZIP 可由維護者自行放到 [共用資料夾](https://drive.google.com/drive/folders/1QeCQjNet73n8FYnrJ4WLmnpGKPnNc7zt)，若找不到請索取。舊的 SILAB_FA_bge_finetuned_20261115-082422.zip 只包含微調版，不能代替完整 B/C 模型包。

## 3. 建立唯一 Windows Python 環境

~~~powershell
python -m venv .venv-windows
$py = ".\.venv-windows\Scripts\python.exe"
& $py -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
& $py -m pip install -r Solution/requirements-embedding.txt
& $py -m pip check
~~~

requirements-embedding.txt 同時安裝 TF-IDF、BGE、微調推論需要的固定套件；Agent 使用同一個 Python。不要複製其他作業系統的 .venv，不要用 WSL 的解譯器。

## 4. 完成 Agent 模型與原生 runtime

~~~powershell
& $py agent/setup.py
~~~

安裝器在準備階段下載並校驗 Qwen3-1.7B Q4 GGUF 與固定 b12366 的 Windows CPU llama.cpp，放到：

~~~text
agent_assets/
├── Qwen_Qwen3-1.7B-Q4_K_M.gguf
├── manifest.json
└── runtime/（llama-server.exe 及全部配套 DLL）
~~~

保留完整 runtime，不可只搬 exe。已存在且校驗符合的檔案會重用；推論不自行下載模型。

## 5. 確認環境已齊全

~~~powershell
& $py scripts/check_environment.py
~~~

預期 exit code 0 且 JSON ready 為 true；problems 會列出缺少的檔案／套件。檢查包含六種批次方法所需資產、Qwen 雜湊、runtime 能啟動、Git 與本機資料檔存在；不開啟任何工單內容，也不讀 dev。這不是準確率／完整效能驗收。

## 6. 使用同一環境跑兩階段測試

~~~powershell
& $py Solution/scripts/test_development.py
& $py -m unittest discover -s agent/tests -v
& $py predict.py data/tickets.jsonl reports/rules_predictions.jsonl
& $py predict.py data/tickets.jsonl reports/finetuned_predictions.jsonl --method bge-finetuned --model-path transfer/finetuned
& $py ask.py "B1 動力室有哪些設備？" --trace
& $py patrol.py
& $py evals/run.py --suite synthetic
& $py evals/run.py --suite history
& $py evals/run_patrol.py --dataset tickets
~~~

固定模型另用 --method tfidf-lr、tfidf-svc、bge-lr、bge-svc，分別指定自己的輸出檔名。正式推論使用本機 CPU、不得呼叫外部 API；--trace 在 stderr，Agent 最終 JSON 在 stdout。

開發測試會封鎖 dev 並跳過未啟用的 dev 驗證；真實 Agent 題庫可能有既有能力限制，環境可執行不等於每題都正確。完整目的、輸出與結果欄位見 [41 項驗收清單](../testing/cross_device_test_checklist.json)。dev 僅在候選版本與設定固定後交由驗證程序評分，逐筆資料只留本機，不用來調整算法。

## 7. 開啟同一個測試 UI

~~~powershell
& $py helper_tool/ui_server.py --no-browser --port 8765
~~~

依終端實際網址開啟瀏覽器。同一服務可操作批次與 Agent；微調預測選 bge-finetuned 並填 transfer/finetuned。關閉時按 Ctrl+C。

本機實測及範圍見 [驗證紀錄](validation-2026-11-15.md)。重新開終端後，先切到 repo、設定 $env:PYTHONUTF8="1"，再指定 $py 即可；不用重裝套件或模型。
