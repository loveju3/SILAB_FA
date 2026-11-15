# Mac：一套環境跑第一、第二階段

本流程針對 **Apple Silicon（M1／M2／M3 等 arm64）**，使用 Python 3.12／3.13。第一與第二階段共用同一個原生 Python 環境，不要求 Docker 或 2 GB 演練。

**本次沒有 Mac 實機，以下尚未完成端到端驗證。** 已查核固定 runtime 發行檔、Apple Silicon 檔案雜湊／目錄結構，並測試安裝器邏輯。

## 1. 確認工具與架構

先安裝 Git、arm64 Python 3.12 或 3.13，取得最新 repo。在終端執行：

~~~bash
cd /path/to/SILAB_FA
uname -m
python3 --version
python3 -c "import platform; print(platform.machine())"
git --version
~~~

uname 與 Python 架構都應為 arm64，請使用原生 arm64 Python。

## 2. 放好資料與完整測試模型包

本機準備 data/devices.json、data/error_codes.json、data/tickets.jsonl、data/history.jsonl。資料不隨 Git 提供，也不傳到雲端。

向維護者取得 SILAB_FA_test_models_20261115.zip，解壓到 repo 根目錄。維護者可自行放至 [共用資料夾](https://drive.google.com/drive/folders/1QeCQjNet73n8FYnrJ4WLmnpGKPnNc7zt)，未找到時索取，不會由推論自動下載。確認：

~~~text
Solution/models/
├── tfidf-lr.joblib、tfidf-lr.json
├── tfidf-svc.joblib、tfidf-svc.json
├── bge-lr.joblib、bge-lr.json
├── bge-svc.joblib、bge-svc.json
└── bge-small-zh-v1.5/（完整 encoder 與 manifest）
transfer/finetuned/
├── model.json
├── heads.safetensors
└── encoder/（完整微調 encoder 與 manifest）
TEST_MODELS_MANIFEST.json
~~~

這些是已訓練完成的推論資產，不需要在這台重新訓練。只有 finetuned 的舊 ZIP 不足以跑所有 B/C 方法。

## 3. 建立唯一 Mac Python 環境

~~~bash
python3 -m venv .venv-macos
.venv-macos/bin/python -m pip install -r Solution/requirements-embedding.txt
.venv-macos/bin/python -m pip check
~~~

不要使用 Windows／WSL 的虛擬環境，也不要在 Mac 指定 Linux／Windows CPU wheel index。這一份 requirements 已包含 TF-IDF、BGE、微調推論及 Agent 共用的 Python 環境所需套件。

## 4. 下載 Agent 模型與原生 runtime

~~~bash
.venv-macos/bin/python agent/setup.py
~~~

安裝器依原生架構選擇固定 [llama.cpp b12366 官方發行檔](https://github.com/ggml-org/llama.cpp/releases/tag/b12366)，下載並校驗 GGUF 與 runtime，解壓成：

~~~text
agent_assets/
├── Qwen_Qwen3-1.7B-Q4_K_M.gguf
├── manifest.json
└── runtime/
    ├── llama-server
    └── 配套 dylib／相對符號連結（須完整保留）
~~~

不需手動改寫 --server，ask.py、patrol.py 與 UI 使用平台原生路徑。模型實際啟動使用 -ngl 0，仍按 CPU 推論測試。

## 5. 確認環境已齊全

~~~bash
.venv-macos/bin/python scripts/check_environment.py
~~~

預期 exit code 0、JSON ready=true；缺套件、模型、runtime 或資料時 problems 會列出原因。檢查不讀任何工單內容／dev，也不代表推論準確率或效能達標。若 Mac runtime 不能啟動，保留錯誤摘要回報；不要把只完成下載當成實機驗證。

## 6. 使用同一環境跑兩階段測試

~~~bash
.venv-macos/bin/python Solution/scripts/test_development.py
.venv-macos/bin/python -m unittest discover -s agent/tests -v
.venv-macos/bin/python predict.py data/tickets.jsonl reports/rules_predictions.jsonl
.venv-macos/bin/python predict.py data/tickets.jsonl reports/finetuned_predictions.jsonl --method bge-finetuned --model-path transfer/finetuned
.venv-macos/bin/python ask.py "B1 動力室有哪些設備？" --trace
.venv-macos/bin/python patrol.py
.venv-macos/bin/python evals/run.py --suite synthetic
.venv-macos/bin/python evals/run.py --suite history
.venv-macos/bin/python evals/run_patrol.py --dataset tickets
~~~

固定模型另用 --method tfidf-lr、tfidf-svc、bge-lr、bge-svc，使用各自輸出檔名。正式推論離線；完整資料、預測、逐筆 trace／報告只留本機。開發測試禁止 dev；dev 僅在候選版本固定後由驗證程序使用。題庫有既有能力限制，安裝成功不代表每題都會通過。完整驗收見 [JSON 清單](../testing/cross_device_test_checklist.json)。

## 7. 開啟同一個測試 UI

~~~bash
.venv-macos/bin/python helper_tool/ui_server.py --no-browser --port 8765
~~~

依終端實際網址開啟瀏覽器。批次與 Agent 共用這個服務；Ctrl+C 結束。使用明確的 .venv-macos/bin/python，避免 Finder 啟動器拿到另一套 Python。
