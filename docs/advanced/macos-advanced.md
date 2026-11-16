# Mac：完整驗收與 Agent

> **進階／歷史文件。** 一般安裝請用 [docs/setup](../setup/README.md) 的 `python3 install.py`，不需要照本頁操作。本頁保留先前的手動流程與實測紀錄：文中的 `.venv-windows`、`.venv-macos` 現已統一為 `.venv`；`SILAB_FA_test_models_20261115.zip` 完整模型包只有要跑 `--method tfidf-lr／tfidf-svc／bge-lr／bge-svc` 時才需要，預設的微調模型由 `install.py` 自動下載。

一般使用者請先完成 [Mac 安裝](macos.md)。以下是完整模型驗收、Agent 與開發測試，並非微調模型安裝的必要步驟。

## 2. 按需放入資料與模型

完整兩階段驗收需在本機準備 data/devices.json、data/error_codes.json、data/tickets.jsonl、data/history.jsonl。資料不隨 Git 提供，也不傳到雲端。這些檔案不影響前一步 Python 環境安裝。

若先只跑微調推論，可稍後放入完整的 `transfer/finetuned/`（下列 model.json、heads.safetensors 與完整 encoder/），不需準備其他四組分類模型或重新訓練。推論仍需設備／錯誤碼表與輸入工單；完整環境檢查則會要求所有驗收資產。

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

## 3. 使用 Agent 時下載模型與原生 runtime

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

## 4. 完整驗收前確認所有資產齊全

~~~bash
.venv-macos/bin/python scripts/check_environment.py
~~~

預期 exit code 0、JSON ready=true；缺套件、模型、runtime 或資料時 problems 會列出原因。檢查不讀任何工單內容／dev，也不代表推論準確率或效能達標。若 Mac runtime 不能啟動，保留錯誤摘要回報；不要把只完成下載當成實機驗證。

## 5. 使用同一環境跑兩階段測試

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

## 6. 開啟同一個測試 UI

~~~bash
.venv-macos/bin/python helper_tool/ui_server.py --no-browser --port 8765
~~~

依終端實際網址開啟瀏覽器。批次與 Agent 共用這個服務；Ctrl+C 結束。使用明確的 .venv-macos/bin/python，避免 Finder 啟動器拿到另一套 Python。
