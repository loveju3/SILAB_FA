# 環境檢查與測試入口

> **進階／歷史文件。** 一般安裝請用 [docs/setup](../setup/README.md) 的 `python3 install.py`，不需要照本頁操作。本頁保留先前的手動流程與實測紀錄：文中的 `.venv-windows`、`.venv-macos` 現已統一為 `.venv`；`SILAB_FA_test_models_20261115.zip` 完整模型包只有要跑 `--method tfidf-lr／tfidf-svc／bge-lr／bge-svc` 時才需要，預設的微調模型由 `install.py` 自動下載。

首次建置先依 [Windows](windows.md) 或 [Mac](macos.md) 完成完整流程。一般測試不要求 WSL、Docker、2 GB 硬限制或重新訓練。尚未實測的 Mac 端到端限制見 [驗證紀錄](validation-2026-11-15.md)。

## 準備完成後執行一次

- Windows PowerShell：設定 `$env:PYTHONUTF8 = "1"`，執行 `.\.venv-windows\Scripts\python.exe scripts/check_environment.py`。
- Mac：`.venv-macos/bin/python scripts/check_environment.py`。

預期 exit 0、JSON `ready: true`。此檢查會載入必要 Python 套件、確認本機資料檔存在、驗證固定／微調 encoder 與分類頭及 Qwen 雜湊、檢查 native runtime 的 `--version` 與 Git checkout。它不讀工單原文或 dev，也不執行準確率／效能評分。

若缺少資產，依 `problems` 清單補齊。完整模型包 `SILAB_FA_test_models_20261115.zip` 解壓到 repo 根目錄；微調模型會在 `transfer/finetuned/`，其他分類模型在 `Solution/models/`。Qwen 與 runtime 由 `agent/setup.py` 放到 `agent_assets/`。資料另行準備於 `data/`。

## 測試入口

下表的 Python 都使用上述同一個環境的解譯器，不要改用全域 Python。

| 測試 | 入口／補充 |
|---|---|
| 第一部分開發測試 | `Solution/scripts/test_development.py`；封鎖 dev，opt-in dev 測試可跳過，須記錄跳過項目 |
| 六種批次方法 | `predict.py <input> <output> --method <method>`；rules、tfidf-lr、tfidf-svc、bge-lr、bge-svc、bge-finetuned |
| 微調指定路徑 | 加 `--model-path transfer/finetuned`；無須重新訓練 |
| Agent 單元測試 | `-m unittest discover -s agent/tests -v`；scripted model 不代表真實模型成績 |
| 真實問答／巡檢 | `ask.py "問題" --trace`、`patrol.py`；正式證據使用 tickets 原文與公共表 |
| 合成／history Evals | `evals/run.py --suite synthetic`、`--suite history`；需要 Git checkout；history 需本機 history |
| 巡檢 smoke | `evals/run_patrol.py --dataset synthetic` 或 `tickets` |
| 同一測試 UI | `helper_tool/ui_server.py --no-browser --port 8765`；批次與 Agent 共用 |
| 一般效能 | 外部計時啟動到輸出；A/B/C 可用 `Solution/scripts/benchmark.py`；微調不在該腳本 method 選項內，直接量 CLI |
| dev 固定版本驗證 | 先凍結版本／設定，再由驗證器執行預測及 `baseline/evaluate.py`，只回報彙總，不看逐筆內容回頭調整 |
| 2 GB／Docker | 開發者進階驗證，見 [CPU／Docker](cpu-models.md)，不阻擋一般測試者開始兩階段測試 |

可執行與答對是不同結果。模型／資料／程式版本、OS／CPU／實體 RAM、耗時、峰值記憶體都須記錄；完整 2000 筆 480 秒、問答 30 秒、巡檢 120 秒及輸出契約仍需實際驗收。使用 [41 項清單](../testing/cross_device_test_checklist.json) 填結果；若不執行其中開發者 2 GB 項目，標 not_run 並寫「由開發者另行驗證」，不要標通過。

原始工單、預測 JSONL、逐筆 trace／報告留本機；雲端只放彙總。本次過程沒有讀取 dev 或逐筆 dev 結果。
