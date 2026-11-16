# Mac 安裝與測試

> **進階／歷史文件。** 一般安裝請用 [docs/setup](../setup/README.md) 的 `python3 install.py`，不需要照本頁操作。本頁保留先前的手動流程與實測紀錄：文中的 `.venv-windows`、`.venv-macos` 現已統一為 `.venv`；`SILAB_FA_test_models_20261115.zip` 完整模型包只有要跑 `--method tfidf-lr／tfidf-svc／bge-lr／bge-svc` 時才需要，預設的微調模型由 `install.py` 自動下載。

適用 Apple Silicon Mac，需先安裝 Python 3.12 或 3.13。

## 1. 下載

- 下載並解壓 repo，開啟終端機，切換到專案資料夾。
- 將 `SILAB_FA_bge_finetuned_20261115-082422.zip` 下載到 Mac 的「下載項目」，**不用解壓縮**。

## 2. 安裝

~~~bash
python3 scripts/setup_macos.py
~~~

腳本會自動建立環境、安裝套件、匯入模型，並實際測試模型推論。

- **PASS**：環境與模型安裝完成。
- **FAIL**：尚未完成，依最後的錯誤訊息處理後重跑。

模型 ZIP 放在其他位置時，可改用 `python3 scripts/setup_macos.py --model-zip "模型 ZIP 的完整路徑"`。

## 3. 測試自己的工單

將取得的 `devices.json`、`error_codes.json`、`tickets.jsonl` 放進專案的 `data` 資料夾，執行：

~~~bash
python3 scripts/setup_macos.py --test
~~~

看到 **工單測試：PASS** 即完成，結果位於 `reports/finetuned_predictions.jsonl`。資料與結果只留本機。

需要 Agent 或完整驗收時，見 [進階操作](macos-advanced.md)。
