# 2026-11-16：Mac 原生 Python 環境實測

> **進階／歷史文件。** 一般安裝請用 [docs/setup](../setup/README.md) 的 `python3 install.py`，不需要照本頁操作。本頁保留先前的手動流程與實測紀錄：文中的 `.venv-windows`、`.venv-macos` 現已統一為 `.venv`；`SILAB_FA_test_models_20261115.zip` 完整模型包只有要跑 `--method tfidf-lr／tfidf-svc／bge-lr／bge-svc` 時才需要，預設的微調模型由 `install.py` 自動下載。

環境：macOS 15.6.1、arm64、Python 3.13.7。使用 repo 的 `scripts/setup_macos.py` 與兩份 requirements，建立／沿用 `.venv-macos`；本次未更改套件版本。

原先受限制的執行環境無法解析 PyPI 網域。允許安裝程序連網後，執行 `python3 scripts/setup_macos.py` 成功下載原生 Mac wheel 並安裝，exit code 0，最後輸出 `安裝結果：PASS（Python 環境已就緒）`。這不是 NumPy 版本不存在或 requirements 缺套件造成的失敗。

| 項目 | 結果 |
|---|---|
| `.venv-macos` 虛擬環境建立與路徑檢查 | 通過 |
| requirements 所有直接固定版本與套件實際載入 | 通過 |
| pip check | `No broken requirements found.` |
| NumPy／Torch 轉換與專案 NeuralHeads／HeadAdapter | 使用 `(2, 8)` float32 合成陣列完成分類，通過 |
| Transformers CPU 前向運算 | 由本機 BertConfig 建立隨機一層小型 BERT，輸出 shape `(2, 4, 8)`，通過；未下載 encoder |
| scikit-learn／SciPy 基本運算 | 四筆合成文字經 TF-IDF 與 LogisticRegression，預測符合合成標籤，通過 |

固定版本：numpy 2.3.4、scipy 1.16.3、scikit-learn 1.7.2、joblib 1.5.2、threadpoolctl 3.6.0、torch 2.9.0、transformers 4.57.1、huggingface-hub 0.36.0、tokenizers 0.22.1、safetensors 0.6.2。間接依賴由 pip 解析，並非完整 lockfile 重現驗證。

後續整合模型匯入流程：執行相同的 `python3 scripts/setup_macos.py`，自動從 Downloads 取得 `SILAB_FA_bge_finetuned_20261115-082422.zip`，依包內清單校驗並安裝到 `transfer/finetuned`。實際載入微調 encoder 與分類頭，對一筆合成文字完成推論，exit code 0，輸出 `安裝結果：PASS（環境與微調模型已就緒，模型推論測試通過）`。相關安裝與匯入測試共 14 項通過，包含失敗即停、損毀 ZIP、路徑穿越與保留既有模型。

尚未執行完整工單推論、Agent runtime、問答、巡檢、UI 或效能驗收；未讀取工單與 dev 資料。
