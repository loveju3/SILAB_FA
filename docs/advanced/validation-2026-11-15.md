# 2026-11-15：依實測整理 Windows／Mac 建置入口

> **進階／歷史文件。** 一般安裝請用 [docs/setup](../setup/README.md) 的 `python3 install.py`，不需要照本頁操作。本頁保留先前的手動流程與實測紀錄：文中的 `.venv-windows`、`.venv-macos` 現已統一為 `.venv`；`SILAB_FA_test_models_20261115.zip` 完整模型包只有要跑 `--method tfidf-lr／tfidf-svc／bge-lr／bge-svc` 時才需要，預設的微調模型由 `install.py` 自動下載。

結論：Windows x64 可以將兩階段放在同一個原生 Python 環境；一般測試不需要 WSL 或 Docker。Mac 可準備對應的原生流程，但本次只有 Windows 實機，不能宣稱 Mac 端到端已通過。2 GB、重新訓練與 Docker 打包移至進階文件。

## Windows 實際執行

Windows 11 x64、Python 3.13.5，在 repo 新建 `.venv-windows/`。安裝 CPU torch 2.9.0+cpu、transformers 4.57.1、scikit-learn 1.7.2、numpy 2.3.4、scipy 1.16.3、safetensors 0.6.2；pip check 通過。安裝重用下載快取，Qwen 與 runtime 重用已有檔案並校驗，不是另一台全新機器的完整網路下載測試。

| 項目 | 結果 | 實際範圍 |
|---|---|---|
| rules、tfidf-lr、tfidf-svc、bge-lr、bge-svc、bge-finetuned | 六種全部 exit 0 | 相同 Windows 原生環境、既有合成 fixture；檢查筆數與必要輸出型別，不是完整 tickets 效能或準確率驗收 |
| 第一部分及 helper_tool 開發測試 | 90 項中 88 通過、2 跳過 | `Solution/scripts/test_development.py` 封鎖 dev；未啟用的驗證測試依規則跳過 |
| Agent／安裝器單元測試 | 31 項通過 | 包含模型決策控制與新增跨平台 runtime 選擇、安全解壓、版本目錄攤平、重複安裝檔案重用 |
| 真實 Qwen 問答 | public-location 1/1，9.450 秒 | 新程序、CPU llama.cpp；不是完整 30 題評測通過宣稱 |
| 真實 Qwen 巡檢 | 合成 contract PASS，16.561 秒 | 引用、十張、唯一性、格式與時間；不是完整 tickets 排序品質驗收 |
| 環境檢查器 | ready=true | 必要套件、固定／微調模型雜湊、Qwen、原生 runtime、Git 及資料存在 |

實測發現並處理：

- Windows 預設 CP950 會讓部分現有測試以錯誤編碼讀取 UTF-8 輸出。建置流程要求 `$env:PYTHONUTF8 = "1"`，套用到後續 Python 與測試子程序；設定後開發測試通過。
- 本機 HTTP 測試與 Agent 需要 loopback。沙箱阻擋 socket 時會出現權限錯誤／逾時；允許 loopback 後重驗成功。推論不使用外部雲端 API，但不是禁止一切本機 socket。
- 重跑 setup 時若有模型程序使用 Windows DLL，不能重寫相同 DLL。安裝器會比對既有內容，重用符合的檔案；檔案變更仍須先關閉使用它的模型程序。

## Mac 已查核與尚未查核

已由 [llama.cpp b12366 官方 release](https://github.com/ggml-org/llama.cpp/releases/tag/b12366) 核對 macOS arm64 資產與固定 SHA-256；實際下載 arm64 的 tar.gz，確認雜湊及內含版本目錄、llama-server、配套 dylib／相對符號連結。

- arm64 archive SHA-256：`6263965248c6c17b942eee1d91d9f810566d2ed1a7ec884130deb2339baee842`。
- 安裝器新增 Mac 架構選擇、安全 tar 解壓及完整 dylib 保留；相關邏輯在 Windows 上用合成壓縮檔測試。
- `ask.py`／`patrol.py` 改用平台原生 server 預設路徑，不再一律指向 Windows exe。

**沒有 Mac 實機**：沒有實際執行 Mac pip 安裝、GGUF 下載器、runtime、六種模型推論、Agent、巡檢或 UI。macOS 權限、動態函式庫載入、原生套件相容性與效能仍待 Apple Silicon 實機驗證。Windows 或 Linux Docker 成績不能代替這些驗證。

## 模型與資料邊界

新增完整推論模型包 `SILAB_FA_test_models_20261115.zip`，解壓到 repo 根目錄即可放好 B/C 分類頭、固定 encoder 與 `transfer/finetuned/`。共 24 個模型檔案及一份 manifest，ZIP CRC 與每檔 SHA-256 已核對。Qwen／平台 runtime 仍由安裝器另行取得。模型包由維護者自行上傳，不隨 Git 提供。

全部新驗證只用合成資料與公共表；沒有讀 dev 或其逐筆衍生結果，也沒有根據 dev 選模型。原始資料、模型、預測及逐筆 log 都留在 Git 忽略目錄。沿用方案 A 資產的既有 dev 影響限制保持不變。

完整 2000 筆 480 秒、單題全面評測、正式巡檢 120 秒、8 GB 實機、2 GB 硬限制與隱藏題均不在本次環境整合結果內。完整 Evals 仍可能有既有能力上的失敗；環境可以跑不代表全部題目正確。
