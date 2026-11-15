# Windows 原生 Agent 環境

先閱讀 [環境索引與資料準備](README.md)。本頁使用 Windows PowerShell 與 Windows Python，適用第二部分問答 Agent、巡檢及 Agent 測試 UI。

## 1. 確認 Python

先準備 Windows Python 3.10 以上；既有 Agent 實測使用 Python 3.13.5。切換到你的專案根目錄：

```powershell
Set-Location 'C:\path\to\SILAB_FA'
python --version
```

請替換範例專案路徑。Agent 本身只用標準函式庫，不需安裝 TF-IDF／BGE 訓練套件，也不需使用 WSL 的 `.venv`。

## 2. 下載模型與 CPU runtime

```powershell
python agent/setup.py
```

此步驟需要連網，下載約 1.3 GB 的 Qwen3-1.7B Q4_K_M 與 Windows CPU llama.cpp runtime。腳本固定模型 revision、模型 SHA256、llama.cpp b12366 及發行檔 SHA256；完成後保存本機 `agent_assets/manifest.json`。Q4 模型由 bartowski 發布，原始模型為 Qwen。權重與 runtime 均由 Git 忽略。

只在這個準備階段下載；正式問答不自動下載資產，不呼叫外部模型服務。

環境建立完成前，確認以下路徑存在：

```text
agent_assets/
├── Qwen_Qwen3-1.7B-Q4_K_M.gguf
├── manifest.json
└── runtime/
    ├── llama-server.exe
    └── 配套 DLL（保留 setup 解壓出的完整 runtime）
```

若從另一台已完成 setup 的 Windows 電腦搬移，保留這三部分的路徑與配套版本；不能只搬 `llama-server.exe` 或拿 Linux 執行檔替代。完整測試前置清單見 [test-readiness.md](test-readiness.md)。

## 3. 檢查命令列與評測

```powershell
python ask.py "B1 動力室有哪些設備？" --trace
python evals/run.py --suite synthetic
```

需要正式工單問答或完整巡檢時，先備妥 `data/tickets.jsonl`：

```powershell
python patrol.py --trace
```

合成評測不是完整真實工單能力或隱藏題成績；支援範圍見 [Agent 操作文件](../../agent/README.md)。

## 4. 啟動 Agent 測試 UI

```powershell
python -X utf8 helper_tool/ui_server.py --no-browser --port 8785
```

依終端機顯示的網址開啟瀏覽器，進入「問答 Agent」。若 8785 可用，直接網址為 `http://127.0.0.1:8785/#agent`；被占用時使用終端機實際顯示的埠號。保留 PowerShell 視窗，Ctrl+C 結束服務。UI 可顯示總耗時、逐題結果及保存的測試紀錄。

## 和第一部分環境分開看待

第一部分的 `predict.py` 已加入 Windows 原生記憶體量測，規則版兩參數入口可直接執行。這不等於 TF-IDF／BGE 的模型依賴與完整效能都已在 Windows 重測；需要既有 CPU 模型及 Docker 操作流程時，仍可依 [Windows／WSL](windows-wsl.md) 準備。

訓練頁的新 CPU／CUDA／Modal 後端另依 [功能與運算環境說明](../../Solution/docs/訓練頁與運算環境.md) 準備。這些依賴不屬於 Agent 的標準函式庫環境，也不會由 `agent/setup.py` 安裝或啟動。
