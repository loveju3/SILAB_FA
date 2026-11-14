# macOS 環境

先閱讀 [環境索引與資料準備](README.md)。本頁整理既有 macOS 操作方式；本次未在 macOS 重新執行安裝。

## 1. Python 與專案目錄

先準備 Python 3.12 或 3.13，以及專案原始碼。以下路徑為範例，請換成你的專案位置：

```bash
cd /path/to/SILAB_FA
python3 --version
```

只使用方案 A 時不需要模型套件，可在備妥本機資料後執行：

```bash
python3 predict.py data/tickets.jsonl Solution/predictions/tickets_predictions.jsonl
```

## 2. CPU 模型環境（選用）

若要使用 TF-IDF、BGE 或模型訓練，先在 macOS 建立虛擬環境，再依 [CPU 模型與 Docker](cpu-models.md) 安裝所需套件並建立模型：

```bash
python3 -m venv .venv
.venv/bin/python --version
```

不要使用從 WSL／Windows 複製過來的 `.venv`。本文件直接指定虛擬環境的 Python 路徑，不依賴 shell 是否已啟用環境。

## 3. 啟動測試 UI

沒有使用模型套件時，可使用系統 Python：

```bash
python3 helper_tool/ui_server.py --no-browser
```

已完成 CPU 模型環境時，使用：

```bash
.venv/bin/python helper_tool/ui_server.py --no-browser
```

依終端機顯示的網址開啟瀏覽器；預設為 `http://127.0.0.1:8765`，被占用時以實際輸出為準。保留終端機，按 Control+C 結束。

也可從 Finder 雙擊 `helper_tool/啟動工具.command`。此腳本使用 PATH 中的 `python3`，不會自動啟用 `.venv`；若要使用訓練套件，請採用上方明確指定 `.venv/bin/python` 的命令。

## 第二部分 Agent 的支援邊界

目前 `agent/setup.py` 只下載 Windows CPU runtime，沒有 macOS 自動安裝步驟。若已自行準備相同 GGUF 模型與原生 `llama-server`，可使用 `ask.py`／`patrol.py` 的 `--model`、`--server` 指定路徑；Agent UI 則檢查 `agent_assets/runtime/llama-server`。

這是手動 runtime 介面，不代表完整 macOS Agent 安裝流程或效能已驗證。需要現成的 Agent 安裝流程時，請依 [Windows 原生 Agent](windows-native-agent.md) 使用 Windows 環境。
