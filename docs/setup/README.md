# 建立執行環境

環境建立文件統一放在本目錄。第一次使用請先選作業系統，再依需要安裝模型依賴；功能文件中的使用範例與歷史實測紀錄不取代這裡的環境說明。

| 你的環境與用途 | 依循文件 |
|---|---|
| macOS：批次分流、CPU 模型與測試 UI | [macOS](macos.md) |
| Windows：透過 WSL 執行批次分流、CPU 模型與 Docker 驗證 | [Windows／WSL](windows-wsl.md) |
| Windows 原生：問答 Agent、巡檢與 Agent 測試 UI | [Windows 原生 Agent](windows-native-agent.md) |
| macOS／WSL：安裝 TF-IDF、BGE、建立模型與 Docker 映像 | [CPU 模型與 Docker](cpu-models.md) |

## 使用前先確認

- 所有命令都從專案根目錄執行，也就是包含 `predict.py`、`ask.py`、`Solution/` 的目錄。
- 第一部分規則版僅使用標準函式庫，已加入 Windows 原生峰值工作集量測；規則版兩參數入口可在 Windows 執行。TF-IDF／BGE 與 Docker 的既有環境步驟主要使用 macOS／WSL，不代表所有模型與效能實驗已跨平台驗證。
- 第二部分 Agent 已有 Windows 原生安裝腳本。macOS／WSL 尚無同等自動安裝流程，不能直接執行 Windows 的 `agent/setup.py`。
- CPU 分類環境沿用既有 Python 3.12／3.13 安裝紀錄；Agent 文件的最低版本為 Python 3.10。不要直接假設其他 Python 小版本與固定套件版本相容。
- `.venv` 必須在實際執行的作業系統內建立，不能將 macOS、WSL 與 Windows 的虛擬環境互相複製或混用。已有不同平台的 `.venv` 時，請使用另一份專案工作目錄建立環境，不要直接覆蓋。

模型訓練頁另有新增的 CPU／CUDA／Modal 後端，其功能與限制見 [訓練頁與運算環境](../../Solution/docs/訓練頁與運算環境.md)。這份環境索引不代表那些後端已在每個平台重新驗證，也不會自動安裝 CUDA、登入 Modal 或啟動付費工作。

## 準備本機資料

自行將設備表與錯誤碼表放在 `data/devices.json`、`data/error_codes.json`。批次推論與正式問答需要 `data/tickets.jsonl`；模型訓練及 history 開發評測需要 `data/history.jsonl`。原始資料不隨 Git 提供。

安裝套件與下載公開模型需要連網；正式推論使用本機資產。模型、虛擬環境、原始資料與逐筆報告維持 Git 忽略。環境建立不需要開啟 dev；資料使用一律依 [AGENTS.md](../../AGENTS.md)，dev 只供固定版本驗證。

## 啟動之後

- [第一部分操作與架構](../../Solution/README.md)
- [測試 UI 操作](../../helper_tool/README.md)
- [Agent 工具、評測與限制](../../agent/README.md)

Python 虛擬環境的跨平台限制與使用方式可參閱 [Python 官方 venv 文件](https://docs.python.org/3/library/venv.html)。本次為文件整理，沒有重新安裝環境或重跑跨平台效能測試；既有實測結果仍以各驗證文件為準。
