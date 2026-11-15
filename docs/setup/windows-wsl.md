# 進階開發：既有 Windows／WSL 環境

一般 Windows 測試者請使用 [Windows 完整原生環境](windows.md)，同一環境即可跑兩階段，不需要 WSL。本頁保留給既有 WSL 開發、Linux CPU 模型及 Docker／2 GB 驗證；不作一般測試者的必經流程。

## 1. 安裝與開啟 WSL Ubuntu

尚未安裝 WSL 時，以系統管理員身分開啟 PowerShell：

```powershell
wsl --install -d Ubuntu
```

依提示重新啟動電腦，首次開啟 Ubuntu 時建立 Linux 使用者與密碼。已安裝的使用者不需重裝；可在 PowerShell 檢查並啟動指定發行版：

```powershell
wsl --list --verbose
wsl -d Ubuntu
```

舊版 Windows 或安裝異常時，依 [Microsoft 官方 WSL 安裝文件](https://learn.microsoft.com/en-us/windows/wsl/install) 處理。

## 2. 在 Ubuntu 建立 Python 環境

以下全部在 WSL Ubuntu 執行；`/mnt/c/path/to/SILAB_FA` 請換成實際專案路徑。Windows 的 `C:\...` 對應 WSL 的 `/mnt/c/...`。

```bash
cd /mnt/c/path/to/SILAB_FA
python3 --version
sudo apt-get update
sudo apt-get install python3-venv
python3 -m venv .venv
.venv/bin/python --version
```

既有環境使用 WSL Python 3.12.3；固定 ML 套件沿用 Python 3.12／3.13 的安裝紀錄。如果安裝特定 Python 小版本，venv 套件也要與該版本相符，例如 `python3.12-venv`。

既有 `.venv` 若已是 WSL 環境，不需重建。它的 Python 在 `.venv/bin/python`，不能用 Windows 的 `.venv/Scripts/python.exe` 啟動。

## 3. 選擇要執行的功能

方案 A 不需要第三方模型套件，備妥資料後可直接使用兩參數入口：

```bash
.venv/bin/python predict.py data/tickets.jsonl Solution/predictions/tickets_predictions.jsonl
```

需要 TF-IDF、BGE 或 Docker 2 GB 模式時，接著依 [CPU 模型與 Docker](cpu-models.md) 完成安裝、模型建立與映像建置。

## 4. 啟動 UI

```bash
.venv/bin/python helper_tool/ui_server.py --no-browser
```

保留 Ubuntu 終端機，從 Windows 瀏覽器開啟終端機顯示的網址，預設為 `http://127.0.0.1:8765`。以實際埠號為準；Ctrl+C 結束服務。

第二部分 Agent 需要 Linux 原生 `llama-server`；Windows 的 `agent/setup.py` 不支援在 WSL 安裝這個 runtime。不要因共用同一個專案目錄，就把 Windows exe 當成 WSL 原生執行檔。需要現成安裝流程時，另在 Windows PowerShell 依 [Windows 原生 Agent](windows-native-agent.md) 操作。

## 既有實測與疑難排解

[Windows／WSL 2 GB 驗證紀錄](../../Solution/docs/Windows-WSL環境與2GB驗證.md) 保留既有套件版本、容器限制與執行紀錄，供追溯。本頁是環境建立入口；該紀錄不代表本次重新測過所有功能。

若出現 `failed to make path absolute`，先重新 `cd` 到實際專案根目錄再執行，不必刪除已安裝的套件或實驗結果。
