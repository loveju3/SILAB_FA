# 建立環境與執行

三個步驟：取得 repo、放入題目資料、執行 `install.py`。不需要手動建立虛擬環境、下載模型或設定路徑。

## 支援環境

| 作業系統 | 第一部分（批次分流） | 第二部分（Agent） |
|---|---|---|
| macOS Apple Silicon | ✅ | ✅ |
| Windows x64 | ✅ | ✅ |
| Linux x64／arm64 | ✅ | ✅（需 glibc 2.38 以上，例如 Ubuntu 24.04+） |
| Intel Mac | ❌ torch 2.9 沒有 Intel Mac 套件 | — |

- **Python 3.12 或 3.13**。macOS 內建的 `/usr/bin/python3`（3.9）也可以用來啟動 `install.py`，它會自己找 3.12／3.13；都找不到時會告訴你怎麼安裝（若有安裝 [uv](https://docs.astral.sh/uv/)，會自動下載 Python 3.13）。
- 磁碟：約 1.5 GB；加裝 Agent 再約 1.3 GB。只有安裝時需要網路，推論全程離線、只用 CPU。
- Windows 請把下文的 `python3` 換成 `python`（或 `py`）。

## 三個步驟

```bash
# 1. 取得 repo
git clone https://github.com/loveju3/SILAB_FA.git
cd SILAB_FA

# 2. 把題目附的 5 個資料檔放進 data/
#    devices.json、error_codes.json、tickets.jsonl、history.jsonl、dev.jsonl

# 3. 安裝（第二部分 Agent 要加 --with-agent）
python3 install.py --with-agent
```

最後看到 `安裝結果：PASS` 就完成了。中途失敗時，畫面會顯示失敗的階段與原因；修正後重跑同一個指令即可，已完成的部分會沿用。

`install.py` 依序做這些事：

| 階段 | 內容 |
|---|---|
| 建立 `.venv` | 用 Python 3.12／3.13 在 repo 內建立虛擬環境 |
| 安裝套件 | 安裝 `Solution/requirements-embedding.txt` 的固定版本；Windows／Linux x64 從 PyTorch CPU 來源裝 torch，避免下載 GPU 版 |
| 準備模型 | 下載微調模型 ZIP（見下方），校驗 SHA-256 後解壓到 `transfer/finetuned/` |
| 驗證 | 在 `.venv` 中檢查所有套件版本，實際載入模型並推論一筆合成文字 |
| 檢查資料 | 確認 `data/` 的檔案（只看檔案在不在，不讀內容）；缺 `devices.json`／`error_codes.json` 會失敗，缺工單檔只提醒 |
| Agent（`--with-agent`） | 下載 Qwen3-1.7B Q4 GGUF 與 llama.cpp b12366 CPU 版到 `agent_assets/`，校驗後確認 `llama-server` 能執行 |

## 安裝後怎麼跑

**不需要啟用 `.venv`。** 根目錄的 `predict.py`、`ask.py`、`patrol.py`、`run_demo.py`、`evals/run.py` 被其他 Python 啟動時，會自動改用 `.venv` 的 Python 重新執行，所以照題目規格直接打 `python3 ...` 就可以。不想要這個行為時設定 `SILAB_NO_VENV=1`。

```bash
python3 run_demo.py                 # 一鍵：dev 評分、兩份 tickets 預測、Agent 示範
python3 run_demo.py --tests         # 另外跑全部開發測試
python3 run_demo.py --evals         # 另外跑 Agent 評測（每題冷啟動，較慢）
```

個別指令：

| 目的 | 指令 |
|---|---|
| 第一部分預測（預設：BGE 聯合微調） | `python3 predict.py data/tickets.jsonl tickets_predictions.jsonl` |
| dev 評分 | `python3 predict.py data/dev.jsonl dev_pred.jsonl && python3 baseline/evaluate.py dev_pred.jsonl data/dev.jsonl` |
| 2 GB 版本（同一個模型，加上 2 GiB 上限） | `python3 predict.py data/tickets.jsonl tickets_predictions_2gb.jsonl --mem-limit 2g` |
| Docker 2 GiB 驗證（需 Docker Desktop） | `bash scripts/docker_2gb_check.sh` |
| 時間與 2 GiB 量測證據 | `python3 scripts/collect_evidence.py --runs 3 --output evidence/<名稱>` |
| 指定其他方法 | `--method rules`（另有 `tfidf-*`、`bge-lr`、`bge-svc`，需要[完整模型包](../advanced/cpu-models.md)） |
| 問答 Agent | `python3 ask.py "B1 動力室有哪些設備？" --trace` |
| 巡檢 | `python3 patrol.py` |
| Agent 評測 | `python3 evals/run.py --suite synthetic`、`--suite history` |
| 測試 UI | `python3 helper_tool/ui_server.py --port 8765`，瀏覽器開啟終端機顯示的網址 |
| 環境檢查 | `.venv/bin/python scripts/check_environment.py`（Windows：`.venv\Scripts\python.exe`） |

## 模型從哪裡來

預設的微調模型不放在 Git 裡，`install.py` 會從本 repo 的 GitHub Release 下載：

- 檔案：`SILAB_FA_bge_finetuned_20261115-082422.zip`（76 MB，SHA-256 `4e721a87…59aa3f`）
- 網址：`https://github.com/loveju3/SILAB_FA/releases/download/models-20261115/SILAB_FA_bge_finetuned_20261115-082422.zip`
- 內容：微調後的 bge-small-zh-v1.5 encoder、分類頭、tokenizer，以及每個檔案的 SHA-256 清單；不含任何工單資料

無法連到 GitHub 時，手動下載這個 ZIP 放到 repo 根目錄或「下載項目」（`~/Downloads`），`install.py` 會自動找到；放在其他位置用 `python3 install.py --model-zip <路徑>`。

## 常見問題

| 狀況 | 處理 |
|---|---|
| `找不到 Python 3.12 或 3.13` | macOS：`brew install python@3.13` 或到 python.org 安裝；Windows：`winget install Python.Python.3.13`（勾選 Add to PATH）；Linux：`sudo apt install python3.12 python3.12-venv`，或安裝 uv 後重跑 |
| `建立 .venv 失敗`（Debian／Ubuntu） | `sudo apt install python3.12-venv`（或對應版本） |
| `CERTIFICATE_VERIFY_FAILED` | python.org 版 Python 在 macOS 預設沒有 SSL 根憑證。下載已改用系統 `curl`（用作業系統憑證）避開這個問題；若電腦沒有 `curl`，執行「/Applications/Python 3.x/Install Certificates.command」後重跑 |
| 套件安裝看起來停住 | torch 約 100–200 MB，第一次安裝需要幾分鐘 |
| `.venv 已存在但不是 Python 3.12/3.13` | 刪除 `.venv` 資料夾後重跑 |
| `transfer/finetuned 已存在但不完整` | 刪除 `transfer/finetuned` 後重跑 |
| Linux 上 `llama-server 無法啟動 … GLIBC` | 官方 llama.cpp 需要 glibc 2.38 以上；請改用 Ubuntu 24.04+ 或 macOS／Windows 跑第二部分 |
| Windows 讀檔出現 cp950 編碼錯誤 | 在 PowerShell 先執行 `$env:PYTHONUTF8 = "1"`（經由自動切換 `.venv` 執行時已自動設定） |
| 想確認目前狀態 | `python3 install.py --check`：不安裝、不連網，只檢查 `.venv`、模型與資料 |

## 驗證程度（2026-11-16）

- **Linux arm64**（VM、Python 3.13）：從全新 clone 跑完 `install.py` → `run_demo.py --tests`，全部開發測試通過；Agent 的 Qwen 下載在該 VM 被網路政策擋住，llama.cpp Linux 版因 VM 的 glibc 2.35 無法執行，所以第二部分未在 Linux 實跑。
- **macOS Apple Silicon**：以同一組固定套件實測過安裝與模型推論（[紀錄](../advanced/validation-2026-11-16-macos.md)）；`install.py` 的整段流程請以實際執行結果為準。
- **Windows x64**：先前已用同一組固定套件與 Agent 完成兩階段實測（[紀錄](../advanced/validation-2026-11-15.md)）；`install.py` 的 Windows 分支依程式邏輯撰寫，尚未在 Windows 實機執行。

開發者工作（重新訓練、Docker／2 GB 容器、Modal GPU、舊的手動流程）見 [進階文件](../advanced/README.md)。資料使用規範見 [AGENTS.md](../../AGENTS.md)。
