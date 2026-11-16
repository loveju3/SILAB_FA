# Windows／WSL 執行與 2 GB 驗證紀錄

新的環境建立步驟統一依循 [docs/setup/](../../docs/setup/README.md)，Windows 批次分流請讀 [WSL 指南](../../docs/advanced/windows-wsl.md)。以下保留先前安裝與驗證時的命令、版本及結果，供追溯，不取代統一環境指南。

第一部分批次入口使用 Unix 的 `resource`，Windows 請使用 WSL；第二部分 Agent 已有 [Windows 原生安裝方式](../../docs/advanced/windows-native-agent.md)。專案內既有 `.venv` 是 Linux 虛擬環境，不能用 Windows Python 啟動。

## 環境準備

在 WSL Ubuntu 執行：

```bash
cd /mnt/c/Users/lovej/Desktop/SILAB_FA
# 首次安裝；若 python3 -m venv 已可用則不需執行。
sudo apt-get install python3.12-venv
python3 -m venv .venv
.venv/bin/python -m pip install --no-compile -r Solution/requirements-ml.txt
.venv/bin/python -m pip install --no-compile torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install --no-compile -r Solution/requirements-embedding.txt
.venv/bin/python Solution/scripts/download_encoder.py
.venv/bin/python -m pip check
.venv/bin/python Solution/scripts/test_development.py
```

Ubuntu 的 Python 小版本可能不同，請安裝與 `python3 --version` 相符的 venv 套件。套件及公開 encoder 的下載需要連網，推論只載入本機資產。

`--no-compile` 略過安裝時預先產生 `.pyc`，減少 Windows 掛載磁碟的大量小檔案寫入；不影響 Python 執行套件，第一次匯入仍可能較慢。

`test_development.py` 關閉 dev 驗證並攔截測試程序直接開啟 `dev.jsonl`；仍須遵守 [AGENTS.md](../../AGENTS.md)，不使用 dev 的副本或逐筆衍生資料。現有 A 與沿用模組的來源限制持續適用。

## 啟動 UI

```bash
.venv/bin/python helper_tool/ui_server.py --no-browser
```

保留終端機，從 Windows 瀏覽器開啟 `http://127.0.0.1:8765`。按 Ctrl+C 結束。若其他埠號正在使用，以終端機實際顯示的網址為準。UI 會對不同連線使用獨立執行緒，應用操作仍循序執行，避免官方評分器的 stdout 擷取互相干擾。

## Docker 2 GiB 推論驗證

先啟動 Docker Desktop 並啟用 Ubuntu 的 WSL integration，確認 `docker version` 能取得 Server 資訊。

```bash
# 模型訓練只使用 history，沿用既有固定比較流程。
.venv/bin/python Solution/train.py
.venv/bin/python Solution/train_embedding.py
docker build -f Solution/Dockerfile -t silab-fa:scheme-a .
docker build -f Solution/Dockerfile.ml -t silab-fa:scheme-b .
docker build -f Solution/Dockerfile.embedding -t silab-fa:scheme-c .

.venv/bin/python Solution/scripts/run_budget.py --method rules --runs 1 --probe
.venv/bin/python Solution/scripts/run_budget.py --method tfidf-lr --runs 1
.venv/bin/python Solution/scripts/run_budget.py --method tfidf-svc --runs 1
.venv/bin/python Solution/scripts/run_budget.py --method bge-lr --runs 1
.venv/bin/python Solution/scripts/run_budget.py --method bge-svc --runs 1
```

上述程式預設只用 history 與 tickets 檢查可執行性，不指定 dev。每次建立獨立容器，設定 2 GiB RAM、禁止 swap、2 CPU、關閉網路與禁止自動拉映像；同時保存 Docker 設定及 cgroup 限制。`--probe` 刻意配置超限記憶體，預期 exit 137 且 `OOMKilled=true`。訓練在本機執行，沒有 2 GiB 限制；不能把推論結果解讀成訓練也通過相同限制。

UI 的 Docker 流程使用 WSL 使用者的 UID／GID 執行容器，讓 UI 可以讀取容器原子寫入的私有預測檔，避免 Linux 暫存目錄出現 root 擁有檔案而讀取失敗。

本機測試曾在 Docker 操作後遇到舊 shell 工作目錄失效，Python 顯示 `failed to make path absolute` 或 `error evaluating path`。請先重新執行 `cd /mnt/c/Users/lovej/Desktop/SILAB_FA`，再重跑該指令；已完成的安裝或測試紀錄不需刪除。

執行紀錄存於 `Solution/experiments/runs/budget-*/`，預測與逐筆資料只留本機。`*-limits.json` 記錄實際限制、記憶體峰值及退出碼，`docker_runs.json` 記錄容器狀態。history 是模型訓練來源，其回測不可當作獨立測試成績。

Docker 建置只複製設備／錯誤碼表；`.dockerignore` 排除所有 JSONL 與 UI 訓練工作目錄。單獨啟動映像時需掛載輸入檔與輸出資料夾，不能期待映像內含原始工單。既有 `run_budget.py` 與 UI 的 Docker 執行流程會提供掛載。

## 2026-11-14 本機容器實測

Windows＋WSL Ubuntu，Docker Desktop Linux amd64，容器 Python 3.13；每種方法各跑一次 history 與 tickets。下列是兩份資料執行的 cgroup 記憶體峰值最大值，包含容器記憶體帳務，並非單一 Python 程序 RSS。測試與套件安裝有同時執行，僅確認可執行性，不作效能排名。

| 方法 | history／tickets | 峰值最大值（MiB） | 本機紀錄目錄（Solution/experiments/runs/ 下） |
|---|---|---:|---|
| rules | 通過／通過 | 50.3 | budget-rules-f15591c3fc1c |
| tfidf-lr | 通過／通過 | 89.0 | budget-tfidf-lr-907a19aee606 |
| tfidf-svc | 通過／通過 | 87.4 | budget-tfidf-svc-d23976b9a51e |
| bge-lr | 通過／通過 | 506.8 | budget-bge-lr-2b6cd78aa1e0 |
| bge-svc | 通過／通過 | 429.7 | budget-bge-svc-52017bd8f704 |

全部確認 `memory.max=2147483648`、`memory.swap.max=0`、2 CPU、`network=none`、退出碼 0。獨立超限探測取得退出碼 137 與 `OOMKilled=true`。UI API 另外完成 history 的原生 TF-IDF SVC、原生 BGE SVC，以及 2 GiB 下 A、TF-IDF SVC、BGE SVC 的預測、評分與保存。未執行 dev 驗證，也未執行完整 BGE 聯合微調訓練的資源測試。

專案 `.venv` 使用 WSL Python 3.12.3，安裝固定 ML／embedding 依賴與 CPU PyTorch 2.9.0+cpu，`pip check` 通過。完整開發測試共 63 項，61 項通過，只略過 2 項需要明確啟用的 dev 驗證，耗時 66.269 秒；包含真實 BGE 離線載入、模型保存／載入、神經訓練契約與背景工作測試。未啟用 `SILAB_VALIDATE_DEV`。

本機追溯資料位於 `Solution/reports/environment/`：`development-tests.txt`、`installed-requirements.txt` 與 `docker-validation.json`。這些產物依 Git 忽略規則保留本機。
