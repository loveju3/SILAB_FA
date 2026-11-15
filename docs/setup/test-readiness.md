# 另一台電腦：完整測試環境準備清單

這份清單是建立環境的最後一步。先準備後續所需的資料、模型、runtime 與映像，再開始測試。它不是一鍵安裝器，也不代表準備完成就已通過驗收。所有路徑均相對於 repo 根目錄。

## 1. 選擇執行路線

- **Windows 全功能驗收**：第一部分依 [Windows／WSL](windows-wsl.md) 建立 Linux `.venv`；第二部分依 [Windows 原生 Agent](windows-native-agent.md) 使用 Windows Python 與 Windows llama.cpp。兩邊各用自己的解譯器，不能混用 `.venv`。
- **macOS／Linux**：批次依 [macOS](macos.md)／[CPU 模型](cpu-models.md)；Agent 需自行取得與 GGUF 相容的原生 CPU llama.cpp。現有 `agent/setup.py` 只支援 Windows，尚無所有平台自動裝完全部功能的流程。
- **只驗第一部分微調版 Docker**：取得 repo、公共表、微調 ZIP，安裝 Docker 即可；不必先建立本機 BGE／TF-IDF 分類環境。直接走步驟 4 的專用映像。
- **只跑既有可攜 image**：可不下載 repo，但只涵蓋映像內的批次功能；不能因此執行 repo 的全部測試、Agent 或訓練。見 [映像使用說明](../testing/portable_image_readme.md)。

取得與模型相容的程式版本；只更新模型、不更新不支援其格式的舊 repo，仍會載入失敗。

## 2. 在本機放好資料

| 路徑 | 後續用途 |
|---|---|
| `data/devices.json`、`data/error_codes.json` | 批次、Agent、Docker 建置與合成問答評測都需要的公共表 |
| `data/tickets.jsonl` | 正式批次、問答、完整巡檢與效能測試 |
| `data/history.jsonl` | 建立 B/C 模型、history 開發評測、預設 budget runner |
| `data/dev.jsonl` | 僅供版本／設定固定後驗證；不屬於日常開發環境檢查 |

資料不隨 Git 提供，需由維護者另行在本機傳送。原始工單、預測 JSONL、逐筆報告不得上傳雲端或加入 Git。建立環境不需要開啟、抽樣或分析 dev，也不得用改名副本繞過限制。

建立一個可寫的本機 `reports/` 或 `results/` 目錄保存測試結果；先確認不會覆蓋需要保留的舊實驗。

## 3. 在建置階段完成模型下載／放置

| 功能 | 取得方式 | 正確位置／必要檔案 |
|---|---|---|
| A 規則 | 無需模型下載 | repo 的 `Solution/config/` 加公共表 |
| B TF-IDF | 安裝 ML 依賴後用 history 執行 `Solution/train.py`，或搬入可信且版本相容的模型 | `Solution/models/tfidf-lr.joblib`、`tfidf-svc.joblib` 與各自同名 `.json` |
| C 固定 BGE | 執行 `Solution/scripts/download_encoder.py` 下載固定 revision，再以 history 執行 `Solution/train_embedding.py`；或搬入整套配套資產 | `Solution/models/bge-small-zh-v1.5/`（完整權重、tokenizer、設定與 `encoder_manifest.json`）及同層 `bge-lr.joblib`、`bge-svc.joblib` 與同名 `.json` |
| 微調 BGE | 向維護者取得 `SILAB_FA_bge_finetuned_20261115-082422.zip`，解壓至 `transfer/`；無須重新訓練 | `transfer/finetuned/model.json`、`heads.safetensors`、完整 `encoder/`；不能只放單一權重檔 |
| 問答／巡檢 Qwen | Windows 原生執行 `python agent/setup.py`，自動下載與驗證；其他平台自行準備同版 GGUF 與原生 runtime | `agent_assets/Qwen_Qwen3-1.7B-Q4_K_M.gguf`、`agent_assets/runtime/llama-server.exe`（Windows）與全部配套 DLL、`agent_assets/manifest.json` |

微調 ZIP 由維護者自行上傳至 [共用雲端資料夾](https://drive.google.com/drive/folders/1QeCQjNet73n8FYnrJ4WLmnpGKPnNc7zt)，若找不到請索取；Git clone 與推論都不會自動下載這份私人訓練產物。ZIP 的 `MANIFEST.json` 是移轉檔案清單；`finetuned/model.json` 是模型格式 metadata，兩者不要互換。

### macOS／WSL：本機 B/C 套件與資產

先完成平台 `.venv`。以下指令從 repo 根目錄執行；Windows 原生 PowerShell 不使用 `.venv/bin/python`。

```bash
.venv/bin/python -m pip install -r Solution/requirements-ml.txt
.venv/bin/python Solution/train.py
```

WSL／Linux CPU BGE 先裝 CPU torch，再安裝固定依賴：

```bash
.venv/bin/python -m pip install --no-compile torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install --no-compile -r Solution/requirements-embedding.txt
.venv/bin/python Solution/scripts/download_encoder.py
.venv/bin/python Solution/train_embedding.py
.venv/bin/python -m pip check
```

macOS 使用 `pip install -r Solution/requirements-embedding.txt`，不使用 Linux CPU wheel index。已搬入配套 B/C 模型時可略過對應訓練／下載，仍須安裝套件。只測微調模型時，安裝 embedding 依賴並放好微調 ZIP 即可，不必額外下載原版 encoder、訓練 B/C 或重新 fine-tune。

### Windows 原生：Agent 資產

```powershell
python agent/setup.py
```

完成後模型與 runtime 已在上述固定位置，保留整個 runtime 的配套 DLL。`evals/run.py` 使用固定 GGUF 路徑，不能只把 GGUF 放到任意下載資料夾。macOS／WSL 的 `ask.py`／`patrol.py` 可用 `--server` 指定原生執行檔；Evals runner 也支援 `--server`，GGUF 仍須放上述固定位置。取得原生 runtime 的程序與效能尚需目標平台驗證。

## 4. 在建置階段備妥 Docker 映像

先啟動 Docker；`docker version` 須同時顯示 Client 與 Server。Windows 原生使用 Linux containers；WSL 使用 Docker Desktop 的發行版 integration。建置／下載階段可連網，之後用 `--network none` 做離線推論。

| 要測的方法 | 建置命令／條件 |
|---|---|
| rules | `docker build -f Solution/Dockerfile -t silab-fa:scheme-a .`；需公共表 |
| tfidf-lr／tfidf-svc | `docker build -f Solution/Dockerfile.ml -t silab-fa:scheme-b .`；先備齊 B 模型與 metadata |
| bge-lr／bge-svc | `docker build -f Solution/Dockerfile.embedding -t silab-fa:scheme-c .`；先備齊 C 分類頭、metadata 及 encoder |
| bge-finetuned | `docker build -f docs/testing/Dockerfile.finetuned -t silab-fa:finetuned-runtime-20261115 .`；只需 repo、公共表、Docker，權重執行時掛載 |

只測一種方法不必建立所有映像；要驗全部方法則建立所有所需映像。專用微調 Dockerfile 不依賴 scheme-c，但舊 `Dockerfile.portable` 依賴它。若使用 ZIP 內的配套 Dockerfile，可將上表 `-f` 改成 `transfer/Dockerfile.finetuned`。

微調執行時將 `transfer/finetuned/` 唯讀掛至 `/model`，指定 `--method bge-finetuned --model-path /model`；完整資料／結果掛載與 2 GiB 指令見 [微調模型移轉](../testing/finetuned_transfer_readme.md)。微調模型不會自動放進映像，不能省略 mount。

## 測試前逐項確認

先確認下列各列的必要資產與環境都已準備，再跑對應測試。單元測試不等於真實模型／完整資料評測。

| 測試入口 | 測試前必須備妥 |
|---|---|
| `python predict.py data/tickets.jsonl reports/rules.jsonl` | 公共表、tickets、Python；預設 rules，無第三方模型依賴 |
| 同入口加 `--method tfidf-lr`／`tfidf-svc` | 同上，加 ML 套件與對應 joblib／metadata |
| 同入口加 `--method bge-lr`／`bge-svc` | 同上，加 embedding 套件、分類頭與配套 encoder |
| 同入口加 `--method bge-finetuned --model-path transfer/finetuned` | 公共表、tickets、embedding 套件、完整微調 checkpoint；或專用 Docker 映像與 mount |
| `Solution/scripts/test_development.py` | 在依用途準備的批次環境執行；此入口封鎖 dev 並預設停用 dev 驗證，部分 opt-in 測試可能跳過，需如實記錄 |
| `Solution/scripts/benchmark.py --method rules --runs 3` | 對應本機套件／模型、tickets、可寫輸出目錄；腳本目前只支援五種 A/B/C 方法，不接受 bge-finetuned |
| `Solution/scripts/run_budget.py --method rules --runs 1 --probe` | Docker 及對應映像、公共表、history、tickets；預設評 history，需 cgroup v2 的 2 GiB／無 swap 硬限制 |
| 微調版 2 GiB／效能 | 依移轉文件直接執行容器與外部計時；現有 run_budget.py 不接受 bge-finetuned |
| `python -m unittest discover -s agent/tests -v` | Agent 程式與測試依賴；scripted model 單元測試不能代表真實 LLM 通過 |
| `python ask.py "B1 動力室有哪些設備？" --trace` | 公共表、正式 tickets、Qwen GGUF、對應平台完整 CPU runtime |
| `python patrol.py` | 同上；完整巡檢用正式 tickets，不用合成 smoke 代替 |
| `python evals/run.py --suite synthetic` | 公共表、Git checkout／git 指令、Qwen 與 runtime；工單 fixture 由 runner 產生，不必提供 tickets |
| `python evals/run.py --suite history` | 同上，加本機 history |
| `python evals/run_patrol.py --dataset synthetic`／`tickets` | 公共表、Git、Qwen/runtime；tickets 模式另需正式 tickets |
| `python helper_tool/ui_server.py --no-browser` | 對應功能所需環境與資產；WSL 批次 UI 與 Windows Agent UI 需分別使用各自的 Python/runtime |
| `baseline/evaluate.py <預測> data/dev.jsonl` | 先固定候選版本、設定及來源；由驗證程序讀 dev，只回報彙總；原始 stdout 含範例，留本機，不用來開發 |

上表 `python` 表示該平台已準備的解譯器；macOS／WSL 批次請換成 `.venv/bin/python`，Windows Agent 使用 Windows Python。若只完成某些功能的環境，未備妥的測試應標 blocked／not_run，不得標通過。

最後記錄 OS、CPU、實體 RAM、Python／套件版本、Git revision／工作樹差異、模型與資料雜湊。完成建置後再量完整 2000 筆 480 秒、單題 30 秒、巡檢 120 秒、峰值記憶體與正確率；本文件沒有替其他電腦宣告通過。完整測試目的、預期輸出與結果欄位另見 [跨電腦 JSON 清單](../testing/cross_device_test_checklist.json)。
