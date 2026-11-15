# 進階開發：CPU 模型訓練與 Docker

一般測試者先依 [Windows](windows.md) 或 [Mac](macos.md) 完成一套環境；下列重新訓練與 2 GB Docker 為開發者進階工作。

**本頁 Docker 指令的前置條件：**若電腦尚未安裝 Docker，先安裝並啟動 [Docker Desktop](https://www.docker.com/products/docker-desktop/)，執行 `docker version` 確認有 Client 與 Server 後，才能執行下方任何 `docker build`／`docker run`。Windows 使用 Linux containers；從 WSL 執行時另啟用該發行版的 WSL integration。

本頁沿用既有 Unix 開發環境 `.venv`，Windows／WSL 可依 [既有 WSL 路線](windows-wsl.md) 建立；若已依 [Mac 完整流程](macos.md) 建好 `.venv-macos`，將以下 `.venv/bin/python` 換成 `.venv-macos/bin/python` 即可，不需要為一般測試再建一份環境。資料準備見 [完整流程索引](README.md)。命令從 repo 根目錄執行；以下 Unix 路徑不適用 Windows PowerShell。

## TF-IDF（方案 B）

```bash
.venv/bin/python -m pip install -r Solution/requirements-ml.txt
.venv/bin/python Solution/train.py
```

訓練來源為 history。模型資產留在本機，推論時不自動訓練或改選其他方法。

完成後預設模型位於 `Solution/models/tfidf-lr.joblib`、`tfidf-svc.joblib`，各自須有同名 `.json` metadata。若改用另一台電腦已訓練的可信模型，將這兩組檔案放到相同路徑即可，不必重新訓練；只放 `.joblib` 不完整。

## BGE（方案 C）

WSL／Linux 的 CPU 環境先安裝固定 CPU PyTorch，再裝其他固定依賴：

```bash
.venv/bin/python -m pip install --no-compile torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install --no-compile -r Solution/requirements-embedding.txt
```

macOS 使用專案的固定 embedding 依賴，不使用上面的 Linux CPU wheel index：

```bash
.venv/bin/python -m pip install -r Solution/requirements-embedding.txt
```

接著在各自環境下載公開 encoder、以 history 建立分類頭並檢查依賴：

```bash
.venv/bin/python Solution/scripts/download_encoder.py
.venv/bin/python Solution/train_embedding.py
.venv/bin/python -m pip check
```

`--no-compile` 可減少 Windows 掛載磁碟的 `.pyc` 寫入；不是關閉推論或略過模型安裝。下載只發生在準備階段。模型的訓練／驗證分數與正式推論資源測試分開看待。

公開 encoder 預設下載到 `Solution/models/bge-small-zh-v1.5/`，內含權重、tokenizer、設定及 `encoder_manifest.json`。訓練後須有 `Solution/models/bge-lr.joblib`、`bge-svc.joblib` 與各自同名 `.json`。若搬入可信模型，分類頭、metadata 與同層 encoder 資料夾須一起搬入，不能用重新下載的任意 revision 取代配套 encoder。

## 已有微調 BGE：在建立環境時放好模型

微調版不是上述 `bge-lr`／`bge-svc`。取得維護者提供的 `SILAB_FA_bge_finetuned_20261115-082422.zip`，解壓至 repo 根目錄 `transfer/`；確認 `transfer/finetuned/` 下有 `model.json`、`heads.safetensors`、完整 `encoder/`。ZIP 本身不含 Python 套件，需完成前述 BGE CPU 依賴安裝；已有微調 checkpoint 時不用再跑 `train_embedding.py` 或重新 fine-tune。

本機 CPU 推論（macOS／WSL）：

```bash
.venv/bin/python predict.py data/tickets.jsonl reports/finetuned_predictions.jsonl --method bge-finetuned --model-path transfer/finetuned
```

若只用 Docker 執行微調版，不必先安裝本機 ML 套件／建立其他分類模型。準備公共表與 ZIP 後，可直接使用 repo 的專用 Dockerfile：

```bash
docker build -f docs/testing/Dockerfile.finetuned -t silab-fa:finetuned-runtime-20261115 .
```

此映像不包含微調權重；執行時須將 `transfer/finetuned/` 唯讀掛到 `/model`，並指定 `--method bge-finetuned --model-path /model`。完整 Windows／Unix 掛載與 2 GiB 指令見 [微調模型移轉說明](../testing/finetuned_transfer_readme.md)。`run_budget.py`、`benchmark.py` 目前不接受 `bge-finetuned`，微調版使用該說明的直接容器命令及外部計時，不要套用下方五種方法 runner。

完成對應依賴後，可執行既有開發測試入口；它預設關閉 dev 驗證：

```bash
.venv/bin/python Solution/scripts/test_development.py
```

新增訓練後端的操作與額外依賴見 [訓練頁與運算環境](../../Solution/docs/訓練頁與運算環境.md)，不要用本頁的 CPU 安裝步驟當成 CUDA／Modal 環境已就緒的證明。

## Docker 2 GB 推論（選用）

先啟動 Docker Desktop。Windows 需啟用 Ubuntu 的 WSL integration，並在 WSL 執行；macOS 在自己的終端機執行。確認 `docker version` 顯示 Server 後，建置要使用的方法：

```bash
docker build -f Solution/Dockerfile -t silab-fa:scheme-a .
docker build -f Solution/Dockerfile.ml -t silab-fa:scheme-b .
docker build -f Solution/Dockerfile.embedding -t silab-fa:scheme-c .
```

B／C 映像需要先完成相應的本機模型建立；只使用方案 A 不必建置其餘兩個映像。以下範例用規則版跑一次硬限制驗證，預設檢查 history 與 tickets：

```bash
.venv/bin/python Solution/scripts/run_budget.py --method rules --runs 1 --probe
```

其他方法使用 `--method tfidf-lr`、`tfidf-svc`、`bge-lr` 或 `bge-svc`。限制為 2 GiB、禁止 swap、2 CPU、推論無網路且不自動拉映像；獨立 `--probe` 預期超限退出 137，不是正式推論失敗。

輸入與輸出需要由 runner／UI 掛載，不依賴映像包含原始工單。完整量測定義與歷史數字見 [budget.md](../../Solution/budget.md)，不把訓練或安裝環境視為已通過相同的記憶體限制。
