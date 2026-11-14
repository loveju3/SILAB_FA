# CPU 模型與 Docker

先依 [macOS](macos.md) 或 [Windows／WSL](windows-wsl.md) 在對應作業系統建立 `.venv`，並準備 [本機資料](README.md#準備本機資料)。以下命令從專案根目錄執行，使用 Unix 的 `.venv/bin/python`；不適用 Windows PowerShell 的原生 Agent 環境。

## TF-IDF（方案 B）

```bash
.venv/bin/python -m pip install -r Solution/requirements-ml.txt
.venv/bin/python Solution/train.py
```

訓練來源為 history。模型資產留在本機，推論時不自動訓練或改選其他方法。

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
