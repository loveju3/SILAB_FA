# 微調 BGE：另一台電腦自行建置 image 與掛載模型

一般使用者請依 [docs/setup](../setup/README.md) 執行 `python3 install.py`，一個指令即可建立環境並匯入本模型 ZIP。以下僅供需要 Docker 移轉的使用者。

## 這個 ZIP 包含什麼

- 檔名：SILAB_FA_bge_finetuned_20261115-082422.zip
- 使用本機最近一次訓練工作 20261115-082422-580651-ea6d1ae8 的 models/finetuned/，不是宣稱分數最高的版本。
- 內含完整 finetuned/、Dockerfile.finetuned、本說明及模型檔案 SHA-256 manifest。
- 不含 history/dev/tickets、訓練資料快照、逐筆預測或訓練報告。這不是 Docker image，也不包含 Python 套件。
- ZIP 由專案擁有者自行上傳；本次僅將本說明上傳雲端。
- 所用共用規則／設備模組有方案 A 既有 dev 影響限制；不能宣稱獨立未觀察 dev 的測試成績。

## 1. 準備另一台電腦

**若這台電腦尚未安裝 Docker，請先安裝 [Docker Desktop](https://www.docker.com/products/docker-desktop/)**，啟動後等待引擎就緒；Windows 使用 Linux containers。模型 ZIP／image 檔本身不會安裝 Docker。

在終端執行 `docker version`，確認同時顯示 Client 與 Server，再繼續建置。若找不到指令，確認安裝完成並重開終端；若只有 Client 或連不上引擎，先啟動 Docker Desktop。建置時需要網路下載 Python 基底映像與 CPU 套件；正式推論禁止外部網路。

取得與目前程式相容的 repo（打包時 HEAD：355943ed5b77e3c799fba0472139f62408267a94；實際工作樹程式雜湊另列 ZIP 的 MANIFEST.json）。若 Git 遠端尚未更新到這版，需另外同步所需程式；只 clone 舊版不保證支援 bge-finetuned。訓練不需重新執行。

Linux amd64 已作本機基本驗證；Apple Silicon 的原生相容性及效能未驗證。需重現本機平台時在 build 和 run 都加 --platform linux/amd64，Apple Silicon 此時是模擬執行。

## 2. 解壓 ZIP，準備資料

建議將 ZIP 解壓至 repo 根目錄下 transfer/，得到：

~~~text
SILAB_FA/
├── predict.py
├── Solution/
├── .dockerignore
├── data/
│   ├── devices.json       ← 另行準備，repo 不包含
│   ├── error_codes.json   ← 另行準備，repo 不包含
│   └── tickets.jsonl      ← 另行準備，只留本機
├── transfer/
│   ├── Dockerfile.finetuned
│   ├── README.md
│   ├── MANIFEST.json
│   └── finetuned/
│       ├── model.json
│       ├── heads.safetensors
│       └── encoder/       ← 完整模型、tokenizer、設定及 encoder_manifest.json
└── results/
~~~

保留 repo 原有 .dockerignore，避免將 JSONL 與報告送入建置 context。可在 repo 根目錄建立 transfer/.dockerignore 不會生效；必須使用根目錄的 .dockerignore。模型在容器執行時掛載，不會 COPY 進這個映像。

不要只搬 encoder/model.safetensors；神經分類頭、metadata、tokenizer 和 manifest 都必須一起保留。

## 3. 從 repo 根目錄 build

~~~bash
docker build -f transfer/Dockerfile.finetuned -t silab-fa:finetuned-runtime-20261115 .
~~~

這個專用 Dockerfile 直接從 Python 基底安裝 CPU 推論套件，不依賴你本機已有 silab-fa:scheme-c，也不要求先訓練 TF-IDF 或原版 BGE 分類頭。

建置需要 data/devices.json 與 data/error_codes.json；不需要 history、dev 或完整工單。Dockerfile 未包含 ask.py、patrol.py、訓練或測試 UI。

## 4. Windows PowerShell 執行

在 repo 根目錄執行：

~~~powershell
New-Item -ItemType Directory -Force results | Out-Null
$modelDir = (Resolve-Path .\transfer\finetuned).Path
$inputDir = (Resolve-Path .\data).Path
$resultDir = (Resolve-Path .\results).Path
docker run --rm --network none --memory 2g --memory-swap 2g --cpus 2 --mount "type=bind,source=$modelDir,target=/model,readonly" --mount "type=bind,source=$inputDir,target=/input,readonly" --mount "type=bind,source=$resultDir,target=/results" silab-fa:finetuned-runtime-20261115 Solution/scripts/check_limits.py --report /results/finetuned_limits.json --run predict.py /input/tickets.jsonl /results/finetuned_predictions.jsonl --method bge-finetuned --model-path /model --metrics /results/finetuned_metrics.json
$LASTEXITCODE
~~~

## 5. Linux／macOS shell 執行

同樣從 repo 根目錄執行：

~~~bash
mkdir -p results
docker run --rm --network none --memory 2g --memory-swap 2g --cpus 2 \
  --mount "type=bind,source=$(pwd)/transfer/finetuned,target=/model,readonly" \
  --mount "type=bind,source=$(pwd)/data,target=/input,readonly" \
  --mount "type=bind,source=$(pwd)/results,target=/results" \
  silab-fa:finetuned-runtime-20261115 \
  Solution/scripts/check_limits.py --report /results/finetuned_limits.json \
  --run predict.py /input/tickets.jsonl /results/finetuned_predictions.jsonl \
  --method bge-finetuned --model-path /model \
  --metrics /results/finetuned_metrics.json
echo $?
~~~

--model-path /model 是容器內掛載路徑，不能填原電腦的 C:\... 路徑。權重可 CPU 推論，接收者不需要 GPU 或重新 fine-tune。

## 6. 預期結果與測試

成功退出碼為 0，results/ 應有：
- finetuned_predictions.jsonl：每筆輸入恰好一筆 JSONL；七欄 ticket_id、device_id、category、urgency、duplicate_of、llm_used、suggestion。
- finetuned_metrics.json：量測與逐筆耗時，只留本機。
- finetuned_limits.json：memory.max 預期 2147483648、memory.swap.max 預期 0、exit_code 預期 0，包含子程序耗時與 cgroup 峰值。

完整 2000 筆 tickets 需自行量測全程序總時間是否 <=480 秒；逐筆記錄 P50、P95、最大值與實際硬體。2 GiB、無 swap 需 Docker Linux cgroup v2。內部 timeout 或幾筆 smoke 不代表完整效能達標。

完整驗收與結果欄位可搭配同資料夾的 SILAB_FA_cross_device_test_checklist.json。原始工單、預測、逐筆量測與 trace 不上傳；雲端只填彙總。dev 只在版本與設定固定後交由驗證器使用，不檢視逐筆結果或據此調整算法。

本次驗證結果：專用 Dockerfile 在本機 Linux amd64 建置成功（套件層使用既有快取，未驗證全新機器下載）；從待移轉資料夾唯讀掛載模型，在 --network none、2 GiB、無 swap、2 CPU 下完成一筆合成工單，exit 0、七欄輸出合法，約 11.88 秒。未跑完整 2000 筆或 dev，未在另一台電腦實測。

## 7. 後續更新

- 只換同格式 finetuned/：傳送新模型資料夾，改 mount 來源即可，不用重建 image。
- 更換工單或輸出：不用重建。
- 修改 repo 的 Python／規則／公共表：重新 build；不一定需要重裝套件，Docker 可利用快取。
- 修改模型格式或套件需求：同步相容程式並重新 build，重新測試。
- 選定模型後再做 dev 驗證，不根據 dev 成績反覆挑模型。


## 本次壓縮檔校驗

檔名：SILAB_FA_bge_finetuned_20261115-082422.zip

大小：76005729 bytes

SHA-256：`4e721a8722f57720ed87c1cf02274ff1b11e3c43f648044b19843e279d59aa3f`

Windows 可用 `Get-FileHash .\SILAB_FA_bge_finetuned_20261115-082422.zip -Algorithm SHA256` 核對。ZIP 內已驗證全部 9 個模型檔案的雜湊及壓縮 CRC。
