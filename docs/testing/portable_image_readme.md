# SILAB_FA 可攜式批次分流映像（2026-11-15）

## 先安裝並啟動 Docker

1. 如果這台電腦還沒有 Docker，先安裝 [Docker Desktop](https://www.docker.com/products/docker-desktop/)。下載 image 檔不會一併安裝 Docker。
2. 啟動 Docker Desktop，等待引擎就緒；Windows 使用 Linux containers。
3. 在終端執行 `docker version`，確認同時顯示 Client 與 Server，再繼續下載／匯入與測試。若顯示找不到指令，先確認安裝完成並重開終端；若只有 Client 或連線引擎失敗，先確認 Docker Desktop 已啟動。

## 包含內容與限制

- 映像標籤：silab-fa:portable-batch-20261115。
- 平台：Linux amd64（x86-64）。Windows 使用 Docker Desktop 的 Linux containers；Apple Silicon 執行此映像需模擬 amd64，效能不可視為原生。
- 包含 Python、CPU 模型套件、BGE encoder、TF-IDF／BGE 分類模型、目前批次程式與公開設備／錯誤碼表。
- 支援 rules（預設）、tfidf-lr、tfidf-svc、bge-lr、bge-svc。
- 不包含原始工單、history/dev、預測、逐筆報告、Git、憑證或 Windows Agent runtime。
- 不包含 ask.py／patrol.py 的可執行環境，也不包含新增的 bge-finetuned 訓練工作產物。這份不是全專案所有功能的映像。
- 封存檔包含所有執行所需映像層；接收者不用安裝本機 Python／ML 套件或另外下載基底映像。
- 仍須先安裝並啟動 Docker，並由測試者在本機另外準備工單資料。
- 沿用方案 A 的規則與設備模組，存在既有 dev 影響限制，不能宣稱未觀察 dev 的獨立開發或測試。

## 下載與校驗

專案擁有者已手動上傳完整映像，請下載單一 silab-fa-portable-batch-20261115.tar.gz（或未壓縮 .tar）即可，不需下載分卷或執行 assemble_image.py。資料夾內若有舊分卷，這次請忽略；合併腳本只適用於完整分卷交付，不是測試腳本。

另下載 manifest.json 與 SHA256SUMS.txt；其中 archive 與校驗檔記錄的是 .tar.gz 的大小與 SHA-256，transfer.parts 是先前分卷資訊，完整檔案不需要用到。至少保留約 5 GB 可用空間供封存檔與 Docker 匯入映像使用。

此封存檔是 docker save 格式，不是可直接解開後執行的應用程式。若檔名有變更，請自行替換以下指令的檔名。

Windows PowerShell：

~~~powershell
Get-FileHash .\silab-fa-portable-batch-20261115.tar.gz -Algorithm SHA256
~~~

Linux：

~~~bash
sha256sum silab-fa-portable-batch-20261115.tar.gz
~~~

macOS 可用 shasum -a 256；確認完整 .tar.gz 的結果與 SHA256SUMS.txt 第一行／manifest.json 的 archive.sha256 相同。舊校驗檔另含分卷項目，未下載分卷時不必逐卷檢查。未壓縮 .tar 的雜湊不同，不要拿 .tar.gz 校驗碼比對。

## 匯入

~~~bash
docker load --input silab-fa-portable-batch-20261115.tar.gz
docker image inspect silab-fa:portable-batch-20261115
~~~

Docker 可直接載入 gzip 壓縮的 docker save 封存檔，不必先手動解壓。匯入後核對 image ID 與 manifest.json。不要改用 docker import，否則會失去映像設定。

若下載的是未壓縮 .tar，改執行 docker load --input silab-fa-portable-batch-20261115.tar；之後的執行指令相同。

## 另一台電腦要怎麼測試

完整的 41 項需求驗收請搭配同資料夾的 [SILAB_FA_cross_device_test_checklist.json](https://drive.google.com/file/d/1qO6xW6S2VFwegVU81Eg_wWRhnY9PB9Xp/view)。本使用說明負責映像操作；JSON 清單列出各項測試目的、命令、預期結果與待填寫的通過狀態。合併腳本／校驗檔只檢查傳輸完整性，不會自動跑完驗收。

建議依下面順序操作：

1. 確認 Docker 使用 Linux containers，執行 docker version，記錄 CPU、RAM、OS、Docker 版本。下載映像並完成上面的校驗與匯入。
2. 自行在本機建立 data/ 與 results/，準備 data/tickets.jsonl；映像未包含 2000 筆工單，只有映像檔不能跑完整資料測試。
3. 先用下方 Windows 或 Linux/macOS 指令測 rules。必須 exit code 0，results 內產生預測、metrics.json 與 limits.json。PowerShell 用 $LASTEXITCODE、Unix shell 用 echo $? 查上一條命令退出碼。
4. 檢查預測每個非空行都是 JSON object，輸入／輸出 ticket_id 一對一、不重複，筆數相同。七欄必須包含 ticket_id、device_id、category、urgency、duplicate_of、llm_used、suggestion；llm_used 是布林，urgency 為 low/normal/high。完整型別規則見 JSON 清單 BATCH-01/02。
5. 分別改 --method 為 tfidf-lr、tfidf-svc、bge-lr、bge-svc。每次使用不同結果資料夾／檔名。核對 limits.json 的 memory.max=2147483648、memory.swap.max=0、exit_code=0；確認以 --network none 執行。對每次執行做外部全程序計時，完整 2000 筆須 <=480 秒，記錄逐筆 P50/P95/最大值及峰值記憶體。不要把內部三筆 smoke 數字當成這台機器的完整成績。
6. 在 JSON 清單對應項目的 result 填 status、passed、實際命令、耗時、記憶體、觀察行為與問題；未測項目保留 not_run，缺少環境或隱藏題標 blocked 並說明。彙總可放雲端，原始資料／預測／逐筆報告留本機。

這份映像可直接測第一部分批次入口與 2 GB 限制。第二部分 ask.py／patrol.py、自製 Agent Evals、README／交付文件與隱藏題驗收不會隨 docker load 自動完成，須另外取得專案程式與相應環境。dev 評分也需在固定版本後依清單執行；映像未包含 dev 或 baseline/evaluate.py。沒有公開最低準確率門檻，不要自行指定一個分數當作通過條件。

## Windows PowerShell：2 GiB、無 swap、離線執行

在你的工作資料夾下準備 data/tickets.jsonl，並建立 results。以下使用目前資料夾的絕對路徑，路徑含空白時保留引號：

~~~powershell
New-Item -ItemType Directory -Force results | Out-Null
$inputDir = (Resolve-Path .\data).Path
$resultDir = (Resolve-Path .\results).Path
docker run --rm --network none --memory 2g --memory-swap 2g --cpus 2 --mount "type=bind,source=$inputDir,target=/input,readonly" --mount "type=bind,source=$resultDir,target=/results" silab-fa:portable-batch-20261115 Solution/scripts/check_limits.py --report /results/limits.json --run predict.py /input/tickets.jsonl /results/tickets_predictions.jsonl --method rules --metrics /results/metrics.json
~~~

## Linux／macOS shell：相同限制

在目前資料夾下準備 data/tickets.jsonl：

~~~bash
mkdir -p results
docker run --rm --network none --memory 2g --memory-swap 2g --cpus 2 \
  --mount "type=bind,source=$(pwd)/data,target=/input,readonly" \
  --mount "type=bind,source=$(pwd)/results,target=/results" \
  silab-fa:portable-batch-20261115 \
  Solution/scripts/check_limits.py --report /results/limits.json \
  --run predict.py /input/tickets.jsonl /results/tickets_predictions.jsonl \
  --method rules --metrics /results/metrics.json
~~~

其他方法：將 --method rules 換成 tfidf-lr、tfidf-svc、bge-lr 或 bge-svc。分別指定不同輸出檔名，避免覆寫要保留的結果。

若不需要硬限制檢查，可把映像後面的命令換成：
predict.py /input/tickets.jsonl /results/tickets_predictions.jsonl
這仍保留題目要求的兩位置參數入口。不要自行加未實作的 --mem-limit。

## 預期結果與資料保護

- 成功 exit code 0；每筆輸入對應一筆 UTF-8 JSONL 七欄預測。
- results/limits.json 保存 cgroup 限制、子程序退出碼、耗時與峰值。
- results/metrics.json 含逐筆耗時與量測範圍。兩者數值不能代替外部完整程序計時。
- 本映像的 2g 為 2147483648 bytes（2 GiB）；swap.max 預期 0，需要 Linux cgroup v2。
- --network none 禁止推論連網；不需要在推論時拉取基底映像或模型。
- 完整 tickets 預算仍為 480 秒／2000 筆；另一台電腦須自行測量。
- 目前只用三筆合成工單測過五方法的執行與輸出格式，未在本次打包重新量完整 tickets／dev 準確率，也未驗證另一台電腦或 Apple Silicon。
- 原始資料、預測、逐筆 metrics／trace 與錯題只留本機。雲端清單只填彙總，不上傳這些檔案。
- dev 只在候選版本與設定固定後由驗證器讀取；不得以逐筆內容或分數反覆調整算法。

## 改 code 後需要重包嗎？

目前這份是「程式、設定與模型都包在映像內」的交付方式。修改本機檔案不會自動改變既有映像或已上傳的封存檔。

| 修改內容 | 是否要重建／重包 |
|---|---|
| predict.py、Solution/predict.py、Solution/triage/ | 正式交付新版時要重建、重新匯出與上傳 |
| Solution/config/ 規則／別名、映像內 devices.json／error_codes.json | 正式交付新版時要；這些也是算法／資料版本的一部分 |
| TF-IDF／BGE 模型或 encoder 權重 | 要重新建立含新資產的映像，再匯出與上傳；目前 portable Dockerfile 只更新四個分類頭，不會更新基底映像的 encoder |
| Python 套件／版本 | 要先重建相應的 CPU 依賴映像，再建置 portable 映像 |
| 外掛 data/tickets.jsonl、輸出路徑、--method 等參數 | 不用重包；重新執行並記錄輸入／參數即可 |
| README 或本 JSON 測試清單 | 不影響映像執行，更新說明檔即可 |
| ask.py／patrol.py | 這份映像沒有這些功能；修改不會更新它們的執行環境 |

### 開發時只改 Python：可以掛載，不必每次重包

若另一台電腦也持有最新程式碼，可把必要程式路徑唯讀掛入容器，沿用映像中的 Python 套件與模型。以下 PowerShell 範例從專案根目錄執行；先備妥 data/tickets.jsonl 及 results 資料夾：

~~~powershell
$projectDir = (Get-Location).Path
$inputDir = (Resolve-Path .\data).Path
$resultDir = (Resolve-Path .\results).Path
docker run --rm --network none --memory 2g --memory-swap 2g --cpus 2 --mount "type=bind,source=$projectDir\predict.py,target=/app/predict.py,readonly" --mount "type=bind,source=$projectDir\Solution\predict.py,target=/app/Solution/predict.py,readonly" --mount "type=bind,source=$projectDir\Solution\triage,target=/app/Solution/triage,readonly" --mount "type=bind,source=$projectDir\Solution\config,target=/app/Solution/config,readonly" --mount "type=bind,source=$inputDir,target=/input,readonly" --mount "type=bind,source=$resultDir,target=/results" silab-fa:portable-batch-20261115 predict.py /input/tickets.jsonl /results/tickets_predictions.jsonl --method rules
~~~

這只替換列出的程式與設定；不替換套件、模型或映像內公共表。如果修改新增依賴、改模型路徑或與舊模型格式不相容，仍需重建。不要把整個專案掛到 /app，否則會遮住映像內模型。正式重現報告需同時記錄映像 ID、掛載原始碼／設定雜湊，不能只寫映像版本。

### 正式交付新版的流程

1. 若只改 portable Dockerfile 已 COPY 的程式／設定／分類頭，可在保留 silab-fa:scheme-c 基底映像的建置機重新 build。若 encoder／套件有變，先依 docs/advanced/cpu-models.md 重建 scheme-c；其他新增路徑也要更新 Dockerfile 的 COPY 清單。
2. 使用新標籤，例如 silab-fa:portable-batch-20261115-v2，避免與舊版混淆。重跑合成 smoke、離線與所需資源限制測試。
3. docker save 匯出新標籤；視需要壓縮，產生本次檔案的 SHA-256 與 manifest，不沿用舊校驗碼。正常交付單檔即可，不必分卷。
4. 上傳到新的版本子資料夾，保留舊版；接收者重新下載、校驗並 docker load。

~~~bash
docker build --pull=false -f docs/testing/Dockerfile.portable -t silab-fa:portable-batch-20261115-v2 .
docker save --output silab-fa-portable-batch-20261115-v2.tar silab-fa:portable-batch-20261115-v2
~~~

Docker 可重用未變的建置快取，所以改 Python 通常不用重裝全部套件；目前以封存檔交付時，仍須提供新版完整封存。以上掛載與新版示範指令尚未在另一台電腦實測。

## 封存來源

由既有 silab-fa:scheme-c 的 CPU 依賴層加上目前批次程式建置。docs/testing/Dockerfile.portable 用於這次本機建置；接收者直接 docker load 即可，不需要原本的 scheme-c 映像。

詳細 image ID、來源 Git revision、打包檔案雜湊、依賴與基本測試結果見 manifest.json。
