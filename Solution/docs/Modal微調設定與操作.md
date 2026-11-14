# Modal 微調設定與操作

本專案使用 Modal 的 GPU 執行既有 **BGE-small-zh-v1.5 encoder＋線性神經分類頭聯合微調**。固定 BGE＋LR／SVC、固定 BGE＋神經頭會一併作為對照，使用相同 history 切分與既有評估指標。這不是在 Modal 網頁上手動建立一個模型 Endpoint；程式會透過 SDK 準備 GPU 與環境。

## 1. 在 Modal 帳號準備什麼

1. 到 [Modal](https://modal.com/) 註冊／登入，確認使用的 Workspace。
2. 在 Workspace 的 **Usage & Billing** 確認付款方式與可用額度。Modal 官方目前要求 GPU 使用者具備有效付款方式；不要假定有免費額度就不需設定。[GPU 官方文件](https://modal.com/docs/guide/gpu)
3. 依你的預算設定 usage budget／spend limit；這些是帳號層級設定，不是下方 JSON 的 `timeout_seconds`。頁面名稱與規則依 [Modal Budgets 文件](https://modal.com/docs/guide/budgets) 為準。

不需要先建立 Secret、Volume、Web Endpoint 或手動上傳整個專案。帳號登入、付款資料及預算由你在 Modal 完成；專案設定檔不存 token 或信用卡。

## 2. 使用與訓練 UI 相同的 Python

目前這台 Windows 已建立隔離訓練環境。開啟 PowerShell，執行：

```powershell
Set-Location 'C:\Users\lovej\Desktop\SILAB_FA'
$trainingPython = '.\helper_tool\reports\training-runtime\Scripts\python.exe'
& $trainingPython -m pip install -r Solution/requirements-modal.txt
& $trainingPython -m modal --version
```

`requirements-modal.txt` 固定 Modal SDK 1.6.4。此機先前已安裝，重跑安裝只會確認版本；本機可以使用 CPU 版 PyTorch，雲端 GPU 的依賴由 Modal Image 另外安裝，不需要為了 Modal 在本機安裝 CUDA。

新電腦若沒有上述環境，先建立，再安裝模型與 Modal 依賴：

```powershell
python -m venv helper_tool/reports/training-runtime
$trainingPython = '.\helper_tool\reports\training-runtime\Scripts\python.exe'
& $trainingPython -m pip install -r Solution/requirements-embedding.txt -r Solution/requirements-modal.txt
```

WSL 使用 `.venv/bin/python` 執行相同的 `-m pip`、`-m modal` 命令，並從 WSL 啟動 UI。**Windows 與 WSL 的登入設定不是同一份**，不要只在 WSL 登入，卻從 Windows UI 啟動。

## 3. 登入 Modal 並確認 Workspace

在同一個 PowerShell 執行：

```powershell
& $trainingPython -m modal setup
& $trainingPython -m modal token info
```

第一個命令會引導瀏覽器登入／授權，完成後回到終端機。第二個命令用來確認目前 token 可以使用。這兩步不會啟動本專案訓練，也不會上傳 history。[Modal 官方入門](https://modal.com/docs/guide/getting-started)

如果你有多個登入 profile：

```powershell
& $trainingPython -m modal profile list
& $trainingPython -m modal profile activate 你的Profile名稱
& $trainingPython -m modal token info
```

確認使用的 Workspace 正確。登入憑證由 Modal SDK 管理，不要把 token 複製到本專案的 JSON、Git 或訓練表單。已開啟的 UI 建議在登入／切換 profile 後重新啟動。

## 4. 專案設定檔改哪裡

編輯 **`Solution/config/modal_finetune.json`**：

```json
{
  "app_name": "silab-bge-training",
  "gpu": "T4",
  "cpu": 2,
  "memory_mb": 8192,
  "timeout_seconds": 3600
}
```

| 欄位 | 用途與可填值 |
|---|---|
| `app_name` | Modal 工作名稱，方便在 Dashboard 找到；3～63 字的小寫英文、數字、連字號，以英文開頭 |
| `gpu` | 單張 GPU；本專案允許 T4、L4、A10、L40S、A100-40GB、A100-80GB；預設 T4 是起始設定，不代表已實測最佳或最快 |
| `cpu` | CPU 資源請求，整數 1～8，預設 2 |
| `memory_mb` | 傳入 Modal `memory` 的 RAM 資源請求，實際單位為 **MiB**；整數 1024～65536，8192 表示 8 GiB；不是 GPU VRAM |
| `timeout_seconds` | 遠端函式執行上限，整數 60～14400；3600 表示最多 1 小時，包含訓練、評估與封裝／串流，不是金額上限，也不涵蓋整個本機啟動等待時間 |

CPU／RAM 單值為最低資源請求，不是硬性最大值；Modal 的 CPU／RAM 計費會比較請求與實際用量。修改資源也會影響成本。[資源與計費說明](https://modal.com/docs/guide/resources)

這是**本專案讀取的設定檔**，不用上傳到 Modal 後台。新建 Modal 工作時會驗證並複製到該次 `config.json` 的 `modal_settings`；執行時另保存 `modal-runtime.json`，記錄實際使用的資源與 SDK 版本。修改 JSON 只影響之後新建的 Modal 工作，CPU／本機 GPU 訓練不讀取這份檔案。

學習率、epochs、dropout 等**模型訓練參數仍在 UI 設定**，另存為同次 `config.json` 的 `settings`。GPU 資源與模型超參數分開保存；JSON 不會蓋掉 UI 的 epochs。遠端 Python 會自動對齊啟動訓練 worker 的版本（目前允許 3.12／3.13），以符合 `serialized=True` 的相容性要求；套件版本仍由既有 requirements 固定。

## 5. 從 UI 啟動微調

1. 雙擊 `helper_tool/啟動訓練UI.cmd`，或在上述環境執行：

   ```powershell
   & $trainingPython -X utf8 helper_tool/ui_server.py
   ```

2. 打開終端機顯示的本機網址，進入側欄「模型訓練」。
3. 訓練方式選「BGE encoder＋分類頭聯合微調，與凍結版比較」。
4. 訓練環境選「Modal 雲端 GPU · 依專案設定」。
5. 來源選專案 history 或自己選取的 history JSONL，指定保存根目錄。來源必須有五類／null 分類標籤；完整四欄評估還需要既有 history 的 device_id、urgency、duplicate_of 標籤。不得把 dev 當成訓練資料。
6. 檢查 epochs、學習率等 UI 設定，再勾選 history 來源確認及本次 Modal 上傳／GPU 費用確認。
7. 按「開始訓練與保存」。**這一步才會啟動遠端工作，並傳送選取的 history。**

固定 BGE 資產需先存在 `Solution/models/bge-small-zh-v1.5/`。此機已準備；新電腦若沒有，可明確執行下載：

```powershell
& $trainingPython Solution/scripts/download_encoder.py
```

訓練本身不會臨時拉取另一個 encoder revision。Modal 會為遠端 Image 安裝 requirements 中的套件；首次 Image 建置與約百 MB 等級模型傳送通常會有額外等待，不能拿先前 CPU 約 46 秒的數字當成 Modal 全流程耗時預測。

## 6. 查看雲端工作與進度

本機頁面會顯示準備環境、方法、epoch、batch、驗證與模型下載等階段。每個 epoch 的指標會同步更新，完整結果完成一組便寫入比較紀錄。整體百分比是階段估計；完成保存才到 100%。Modal 串流的中間結果以事件回傳，完整比較表與模型在下載完成後補齊。

也可以開啟 Dashboard：

```powershell
& $trainingPython -m modal dashboard
```

在 Apps／執行紀錄中找設定的 `silab-bge-training`，查看 image build、GPU 執行狀態、log 與費用。這個流程使用 `app.run()`，不需要事先執行 `modal deploy`。`Solution/modal_training.py` 是由本機 worker 呼叫的模組，不是供 `modal run Solution/modal_training.py` 使用的獨立 CLI 入口。

保持電腦開機與網路連線，讓本機 worker 持續接收進度與下載模型。切換 UI 頁面不會停止工作；關閉瀏覽器後可再回來查看。但本機程序中止／電腦重開後，尚未實作遠端工作續接。若中斷，先到 Modal Dashboard 確認遠端是否仍在執行，再決定是否重建工作，避免重複啟動。

## 7. 訓練完成後在哪裡

每次工作會在指定保存根目錄建立「台北時間＋唯一編號」子資料夾，重點檔案如下：

```text
該次工作/
  config.json                 # 資料來源、UI 超參數、固定的 Modal 資源設定
  modal-runtime.json          # 使用的資源、Modal SDK 與遠端 Python 版本
  status.json                 # 完成狀態與總耗時
  events.jsonl                # 進度與 epoch 指標
  worker.log                  # 本機 worker 與連線錯誤
  source/                     # 本次程式與設定快照
  models/
    comparison.json           # 四組方法、既有 metrics 與本機模型路徑
    frozen/                   # 固定 BGE 與 LR／SVC
    frozen-neural/            # 固定 encoder 的神經頭對照
    finetuned/                # 微調後 encoder、分類頭與 metadata
```

模型自動下載回本機，不需手動從 Modal 下載。UI 顯示分類 F1、六類 F1、null recall、四欄管線指標與總分；「帶入預測」會填入本機模型路徑。history 驗證段兼用於選 epoch，不是獨立測試；沿用 A 規則曾受 dev 觀察影響的來源限制仍保留。

## 常見問題

| 訊息／狀況 | 處理方式 |
|---|---|
| serialized Function 的 Python 版本不相容 | 已修正為 Image 自動對齊本機 worker 的 Python minor 版本；建立新工作即可，舊失敗紀錄不自動重跑 |
| 沒有 token／未登入／Unauthenticated | 用**啟動 UI 的同一個 Python 與作業環境**重新執行 `-m modal setup`，再用 `token info` 確認 |
| Workspace 權限或付款／GPU 額度不足 | 在 Modal 帳號檢查 Workspace、角色、付款方式與限制；改 JSON 不能繞過帳號限制 |
| 找不到固定 BGE encoder | 先明確執行 `Solution/scripts/download_encoder.py`，再建立新工作 |
| JSON 設定不合法 | 使用標準 JSON，不放註解／尾逗號；只能填上述五個欄位，不可放 token |
| GPU 等待較久／資源不可用 | 查看 Dashboard；如需改 GPU，先確認舊工作已停止，再修改設定並建立新工作 |
| OOM 或 timeout | 查看 worker／Modal log；GPU OOM 檢查 VRAM／GPU 型號，主記憶體不足檢查 RAM，timeout 檢查執行上限與訓練設定。UI 目前固定 batch=8；timeout 不會增加記憶體，不要用 dev 調參 |
| ZIP does not support timestamps before 1980 | Modal 檔案可能帶有 ZIP 無法表示的舊時間戳；封裝已改為將 ZIP 時間戳限制在格式允許範圍，模型內容與 SHA-256 不變。此錯誤可能發生在訓練後，不能把訓練完成當成下載成功 |
| 模型下載不完整 | 工作會標記 failed 並保留紀錄；只有下載與校驗完成才是成功，不自動重試付費訓練 |

初期只完成離線接線測試，未涵蓋實際雲端生命週期。2026-11-14 修正 Python 版本與 ZIP 舊時間戳問題後，已用本次授權的 history／設定完成真實 Modal T4 訓練、下載、四模型本機載入及 UI 預測 API 驗證，詳見下方紀錄。資料上傳只限你已授權的當次 history；不掛載原始 data／既有 reports 或憑證目錄。未建立永久 Volume 不代表 Modal 立即刪除平台快取或執行紀錄。

本次變更驗證：20 項相關測試中 19 項通過、1 項 dev opt-in 驗證未啟用；預設資源定義已由本機 Modal SDK 1.6.4 接受，未進入 `app.run()` 或啟動任何遠端工作。

## 實際 Modal 全流程驗證（2026-11-14）

- 成功工作：`20261114-175411-405736-557d7766`，狀態 `succeeded`，進度 100%。本次直接沿用前次已授權 history 與設定，沒有根據分數調整算法。
- 執行環境：Modal T4／CUDA、Python 3.13，最多 10 epochs；兩個神經模型均在第 5 輪提早停止，最佳 checkpoint 為第 4 輪。finetuned 的 encoder 權重確實更新，frozen-neural 保持不變。
- 遠端四方法訓練與完整管線評估約 **46.78 秒**；本機 worker 全流程（含準備、封裝、傳輸、校驗與解壓保存）約 **302.33 秒**，不能把兩者混稱為純訓練時間。
- ZIP 修正使用 `strict_timestamps=False`，讓格式不能表示的舊日期被限制到可表示範圍；不改動模型內容。新增測試直接執行 remote generator，使用 Unix epoch 日期的合成模型，驗證封裝、SHA-256 與還原。15 項相關測試全數通過。
- 四個真實雲端模型已逐一在 Windows CPU 載入，每個模型皆執行兩筆合成輸入推論；微調模型另透過本機 `/api/run` 成功完成一筆合成工單預測。這是使用流程驗證，不是準確率評測。
- 產物留在本機 `Solution/models/training-runs/20261114-175411-405736-557d7766/`，其中 `end-to-end-verification.json` 保存驗證結果，`models/comparison.json` 保存既有指標，`status.json` 保存全流程耗時。原始 history、逐筆評估與模型不加入 Git。
- 本次微調六類 F1 約 0.616358，凍結 SVC 約 0.891131；這些都是 history 選型驗證結果，流程跑通不代表模型優於對照。未使用 dev，沿用 A 的來源限制仍適用。
- 前次工作 `20261114-174252-938227-27278570` 在訓練完成後因 ZIP 日期錯誤失敗，本機封裝為 0 bytes；遠端使用暫存目錄，沒有可供此次續接的持久化模型，因此本次為明確的新工作，非恢復前次 checkpoint。
