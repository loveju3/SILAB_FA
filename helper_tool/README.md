# 開發輔助工具

只需 Python 3，沒有第三方套件。以下指令從專案根目錄執行；工具也能從其他目錄以絕對路徑執行。使用者提供的檔案路徑相對於執行目錄，預設資料路徑則固定相對於專案根目錄。

## 網頁操作版（建議）

在 macOS Finder 打開 `helper_tool` 資料夾，**雙擊「啟動工具.command」**。它會自動啟動本機服務並開啟瀏覽器，不需要每次輸入指令。使用期間保留啟動視窗；結束時在該視窗按 Control+C。

介面有兩個分頁：

- **評分分析**：選擇／拖入預測 JSONL，選 dev、history 或自訂答案檔，按「開始評分」。也可以直接按「載入 Baseline 範例」。畫面顯示總分、四項指標、錯誤長條圖及分類混淆矩陣。錯題可依欄位、ID、原文或回報人篩選，按「查看差異」對照原文與四欄正解／預測。可下載錯題 CSV、JSONL 及指標 JSON。
- **工單查詢**：輸入一個或多個 ticket_id，按查詢或 Enter。直接呈現原文、回報人、時間、標籤、來源檔與行號；重複工單編號可點擊追查。

評分檔只在本機讀取，原始資料不會被修改。每次評分報告保留在 `helper_tool/reports/ui/<時間與唯一編號>/`；重啟後檔案仍保留，網頁上的分析狀態則不保留。評分規則及檔案檢查與下方命令列工具相同。選檔每檔上限 8 MB，請求總量上限 16 MB。

預設網址為 `http://127.0.0.1:8765`，埠號被占用時自動嘗試後續 19 個埠，實際網址顯示在啟動視窗。只監聽本機，不使用外部字型、CDN 或雲端服務。若瀏覽器未自動打開，點擊或複製啟動視窗內的網址。

其他系統或偏好自行啟動時可使用 `python3 helper_tool/ui_server.py`。UI 檔案位於 `helper_tool/web/`，後端入口為 `ui_server.py`。

## 1. 自動評分與錯誤分析

先產生預測，再評分。每次調整算法後，對新預測檔執行第二行即可。

```bash
python3 baseline/predict.py data/dev.jsonl /tmp/baseline_dev.jsonl
python3 helper_tool/evaluate.py /tmp/baseline_dev.jsonl

# 自選答案檔及報告位置：不同實驗使用不同目錄即可保留結果
python3 helper_tool/evaluate.py /tmp/baseline_dev.jsonl data/dev.jsonl --output-dir helper_tool/reports/baseline
```

終端顯示官方各欄分數、總分、較精確的六位小數總分、百分制換算與各欄問題筆數。總分直接呼叫原本的 `baseline/evaluate.py`，不修改官方程式：

`total = 0.4 × category macro-F1 + 0.2 × device accuracy + 0.2 × urgency macro-F1 + 0.2 × duplicate F1`

預設輸出到 `helper_tool/reports/<預測檔名去副檔名>/`，同一目錄重跑會覆寫。自動產生的 reports 不加入 Git。

| 報告 | 用法 |
|---|---|
| `report.md` | 快速看總分、各欄錯誤數、前十種正解→預測混淆 |
| `summary.json` | 完整精度指標、各類 precision/recall/F1/support、duplicate TP/FP/FN、缺失/多餘 ID、無效預測及建議缺漏 |
| `errors.csv` | 用試算表篩選 `wrong_fields`；同列對照原文、各欄正解、預測及原因，UTF-8 BOM 支援 Excel |
| `errors.jsonl` | 每張有問題的工單一列，含原始資料、預測、差異、來源行號與格式問題 |
| `official.txt` | 原始官方評分輸出，方便核對 |

錯誤原因：

- `wrong_value`：欄位值錯誤。
- `abstention_requires_both_null`：標準答案 device/category 同為 null，預測未同時給 null，device 不得分。
- `false_positive` / `false_negative` / `wrong_target`：duplicate 誤報、漏報或指錯單。指錯單同時計入 FP 和 FN。
- `missing_prediction` / `invalid_prediction`：漏掉整筆，或缺必要欄位／llm_used 型別錯誤。這些工單的四欄都標示問題；分數仍完全依官方規則計算。

category 的 gold=null 不列入分類 F1，也不列為分類錯題；雙 null 棄答錯誤列入 device。duplicate 的 null/null 不是 TP。錯題筆數不能直接當作 F1 的扣分，也不能把各欄錯誤數相加當作有錯工單數。`suggestion` 不計分，缺漏記在摘要。多餘 ID 由摘要列出，不計入 gold 錯題清單或總分。

只接受有標籤的 dev/history 作為答案檔；tickets 沒有答案，不能拿來算分。重複 ticket_id、壞 JSON、非 object、空答案檔及計分欄位非字串/null 會明確報錯並退出，避免產生含糊結果。這比官方對重複 ID「後一筆覆蓋」更嚴格；必要欄位缺漏及 llm_used 非 boolean 則依官方方式評分並標記無效。這是開發診斷工具，不是完整輸出 schema validator。

成功評分退出碼為 0，即使有答錯或預測格式問題；無法讀取／解析／評分時為 2。查看 `summary.json` 的檢查結果判斷完整性。

## 2. 快速查詢工單

```bash
# 預設查詢 tickets、dev、history，保留原文與所有存在的欄位
python3 helper_tool/lookup.py T02306

# 同時查多筆，方便比較 duplicate 前後單
python3 helper_tool/lookup.py T02306 T02304

# 不帶 ID：載入一次，連續輸入編號查詢，輸入 q 離開
python3 helper_tool/lookup.py

# 輸出 JSON，可供其他程式使用
python3 helper_tool/lookup.py T02306 --json

# 查指定檔案，也可以同時查原始資料及自己的預測
python3 helper_tool/lookup.py T02306 --files data/dev.jsonl /tmp/baseline_dev.jsonl
```

每筆結果包含來源檔案、原始行號及完整 JSON。精確比對 ticket_id，不改寫缺字符號或原文。跨檔同 ID 的資料會全部顯示；這只是查詢，不會跨檔推論 duplicate 或上下文。同檔重複 ID 會報錯。

找到全部 ID 退出碼為 0；任一 ID 查無資料為 1（JSON 模式對應空陣列）；讀檔或格式錯誤為 2。`--files` 接受多個路徑，請放在 ID 後面，如上例。

## 驗證

```bash
python3 -m unittest discover -s helper_tool -p 'test_*.py' -v
```

測試會比較 helper 的指標加權結果與官方評分，涵蓋真實 baseline、雙 null、缺失／無效預測、duplicate 指錯對象、重複 ID、壞 JSON 及跨來源查詢。測試產物放在暫存目錄。
