# SILAB FA：工單分流程式

目前包含方案 A、評分／查詢 UI、測試、流程圖及彙總效能說明；內容由 codex/solution-a-no-ticket-data 合併至 main。

- [Solution 操作與架構](Solution/README.md)
- [演算法流程圖](Solution/docs/演算法流程.md)
- [輔助工具](helper_tool/README.md)
- [CPU 量測](Solution/bench.md)
- [2 GB 驗證](Solution/budget.md)

目前版本不追蹤原始工單、預測結果、逐筆錯誤報告、含工單 ID 的摘要，以及含工單摘錄的原始題目／分析文件。測試程式中的範例用於驗證契約，執行真實資料評測時仍需自行在本機提供資料。

請自行在本機準備 data/devices.json、data/error_codes.json，以及需要的工單 JSONL。data/、JSONL 與報告已設為忽略；不應將任何工單資料加入 Git。

本次合併保留 main 既有歷史；舊提交中的資料仍可能存在。之後產生的預測 JSONL 僅留本機，不加入 Git，也不推送至 GitHub。
