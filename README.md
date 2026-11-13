# SILAB FA：工單分流程式

目前包含方案 A 規則版、方案 B 字元 TF-IDF、方案 C 固定 BGE encoder（B/C 均提供 LR 與 LinearSVC）、評分／查詢 UI、測試、流程圖及彙總效能說明。

工作台另提供訓練 modal：上傳 history、指定模型保存路徑、執行 BGE encoder＋分類頭聯合微調及凍結對照，並保存逐輪紀錄與時間戳模型。操作與限制見 [UI 訓練文件](Solution/docs/BGE聯合微調與UI訓練.md)。

**重要開發規則：不得觀察 `dev.jsonl` 的資料來設計算法；dev 僅供固定版本驗證。** 開發與調參使用 history，完整規範見 [AGENTS.md](AGENTS.md)。既有方案 A 曾參考 dev，其成績只作歷史開發驗證紀錄。

- [Solution 操作與架構](Solution/README.md)
- [演算法流程圖](Solution/docs/演算法流程.md)
- [TF-IDF 實作、history 比較與操作](Solution/docs/TF-IDF實作與比較.md)
- [BGE 方案 C、批次流程與實驗](Solution/docs/BGE方案C方法與實驗.md)
- [輔助工具](helper_tool/README.md)
- [CPU 量測](Solution/bench.md)
- [2 GB 驗證](Solution/budget.md)

目前版本不追蹤原始工單、預測結果、逐筆錯誤報告、含工單 ID 的摘要，以及含工單摘錄的原始題目／分析文件。測試程式中的範例用於驗證契約，執行真實資料評測時仍需自行在本機提供資料。

請自行在本機準備 data/devices.json、data/error_codes.json，以及需要的工單 JSONL。data/、JSONL 與報告已設為忽略；不應將任何工單資料加入 Git。

本次合併保留 main 既有歷史；舊提交中的資料仍可能存在。之後產生的預測 JSONL 僅留本機，不加入 Git，也不推送至 GitHub。
