# 技術選型與實作進度

dev 僅供固定版本驗證，不參與規則設計、選型或調參；開發比較使用 history 的內部分割。既有 A 曾參考 dev，表內 A 分數只作歷史開發驗證紀錄。沿用 A 模組的後續方案須保留其來源限制。完整規範見 [AGENTS.md](../AGENTS.md)。

| 方案 | 模型／產物大小 | dev 總分 | 狀態與判斷 |
|---|---|---:|---|
| 原始 baseline | 無權重；原始 Python 關鍵詞規則 | 0.441574 | 官方對照，duplicate 永遠 null，無完整 urgency 與正規化 |
| A：完整規則 | 無權重；推論入口＋triage＋config 共 22,049 bytes | 0.908779 | 已實測並保留；低依賴、易追查，口語分類召回不足 |
| B-LR：字元 TF-IDF＋Logistic Regression | 本機 joblib 148,297 bytes | 未驗證 | 已完成四候選比較；選遮罩版，history 內部 category F1=0.746499 |
| B-SVC：字元 TF-IDF＋LinearSVC | 本機 joblib 152,327 bytes | 未驗證 | 遮罩對照版，history 內部 category F1=0.752058；不以 dev 選型 |
| C-LR：固定 BGE-small-zh-v1.5＋LR | 共用 encoder 96,377,581 bytes＋頭檔 24,720 bytes | 未驗證 | 已實作；保留名稱，history category F1=0.793935、六類 F1=0.805729 |
| C-SVC：固定 BGE-small-zh-v1.5＋SVC | 共用同一 encoder＋頭檔 24,762 bytes | 未驗證 | 保留名稱，history category F1=0.897928、六類 F1=0.891131 |

此為目前進度，B/C 已實作，尚未完成 dev 固定版本驗證或最終產品選型。history 分割與限制見 [TF-IDF](docs/TF-IDF實作與比較.md) 及 [BGE](docs/BGE方案C方法與實驗.md)。C 的遮罩版 category 更高，但原先六類選型規則偏好保留名稱；不混用不同選型指標。產物大小不等於程序 RAM；延遲及記憶體統一見 [bench.md](bench.md)，2 GB 見 [budget.md](budget.md)。

## C 的聯合微調實驗

已新增 UI 訓練 modal、history 上傳、獨立時間戳模型及紀錄保存，並比較固定 BGE＋相同神經頭與 encoder＋頭聯合微調。預設設定下，history 內部分類 F1 分別為 0.686637／0.792377，六類 F1 為 0.553206／0.639588；目前仍不及凍結 SVC 的 0.897928／0.891131。微調最佳 epoch=4，unknown recall=0，未據此再調參或替換預設模型；驗證段只有 5 筆 null。

聯合微調保留 train-only 的最佳 checkpoint，不以完整 history 重訓，與既有 B/C 保存全 history 重訓資產的流程不同。模型與完整紀錄均留本機，來源限制仍適用。操作、loss、保守超參數與控制組設計見 [BGE 聯合微調與 UI 訓練](docs/BGE聯合微調與UI訓練.md)。
