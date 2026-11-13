# 技術選型與實作進度

dev 僅供固定版本驗證，不參與規則設計、選型或調參；開發比較使用 history 的內部分割。既有 A 曾參考 dev，表內 A 分數只作歷史開發驗證紀錄。沿用 A 模組的後續方案須保留其來源限制。完整規範見 [AGENTS.md](../AGENTS.md)。

| 方案 | 模型／產物大小 | dev 總分 | 狀態與判斷 |
|---|---|---:|---|
| 原始 baseline | 無權重；原始 Python 關鍵詞規則 | 0.441574 | 官方對照，duplicate 永遠 null，無完整 urgency 與正規化 |
| A：完整規則 | 無權重；推論入口＋triage＋config 共 22,049 bytes | 0.908779 | 已實測並保留；低依賴、易追查，口語分類召回不足 |
| B-LR：字元 TF-IDF＋Logistic Regression | 本機 joblib 148,297 bytes | 未驗證 | 已完成四候選比較；選遮罩版，history 內部 category F1=0.746499 |
| B-SVC：字元 TF-IDF＋LinearSVC | 本機 joblib 152,327 bytes | 未驗證 | 遮罩對照版，history 內部 category F1=0.752058；不以 dev 選型 |
| C：A＋BGE-small-zh／分類頭 | 尚未下載或產生 | 尚未實測 | 預定實驗，須由分類收益與資源實測決定是否採用 |

此為目前進度，B 已實作，C 尚未實作，不宣稱已完成最終 A/B/C 選型。history 分割與限制見 [TF-IDF 實作與比較](docs/TF-IDF實作與比較.md)。產物大小不等於程序 RAM；延遲及記憶體統一見 [bench.md](bench.md)，2 GB 見 [budget.md](budget.md)。
