# 演算法文件

| 想了解什麼 | 文件 |
|---|---|
| macOS／Windows 環境建立、模型依賴與啟動方式 | [統一環境建立文件](../../docs/setup/README.md) |
| 既有 Windows／WSL Docker 2 GiB 實測與疑難排解 | [Windows／WSL 驗證紀錄](Windows-WSL環境與2GB驗證.md) |
| Modal 帳號登入、設定檔、雲端微調與模型下載的完整步驟 | [Modal 微調設定與操作](Modal微調設定與操作.md) |
| 訓練頁、CPU／CUDA GPU／Modal、進度條與同步評估 | [訓練頁與運算環境](訓練頁與運算環境.md) |
| 方案 A 如何判斷工單 | [演算法流程](演算法流程.md) |
| 方案 B 的概念、設計重點與流程圖 | [字元 TF-IDF 分類方法說明](字元TF-IDF分類方法說明.md) |
| 方案 B 的實作、實驗數字與操作 | [TF-IDF 實作與比較](TF-IDF實作與比較.md) |
| 方案 C 的白話分工、三張流程圖、固定 encoder、實驗與操作 | [BGE 方案 C 方法與實驗](BGE方案C方法與實驗.md) |
| 從網頁上傳 history、聯合微調 encoder 與分類頭、保存紀錄及模型 | [BGE 聯合微調與 UI 訓練](BGE聯合微調與UI訓練.md) |

閱讀方案 B 時，建議先看方法說明，再看實驗比較。所有算法開發遵循 [專案資料使用規範](../../AGENTS.md)：dev 只用於固定版本驗證，不用於算法開發。

方案 C 的流程圖使用隨文件保存的本機 SVG，可直接用 VS Code 的 **Markdown: Open Preview** 開啟，不需要 Mermaid 擴充套件。分享文件時請一併保留 `assets/` 圖檔。
