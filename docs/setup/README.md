# 建立環境

一般測試者只需依作業系統完成一份文件，準備完就能執行第一、第二階段與同一個測試 UI：

| 作業系統 | 完整流程 | 本次驗證程度 |
|---|---|---|
| Windows x64 | [Windows](windows.md) | 同一原生環境已跑批次六方法、開發測試與真實 Agent／合成巡檢 |
| macOS Apple Silicon | [Mac](macos.md) | 安裝器／官方資產已查核；無 Mac 實機，端到端待驗 |

兩份流程包含資料、完整測試模型包、Python 套件、Qwen／原生 llama.cpp、環境檢查及各項測試入口。模型包由維護者提供，不需要測試者重新訓練。原始工單另行在本機準備；模型、虛擬環境與逐筆報告不加入 Git。

[本次實測與限制](validation-2026-11-15.md)；[環境檢查與驗收入口](test-readiness.md)。

## 開發者進階工作

以下不是一般測試者建立環境的必要步驟：

若要執行下面的 Docker 建置、映像匯入或 2 GB 容器測試，尚未安裝者需先安裝並啟動 [Docker Desktop](https://www.docker.com/products/docker-desktop/)。執行 `docker version`，確認同時顯示 Client 與 Server，再繼續操作。

- [CPU 模型訓練與 Docker／2 GB](cpu-models.md)
- [既有 WSL 開發環境](windows-wsl.md)
- [微調模型與 Docker 移轉](../testing/finetuned_transfer_readme.md)
- [既有可攜映像匯入](../testing/portable_image_readme.md)
- [GPU／Modal 訓練後端](../../Solution/docs/訓練頁與運算環境.md)

資料使用依 [AGENTS.md](../../AGENTS.md)：dev 僅供固定版本驗證，不能拿原文、標籤、逐筆結果或分數反覆設計與調整算法；沿用方案 A 的來源限制保留。
