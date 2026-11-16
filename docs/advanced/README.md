# 進階與歷史文件

一般安裝與測試只需要 [docs/setup](../setup/README.md)（`python3 install.py`）。這裡保留開發者工作與先前的實測紀錄：

| 文件 | 內容 |
|---|---|
| [cpu-models.md](cpu-models.md) | 重新訓練 TF-IDF／固定 BGE、Docker 映像與 2 GB 容器演練 |
| [windows.md](windows.md)、[macos.md](macos.md)、[macos-advanced.md](macos-advanced.md) | `install.py` 出現前的手動安裝流程（含完整模型包與 Agent） |
| [windows-wsl.md](windows-wsl.md)、[windows-native-agent.md](windows-native-agent.md) | 既有 WSL 開發環境與 Windows 原生 Agent 說明 |
| [python-dependencies.md](python-dependencies.md) | 固定套件版本的理由與疑難排解 |
| [test-readiness.md](test-readiness.md) | 環境檢查與 41 項跨裝置驗收清單的用法 |
| [validation-2026-11-15.md](validation-2026-11-15.md)、[validation-2026-11-16-macos.md](validation-2026-11-16-macos.md) | 先前 Windows／Mac 的實測紀錄與限制 |
| [../testing/](../testing/finetuned_transfer_readme.md) | 微調模型的 Docker 移轉、可攜映像匯入 |
| [GPU／Modal 訓練](../../Solution/docs/訓練頁與運算環境.md) | 雲端 GPU 微調後端 |
