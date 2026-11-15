# Windows 原生 Agent 環境

Agent 已納入 [Windows 完整環境建置](windows.md)。新使用者依該份流程一次準備第一、第二階段，不必另建 Agent 環境或切換 WSL。

若已有相容的 Windows Python，仍可執行 `python agent/setup.py`，下載並驗證 `agent_assets/Qwen_Qwen3-1.7B-Q4_K_M.gguf` 及完整 `agent_assets/runtime/`。只搬 exe 不完整。一般測試請使用完整流程的同一個虛擬環境，並啟用 UTF-8。

macOS 的原生 runtime 也已加入安裝器；完整步驟與尚未實機驗證的限制見 [Mac](macos.md)。Linux／WSL 自備原生 runtime，屬進階路線。
