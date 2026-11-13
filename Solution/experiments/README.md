# 實驗紀錄

`scheme_A_summary.json` 保存本階段本機摘要，含工單 ID，因此與 `runs/` 原始紀錄一同排除於 Git。每次 runner 建立獨立目錄，避免覆蓋前次結果。

本機量測保存版本、程式/輸入雜湊、設定、輸出與完整逐筆耗時；Docker 演練另保存限制、OOM 狀態及 cgroup 峰值。正式預測副本留在 `Solution/predictions/`，僅供本機使用，不上傳 GitHub。

進入 B/C 時沿用相同輸出契約與評分器，以獨立執行目錄保存各方案結果；沒有實測的方案不填推測分數。
