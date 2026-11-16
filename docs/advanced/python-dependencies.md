# Python 依賴與版本固定依據

> **進階／歷史文件。** 一般安裝請用 [docs/setup](../setup/README.md) 的 `python3 install.py`，不需要照本頁操作。本頁保留先前的手動流程與實測紀錄：文中的 `.venv-windows`、`.venv-macos` 現已統一為 `.venv`；`SILAB_FA_test_models_20261115.zip` 完整模型包只有要跑 `--method tfidf-lr／tfidf-svc／bge-lr／bge-svc` 時才需要，預設的微調模型由 `install.py` 自動下載。

`Solution/requirements-embedding.txt` 是 TF-IDF、BGE 與微調 BGE 的共用安裝入口，透過 `-r requirements-ml.txt` 引入基本 ML 套件。NumPy 已包含在這個入口，不需要再手動安裝一次。pip 會依各套件的 metadata 安裝間接依賴；目前兩份檔案並非涵蓋所有間接依賴與雜湊的完整 lockfile。

| 套件 | 專案用途 |
|---|---|
| NumPy | `Solution/triage/embedding.py` 的 `BgeEncoder.transform/encode` 建立空陣列、串接 batch，並將 encoder 的 Torch tensor 轉為 NumPy array；微調推論沿用這條路徑。`neural.py` 的 `HeadAdapter.predict` 再將陣列轉回 Torch tensor。 |
| SciPy、scikit-learn | TF-IDF、線性分類與相關數值運算；`linear.py`、`embedding.py` 載入分類模型時會核對 scikit-learn 版本。 |
| joblib、threadpoolctl | 本機分類模型保存／載入，以及 scikit-learn 使用的數值執行緒控制依賴。 |
| torch | BGE encoder 與神經分類頭推論、訓練，以及 Tensor／NumPy 轉換。 |
| transformers、huggingface-hub、tokenizers | 載入本機 BGE encoder、設定與 tokenizer。 |
| safetensors | 讀取 encoder 與微調神經分類頭權重。 |

純 `rules` 方法透過 lazy loading 保持不依賴 ML 套件；使用 Mac 共用安裝腳本時，仍會一次安裝上述依賴。

## 為什麼是 NumPy 2.3.4

需要 NumPy，但目前檢查到的程式沒有要求 2.3.4 專屬 API。`numpy==2.3.4` 是既有 repo 為固定環境採用的版本，[既有 Windows 驗證紀錄](validation-2026-11-15.md) 使用此版本並通過 pip check；完整環境檢查器也依此版本驗證。這不能當作 Mac 實機安裝已通過的證據。

上游套件的 runtime 宣告也沒有要求精確等於 2.3.4：

- [SciPy 1.16.3 的 pyproject.toml](https://github.com/scipy/scipy/blob/v1.16.3/pyproject.toml)：`numpy>=1.25.2,<2.6`（一般 CPython 的 runtime 範圍）。
- [scikit-learn 1.7.2 的 pyproject.toml](https://github.com/scikit-learn/scikit-learn/blob/1.7.2/pyproject.toml)：`numpy>=1.22.0`。
- [Transformers 4.57.1 的 setup.py](https://github.com/huggingface/transformers/blob/v4.57.1/setup.py)：`numpy>=1.17`。

2.3.4 落在以上宣告範圍內；宣告相容不等於整套推論已完成實機驗證。現階段保留既有精確版本，避免在未驗證下改變模型執行環境。若要換版本，需同步檢查 requirements、環境檢查器與模型推論相容性。

## 安裝失敗怎麼判斷

[NumPy 2.3.4 官方 PyPI](https://pypi.org/project/numpy/2.3.4/#files) 提供 CPython 3.12／3.13 的 macOS arm64 wheel，其中包含 `numpy-2.3.4-cp313-cp313-macosx_14_0_arm64.whl`，對應本機 macOS 15.6.1、Python 3.13、arm64 平台條件。

`No matching distribution found` 不能單獨證明版本不存在或缺少依賴。應一起看前面的連線、TLS、套件索引與平台錯誤；無法讀取索引時，pip 也可能列出零個候選版本。補上其他套件無法修復 DNS 問題。

最初 Mac 安裝嘗試在受限制的執行環境中失敗，對 `pypi.org` 與 `files.pythonhosted.org` 的 DNS 查詢均回報 `nodename nor servname provided, or not known`。2026-11-16 允許連網後，同一安裝腳本已成功下載並安裝原有固定版本，所有相依性、版本與載入檢查通過，無須更換 NumPy 版本或新增套件。詳見 [Mac 實測紀錄](validation-2026-11-16-macos.md)。

排除套件來源連線問題後，從 repo 根目錄重新執行 `python3 scripts/setup_macos.py`，由腳本的 PASS／FAIL 判斷 Python 環境是否完成。模型與資料另行準備，不涉及 dev 內容檢視。
