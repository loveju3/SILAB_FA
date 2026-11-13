# TF-IDF＋Logistic Regression／LinearSVC 核心程式碼導讀

本文件依目前工作區程式整理。程式片段為原碼節錄或標明的流程簡化，方便對照，不是另一份訓練實作。本次整理未讀取 dev 或逐筆預測報告，也未重新訓練。

來源限制：目前訓練與選型僅使用 history，但沿用方案 A 的正規化、別名、遮罩及其他規則曾受 dev 觀察影響，因此不能宣稱全流程為獨立、未觀察 dev 的開發結果。

## 1. 先看整體流程

```text
訓練：history → 時間／近似模板切分 → 文字前處理 → 可選設備名稱遮罩
             → TF-IDF → 類別分類器（LR 或 LinearSVC）
                      → 資訊不足分類器（固定使用另一個 LR）
             → history 驗證集比較 → 固定設定 → 全 history 重訓 → 儲存

預測：四個輸入欄位 → 文字前處理 → 依模型設定遮罩
                   → 已知錯誤碼優先；否則 TF-IDF.transform
                   → 資訊不足判斷 → 若資訊足夠，再預測問題類別
                   → 其他既有流程組合最終輸出
```

TF-IDF 模型只負責問題類別與資訊不足判斷。設備辨識、上下文、急迫性、重複單與建議文字另由既有流程處理。

## 2. Input：只接收四個欄位

位置：`Solution/triage/io.py` 的 `read_tickets()`、`Solution/triage/contracts.py` 的 `Ticket`。

```python
@dataclass(frozen=True)
class Ticket:
    ticket_id: str
    created_at: datetime
    reporter: str
    raw_text: str
```

讀取 JSONL 時，以下投影確保其他欄位不會進入推論資料物件：

```python
values = {k: row[k] for k in ("ticket_id", "created_at", "reporter", "raw_text")}
```

`read_tickets()` 也檢查欄位型別、ID 重複、非空白 ID／reporter，以及時間是否包含時區，並將時間轉成 UTC。

**真正送進文字分類器的特徵來源是 `raw_text`。** 時間用於排序；reporter 用於其他上下文與重複判定流程。訓練標籤則由 `load_history()` 另行讀取，不混入輸入特徵。

## 3. 文字前處理與設備遮罩

位置：`Solution/triage/text.py`。

```python
def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_MAP).lower()
    text = re.sub(r"[\u200b-\u200d\ufeff]", "", text)
    return re.sub(r"([\u3400-\u9fff])\1+", r"\1", text)


def classification_text(text: str) -> str:
    text = normalize(text)
    return re.split(r"\n\s*--\s*\n|從我的\s*iphone\s*傳送", text)[0]
```

這裡做 Unicode NFKC 正規化、依既有 `_MAP` 做有限字元轉換、轉小寫、去零寬字元、壓縮連續重複中文字，並截除特定簽名尾段。它不是完整的簡繁轉換器。

位置：`Solution/train.py`；以下為原碼節錄。

```python
texts = [classification_text(t.raw_text) for t in tickets]
if masked:
    texts = [resolver.mask_names(text) for text in texts]
```

`mask_names()` 定義在 `Solution/triage/devices.py`，依設備表與別名，把名稱、位置、型號及別名替換成空白。遮罩方案讓分類器較集中在症狀文字；是否使用遮罩由 history 驗證集比較決定。訓練與預測必須使用同一個遮罩設定。

## 4. 訓練資料：history 切分與標籤

位置：`Solution/training/data.py`、`Solution/train.py`。

```python
history = ROOT / "data/history.jsonl"
tickets, gold = load_history(history)
train, validation, purged = history_split(tickets, resolver)
train_labels = [gold[tickets[i].ticket_id]["category"] for i in train]
```

`load_history()` 限制檔名為 `history.jsonl`、拒絕符號連結，檢查 category 必須是五類之一或 `null`，並依時間排序。

`history_split()` 預設以排序後約 70% 的位置建立時間邊界，較早資料作訓練、較晚資料作驗證；相同邊界時間的資料都進驗證。它還會先遮罩設備名稱，利用 `SequenceMatcher` 的相似度（預設 0.90）建立近似模板群組，移除訓練候選中與驗證集同群的資料，避免近似模板跨集合。

五個問題類別是 `noise`、`temperature`、`power`、`software`、`other`；`None` 表示資訊不足。**`other` 是有效類別，不等於資訊不足。**

## 5. TF-IDF：把文字變成數值特徵

位置：`Solution/training/linear.py` 的 `fit_bundle()`；以下節錄特徵建立部分。

```python
vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 4),
    lowercase=False,
    sublinear_tf=True,
    min_df=1,
    norm="l2",
)
features = vectorizer.fit_transform(texts)
```

| 設定 | 意義 |
| --- | --- |
| `analyzer="char"` | 以字元片段做特徵，不需要中文斷詞 |
| `ngram_range=(2, 4)` | 擷取連續 2～4 個字元 |
| `lowercase=False` | 前處理已統一小寫，此處不重複轉換 |
| `sublinear_tf=True` | 非零詞頻使用 `1 + log(tf)`，降低重複出現的影響 |
| `min_df=1` | 在訓練文字出現至少一次即可保留 |
| `norm="l2"` | 每筆特徵向量做 L2 正規化 |

例如自編文字「馬達過熱」會產生「馬達」「達過」「過熱」「馬達過」「達過熱」「馬達過熱」等特徵。

**`fit_transform()` 學習詞彙表和 IDF；驗證、預測只呼叫 `transform()`。** 比較候選方案時，這裡的 `texts` 只包含訓練切分，不包含驗證集。設定固定後的最終重訓，才改用全 history。

## 6. 模型訓練：兩個分類器各自負責一件事

位置：`Solution/training/linear.py` 的 `fit_heads()`；以下節錄通過輸入驗證後的訓練部分。

```python
known = [i for i, label in enumerate(labels) if label is not None]

category = (
    LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=2000, random_state=42
    )
    if method.endswith("-lr") else
    LinearSVC(
        C=1.0, class_weight="balanced", dual="auto",
        max_iter=5000, random_state=42
    )
)
category.fit(features[known], [labels[i] for i in known])

information = LogisticRegression(
    C=1.0, class_weight="balanced", max_iter=2000, random_state=42
)
information.fit(
    features,
    ["unknown" if label is None else "known" for label in labels],
)
```

- `category_model`：只用非空標籤的資料，學習問題類別。`tfidf-lr` 使用 Logistic Regression；`tfidf-svc` 使用 LinearSVC。
- `information_model`：使用全部訓練資料，學習 `known`／`unknown` 二元分類；兩種方案都固定用 Logistic Regression。

`class_weight="balanced"` 依該分類器訓練標籤頻率調整類別權重，讓少數類別有較大權重；`C` 控制正則化強度，值越小通常約束越強；`max_iter` 是最佳化迭代上限。

原函式還會檢查方法名稱、特徵與標籤筆數、至少兩個已知類別，以及是否存在 `None` 範例。正式訓練入口另外要求訓練切分涵蓋五類與 `None`。

## 7. 比較方案與固定設定

位置：`Solution/train.py`；以下為原碼節錄。

```python
for method in ("tfidf-lr", "tfidf-svc"):
    for masked in (False, True):
        # 前處理後，只取訓練切分進行 fit。
        # 後續在 history 驗證切分評估。
        ...

for method in ("tfidf-lr", "tfidf-svc"):
    options = [r for r in manifest["candidates"] if r["method"] == method]
    winners[method] = max(
        options,
        key=lambda r: (r["six_class_macro_f1"], r["masked"]),
    )["name"]
```

共比較四個方案：LR／SVC 各搭配保留或遮罩設備資訊。**程式為 LR 與 SVC 各選一個遮罩設定，不會自動把兩種方法縮成唯一冠軍。** 每種方法以包含 `unknown` 的六類 macro-F1 選設定，同分優先遮罩。

這個選型分數來自 `classifier.classify()`：包含錯誤碼優先規則與資訊不足判斷，但尚未加入完整 pipeline 的上下文修正；不能把它當作純五類模型分類頭的分數。

設定固定後，程式會用全 history 重訓四個方案，並另存每種方法選中的模型供預測使用。dev 不參與這個訓練與選型程式。

## 8. 儲存與載入模型

位置：`Solution/training/linear.py` 的 `fit_bundle()`、`Solution/triage/linear.py` 的 `save_model()`／`load()`。

模型 bundle 的內容如下（結構簡化）：

```python
bundle = {
    "vectorizer": vectorizer,
    "category_model": category,
    "information_model": information,
    "mask_device_names": masked,
}
```

`save_model()` 用 `joblib` 儲存 bundle，另存 JSON metadata，記錄 scikit-learn 版本與模型 SHA-256；訓練入口還提供設定、history 雜湊、原碼雜湊、驗證資訊與來源限制。

`LinearTextClassifier.load()` 會檢查方法、scikit-learn 版本及模型雜湊，再載入 bundle。joblib 只應載入可信的本機模型；雜湊檢查不會讓任意外來模型變成可信。

## 9. 預測：沿用訓練好的 TF-IDF 與模型

位置：`Solution/triage/linear.py`。

```python
def classify_features(self, features) -> CategoryMatch:
    if self.information_model.predict(features)[0] == "unknown":
        return CategoryMatch(None, "history 訓練的資訊不足判斷")
    return CategoryMatch(
        str(self.category_model.predict(features)[0]), self.evidence_label
    )

def classify(self, text: str) -> CategoryMatch:
    return self.code_match(text) or self.classify_features(
        self.vectorizer.transform([text])
    )
```

順序是：

1. 若文字中可識別的已知錯誤碼全部指向同一類別，直接回傳該類別。
2. 否則以既有 TF-IDF 轉換文字；不更新詞彙表或 IDF。
3. 資訊不足分類器預測 `unknown` 時回傳 `None`。
4. 資訊足夠時，使用 LR／LinearSVC 預測問題類別。

這裡**沒有用「類別模型信心低於某個閾值」判定資訊不足**，而是使用獨立訓練的二元分類器。

`classify()` 本身不做 `classification_text()` 或設備遮罩；正式入口在 `Solution/triage/pipeline.py` 先完成這些處理，再呼叫它。完整 pipeline 後續還可能依上下文調整類別，因此最終輸出不一定等於這一段的直接預測。

## 10. 建議閱讀順序與既有測試

依序閱讀 `triage/io.py` → `triage/text.py` → `training/data.py` → `training/linear.py` → `train.py` → `triage/linear.py` → `triage/pipeline.py`，即可掌握輸入、特徵、訓練、選型和推論。

`Solution/tests/test_linear.py` 使用人工合成文字驗證詞彙表不受驗證文字更新、兩個分類頭的標籤、模型存取一致性、錯誤碼優先、模型完整性，以及輸入夾帶答案不影響預測。本次僅新增導讀文件，未執行測試；上述內容是既有測試涵蓋範圍，不代表本次執行結果。
