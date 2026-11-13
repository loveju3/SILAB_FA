"""Fit vocab/IDF and both learned heads strictly on the supplied training rows."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def fit_bundle(texts: list[str], labels: list[str | None], method: str, masked: bool) -> dict:
    if method not in ("tfidf-lr", "tfidf-svc"):
        raise ValueError("只支援 tfidf-lr 與 tfidf-svc")
    if len(texts) != len(labels) or len({c for c in labels if c is not None}) < 2:
        raise ValueError("分類訓練至少需要兩類且文字與標籤筆數一致")
    if None not in labels:
        raise ValueError("資訊不足分類器需要 history 的 null 範例")
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), lowercase=False,
                                 sublinear_tf=True, min_df=1, norm="l2")
    features = vectorizer.fit_transform(texts)
    known = [i for i, label in enumerate(labels) if label is not None]
    category = (LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=42)
                if method == "tfidf-lr" else
                LinearSVC(C=1.0, class_weight="balanced", dual="auto", max_iter=5000, random_state=42))
    category.fit(features[known], [labels[i] for i in known])
    information = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=42)
    information.fit(features, ["unknown" if label is None else "known" for label in labels])
    return {"vectorizer": vectorizer, "category_model": category, "information_model": information,
            "mask_device_names": masked}
