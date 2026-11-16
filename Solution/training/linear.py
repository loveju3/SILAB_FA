"""Fit vocab/IDF and both learned heads strictly on the supplied training rows."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from training.settings import LinearSettings


def fit_heads(features, labels: list[str | None], method: str, settings: LinearSettings | None = None) -> dict:
    settings = settings or LinearSettings()
    if method not in ("tfidf-lr", "tfidf-svc", "bge-lr", "bge-svc"):
        raise ValueError("不支援的分類頭方法")
    if features.shape[0] != len(labels) or len({c for c in labels if c is not None}) < 2:
        raise ValueError("分類訓練至少需要兩類且文字與標籤筆數一致")
    if None not in labels:
        raise ValueError("資訊不足分類器需要 history 的 null 範例")
    weight = None if settings.class_weight == "none" else settings.class_weight
    known = [i for i, label in enumerate(labels) if label is not None]
    category = (LogisticRegression(C=settings.c, class_weight=weight, max_iter=settings.lr_max_iter, random_state=42)
                if method.endswith("-lr") else
                LinearSVC(C=settings.c, class_weight=weight, dual="auto", max_iter=settings.svc_max_iter,
                          random_state=42))
    category.fit(features[known], [labels[i] for i in known])
    information = LogisticRegression(C=settings.information_c, class_weight=weight,
                                     max_iter=settings.lr_max_iter, random_state=42)
    information.fit(features, ["unknown" if label is None else "known" for label in labels])
    return {"category_model": category, "information_model": information}


def fit_bundle(texts: list[str], labels: list[str | None], method: str, masked: bool,
               settings: LinearSettings | None = None) -> dict:
    settings = settings or LinearSettings()
    if method not in ("tfidf-lr", "tfidf-svc") or len(texts) != len(labels):
        raise ValueError("TF-IDF 方法或訓練筆數不合法")
    vectorizer = TfidfVectorizer(analyzer=settings.analyzer, ngram_range=(settings.ngram_min, settings.ngram_max),
                                 lowercase=False, sublinear_tf=settings.sublinear_tf, min_df=settings.min_df,
                                 max_features=settings.max_features or None, norm="l2")
    features = vectorizer.fit_transform(texts)
    return {"vectorizer": vectorizer, **fit_heads(features, labels, method, settings),
            "mask_device_names": masked}
