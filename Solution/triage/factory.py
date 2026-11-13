"""Lazy model loading preserves the dependency-free rules entry point."""
from pathlib import Path

from .classification import RuleClassifier

METHODS = ("rules", "tfidf-lr", "tfidf-svc")


def build_classifier(method: str, codes: dict, rules: list[dict], model_path: Path | None = None):
    if method == "rules":
        return RuleClassifier(rules, codes), {"scheme": "A-rules-v1"}, []
    if method not in METHODS:
        raise ValueError(f"不支援的方法：{method}")
    path = model_path or Path(__file__).resolve().parents[1] / "models" / f"{method}.joblib"
    if not path.is_file() or not path.with_suffix(".json").is_file():
        raise ValueError("尚未建立模型；請先以 .venv/bin/python Solution/train.py 訓練 history")
    try:
        from .linear import LinearTextClassifier
        classifier, metadata = LinearTextClassifier.load(path, codes, method)
    except ModuleNotFoundError as exc:
        raise ValueError("模型方法需要 ML 依賴；請使用 .venv/bin/python 並安裝 Solution/requirements-ml.txt") from exc
    return classifier, {"scheme": f"B-{method}", "model": metadata}, [path, path.with_suffix(".json")]
