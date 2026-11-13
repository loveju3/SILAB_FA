"""Lazy model loading preserves the dependency-free rules entry point."""
from pathlib import Path

from .classification import RuleClassifier

METHODS = ("rules", "tfidf-lr", "tfidf-svc", "bge-lr", "bge-svc", "bge-finetuned")


def build_classifier(method: str, codes: dict, rules: list[dict], model_path: Path | None = None):
    if method == "rules":
        return RuleClassifier(rules, codes), {"scheme": "A-rules-v1"}, []
    if method not in METHODS:
        raise ValueError(f"不支援的方法：{method}")
    if method == "bge-finetuned":
        if model_path is None or not (model_path / "model.json").is_file():
            raise ValueError("BGE 神經模型須指定 --model-path，指向訓練工作中的 finetuned 或 frozen-neural 資料夾")
        try:
            from .neural import NeuralBgeClassifier
            classifier, metadata = NeuralBgeClassifier.load(model_path, codes)
        except ModuleNotFoundError as exc:
            raise ValueError("請安裝 Solution/requirements-embedding.txt 並使用 .venv/bin/python") from exc
        assets = [p for p in model_path.rglob('*') if p.is_file()]
        return classifier, {"scheme": "C-bge-finetuned", "model": metadata}, assets
    path = model_path or Path(__file__).resolve().parents[1] / "models" / f"{method}.joblib"
    if not path.is_file() or not path.with_suffix(".json").is_file():
        script = "Solution/train_embedding.py" if method.startswith("bge-") else "Solution/train.py"
        raise ValueError(f"尚未建立模型；請先以 .venv/bin/python {script} 訓練 history")
    try:
        if method.startswith("bge-"):
            from .embedding import BgeClassifier
            classifier, metadata = BgeClassifier.load(path, codes, method)
        else:
            from .linear import LinearTextClassifier
            classifier, metadata = LinearTextClassifier.load(path, codes, method)
    except ModuleNotFoundError as exc:
        requirements = "requirements-embedding.txt" if method.startswith("bge-") else "requirements-ml.txt"
        raise ValueError(f"模型方法需要依賴；請使用 .venv/bin/python 並安裝 Solution/{requirements}") from exc
    assets = [path, path.with_suffix(".json")]
    if method.startswith("bge-"):
        folder = classifier.vectorizer.directory
        assets.extend(folder / name for name in classifier.vectorizer.manifest["files_sha256"])
        assets.append(folder / "encoder_manifest.json")
    return classifier, {"scheme": f"{'C' if method.startswith('bge-') else 'B'}-{method}", "model": metadata}, assets
