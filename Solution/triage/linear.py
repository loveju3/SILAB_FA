"""History-trained TF-IDF category model and independent information gate."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .classification import CategoryMatch
from .text import error_codes


class LinearTextClassifier:
    def __init__(self, bundle: dict, codes: dict):
        self.vectorizer = bundle["vectorizer"]
        self.category_model = bundle["category_model"]
        self.information_model = bundle["information_model"]
        self.mask_device_names = bundle["mask_device_names"]
        self.codes = codes

    def classify(self, text: str) -> CategoryMatch:
        known = [self.codes[c]["category"] for c in error_codes(text) if c in self.codes]
        if known and len(set(known)) == 1:
            return CategoryMatch(known[0], "已知錯誤碼")
        features = self.vectorizer.transform([text])
        if self.information_model.predict(features)[0] == "unknown":
            return CategoryMatch(None, "history 訓練的資訊不足判斷")
        return CategoryMatch(str(self.category_model.predict(features)[0]), "字元 TF-IDF 線性分類")

    @classmethod
    def load(cls, path: Path, codes: dict, method: str):
        import joblib
        import sklearn

        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if metadata.get("method") != method:
            raise ValueError("模型的方法與 --method 不一致")
        if metadata.get("sklearn_version") != sklearn.__version__:
            raise ValueError("scikit-learn 版本不同，請依 requirements-ml.txt 安裝並重新訓練")
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata.get("artifact_sha256"):
            raise ValueError("模型雜湊不符，請重新訓練")
        # Only load locally generated, trusted artifacts. Metadata/hash integrity
        # is not a trust boundary for arbitrary pickle/joblib files.
        bundle = joblib.load(path)
        return cls(bundle, codes), metadata


def save_model(path: Path, bundle: dict, metadata: dict) -> dict:
    import joblib
    import sklearn

    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".joblib")
    os.close(handle)
    try:
        joblib.dump(bundle, temporary, compress=3)
        digest = hashlib.sha256(Path(temporary).read_bytes()).hexdigest()
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    metadata = {**metadata, "sklearn_version": sklearn.__version__, "artifact_sha256": digest,
                "artifact_bytes": path.stat().st_size}
    from .io import atomic_write
    atomic_write(path.with_suffix(".json"), json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    return metadata
