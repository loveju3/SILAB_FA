"""Frozen local BGE encoder, CPU only, plus history-trained linear heads."""
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

from .classification import BatchClassifications
from .linear import LinearTextClassifier
from .io import file_sha256


class BgeEncoder:
    def __init__(self, directory: Path, *, threads=2, max_length=512):
        if threads < 1 or not 1 <= max_length <= 512:
            raise ValueError("encoder 執行緒或長度設定不合法")
        self.directory = directory
        self.max_length = max_length
        self.manifest = json.loads((directory / "encoder_manifest.json").read_text(encoding="utf-8"))
        for name, expected in self.manifest["files_sha256"].items():
            if Path(name).name != name or file_sha256(directory / name) != expected:
                raise ValueError("BGE encoder 檔案雜湊不符，請重新下載固定 revision")
        import torch
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        torch.set_num_threads(threads)
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(directory, local_files_only=True, trust_remote_code=False,
                                              use_safetensors=True, attn_implementation="eager")
        self.model.to("cpu").eval().requires_grad_(False)
        self.dimension = self.model.config.hidden_size

    def transform(self, texts: list[str]):
        import numpy as np

        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length,
                                return_tensors="pt")
        with self.torch.inference_mode():
            embeddings = self.model(**inputs).last_hidden_state[:, 0]
            embeddings = self.torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

    def encode(self, texts: list[str], batch_size=16):
        import numpy as np

        if batch_size < 1:
            raise ValueError("batch_size 必須大於 0")
        parts = [self.transform(texts[i:i + batch_size]) for i in range(0, len(texts), batch_size)]
        return np.concatenate(parts) if parts else self.transform([])


class BgeClassifier(LinearTextClassifier):
    evidence_label = "固定 BGE embedding 線性分類"

    def __init__(self, bundle: dict, codes: dict, encoder, batch_size=16):
        if batch_size < 1:
            raise ValueError("batch_size 必須大於 0")
        super().__init__({**bundle, "vectorizer": encoder}, codes)
        self.batch_size = batch_size
        self.timing_description = ("per-ticket seconds includes the FULL shared batch classification service time plus individual rule work; "
                                   "batch time is counted once in process total, so per-ticket seconds must not be summed; "
                                   "excludes model load, shared preparation, sort, validation and output")

    def classify_many(self, texts: list[str]) -> BatchClassifications:
        predictions, seconds = [], []
        for start in range(0, len(texts), self.batch_size):
            began = perf_counter()
            batch = texts[start:start + self.batch_size]
            matches = [self.code_match(text) for text in batch]
            pending = [i for i, match in enumerate(matches) if match is None]
            if pending:
                vectors = self.vectorizer.transform([batch[i] for i in pending])
                for offset, i in enumerate(pending):
                    matches[i] = self.classify_features(vectors[offset:offset + 1])
            duration = perf_counter() - began
            predictions.extend(matches)
            seconds.extend([duration] * len(batch))
        return BatchClassifications(predictions, seconds)

    @classmethod
    def load(cls, path: Path, codes: dict, method: str):
        import joblib
        import sklearn

        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if metadata.get("method") != method or metadata.get("sklearn_version") != sklearn.__version__:
            raise ValueError("BGE 分類頭的方法或 scikit-learn 版本不一致")
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata.get("artifact_sha256"):
            raise ValueError("BGE 分類頭雜湊不符")
        bundle = joblib.load(path)  # Locally trained trusted artifact only.
        name = bundle["encoder_folder"]
        if Path(name).name != name:
            raise ValueError("encoder 須與分類頭放在同一模型資料夾")
        encoder = BgeEncoder(path.parent / name)
        if encoder.manifest["revision"] != metadata["encoder"]["revision"]:
            raise ValueError("encoder revision 與分類頭訓練版本不同")
        return cls(bundle, codes, encoder, bundle["batch_size"]), metadata
