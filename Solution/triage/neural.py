"""Small neural heads and local-only BGE checkpoint inference."""
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from .embedding import BgeClassifier, BgeEncoder
from .io import file_sha256

CLASSES = ("noise", "temperature", "power", "software", "other")


class NeuralHeads(torch.nn.Module):
    def __init__(self, dimension, dropout):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.category = torch.nn.Linear(dimension, len(CLASSES))
        self.information = torch.nn.Linear(dimension, 2)

    def forward(self, vectors):
        vectors = self.dropout(vectors)
        return self.category(vectors), self.information(vectors)


class HeadAdapter:
    def __init__(self, layer, labels):
        self.layer, self.labels = layer, labels

    def predict(self, features):
        with torch.inference_mode():
            logits = self.layer(torch.as_tensor(features.copy(), dtype=torch.float32))
        return [self.labels[i] for i in logits.argmax(-1).tolist()]


class NeuralBgeClassifier(BgeClassifier):
    evidence_label = "BGE 神經分類頭（checkpoint）"

    @classmethod
    def load(cls, directory: Path, codes: dict):
        metadata = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        if metadata.get("format") != "bge-neural-v1" or metadata.get("classes") != list(CLASSES):
            raise ValueError("BGE 神經模型格式不符")
        if file_sha256(directory / "heads.safetensors") != metadata["heads_sha256"]:
            raise ValueError("分類頭雜湊不符")
        encoder = BgeEncoder(directory / "encoder", max_length=metadata["settings"]["max_length"])
        heads = NeuralHeads(encoder.dimension, metadata["settings"]["dropout"])
        heads.load_state_dict(load_file(str(directory / "heads.safetensors"), device="cpu"), strict=True)
        heads.eval().requires_grad_(False)
        bundle = {"category_model": HeadAdapter(heads.category, CLASSES),
                  "information_model": HeadAdapter(heads.information, ("unknown", "known")),
                  "mask_device_names": False}
        classifier = cls(bundle, codes, encoder, batch_size=metadata["settings"]["batch_size"])
        return classifier, metadata
