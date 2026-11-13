"""Conservative, bounded settings shared by the UI and offline training worker."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class FineTuneSettings:
    epochs: int = 5
    encoder_lr: float = 1e-5
    head_lr: float = 1e-3
    weight_decay: float = .01
    dropout: float = .2
    label_smoothing: float = .1
    patience: int = 2
    batch_size: int = 8
    max_length: int = 512
    seed: int = 42

    @classmethod
    def from_dict(cls, values=None):
        values = {} if values is None else values
        if not isinstance(values, dict) or set(values) - set(asdict(cls())):
            raise ValueError("不支援的微調設定")
        result = cls(**values)
        ranges = {"epochs": (1, 10), "encoder_lr": (1e-6, 5e-5), "head_lr": (1e-4, .01),
                  "weight_decay": (.01, .3), "dropout": (.1, .5), "label_smoothing": (0, .2),
                  "patience": (1, 3), "batch_size": (1, 16), "seed": (0, 2**31 - 1)}
        for name, (low, high) in ranges.items():
            value = getattr(result, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} 必須介於 {low} 與 {high}")
            if name in ("epochs", "patience", "batch_size", "seed") and type(value) is not int:
                raise ValueError(f"{name} 必須為整數")
        if type(result.max_length) is not int or result.max_length != 512:
            raise ValueError("比較固定使用 max_length=512，以維持相同文字範圍")
        return result
