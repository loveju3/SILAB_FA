"""Conservative, bounded settings shared by the UI and offline training worker."""
from dataclasses import asdict, dataclass, fields
import math
import json
from pathlib import Path
import re


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
        if type(result.epochs) is not int or result.epochs < 1:
            raise ValueError("epochs 必須為大於等於 1 的整數")
        ranges = {"encoder_lr": (1e-6, 5e-5), "head_lr": (1e-4, .01),
                  "weight_decay": (.01, .3), "dropout": (.1, .5), "label_smoothing": (0, .2),
                  "patience": (1, 3), "batch_size": (1, 16), "seed": (0, 2**31 - 1)}
        for name, (low, high) in ranges.items():
            value = getattr(result, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} 必須介於 {low} 與 {high}")
            if name in ("patience", "batch_size", "seed") and type(value) is not int:
                raise ValueError(f"{name} 必須為整數")
        if type(result.max_length) is not int or result.max_length != 512:
            raise ValueError("比較固定使用 max_length=512，以維持相同文字範圍")
        return result


@dataclass(frozen=True)
class ModalSettings:
    """Public cloud resources only; credentials remain managed by the Modal SDK."""
    app_name: str
    gpu: str
    cpu: int
    memory_mb: int
    timeout_seconds: int

    @classmethod
    def load(cls, path=None):
        path = Path(path) if path is not None else Path(__file__).resolve().parents[1] / 'config/modal_finetune.json'
        try:
            values = json.loads(path.read_text(encoding='utf-8-sig'))
        except (OSError, ValueError) as exc:
            raise ValueError(f'Modal 設定檔無法讀取：{path}；請確認檔案存在且為合法 JSON') from exc
        return cls.from_dict(values)

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, dict) or set(values) != {field.name for field in fields(cls)}:
            raise ValueError('Modal 設定僅接受 app_name、gpu、cpu、memory_mb、timeout_seconds，且每項皆必填；不要填入 token')
        result = cls(**values)
        if not isinstance(result.app_name, str) or not re.fullmatch(r'[a-z][a-z0-9-]{2,62}', result.app_name):
            raise ValueError('Modal app_name 須為 3～63 字的小寫英文、數字或連字號，並以英文開頭')
        if result.gpu not in ('T4', 'L4', 'A10', 'L40S', 'A100-40GB', 'A100-80GB'):
            raise ValueError('Modal gpu 請選 T4、L4、A10、L40S、A100-40GB 或 A100-80GB')
        for name, low, high in (('cpu', 1, 8), ('memory_mb', 1024, 65536), ('timeout_seconds', 60, 14400)):
            value = getattr(result, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'Modal {name} 須為 {low}～{high} 的整數')
        return result


@dataclass(frozen=True)
class LinearSettings:
    """TF-IDF／凍結 BGE 的 LR／SVC 訓練參數；預設值即原本固定的配方。

    分類頭欄位兩種訓練都用；TF-IDF 特徵欄位只在 TF-IDF 訓練時使用。
    """
    c: float = 1.0
    class_weight: str = "balanced"
    information_c: float = 1.0
    lr_max_iter: int = 2000
    svc_max_iter: int = 5000
    analyzer: str = "char"
    ngram_min: int = 2
    ngram_max: int = 4
    min_df: int = 1
    max_features: int = 0
    sublinear_tf: bool = True

    HEAD_FIELDS = ("c", "class_weight", "information_c", "lr_max_iter", "svc_max_iter")
    TFIDF_FIELDS = ("analyzer", "ngram_min", "ngram_max", "min_df", "max_features", "sublinear_tf")

    @classmethod
    def from_dict(cls, values=None, kind="tfidf"):
        values = {} if values is None else values
        allowed = cls.HEAD_FIELDS + (cls.TFIDF_FIELDS if kind == "tfidf" else ())
        if not isinstance(values, dict) or set(values) - set(allowed):
            raise ValueError("不支援的 LR／SVC 訓練設定" + ("；凍結 BGE 不使用 TF-IDF 特徵設定" if kind != "tfidf" else ""))
        result = cls(**values)
        for name, (low, high) in {"c": (1e-3, 1e3), "information_c": (1e-3, 1e3)}.items():
            value = getattr(result, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} 必須介於 {low} 與 {high}")
        for name, (low, high) in {"lr_max_iter": (100, 100000), "svc_max_iter": (100, 100000),
                                  "ngram_min": (1, 6), "ngram_max": (1, 6), "min_df": (1, 50),
                                  "max_features": (0, 1000000)}.items():
            value = getattr(result, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} 必須為 {low}～{high} 的整數")
        if result.class_weight not in ("balanced", "none"):
            raise ValueError("class_weight 只能是 balanced 或 none")
        if result.analyzer not in ("char", "char_wb"):
            raise ValueError("analyzer 只能是 char 或 char_wb")
        if result.ngram_min > result.ngram_max:
            raise ValueError("ngram_min 不可大於 ngram_max")
        if 0 < result.max_features < 100:
            raise ValueError("max_features 填 0 代表不限制，否則至少 100")
        if type(result.sublinear_tf) is not bool:
            raise ValueError("sublinear_tf 必須為 true 或 false")
        return result

    def to_dict(self, kind="tfidf"):
        values = asdict(self)
        return {k: values[k] for k in self.HEAD_FIELDS + (self.TFIDF_FIELDS if kind == "tfidf" else ())}
