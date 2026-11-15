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
