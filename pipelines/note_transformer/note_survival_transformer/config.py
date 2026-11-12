"""One validated, JSON-serializable configuration; no clinical assumptions."""

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
from typing import Tuple


@dataclass
class Config:
    input_dim: int = 0  # inferred from the embedding matrix before model creation
    static_dim: int = 0  # inferred from the fitted static preprocessor
    d_model: int = 64
    n_heads: int = 2
    n_layers: int = 2
    dim_ff: int = 128
    dropout: float = 0.1
    max_notes: int = 128
    time_scale_days: float = 180.0
    landmark_scale_days: float = 180.0
    gap_cap_days: float = 1096.0
    landmarks: Tuple[float, ...] = (0.0,)
    num_epochs: int = 20
    min_epochs: int = 1
    patience: int = 5
    min_delta: float = 1e-4
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 8  # event-centered groups per optimizer step
    risk_set_max_controls: int = 32
    use_ipw: bool = False
    max_grad_norm: float = 1.0
    scheduler: str = "none"  # none or cosine, stepped once per epoch
    eval_batch_size: int = 32
    seed: int = 42
    device: str = "cpu"
    num_threads: int = 1

    def validate(self, require_dimensions=False):
        positive_ints = ("d_model", "n_heads", "n_layers", "dim_ff", "max_notes",
                         "num_epochs", "min_epochs", "patience", "batch_size",
                         "risk_set_max_controls", "eval_batch_size", "num_threads")
        for name in positive_ints:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("input_dim", "static_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if require_dimensions and self.input_dim < 1:
            raise ValueError("input_dim must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        for name in ("time_scale_days", "landmark_scale_days", "gap_cap_days", "lr"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("weight_decay", "max_grad_norm", "min_delta"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        self.landmarks = tuple(float(x) for x in self.landmarks)
        if not self.landmarks or not all(math.isfinite(x) for x in self.landmarks):
            raise ValueError("landmarks must be nonempty and finite")
        if len(set(self.landmarks)) != len(self.landmarks):
            raise ValueError("landmarks must be unique")
        self.landmarks = tuple(sorted(self.landmarks))
        if self.min_epochs > self.num_epochs:
            raise ValueError("min_epochs cannot exceed num_epochs")
        if self.scheduler not in ("none", "cosine"):
            raise ValueError("scheduler must be none or cosine")
        if not isinstance(self.use_ipw, bool):
            raise ValueError("use_ipw must be boolean")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2 ** 32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if self.device != "cpu" and not self.device.startswith("cuda"):
            raise ValueError("device must be cpu or a CUDA device")
        return self

    def to_dict(self):
        result = asdict(self)
        result["landmarks"] = list(self.landmarks)
        return result

    @classmethod
    def from_dict(cls, values):
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        return cls(**values).validate()

    @classmethod
    def from_json(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))
