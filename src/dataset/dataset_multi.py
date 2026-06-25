"""MultiDataset — weighted mix of several sub-datasets (e.g. DL3DV + DynamicVerse)
for one training run.

Each sub-dataset keeps its OWN view_sampler and config (DL3DV→bounded,
DynamicVerse→caption_window); this wrapper only routes `__getitem__` to a
weighted-randomly chosen sub-dataset (sample-level mixing). All sub-datasets MUST
emit the same batch schema (keys + shapes) so a mixed batch collates — enforce
matching num_context/num_target/shapes and `text` always present (DL3DV:
force_empty_text=true).

Construction is special-cased in `get_dataset` (which has step_tracker/generator):
it builds each sub via `get_dataset(sub_cfg, ...)` then wraps them here.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from .dtypes import Stage
from .view_sampler import ViewSamplerCfg


@dataclass
class MultiDatasetCfg:
    name: Literal["multi"]
    # `view_sampler` here is NOT used for sampling (each sub-dataset uses its own).
    # It only exposes the SHARED temporal layout the wrapper reads
    # (`dataset_cfg.view_sampler.num_target_views/offset/...`). Set it to match
    # both subs (e.g. caption_window, num_target_views=10, offset=1).
    view_sampler: ViewSamplerCfg
    # Wrapper also reads these directly:
    fps: int = 24
    precomputed_latents: dict = field(default_factory=lambda: {"context": False, "target": False})
    # Raw sub-dataset config dicts (each has its own `name` + `view_sampler`).
    # Parsed per-name in `get_dataset` (avoids dacite union-in-list ambiguity).
    datasets: list[Any] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    # Nominal train epoch length (sample-level random mixing ignores idx).
    length: int = 100_000


class MultiDataset(Dataset):
    def __init__(self, cfg: MultiDatasetCfg, sub_datasets: list[Dataset],
                 stage: Stage, force_shuffle: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        self.subs = sub_datasets
        self.stage = stage
        w = np.asarray(cfg.weights if cfg.weights else [1.0] * len(sub_datasets), dtype=np.float64)
        assert len(w) == len(sub_datasets), f"weights {len(w)} != datasets {len(sub_datasets)}"
        self.w = w / w.sum()
        self._rng: np.random.RandomState | None = None
        print(f"(MultiDataset) {len(self.subs)} sub-datasets, weights={self.w.round(3).tolist()}, "
              f"lens={[len(s) for s in self.subs]}, stage={stage}")

    def _rng_(self) -> np.random.RandomState:
        # Per-worker RNG so different workers draw different samples.
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            seed = int(info.seed % (2 ** 31)) if info is not None else 0
            self._rng = np.random.RandomState(seed)
        return self._rng

    def __len__(self) -> int:
        if self.stage == "train":
            return self.cfg.length
        return sum(len(s) for s in self.subs)

    def __getitem__(self, idx: int):
        if self.stage == "train":
            rng = self._rng_()
            di = int(rng.choice(len(self.subs), p=self.w))
            s = self.subs[di]
            return s[int(rng.randint(len(s)))]
        # val/test: deterministic concat indexing
        for s in self.subs:
            if idx < len(s):
                return s[idx]
            idx -= len(s)
        raise IndexError(idx)
