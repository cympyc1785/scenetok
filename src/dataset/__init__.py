

from torch import Generator
from torch.utils.data import Dataset

# Custom modules
from .dtypes import Stage
from .view_sampler import get_view_sampler
from ..misc.step_tracker import StepTracker
from .dataset_re10k import DatasetRE10k, DatasetRE10kCfg
from .dataset_dl3dv import DatasetDL3DV, DatasetDL3DVCfg
from .dataset_re10k import DatasetRE10k, DatasetRE10kCfg
from .dataset_latent import DatasetLatent, DatasetLatentCfg
from .dataset_davis import DatasetDAVIS, DatasetDAVISCfg
from .dataset_dynamicverse import DatasetDynamicverse, DatasetDynamicverseCfg
from .dataset_multi import MultiDataset, MultiDatasetCfg



DATASETS: dict[str, Dataset] = {
    "re10k": DatasetRE10k,
    "dl3dv": DatasetDL3DV,
    "latent": DatasetLatent,
    "davis": DatasetDAVIS,
    "dynamicverse": DatasetDynamicverse,
}

# name → typed cfg dataclass, for manually parsing MultiDataset sub-configs
# (avoids dacite union-in-list ambiguity; sub dicts are parsed here per-name).
DATASET_CFG_BY_NAME = {
    "re10k": DatasetRE10kCfg,
    "dl3dv": DatasetDL3DVCfg,
    "latent": DatasetLatentCfg,
    "davis": DatasetDAVISCfg,
    "dynamicverse": DatasetDynamicverseCfg,
}


DatasetCfg = (
    DatasetDL3DVCfg
    | DatasetLatentCfg
    | DatasetRE10kCfg
    | DatasetDAVISCfg
    | DatasetDynamicverseCfg
    | MultiDatasetCfg
)


def _parse_sub_cfg(d):
    """Parse a raw MultiDataset sub-config dict → typed DatasetCfg (by `name`)."""
    from pathlib import Path
    from dacite import Config, from_dict
    from omegaconf import OmegaConf, DictConfig
    if isinstance(d, DictConfig):
        d = OmegaConf.to_container(d, resolve=True)
    name = d["name"]
    return from_dict(DATASET_CFG_BY_NAME[name], d, config=Config(type_hooks={Path: Path}))


def get_dataset(
    cfg: DatasetCfg,
    stage: Stage,
    step_tracker: StepTracker | None,
    generator: Generator | None = None,
    force_shuffle: bool = False
) -> Dataset:

    if cfg.name == "multi":
        subs = [
            get_dataset(_parse_sub_cfg(d), stage, step_tracker, generator, force_shuffle)
            for d in cfg.datasets
        ]
        return MultiDataset(cfg, subs, stage, force_shuffle)

    view_sampler = get_view_sampler(
        cfg.view_sampler,
        stage,
        cfg.cameras_are_circular,
        step_tracker,
        generator
    )
    return DATASETS[cfg.name](cfg, stage, view_sampler, force_shuffle)
