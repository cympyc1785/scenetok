
from .plucker import Plucker, PluckerCfg
from .ray import Ray, RayCfg
from .wan_plucker import WanPlucker, WanPluckerCfg

from dataclasses import dataclass
from jaxtyping import Float
from torch import Tensor


CameraCfg = PluckerCfg | RayCfg | WanPluckerCfg
Camera = Plucker | Ray | WanPlucker

CAMERA = {
    "plucker": Plucker,
    "ray": Ray,
    "wan_plucker": WanPlucker,
}

def get_camera(cfg: CameraCfg, **kwargs) -> Camera:

    return CAMERA[cfg.name](cfg, **kwargs)
