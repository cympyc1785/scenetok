
from .lightningdit import LightningDiT, LightningDiTCfg
from .wan_ti2v import WanTI2V5BCfg, WanTI2V5BDenoiser
from .wan_t2v_14B import WanT2V14BCfg, WanT2V14BDenoiser


DenoiserCfg = LightningDiTCfg | WanTI2V5BCfg | WanT2V14BCfg
Denoiser = LightningDiT | WanTI2V5BDenoiser | WanT2V14BDenoiser

DENOISER = {
    "lightningdit": LightningDiT,
    "wan_ti2v_5b": WanTI2V5BDenoiser,
    "wan_t2v_14b": WanT2V14BDenoiser,
}


def get_denoiser(
    denoiser_cfg: DenoiserCfg,
    **kwargs
) -> Denoiser:
    return DENOISER[denoiser_cfg.name](denoiser_cfg, **kwargs)


