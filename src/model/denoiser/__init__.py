
from .lightningdit import LightningDiT, LightningDiTCfg
from .wan_ti2v import WanTI2V5BCfg, WanTI2V5BDenoiser


DenoiserCfg = LightningDiTCfg | WanTI2V5BCfg
Denoiser = LightningDiT | WanTI2V5BDenoiser

DENOISER = {
    "lightningdit": LightningDiT,
    "wan_ti2v_5b": WanTI2V5BDenoiser,
}


def get_denoiser(
    denoiser_cfg: DenoiserCfg,
    **kwargs
) -> Denoiser:
    return DENOISER[denoiser_cfg.name](denoiser_cfg, **kwargs)


