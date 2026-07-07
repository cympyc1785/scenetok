

from typing import Union

from .mvae_compressor import MVAECompressor, MVAECompressorCfg
from .lagernvs_compressor import LagerNVSCompressor, LagerNVSCompressorCfg


COMPRESSOR = {
    "mvae_compressor": MVAECompressor,
    "lagernvs_compressor": LagerNVSCompressor,
}

Compressor = MVAECompressor
# Discriminated (by `name` Literal) union so dacite can type both compressor
# families on `model.compressor`. Order doesn't matter — the `name` Literal
# disambiguates which member validates.
CompressorCfg = Union[MVAECompressorCfg, LagerNVSCompressorCfg]


def get_compressor(
    cfg: CompressorCfg,
    **kwargs
) -> Compressor:

    return COMPRESSOR[cfg.name](cfg, **kwargs)