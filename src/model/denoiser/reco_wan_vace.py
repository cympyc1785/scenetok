"""ReCo (Wan2.1 VACE 1.3B) denoiser with a LightningDiT ControlNet branch.

핵심 아이디어 (plan: bubbly-floating-snail):
  - main denoiser = ReCo = Wan2.1 1.3B DiT + VACE (ReCo pretrained LoRA + 추가 LoRA fine-tune)
  - ctrl branch = LightningDiT (SceneTok rectified flow decoder, va-wan_dl3dv ckpt warm-start)
    가 scene token + camera로부터 비디오 latent을 생성 → `ldt2reco_proj`(48→16, zero-init)로
    Wan2.1 latent 도메인에 매핑 → ReCo VACE의 ref slot(=vace_context의 source latent)에 주입.
  - ReCo width-doubled 출력: 왼쪽=recon(background), 오른쪽=dynamic insertion.
    loss는 ReCo 출력에만 (wrapper에서 좌/우 split rectified-flow loss).
  - LightningDiT는 **trainable** (joint end-to-end): ReCo 출력 loss가 ctrl branch까지 역전파.

메인 DiffSynth-Studio(diffsynth)가 VACE + 1.3B WanModel을 모두 지원하므로 ReCo의 vendored
diffsynth를 따로 import하지 않는다 (충돌 회피). DiT/VACE/text-encoder 모두 메인 diffsynth로 로드.
"""
from __future__ import annotations

import sys
import glob
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import torch
from torch import nn, Tensor
from jaxtyping import Float

DIFFSYNTH_ROOT = Path(__file__).resolve().parents[1] / "DiffSynth-Studio"
if str(DIFFSYNTH_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFSYNTH_ROOT))

from diffsynth.core import load_state_dict
from diffsynth.models.model_loader import ModelPool
from diffsynth.models.wan_video_text_encoder import HuggingfaceTokenizer
from diffsynth.pipelines.wan_video import model_fn_wan_video
from peft import LoraConfig, inject_adapter_in_model

from .denoiser import Denoiser
from ..types import DenoiserInputs, CameraInputs
from ..camera import CameraCfg

from colorama import Fore
def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


@dataclass
class RecoLoRACfg:
    enabled: bool = True
    rank: int = 128
    alpha: int | None = 128
    target_modules: str = "q,k,v,o,ffn.0,ffn.2"
    checkpoint: str | Path | None = (
        "src/model/ReCo/checkpoints/ReCo/2026_01_16_v1_release.ckpt"
    )


@dataclass
class RecoWanVace1_3BCfg:
    name: Literal["reco_wan_vace_1_3b"]
    camera: CameraCfg | None = None
    model_root: str | Path = Path("src/model/ReCo/checkpoints/Wan2.1-VACE-1.3B")
    dit_pattern: str = "diffusion_pytorch_model*.safetensors"
    text_encoder_path: str | Path | None = None   # default: <model_root>/models_t5_umt5-xxl-enc-bf16.pth
    tokenizer_path: str | Path | None = None       # default: <model_root>/google/umt5-xxl
    seq_len: int = 512
    clean: str = "whitespace"
    gradient_checkpointing: bool = True
    # ── LightningDiT ctrl branch (warm-start from va-wan_dl3dv ckpt) ──────────
    lightningdit_ckpt_path: str | Path | None = "checkpoints/va-wan_dl3dv_256-480.ckpt"
    lightningdit_hidden_size: int = 1024
    lightningdit_num_heads: int = 16
    lightningdit_mlp_ratio: float = 4.0
    lightningdit_in_channels: int = 48     # va-wan_dl3dv LightningDiT는 Wan2.2 48ch
    # ReCo VACE latent (Wan2.1) 채널 — ldt 출력(48ch)을 이리로 매핑.
    reco_latent_channels: int = 16
    # ── conditioning routing (호환용; ReCo는 항상 LightningDiT ctrl 사용) ─────
    scene_input_type: Literal["none", "controlnet"] = "controlnet"
    num_target_split: int = 1
    input_shape: int | list[int] = field(default_factory=lambda: [30, 52])
    noise_seed: int | None = None
    vace_scale: float = 1.0
    # ReCo 출력 좌/우 split loss 가중치 (wrapper에서 사용). left=recon, right=dynamic.
    recon_loss_weight: float = 1.0
    ckpt_path: str | Path | None = None
    load_strict: bool = False
    lora: RecoLoRACfg = field(default_factory=RecoLoRACfg)


class RecoWanVace1_3BDenoiser(Denoiser[RecoWanVace1_3BCfg]):
    def __init__(
        self,
        cfg: RecoWanVace1_3BCfg,
        cond_dim: int | None = 64,
        num_scene_tokens: int = 1,
        temporal_downsample: int = 1,
        using_wan: bool = True,
        **_: object,
    ) -> None:
        super().__init__(cfg)
        self.model_root = Path(cfg.model_root)
        self.temporal_downsample = int(temporal_downsample)
        self.cond_dim = 64 if cond_dim is None else int(cond_dim)
        self.num_scene_tokens = int(num_scene_tokens)

        # 1) ReCo DiT (Wan2.1 1.3B) + VACE 로드 (단일 safetensors에 둘 다 포함)
        dit_paths = self._resolve_dit_paths()
        pool = ModelPool()
        pool.auto_load_model(dit_paths)
        self.model = pool.fetch_model("wan_video_dit")
        self.vace = pool.fetch_model("wan_video_vace")
        if self.model is None or self.vace is None:
            raise RuntimeError(f"Failed to load wan_video_dit / wan_video_vace from {dit_paths}")
        print(cyan(f"(ReCo) DiT in_dim={self.model.in_dim} dim={self.model.dim} "
                   f"blocks={len(self.model.blocks)} | VACE blocks={len(self.vace.vace_blocks)}"))
        self.supports_scene_tokens = True
        self.supports_condition_latents = True
        self.uses_internal_text_encoder = True

        # 2) text encoder
        self.text_encoder = self._load_model("wan_video_text_encoder", self._resolve_text_encoder_path())
        self.tokenizer = HuggingfaceTokenizer(
            name=str(self._resolve_tokenizer_path()), seq_len=cfg.seq_len, clean=cfg.clean
        )
        self.negative_prompt = None

        # 3) LoRA inject (DiT + VACE) + ReCo pretrained LoRA 로드
        if cfg.lora.enabled:
            self._enable_lora(self.model, cfg.lora, prefix="(DiT)")
            self._enable_lora(self.vace, cfg.lora, prefix="(VACE)")
            if cfg.lora.checkpoint is not None:
                self._load_reco_lora(str(cfg.lora.checkpoint))

        # 4) LightningDiT ctrl branch (warm-start) + ldt2reco_proj (zero-init)
        ref_param = next(self.model.parameters())
        self._build_lightningdit_branch(cfg, num_scene_tokens, temporal_downsample, ref_param)
        self.ldt2reco_proj = nn.Conv3d(
            cfg.lightningdit_in_channels, cfg.reco_latent_channels, kernel_size=1
        ).to(device=ref_param.device, dtype=ref_param.dtype)
        nn.init.zeros_(self.ldt2reco_proj.weight)
        nn.init.zeros_(self.ldt2reco_proj.bias)

        # logging / loss용 stash
        self._last_ldt_pred = None     # LightningDiT 출력 latent (logging)
        self._last_ref_latent = None   # projector 후 ref slot latent

        self._set_trainable_parameters()

    # ─────────────────────── builders / loaders ───────────────────────
    def _resolve_dit_paths(self) -> list[str]:
        paths = sorted(glob.glob(str(self.model_root / self.cfg.dit_pattern)))
        if not paths:
            raise FileNotFoundError(f"No ReCo DiT weights under {self.model_root / self.cfg.dit_pattern}")
        return paths

    def _resolve_text_encoder_path(self) -> str:
        if self.cfg.text_encoder_path is not None:
            return str(self.cfg.text_encoder_path)
        return str(self.model_root / "models_t5_umt5-xxl-enc-bf16.pth")

    def _resolve_tokenizer_path(self) -> Path:
        if self.cfg.tokenizer_path is not None:
            return Path(self.cfg.tokenizer_path)
        return self.model_root / "google" / "umt5-xxl"

    def _load_model(self, model_name: str, path) -> nn.Module:
        pool = ModelPool()
        pool.auto_load_model(path)
        model = pool.fetch_model(model_name)
        if model is None:
            raise RuntimeError(f"Failed to load `{model_name}` from {path}.")
        return model

    def _enable_lora(self, module: nn.Module, lora_cfg: RecoLoRACfg, prefix: str = "") -> None:
        alpha = lora_cfg.alpha if lora_cfg.alpha is not None else lora_cfg.rank
        targets = [m.strip() for m in lora_cfg.target_modules.split(",") if m.strip()]
        inject_adapter_in_model(
            LoraConfig(r=lora_cfg.rank, lora_alpha=alpha, init_lora_weights=True, target_modules=targets),
            module,
        )
        n = sum(1 for name, _ in module.named_parameters() if "lora_" in name)
        print(cyan(f"(ReCo LoRA) {prefix} injected {n} lora params (rank={lora_cfg.rank})"))

    def _load_reco_lora(self, path: str) -> None:
        sd = load_state_dict(path) if path.endswith(".safetensors") else torch.load(path, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        vace_sd = {k: v for k, v in sd.items() if k.startswith("vace")}
        dit_sd = {k: v for k, v in sd.items() if not k.startswith("vace")}
        m_d, u_d = self.model.load_state_dict(dit_sd, strict=False)
        m_v, u_v = self.vace.load_state_dict(vace_sd, strict=False)
        print(cyan(f"(ReCo LoRA) loaded DiT {len(dit_sd) - len(u_d)}/{len(dit_sd)} "
                   f"(unexpected={len(u_d)}), VACE {len(vace_sd) - len(u_v)}/{len(vace_sd)} "
                   f"(unexpected={len(u_v)})"))

    def _build_lightningdit_branch(self, cfg, num_scene_tokens, temporal_downsample, ref_param) -> None:
        from .lightningdit import LightningDiT, LightningDiTCfg, LightningDiTKwargsCfg

        litdit_kwargs = LightningDiTKwargsCfg(
            patch_size=1,
            in_channels=int(cfg.lightningdit_in_channels),
            hidden_size=int(cfg.lightningdit_hidden_size),
            depth=24,
            num_heads=int(cfg.lightningdit_num_heads),
            mlp_ratio=float(cfg.lightningdit_mlp_ratio),
            class_dropout_prob=0.0,
            num_classes=1000,
            learn_sigma=False,
            use_qknorm=True,
            use_swiglu=True,
            use_rope=False,
            use_rope_3d=True,
            use_rmsnorm=True,
            wo_shift=False,
            frequency_embedding_size=256,
        )
        ld_camera_cfg = deepcopy(cfg.camera)
        ld_emb_cfg = getattr(ld_camera_cfg, "embedding", None) if ld_camera_cfg is not None else None
        if temporal_downsample > 1 and ld_emb_cfg is not None and getattr(ld_emb_cfg, "name", None) == "time_embed":
            ld_emb_cfg.in_channels *= temporal_downsample
        litdit_cfg = LightningDiTCfg(
            name="lightningdit",
            camera=ld_camera_cfg,
            kwargs=litdit_kwargs,
            single_dim_tokens=False,
            num_target_split=int(cfg.num_target_split),
            camera_conditioning="add",
            input_shape=list(cfg.input_shape) if not isinstance(cfg.input_shape, int) else [cfg.input_shape, cfg.input_shape],
            gradient_checkpointing=True,
            pretrained_from=None,
            ckpt_path=str(cfg.lightningdit_ckpt_path) if cfg.lightningdit_ckpt_path is not None else None,
            load_strict=False,
            causal_attention=False,
            text_cond_dim=None,
        )
        self.ldt_branch = LightningDiT(
            cfg=litdit_cfg,
            cond_dim=self.cond_dim,
            num_scene_tokens=int(num_scene_tokens),
            num_views=int(getattr(cfg, "num_target_split", 1)) or 1,
            temporal_downsample=int(temporal_downsample),
            using_wan=True,
            cfg_train=False,
        ).to(device=ref_param.device, dtype=ref_param.dtype)
        if cfg.lightningdit_ckpt_path is not None:
            print(cyan(f"(ReCo) LightningDiT ctrl branch warm-started from {cfg.lightningdit_ckpt_path}"))

    def _set_trainable_parameters(self) -> None:
        # 전체 freeze 후, trainable: ReCo LoRA(DiT+VACE) + LightningDiT 전체 + projector.
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.vace.parameters():
            p.requires_grad = False
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        for name, p in self.model.named_parameters():
            if "lora_" in name:
                p.requires_grad = True
        for name, p in self.vace.named_parameters():
            if "lora_" in name:
                p.requires_grad = True
        for p in self.ldt_branch.parameters():
            p.requires_grad = True
        for p in self.ldt2reco_proj.parameters():
            p.requires_grad = True

        groups: dict[str, int] = {}
        for name, p in self.named_parameters():
            if p.requires_grad:
                key = name.split(".")[0] + ("/lora" if "lora_" in name else "")
                groups[key] = groups.get(key, 0) + p.numel()
        total = sum(groups.values())
        print(cyan("\n[ReCoWanVace] Trainable groups:"))
        for k, v in sorted(groups.items()):
            print(cyan(f"  - {k}: {v:,}"))
        print(cyan(f"[ReCoWanVace] Total trainable: {total:,}\n"))

    # ─────────────────────── forward ───────────────────────
    def load_weights(self, path, **kwargs):
        # ckpt_path는 wrapper의 load_state_dict 경로로 처리됨 (별도 no-op).
        return

    def _forward(self, inputs: DenoiserInputs, **kwargs) -> Float[Tensor, "batch view channel height width"]:
        # NOTE: 실제 forward(vace_context 96ch 구성 + width-doubled split)는 다음 단계(#33)에서
        # 구현. 현 단계는 loader/branch 구성 검증용 — 호출 시 명확히 막아둔다.
        raise NotImplementedError(
            "RecoWanVace1_3BDenoiser._forward는 아직 미구현 (loader/branch 검증 단계). "
            "다음 단계에서 vace_context 구성 + ReCo forward + width split을 추가."
        )
