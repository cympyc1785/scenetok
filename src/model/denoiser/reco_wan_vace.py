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

from einops import rearrange

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
    # ── 내부 보유 background VAE (Wan2.2 48ch) — ldt branch 입력 인코딩용 ──────
    bg_vae_ckpt_path: str | Path = "checkpoints/Wan2.2_VAE.pth"
    bg_latent_channels: int = 48
    bg_scaling_factor: float = 1.0
    # ── conditioning routing (호환용; ReCo는 항상 LightningDiT ctrl 사용) ─────
    scene_input_type: Literal["none", "controlnet"] = "controlnet"
    # `_derive_camera_shapes`가 latent-domain ray로 인식하도록 (camera.input_shape
    # = latent_shape). ReCo 내부에선 routing에 안 쓰이고 ldt branch가 latent grid
    # ray를 쓰므로 controlnet_lightningdit으로 둔다.
    camera_input_type: str = "controlnet_lightningdit"
    num_target_split: int = 1
    input_shape: int | list[int] = field(default_factory=lambda: [30, 52])
    noise_seed: int | None = None
    vace_scale: float = 1.0
    # ReCo 출력 좌/우 split loss 가중치 (wrapper에서 사용). left=recon, right=dynamic.
    recon_loss_weight: float = 1.0
    dynamic_loss_weight: float = 1.0
    # True면 ReCo(DiT/VACE LoRA) freeze → LightningDiT ctrl branch + ldt2reco_proj만 학습.
    # recon-우선 phase: freeze_reco=true + dynamic_loss_weight=0 으로 ldt만 recon에 fit.
    freeze_reco: bool = False
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

        # 3.5) 내부 background VAE (Wan2.2 48ch, frozen) — ldt branch 입력 인코딩
        from src.model.autoencoder.autoencoder_wan import AutoencoderWan, WanKwargsCfg
        self.bg_vae = AutoencoderWan(WanKwargsCfg(
            latent_channels=cfg.bg_latent_channels, scaling_factor=cfg.bg_scaling_factor
        ))
        self.bg_vae.from_pretrained(str(cfg.bg_vae_ckpt_path))
        for p in self.bg_vae.parameters():
            p.requires_grad = False
        self.bg_scaling_factor = float(cfg.bg_scaling_factor)

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

    def encode_text_condition(self, text, device: torch.device):
        """T5 text encoder → raw context (B, L, 4096). model_fn_wan_video가 내부에서
        `dit.text_embedding`(4096→1536)을 적용하므로 여기선 projection 하지 않는다."""
        if text is None:
            text = ""
        ids, mask = self.tokenizer(text, return_mask=True)
        ids, mask = ids.to(device), mask.to(device)
        with torch.no_grad():
            text_state = self.text_encoder(ids, mask)
            seq_lens = mask.gt(0).sum(dim=1).long()
            for i, v in enumerate(seq_lens):
                text_state[i, v:] = 0
        return text_state.to(dtype=next(self.model.parameters()).dtype)

    @torch.no_grad()
    def encode_bg(self, frames: Tensor) -> Tensor:
        """inpaint_result frames (B, V, 3, H, W) → Wan2.2 48ch background latent (B, V, 48, h, w)."""
        vae_dtype = next(self.bg_vae.parameters()).dtype
        out_dtype = frames.dtype
        lat = self.bg_vae.encode(frames.to(dtype=vae_dtype))
        if self.bg_scaling_factor != 1.0:
            lat = lat * self.bg_scaling_factor
        return lat.to(dtype=out_dtype)

    def _set_trainable_parameters(self) -> None:
        # 전체 freeze 후, trainable: ReCo LoRA(DiT+VACE) + LightningDiT 전체 + projector.
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.vace.parameters():
            p.requires_grad = False
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        for p in self.bg_vae.parameters():
            p.requires_grad = False
        if not self.cfg.freeze_reco:                        # recon-우선 phase면 ReCo LoRA도 freeze
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

    def _build_vace_context(self, ref_latent: Tensor) -> Tensor:
        """ref_latent (B, 16, F, H, W_half) → ReCo vace_context (B, 96, F, H, 2*W_half).

        plan: 좌=recon(background render, mask 0 → 보존/복원), 우=dynamic(mask 1 → 생성).
          inactive = [ref_render | zeros]     (preserve 영역에 LightningDiT render)
          reactive = zeros                    (edit 영역 원본 content 없음 → 0)
          mask(64ch) = [0(좌) | 1(우)]         (latent 도메인에서 직접 구성)
          vace_context = [inactive(16) | reactive(16) | mask(64)] = 96ch
        """
        b, c, f, h, wh = ref_latent.shape
        zeros = torch.zeros_like(ref_latent)
        inactive = torch.cat([ref_latent, zeros], dim=-1)          # (B,16,F,H,2W)
        reactive = torch.zeros_like(inactive)                      # (B,16,F,H,2W)
        video_lat = torch.cat([inactive, reactive], dim=1)         # (B,32,F,H,2W)
        mask = torch.zeros((b, 64, f, h, 2 * wh), dtype=ref_latent.dtype, device=ref_latent.device)
        mask[..., wh:] = 1.0                                       # 우측(dynamic) 생성 영역
        return torch.cat([video_lat, mask], dim=1)                 # (B,96,F,H,2W)

    def _forward(self, inputs: DenoiserInputs, temporal_downsample: int = 1,
                 chunk_targets: bool = False, **kwargs) -> Float[Tensor, "batch view channel height width"]:
        """ReCo(Wan2.1 VACE) + LightningDiT ctrl branch, same-t co-sampling.

        기대 입력 (wrapper가 채움):
          inputs.view             : (B, V, 16, H, 2W) ReCo width-doubled noisy latent (좌 recon / 우 dynamic)
          inputs.condition_latents: (B, V, 48, H, W)  background(inpaint_result) va-wan noisy latent (ldt용, same t)
          inputs.raw_state        : (B, N, 64)        raw scene tokens (ldt cnd_proj용)
          inputs.text             : (B, L, 4096)      text context embeddings
          inputs.pose, inputs.timestep
        """
        latents = rearrange(inputs.view, "b v c h w -> b c v h w")   # (B,16,F,H,2W)
        timestep = inputs.timestep if inputs.timestep.ndim >= 1 else inputs.timestep.unsqueeze(0)

        # 1) LightningDiT ctrl branch: 48ch background latent을 same-t에서 denoise
        bg = inputs.condition_latents
        if bg is None:
            raise ValueError("RecoWanVace._forward: condition_latents(48ch background latent)가 필요합니다.")
        raw_scene = getattr(inputs, "raw_state", None)
        proj_scene = self.ldt_branch.cnd_proj(raw_scene) if raw_scene is not None else None
        # bg(Wan2.2 VAE /16)와 ldt x_embedder가 기대하는 grid(va-wan ckpt native /8)가 다를 수 있음
        # → ldt img_size로 resize 후 입력.
        ldt_hw = tuple(int(s) for s in self.ldt_branch.model.x_embedder.img_size)
        if bg.shape[-2:] != ldt_hw:
            bf, bv = bg.shape[0], bg.shape[1]
            bg_flat = rearrange(bg, "b v c h w -> (b v) c h w")
            bg_flat = torch.nn.functional.interpolate(bg_flat, size=ldt_hw, mode="bilinear", align_corners=False)
            bg = rearrange(bg_flat, "(b v) c h w -> b v c h w", b=bf, v=bv)
        ctrl_inputs = DenoiserInputs(
            view=bg, pose=inputs.pose, timestep=timestep, state=proj_scene, text=None, condition_latents=None,
        )
        ldt_pred, _ = self.ldt_branch._forward(
            inputs=ctrl_inputs, temporal_downsample=temporal_downsample, chunk_targets=chunk_targets,
        )
        ldt_pred = rearrange(ldt_pred, "b v c h w -> b c v h w")     # (B,48,F,H,W)
        self._last_ldt_pred = ldt_pred                               # logging용

        # 2) 48→16 projector (zero-init) → ReCo VACE source slot
        ref_latent = self.ldt2reco_proj(ldt_pred)                    # (B,16,F,H_ld,W_ld)
        # ldt(Wan2.2 VAE /16) grid ≠ ReCo(Wan2.1 VAE /8) grid → ReCo half-width 해상도로 resize.
        H_r, Wd_r = latents.shape[-2], latents.shape[-1]
        Wh_r = Wd_r // 2
        if ref_latent.shape[-2:] != (H_r, Wh_r):
            bc, cc, fc = ref_latent.shape[:3]
            flat = rearrange(ref_latent, "b c f h w -> (b f) c h w")
            flat = torch.nn.functional.interpolate(flat, size=(H_r, Wh_r), mode="bilinear", align_corners=False)
            ref_latent = rearrange(flat, "(b f) c h w -> b c f h w", b=bc, f=fc)
        self._last_ref_latent = ref_latent

        # 3) vace_context (96ch, width-doubled, ReCo grid)
        vace_context = self._build_vace_context(ref_latent)

        # 4) ReCo VACE + main DiT forward (text context는 wrapper가 인코딩해 inputs.text로 전달)
        context = inputs.text
        # model_fn_wan_video는 1-D timestep(B,) 기대 (seperated_timestep=False). uniform
        # sampling이라 per-view 값이 동일 → 첫 view 값 사용.
        ts_main = timestep if timestep.ndim == 1 else timestep[:, 0]
        pred = model_fn_wan_video(
            dit=self.model,
            vace=self.vace,
            latents=latents,
            timestep=ts_main,
            context=context,
            vace_context=vace_context,
            vace_scale=float(self.cfg.vace_scale),
            use_gradient_checkpointing=self.cfg.gradient_checkpointing and self.training,
        )
        return rearrange(pred, "b c v h w -> b v c h w")             # (B,V,16,H,2W)
