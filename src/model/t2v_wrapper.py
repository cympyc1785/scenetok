import math
import os
import contextlib

from .diffusion import first_stage_encode, last_stage_decode
from .diffusion_wrapper import *
from .warp import multi_view_warp_to_target
from einops import repeat


class T2VWrapper(DiffusionWrapper):
    def should_use_condition_latents(self):
        return (
            getattr(self.denoiser, "supports_condition_latents", False)
            and getattr(self.denoiser, "condition_latents_input_type", "none")
            not in ("none", "first_frame", "first_frame_random", "first_frame_depth", "first_frame_depth_soft")
        )

    def should_replace_first_frame_latent(self, training: bool = False):
        condition_type = getattr(self.denoiser, "condition_latents_input_type", "none")
        if condition_type == "first_frame":
            return True
        if condition_type == "first_frame_random":
            return bool(np.random.choice([True, False])) if training else False
        if condition_type in ("first_frame_depth", "first_frame_depth_soft"):
            return True
        return False

    def preprocess_scene_tokens(self, scene_tokens, shape, device, token_mask=None):
        if not getattr(self.denoiser, "supports_scene_tokens", True):
            return None

        return super().preprocess_scene_tokens(scene_tokens, shape, device, token_mask)

    def _init_text_encoder(self):
        if self.model_cfg.text_encoder is None:
            return
        if getattr(self.denoiser, "uses_internal_text_encoder", False):
            self.text_tokenizer = getattr(self.denoiser, "text_tokenizer", None)
            self.text_encoder = getattr(self.denoiser, "text_encoder", None)
            self.text_condition_proj = None
            return
        return super()._init_text_encoder()

    def get_text_condition(self, batch, device=None):
        target_device = self.device if device is None else device
        if getattr(self.denoiser, "uses_internal_text_encoder", False):
            if "text_embedding" in batch and batch["text_embedding"] is not None:
                return batch["text_embedding"].to(target_device)
            text = batch.get("text")
            if text is None:
                return None
            return self.denoiser.encode_text_condition(text, device=target_device)

        return super().get_text_condition(batch, device=device)

    def get_autoencoder_name(self, view_type: str):
        cfg = getattr(self.model_cfg.autoencoders, view_type)
        return None if cfg is None else cfg.name

    def get_autoencoder_scaling_factor(self, view_type: str):
        cfg = getattr(self.model_cfg.autoencoders, view_type)
        return 1.0 if cfg is None else cfg.kwargs.scaling_factor

    def get_condition_latents(self, batch, device, dtype, target_num_views=None, first_frame_only=False):
        if not self.should_use_condition_latents():
            return None

        target_cfg = getattr(self.model_cfg.autoencoders, "target")
        target_name = target_cfg.name
        target_scale = target_cfg.kwargs.scaling_factor

        if self.dataset_cfg.precomputed_latents["context"]:
            raise ValueError(
                "Width-concat Wan TI2V conditioning requires raw context images so they can be "
                "repeated to the target view count and encoded with the target Wan autoencoder."
            )

        context_views = batch["context"]["latent"]
        if target_num_views is None:
            target_num_views = batch["target"]["latent"].shape[1]
        repeat_count = math.ceil(target_num_views / context_views.shape[1])
        context_views = repeat(context_views, "b v c h w -> b (r v) c h w", r=repeat_count)
        context_views = context_views[:, :target_num_views]

        target_image_shape = batch["target"]["latent"].shape[-2:]
        if context_views.shape[-2:] != target_image_shape:
            b, v, c, _, _ = context_views.shape
            context_views = rearrange(context_views, "b v c h w -> (b v) c h w")
            context_views = F.interpolate(
                context_views,
                size=target_image_shape,
                mode="bilinear",
                align_corners=False,
            )
            context_views = rearrange(context_views, "(b v) c h w -> b v c h w", b=b, v=v)

        condition_latents = first_stage_encode(
            autoencoder=self.autoencoder,
            inputs=context_views,
            view_type="target",
            autoencoder_name=target_name,
            scaling_factor=target_scale,
            chunk_targets=getattr(self.dataset_cfg.view_sampler, "chunk_targets", True),
        )
        
        if first_frame_only:
            condition_latents = condition_latents[:, :1]

        return condition_latents.to(device=device, dtype=dtype)

    def _build_first_frame_from_depth(self, batch, device, dtype):
        """Forward-warp context views into target view's first frame using DA3
        depth + npz cameras (loaded by `dataset.load_da3_depth=true`). Returns
        a (B, 3, H_target, W_target) image in [-1, 1] to be used in place of
        the GT target first frame for `first_frame_depth` conditioning."""
        ctx = batch["context"]
        tgt = batch["target"]
        required = ("depth_at_target_shape", "da3_w2c_at_target_shape", "da3_intrinsics_at_target_shape")
        for k in required:
            if k not in ctx:
                raise KeyError(
                    f"first_frame_depth requires `context['{k}']` from `dataset.load_da3_depth=true`."
                )
        if "da3_w2c_first" not in tgt or "da3_intrinsics_first" not in tgt:
            raise KeyError(
                "first_frame_depth requires `target['da3_w2c_first']`/`da3_intrinsics_first` "
                "from `dataset.load_da3_depth=true`."
            )

        # Context RGB at target shape — resize on the fly (context["latent"]
        # was preprocessed to context_shape).
        ctx_imgs = ctx["latent"].to(device=device, dtype=dtype)
        b, v, c, ch, cw = ctx_imgs.shape
        target_shape = tuple(batch["target"]["latent"].shape[-2:])
        if (ch, cw) != target_shape:
            ctx_imgs = rearrange(ctx_imgs, "b v c h w -> (b v) c h w")
            ctx_imgs = F.interpolate(ctx_imgs, size=target_shape, mode="bilinear", align_corners=False)
            ctx_imgs = rearrange(ctx_imgs, "(b v) c h w -> b v c h w", b=b, v=v)
        # `forward_warp` operates in [-1, 1] (holes filled with -1 when
        # is_image=True); our dataset RGB lives in [0, 1]. Convert before warp
        # and convert back so the downstream `first_stage_encode` (which does
        # `inputs * 2 - 1` internally) sees a clean [0, 1] tensor with
        # 0-valued (black) holes — not double-converted -3 garbage.
        ctx_imgs = ctx_imgs * 2.0 - 1.0

        ctx_depth = ctx["depth_at_target_shape"].to(device=device, dtype=dtype)
        ctx_w2c = ctx["da3_w2c_at_target_shape"].to(device=device, dtype=dtype)
        ctx_intr = ctx["da3_intrinsics_at_target_shape"].to(device=device, dtype=dtype)
        tgt_w2c = tgt["da3_w2c_first"].to(device=device, dtype=dtype)
        tgt_intr = tgt["da3_intrinsics_first"].to(device=device, dtype=dtype)

        warped, mask = multi_view_warp_to_target(
            context_images=ctx_imgs,
            context_depths=ctx_depth,
            context_w2cs=ctx_w2c,
            context_intrinsics=ctx_intr,
            target_w2c=tgt_w2c,
            target_intrinsics=tgt_intr,
            topk=2,
        )
        # warped: (B, 3, H, W) in [-1, 1], holes are -1. mask: {0, 1}.
        # Bring back to dataset's [0, 1] convention (holes → 0, "black").
        # `first_stage_encode` will subsequently re-shift to [-1, 1] for VAE.
        warped = ((warped + 1.0) / 2.0).clamp(0.0, 1.0)
        return warped, mask

    def get_first_frame_latents(self, batch, device, dtype, repeat_factor: int = 1):
        target_cfg = getattr(self.model_cfg.autoencoders, "target")
        if target_cfg is None or target_cfg.name != "wan":
            raise ValueError("first_frame conditioning currently requires the target Wan autoencoder.")
        if self.dataset_cfg.precomputed_latents["target"]:
            raise ValueError("first_frame conditioning requires raw target images, not precomputed target latents.")

        condition_type = getattr(self.denoiser, "condition_latents_input_type", "none")
        first_frame_mask = None  # pixel-space visibility, used for soft blend
        if condition_type in ("first_frame_depth", "first_frame_depth_soft"):
            warped, vis_mask = self._build_first_frame_from_depth(batch, device=device, dtype=dtype)
            first_frame = warped.unsqueeze(1)                                 # (B, 1, 3, H, W)
            # `first_frame_depth`      → hard replacement (mask = 1 everywhere)
            # `first_frame_depth_soft` → soft blend with the actual visibility
            if condition_type == "first_frame_depth_soft":
                first_frame_mask = vis_mask
        else:
            first_frame = batch["target"]["latent"][:, :1].to(device=device, dtype=dtype)
        padding = torch.zeros(
            first_frame.shape[0],
            3,
            *first_frame.shape[2:],
            device=device,
            dtype=dtype,
        )
        first_frame_video = torch.cat([first_frame, padding], dim=1)
        first_frame_latents = first_stage_encode(
            autoencoder=self.autoencoder,
            inputs=first_frame_video,
            view_type="target",
            autoencoder_name=target_cfg.name,
            scaling_factor=target_cfg.kwargs.scaling_factor,
            chunk_targets=False,
        )[:, :1]
        if repeat_factor != 1:
            first_frame_latents = repeat(first_frame_latents, "b ... -> (b r) ...", r=repeat_factor)

        # Build per-latent-token blend mask **only** for the soft variant.
        # For hard replacement (`first_frame` / `first_frame_random` /
        # `first_frame_depth`) we return mask_latent=None so the downstream
        # blend short-circuits to direct assignment `x_t[:, 0:1] = first_frame_latents`.
        latent_h, latent_w = first_frame_latents.shape[-2:]
        if first_frame_mask is not None:
            mask_latent = F.avg_pool2d(first_frame_mask.float(), kernel_size=16, stride=16)
            # Match latent grid even if H/16 != latent_h (e.g., Wan VAE quirks).
            if mask_latent.shape[-2:] != (latent_h, latent_w):
                mask_latent = F.interpolate(
                    mask_latent, size=(latent_h, latent_w), mode="bilinear", align_corners=False
                )
            mask_latent = mask_latent.clamp(0, 1).unsqueeze(1)  # (B, 1, 1, h, w)
            if repeat_factor != 1:
                mask_latent = repeat(mask_latent, "b ... -> (b r) ...", r=repeat_factor)
            mask_latent = mask_latent.to(device=device, dtype=dtype)
        else:
            mask_latent = None   # → caller does hard replace

        return (
            first_frame_latents.to(device=device, dtype=dtype),
            mask_latent,
        )

    def prepare_target_pose(self, target, num_latents: Optional[int] = None):
        target_cfg = getattr(self.model_cfg.autoencoders, "target")
        target_name = None if target_cfg is None else target_cfg.name
        temporal_downsample = 4 if target_name in ["video_dc", "wan"] else 1
        intrinsics = target["intrinsics"]
        extrinsics = target["extrinsics"]
        num = num_latents

        if target_name == "video_dc":
            if num is None:
                num = extrinsics.shape[1] // temporal_downsample
            num_pose_views = num * temporal_downsample
            intrinsics = intrinsics[:, :num_pose_views]
            extrinsics = extrinsics[:, :num_pose_views]
        elif target_name == "wan":
            if getattr(self.dataset_cfg.view_sampler, "chunk_targets", True):
                if num is None:
                    num_chunks = extrinsics.shape[1] // 17
                else:
                    num_chunks = num_latents // 5
                num = num_chunks * 5
                num_pose_views = num_chunks * 17
            else:
                if num is None:
                    num = 1 + (extrinsics.shape[1] - 1) // temporal_downsample
                num_pose_views = 1 + (num - 1) * temporal_downsample
            intrinsics = intrinsics[:, :num_pose_views]
            extrinsics = extrinsics[:, :num_pose_views]

        return CameraInputs(intrinsics=intrinsics, extrinsics=extrinsics), temporal_downsample, num

    @torch.no_grad()
    def generate_batch_with_scene(self, batch, sampler: Sampler, repeat_factor: int = 1):
        if os.environ.get("FORCE_FP32"):
            self.denoiser.model.to(torch.float32)
            if getattr(self.denoiser, "text_encoder", None) is not None:
                self.denoiser.text_encoder.to(torch.float32)
            import diffsynth.models.wan_video_dit as _wvd
            _wvd.FLASH_ATTN_3_AVAILABLE = False
            _wvd.FLASH_ATTN_2_AVAILABLE = False
            _wvd.SAGE_ATTN_AVAILABLE = False
            print("[T2VWrapper] FORCE_FP32: cast model+text_encoder → fp32, disable FlashAttention")
        if self.model_cfg.compressor is not None:
            context_latents = get_latents(
                autoencoder=self.autoencoder,
                inputs=batch["context"],
                view_type="context",
                precomputed_latents=self.dataset_cfg.precomputed_latents,
                autoencoder_name=self.get_autoencoder_name("context"),
                scaling_factor=self.get_autoencoder_scaling_factor("context"),
            )
            device = context_latents.device
            dtype = context_latents.dtype
        else:
            context_latents = None
            target_latent = batch["target"].get("latent")
            if target_latent is not None:
                device = target_latent.device
                dtype = target_latent.dtype
            else:
                device = batch["target"]["extrinsics"].device
                dtype = next(self.denoiser.parameters()).dtype

        target = repeat_batch(batch["target"], repeat_factor)
        text_state = self.get_text_condition(batch, device=device)
        if text_state is not None:
            text_state = einops.repeat(text_state, "b n d -> (b r) n d", r=repeat_factor)

        b, v_t, *_ = target["extrinsics"].shape
        target_pose, temporal_downsample, num = self.prepare_target_pose(target)
        if num is None:
            num = v_t

        target_cfg = getattr(self.model_cfg.autoencoders, "target")
        if target_cfg is not None:
            if target_cfg.name in ["kl"]:
                c = target_cfg.kwargs.latent_channels // 2
            else:
                c = target_cfg.kwargs.latent_channels
        else:
            c = 3

        input_shape = self.model_cfg.denoiser.input_shape
        if isinstance(input_shape, int):
            h = w = input_shape
        else:
            h, w = input_shape

        noise_seed = getattr(getattr(self.denoiser, "cfg", None), "noise_seed", None)
        noise_generator = None
        if noise_seed is not None:
            noise_generator = torch.Generator(device=device)
            noise_generator.manual_seed(int(noise_seed))
            print(f"(Wan TI2V) Using noise seed: {noise_seed}")

        x_t = torch.randn((b, num, c, h, w), device=device, dtype=dtype, generator=noise_generator)
        x_t *= self.scheduler.init_noise_sigma
        _inject_path = os.environ.get("INJECT_NOISE_PATH")
        if _inject_path:
            injected = torch.load(_inject_path, map_location=device).to(dtype=dtype)
            assert injected.shape == x_t.shape, f"injected noise shape {tuple(injected.shape)} != x_t {tuple(x_t.shape)}"
            print(f"[T2VWrapper] Overriding x_t with noise from {_inject_path}  sum={injected.float().sum().item():.4f}")
            x_t = injected
        first_frame_latents = None
        first_frame_mask_latent = None
        if self.should_replace_first_frame_latent():
            first_frame_latents, first_frame_mask_latent = self.get_first_frame_latents(
                batch,
                device=device,
                dtype=dtype,
                repeat_factor=repeat_factor,
            )
            if first_frame_latents is not None:
                if first_frame_mask_latent is None:
                    # Hard replace — no blend, just assign.
                    x_t[:, 0:1] = first_frame_latents
                else:
                    m = first_frame_mask_latent  # (B, 1, 1, h, w) broadcastable
                    x_t[:, 0:1] = m * first_frame_latents + (1.0 - m) * x_t[:, 0:1]

        context_camera = CameraInputs(
            intrinsics=batch["context"]["intrinsics"],
            extrinsics=batch["context"]["extrinsics"],
        )

        if self.model_cfg.compressor is None:
            scene_tokens = None
        else:
            context_inputs = CompressorInputs(
                view=context_latents,
                pose=context_camera,
                mask=None,
            )
            tokens, _ = self.compressor._forward(inputs=context_inputs)
            if self.model_cfg.compressor.scene_token_projection == "kl":
                scene_tokens = tokens.sample()
            else:
                scene_tokens = tokens
            scene_tokens = repeat(scene_tokens, "b ... -> (b n) ...", n=repeat_factor)

        sampler.set_scheduling_matrix(
            horizon=num,
            steps=self.model_cfg.scheduler.num_inference_steps,
            concurrency=self.dataset_cfg.view_sampler.num_target_views,
            device=device,
            dtype=dtype,
            cond_mask_indices=None,
        )
        if self.model_cfg.scheduler.kwargs.weighting == "shifted":
            print(f"(Sampler) Shifting scheduling matrix by {self.model_cfg.scheduler.kwargs.timestep_shift}")
            sampler.shift_scheduling_matrix(self.model_cfg.scheduler.kwargs.timestep_shift)
        sampler.log_vis(self.logger, step=self.step_tracker, name=f"({sampler.cfg.name})")
        print("Shape of latents: ", x_t.shape)
        if "negative_prompt" in batch and hasattr(self.denoiser, "negative_prompt"):
            self.denoiser.negative_prompt = batch["negative_prompt"]
        return *sample(
            model=self.denoiser,
            x_t=x_t,
            target_pose=target_pose,
            cond_state=scene_tokens,
            text_state=text_state,
            sampler=sampler,
            scheduler=self.scheduler,
            autoencoder=self.autoencoder,
            temporal_downsample=temporal_downsample,
            cfg_scale=self.model_cfg.cfg_scale,
            autoencoder_name=target_cfg.name,
            scaling_factor=target_cfg.kwargs.scaling_factor,
            chunk_index_gap=getattr(self.dataset_cfg.view_sampler, "chunk_index_gap", 1),
            offset=self.dataset_cfg.view_sampler.offset,
            chunk_targets=getattr(self.dataset_cfg.view_sampler, "chunk_targets", True),
            first_frame_latents=first_frame_latents,
            first_frame_mask_latent=first_frame_mask_latent,
        ), scene_tokens

    def test_step(self, batch, batch_idx):
        if batch is None:
            return None

        step = self.step_tracker.get_step()

        prompt = getattr(self.test_cfg, "prompt", None)
        negative_prompt = getattr(self.test_cfg, "negative_prompt", None)

        if prompt:
            batch["text"] = [prompt] * len(batch["scene"])
            print(f"\n\nPrompt: {batch['text']}")
        if negative_prompt:
            batch["negative_prompt"] = negative_prompt
            print(f"Negative prompt: {batch['negative_prompt']}\n")
        
        print(
            f"Current epoch {step}; "
            f"Step {batch_idx}; "
            f"Scene = {batch['scene']}; "
            f"Context = {batch['context']['index'].tolist()}; "
            f"Target = {batch['target']['index'].tolist()}; "
        )

        b, v_t, *_ = batch["target"]["extrinsics"].shape
        b, v_c, *_ = batch["context"]["extrinsics"].shape

        print(f"Number of context views: {v_c}")
        print(f"Number of target views: {v_t}")

        target_views=get_images(
            autoencoder=self.autoencoder,
            inputs=batch["target"],
            view_type="target",
            precomputed_latents=self.dataset_cfg.precomputed_latents,
            autoencoder_name=getattr(self.model_cfg.autoencoders, "target").name,
            scaling_factor=getattr(self.model_cfg.autoencoders, "target").kwargs.scaling_factor,
            chunk_targets=getattr(self.dataset_cfg.view_sampler, "chunk_targets", True),
        )
        context_views=get_images(
            autoencoder=self.autoencoder,
            inputs=batch["context"],
            view_type="context",
            precomputed_latents=self.dataset_cfg.precomputed_latents,
            autoencoder_name=getattr(self.model_cfg.autoencoders, "context").name,
            scaling_factor=getattr(self.model_cfg.autoencoders, "context").kwargs.scaling_factor,
            chunk_targets=getattr(self.dataset_cfg.view_sampler, "chunk_targets", True),
        )

        # Relative camera w.r.t middle context camera (can be any other context camera)
        batch = preprocess_batch(batch, index=v_c//2)
        sampled_views, uncertainty_maps, _ = self.generate_batch_with_scene(
            batch, 
            self.sampler, 
            repeat_factor=1,
        )

        # for j in tqdm(range(b), desc="Saving Uncertainty Maps: "):
        #     save_image_video(
        #         images=uncertainty_maps[j], 
        #         indices=torch.arange(0, uncertainty_maps[j].shape[0]), 
        #         output_dir=self.output_dir / "uncertainty" / batch["scene"][j],
        #         name=self.sampler.cfg.name, save_img=True, save_video=True, fps=self.dataset_cfg.fps
        #     )

        for j in tqdm(range(b), desc="Saving Sampled Views: "):
            save_image_video(
                images=sampled_views[j].float(), 
                indices=torch.arange(0, sampled_views[j].shape[0]), 
                output_dir=self.output_dir / "predicted" / batch["scene"][j],
                name=self.sampler.cfg.name, save_img=True, save_video=True, fps=self.dataset_cfg.fps
            )
        # for j in tqdm(range(b), desc="Saving Original Views: "):
        #     save_image_video(
        #         images=target_views[j].float(), 
        #         indices=torch.arange(0, target_views[j].shape[0]), 
        #         output_dir=self.output_dir / "gt" / batch["scene"][j],
        #         name="original", save_img=True, save_video=True, fps=self.dataset_cfg.fps
        #     )

        #     save_image_video(
        #         images=context_views[j].float(), 
        #         indices=batch["context"]["index"][j], 
        #         output_dir=self.output_dir / "context" / batch["scene"][j],
        #         name="context", save_img=True, save_video=True, fps=self.dataset_cfg.fps
        #     )
        return None

    def _reco_training_step(self, batch, batch_idx):
        """ReCo(Wan2.1 VACE 1.3B) + LightningDiT ctrl branch.

        - main 16ch ReCo latent (width-doubled): 좌=inpaint_result(recon) / 우=video_input(dynamic).
        - bg 48ch latent (ldt branch 입력): inpaint_result, denoiser 내부 Wan2.2 VAE로 인코딩.
        - same timestep t로 둘 다 noise. 출력 좌/우 split → recon + dynamic rectified-flow loss.
        """
        batch = preprocess_batch(batch)
        ae_name = self.get_autoencoder_name("target")
        ae_scale = self.get_autoencoder_scaling_factor("target")
        chunk = getattr(self.dataset_cfg.view_sampler, "chunk_targets", True)

        # 16ch latents: video_input(dynamic, batch target latent) + inpaint_result(recon_video)
        dyn_lat = get_latents(autoencoder=self.autoencoder, inputs=batch["target"], view_type="target",
                              precomputed_latents=self.dataset_cfg.precomputed_latents,
                              autoencoder_name=ae_name, scaling_factor=ae_scale, chunk_targets=chunk)
        recon_video = batch["target"].get("recon_video")
        if recon_video is None:
            raise ValueError("ReCo 학습은 dataset.recon_target_video_name(inpaint_result.mp4)이 필요합니다.")
        recon_lat = get_latents(autoencoder=self.autoencoder, inputs={**batch["target"], "latent": recon_video},
                                view_type="target", precomputed_latents=self.dataset_cfg.precomputed_latents,
                                autoencoder_name=ae_name, scaling_factor=ae_scale, chunk_targets=chunk)

        device, dtype = dyn_lat.device, dyn_lat.dtype
        target_pose, temporal_downsample, num_target_latents = self.prepare_target_pose(
            batch["target"], num_latents=dyn_lat.shape[1])
        if num_target_latents is not None and num_target_latents != dyn_lat.shape[1]:
            dyn_lat = dyn_lat[:, :num_target_latents]
            recon_lat = recon_lat[:, :num_target_latents]
        # width-doubled main target: 좌 recon | 우 dynamic
        target_reco = torch.cat([recon_lat, dyn_lat], dim=-1)        # (B,V,16,H,2W)
        b, v_t = target_reco.shape[:2]

        # bg 48ch latent (inpaint_result frames → denoiser 내부 VAE)
        bg_clean = self.denoiser.encode_bg(recon_video.to(device=device, dtype=dtype))
        if num_target_latents is not None and num_target_latents != bg_clean.shape[1]:
            bg_clean = bg_clean[:, :num_target_latents]

        # scene tokens (context → compressor)
        scene_tokens = None
        tokens = None
        if self.model_cfg.compressor is not None:
            context_latents = get_latents(autoencoder=self.autoencoder, inputs=batch["context"], view_type="context",
                                          precomputed_latents=self.dataset_cfg.precomputed_latents,
                                          autoencoder_name=self.get_autoencoder_name("context"),
                                          scaling_factor=self.get_autoencoder_scaling_factor("context"))
            context_inputs = CompressorInputs(
                view=context_latents,
                pose=CameraInputs(intrinsics=batch["context"]["intrinsics"], extrinsics=batch["context"]["extrinsics"]),
                mask=None)
            if self.frozen_compressor:
                with torch.no_grad():
                    tokens, *_ = self.compressor(inputs=context_inputs)
            else:
                tokens, *_ = self.compressor(inputs=context_inputs)
            scene_tokens = tokens.sample() if self.model_cfg.compressor.scene_token_projection == "kl" else tokens

        # timestep (b,v) + noise (same t for both branches, independent noise per channel-count)
        timestep = self.get_noise_level((b,), dtype=dtype)
        timestep = repeat(timestep, "b -> b v", v=v_t)
        noise_reco = torch.randn_like(target_reco)
        noisy_reco = self.scheduler.add_noise(target_reco, noise_reco, timestep)
        noise_bg = torch.randn_like(bg_clean)
        noisy_bg = self.scheduler.add_noise(bg_clean, noise_bg, timestep)

        # scene tokens은 ldt branch의 cnd_proj로만 들어감 (main ReCo DiT는 scene 직접 미사용)
        # → raw_state로만 전달, state(projected)는 None.
        raw_scene_tokens = scene_tokens if scene_tokens is not None else torch.zeros(
            (b, self.denoiser.num_scene_tokens, self.denoiser.cond_dim), device=device, dtype=dtype)

        denoiser_input = DenoiserInputs(
            view=noisy_reco, pose=target_pose, timestep=self.rescale_timesteps(timestep=timestep),
            state=None, text=self.get_text_condition(batch, device=device),
            condition_latents=noisy_bg, raw_state=raw_scene_tokens)

        pred = self.denoiser(inputs=denoiser_input, temporal_downsample=temporal_downsample, chunk_targets=chunk)
        if isinstance(pred, tuple):
            pred = pred[0]
        gt = self.process_gt(target_reco, noise_reco, timestep)      # (B,V,16,H,2W)

        wh = gt.shape[-1] // 2
        loss_recon = F.mse_loss(pred[..., :wh], gt[..., :wh])        # 좌: background recon
        loss_dyn = F.mse_loss(pred[..., wh:], gt[..., wh:])          # 우: dynamic
        recon_w = float(getattr(self.denoiser.cfg, "recon_loss_weight", 1.0))
        loss = recon_w * loss_recon + loss_dyn

        current_lr = self.optimizers().param_groups[0]["lr"]
        if self.global_rank == 0:
            print(f"Train step {self.step_tracker.get_step()}; loss = {loss.item():.4f} "
                  f"(recon {loss_recon.item():.4f} dyn {loss_dyn.item():.4f}) lr = {current_lr}")
        self.log("loss/diffusion", loss)
        self.log("loss/reco_recon", loss_recon)
        self.log("loss/reco_dynamic", loss_dyn)
        return loss

    def _reco_encode_dual_latents(self, batch):
        """ReCo 16ch width-doubled GT(좌 recon / 우 dynamic) + 48ch bg GT + scene/pose/text.
        training/validation 공용. returns dict."""
        ae_name = self.get_autoencoder_name("target")
        ae_scale = self.get_autoencoder_scaling_factor("target")
        chunk = getattr(self.dataset_cfg.view_sampler, "chunk_targets", True)
        dyn_lat = get_latents(autoencoder=self.autoencoder, inputs=batch["target"], view_type="target",
                              precomputed_latents=self.dataset_cfg.precomputed_latents,
                              autoencoder_name=ae_name, scaling_factor=ae_scale, chunk_targets=chunk)
        recon_video = batch["target"].get("recon_video")
        if recon_video is None:
            raise ValueError("ReCo는 dataset.recon_target_video_name(inpaint_result.mp4)이 필요합니다.")
        recon_lat = get_latents(autoencoder=self.autoencoder, inputs={**batch["target"], "latent": recon_video},
                                view_type="target", precomputed_latents=self.dataset_cfg.precomputed_latents,
                                autoencoder_name=ae_name, scaling_factor=ae_scale, chunk_targets=chunk)
        target_pose, temporal_downsample, num_target_latents = self.prepare_target_pose(
            batch["target"], num_latents=dyn_lat.shape[1])
        if num_target_latents is not None and num_target_latents != dyn_lat.shape[1]:
            dyn_lat, recon_lat = dyn_lat[:, :num_target_latents], recon_lat[:, :num_target_latents]
        target_reco = torch.cat([recon_lat, dyn_lat], dim=-1)
        device, dtype = target_reco.device, target_reco.dtype
        bg_clean = self.denoiser.encode_bg(recon_video.to(device=device, dtype=dtype))
        if num_target_latents is not None and num_target_latents != bg_clean.shape[1]:
            bg_clean = bg_clean[:, :num_target_latents]
        scene_tokens = None
        if self.model_cfg.compressor is not None:
            ctx_lat = get_latents(autoencoder=self.autoencoder, inputs=batch["context"], view_type="context",
                                  precomputed_latents=self.dataset_cfg.precomputed_latents,
                                  autoencoder_name=self.get_autoencoder_name("context"),
                                  scaling_factor=self.get_autoencoder_scaling_factor("context"))
            ci = CompressorInputs(view=ctx_lat, pose=CameraInputs(
                intrinsics=batch["context"]["intrinsics"], extrinsics=batch["context"]["extrinsics"]), mask=None)
            with torch.no_grad() if self.frozen_compressor else contextlib.nullcontext():
                tokens, *_ = self.compressor(inputs=ci)
            scene_tokens = tokens.sample() if self.model_cfg.compressor.scene_token_projection == "kl" else tokens
        b = target_reco.shape[0]
        raw_scene = scene_tokens if scene_tokens is not None else torch.zeros(
            (b, self.denoiser.num_scene_tokens, self.denoiser.cond_dim), device=device, dtype=dtype)
        return dict(target_reco=target_reco, dyn_lat=dyn_lat, recon_lat=recon_lat, bg_clean=bg_clean,
                    raw_scene=raw_scene, target_pose=target_pose, temporal_downsample=temporal_downsample,
                    device=device, dtype=dtype)

    # 메트릭(PSNR/FVD)용 누적 영상 상한 — full-seq sampling이 배치마다 비싸서 cap.
    RECO_METRIC_MAX_VIDEOS = 24

    @torch.no_grad()
    def _reco_validation_step(self, batch, batch_idx, dataloader_idx=None):
        if batch is None or self.global_rank != 0:
            return None
        step = self.step_tracker.get_step()
        val_step = (step + 1) // self.val_check_interval
        loader_name = self.validation_loader_names.get(dataloader_idx or 0, f"val_{dataloader_idx}")
        # validation 라운드가 바뀌면 buffer reset (sanity-val 잔여물이 섞이지 않게).
        if getattr(self, "_reco_metric_val_step", None) != val_step:
            self._reco_metric_buf = {}
            self._reco_metric_val_step = val_step
        buf = self._reco_metric_buf.setdefault(
            loader_name, {"pred_recon": [], "gt_recon": [], "pred_dyn": [], "gt_dyn": []})
        n_done = len(buf["pred_dyn"])
        # 첫 배치(영상 로깅)는 항상, 이후 배치는 메트릭 cap까지만 sampling.
        if batch_idx != 0 and n_done >= self.RECO_METRIC_MAX_VIDEOS:
            return None
        b, v_c = batch["context"]["extrinsics"].shape[:2]
        batch = preprocess_batch(batch, index=v_c // 2)
        d = self._reco_encode_dual_latents(batch)
        target_reco, bg_clean = d["target_reco"], d["bg_clean"]
        text = self.denoiser.encode_text_condition(batch.get("text"), device=d["device"])
        td = d["temporal_downsample"]
        b, v_t = target_reco.shape[:2]
        wh = target_reco.shape[-1] // 2

        # full-sequence flow sampling (bg teacher-forced from GT at each t)
        self.scheduler.set_timesteps(self.model_cfg.scheduler.num_inference_steps)
        ts_list = self.scheduler.timesteps  # cpu (next_timestep이 cpu 비교) — step 내부서 device 처리
        x = torch.randn_like(target_reco)
        noise_bg = torch.randn_like(bg_clean)
        for i in range(len(ts_list) - 1):
            t = ts_list[i]
            ts_full = t.reshape(1, 1).expand(b, v_t).to(device=d["device"], dtype=d["dtype"])
            x_in = self.scheduler.scale_model_input(x, ts_full)
            bg_t = self.scheduler.add_noise(bg_clean, noise_bg, ts_full)
            inp = DenoiserInputs(view=x_in, pose=d["target_pose"], timestep=self.rescale_timesteps(ts_full),
                                 state=None, text=text, condition_latents=bg_t, raw_state=d["raw_scene"])
            pred = self.denoiser(inputs=inp, temporal_downsample=td, chunk_targets=False)
            if isinstance(pred, tuple):
                pred = pred[0]
            x = self.scheduler.step(pred, t, x).prev_sample

        ae = self.autoencoder["target"]
        # decode는 [-1,1] → [0,1] (last_stage_decode와 동일). 로깅/메트릭 공통.
        to01 = lambda v: (v.float() / 2 + 0.5).clamp(0, 1)
        recon_vid = to01(ae.decode(x[..., :wh].to(d["dtype"])))      # (b, vd, 3, H, W)
        dyn_vid = to01(ae.decode(x[..., wh:].to(d["dtype"])))
        # 메트릭용 GT 디코드 (좌=recon GT, 우=dynamic GT)
        gt_recon = to01(ae.decode(target_reco[..., :wh].to(d["dtype"])))
        gt_dyn = to01(ae.decode(target_reco[..., wh:].to(d["dtype"])))
        if n_done < self.RECO_METRIC_MAX_VIDEOS:
            buf["pred_recon"].append(recon_vid.half().cpu()); buf["gt_recon"].append(gt_recon.half().cpu())
            buf["pred_dyn"].append(dyn_vid.half().cpu()); buf["gt_dyn"].append(gt_dyn.half().cpu())

        if batch_idx == 0:  # 정성 영상 로깅 (첫 배치만)
            ref_lat = rearrange(self.denoiser._last_ref_latent, "b c f h w -> b f c h w")
            ldt_vid = to01(ae.decode(ref_lat.to(d["dtype"])))
            scenes = batch.get("scene", [f"val{j}" for j in range(b)])
            log_tensor_as_video(self.logger, recon_vid, f"{loader_name}/ReCo Recon (left)", fps=8, step=val_step, caption=scenes)
            log_tensor_as_video(self.logger, dyn_vid, f"{loader_name}/ReCo Dynamic (right)", fps=8, step=val_step, caption=scenes)
            log_tensor_as_video(self.logger, ldt_vid, f"{loader_name}/LightningDiT ref (ldt→VACE)", fps=8, step=val_step, caption=scenes)
        torch.cuda.empty_cache()
        return None

    def _reco_on_validation_end(self):
        """ReCo PSNR/FVD flush: recon(좌)·dynamic(우) 각각 GT 대비. rank0 only."""
        if self.global_rank != 0:
            return
        step = self.step_tracker.get_step()
        val_step = (step + 1) // self.val_check_interval
        for loader_name, buf in getattr(self, "_reco_metric_buf", {}).items():
            for tag in ("recon", "dyn"):
                preds, gts = buf[f"pred_{tag}"], buf[f"gt_{tag}"]
                if not preds:
                    continue
                pred = torch.cat(preds); gt = torch.cat(gts)          # (N, vd, 3, H, W) fp16 cpu
                vd = pred.shape[1]
                print(f"[reco-metric] {loader_name}/{tag}: {pred.shape[0]} videos × {vd} frames")
                pf = rearrange(pred, "b v c h w -> (b v) c h w")
                gf = rearrange(gt, "b v c h w -> (b v) c h w")
                psnr = self.metric(pf.float().to(self.device), gf.float().to(self.device), psnr=True).get("psnr")
                if psnr is not None:
                    self.logger.log_metrics({f"{loader_name}/{tag}/psnr": psnr}, val_step)
                # FVD는 분포 메트릭이라 표본이 너무 적으면(예: sanity-val 1개) 불안정/에러 → skip.
                if pred.shape[0] >= 4:
                    try:
                        fvd = self.metric(pf.float(), gf.float(), num_views=vd, fvd=True).get("fvd")
                    except Exception as e:
                        print(f"[reco-metric] FVD skip ({loader_name}/{tag}): {e}"); fvd = None
                    self.metric.reset_fvd()
                    if fvd is not None:
                        self.logger.log_metrics({f"{loader_name}/{tag}/fvd": fvd}, val_step)
                else:
                    print(f"[reco-metric] FVD skip ({loader_name}/{tag}): only {pred.shape[0]} video(s)")
        self._reco_metric_buf = {}

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        from src.model.denoiser.reco_wan_vace import RecoWanVace1_3BDenoiser
        if isinstance(self.denoiser, RecoWanVace1_3BDenoiser):
            return self._reco_validation_step(batch, batch_idx, dataloader_idx)
        return super().validation_step(batch, batch_idx, dataloader_idx)

    def on_validation_end(self):
        from src.model.denoiser.reco_wan_vace import RecoWanVace1_3BDenoiser
        if isinstance(self.denoiser, RecoWanVace1_3BDenoiser):
            return self._reco_on_validation_end()
        return super().on_validation_end()

    def training_step(self, batch, batch_idx):
        if batch is None:  # safe_collate returned None (entire batch was None-filtered)
            return None
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)
            self.log("step_tracker/step", self.step_tracker.get_step())

        # ReCo(Wan2.1 VACE) + LightningDiT ctrl branch — width-doubled dual-loss path.
        from src.model.denoiser.reco_wan_vace import RecoWanVace1_3BDenoiser
        if isinstance(self.denoiser, RecoWanVace1_3BDenoiser):
            return self._reco_training_step(batch, batch_idx)

        batch = preprocess_batch(batch)
        target_latents = get_latents(
            autoencoder=self.autoencoder,
            inputs=batch["target"],
            view_type="target",
            precomputed_latents=self.dataset_cfg.precomputed_latents,
            autoencoder_name=self.get_autoencoder_name("target"),
            scaling_factor=self.get_autoencoder_scaling_factor("target"),
            chunk_targets=getattr(self.dataset_cfg.view_sampler, "chunk_targets", True),
        )

        device = target_latents.device
        dtype = target_latents.dtype
        target_pose, temporal_downsample, num_target_latents = self.prepare_target_pose(
            batch["target"],
            num_latents=target_latents.shape[1],
        )
        if num_target_latents is not None and num_target_latents != target_latents.shape[1]:
            target_latents = target_latents[:, :num_target_latents]
        b, v_t, *_ = target_latents.shape

        conditional_tokens = True
        if self.train_cfg.cfg_train and not self.freeze_cfg.denoiser:
            conditional_tokens = np.random.choice([True, False], 1, p=[0.90, 0.10])

        token_mask = None
        if conditional_tokens and self.model_cfg.compressor is not None:
            context_latents = get_latents(
                autoencoder=self.autoencoder,
                inputs=batch["context"],
                view_type="context",
                precomputed_latents=self.dataset_cfg.precomputed_latents,
                autoencoder_name=self.get_autoencoder_name("context"),
                scaling_factor=self.get_autoencoder_scaling_factor("context"),
            )
            b_c, v_c, *_ = context_latents.shape
            if self.model_cfg.mask_context and np.random.choice([True, False], p=[0.4, 0.6]):
                context_mask = generate_biased_boolean_mask((b_c, v_c), self.dataset_cfg.view_sampler.min_context_views).to(context_latents.device)
            else:
                context_mask = None

            context_inputs = CompressorInputs(
                view=context_latents,
                pose=CameraInputs(
                    intrinsics=batch["context"]["intrinsics"],
                    extrinsics=batch["context"]["extrinsics"],
                ),
                mask=context_mask,
            )
            if self.frozen_compressor:
                with torch.no_grad():
                    tokens, *_ = self.compressor(inputs=context_inputs)
            else:
                tokens, *_ = self.compressor(inputs=context_inputs)

            if self.model_cfg.compressor.scene_token_projection == "kl":
                scene_tokens = tokens.sample()
            else:
                scene_tokens = tokens

            if self.model_cfg.noisy_scene_tokens and np.random.choice([True, False], p=[self.model_cfg.noise_prob, 1 - self.model_cfg.noise_prob]):
                scene_noise = torch.randn_like(scene_tokens, device=device)
                timestep_scene = self.get_noise_level((b, self.compressor.num_scene_tokens), dtype=dtype, mu=self.model_cfg.mu, sigma=self.model_cfg.sigma)
                scene_tokens = self.scheduler.add_noise(scene_tokens, scene_noise, timestep_scene)

            if self.model_cfg.mask_tokens:
                token_mask, _, _ = random_mask_biased(B=scene_tokens.shape[0], N=scene_tokens.shape[1], M=0.6, device="cpu")
        else:
            scene_tokens = None

        # use_scalar_timestep = (
        #     self.should_use_condition_latents()
        #     and not getattr(self.denoiser, "uses_internal_text_encoder", False)
        # )
        # if use_scalar_timestep:
        #     timestep_shape = ()
        # if not getattr(self.denoiser, "supports_per_view_timestep", True):
        #     timestep_shape = (b,)
        if self.model_cfg.scheduler.sampling_type == "random_uniform":
            timestep_shape = (b,)
        elif self.model_cfg.scheduler.sampling_type == "random_chunked_uniform":
            timestep_shape = (b, self.dataset_cfg.view_sampler.num_target_split)
        elif self.model_cfg.scheduler.sampling_type == "random_independent":
            timestep_shape = (b, v_t)
        else:
            raise NotImplementedError(f"Sampling type in scheduler is not correctly specified and instead got {self.model_cfg.scheduler.sampling_type}")

        # if (
        #     not use_scalar_timestep
        #     and np.random.choice([True, False], p=[0.2, 0.8])
        #     and self.model_cfg.enforce_uniform_noise
        # ):
        #     timestep_shape = (b,)

        timestep = self.get_noise_level(timestep_shape, dtype=dtype)
        if timestep.ndim == 1:
            timestep = repeat(timestep, "b -> b v", v=v_t)
        elif timestep.ndim == 2 and self.model_cfg.scheduler.sampling_type == "random_chunked_uniform":
            timestep = repeat(timestep, "b n -> b (n v)", v=self.dataset_cfg.view_sampler.num_target_views // self.dataset_cfg.view_sampler.num_target_split)

        # Experimental: Force zero noise-levels for conditioning 
        if self.model_cfg.force_clean:
            # if use_scalar_timestep:
            #     raise ValueError("force_clean is not supported with Wan TI2V locked first-frame conditioning.")
            target_cond_mask = self.get_conditioning_mask((b, v_t), device=device, dtype=dtype)
            timestep = timestep * target_cond_mask

        noise = torch.randn_like(target_latents, device=device)
        noisy_latents = self.scheduler.add_noise(target_latents, noise, timestep)
        first_frame_latents = None
        first_frame_mask_latent = None
        if self.should_replace_first_frame_latent(training=True):
            first_frame_latents, first_frame_mask_latent = self.get_first_frame_latents(
                batch,
                device=device,
                dtype=dtype,
            )
        if first_frame_latents is not None:
            if first_frame_mask_latent is None:
                # Hard replace — no blend, just assign.
                noisy_latents[:, 0:1] = first_frame_latents
            else:
                m = first_frame_mask_latent  # (B, 1, 1, h, w) broadcastable
                noisy_latents[:, 0:1] = m * first_frame_latents + (1.0 - m) * noisy_latents[:, 0:1]
        condition_latents = self.get_condition_latents(
            batch,
            device=device,
            dtype=dtype,
            target_num_views=batch["target"]["latent"].shape[1],
        )

        # Stash raw (pre-cnd_proj) cond tokens for branches that need cond_dim
        # input (controlnet_lightningdit uses ckpt-warm-started Linear(64→1024)).
        raw_scene_tokens = scene_tokens if scene_tokens is not None else torch.zeros(
            (b, self.denoiser.num_scene_tokens, self.denoiser.cond_dim),
            device=device, dtype=dtype,
        )
        scene_tokens = self.preprocess_scene_tokens(
            scene_tokens=scene_tokens,
            shape=(b, self.denoiser.num_scene_tokens, self.denoiser.cond_dim),
            device=device,
            token_mask=token_mask,
        )

        denoiser_input = DenoiserInputs(
            view=noisy_latents,
            pose=target_pose,
            timestep=self.rescale_timesteps(timestep=timestep),
            state=scene_tokens,
            text=self.get_text_condition(batch, device=device),
            condition_latents=condition_latents,
            raw_state=raw_scene_tokens,
        )

        # Clear stashed ctrl pred before forward; populated by wan_ti2v._forward
        # when camera_input_type=controlnet_lightningdit.
        if hasattr(self.denoiser, "_last_ctrl_pred_raw"):
            self.denoiser._last_ctrl_pred_raw = None

        pred, _ = self.denoiser(
            inputs=denoiser_input,
            temporal_downsample=temporal_downsample,
            chunk_targets=getattr(self.dataset_cfg.view_sampler, "chunk_targets", True),
        )
        gt = self.process_gt(target_latents, noise, timestep)
        if first_frame_latents is not None:
            pred = pred[:, 1:]
            gt = gt[:, 1:]
        if pred.shape[1] != gt.shape[1]:
            raise RuntimeError("prediction shape mismatch", pred.shape, gt.shape)

        loss = F.mse_loss(pred, gt, reduction="none")
        if self.model_cfg.force_clean:
            loss = einops.reduce(loss, "b v c h w -> b v", "mean")
            loss = loss * target_cond_mask
            loss = loss.sum(-1) / target_cond_mask.sum(-1)
        else:
            loss = einops.reduce(loss, "b v c h w -> b", "mean")

        # ── Joint recon loss for controlnet_lightningdit ────────────────────
        # When `model.denoiser.litdit_recon_loss_weight > 0` AND the dataset
        # provided `target["recon_video"]` (background frames), supervise the
        # LightningDiT ctrl branch's raw (pre-gate) prediction toward the
        # rectified-flow velocity for the background video. Uses the SAME
        # noise + timestep as the main forward — `noisy_latents` already
        # encodes that noise level — so only the target shifts.
        litdit_recon_w = float(getattr(self.denoiser.cfg, "litdit_recon_loss_weight", 0.0))
        recon_pred = getattr(self.denoiser, "_last_ctrl_pred_raw", None)
        recon_video = batch["target"].get("recon_video") if "target" in batch else None
        if litdit_recon_w > 0.0 and recon_pred is not None and recon_video is not None:
            recon_batch = {**batch["target"], "latent": recon_video}
            recon_target_latents = get_latents(
                autoencoder=self.autoencoder,
                inputs=recon_batch,
                view_type="target",
                precomputed_latents=self.dataset_cfg.precomputed_latents,
                autoencoder_name=self.get_autoencoder_name("target"),
                scaling_factor=self.get_autoencoder_scaling_factor("target"),
                chunk_targets=getattr(self.dataset_cfg.view_sampler, "chunk_targets", True),
            )
            if num_target_latents is not None and num_target_latents != recon_target_latents.shape[1]:
                recon_target_latents = recon_target_latents[:, :num_target_latents]
            gt_recon = self.process_gt(recon_target_latents, noise, timestep)
            recon_pred_aligned = recon_pred
            if first_frame_latents is not None:
                recon_pred_aligned = recon_pred_aligned[:, 1:]
                gt_recon = gt_recon[:, 1:]
            # ctrl branch operates in (B, C, V, H, W); main pred is (B, V, C, H, W).
            # `process_gt` returns (B, V, C, H, W); align by rearranging recon_pred.
            if recon_pred_aligned.shape != gt_recon.shape:
                recon_pred_aligned = rearrange(recon_pred_aligned, "b c v h w -> b v c h w")
            recon_loss = F.mse_loss(recon_pred_aligned, gt_recon, reduction="none")
            recon_loss = einops.reduce(recon_loss, "b v c h w -> b", "mean")
            loss = loss + litdit_recon_w * recon_loss
            self.log("loss/litdit_recon", recon_loss.mean())

        if self.model_cfg.compressor is not None and self.model_cfg.compressor.scene_token_projection == "kl" and conditional_tokens and not self.frozen_compressor:
            kl_raw = tokens.kl()
            kl = kl_raw / (tokens.mean.shape[1] * tokens.mean.shape[2])
            kl_weight = 0.0
            if self.global_step <= self.model_cfg.compressor.kl_schedule[0]:
                kl_weight = self.model_cfg.compressor.kl_weights[0]
            elif self.global_step <= self.model_cfg.compressor.kl_schedule[1]:
                t = (self.global_step - self.model_cfg.compressor.kl_schedule[0]) / (self.model_cfg.compressor.kl_schedule[1] - self.model_cfg.compressor.kl_schedule[0])
                kl_weight = (1 - t) * self.model_cfg.compressor.kl_weights[0] + t * self.model_cfg.compressor.kl_weights[1]
            else:
                kl_weight = self.model_cfg.compressor.kl_weights[1]
            loss = loss + kl_weight * kl
            self.log("loss/kl", kl.mean())
            self.log("loss/kl_raw", kl_raw.mean())

        loss = loss.mean()
        current_lr = self.optimizers().param_groups[0]["lr"]
        if self.global_rank == 0:
            print(f"Train step {self.step_tracker.get_step()}; loss = {loss.item():.4f} lr = {current_lr}")
        self.log("loss/diffusion", loss)
        return loss

    def on_train_batch_start(self, batch, batch_idx):
        step = self.global_step
        if (
            self.model_cfg.compressor is not None
            and step >= self.model_cfg.compressor.freeze_after
            and self.model_cfg.compressor.freeze_after != -1
            and not self.frozen_compressor
        ):
            print(f"[INFO] Freezing Compressor after {step} steps!")
            freeze(self.compressor)
            self.frozen_compressor = True

        if self.optimizer_cfg.scheduler is None:
            return

        if type(self.optimizer_cfg.scheduler) != list:
            warmup_iters = self.optimizer_cfg.scheduler.kwargs.get("total_iters", 0)
            if step < warmup_iters and not self.override_applied:
                self.print(f"[INFO] Warmup not done yet! Current step {step} < {warmup_iters}. Overriding will happen afterwards!")
                self.override_applied = True

            if step >= warmup_iters and not self.override_applied:
                for group in self.trainer.optimizers[0].param_groups:
                    ckpt_lr = group["lr"]
                    if ckpt_lr != self.lr:
                        group["lr"] = self.lr
                        self.print(f"[INFO] Warmup done at step {step}. Overriding LR from {ckpt_lr} to {self.lr}")
                        self.override_applied = True
        else:
            if self.optimizer_cfg.override_lr is not None and not self.override_applied:
                for group in self.trainer.optimizers[0].param_groups:
                    ckpt_lr = group["lr"]
                    if ckpt_lr != self.optimizer_cfg.override_lr:
                        group["lr"] = self.optimizer_cfg.override_lr
                        self.print(f"[INFO] Overriding LR from {ckpt_lr} to {self.optimizer_cfg.override_lr}")
                        self.override_applied = True

    @staticmethod
    def get_lr_scheduler(opt: optim.Optimizer, optim_cfg: OptimizerCfg):
        lr_scheduler_cfg = optim_cfg.scheduler
        if lr_scheduler_cfg is None:
            return None
        if type(lr_scheduler_cfg) == list:
            schedulers = [
                getattr(optim.lr_scheduler, cfg.name)(
                    opt,
                    **(cfg.kwargs if cfg.kwargs is not None else {}),
                )
                for cfg in lr_scheduler_cfg
            ]
            if len(schedulers) == 1:
                return schedulers[0]
            return optim.lr_scheduler.SequentialLR(
                optimizer=opt,
                schedulers=schedulers,
                milestones=optim_cfg.milestones,
            )
        return getattr(optim.lr_scheduler, lr_scheduler_cfg.name)(
            opt,
            **(lr_scheduler_cfg.kwargs if lr_scheduler_cfg.kwargs is not None else {}),
        )

    def configure_optimizers(self):
        optimizer = self.get_optimizer(
            self.optimizer_cfg,
            [{"params": self.denoiser.parameters()}],
            self.lr,
        )
        if self.optimizer_cfg.scheduler is None:
            return optimizer
        if type(self.optimizer_cfg.scheduler) == list:
            frequency = self.optimizer_cfg.scheduler[0].frequency
            interval = self.optimizer_cfg.scheduler[0].interval
        else:
            frequency = self.optimizer_cfg.scheduler.frequency
            interval = self.optimizer_cfg.scheduler.interval
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": self.get_lr_scheduler(optimizer, self.optimizer_cfg),
                "frequency": frequency,
                "interval": interval,
            },
        }
