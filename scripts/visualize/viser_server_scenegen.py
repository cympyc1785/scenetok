"""Interactive viser server for **SceneGen** (scene-token GENERATION) on RE10K.

For ONE re10k scene: condition SceneGen on the scene's context views, GENERATE
scene tokens (stochastic diffusion), then interactively move the TARGET camera
(gizmo / preset patterns) and RENDER the generated scene from the moved views.

Scene tokens are generated ONCE per "Generate scene tokens" click (they are
camera-independent — a function of the conditioning views, not the target). The
"Render video" button renders the *current* (edited) target trajectory from the
last-generated tokens. "Regenerate" resamples new stochastic tokens.

Model pieces (mirrors scripts/infer_scenegen.py):
  - scene_generator : checkpoints/scenegen_shift12_re10k.ckpt          (generates tokens)
  - compressor+denoiser+VAVAE/VideoDC : checkpoints/va-videodc_re10k_scene.ckpt
Loaded once at startup via the `scenegen_shift12_re10k` experiment + re10k overrides.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/visualize/viser_server_scenegen.py --scene 004e9db3337e8206
  # headless smoke (no viser UI): generate + render original target, save video
  CUDA_VISIBLE_DEVICES=0 python scripts/visualize/viser_server_scenegen.py --headless --scene 004e9db3337e8206
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

DEFAULT_RE10K_ROOT = str((REPO / "../dataset/re10k/re10k").resolve())
DEFAULT_EVAL_INDEX = "./assets/evaluation_index/re10k_c1_192.json"


# ─────────────────────────── small helpers ───────────────────────────
def save_gif(images, path, fps=12):
    from PIL import Image
    arr = (images.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype("uint8")
    frames = [Image.fromarray(a) for a in arr]
    frames[0].save(str(path), save_all=True, append_images=frames[1:],
                   duration=int(1000 / max(fps, 1)), loop=0)


def _wxyz_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]], dtype=np.float64)


def _pose44(wxyz, pos):
    M = np.eye(4); M[:3, :3] = _wxyz_to_R(np.asarray(wxyz, float)); M[:3, 3] = np.asarray(pos, float); return M


def _parse_sel(text, n):
    text = text.strip().lower()
    if text in ("", "all", "*"):
        return list(range(n))
    out = []
    for tok in text.replace(" ", "").split(","):
        if "-" in tok:
            a, b = tok.split("-"); out += list(range(int(a), int(b) + 1))
        elif tok:
            out.append(int(tok))
    return [i for i in out if 0 <= i < n]


def _rotx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _roty(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _make_pattern_poses(name, T, amount, deg, base=None):
    """pose[i] = base @ delta_local(i), a local move/rotate ramped over T frames.
    OpenCV cam local axes: +X right, +Y down, +Z forward."""
    base = np.eye(4, dtype=np.float64) if base is None else np.asarray(base, dtype=np.float64)
    th = np.radians(deg)
    move = {"move_forward": (0, 0, 1), "move_back": (0, 0, -1),
            "move_right": (1, 0, 0), "move_left": (-1, 0, 0)}
    poses = []
    for t in range(T):
        f = (t / (T - 1)) if T > 1 else 1.0
        delta = np.eye(4)
        if name in move:
            delta[:3, 3] = f * amount * np.array(move[name], dtype=np.float64)
        else:
            a = f * th
            if name == "rotate_up":      delta[:3, :3] = _rotx(a)
            elif name == "rotate_down":  delta[:3, :3] = _rotx(-a)
            elif name == "rotate_right": delta[:3, :3] = _roty(a)
            else:                        delta[:3, :3] = _roty(-a)   # rotate_left
        poses.append(base @ delta)
    return np.stack(poses)


# ─────────────────────────── model + scene ───────────────────────────
class SceneGenEngine:
    """Loads SceneGen + SceneTok renderer once; generates tokens and renders."""

    def __init__(self, args):
        import torch
        from einops import repeat  # noqa: F401 (used by sample fns via closure)
        from hydra import compose, initialize_config_dir
        from src.config import load_typed_config, ModelCfg
        from src.model.compressor import MVAECompressorCfg, MVAECompressor
        from src.model.denoiser import LightningDiTCfg, LightningDiT
        from src.model.scene_generator import SceneGenerator, SceneGeneratorCfg
        from src.model.autoencoder import get_autoencoder, AutoencodersCfg
        from src.model.scheduler import RectifiedFlowMatchingScheduler, RectifiedFlowMatchingSchedulerCfg
        from src.model.sampler.full_sequence import FullSequenceSampler, FullSequenceSamplerCfg
        from src.misc.torch_utils import freeze
        self.args = args
        self.device = args.device

        overrides = [
            "dataset=re10k", f"dataset.root={args.re10k_root}",
            "dataset/view_sampler=evaluation_video", "dataset.view_sampler.max_cond_number=3",
            f"+experiment={args.experiment}",
            f"model.denoiser.ckpt_path={args.scenetok_ckpt}",
            f"model.compressor.ckpt_path={args.scenetok_ckpt}",
            f"model.scene_generator.ckpt_path={args.model_ckpt}",
            "model.scene_generator.load_strict=false",
            "dataset.view_sampler.num_target_views=8",
            "dataset.view_sampler.temporal_downsample=4",
            "dataset.view_sampler.num_context_views=12",
            f"dataset.view_sampler.index_path={args.eval_index}",
            "dataset.precomputed_latents.context=false",
            "dataset.precomputed_latents.target=false",
            "wandb.activated=false",
        ]
        with initialize_config_dir(config_dir=str(REPO / "config"), version_base=None):
            cfg = compose(config_name="main", overrides=overrides)

        self.model_cfg = load_typed_config(cfg.model, ModelCfg)
        self.compressor_cfg = load_typed_config(cfg.model.compressor, MVAECompressorCfg)
        self.denoiser_cfg = load_typed_config(cfg.model.denoiser, LightningDiTCfg)
        self.autoencoders_cfg = load_typed_config(cfg.model.autoencoders, AutoencodersCfg)
        self.scheduler_cfg = load_typed_config(cfg.model.scheduler, RectifiedFlowMatchingSchedulerCfg)
        self.scene_scheduler_cfg = load_typed_config(cfg.model.scene_scheduler, RectifiedFlowMatchingSchedulerCfg)
        self.sampler_cfg = load_typed_config(cfg.sampler, FullSequenceSamplerCfg)
        self.scenegen_cfg = load_typed_config(cfg.model.scene_generator, SceneGeneratorCfg)
        from src.dataset import DatasetRE10kCfg
        self.dataset_cfg = load_typed_config(cfg.dataset, DatasetRE10kCfg)

        # autoencoder pretrained paths are repo-relative; ensure resolvable from cwd
        for k in ("context", "target"):
            pf = getattr(self.autoencoders_cfg, k).pretrained_from
            if pf and not os.path.isabs(pf) and not os.path.exists(pf):
                getattr(self.autoencoders_cfg, k).pretrained_from = str(REPO / pf)

        self.temporal_downsample = 4 if getattr(self.autoencoders_cfg, "target").name in ("video_dc", "wan") else 1
        self.num_scene_tokens = self.compressor_cfg.num_scene_tokens

        print("[scenegen-viser] loading models...")
        self.scheduler = RectifiedFlowMatchingScheduler(**self.scheduler_cfg.kwargs.__dict__)
        self.sampler = FullSequenceSampler(cfg=self.sampler_cfg)
        self.compressor = MVAECompressor(
            cfg=self.compressor_cfg, in_channels=self.autoencoders_cfg.context.kwargs.latent_channels,
            num_views=self.dataset_cfg.view_sampler.num_context_views, temporal_downsample=1,
        ).to(self.device).to(torch.bfloat16)
        self.denoiser = LightningDiT(
            cfg=self.denoiser_cfg, cond_dim=self.compressor_cfg.token_dim,
            num_scene_tokens=self.num_scene_tokens, num_views=self.dataset_cfg.view_sampler.num_target_views,
            temporal_downsample=self.temporal_downsample if not self.model_cfg.force_incorrect else 1,
            using_wan="wan" in getattr(self.autoencoders_cfg, "target").name,
        ).to(self.device).to(torch.bfloat16)
        self.scenegen = SceneGenerator(
            cfg=self.scenegen_cfg, cond_dim=self.compressor_cfg.token_dim,
            num_scene_tokens=self.num_scene_tokens, temporal_downsample=1,
        ).to(self.device).to(torch.bfloat16)
        self.autoencoders = {
            "context": get_autoencoder(self.autoencoders_cfg.context).to(self.device).to(torch.bfloat16),
            "target": get_autoencoder(self.autoencoders_cfg.target).to(self.device).to(torch.bfloat16),
        }
        for m in (self.denoiser, self.compressor, self.scenegen,
                  self.autoencoders["context"], self.autoencoders["target"]):
            freeze(m)
        print("[scenegen-viser] models ready.")

    # ── scene loading ──
    def load_scene(self, scene_id):
        import torch
        from src.dataset import get_dataset
        from src.misc.batch_utils import preprocess_batch, batch_expand, batch_cast
        cfg = self.dataset_cfg
        ds = get_dataset(cfg, stage="test", step_tracker=None)
        ds.overfit_to_scene = [scene_id]
        batch = ds[0]
        batch["context"] = batch_expand(batch["context"])
        batch["target"] = batch_expand(batch["target"])
        batch = preprocess_batch(batch, index=0)   # all poses relative to context[0]
        for k in ("context", "target"):
            batch[k] = batch_cast(batch[k], torch.bfloat16)
            batch[k] = batch_cast(batch[k], torch.device(self.device))

        # Conditioning views for SceneGen: first + last target frames (matches
        # infer_scenegen; num_cond controls how many are actually attended).
        tgt = batch["target"]
        batch["cond"] = {k: tgt[k].clone()[:, [0, -1, -1]] for k in ("extrinsics", "intrinsics", "latent", "index")}
        # Context views for the compressor path: uniform over the target sequence.
        n_ctx = cfg.view_sampler.num_context_views
        ctx_idx = torch.linspace(0, tgt["extrinsics"].shape[1] - 1, n_ctx, device=tgt["extrinsics"].device).long()
        batch["context"] = {k: tgt[k].clone()[:, ctx_idx] for k in ("extrinsics", "intrinsics", "latent", "index")}
        self.batch = batch
        # base target trajectory (relative to ctx0) that the user will edit
        self.base_target_c2w = tgt["extrinsics"][0].float().cpu().numpy().astype(np.float64)
        self.target_K = tgt["intrinsics"][0, 0].clone()      # one K, repeated per view
        self.context_c2w = batch["context"]["extrinsics"][0].float().cpu().numpy().astype(np.float64)
        self.context_images = batch["context"]["latent"][0].float().clamp(0, 1).cpu().numpy()  # (Vc,3,H,W)
        self.tokens = None
        print(f"[scenegen-viser] scene {scene_id}: ctx={self.context_c2w.shape[0]} "
              f"base_target={self.base_target_c2w.shape[0]}")
        return self.base_target_c2w

    # ── token generation (once) ──
    def generate_tokens(self, num_cond=1, guidance=3.0):
        from einops import repeat
        from src.model.types import CameraInputs, SceneGeneratorInputs
        from src.model.diffusion import get_latents
        from src.model.sampler.full_sequence import FullSequenceSampler, FullSequenceSamplerCfg
        b = self.batch
        # Conditioning views = num_cond UNIFORM(diverse) views over the target
        # sequence (generalizes the notebook's [0,-1,-1]; num_cond=1 → frame 0).
        # ⚠️ SceneGen was trained/evaluated with max_cond_number=3 ("a few images");
        # num_cond>3 is out-of-distribution.
        tgt = b["target"]; T_all = tgt["extrinsics"].shape[1]
        nc = max(1, int(num_cond))
        cond_idx = torch.linspace(0, T_all - 1, nc, device=tgt["extrinsics"].device).long()
        cond = {k: tgt[k][:, cond_idx] for k in ("extrinsics", "intrinsics", "latent", "index")}
        cond_latents = get_latents(
            autoencoder=self.autoencoders, inputs=cond, view_type="context",
            precomputed_latents=self.dataset_cfg.precomputed_latents,
            autoencoder_name=self.autoencoders_cfg.context.name,
            scaling_factor=self.autoencoders_cfg.context.kwargs.scaling_factor,
        )
        anchor_pose = CameraInputs(intrinsics=b["context"]["intrinsics"], extrinsics=b["context"]["extrinsics"])
        cond_pose = CameraInputs(intrinsics=cond["intrinsics"], extrinsics=cond["extrinsics"])
        device = cond_latents.device
        scene_sampler = FullSequenceSampler(FullSequenceSamplerCfg(name="full_sequence"))
        scene_sampler.set_scheduling_matrix(
            horizon=self.num_scene_tokens, steps=self.scene_scheduler_cfg.num_inference_steps,
            concurrency=self.num_scene_tokens, device=device, dtype=cond_latents.dtype, cond_mask_indices=None)
        shift = self.scene_scheduler_cfg.kwargs.timestep_shift or 1
        scene_sampler.shift_scheduling_matrix(shift)

        x_t = torch.randn((1, self.num_scene_tokens, self.compressor.output_dim), device=device)
        # cond_mask width = nc (all conditioning views attended). Bypasses the
        # eval sampler's max_cond_number cap so num_cond>3 can be probed (OOD).
        cond_mask = torch.ones((1, nc), device=device, dtype=torch.bool)
        with torch.no_grad():
            for m in range(scene_sampler.global_steps):
                ts, _ = scene_sampler(m); ts_next, _ = scene_sampler(m + 1)
                ts = repeat(ts, "n -> b n", b=1).to(device); ts_next = repeat(ts_next, "n -> b n", b=1).to(device)
                self.scheduler.set_scheduling_matrix(ts_next)
                t = (ts * self.scheduler.num_train_timesteps - 1).clip(min=0)
                gi = SceneGeneratorInputs(view=cond_latents, pose=cond_pose, anchor_pose=anchor_pose,
                                          timestep=t, state=self.scheduler.scale_model_input(x_t, ts).clone())
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred_c, _ = self.scenegen._forward(inputs=gi, cond_mask=cond_mask)
                    if guidance > 1.0:
                        gi.state = self.scheduler.scale_model_input(x_t, ts).clone()
                        pred_u, _ = self.scenegen._forward(inputs=gi, cond_mask=torch.zeros_like(cond_mask))
                        pred = pred_u + guidance * (pred_c - pred_u)
                    else:
                        pred = pred_c
                x_t = self.scheduler.step(pred, ts, x_t).prev_sample
                self.scheduler.unset_scheduling_matrix()
        self.tokens = (x_t / self.scenegen_cfg.scale_factor).bfloat16()
        print(f"[scenegen-viser] generated tokens {tuple(self.tokens.shape)}")
        return self.tokens

    # ── render tokens at (edited) target cameras ──
    @torch.no_grad()
    def render(self, target_c2w_rel):
        import torch
        from einops import repeat
        from src.model.types import CameraInputs, DenoiserInputs
        from src.model.diffusion import last_stage_decode
        if self.tokens is None:
            raise RuntimeError("no scene tokens yet — Generate first")
        dev = self.device
        td = self.temporal_downsample
        ext = torch.as_tensor(np.asarray(target_c2w_rel), dtype=self.target_K.dtype, device=dev)  # (T,4,4)
        T = (ext.shape[0] // td) * td
        ext = ext[:T].unsqueeze(0)                                    # (1,T,4,4)
        K = self.target_K.to(dev).unsqueeze(0).repeat(T, 1, 1).unsqueeze(0)  # (1,T,3,3)
        target_pose = CameraInputs(intrinsics=K, extrinsics=ext)
        num = T // td
        c = self.autoencoders_cfg.target.kwargs.latent_channels
        h, w = self.denoiser_cfg.input_shape
        x_t = torch.randn((1, num, c, h, w), device=dev, dtype=self.tokens.dtype) * self.scheduler.init_noise_sigma
        # clean_targets MUST match the canonical scenegen inference (notebook /
        # infer_scenegen render() → sampler_cfg.clean_targets, =4 for re10k). 0 was
        # wrong and degraded the sampling schedule ("무너짐").
        self.sampler.set_scheduling_matrix(
            horizon=num, steps=self.scheduler_cfg.num_inference_steps,
            concurrency=self.dataset_cfg.view_sampler.num_target_views, device=dev,
            dtype=self.tokens.dtype, cond_mask_indices=None,
            clean_targets=self.sampler_cfg.clean_targets)
        self.sampler.shift_scheduling_matrix(shift=self.scheduler_cfg.kwargs.timestep_shift or 1)
        cond_state = self.denoiser.cnd_proj(self.tokens)
        for m in range(self.sampler.global_steps):
            ts, dmask = self.sampler(m); ts_next, _ = self.sampler(m + 1)
            ts = repeat(ts, "v -> b v", b=1).to(dev); ts_next = repeat(ts_next, "v -> b v", b=1).to(dev)
            self.scheduler.set_scheduling_matrix(ts_next[:, dmask])
            new_dmask = repeat(dmask, "n -> (n t)", t=td)
            t = (ts[:, dmask] * self.scheduler.num_train_timesteps - 1).clip(min=0)
            di = DenoiserInputs(view=self.scheduler.scale_model_input(x_t[:, dmask], ts[:, dmask]).clone(),
                                pose=target_pose[:, new_dmask], timestep=t, state=cond_state)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred, _ = self.denoiser._forward(inputs=di, temporal_downsample=td)
            x_t[:, dmask] = self.scheduler.step(pred, ts[:, dmask], x_t[:, dmask]).prev_sample
            self.scheduler.unset_scheduling_matrix()
        decoded = last_stage_decode(
            autoencoder=self.autoencoders, latents=x_t, view_type="target",
            autoencoder_name=self.autoencoders_cfg.target.name,
            scaling_factor=self.autoencoders_cfg.target.kwargs.scaling_factor)
        return decoded[0].float().clamp(0, 1).cpu()   # (Tpix,3,H,W)


# ─────────────────────────── headless smoke ───────────────────────────
def run_headless(engine, args):
    import torch
    from src.misc.image_io import save_image_video
    engine.load_scene(args.scene)
    engine.generate_tokens(num_cond=args.num_cond, guidance=args.guidance)
    frames = engine.render(engine.base_target_c2w)
    out = Path(args.gen_out) / "scenegen_re10k" / f"{args.scene[:16]}_{time.strftime('%m%d_%H%M%S')}_headless"
    out.mkdir(parents=True, exist_ok=True)
    save_image_video(images=frames, indices=torch.arange(frames.shape[0]), output_dir=out,
                     name="generated", save_img=False, save_video=True, fps=args.fps)
    save_gif(frames, out / "generated.gif", fps=args.fps)
    print(f"[scenegen-viser] HEADLESS OK → {out} frames={tuple(frames.shape)}")


# ─────────────────────────── viser server ───────────────────────────
def run_viser(engine, args):
    import torch
    from src.misc.viser_frustum import start_server, add_view_frustums, _rotmat_to_wxyz, path_points_from_origins
    from src.misc.image_io import save_image_video

    def _from44(M):
        return _rotmat_to_wxyz(M[:3, :3]), tuple(float(v) for v in M[:3, 3])

    server = start_server(host=args.host, port=args.port)
    print(f"[scenegen-viser] server up → http://{args.host}:{args.port}")

    gui_scene = server.gui.add_text("re10k scene id", args.scene)
    gui_status = server.gui.add_text("status", "idle")
    gui_load = server.gui.add_button("Load scene")
    gui_numcond = server.gui.add_number("num_cond", args.num_cond, step=1)
    gui_guid = server.gui.add_number("guidance", args.guidance, step=0.5)
    gui_gen = server.gui.add_button("Generate scene tokens")
    PATTERNS = ["move_forward", "move_back", "move_left", "move_right",
                "rotate_up", "rotate_down", "rotate_left", "rotate_right"]
    gui_pattern = server.gui.add_dropdown("pattern", PATTERNS, initial_value=PATTERNS[0])
    gui_translate = server.gui.add_number("translate amount", 0.3, step=0.01)
    gui_rotate = server.gui.add_number("rotate deg", 20.0, step=1.0)
    gui_apply = server.gui.add_button("Apply pattern")
    gui_sel = server.gui.add_text("target idx (0-5,10 / all)", "all")
    gui_select = server.gui.add_button("Select for edit")
    gui_reset = server.gui.add_button("Reset target")
    gui_render = server.gui.add_button("Render video")

    S = {"ctx_h": [], "tgt_h": [], "tgt_poses": None, "tgt_orig": None,
         "gizmo": None, "tgt_gen": 0, "tgt_path": None, "scale": 0.3}

    def _clear_gizmo():
        if S["gizmo"] is not None:
            try: S["gizmo"].remove()
            except Exception: pass
            S["gizmo"] = None

    def _refresh_path():
        if S["tgt_path"] is not None and S["tgt_poses"] is not None and S["tgt_poses"].shape[0] >= 2:
            S["tgt_path"].points = path_points_from_origins(S["tgt_poses"][:, :3, 3])

    def _set_target(poses):
        poses = np.asarray(poses, dtype=np.float64)
        if poses.ndim == 2: poses = poses[None]
        _clear_gizmo()
        old = {id(h) for h in S["tgt_h"]}
        for h in S["tgt_h"]:
            try: h.remove()
            except Exception: pass
        S["ctx_h"] = [c for c in S["ctx_h"] if id(c) not in old]
        T = poses.shape[0]
        K0 = engine.target_K.float().cpu().numpy()
        intr = np.tile(K0, (T, 1, 1))
        S["tgt_gen"] += 1
        th = add_view_frustums(server, poses, intrinsics=intr, scale=S["scale"] * 0.6,
                               prefix=f"target_v{S['tgt_gen']}", color_start=(40, 220, 120),
                               color_end=(120, 40, 220), path_color=(255, 40, 40),
                               add_world_axes=False, add_gui_color=False, return_all=True)
        S["ctx_h"] += th; S["tgt_h"] = th
        S["tgt_frustums"] = th[:T]
        S["tgt_path"] = next((h for h in th if getattr(h, "_is_frustum_path", False)), None)
        S["tgt_poses"] = poses.copy(); S["tgt_orig"] = poses.copy()
        return T

    def load_scene(_=None):
        try:
            gui_status.value = f"loading scene {gui_scene.value.strip()}..."
            for h in S["ctx_h"]:
                try: h.remove()
                except Exception: pass
            S["ctx_h"] = []; S["tgt_h"] = []
            base = engine.load_scene(gui_scene.value.strip())
            cpos = engine.context_c2w[:, :3, 3]
            if len(cpos) >= 2:
                S["scale"] = float(round(np.linalg.norm(cpos[:, None] - cpos[None], axis=-1).max() * 0.3, 4))
            imgs = (engine.context_images * 255).astype("uint8")
            ch = add_view_frustums(server, engine.context_c2w, intrinsics=None, images=imgs,
                                   scale=S["scale"], prefix="context", color_start=(40, 120, 255),
                                   color_end=(255, 80, 40), path_color=(0, 80, 255), return_all=True)
            S["ctx_h"] += ch
            _set_target(base)
            gui_status.value = f"loaded {gui_scene.value.strip()}: ctx={len(engine.context_c2w)} tgt={base.shape[0]} — Generate tokens next"
            print("[scenegen-viser]", gui_status.value)
        except Exception as e:
            import traceback; traceback.print_exc(); gui_status.value = f"load ERROR: {e}"

    def gen_tokens(_=None):
        try:
            gui_status.value = "generating scene tokens... (~1min)"
            engine.generate_tokens(num_cond=int(gui_numcond.value), guidance=float(gui_guid.value))
            gui_status.value = "scene tokens ready — move target & Render"
        except Exception as e:
            import traceback; traceback.print_exc(); gui_status.value = f"generate ERROR: {e}"

    def select(_=None):
        if S["tgt_poses"] is None:
            gui_status.value = "load a scene first"; return
        sel = _parse_sel(gui_sel.value, S["tgt_poses"].shape[0])
        if not sel:
            gui_status.value = "empty selection"; return
        _clear_gizmo()
        centroid = S["tgt_poses"][sel, :3, 3].mean(0)
        gizmo_init = np.eye(4); gizmo_init[:3, 3] = centroid
        snap = S["tgt_poses"][sel].copy()
        giz = server.scene.add_transform_controls("/edit_gizmo", scale=0.6,
                                                   position=tuple(float(v) for v in centroid))
        def on_move(_=None):
            cur = _pose44(giz.wxyz, giz.position)
            delta = cur @ np.linalg.inv(gizmo_init)
            for j, i in enumerate(sel):
                M = delta @ snap[j]; S["tgt_poses"][i] = M
                w, p = _from44(M); S["tgt_frustums"][i].wxyz = w; S["tgt_frustums"][i].position = p
            _refresh_path()
        giz.on_update(on_move); S["gizmo"] = giz
        gui_status.value = f"editing {len(sel)} target cam(s)"

    def apply_pattern(_=None):
        if S["tgt_orig"] is None:
            gui_status.value = "load a scene first"; return
        T = S["tgt_orig"].shape[0]
        base = S["tgt_orig"][0]
        poses = _make_pattern_poses(gui_pattern.value, T, float(gui_translate.value),
                                    float(gui_rotate.value), base=base)
        _set_target(poses)
        gui_status.value = f"pattern {gui_pattern.value}: T={T}"

    def reset(_=None):
        if S["tgt_orig"] is None:
            gui_status.value = "nothing to reset"; return
        _set_target(engine.base_target_c2w.copy()); gui_status.value = "target reset"

    def render(_=None):
        if engine.tokens is None:
            gui_status.value = "generate scene tokens first"; return
        if S["tgt_poses"] is None:
            gui_status.value = "no target"; return
        try:
            gui_status.value = "rendering... (~1min)"
            frames = engine.render(S["tgt_poses"])
            out = Path(args.gen_out) / "scenegen_re10k" / \
                f"{gui_scene.value.strip()[:16]}_{time.strftime('%m%d_%H%M%S')}"
            out.mkdir(parents=True, exist_ok=True)
            save_image_video(images=frames, indices=torch.arange(frames.shape[0]), output_dir=out,
                             name="generated", save_img=False, save_video=True, fps=args.fps)
            try: save_gif(frames, out / "generated.gif", fps=args.fps)
            except Exception as ge: print("gif fail", ge)
            torch.save({"target_c2w_rel": torch.tensor(S["tgt_poses"], dtype=torch.float32),
                        "scene": gui_scene.value.strip()}, out / "poses.pt")
            gui_status.value = f"saved → {out.name}/generated.mp4 {tuple(frames.shape)}"
            print("[scenegen-viser]", gui_status.value)
        except Exception as e:
            import traceback; traceback.print_exc(); gui_status.value = f"render ERROR: {e}"

    gui_load.on_click(load_scene); gui_gen.on_click(gen_tokens)
    gui_select.on_click(select); gui_apply.on_click(apply_pattern)
    gui_reset.on_click(reset); gui_render.on_click(render)
    load_scene()
    print("[scenegen-viser] panel: Load scene / Generate scene tokens / Apply pattern / Select / Render. Ctrl-C to stop.")
    try:
        while True: time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[scenegen-viser] stopping."); server.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--scene", default="004e9db3337e8206", help="re10k scene id (key in eval index)")
    ap.add_argument("--experiment", default="scenegen_shift12_re10k")
    ap.add_argument("--model_ckpt", default=str(REPO / "checkpoints/scenegen_shift12_re10k.ckpt"))
    ap.add_argument("--scenetok_ckpt", default=str(REPO / "checkpoints/va-videodc_re10k_scene.ckpt"))
    ap.add_argument("--re10k_root", default=DEFAULT_RE10K_ROOT)
    ap.add_argument("--eval_index", default=DEFAULT_EVAL_INDEX)
    ap.add_argument("--num_cond", type=int, default=1)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--infer_steps", type=int, default=None, help="renderer steps (default: config)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gen_out", default=str(REPO / "results/viser_generate"))
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--headless", action="store_true", help="no viser: gen+render original target, save video")
    args = ap.parse_args()

    engine = SceneGenEngine(args)
    if args.infer_steps is not None:
        engine.scheduler_cfg.num_inference_steps = args.infer_steps
    if args.headless:
        run_headless(engine, args)
    else:
        run_viser(engine, args)


if __name__ == "__main__":
    main()
