"""Launch a viser server, visualize view frustums from a JSON-configured bundle,
interactively EDIT target-camera poses, AND generate video from the current
context + (edited) target poses using a loaded SceneTok va-wan model.

Panel:
  - config json (path) + Reload  : swap scene / options live (edit the JSON)
  - context/target start·end color pickers, frustum colors gradient first→last
  - Edit target poses:
      "target idx" text  (e.g. "0-5,10" or "all") + Select
        → one transform gizmo at the centroid of the selected target cams;
          dragging moves ALL selected cameras rigidly (group move).
      Save edited  → writes <bundle>_edited.pt with edited target poses
      Reset        → restore original target poses
  - Generate video : run the loaded model on this scene's context + current
      (edited) target poses → mp4 under <gen_out>/<scene16>_<ts>/.

Model is loaded ONCE at startup (like a server) from --model_ckpt with the
matching --model_experiment, using an `evaluation` view sampler keyed by the
same eval index the bundles were built from — so the dataset reproduces the
bundle's exact 16 context views (correct 256x448 crop + intrinsics + reference
frame). Edited target poses (bundle-relative, == dataset preprocess_batch
frame) are mapped to world via ctx0_abs then re-relativized by preprocess_batch.
Pass --no_model to skip model loading (pure visualization).

Config JSON (default scripts/visualize/viser_config.json):
  {"data": "<bundle.pt/.npz>", "scale": 0.3, "no_target": false}
Bundle keys: c2w (N,4,4 OpenCV), intrinsics, images, scene; optional target_c2w/
target_intrinsics/target_images; optional ref_world (4,4) for save round-trip.

Usage:  CUDA_VISIBLE_DEVICES=3 python scripts/visualize/viser_server.py
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
LAGERNVS_REPO = REPO / "submodules/lagernvs"
LAGERNVS_PY = "/NHNHOME/WORKSPACE/0226010013_A/anaconda3/envs/lagernvs/bin/python"
sys.path.insert(0, str(REPO))
from src.misc.viser_frustum import start_server, add_view_frustums, _rotmat_to_wxyz, path_points_from_origins

DEFAULT_CONFIG = Path(__file__).resolve().parent / "viser_config.json"
DEFAULT_EVAL_INDEX = REPO / "assets/evaluation_index/dl3dv_c16_37_caption_standard.json"

# ckpt stem → (experiment, "H,W"). The compressor feat_rope size differs per
# resolution (256x256 → [64,64], 256x448 → [112,64]), so the config MUST match
# the ckpt or load_state_dict raises a size mismatch (strict=False does NOT
# silence size mismatches). Auto-paired from --model_ckpt when experiment/shape
# are not given explicitly.
CKPT_PRESETS = {
    "va-wan_dl3dv": ("scenetok_va-wan_shift4_dl3dv_finetuned", "256,256"),
    "va-wan_dl3dv_256x448": ("custom/scenetok_va-wan_shift8_dl3dv_finetuned_wide", "256,448"),
}


def save_gif(images, path, fps=12):
    """images: (T,3,H,W) float[0,1] tensor → animated gif."""
    from PIL import Image
    arr = (images.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype("uint8")
    frames = [Image.fromarray(a) for a in arr]
    frames[0].save(str(path), save_all=True, append_images=frames[1:],
                   duration=int(1000 / max(fps, 1)), loop=0)


def load_bundle(path: str):
    p = Path(path)
    if p.suffix == ".npz":
        return dict(np.load(p, allow_pickle=True))
    import torch
    return torch.load(p, map_location="cpu")


def _wxyz_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),     1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),     2*(y*z+x*w),   1-2*(x*x+y*y)],
    ], dtype=np.float64)


def _pose44(wxyz, pos):
    M = np.eye(4); M[:3, :3] = _wxyz_to_R(np.asarray(wxyz, float)); M[:3, 3] = np.asarray(pos, float); return M


def _from44(M):
    return _rotmat_to_wxyz(M[:3, :3]), tuple(float(v) for v in M[:3, 3])


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


def build_model(args):
    """Load the SceneTok va-wan wrapper + an `evaluation`-sampler DL3DV loader
    keyed by the same eval index the bundles were built from. Returns
    (wrapper, loader, device, precision)."""
    import torch
    from omegaconf import OmegaConf
    from hydra import compose, initialize_config_dir
    from torch.utils.data import DataLoader
    from src.config import load_typed_root_config
    from src.dataset import get_dataset
    from src.dataset.data_module import safe_collate
    from src.misc.step_tracker import StepTracker
    from src.model.diffusion_wrapper import DiffusionWrapper

    shape = [int(x) for x in args.model_shape.split(",")]

    def _compose(eval_sampler: bool):
        with initialize_config_dir(config_dir=str(REPO / "config"), version_base=None):
            cfg_dict = compose(
                config_name="main",
                overrides=[f"+experiment={args.model_experiment}", "dataset=dl3dv",
                           "mode=test", "wandb.activated=false"],
            )
        OmegaConf.set_struct(cfg_dict, False)
        for key in ("context_root", "target_root", "map_dict"):
            if key in cfg_dict.dataset:
                del cfg_dict.dataset[key]
        cfg_dict.dataset.root = "./DATA/DL3DV/DL3DV-960"
        cfg_dict.mode = "test"
        cfg_dict.wandb.activated = False
        cfg_dict.data_loader.test.num_workers = 0
        cfg_dict.data_loader.test.batch_size = 1
        cfg_dict.freeze.denoiser = True
        cfg_dict.freeze.compressor = True
        cfg_dict.freeze.autoencoder = True
        cfg_dict.dataset.smallset = False
        cfg_dict.dataset.stage_override = "train"
        cfg_dict.dataset.val_seen = True
        cfg_dict.dataset.scene_id = None
        cfg_dict.dataset.context_shape = shape
        cfg_dict.dataset.target_shape = shape
        cfg_dict.dataset.evaluation_index_path = str(args.eval_index)
        if eval_sampler:
            # Loader-only: evaluation sampler reproduces each bundle's exact
            # context/target views (eval-index order) instead of bounded random.
            cfg_dict.dataset.view_sampler = OmegaConf.create({
                "name": "evaluation", "index_path": str(args.eval_index),
                "num_context_views": 16, "num_target_views": 36})
        cfg_dict.model.cfg_scale = args.cfg_scale
        cfg_dict.model.scheduler.num_inference_steps = args.infer_steps
        if "noise_seed" in cfg_dict.model.denoiser:
            cfg_dict.model.denoiser.noise_seed = args.seed
        OmegaConf.set_struct(cfg_dict, True)
        return load_typed_root_config(cfg_dict)

    # Wrapper keeps the experiment's own (bounded) view_sampler — generate_batch_
    # with_scene reads its chunk_index_gap/chunk_targets/temporal_downsample.
    print(f"[viser-model] hydra-compose: {args.model_experiment}")
    cfg = _compose(eval_sampler=False)
    step_tracker = StepTracker(0)
    wrapper = DiffusionWrapper(
        model_cfg=cfg.model, dataset_cfg=cfg.dataset, freeze_cfg=cfg.freeze,
        optimizer_cfg=cfg.optimizer, test_cfg=cfg.test, train_cfg=cfg.train,
        val_cfg=cfg.val, sampler_cfg=cfg.sampler, step_tracker=step_tracker,
        output_dir=None, batch_size=1, val_check_interval=cfg.trainer.val_check_interval,
        mode="test")
    state = torch.load(Path(args.model_ckpt), map_location="cpu", weights_only=False)
    sd = state["state_dict"] if "state_dict" in state else state
    missing, unexpected = wrapper.load_state_dict(sd, strict=False)
    print(f"[viser-model] state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    wrapper.eval().to(args.device)
    if hasattr(wrapper, "sampler") and hasattr(wrapper.sampler, "log_vis"):
        wrapper.sampler.log_vis = lambda *a, **kw: None

    # Separate loader with the evaluation sampler (exact bundle context).
    cfg_eval = _compose(eval_sampler=True)
    ds = get_dataset(cfg_eval.dataset, "test", step_tracker, generator=None, force_shuffle=False)
    loader = DataLoader(ds, batch_size=1, num_workers=0, collate_fn=safe_collate, shuffle=False)
    precision = (torch.bfloat16 if cfg.trainer.precision in ("bf16-mixed", "bf16", "bf16-true")
                 else torch.float32)
    print(f"[viser-model] ready on {args.device} (precision={precision})")
    return wrapper, loader, args.device, precision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    # in-server model generation
    ap.add_argument("--no_model", action="store_true", help="skip model load (pure viz)")
    # experiment/shape default to None → auto-derived from the ckpt name (see
    # CKPT_PRESETS) so swapping --model_ckpt alone uses the matching config.
    ap.add_argument("--model_experiment", default=None)
    ap.add_argument("--model_ckpt", default=str(REPO / "checkpoints/va-wan_dl3dv.ckpt"))
    ap.add_argument("--model_shape", default=None)
    ap.add_argument("--eval_index", default=str(DEFAULT_EVAL_INDEX))
    ap.add_argument("--infer_steps", type=int, default=25)
    ap.add_argument("--cfg_scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gen_out", default=str(REPO / "results/viser_generate"))
    ap.add_argument("--fps", type=int, default=15)
    # ── LagerNVS (separate `lagernvs` conda env, run as subprocess) ──
    ap.add_argument("--lagernvs_python", default=LAGERNVS_PY)
    ap.add_argument("--lagernvs_repo", default=str(LAGERNVS_REPO))
    ap.add_argument("--lagernvs_ckpt",
                    default=str(LAGERNVS_REPO / "checkpoints/lagernvs_general_512/model.pt"))
    ap.add_argument("--lagernvs_size", type=int, default=512)
    ap.add_argument("--lagernvs_gpu", default=None,
                    help="CUDA_VISIBLE_DEVICES for the LagerNVS subprocess (default: inherit)")
    args = ap.parse_args()

    # auto-pair experiment/shape from the ckpt name when not given explicitly
    stem = Path(args.model_ckpt).stem
    exp_d, shape_d = CKPT_PRESETS.get(stem, ("scenetok_va-wan_shift4_dl3dv_finetuned", "256,256"))
    if args.model_experiment is None:
        args.model_experiment = exp_d
    if args.model_shape is None:
        args.model_shape = shape_d
    if stem not in CKPT_PRESETS:
        print(f"[viser] WARN: unknown ckpt '{stem}', using experiment={args.model_experiment} "
              f"shape={args.model_shape}; pass --model_experiment/--model_shape if it mismatches.")
    print(f"[viser] model: {stem} → experiment={args.model_experiment} shape={args.model_shape}")

    server = start_server(host=args.host, port=args.port)
    print(f"[viser] server up → http://{args.host}:{args.port}")

    gui_cfg = server.gui.add_text("config json", args.config)
    gui_status = server.gui.add_text("status", "idle")
    gui_reload = server.gui.add_button("Reload")
    # ── preset target-camera patterns (before gizmo editing) ──
    PATTERNS = ["move_forward", "move_back", "move_left", "move_right",
                "rotate_up", "rotate_down", "rotate_left", "rotate_right"]
    gui_pattern = server.gui.add_dropdown("pattern", PATTERNS, initial_value=PATTERNS[0])
    gui_translate = server.gui.add_number("translate amount", 0.5, step=0.01)
    gui_rotate = server.gui.add_number("rotate deg", 30.0, step=1.0)
    gui_apply = server.gui.add_button("Apply pattern")
    # ── load target poses from a saved .pt ──
    gui_ptpath = server.gui.add_text("target .pt path", "")
    gui_loadpt = server.gui.add_button("Load target .pt")
    # ── manual gizmo editing ──
    gui_sel = server.gui.add_text("target idx (e.g. 0-5,10 / all)", "all")
    gui_select = server.gui.add_button("Select for edit")
    gui_save = server.gui.add_button("Save edited")
    gui_reset = server.gui.add_button("Reset target")
    gui_generate = server.gui.add_button("Generate video")
    gui_generate_lager = server.gui.add_button("Generate (LagerNVS)")

    S = {"current": [], "tgt_frustums": [], "tgt_path": None, "tgt_poses": None,
         "tgt_orig": None, "gizmo": None, "bundle_path": None,
         "tgt_handles": [], "bundle": None, "scale": 0.3,
         "tgt_gen": 0, "tgt_reset": None}
    # model state (loaded once at startup)
    G = {"wrapper": None, "loader": None, "device": args.device, "precision": None,
         "cache": {}, "scanned": False}

    if not args.no_model:
        try:
            gui_status.value = "loading model..."
            w, ld, dev, prec = build_model(args)
            G.update(wrapper=w, loader=ld, device=dev, precision=prec)
            gui_status.value = f"model ready ({Path(args.model_ckpt).name})"
        except Exception as e:
            import traceback; traceback.print_exc()
            gui_status.value = f"model load FAILED: {e}"
            print("[viser] model load failed:", e)

    def _find_batch(scene_hash):
        """Return the dataset batch whose scene contains scene_hash (cached)."""
        if scene_hash in G["cache"]:
            return G["cache"][scene_hash]
        if G["loader"] is None:
            return None
        for b in G["loader"]:
            if b is None:
                continue
            name = b["scene"][0] if isinstance(b.get("scene"), (list, tuple)) else b.get("scene")
            if name is None:
                continue
            G["cache"][str(name)] = b
            if scene_hash in str(name):
                G["cache"][scene_hash] = b
                return b
        return None

    def _refresh_path():
        if S["tgt_path"] is not None and S["tgt_poses"] is not None and S["tgt_poses"].shape[0] >= 2:
            S["tgt_path"].points = path_points_from_origins(S["tgt_poses"][:, :3, 3])

    def _clear_gizmo():
        if S["gizmo"] is not None:
            try: S["gizmo"].remove()
            except Exception: pass
            S["gizmo"] = None

    def reload(_=None):
        _clear_gizmo()
        for h in S["current"]:
            try: h.remove()
            except Exception: pass
        S["current"] = []; S["tgt_frustums"] = []; S["tgt_path"] = None
        S["tgt_poses"] = None; S["tgt_orig"] = None; S["tgt_handles"] = []
        try:
            cfg = json.loads(Path(gui_cfg.value.strip()).read_text())
            S["bundle_path"] = cfg["data"]
            d = load_bundle(cfg["data"]); scale = float(cfg.get("scale", 0.3))
            S["bundle"] = d; S["scale"] = scale
            ch = add_view_frustums(server, d["c2w"], intrinsics=d.get("intrinsics"),
                                   images=d.get("images"), scale=scale, prefix="context",
                                   color_start=(40,120,255), color_end=(255,80,40),
                                   path_color=(0,80,255), return_all=True)
            S["current"] += ch
            # default translate amount = farthest pairwise distance among context cams
            cpos = np.asarray(d["c2w"], dtype=np.float64)[:, :3, 3]
            if len(cpos) >= 2:
                dmat = np.linalg.norm(cpos[:, None] - cpos[None], axis=-1)
                gui_translate.value = float(round(dmat.max(), 4))
            nt = 0
            if not cfg.get("no_target", False) and "target_c2w" in d:
                tc = np.asarray(d["target_c2w"], dtype=np.float64)
                nt = tc.shape[0]
                th = add_view_frustums(server, tc, intrinsics=d.get("target_intrinsics"),
                                       images=d.get("target_images"), scale=scale*0.6, prefix="target",
                                       color_start=(40,220,120), color_end=(120,40,220),
                                       path_color=(255,40,40), add_world_axes=False, return_all=True)
                S["current"] += th; S["tgt_handles"] = th
                S["tgt_frustums"] = th[:nt]                 # frustum handles come first
                S["tgt_path"] = next((h for h in th if getattr(h, "_is_frustum_path", False)), None)
                S["tgt_poses"] = tc.copy(); S["tgt_orig"] = tc.copy()
                S["tgt_reset"] = tc.copy()                  # original bundle target → Reset baseline
            gui_status.value = f"loaded {Path(cfg['data']).name}: ctx={len(d['c2w'])} tgt={nt}"
            print("[viser]", gui_status.value)
        except Exception as e:
            gui_status.value = f"ERROR: {e}"; print("[viser] reload error:", e)

    def select(_=None):
        if S["tgt_poses"] is None:
            gui_status.value = "no target to edit"; return
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
            delta = cur @ np.linalg.inv(gizmo_init)         # rigid delta in world
            for j, i in enumerate(sel):
                M = delta @ snap[j]
                S["tgt_poses"][i] = M
                w, p = _from44(M)
                S["tgt_frustums"][i].wxyz = w
                S["tgt_frustums"][i].position = p
            _refresh_path()                                  # 경로선도 같이 갱신
        giz.on_update(on_move)
        S["gizmo"] = giz
        gui_status.value = f"editing {len(sel)} target cam(s): {sel[:6]}{'...' if len(sel)>6 else ''}"

    def save(_=None):
        if S["tgt_poses"] is None:
            gui_status.value = "nothing to save"; return
        import torch
        out = Path(S["bundle_path"]).with_name(Path(S["bundle_path"]).stem + "_edited.pt")
        d = load_bundle(S["bundle_path"])
        payload = {"target_c2w_edited": torch.tensor(S["tgt_poses"], dtype=torch.float32),
                   "frame": "viser-relative-OpenCV (same as bundle target_c2w)",
                   "source_bundle": str(S["bundle_path"])}
        if "ref_world" in d: payload["ref_world"] = d["ref_world"]
        torch.save(payload, out)
        gui_status.value = f"saved → {out.name}"; print("[viser] saved", out)

    def reset(_=None):
        # Restore the ORIGINAL bundle target (undo patterns / .pt loads / gizmo
        # edits). Rebuild via _set_target so frustums refresh reliably.
        base = S["tgt_reset"] if S["tgt_reset"] is not None else S["tgt_orig"]
        if base is None:
            gui_status.value = "nothing to reset"; return
        _set_target(base.copy())
        gui_status.value = "target reset to bundle original"
        print("[viser] target reset")

    def _rotx(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)

    def _roty(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)

    def _make_pattern_poses(name, T, amount, deg, base=None):
        """Trajectory anchored at `base` (4x4 c2w; default identity), ramping a
        LOCAL move/rotate over T frames: pose[i] = base @ delta_local(i). So the
        pattern is a relative-pose offset from `base` in base's own camera frame
        (move along base's ±X/±Z, rotate about base's local axes).
        OpenCV camera local axes: +X right, +Y down, +Z forward."""
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

    def _set_target(poses):
        """Replace the target-camera frustums/path with `poses` (N,4,4 in bundle
        frame) and reset tgt_poses/tgt_orig. Used by pattern + .pt loaders."""
        poses = np.asarray(poses, dtype=np.float64)
        if poses.ndim == 2:
            poses = poses[None]
        _clear_gizmo()
        old_ids = {id(h) for h in S["tgt_handles"]}
        for h in S["tgt_handles"]:
            try: h.remove()
            except Exception: pass
        # identity-based removal — `in`/`.remove` would trigger handle __eq__
        # (element-wise numpy compare on .points → broadcast error)
        S["current"] = [c for c in S["current"] if id(c) not in old_ids]
        d = S["bundle"] or {}; scale = S["scale"]; T = poses.shape[0]
        K0 = np.asarray(d["target_intrinsics"])[0] if "target_intrinsics" in d else (
            np.asarray(d["intrinsics"])[0] if "intrinsics" in d else None)
        intr = np.tile(K0, (T, 1, 1)) if K0 is not None else None
        # Unique node prefix each call: re-adding the SAME /target/* names right
        # after removing them makes viser drop the update (stale frustums). A fresh
        # prefix guarantees the new poses render.
        S["tgt_gen"] += 1
        th = add_view_frustums(server, poses, intrinsics=intr, scale=scale * 0.6,
                               prefix=f"target_v{S['tgt_gen']}", color_start=(40, 220, 120),
                               color_end=(120, 40, 220), path_color=(255, 40, 40),
                               add_world_axes=False, add_gui_color=False, return_all=True)
        S["current"] += th; S["tgt_handles"] = th
        S["tgt_frustums"] = th[:T]
        S["tgt_path"] = next((h for h in th if getattr(h, "_is_frustum_path", False)), None)
        S["tgt_poses"] = poses.copy(); S["tgt_orig"] = poses.copy()
        return T

    def apply_pattern(_=None):
        if S["bundle"] is None:
            gui_status.value = "load a bundle first (Reload)"; return
        T = S["tgt_orig"].shape[0] if S["tgt_orig"] is not None else 37
        amount = float(gui_translate.value); deg = float(gui_rotate.value)
        name = gui_pattern.value
        # Anchor at the FIRST TARGET camera (its original pose+orientation) and
        # generate the pattern as a relative-pose offset from it — don't transform
        # the existing target cameras. Falls back to context[0] (identity) if the
        # bundle has no target.
        d = S["bundle"]
        if "target_c2w" in d:
            base = np.asarray(d["target_c2w"], dtype=np.float64)[0]
        else:
            base = np.eye(4, dtype=np.float64)
        poses = _make_pattern_poses(name, T, amount, deg, base=base)
        _set_target(poses)
        gui_status.value = f"pattern {name} @target0: T={T} amt={amount:.3f} deg={deg:.0f}"
        print("[viser]", gui_status.value)

    def load_target_pt(_=None):
        """Load target camera poses from a saved .pt (bundle frame). Accepts a
        dict with target_c2w_edited / target_c2w / poses / c2w, or a raw
        tensor/array of shape (N,4,4)."""
        path = gui_ptpath.value.strip()
        if not path:
            gui_status.value = "enter a .pt path"; return
        if S["bundle"] is None:
            gui_status.value = "load a bundle first (Reload)"; return
        try:
            import torch
            obj = torch.load(path, map_location="cpu", weights_only=False)
            poses = None
            if isinstance(obj, dict):
                for k in ("target_c2w_edited", "target_c2w", "poses", "c2w", "extrinsics"):
                    if k in obj:
                        poses = obj[k]; break
                if poses is None:  # single-entry dict → take the only value
                    vals = [v for v in obj.values() if hasattr(v, "shape")]
                    poses = vals[0] if len(vals) == 1 else None
            else:
                poses = obj
            if poses is None:
                gui_status.value = "no pose tensor found in .pt"; return
            if hasattr(poses, "detach"):
                poses = poses.detach().cpu().numpy()
            poses = np.asarray(poses, dtype=np.float64)
            if poses.ndim == 2 and poses.shape == (4, 4):
                poses = poses[None]
            if poses.ndim != 3 or poses.shape[-2:] != (4, 4):
                gui_status.value = f"bad pose shape {poses.shape}, want (N,4,4)"; return
            T = _set_target(poses)
            gui_status.value = f"loaded {Path(path).name}: {T} target poses"
            print("[viser]", gui_status.value)
        except Exception as e:
            gui_status.value = f"load .pt ERROR: {e}"; print("[viser] load .pt error:", e)

    def generate(_=None):
        if G["wrapper"] is None:
            gui_status.value = "model not loaded (--no_model?)"; return
        if S["tgt_poses"] is None or S["bundle_path"] is None:
            gui_status.value = "no target poses to generate"; return
        import torch
        from src.misc.batch_utils import preprocess_batch
        from src.misc.image_io import save_image_video
        try:
            bundle = load_bundle(S["bundle_path"])
            scene_hash = str(bundle.get("scene", Path(S["bundle_path"]).stem))
            gui_status.value = f"generating {scene_hash[:12]}... (this takes ~1min)"
            print(f"[viser-gen] scene={scene_hash}")
            batch = _find_batch(scene_hash)
            if batch is None:
                gui_status.value = f"scene {scene_hash[:12]} not found in eval loader"; return
            dev = G["device"]

            def _to_dev(o):
                if torch.is_tensor(o): return o.to(dev)
                if isinstance(o, dict): return {k: _to_dev(v) for k, v in o.items()}
                if isinstance(o, list): return [_to_dev(v) for v in o]
                return o
            b = _to_dev(batch)

            ctx_ext = b["context"]["extrinsics"]                 # (1,V,4,4) absolute (world)
            ctx_lat = b["context"]["latent"]
            v_c = ctx_ext.shape[1]
            ctx0_abs = ctx_ext[0, 0]                             # dataset ctx0 == bundle ctx0
            edited_rel = torch.tensor(S["tgt_poses"], dtype=ctx_ext.dtype, device=dev)  # (T,4,4) rel→ctx0
            edited_abs = ctx0_abs.unsqueeze(0) @ edited_rel      # back to world; preprocess re-relativizes
            T = edited_abs.shape[0]
            # Target intrinsics MUST be the dataset's GT target K (NOT context K):
            # config sets scale_context_focal_by_256 → context focal is scaled ~15x,
            # which is wrong for target rays. DL3DV uses one K per scene → repeat [0].
            gt_tgt_int = b["target"]["intrinsics"]
            tgt_int = gt_tgt_int[0, 0].unsqueeze(0).repeat(T, 1, 1).to(gt_tgt_int.dtype)
            b["target"] = {
                "extrinsics": edited_abs.unsqueeze(0),
                "intrinsics": tgt_int.unsqueeze(0),
                "latent": torch.zeros((1, T, ctx_lat.shape[2], ctx_lat.shape[3], ctx_lat.shape[4]),
                                      device=dev, dtype=ctx_lat.dtype),
                "index": torch.arange(T, device=dev).unsqueeze(0),
            }
            # Reference frame = FIRST context view (index 0). Target poses (bundle
            # frame / preset patterns) are defined identity-at-context[0], so making
            # context[0] the generation reference keeps a pattern's identity start
            # truly identity (no context[0]-vs-context[v_c//2] rotation leak). The
            # model was trained with a RANDOM context reference, so index 0 is valid
            # (bundle-GT PSNR 18.7 vs 19.2 at v_c//2 — negligible).
            # Sanity (before preprocess): bundle-derived world poses vs dataset GT world.
            ds_ctx_abs = ctx_ext[0].float().cpu().numpy().astype(np.float64)
            ds_ctx_rel0 = np.linalg.inv(ds_ctx_abs[0])[None] @ ds_ctx_abs   # relative to ctx0
            bundle_ctx = np.asarray(bundle["c2w"], dtype=np.float64)
            n = min(len(bundle_ctx), len(ds_ctx_rel0))
            diff = float(np.abs(bundle_ctx[:n] - ds_ctx_rel0[:n]).max())
            print(f"[viser-gen] context frame check: max|bundle-dataset|={diff:.4f} (want ~0)")
            if diff > 0.1:
                print("[viser-gen] WARN: context frames diverge — edited poses may be misaligned")
            b = preprocess_batch(b, index=0)

            wrapper = G["wrapper"]; prec = G["precision"]
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=prec,
                                                     enabled=(prec != torch.float32)):
                sampled, _, _ = wrapper.generate_batch_with_scene(b, wrapper.sampler, repeat_factor=1)
            sampled = sampled.float().clamp(0, 1)

            ckpt_name = Path(args.model_ckpt).stem
            out_dir = Path(args.gen_out) / ckpt_name / f"{scene_hash[:16]}_{time.strftime('%m%d_%H%M%S')}"
            out_dir.mkdir(parents=True, exist_ok=True)
            save_image_video(images=sampled[0], indices=torch.arange(sampled.shape[1]),
                             output_dir=out_dir, name="generated", save_img=False,
                             save_video=True, fps=args.fps)
            try:
                save_gif(sampled[0], out_dir / "generated.gif", fps=args.fps)
            except Exception as ge:
                print("[viser-gen] gif save failed:", ge)
            torch.save({"target_c2w_edited": torch.tensor(S["tgt_poses"], dtype=torch.float32),
                        "scene": scene_hash, "source_bundle": str(S["bundle_path"]),
                        "context_frame_check": diff}, out_dir / "poses.pt")

            # Screenshot the viser 3D scene from each connected client's viewpoint.
            shots = 0
            try:
                clients = server.get_clients()
                from PIL import Image
                for cid, cl in clients.items():
                    try:
                        asp = float(getattr(cl.camera, "aspect", 0) or 0)
                        h = 900; wpx = int(round(h * asp)) if asp > 0 else 1600
                        img = cl.get_render(height=h, width=wpx, transport_format="png")
                        name = "viser_screenshot.png" if len(clients) == 1 else f"viser_screenshot_client{cid}.png"
                        Image.fromarray(img).save(out_dir / name)
                        shots += 1
                    except Exception as ce:
                        print(f"[viser-gen] screenshot client {cid} failed: {ce}")
                if not clients:
                    print("[viser-gen] no connected client → no screenshot")
            except Exception as se:
                print("[viser-gen] screenshot error:", se)

            gui_status.value = (f"saved → {out_dir.name}/generated.mp4 {tuple(sampled.shape)}"
                                f"{f' + {shots} screenshot' if shots else ''}")
            print(f"[viser-gen] saved → {out_dir} (screenshots={shots})")
        except Exception as e:
            import traceback; traceback.print_exc()
            gui_status.value = f"generate ERROR: {e}"; print("[viser-gen] error:", e)

    def generate_lagernvs(_=None):
        """Render the current (edited) target cameras with LagerNVS general_512.

        Runs in the separate `lagernvs` conda env as a subprocess (env mismatch):
        writes context images + context/target c2w + target intrinsics to a
        payload, calls scripts/visualize/lagernvs_infer.py with the lagernvs
        python, reads the rendered mp4 back. Uses the bundle's RAW context images
        (LagerNVS needs RGB, not VA latents)."""
        if S["tgt_poses"] is None or S["bundle_path"] is None:
            gui_status.value = "no target poses to generate"; return
        import torch
        from PIL import Image
        try:
            bundle = load_bundle(S["bundle_path"])
            scene_hash = str(bundle.get("scene", Path(S["bundle_path"]).stem))
            if "images" not in bundle or bundle["images"] is None:
                gui_status.value = "bundle has no context images (LagerNVS needs RGB)"; return
            gui_status.value = f"[LagerNVS] generating {scene_hash[:12]}... (subprocess)"
            print(f"[viser-lagernvs] scene={scene_hash}")
            out_dir = Path(args.gen_out) / "lagernvs_general_512" / \
                f"{scene_hash[:16]}_{time.strftime('%m%d_%H%M%S')}"
            img_dir = out_dir / "context_images"
            img_dir.mkdir(parents=True, exist_ok=True)

            imgs = np.asarray(bundle["images"])          # (Vc,3,H,W) uint8
            img_paths = []
            for i in range(imgs.shape[0]):
                arr = imgs[i].transpose(1, 2, 0).astype("uint8")   # HWC RGB
                p = img_dir / f"ctx_{i:03d}.png"
                Image.fromarray(arr).save(p); img_paths.append(str(p))

            tgt_c2w = torch.as_tensor(np.asarray(S["tgt_poses"]), dtype=torch.float32)
            ctx_c2w = torch.as_tensor(np.asarray(bundle["c2w"]), dtype=torch.float32)
            # GT target intrinsics (normalized), one K per DL3DV scene → repeat.
            tk = bundle.get("target_intrinsics", bundle["intrinsics"])
            tk0 = torch.as_tensor(np.asarray(tk), dtype=torch.float32)[0]
            tgt_K = tk0.unsqueeze(0).repeat(tgt_c2w.shape[0], 1, 1)

            payload = out_dir / "payload.pt"
            torch.save({"context_image_paths": img_paths, "context_c2w": ctx_c2w,
                        "target_c2w": tgt_c2w, "target_intrinsics_norm": tgt_K,
                        "scene": scene_hash}, payload)

            out_mp4 = out_dir / "generated.mp4"
            cmd = [args.lagernvs_python, str(REPO / "scripts/visualize/lagernvs_infer.py"),
                   "--repo", args.lagernvs_repo, "--payload", str(payload),
                   "--output", str(out_mp4), "--ckpt", args.lagernvs_ckpt,
                   "--target_size", str(args.lagernvs_size)]
            env = dict(os.environ)
            if args.lagernvs_gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(args.lagernvs_gpu)
            r = subprocess.run(cmd, cwd=args.lagernvs_repo, env=env,
                               capture_output=True, text=True)
            if r.stdout:
                print(r.stdout[-3000:])
            if r.returncode != 0 or not out_mp4.exists():
                if r.stderr:
                    print("[viser-lagernvs] stderr:\n", r.stderr[-3000:])
                gui_status.value = f"[LagerNVS] FAILED (rc={r.returncode}) — see console"; return

            torch.save({"target_c2w_edited": tgt_c2w, "scene": scene_hash,
                        "source_bundle": str(S["bundle_path"])}, out_dir / "poses.pt")
            gui_status.value = f"[LagerNVS] saved → {out_dir.name}/generated.mp4"
            print(f"[viser-lagernvs] saved → {out_dir}")
        except Exception as e:
            import traceback; traceback.print_exc()
            gui_status.value = f"[LagerNVS] ERROR: {e}"; print("[viser-lagernvs] error:", e)

    gui_reload.on_click(reload); gui_select.on_click(select)
    gui_save.on_click(save); gui_reset.on_click(reset)
    gui_apply.on_click(apply_pattern); gui_loadpt.on_click(load_target_pt)
    gui_generate.on_click(generate)
    gui_generate_lager.on_click(generate_lagernvs)
    reload()
    print("[viser] panel: Reload / Select for edit / Save edited / Reset / Generate video. Ctrl-C to stop.")
    try:
        while True: time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[viser] stopping."); server.stop()


if __name__ == "__main__":
    main()
