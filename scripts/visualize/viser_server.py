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
import argparse, json, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.misc.viser_frustum import start_server, add_view_frustums, _rotmat_to_wxyz, path_points_from_origins

DEFAULT_CONFIG = Path(__file__).resolve().parent / "viser_config.json"
DEFAULT_EVAL_INDEX = REPO / "assets/evaluation_index/dl3dv_c16_37_caption_standard.json"


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
    ap.add_argument("--model_experiment", default="scenetok_va-wan_shift4_dl3dv_finetuned")
    ap.add_argument("--model_ckpt", default=str(REPO / "checkpoints/va-wan_dl3dv.ckpt"))
    ap.add_argument("--model_shape", default="256,256")
    ap.add_argument("--eval_index", default=str(DEFAULT_EVAL_INDEX))
    ap.add_argument("--infer_steps", type=int, default=25)
    ap.add_argument("--cfg_scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gen_out", default=str(REPO / "results/viser_generate"))
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()

    server = start_server(host=args.host, port=args.port)
    print(f"[viser] server up → http://{args.host}:{args.port}")

    gui_cfg = server.gui.add_text("config json", args.config)
    gui_status = server.gui.add_text("status", "idle")
    gui_reload = server.gui.add_button("Reload")
    gui_sel = server.gui.add_text("target idx (e.g. 0-5,10 / all)", "all")
    gui_select = server.gui.add_button("Select for edit")
    gui_save = server.gui.add_button("Save edited")
    gui_reset = server.gui.add_button("Reset target")
    gui_generate = server.gui.add_button("Generate video")

    S = {"current": [], "tgt_frustums": [], "tgt_path": None, "tgt_poses": None,
         "tgt_orig": None, "gizmo": None, "bundle_path": None}
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
        S["tgt_poses"] = None; S["tgt_orig"] = None
        try:
            cfg = json.loads(Path(gui_cfg.value.strip()).read_text())
            S["bundle_path"] = cfg["data"]
            d = load_bundle(cfg["data"]); scale = float(cfg.get("scale", 0.3))
            ch = add_view_frustums(server, d["c2w"], intrinsics=d.get("intrinsics"),
                                   images=d.get("images"), scale=scale, prefix="context",
                                   color_start=(40,120,255), color_end=(255,80,40),
                                   path_color=(0,80,255), return_all=True)
            S["current"] += ch
            nt = 0
            if not cfg.get("no_target", False) and "target_c2w" in d:
                tc = np.asarray(d["target_c2w"], dtype=np.float64)
                nt = tc.shape[0]
                th = add_view_frustums(server, tc, intrinsics=d.get("target_intrinsics"),
                                       images=d.get("target_images"), scale=scale*0.6, prefix="target",
                                       color_start=(40,220,120), color_end=(120,40,220),
                                       path_color=(255,40,40), add_world_axes=False, return_all=True)
                S["current"] += th
                S["tgt_frustums"] = th[:nt]                 # frustum handles come first
                S["tgt_path"] = next((h for h in th if getattr(h, "_is_frustum_path", False)), None)
                S["tgt_poses"] = tc.copy(); S["tgt_orig"] = tc.copy()
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
        if S["tgt_orig"] is None: return
        _clear_gizmo()
        S["tgt_poses"] = S["tgt_orig"].copy()
        for i, h in enumerate(S["tgt_frustums"]):
            w, p = _from44(S["tgt_poses"][i]); h.wxyz = w; h.position = p
        _refresh_path()
        gui_status.value = "target poses reset"

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
            # Reference frame = MIDDLE context view (v_c//2), matching the canonical
            # eval_context_view_count sweep — the model's camera-ray embedding was
            # trained with this reference, not index 0.
            # Sanity (before preprocess): bundle-derived world poses vs dataset GT world.
            ds_ctx_abs = ctx_ext[0].float().cpu().numpy().astype(np.float64)
            ds_ctx_rel0 = np.linalg.inv(ds_ctx_abs[0])[None] @ ds_ctx_abs   # relative to ctx0
            bundle_ctx = np.asarray(bundle["c2w"], dtype=np.float64)
            n = min(len(bundle_ctx), len(ds_ctx_rel0))
            diff = float(np.abs(bundle_ctx[:n] - ds_ctx_rel0[:n]).max())
            print(f"[viser-gen] context frame check: max|bundle-dataset|={diff:.4f} (want ~0)")
            if diff > 0.1:
                print("[viser-gen] WARN: context frames diverge — edited poses may be misaligned")
            b = preprocess_batch(b, index=v_c // 2)

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

    gui_reload.on_click(reload); gui_select.on_click(select)
    gui_save.on_click(save); gui_reset.on_click(reset)
    gui_generate.on_click(generate)
    reload()
    print("[viser] panel: Reload / Select for edit / Save edited / Reset / Generate video. Ctrl-C to stop.")
    try:
        while True: time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[viser] stopping."); server.stop()


if __name__ == "__main__":
    main()
