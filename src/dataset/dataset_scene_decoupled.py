"""Dataset for the CineScene **Scene-Decoupled Video Dataset** (KlingTeam,
arXiv 2602.06959).

Layout (under ``<root>``)::

    camera/{whuman,wohuman}/<scene_id>/<scene_id>_cam.json
    panorama/<scene_id>/<scene_id>_pano.jpeg            # shared 360° equirect
    video/{whuman,wohuman}/<scene_id>/<scene_id>_<NN>_24mm.mp4
    text/whuman/<scene_id>/action.json                 # caption (+ action-only)

Each scene has **7 camera trajectories** (NN = 01..07); each trajectory is one
81-frame clip (672×384, 15 fps). ``whuman`` (with subject) and ``wohuman``
(background only) share the *same* scene_id, the *same* per-trajectory camera
path (verified byte-identical), and a shared panorama.

We emit one training sample per **(scene_id, trajectory)** pair (~7× the scene
count). The default **decoupled** mode mirrors the dataset's purpose and this
project's research goal — *background NVS + text-driven dynamic foreground*:

  * **context** views (scene-token input)  ← ``wohuman`` (background-only) video
  * **target** views (video to generate)   ← ``whuman``  (with-subject) video
  * **text** (foreground driver)            ← ``caption_action_only``

This reuses the wrapper's existing context/target split (the same mechanism
``dataset_dynamicverse`` uses with ``target_video_name``), so the existing
SceneTok + Wan TI2V (T2VWrapper) training runs unchanged on this data.

Camera JSON note: ``<scene_id>_cam.json`` is ``{frame_str: {view_str: "<4×4>"}}``
where the 4×4 is an Unreal-Engine **row-vector** camera-to-world (last row =
camera position, cm; z-up, left-handed). We parse it to a column-convention c2w
(``c2w = M.T``), optionally convert the camera-local axes to OpenCV
(x-right/y-down/z-forward) and the world to right-handed, and (default) apply
per-scene scene-radius normalization — matching ``dataset_dynamicverse``'s
flag-driven, validated approach. Intrinsics are derived from the 24 mm focal
length + sensor width (normalized [0,1], principal point centered).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import imageio.v3 as iio

from .dataset import DatasetCfgCommon
from .dtypes import Stage
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim_to_views
from .view_sampler import ViewSampler


@dataclass
class DatasetSceneDecoupledCfg(DatasetCfgCommon):
    name: Literal["scene_decoupled"]
    root: Path | None
    # Which video category supplies the TARGET (video to generate).
    target_category: Literal["whuman", "wohuman"] = "whuman"
    # Decoupled mode: CONTEXT (scene tokens) from `wohuman` (background-only),
    # TARGET from `target_category`. When False, both come from `target_category`
    # (same-video recon-style, à la a single dynamicverse clip).
    decoupled_context: bool = True
    context_category: Literal["whuman", "wohuman"] = "wohuman"
    # Camera intrinsics from the 24 mm lens. Unreal CineCamera default filmback
    # is 36 mm wide; square pixels => fx_px = fy_px = focal/sensor_w * W. Stored
    # NORMALIZED ([0,1], principal point centered) like DL3DV/RE10K.
    focal_mm: float = 24.0
    sensor_width_mm: float = 36.0
    # Convert the parsed Unreal camera-local axes (x-fwd, y-right, z-up) to
    # OpenCV (x-right, y-down, z-fwd) so inv(K) ray directions are correct.
    convert_unreal_to_opencv: bool = True
    # Flip the (left-handed) Unreal world to right-handed (negate world Z).
    flip_world_handedness: bool = True
    # Per-scene scene-radius normalization (centroid-subtract, divide by max
    # camera distance) — homogenizes the cm-scale translations to a unit sphere.
    normalize_scene_scale: bool = True
    # Prompt: text/whuman/<scene_id>/<prompt_file>. `caption_action_only` keeps
    # the text to foreground motion (scene comes from tokens); `caption` is the
    # full scene+action description.
    prompt_file: str = "action.json"
    prompt_key: Literal["caption_action_only", "caption"] = "caption_action_only"
    require_prompt: bool = True
    force_empty_text: bool = False
    # Deterministic split: the last `num_val_scenes` scene_ids (sorted) are held
    # out for validation; train uses the rest. val_seen routes the held-out tail:
    #   val_seen=True  → first half of the tail   (val.standard)
    #   val_seen=False → second half of the tail  (val.unseen)
    num_val_scenes: int = 64
    val_seen: bool = True
    max_frames: int = 81
    baseline_epsilon: float = 1e-3
    make_baseline: bool = False
    # Per-view-type shape overrides (fall back to `shape` when None).
    context_shape: list[int] | None = None
    target_shape: list[int] | None = None


# OpenCV-cam basis vectors expressed in Unreal-cam coords (columns):
#   OpenCV +x (right)   = Unreal +Y
#   OpenCV +y (down)    = Unreal -Z
#   OpenCV +z (forward) = Unreal +X
_OPENCV_TO_UNREAL_CAM = torch.tensor(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=torch.float32,
)


class DatasetSceneDecoupled(Dataset):
    cfg: DatasetSceneDecoupledCfg
    stage: Stage
    view_sampler: ViewSampler

    def __init__(
        self,
        cfg: DatasetSceneDecoupledCfg,
        stage: Stage,
        view_sampler: ViewSampler,
        force_shuffle: bool = False,
    ) -> None:
        super().__init__()
        if cfg.root is None:
            raise ValueError(
                "Root of Scene-Decoupled dataset is not set. "
                "Set dataset.root=<path-to-Scene-Decoupled-Video-dataset>."
            )
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.force_shuffle = force_shuffle
        self.root = Path(cfg.root)
        self.prompt_cache: dict[str, str | None] = {}
        # (scene_id, trajectory) pairs — one clip each.
        self.samples = self._collect_samples()

    # ─── discovery ──────────────────────────────────────────────────────────

    def _video_path(self, category: str, scene_id: str, traj: str) -> Path:
        return self.root / "video" / category / scene_id / f"{scene_id}_{traj}_24mm.mp4"

    def _camera_path(self, scene_id: str) -> Path:
        # whuman / wohuman cam.json are byte-identical; use whuman (always present).
        return self.root / "camera" / "whuman" / scene_id / f"{scene_id}_cam.json"

    def _text_path(self, scene_id: str) -> Path:
        return self.root / "text" / "whuman" / scene_id / self.cfg.prompt_file

    def _collect_samples(self) -> list[tuple[str, str]]:
        cam_root = self.root / "camera" / "whuman"
        if not cam_root.is_dir():
            raise FileNotFoundError(f"Expected {cam_root} to exist.")
        scene_ids = sorted(d.name for d in cam_root.iterdir() if d.is_dir())

        # Deterministic train / val split on the sorted scene-id tail.
        n_val = min(self.cfg.num_val_scenes, max(0, len(scene_ids) - 1))
        val_ids = scene_ids[len(scene_ids) - n_val:]
        train_ids = scene_ids[: len(scene_ids) - n_val]
        if self.stage == "train":
            chosen = train_ids
        else:
            half = len(val_ids) // 2
            chosen = val_ids[:half] if self.cfg.val_seen else val_ids[half:]
            if not chosen:  # tiny dataset fallback
                chosen = val_ids or train_ids

        ctx_cat = self.cfg.context_category if self.cfg.decoupled_context else self.cfg.target_category
        tgt_cat = self.cfg.target_category

        samples: list[tuple[str, str]] = []
        dropped_no_prompt = 0
        for sid in chosen:
            cam_path = self._camera_path(sid)
            if not cam_path.exists():
                continue
            try:
                with cam_path.open() as f:
                    trajs = sorted(json.load(f)["0"].keys())  # ["01_24mm", ...]
            except (OSError, json.JSONDecodeError, KeyError):
                continue
            if not (self.cfg.force_empty_text) and self.cfg.require_prompt:
                if self._load_prompt(sid) is None:
                    dropped_no_prompt += 1
                    continue
            for tkey in trajs:
                traj = tkey.split("_")[0]  # "01_24mm" -> "01"
                if not self._video_path(ctx_cat, sid, traj).exists():
                    continue
                if not self._video_path(tgt_cat, sid, traj).exists():
                    continue
                samples.append((sid, traj))

        if dropped_no_prompt:
            print(f"[scene_decoupled] dropped {dropped_no_prompt} scenes lacking `{self.cfg.prompt_key}`")
        if not samples:
            raise ValueError(f"No valid Scene-Decoupled samples under {self.root} (stage={self.stage}).")
        print(f"[scene_decoupled] stage={self.stage} samples={len(samples)} (scenes={len(chosen)})")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    # ─── per-sample loading ───────────────────────────────────────────────────

    def __getitem__(self, idx: int):
        scene_id, traj = self.samples[idx]
        scene = f"{scene_id}_{traj}"

        extrinsics, intrinsics = self._load_cameras(self._camera_path(scene_id), traj)
        num_cameras = extrinsics.shape[0]

        ctx_cat = self.cfg.context_category if self.cfg.decoupled_context else self.cfg.target_category
        tgt_cat = self.cfg.target_category
        ctx_frames = self._load_video(self._video_path(ctx_cat, scene_id, traj), max_frames=num_cameras)
        if ctx_cat == tgt_cat:
            tgt_frames = ctx_frames
        else:
            tgt_frames = self._load_video(self._video_path(tgt_cat, scene_id, traj), max_frames=num_cameras)

        num_views = min(ctx_frames.shape[0], tgt_frames.shape[0], extrinsics.shape[0])
        if num_views < max(self.view_sampler.num_context_views, 2):
            raise ValueError(
                f"Scene {scene} has too few frames ({num_views}) for "
                f"num_context_views={self.view_sampler.num_context_views}."
            )
        ctx_frames = ctx_frames[:num_views]
        tgt_frames = tgt_frames[:num_views] if tgt_frames is not ctx_frames else ctx_frames
        extrinsics = extrinsics[:num_views]
        intrinsics = intrinsics[:num_views]

        ctx_shape = tuple(self.cfg.context_shape or self.cfg.shape)
        tgt_shape = tuple(self.cfg.target_shape or self.cfg.shape)
        min_shape = (max(ctx_shape[0], tgt_shape[0]), max(ctx_shape[1], tgt_shape[1]))
        ctx_frames, intrinsics = self._ensure_min_shape(ctx_frames, intrinsics, min_shape)
        if tgt_frames is not ctx_frames:
            tgt_frames, _ = self._ensure_min_shape(tgt_frames, intrinsics, min_shape)

        context_extrinsics = extrinsics.clone()
        target_extrinsics = extrinsics.clone()
        context_intrinsics = intrinsics.clone()
        target_intrinsics = intrinsics.clone()

        view_indices, _ = self.view_sampler.sample(
            num_views=num_views,
            num_latents=num_views,
            stage=self.stage,
            extrinsics=context_extrinsics,
            scene=scene,
        )

        if view_indices.context is not None and torch.any(view_indices.context >= num_views):
            raise ValueError(f"Scene {scene}: context index out of range.")
        if view_indices.target is not None and torch.any(view_indices.target >= num_views):
            raise ValueError(f"Scene {scene}: target index out of range.")

        sample = {"scene": scene}

        if self.cfg.make_baseline:
            ctxt = context_extrinsics[view_indices.context]
            scale = (ctxt[0, :3, 3] - ctxt[-1, :3, 3]).norm()
            if scale < self.cfg.baseline_epsilon:
                raise ValueError(f"Skipped {scene}: insufficient baseline {scale:.6f}")
            context_extrinsics[:, :3, 3] /= scale
            target_extrinsics[:, :3, 3] /= scale

        if view_indices.context is not None:
            sample["context"] = {
                "extrinsics": context_extrinsics[view_indices.context].float(),
                "intrinsics": context_intrinsics[view_indices.context].float(),
                "latent": ctx_frames[view_indices.context].float(),
                "index": view_indices.context,
            }
        if view_indices.target is not None:
            sample["target"] = {
                "extrinsics": target_extrinsics[view_indices.target].float(),
                "intrinsics": target_intrinsics[view_indices.target].float(),
                "latent": tgt_frames[view_indices.target].float(),
                "index": view_indices.target,
            }

        if self.stage == "train" and self.cfg.augment:
            sample = apply_augmentation_shim(sample)

        if "context" in sample:
            sample["context"] = apply_crop_shim_to_views(sample["context"], ctx_shape)
        if "target" in sample:
            sample["target"] = apply_crop_shim_to_views(sample["target"], tgt_shape)

        if self.cfg.force_empty_text:
            sample["text"] = ""
        else:
            prompt = self._load_prompt(scene_id)
            if prompt is not None:
                sample["text"] = prompt
        return sample

    # ─── helpers ──────────────────────────────────────────────────────────────

    def _ensure_min_shape(self, latents, intrinsics, shape):
        target_h, target_w = shape
        _, _, h, w = latents.shape
        if h >= target_h and w >= target_w:
            return latents, intrinsics
        scale = max(target_h / h, target_w / w)
        new_h = max(int(np.ceil(h * scale)), target_h)
        new_w = max(int(np.ceil(w * scale)), target_w)
        latents = F.interpolate(latents, size=(new_h, new_w), mode="bilinear", align_corners=False)
        # Normalized intrinsics are scale-invariant — no update needed.
        return latents, intrinsics

    def _load_video(self, path: Path, max_frames: int | None = None) -> torch.Tensor:
        for loader in (self._load_video_imageio, self._load_video_decord):
            try:
                frames = loader(path, max_frames=max_frames)
                if frames.numel() == 0:
                    continue
                return frames
            except ImportError:
                continue
        raise ImportError(
            "No mp4 decoder available. `pip install imageio imageio-ffmpeg` or `pip install decord`."
        )

    def _load_video_imageio(self, path: Path, max_frames: int | None = None) -> torch.Tensor:
        last_error = None
        for plugin in (None, "pyav", "FFMPEG", "ffmpeg"):
            try:
                frames = iio.imread(path) if plugin is None else iio.imread(path, plugin=plugin)
                if frames.ndim != 4:
                    raise ValueError(f"Unexpected video shape for {path}: {frames.shape}")
                if max_frames is not None:
                    frames = frames[:max_frames]
                frames = torch.from_numpy(np.asarray(frames)).float() / 255.0
                return frames.permute(0, 3, 1, 2).contiguous()
            except Exception as err:  # noqa: BLE001
                last_error = err
        raise ImportError(f"imageio could not decode {path}.") from last_error

    def _load_video_decord(self, path: Path, max_frames: int | None = None) -> torch.Tensor:
        from decord import VideoReader, cpu

        reader = VideoReader(str(path), ctx=cpu(0))
        num_frames = len(reader) if max_frames is None else min(len(reader), max_frames)
        frames = reader.get_batch(list(range(num_frames))).asnumpy()
        frames = torch.from_numpy(frames).float() / 255.0
        return frames.permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _parse_matrix(s: str) -> torch.Tensor:
        """Parse "[a b c 0] [d e f 0] [g h i 0] [tx ty tz 1] " → (4,4) float tensor."""
        nums = [float(x) for x in s.replace("[", " ").replace("]", " ").split()]
        return torch.tensor(nums, dtype=torch.float32).reshape(4, 4)

    def _load_cameras(self, path: Path, traj: str) -> tuple[torch.Tensor, torch.Tensor]:
        with path.open() as f:
            cam = json.load(f)
        view_key = f"{traj}_24mm"
        frame_ids = sorted(cam.keys(), key=lambda k: int(k))

        extrinsics = []
        for fid in frame_ids:
            entry = cam[fid].get(view_key)
            if entry is None:
                continue
            m = self._parse_matrix(entry)         # Unreal row-vector c2w (last row = position)
            c2w = m.t().contiguous()              # column-convention c2w (last col = position)
            if self.cfg.convert_unreal_to_opencv:
                # camera-local axes Unreal -> OpenCV: rot' = rot @ (OpenCV->Unreal cam)
                c2w[:3, :3] = c2w[:3, :3] @ _OPENCV_TO_UNREAL_CAM
            extrinsics.append(c2w)
        extrinsics = torch.stack(extrinsics)      # (N,4,4)

        if self.cfg.flip_world_handedness:
            # Left-handed Unreal world -> right-handed: negate world Z (rows of the
            # world-side: position z and the z-row of rotation).
            extrinsics = extrinsics.clone()
            extrinsics[:, 2, :] = -extrinsics[:, 2, :]

        if self.cfg.normalize_scene_scale:
            cam_origins = extrinsics[:, :3, 3]
            centroid = cam_origins.mean(dim=0)
            radius = (cam_origins - centroid).norm(dim=-1).max().clamp(min=1e-6)
            extrinsics = extrinsics.clone()
            extrinsics[:, :3, 3] = (cam_origins - centroid) / radius

        # Intrinsics from focal/sensor, NORMALIZED ([0,1], principal point centered).
        # Square pixels: fx_px = fy_px = focal/sensor_w * W  →  fx_norm = focal/sensor_w,
        # fy_norm = fx_px / H = (focal/sensor_w) * (W/H). Resolution-independent in
        # normalized form (W/H = video aspect = 672/384).
        n = extrinsics.shape[0]
        aspect = 672.0 / 384.0
        fx_norm = self.cfg.focal_mm / self.cfg.sensor_width_mm
        fy_norm = fx_norm * aspect
        K = torch.eye(3, dtype=torch.float32)
        K[0, 0] = fx_norm
        K[1, 1] = fy_norm
        K[0, 2] = 0.5
        K[1, 2] = 0.5
        intrinsics = K.unsqueeze(0).repeat(n, 1, 1)
        return extrinsics, intrinsics

    def _load_prompt(self, scene_id: str) -> str | None:
        if scene_id in self.prompt_cache:
            return self.prompt_cache[scene_id]
        path = self._text_path(scene_id)
        value = None
        if path.exists():
            try:
                with path.open() as f:
                    data = json.load(f)
                v = data.get(self.cfg.prompt_key) or data.get("caption")
                if isinstance(v, str) and v.strip():
                    value = v
            except (OSError, json.JSONDecodeError):
                value = None
        self.prompt_cache[scene_id] = value
        return value
