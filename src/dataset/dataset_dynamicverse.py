"""Dataset for WorldTraj/dynamicverse.

Each scene lives under `<root>/<subdataset>/<scene_name>/` and is expected to
contain:

  * ``inpaint_result.mp4`` — RGB video; same file is used for BOTH context and
    target views (no separate source/target distinction).
  * ``cameras.json`` — list of per-frame camera entries
    ``{idx, rotation (3×3), position (3), fx, fy, cx, cy}``; the rotation +
    position together encode a camera-to-world transform (same convention as
    ``dataset_davis.py``).
  * ``prompts.json`` — per-scene language description. We pull the
    ``prompt_scene`` string from the first window entry (key ``"0"`` by
    default — there's typically only one).

The ``dynpose-100k`` subfolder is excluded by default (caller can override
``excluded_subdatasets``). No chunking / split is applied — all scenes go into
a single pool regardless of stage. Built by mirroring ``dataset_davis.py``,
trimmed of split / meta-csv / overfit logic since this dataset is meant to be
trained against as a single bag of scenes.
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
from .shims.crop_shim import apply_crop_shim, apply_crop_shim_to_views
from .view_sampler import ViewSampler


DEFAULT_EXCLUDED_SUBDATASETS = ["dynpose-100k", "logs"]


@dataclass
class DatasetDynamicverseCfg(DatasetCfgCommon):
    name: Literal["dynamicverse"]
    root: Path | None
    excluded_subdatasets: list[str] = field(
        default_factory=lambda: list(DEFAULT_EXCLUDED_SUBDATASETS)
    )
    camera_file: str = "cameras.json"
    video_name: str = "inpaint_result.mp4"
    # When set, target views are read from a DIFFERENT video than context.
    # Use case: context = `inpaint_result.mp4` (background only), target =
    # `video_input.mp4` (original with dynamic objects); train scene-token=
    # background, text=foreground.
    target_video_name: str | None = None
    prompt_file: str = "prompts.json"
    prompt_key: str = "prompt_scene"
    # Prompt loading mode.
    #   "window_dict" (default): existing `prompts.json` schema —
    #     `{"<window_id>": {"prompt_scene": "...", ...}, ...}` — iterate and
    #     pick the first non-empty `prompt_key` string.
    #   "category_first": load `category/category.json` (path = `category_file`)
    #     which has `{"dynamic": ["<dyn1>", ...], "reasoning": {<dyn1>: "<description>", ...}}`,
    #     and use `reasoning[dynamic[0]]` as the prompt.
    prompt_style: Literal["window_dict", "category_first"] = "window_dict"
    category_file: str = "category/category.json"
    # When true, scenes whose `prompts.json` lacks a usable `prompt_key` string
    # are dropped at collection time. Required for text-conditioned training:
    # `default_collate` raises if some samples carry `sample["text"]` and others
    # don't, so the text condition is silently dropped from the whole batch.
    # Set to False for view-only training where text is optional.
    require_prompt: bool = True
    # Sub-datasets reserved for the **unseen** validation pool — never used
    # for training. `val_seen` (set per loader by `data_module.py`) routes:
    #   train + val.standard  → all subdatasets EXCEPT these
    #   val.unseen            → ONLY these subdatasets
    unseen_subdatasets: list[str] = field(default_factory=lambda: ["DAVIS"])
    val_seen: bool = True
    # Optional eval index JSON (`{scene_name: {context: [...], target: [...]}}`).
    # When set, `__getitem__` uses the index's fixed context/target indices for
    # stage in {"val", "test"} (mirrors DL3DV's pattern), and `_collect_scenes`
    # also restricts to the scenes listed in the JSON. Built by
    # `scripts/build_dynamicverse_eval_index.py`.
    evaluation_index_path: Path | None = None
    # Some pipelines store rotation in world-to-camera form. DAVIS / dynamicverse
    # use camera-to-world (rotation + position = c2w), matching `dataset_davis`.
    camera_rotation_is_world_to_camera: bool = False
    baseline_epsilon: float = 1e-3
    make_baseline: bool = False
    # Per-view-type shape overrides. If None, falls back to `shape`. Use these
    # when context (for the scene compressor) and target (for the denoiser) need
    # different spatial layouts — e.g. 256×448 compressor pretrained on a wide
    # context but a 480×832 target latent grid for the Wan TI2V denoiser.
    context_shape: list[int] | None = None
    target_shape: list[int] | None = None


class DatasetDynamicverse(Dataset):
    cfg: DatasetDynamicverseCfg
    stage: Stage
    view_sampler: ViewSampler
    scenes: list[Path]

    def __init__(
        self,
        cfg: DatasetDynamicverseCfg,
        stage: Stage,
        view_sampler: ViewSampler,
        force_shuffle: bool = False,
    ) -> None:
        super().__init__()
        if cfg.root is None:
            raise ValueError(
                "Root directory of dynamicverse dataset is not defined. "
                "Set dataset.root=<path-to-WorldTraj/dynamicverse>."
            )

        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.force_shuffle = force_shuffle
        self.root = cfg.root
        # Worker-local prompt cache keyed by scene_dir absolute path string.
        # Must be created BEFORE `_collect_scenes` since the collection step
        # calls `_load_prompt` to enforce `require_prompt`.
        self.prompt_cache: dict[str, str] = {}
        self.evaluation_index = self._load_evaluation_index()
        self.scenes = self._collect_scenes()

    # ─── scene discovery ────────────────────────────────────────────────────

    def _load_evaluation_index(self) -> dict[str, dict] | None:
        path = self.cfg.evaluation_index_path
        if path is None:
            return None
        with Path(path).open("r") as f:
            return json.load(f)

    def _collect_scenes(self) -> list[Path]:
        excluded = set(self.cfg.excluded_subdatasets)
        unseen = set(self.cfg.unseen_subdatasets)
        # train + val.standard share the "seen" pool (everything except
        # `unseen_subdatasets`); val.unseen routes to ONLY those subdatasets.
        if self.cfg.val_seen:
            allowed = lambda sub_name: sub_name not in excluded and sub_name not in unseen
        else:
            allowed = lambda sub_name: sub_name not in excluded and sub_name in unseen
        scenes: list[Path] = []
        dropped_no_prompt = 0
        for sub_dir in sorted(self.root.iterdir()):
            if not sub_dir.is_dir() or not allowed(sub_dir.name):
                continue
            for scene_dir in sorted(sub_dir.iterdir()):
                if not scene_dir.is_dir():
                    continue
                if not (scene_dir / self.cfg.video_name).exists():
                    continue
                if self.cfg.target_video_name is not None and self.cfg.target_video_name != self.cfg.video_name:
                    if not (scene_dir / self.cfg.target_video_name).exists():
                        continue
                if not (scene_dir / self.cfg.camera_file).exists():
                    continue
                prompt_path = (
                    scene_dir / self.cfg.category_file
                    if self.cfg.prompt_style == "category_first"
                    else scene_dir / self.cfg.prompt_file
                )
                if not prompt_path.exists():
                    continue
                # `require_prompt`: drop scenes whose prompts.json lacks a usable
                # `prompt_key` string. `default_collate` raises (or silently
                # drops the key) when some samples carry `sample["text"]` and
                # others don't → text condition is silently lost for the whole
                # batch. Keep this on for text-conditioned training.
                if self.cfg.require_prompt and self._load_prompt(scene_dir) is None:
                    dropped_no_prompt += 1
                    continue
                scenes.append(scene_dir)
        if self.cfg.require_prompt and dropped_no_prompt > 0:
            print(
                f"[dynamicverse] dropped {dropped_no_prompt} scenes lacking "
                f"`{self.cfg.prompt_key}` in {self.cfg.prompt_file}"
            )
        # If an eval index is loaded, restrict to its scene-name keys.
        if self.evaluation_index is not None:
            allowed_names = set(self.evaluation_index.keys())
            before = len(scenes)
            scenes = [s for s in scenes if s.name in allowed_names]
            print(
                f"[dynamicverse] evaluation_index filter ({self.cfg.evaluation_index_path}): "
                f"{before} → {len(scenes)} scenes"
            )

        # Optional restriction by eval-style view samplers (e.g. evaluation_video)
        # — only keep scenes that the sampler has an entry for.
        if hasattr(self.view_sampler, "index"):
            allowed = set(getattr(self.view_sampler, "index").keys())
            scenes = [
                s for s in scenes
                if s.name in allowed and self._scene_has_valid_sampler_indices(s)
            ]

        if not scenes:
            raise ValueError(
                f"No valid dynamicverse scenes found under {self.root} "
                f"(excluded={sorted(excluded)})."
            )
        return scenes

    def _scene_has_valid_sampler_indices(self, scene_dir: Path) -> bool:
        if not hasattr(self.view_sampler, "index"):
            return True
        entries = getattr(self.view_sampler, "index").get(scene_dir.name)
        if not entries:
            return False
        with open(scene_dir / self.cfg.camera_file, "r") as f:
            num_cameras = len(json.load(f))
        entry = entries[0]
        max_index = -1
        if getattr(entry, "context", None):
            max_index = max(max_index, max(entry.context))
        if getattr(entry, "target", None):
            target = list(entry.target)
            if self.view_sampler.cfg.name == "evaluation_video_wan" and len(target) % 4 == 0:
                target.extend(target[-1] + i for i in range(1, 5))
            max_index = max(max_index, max(target))
        return max_index < num_cameras

    def __len__(self) -> int:
        return len(self.scenes)

    # ─── per-sample loading ────────────────────────────────────────────────

    def __getitem__(self, idx: int):
        scene_dir = self.scenes[idx]
        scene = scene_dir.name

        extrinsics, intrinsics = self._load_cameras(scene_dir / self.cfg.camera_file)
        num_cameras = extrinsics.shape[0]

        # Context frames; target reuses these unless `target_video_name` is set.
        ctx_frames = self._load_video(
            scene_dir / self.cfg.video_name,
            max_frames=num_cameras,
        )
        if self.cfg.target_video_name is not None and self.cfg.target_video_name != self.cfg.video_name:
            tgt_frames = self._load_video(
                scene_dir / self.cfg.target_video_name,
                max_frames=num_cameras,
            )
        else:
            tgt_frames = ctx_frames

        num_views = min(
            ctx_frames.shape[0],
            tgt_frames.shape[0],
            extrinsics.shape[0],
            intrinsics.shape[0],
        )
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
        # `apply_crop_shim_to_views` assumes input ≥ output; rescale frames up
        # to whichever of (context, target) shape is larger so both crops fit.
        min_shape = (max(ctx_shape[0], tgt_shape[0]), max(ctx_shape[1], tgt_shape[1]))
        ctx_frames, intrinsics = self._ensure_min_shape(ctx_frames, intrinsics, min_shape)
        if tgt_frames is not ctx_frames:
            tgt_frames, _ = self._ensure_min_shape(tgt_frames, intrinsics, min_shape)

        # Clone per-view tensors so context/target can be modulated
        # independently downstream (e.g. baseline rescaling).
        context_extrinsics = extrinsics.clone()
        target_extrinsics = extrinsics.clone()
        context_intrinsics = intrinsics.clone()
        target_intrinsics = intrinsics.clone()

        # For stage in {val, test} with an eval index loaded, use its fixed
        # context/target indices (DL3DV pattern). Otherwise call the view
        # sampler normally for train-style stochastic sampling.
        if self.stage in {"val", "test"} and self.evaluation_index is not None:
            entry = self.evaluation_index.get(scene)
            if entry is None:
                raise ValueError(
                    f"scene {scene!r} not found in evaluation_index "
                    f"(path={self.cfg.evaluation_index_path})"
                )
            from .view_sampler.view_sampler import ViewIndex
            view_indices = ViewIndex(
                context=torch.tensor(entry["context"], dtype=torch.long),
                target=torch.tensor(entry["target"], dtype=torch.long),
            )
            latent_indices = None
        else:
            view_indices, latent_indices = self.view_sampler.sample(
                num_views=num_views,
                num_latents=num_views,
                stage=self.stage,
                extrinsics=context_extrinsics,
                scene=scene,
                scene_dir=scene_dir,  # required by caption_window; bounded/unbounded swallow via **kwargs
            )

        if view_indices.context is not None and torch.any(view_indices.context >= num_views):
            raise ValueError(
                f"Scene {scene} has context indices outside available camera range: "
                f"max={view_indices.context.max().item()}, num_views={num_views}"
            )
        if view_indices.target is not None and torch.any(view_indices.target >= num_views):
            raise ValueError(
                f"Scene {scene} has target indices outside available camera range: "
                f"max={view_indices.target.max().item()}, num_views={num_views}"
            )

        sample = {"scene": scene}

        if self.cfg.make_baseline:
            ctxt_extrinsics = context_extrinsics[view_indices.context]
            a = ctxt_extrinsics[0, :3, 3]
            b = ctxt_extrinsics[-1, :3, 3]
            scale = (a - b).norm()
            if scale < self.cfg.baseline_epsilon:
                raise ValueError(
                    f"Skipped {scene} because of insufficient baseline {scale:.6f}"
                )
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

        # Per-view crop: context for the scene compressor, target for the
        # denoiser. When `context_shape` / `target_shape` aren't set they both
        # collapse to `cfg.shape`, matching the simple single-shape path.
        if "context" in sample:
            sample["context"] = apply_crop_shim_to_views(sample["context"], ctx_shape)
        if "target" in sample:
            sample["target"] = apply_crop_shim_to_views(sample["target"], tgt_shape)

        prompt = self._load_prompt(scene_dir)
        if prompt is not None:
            sample["text"] = prompt
        return sample

    # ─── helpers ───────────────────────────────────────────────────────────

    def _ensure_min_shape(
        self,
        latents: torch.Tensor,
        intrinsics: torch.Tensor,
        shape: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_h, target_w = shape
        _, _, h, w = latents.shape
        if h >= target_h and w >= target_w:
            return latents, intrinsics
        scale = max(target_h / h, target_w / w)
        new_h = max(int(np.ceil(h * scale)), target_h)
        new_w = max(int(np.ceil(w * scale)), target_w)
        latents = F.interpolate(
            latents,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )
        intrinsics = intrinsics.clone()
        intrinsics[:, 0, 0] *= new_w / w
        intrinsics[:, 1, 1] *= new_h / h
        intrinsics[:, 0, 2] *= new_w / w
        intrinsics[:, 1, 2] *= new_h / h
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
            "No Python mp4 decoder is available. "
            "Install one of: `pip install imageio imageio-ffmpeg` or `pip install decord`."
        )

    def _load_video_imageio(
        self,
        path: Path,
        max_frames: int | None = None,
    ) -> torch.Tensor:
        last_error = None
        for plugin in (None, "pyav", "FFMPEG", "ffmpeg"):
            try:
                if plugin is None:
                    frames = iio.imread(path)
                else:
                    frames = iio.imread(path, plugin=plugin)
                if frames.ndim != 4:
                    raise ValueError(f"Unexpected video shape for {path}: {frames.shape}")
                if max_frames is not None:
                    frames = frames[:max_frames]
                frames = torch.from_numpy(np.asarray(frames)).float() / 255.0
                return frames.permute(0, 3, 1, 2).contiguous()
            except Exception as err:
                last_error = err
        raise ImportError(
            f"imageio could not decode {path}. "
            "Try `pip install imageio[ffmpeg]` or `pip install av`."
        ) from last_error

    def _load_video_decord(
        self,
        path: Path,
        max_frames: int | None = None,
    ) -> torch.Tensor:
        from decord import VideoReader, cpu

        reader = VideoReader(str(path), ctx=cpu(0))
        num_frames = len(reader) if max_frames is None else min(len(reader), max_frames)
        frames = reader.get_batch(list(range(num_frames))).asnumpy()
        frames = torch.from_numpy(frames).float() / 255.0
        return frames.permute(0, 3, 1, 2).contiguous()

    def _load_cameras(self, path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        with open(path, "r") as f:
            cameras = json.load(f)

        extrinsics = []
        intrinsics = []
        for camera in cameras:
            rotation = torch.tensor(camera["rotation"], dtype=torch.float32)
            position = torch.tensor(camera["position"], dtype=torch.float32)

            if self.cfg.camera_rotation_is_world_to_camera:
                w2c = torch.eye(4, dtype=torch.float32)
                w2c[:3, :3] = rotation
                w2c[:3, 3] = position
                c2w = torch.linalg.inv(w2c)
            else:
                c2w = torch.eye(4, dtype=torch.float32)
                c2w[:3, :3] = rotation
                c2w[:3, 3] = position
            extrinsics.append(c2w)

            intrinsic = torch.eye(3, dtype=torch.float32)
            intrinsic[0, 0] = float(camera["fx"])
            intrinsic[1, 1] = float(camera["fy"])
            intrinsic[0, 2] = float(camera["cx"])
            intrinsic[1, 2] = float(camera["cy"])
            intrinsics.append(intrinsic)

        return torch.stack(extrinsics), torch.stack(intrinsics)

    def _load_prompt(self, scene_dir: Path) -> str | None:
        cache_key = str(scene_dir)
        if cache_key in self.prompt_cache:
            return self.prompt_cache[cache_key]

        if self.cfg.prompt_style == "category_first":
            return self._load_prompt_category_first(scene_dir, cache_key)

        prompt_path = scene_dir / self.cfg.prompt_file
        if not prompt_path.exists():
            return None
        try:
            with open(prompt_path, "r") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

        # `prompts.json` is typically `{"0": {... "prompt_scene": "...", ...}}`.
        # Iterate windows in stored order and pick the first one that carries
        # a non-empty string under `cfg.prompt_key`.
        if isinstance(data, dict):
            iterable = data.values()
        elif isinstance(data, list):
            iterable = data
        else:
            return None

        for entry in iterable:
            if not isinstance(entry, dict):
                continue
            value = entry.get(self.cfg.prompt_key)
            if isinstance(value, str) and value.strip():
                self.prompt_cache[cache_key] = value
                return value
            # Some prompt keys store {"detail", "concise"} dicts (e.g. the
            # camera prompts). Fall back to "concise" then "detail" if so.
            if isinstance(value, dict):
                for sub_key in ("concise", "detail"):
                    sub = value.get(sub_key)
                    if isinstance(sub, str) and sub.strip():
                        self.prompt_cache[cache_key] = sub
                        return sub
        return None

    def _load_prompt_category_first(self, scene_dir: Path, cache_key: str) -> str | None:
        """`category/category.json` lookup:
            {"dynamic": [<name>, ...], "reasoning": {<name>: "<description>", ...}}
        Returns `reasoning[dynamic[0]]` (description for the first dynamic
        object). None if file missing or malformed.
        """
        cat_path = scene_dir / self.cfg.category_file
        if not cat_path.exists():
            return None
        try:
            with cat_path.open() as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        dynamic = data.get("dynamic")
        reasoning = data.get("reasoning")
        if not isinstance(dynamic, list) or not dynamic:
            return None
        if not isinstance(reasoning, dict):
            return None
        first_key = dynamic[0]
        if not isinstance(first_key, str):
            return None
        value = reasoning.get(first_key)
        if isinstance(value, str) and value.strip():
            self.prompt_cache[cache_key] = value
            return value
        return None
