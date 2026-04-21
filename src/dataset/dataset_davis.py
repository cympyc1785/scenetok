import csv
import json
from dataclasses import dataclass
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
from .shims.crop_shim import apply_crop_shim
from .view_sampler import ViewSampler


@dataclass
class DatasetDAVISCfg(DatasetCfgCommon):
    name: Literal["davis"]
    root: Path | None
    meta_file: str | None = None
    meta_scene_column: str = "scene"
    meta_split_column: str | None = "split"
    use_meta_split: bool = True
    overfit_train_all_meta: bool = False
    overfit_val_from_meta_last: bool = False
    baseline_epsilon: float = 1e-3
    make_baseline: bool = False
    camera_file: str = "cameras.json"
    context_video_name: str = "inpaint_result.mp4"
    target_video_name: str = "video_input_resized.mp4"
    train_split_ratio: float = 0.9
    camera_rotation_is_world_to_camera: bool = False
    prompt_file: str | None = "dynamic_prompts.json"
    prompt_source: str | None = "prompt_dynamic_object"


class DatasetDAVIS(Dataset):
    cfg: DatasetDAVISCfg
    stage: Stage
    view_sampler: ViewSampler
    scenes: list[Path]

    def __init__(
        self,
        cfg: DatasetDAVISCfg,
        stage: Stage,
        view_sampler: ViewSampler,
        force_shuffle: bool = False,
    ) -> None:
        super().__init__()
        if cfg.root is None:
            raise ValueError(
                "Root directory of DAVIS dataset is not defined. "
                "Please set dataset.root=<path-to-davis-root>."
            )

        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.force_shuffle = force_shuffle
        self.root = cfg.root
        self.scenes = self._collect_scenes()
        self.prompt_cache: dict[str, str] = {}

    def _collect_scenes(self) -> list[Path]:
        scenes = []
        for scene_dir in sorted(self.root.iterdir()):
            if not scene_dir.is_dir():
                continue
            if not (scene_dir / self.cfg.context_video_name).exists():
                continue
            if not (scene_dir / self.cfg.target_video_name).exists():
                continue
            if not (scene_dir / self.cfg.camera_file).exists():
                continue
            scenes.append(scene_dir)

        scene_metadata = self._load_scene_metadata()
        if scene_metadata is not None:
            valid_scene_names = set(scene_metadata.keys())
            scenes = [
                scene_dir for scene_dir in scenes if scene_dir.name in valid_scene_names
            ]

            if self.stage == "val" and self.cfg.overfit_val_from_meta_last:
                last_scene_name = next(reversed(scene_metadata))
                scenes = [
                    scene_dir for scene_dir in scenes if scene_dir.name == last_scene_name
                ]
            elif (
                not (self.stage == "train" and self.cfg.overfit_train_all_meta)
                and self.cfg.use_meta_split
                and self.cfg.meta_split_column is not None
            ):
                allowed_splits = self._allowed_meta_splits_for_stage()
                scenes = [
                    scene_dir
                    for scene_dir in scenes
                    if scene_metadata[scene_dir.name] in allowed_splits
                ]

        if self.stage in {"val", "test"} and hasattr(self.view_sampler, "index"):
            valid_scene_names = set(getattr(self.view_sampler, "index").keys())
            scenes = [
                scene_dir
                for scene_dir in scenes
                if scene_dir.name in valid_scene_names
                and self._scene_has_valid_sampler_indices(scene_dir)
            ]

        if not scenes:
            raise ValueError(f"No valid DAVIS scenes found under {self.root}.")

        if scene_metadata is not None and (
            self.cfg.use_meta_split
            or (self.stage == "train" and self.cfg.overfit_train_all_meta)
            or (self.stage == "val" and self.cfg.overfit_val_from_meta_last)
        ):
            return scenes

        total = len(scenes)
        split_idx = int(round(total * self.cfg.train_split_ratio))
        split_idx = min(max(split_idx, 1), max(total - 1, 1))

        if self.stage == "train" and total > 1:
            return scenes[:split_idx]
        if self.stage in {"val", "test"} and total > 1:
            return scenes[split_idx:]
        return scenes

    def _load_scene_metadata(self) -> dict[str, str | None] | None:
        if self.cfg.meta_file is None:
            return None

        meta_path = Path(self.cfg.meta_file)
        # if not meta_path.is_absolute():
        #     meta_path = self.root.parent / meta_path

        if not meta_path.exists():
            raise FileNotFoundError(f"DAVIS meta file does not exist: {meta_path}")

        with open(meta_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"DAVIS meta file has no header: {meta_path}")
            if self.cfg.meta_scene_column not in reader.fieldnames:
                raise ValueError(
                    f"Column '{self.cfg.meta_scene_column}' not found in DAVIS meta file: "
                    f"{meta_path}"
                )
            if (
                self.cfg.use_meta_split
                and self.cfg.meta_split_column is not None
                and self.cfg.meta_split_column not in reader.fieldnames
            ):
                raise ValueError(
                    f"Column '{self.cfg.meta_split_column}' not found in DAVIS meta file: "
                    f"{meta_path}"
                )

            metadata = {}
            for row in reader:
                scene = (row.get(self.cfg.meta_scene_column) or "").strip()
                if not scene:
                    continue

                split = None
                if self.cfg.meta_split_column is not None:
                    split_value = row.get(self.cfg.meta_split_column)
                    split = split_value.strip().lower() if split_value else None
                metadata[scene] = split

        if not metadata:
            raise ValueError(f"No scene rows found in DAVIS meta file: {meta_path}")
        return metadata

    def _allowed_meta_splits_for_stage(self) -> set[str]:
        if self.stage == "train":
            return {"train"}
        if self.stage == "val":
            return {"val", "validation", "eval"}
        if self.stage == "test":
            return {"test", "val", "validation", "eval"}
        return set()

    def __len__(self) -> int:
        return len(self.scenes)

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

    def __getitem__(self, idx: int):
        scene_dir = self.scenes[idx]
        scene = scene_dir.name

        extrinsics, intrinsics = self._load_cameras(scene_dir / self.cfg.camera_file)
        num_cameras = extrinsics.shape[0]
        context_latents = self._load_video(
            scene_dir / self.cfg.context_video_name,
            max_frames=num_cameras,
        )
        target_latents = self._load_video(
            scene_dir / self.cfg.target_video_name,
            max_frames=num_cameras,
        )

        num_views = min(
            context_latents.shape[0],
            target_latents.shape[0],
            extrinsics.shape[0],
            intrinsics.shape[0],
        )
        if num_views < max(self.view_sampler.num_context_views, 2):
            raise ValueError(f"Scene {scene} has too few frames: {num_views}")

        context_latents = context_latents[:num_views]
        target_latents = target_latents[:num_views]
        context_extrinsics = extrinsics[:num_views].clone()
        target_extrinsics = extrinsics[:num_views].clone()
        context_intrinsics = intrinsics[:num_views].clone()
        target_intrinsics = intrinsics[:num_views].clone()
        context_latents, context_intrinsics = self._ensure_min_shape(
            context_latents,
            context_intrinsics,
            tuple(self.cfg.shape),
        )
        target_latents, target_intrinsics = self._ensure_min_shape(
            target_latents,
            target_intrinsics,
            tuple(self.cfg.shape),
        )

        view_indices, latent_indices = self.view_sampler.sample(
            num_views=num_views,
            num_latents=num_views,
            stage=self.stage,
            extrinsics=context_extrinsics,
            scene=scene,
        )

        if view_indices.context is not None and torch.any(view_indices.context >= num_views):
            raise ValueError(
                f"Scene {scene} has context indices outside available camera range: "
                f"max={view_indices.context.max().item()}, num_views={num_views}"
            )
        if view_indices.target is not None and torch.any(view_indices.target >= num_views):
            raise ValueError(
                f"Scene {scene} has target indices outside available camera range: "
                f"{view_indices}\n"
                f"max={view_indices.target.max().item()}, num_views={num_views}"
            )

        # if (
        #     view_indices.target is not None
        #     and latent_indices is not None
        #     and getattr(self.view_sampler.cfg, "temporal_downsample", 1) > 1
        #     and hasattr(self.view_sampler, "latent_to_original_index")
        #     and len(view_indices.target) % 17 != 0
        # ):
        #     # 17의 배수만 받음..
        #     start = self.view_sampler.latent_to_original_index(latent_indices[0])
        #     end = self.view_sampler.latent_to_original_index(latent_indices[-1] + 1)
        #     view_indices.target = torch.arange(start, end, dtype=torch.int64)

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
                "latent": context_latents[view_indices.context].float(),
                "index": view_indices.context,
            }

        if view_indices.target is not None:
            target_frame_indices = view_indices.target
            if target_latents.shape[0] != num_views and latent_indices is not None:
                target_frame_indices = latent_indices

            sample["target"] = {
                "extrinsics": target_extrinsics[view_indices.target].float(),
                "intrinsics": target_intrinsics[view_indices.target].float(),
                "latent": target_latents[target_frame_indices].float(),
                "index": view_indices.target,
            }

        if self.stage == "train" and self.cfg.augment:
            sample = apply_augmentation_shim(sample)

        sample = apply_crop_shim(sample, tuple(self.cfg.shape))
        prompt = self._load_prompt(scene_dir)
        if prompt is not None:
            sample["text"] = prompt
        return sample

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

    def _load_prompt(self, scene_dir: Path) -> str | None:
        if self.cfg.prompt_file is None or self.cfg.prompt_source is None:
            return None
        if scene_dir.name in self.prompt_cache:
            return self.prompt_cache[scene_dir.name]

        prompt_path = scene_dir / self.cfg.prompt_file
        if not prompt_path.exists():
            return None

        with open(prompt_path, "r") as f:
            data = json.load(f)

        prompt = self._extract_prompt(data)
        if prompt is not None:
            self.prompt_cache[scene_dir.name] = prompt
        return prompt

    def _extract_prompt(self, data) -> str | None:
        current = data
        if isinstance(current, dict) and "0" in current:
            current = current["0"]

        for key in self.cfg.prompt_source.split("."):
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]

        if isinstance(current, str):
            return current
        return None

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

            c2w = torch.eye(4, dtype=torch.float32)
            if self.cfg.camera_rotation_is_world_to_camera:
                w2c = torch.eye(4, dtype=torch.float32)
                w2c[:3, :3] = rotation
                w2c[:3, 3] = position
                c2w = torch.linalg.inv(w2c)
            else:
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
