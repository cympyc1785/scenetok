import csv
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal

import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset
from tqdm import tqdm
from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.random_transform_shim import apply_random_transform_shim
from .shims.crop_shim import apply_crop_shim
from .dtypes import Stage
from .view_sampler import ViewSampler, ViewSamplerEvaluation
from torch.utils.data import Dataset


def build_re10k_meta(root: Path, min_frames: int = 37, num_workers: int = 8) -> Path:
    """Scan all .torch chunks under `root` and write meta.csv listing scenes where
    len(images) == len(cameras) AND >= min_frames. CSV columns: chunk, key, num_images.
    """
    meta_path = root / "meta.csv"
    if meta_path.exists():
        return meta_path
    chunks = sorted(p for p in root.iterdir() if p.suffix == ".torch")
    print(f"\nBuilding re10k meta for {root} ({len(chunks)} chunks, min_frames={min_frames})...")

    def scan(chunk_path: Path):
        try:
            data = torch.load(chunk_path, weights_only=False)
        except Exception as e:
            print(f"  failed to load {chunk_path}: {e}")
            return []
        out = []
        for x in data:
            n_img = len(x["images"])
            n_cam = len(x["cameras"])
            if n_img != n_cam or n_img < min_frames:
                continue
            out.append({"chunk": chunk_path.name, "key": x["key"], "num_images": n_img})
        return out

    rows = []
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        for chunk_rows in tqdm(ex.map(scan, chunks), total=len(chunks), desc="scanning chunks"):
            rows.extend(chunk_rows)

    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["chunk", "key", "num_images"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {meta_path}: {len(rows)} valid scenes\n")
    return meta_path

@dataclass
class DatasetRE10kCfg(DatasetCfgCommon):
    name: Literal["re10k"]
    root: Path | None
    baseline_epsilon: float
    max_fov: float
    make_baseline: bool
    random_transform_extrinsics = False


class DatasetRE10k(Dataset):
    cfg: DatasetRE10kCfg
    stage: Stage
    view_sampler: ViewSampler
    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 1000.0

    def __init__(
        self,
        cfg: DatasetRE10kCfg,
        stage: Stage,
        view_sampler: ViewSampler,
        force_shuffle: bool = False
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        self.force_shuffle = force_shuffle

        if cfg.root is None:
            raise Exception("Root directory of dataset is not defined. Please specify in your argument as dataset.root=<path-to-root-directory>")

        # train stage reads root/train, val/test read root/test.
        stage_subdir = "train" if stage == "train" else "test"
        root = cfg.root / stage_subdir
        with open(root / "index.json") as f:
            self.map_dict = json.load(f)

        # Build (or read) meta.csv and keep only scenes that pass the filter.
        meta_path = build_re10k_meta(root)
        with open(meta_path, newline="") as f:
            valid_keys = {row["key"] for row in csv.DictReader(f)}

        # Eval samplers (evaluation_video / evaluation_video_wan) expose `.index` to pin
        # which scenes to load; training samplers (unbounded, bounded) don't.
        if hasattr(view_sampler, "index"):
            self.scenes = [s for s in view_sampler.index.keys() if s in self.map_dict and s in valid_keys]
        else:
            self.scenes = [s for s in self.map_dict.keys() if s in valid_keys]
        self.chunks = [root / self.map_dict[s] for s in self.scenes]
        self.root = root

    def __getitem__(self, idx):
        scene = self.scenes[idx]
        chunk = torch.load(self.chunks[idx], weights_only=True)
        example = next(x for x in chunk if x["key"] == scene)
        extrinsics, intrinsics = self.convert_poses(example["cameras"])
        num_views = extrinsics.shape[0]

        try:
            sample_result = self.view_sampler.sample(
                num_views=num_views,
                num_latents=num_views,
                stage=self.stage,
                extrinsics=extrinsics,
                scene=scene,
            )
        except ValueError as err:
            # view_sampler raises ValueError when the scene doesn't have enough frames
            # for the requested context/target window or when produced indices would go OOB.
            # safe_collate will filter the None and the trainer's step methods will skip
            # an all-None batch via their `if batch is None: return None` guard.
            print(f"Skipped {scene}: view_sampler.sample failed ({err}).")
            return None
        # Training samplers (unbounded/bounded) return (ViewIndex, Tensor);
        # eval samplers may return list[ViewIndex]. Normalize to a single ViewIndex.
        if isinstance(sample_result, tuple):
            view_index = sample_result[0]
        elif isinstance(sample_result, list):
            view_index = sample_result[0]
        else:
            view_index = sample_result

        sample = {"scene": scene}

        # Resize the world to make the baseline 1.
        context_extrinsics = extrinsics[view_index.context]
        if context_extrinsics.shape[0] == 2 and self.cfg.make_baseline:
            a, b = context_extrinsics[:, :3, 3]
            scale = (a - b).norm()
            if scale < self.cfg.baseline_epsilon:
                print(
                    f"Skipped {scene} because of insufficient baseline "
                    f"{scale:.6f}"
                )
                return None
            extrinsics[:, :3, 3] /= scale

        for view_type, indices in asdict(view_index).items():
            if indices is None:
                continue
            # Load the images.
            images = [
                example["images"][index.item()] for index in indices
            ]
            images = self.convert_images(images)

            # Skip the example if the images don't have the right shape.
            if images.shape[1:] != (3, 360, 640):
                print(
                    f"Skipped bad example {scene}. "
                    f"{view_type.capitalize()} shape was {images.shape}."
                )
                return None

            sample[view_type] = {
                "extrinsics": extrinsics[indices],
                "intrinsics": intrinsics[indices],
                "latent": images,
                "index": indices,
            }

        if self.stage == "train" and self.cfg.augment:
            sample = apply_augmentation_shim(sample)
        if self.stage in ["train", "val"] and self.cfg.random_transform_extrinsics:
            sample = apply_random_transform_shim(sample)
        return apply_crop_shim(sample, tuple(self.cfg.shape))

    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c.inverse(), intrinsics

    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)



    def __len__(self) -> int:
        return len(self.scenes)
