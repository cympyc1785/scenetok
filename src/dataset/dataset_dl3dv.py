
import csv
import os
import torch
import numpy as np
import torchvision.transforms as tf
import time
import shutil

from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from pathlib import Path
from typing import Literal
from torch.utils.data import Dataset
from dataclasses import asdict, dataclass
from tqdm import tqdm

from .dataset import DatasetCfgCommon
from .dtypes import Stage
from .view_sampler import ViewSampler
from src.misc.dl3dv_utils import load_metadata
from src.misc.camera_utils import rescale_and_crop, reflect_views, convert_poses

from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim


def get_dl3dv_data_dir(scene_dir: Path) -> Path:
    nerfstudio_dir = scene_dir / "nerfstudio"
    return nerfstudio_dir if nerfstudio_dir.exists() else scene_dir

def get_dl3dv_image_folder(data_dir: Path, folder_key: str) -> Path | None:
    return next(
        (
            data_dir / folder
            for folder in [folder_key, "images_8", "images_4", "images"]
            if (data_dir / folder).exists()
        ),
        None,
    )

def check_teleport_camera(extrinsics):
    c2w = extrinsics
    w2c = c2w.inverse()

    t_c2w = c2w[:3, 3]
    t_w2c = w2c[:3, 3]

    diff = t_w2c[1:] - t_w2c[:-1]
    mag = torch.linalg.norm(diff, dim=1)
    invalid_indices_w2c = torch.where(
        (torch.abs(diff) > 10).any(dim=1) | (mag > 15)
    )[0]
    if len(invalid_indices_w2c) > 0:
        return True
    
    invalid_indices_c2w = torch.where(
        (torch.abs(t_c2w) > 50).any(dim=1)
    )[0]
    if len(invalid_indices_c2w) > 0:
        return True
    return False

def build_dl3dv_meta_row(
    cfg: "DatasetDL3DVCfg",
    root: Path,
    scene_dir: Path,
) -> dict[str, str | int] | None:
    data_dir = get_dl3dv_data_dir(scene_dir)
    if not (data_dir / "transforms.json").exists():
        return None

    example = load_metadata(data_dir / "transforms.json")
    extrinsics, intrinsics = convert_poses(example["cameras"])

    image_folder = get_dl3dv_image_folder(data_dir, cfg.folder_key)
    if image_folder is None or not image_folder.exists():
        return None
    image_paths = sorted(image_folder.iterdir())
    num_images = len(image_paths)

    # Check valid length
    if num_images < 34:
        return None
    if num_images != len(extrinsics):
        return None

    # Check valid camera
    if check_teleport_camera(extrinsics):
        return None
    
    # Check valid images
    is_valid = True
    prev_num = 0
    for image_path in image_paths:
        try:
            with Image.open(image_path).convert("RGB") as image:
                width, height = image.size
            if height < 480 or width < 832:
                is_valid = False
                break
            frame_num = int(str(image_path).split(".")[-2].split("frame_")[-1])
            if frame_num - prev_num > 1:
                is_valid = False
                break
            prev_num = frame_num
        except Exception as e:
            print(e)
            is_valid = False
            break
    if not is_valid:
        return None

    # Erase not matching preprocessed data
    if os.path.exists(data_dir / "preprocessed" / "transforms.npz"):
        extrinsics = np.load(data_dir / "preprocessed" / "transforms.npz")["extrinsics"]
        preprocessed_images = np.load(data_dir / "preprocessed" / "images.npy")
        if num_images != len(preprocessed_images) or num_images != len(extrinsics):
            shutil.rmtree(data_dir / "preprocessed")

    return {
        "chunk": str(scene_dir.relative_to(root)),
        "height": height,
        "width": width,
        "num_images": num_images,
    }

def build_dl3dv_meta(cfg: "DatasetDL3DVCfg", root, force: bool = False) -> Path:
    print(f"\nBuilding dataset meta file...\n")
    meta_path = root / f"meta.csv"
    if meta_path.exists() and not force:
        return meta_path

    chunk_roots = [root / f"{i}K" for i in range(1, 12) if (root / f"{i}K").exists()]
    scene_dirs = [
        scene_dir
        for chunk_root in chunk_roots
        for scene_dir in chunk_root.iterdir()
        if scene_dir.is_dir()
    ]
    
    rows = []
    total_cnt = len(scene_dirs)
    num_workers = 8
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(
            lambda scene_dir: build_dl3dv_meta_row(cfg, root, scene_dir),
            scene_dirs,
        )
        for row in tqdm(results, total=total_cnt):
            if row is not None:
                rows.append(row)
    invalid_cnt = total_cnt - len(rows)

    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["chunk", "height", "width", "num_images"])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Invalid Count: {invalid_cnt} / {total_cnt}")

    return meta_path

@dataclass
class DatasetDL3DVCfg(DatasetCfgCommon):
    name: Literal["dl3dv"]
    root: Path
    context_shape: list[int]
    target_shape: list[int]
    flip: bool=False
    scale_focal_by_256: bool=False # Allow compatibility with va-videodc, refer to docs/KNOWN_BUG.md
    scale_context_focal_by_256: bool=False # Allow compatibility with va-wan, refer to docs/KNOWN_BUG.md
    folder_key: str="images_4"
    smallset: bool = False
    target_latent_type: str | None = None
    stage_override: Literal["train", "val", "test"] | None = None
    scene_id: str | None = None
    val_seen: bool = False

class DatasetDL3DV(Dataset):
    cfg: DatasetDL3DVCfg
    stage: Stage
    view_sampler: ViewSampler
    to_tensor: tf.ToTensor
    chunks: list[Path]

    def __init__(
        self,
        cfg: DatasetDL3DVCfg,
        stage: Stage,
        view_sampler: ViewSampler,
        force_shuffle: bool = False
    ) -> None:
        super().__init__()
        
        self.cfg = cfg
        self.stage = stage
        self.data_stage_override = cfg.stage_override or stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        self.root = cfg.root / "train"
        # Collect chunks.
        self.chunks = []

        if self.cfg.scene_id is not None:
            scene_id = Path(self.cfg.scene_id)
            print(f"\n\nUsing single DL3DV scene: {scene_id}\n\n")
            self.chunks = [scene_id]
            return

        meta_path = self.root / f"meta.csv"
        if not meta_path.exists():
            build_dl3dv_meta(cfg, self.root)
        
        with open(meta_path, "r", newline="") as f:
            self.chunks = [Path(row["chunk"]) for row in csv.DictReader(f) if row.get("chunk")]

        # Filter dir
        if self.data_stage_override == "train":
            if cfg.smallset:
                desired_prefix = ["1K"]
            else:
                desired_prefix = [f"{i}K" for i in range(1, 11)]
        else:
            if cfg.val_seen:
                desired_prefix = ["1K"]
            else:
                desired_prefix = ["11K"]

        self.chunks = [chunk for chunk in self.chunks if chunk.parts and chunk.parts[0] in desired_prefix]

    def __getitem__(self, idx):
        chunk_name = self.chunks[idx]
        chunk_path = self.root / chunk_name
        scene = chunk_path.name
        if os.path.exists(chunk_path / "nerfstudio"):
            chunk_path = chunk_path / "nerfstudio"
        # Load raw data
        image_folder = next(
            (
                chunk_path / folder
                for folder in [self.cfg.folder_key, "images_8", "images_4", "images"]
                if (chunk_path / folder).exists()
            ),
            None,
        )
        if image_folder is None:
            raise FileNotFoundError(f"No image folder found under {chunk_path}")
        
        image_paths = sorted(path for path in image_folder.iterdir() if path.is_file())
        num_images = len(image_paths)
        if num_images == 0:
            raise ValueError(f"Empty image folder: {image_folder}")
        
        example = load_metadata(chunk_path / "transforms.json", scale_focal_by_256=self.cfg.scale_focal_by_256)
        extrinsics, intrinsics = convert_poses(example["cameras"])

        num_views = extrinsics.shape[0]

        if num_images != num_views:
            raise ValueError(f"Frame mismatch images={num_images} != cameras={num_views}")

        if num_views < 34:
            raise ValueError("not enough views", chunk_path)

        view_indices, upsampled_indices = self.view_sampler.sample(
            num_views=num_views, 
            num_latents=num_views, 
            stage=self.stage, 
            extrinsics=extrinsics, 
            scene=scene
        )

        view_index_items = []
        for view_type, indices in asdict(view_indices).items():
            if indices is None:
                continue
            if view_type not in {"context", "target"}:
                raise ValueError("Invalid view type", view_type)
            view_index_items.append((view_type, indices))

        if not view_index_items:
            raise ValueError("No views sampled", chunk_path)

        sampled_indices = torch.unique(
            torch.cat([indices.to(dtype=torch.long) for _, indices in view_index_items])
        ).sort().values
        images = []

        for image_idx in sampled_indices.tolist():
            img_path = image_paths[image_idx]
            try:
                im = Image.open(img_path).convert("RGB")
            except OSError as exc:
                raise ValueError(f"Unreadable image {img_path}: {exc}")
            im = np.asarray(im, dtype=np.float32) / 255.0
            im = torch.from_numpy(im)
            im = im.permute(2, 0, 1)
            images.append(im)
        images = torch.stack(images)
        
        if images.shape[2] < 256 or images.shape[3] < 256:
            raise ValueError(f"Bad image shape: {images.shape}")

        flip = self.cfg.flip
        if self.cfg.augment and self.stage == "train":
            flip = np.random.choice([False, True]).astype(int)

        sampled_position_by_index = {
            image_idx: position for position, image_idx in enumerate(sampled_indices.tolist())
        }
        sample = {"scene": scene}
        for view_type, indices in view_index_items:
            image_positions = torch.tensor(
                [sampled_position_by_index[index] for index in indices.tolist()],
                dtype=torch.long,
            )
            view_images = images[image_positions]

            if view_type == "context":
                view_images, view_intrinsics = rescale_and_crop(
                    view_images,
                    intrinsics[indices],
                    tuple(self.cfg.context_shape),
                )
                view_extrinsics = extrinsics[indices]
                if flip:
                    view_images, view_extrinsics = reflect_views(view_images, view_extrinsics)
                sample[view_type] = {
                    "extrinsics": view_extrinsics.clone().contiguous(),
                    "intrinsics": view_intrinsics.clone().contiguous(),
                    "latent": view_images.clone().contiguous(),
                    "index": indices.clone().contiguous()
                }
            elif view_type == "target":
                view_images, view_intrinsics = rescale_and_crop(
                    view_images,
                    intrinsics[indices],
                    tuple(self.cfg.target_shape),
                )
                view_extrinsics = extrinsics[indices]
                if flip:
                    view_images, view_extrinsics = reflect_views(view_images, view_extrinsics)
                sample[view_type] = {
                    "extrinsics": view_extrinsics.clone().contiguous(),
                    "intrinsics": view_intrinsics.clone().contiguous(),
                    "latent": view_images.clone().contiguous(),
                    "index": indices.clone().contiguous()
                }

            if view_type == "context" and self.cfg.scale_context_focal_by_256:
                sample[view_type]["intrinsics"][..., 0, 0] *= 3840/256
                sample[view_type]["intrinsics"][..., 1, 1] *= 2160/256

        return sample
    
        
    @property
    def data_stage(self) -> Stage:

        if self.data_stage_override == "val":
            return "test"
        return self.data_stage_override

    def __len__(self) -> int:
        return len(self.chunks)
   
