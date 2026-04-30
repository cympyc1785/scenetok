
import csv
import os
import torch
import numpy as np
import torchvision.transforms as tf
import time
import shutil

from PIL import Image
from pathlib import Path
from typing import Literal
from torch.utils.data import Dataset
from dataclasses import asdict, dataclass

from .dataset import DatasetCfgCommon
from .dtypes import Stage
from .view_sampler import ViewSampler
from src.misc.dl3dv_utils import load_metadata
from src.misc.camera_utils import rescale_and_crop, reflect_views, convert_poses

from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim


def get_dl3dv_stage_root(cfg: "DatasetDL3DVCfg", stage: Stage) -> Path:
    if cfg.smallset and stage in {"val", "test"}:
        return cfg.root / "train"
    # return cfg.root / ("test" if stage == "val" else stage)
    return cfg.root / "train"

def get_dl3dv_scene_dirs(cfg: "DatasetDL3DVCfg", root: Path, stage: Stage) -> list[Path]:
    if stage == "train" or str(root).endswith("train"):
        chunk_ids = [1] if cfg.smallset else range(1, 12)
        chunk_roots = [root / f"{i}K" for i in chunk_ids if (root / f"{i}K").exists()]
        return [
            scene_dir
            for chunk_root in chunk_roots
            for scene_dir in chunk_root.iterdir()
            if scene_dir.is_dir()
        ]

    if cfg.smallset and (root / "1K").exists(): # 11K
        return [
            scene_dir
            for scene_dir in (root / "1K").iterdir() # 11K
            if scene_dir.is_dir()
        ]

    return [
        scene_dir
        for scene_dir in root.iterdir()
        if scene_dir.is_dir()
    ]


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


def apply_dl3dv_smallset_filter(chunks: list[Path], stage: Stage) -> list[Path]:
    if not chunks:
        return chunks

    desired_prefix = "1K" if stage == "train" else "1K"  # 11K
    filtered = [chunk for chunk in chunks if chunk.parts and chunk.parts[0] == desired_prefix]
    if filtered:
        return filtered

    return chunks

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

def build_dl3dv_meta(cfg: "DatasetDL3DVCfg", stage: Stage, force: bool = False) -> Path:
    from tqdm import tqdm
    root = get_dl3dv_stage_root(cfg, stage)
    meta_path = root / "meta.csv"
    if meta_path.exists() and not force:
        return meta_path

    rows = []
    total_cnt = 0
    invalid_cnt = 0
    for scene_dir in tqdm(get_dl3dv_scene_dirs(cfg, root, stage)):
        total_cnt += 1
        data_dir = get_dl3dv_data_dir(scene_dir)
        if not (data_dir / "transforms.json").exists():
            invalid_cnt += 1
            continue

        example = load_metadata(data_dir / "transforms.json")
        extrinsics, intrinsics = convert_poses(example["cameras"])

        image_folder = get_dl3dv_image_folder(data_dir, cfg.folder_key)
        if not os.path.exists(image_folder):
            invalid_cnt += 1
            continue
        image_paths = sorted(image_folder.iterdir())
        num_images = len(image_paths)

        # Check valid length
        if num_images < 34:
            invalid_cnt += 1
            continue
        if num_images != len(extrinsics):
            invalid_cnt += 1
            continue

        # Check valid camera
        if check_teleport_camera(extrinsics):
            invalid_cnt += 1
            continue
        
        # Check valid images
        is_valid = True
        prev_num = 0
        for image_path in image_paths:
            try:
                with Image.open(image_path).convert("RGB") as image:
                    width, height = image.size
                if height < 256 or width < 256:
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
            invalid_cnt += 1
            continue

         # Erase not matching preprocessed data
        if os.path.exists(data_dir / "preprocessed" / "transforms.npz"):
            extrinsics = np.load(data_dir / "preprocessed" / "transforms.npz")["extrinsics"]
            preprocessed_images = np.load(data_dir / "preprocessed" / "images.npy")
            if num_images != len(preprocessed_images) or num_images != len(extrinsics):
                shutil.rmtree(data_dir / "preprocessed")

        rows.append(
            {
                "chunk": str(scene_dir.relative_to(root)),
                "split": stage,
                "height": height,
                "width": width,
                "num_images": num_images,
            }
        )

    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["chunk", "split", "height", "width", "num_images"])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Invalid Count: {invalid_cnt} / {total_cnt}")

    return meta_path

@dataclass
class DatasetDL3DVCfg(DatasetCfgCommon):
    name: Literal["dl3dv"]
    root: Path
    flip: bool=False
    scale_focal_by_256: bool=False # Allow compatibility with va-videodc, refer to docs/KNOWN_BUG.md
    scale_context_focal_by_256: bool=False # Allow compatibility with va-wan, refer to docs/KNOWN_BUG.md
    folder_key: str="images_4"
    smallset: bool = True
    target_latent_type: str | None = None
    stage_override: Literal["train", "val", "test"] | None = None
    scene_id: str | None = None

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
        self.root = get_dl3dv_stage_root(cfg, self.data_stage_override)
        # Collect chunks.
        self.chunks = []

        meta_path = self.root / "meta.csv"
        if meta_path.exists():
            with open(meta_path, "r", newline="") as f:
                self.chunks = [Path(row["chunk"]) for row in csv.DictReader(f) if row.get("chunk")]
        else:
            self.chunks = sorted(
                scene_dir.relative_to(self.root)
                for scene_dir in get_dl3dv_scene_dirs(self.cfg, self.root, self.data_stage_override)
                if (get_dl3dv_data_dir(scene_dir) / "transforms.json").exists()
                and get_dl3dv_image_folder(get_dl3dv_data_dir(scene_dir), self.cfg.folder_key) is not None
            )
        if cfg.smallset:
            print("\ndl3dv small set!\n")
            self.chunks = apply_dl3dv_smallset_filter(self.chunks, self.data_stage_override)
        if self.cfg.scene_id is not None:
            scene_id = Path(self.cfg.scene_id)
            if scene_id not in self.chunks:
                raise ValueError(
                    f"Requested DL3DV scene_id={scene_id} was not found under {self.root}. "
                    f"Available examples: {self.chunks[:5]}"
                )
            print(f"\n\nUsing single DL3DV scene: {scene_id}\n\n")
            self.chunks = [scene_id]

    def __getitem__(self, idx):
        chunk_name = self.chunks[idx]
        chunk_path = self.root / chunk_name
        scene = chunk_path.name
        if os.path.exists(chunk_path / "nerfstudio"):
            chunk_path = chunk_path / "nerfstudio"
        images = []
        if os.path.exists(chunk_path / "preprocessed" / "images.npy") and os.path.exists(chunk_path / "preprocessed" / "transforms.npz"):
            # Load preprocessed data
            chunk_path = chunk_path / "preprocessed"
            images = torch.from_numpy(np.load(chunk_path / "images.npy"))
            if images.dtype == torch.uint8:
                images = images.float() / 255.0
            else:
                images = images.float()
            cameras = np.load(chunk_path / "transforms.npz")
            extrinsics = torch.from_numpy(cameras["extrinsics"]).float()
            intrinsics = torch.from_numpy(cameras["intrinsics"]).float()
        else:
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
            if len(image_paths) == 0:
                raise ValueError(f"Empty image folder: {image_folder}")

            for img_path in image_paths:
                try:
                    im = Image.open(img_path).convert("RGB")
                except OSError as exc:
                    raise ValueError(f"Unreadable image {img_path}: {exc}")
                im = np.asarray(im) / 255
                im = torch.from_numpy(im)
                im = im.permute(2, 0, 1)
                images.append(im)
            images = torch.stack(images)
        
            example = load_metadata(chunk_path / "transforms.json", scale_focal_by_256=self.cfg.scale_focal_by_256)
            extrinsics, intrinsics = convert_poses(example["cameras"])
            
            if images.shape[2] < 256 or images.shape[3] < 256:
                raise ValueError(f"Bad image shape: {images.shape}")

            images, intrinsics = rescale_and_crop(images, intrinsics, tuple(self.cfg.shape))

            preprocessed_path = chunk_path / "preprocessed"
            preprocessed_path.mkdir(exist_ok=True)
            images_uint8 = (images.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
            np.save(preprocessed_path / "images.npy", images_uint8)
            np.savez(preprocessed_path / "transforms.npz", extrinsics=extrinsics, intrinsics=intrinsics)
        
        flip = self.cfg.flip
        if self.cfg.augment and self.stage == "train":
            flip = np.random.choice([False, True]).astype(int)
        
        if flip:
            images, extrinsics = reflect_views(images, extrinsics)
        
        num_views = extrinsics.shape[0]

        if num_views < 34:
            raise ValueError("not enough views", chunk_path)

        view_indices, upsampled_indices = self.view_sampler.sample(
            num_views=num_views, 
            num_latents=num_views, 
            stage=self.stage, 
            extrinsics=extrinsics, 
            scene=scene
        )
        sample = {"scene": scene}
        for view_type, indices in asdict(view_indices).items():
            if indices is None:
                continue
            
                
            sample[view_type] = {
                "extrinsics": extrinsics[indices].clone().contiguous(),
                "intrinsics": intrinsics[indices].clone().contiguous(),
                "latent": images[indices].clone().contiguous(),
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
   
