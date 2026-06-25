
import csv
import json
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
from .view_sampler.view_sampler import ViewIndex
from src.misc.dl3dv_utils import load_metadata
from src.misc.camera_utils import rescale_and_crop, reflect_views, convert_poses, rescale_and_pad

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
    # `extrinsics` is (N, 4, 4). Previous slicing `c2w[:3, 3]` accidentally
    # picked the first 3 frames' homogeneous row [0,0,0,1] (shape (3, 4))
    # instead of the per-frame translation, so the magnitude checks always
    # passed for any data. Fix: take col-3 of the first three rows for every
    # frame to recover the translation vector of shape (N, 3).
    c2w = extrinsics                          # (N, 4, 4)
    w2c = c2w.inverse()

    t_c2w = c2w[:, :3, 3]                     # (N, 3)
    t_w2c = w2c[:, :3, 3]                     # (N, 3)

    diff = t_w2c[1:] - t_w2c[:-1]             # (N-1, 3)
    mag = torch.linalg.norm(diff, dim=1)      # (N-1,)
    if (torch.abs(diff) > 10).any(dim=1).any() or (mag > 15).any():
        return True
    if (torch.abs(t_c2w) > 50).any(dim=1).any():
        return True
    if not torch.isfinite(t_c2w).all() or not torch.isfinite(t_w2c).all():
        return True
    return False

BLACKLIST_FILENAME = "blacklist.csv"
_BLACKLIST_FIELDNAMES = ["scene", "reason", "step", "loss", "detail"]
_PROMPTS_NOT_LOADED = object()  # sentinel for `_prompts_cache.get(..., _PROMPTS_NOT_LOADED)`


def _resolve_blacklist_path(cfg: "DatasetDL3DVCfg", root: Path) -> Path:
    """`{cfg.blacklist_path}` if set, else `{root}/blacklist.csv` (sibling of meta.csv)."""
    path = getattr(cfg, "blacklist_path", None)
    return Path(path) if path else Path(root) / BLACKLIST_FILENAME


def load_blacklist(path: Path | str | None) -> set[str]:
    """Return the set of blacklisted scene basenames; tolerate missing file."""
    if path is None:
        return set()
    path = Path(path)
    if not path.exists():
        return set()
    with open(path, "r", newline="") as f:
        return {row["scene"] for row in csv.DictReader(f) if row.get("scene")}


def append_blacklist(path: Path | str, entries: list[dict]) -> int:
    """Append entries to the blacklist CSV (dedup against existing). Returns # new rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_blacklist(path)
    new = []
    seen_local: set[str] = set()
    for e in entries:
        s = e.get("scene")
        if not s or s in existing or s in seen_local:
            continue
        seen_local.add(s)
        new.append({k: e.get(k, "") for k in _BLACKLIST_FIELDNAMES})
    if not new:
        return 0
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_BLACKLIST_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(new)
    return len(new)


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

    blacklist = load_blacklist(_resolve_blacklist_path(cfg, root))
    if blacklist:
        before = len(scene_dirs)
        scene_dirs = [d for d in scene_dirs if d.name not in blacklist]
        print(f"Blacklist filter: dropped {before - len(scene_dirs)} / {before} scenes")

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
    do_scale_and_pad: bool = False
    evaluation_index_path: Path | None = None
    blacklist_path: Path | None = None
    # If true, additionally load DA3 depth + cameras from
    # `<scene>/da3/exports/mini_npz/results.npz` for every sampled view.
    # Adds `context["depth"], context["da3_w2c"], context["da3_intrinsics"]`
    # and `target["da3_w2c_first"], target["da3_intrinsics_first"]` to the
    # sample. Used by `condition_latents_input_type=first_frame_depth`.
    load_da3_depth: bool = False
    # If true, read `<scene>/prompts_37.json` (or `prompts_filename`) at
    # __getitem__ time and inject the entry matching the target frame window
    # as `sample["text"]` (the `prompt_scene_simple.concise` field).
    load_prompts: bool = False
    prompts_filename: str = "prompts_37.json"

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
        self.root = cfg.root
        # self.root = cfg.root / "train"
        # Collect chunks.
        self.chunks = []
        self.preprocess = rescale_and_pad if cfg.do_scale_and_pad else rescale_and_crop
        self.evaluation_index = self.load_evaluation_index()
        # Per-scene cache for prompts_*.json (only used when cfg.load_prompts).
        # Worker-local; safe under torch DataLoader's per-worker dataset copy.
        self._prompts_cache: dict[str, dict | None] = {}

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

        # Filter blacklisted scenes (also handles entries that slipped into a
        # pre-existing meta.csv from before the blacklist was populated).
        blacklist = load_blacklist(_resolve_blacklist_path(cfg, self.root))
        if blacklist:
            before = len(self.chunks)
            self.chunks = [c for c in self.chunks if c.name not in blacklist]
            print(f"[DL3DV] blacklist drop: {before - len(self.chunks)} / {before}")

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
        if self.stage in {"val", "test"} and self.evaluation_index is not None:
            self.chunks = [chunk for chunk in self.chunks if chunk.name in self.evaluation_index]

    def load_evaluation_index(self) -> dict[str, dict[str, list[int]]] | None:
        path = self.cfg.evaluation_index_path
        if path is None:
            chunk_targets = getattr(self.view_sampler.cfg, "chunk_targets", None)
            if chunk_targets is None:
                return None
            size = "34" if chunk_targets else "37"
            split = "standard" if self.cfg.val_seen else "unseen"
            # caption_window는 caption-정렬 전용 index를 쓴다
            # (`scripts/build_caption_window_eval_index.py`로 생성).
            variant = "_caption" if getattr(self.view_sampler.cfg, "name", None) == "caption_window" else ""
            path = Path(f"assets/evaluation_index/dl3dv_c16_{size}{variant}_{split}.json")
        with Path(path).open("r") as f:
            return json.load(f)

    def sample_evaluation_index_views(self, scene: str) -> ViewIndex:
        if self.evaluation_index is None:
            raise ValueError("evaluation_index is not loaded")
        entry = self.evaluation_index.get(scene)
        if entry is None:
            raise ValueError(f"No evaluation indices available for scene {scene}.")
        return ViewIndex(
            torch.tensor(entry["context"], dtype=torch.long),
            torch.tensor(entry["target"], dtype=torch.long),
        )

    def _lookup_prompt_for_target(self, scene_root: Path, target_indices: torch.Tensor) -> str:
        """Look up the `prompts_<N>.json` entry whose `frame_idx` window covers
        the target frame range, and return `prompt_scene_simple.concise`.
        Returns empty string if the file is missing or no entry matches.

        Matching rule: `frame_idx[0] <= min(target) <= max(target) < frame_idx[1]`.
        If multiple windows contain the range (shouldn't happen for typical
        non-overlapping windows), the narrowest one wins.
        """
        scene_key = str(scene_root)
        cached = self._prompts_cache.get(scene_key, _PROMPTS_NOT_LOADED)
        if cached is _PROMPTS_NOT_LOADED:
            prompts_path = scene_root / self.cfg.prompts_filename
            if not prompts_path.exists():
                self._prompts_cache[scene_key] = None
                return ""
            try:
                with prompts_path.open() as f:
                    cached = json.load(f)
            except (OSError, json.JSONDecodeError):
                cached = None
            self._prompts_cache[scene_key] = cached
        if not cached:
            return ""

        if target_indices.numel() == 0:
            return ""
        t_min = int(target_indices.min().item())
        t_max = int(target_indices.max().item())

        best = None
        best_span = None
        for entry in cached.values():
            fi = entry.get("frame_idx") if isinstance(entry, dict) else None
            if not (isinstance(fi, (list, tuple)) and len(fi) == 2):
                continue
            lo, hi = int(fi[0]), int(fi[1])  # half-open [lo, hi)
            if lo <= t_min and t_max < hi:
                span = hi - lo
                if best_span is None or span < best_span:
                    best, best_span = entry, span

        if best is None:
            return ""
        scene_simple = best.get("prompt_scene_simple")
        if not isinstance(scene_simple, dict):
            return ""
        return scene_simple.get("concise", "") or ""

    def validate_view_shape(
        self,
        images: torch.Tensor,
        expected_shape: tuple[int, int],
        view_type: str,
        chunk_path: Path,
    ) -> None:
        actual_shape = tuple(images.shape[-2:])
        if actual_shape != expected_shape:
            raise ValueError(
                f"{view_type} images have shape {actual_shape}, expected "
                f"{expected_shape}: {chunk_path}"
            )

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

        if self.stage in {"val", "test"} and self.evaluation_index is not None:
            view_indices = self.sample_evaluation_index_views(scene)
        else:
            view_indices, upsampled_indices = self.view_sampler.sample(
                num_views=num_views,
                num_latents=num_views,
                stage=self.stage,
                extrinsics=extrinsics,
                scene=scene,
                scene_dir=chunk_path,
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
                context_shape = tuple(self.cfg.context_shape)
                view_images, view_intrinsics = self.preprocess(
                    view_images,
                    intrinsics[indices],
                    context_shape,
                )
                self.validate_view_shape(view_images, context_shape, view_type, chunk_path)
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
                target_shape = tuple(self.cfg.target_shape)
                view_images, view_intrinsics = self.preprocess(
                    view_images,
                    intrinsics[indices],
                    target_shape,
                )
                self.validate_view_shape(view_images, target_shape, view_type, chunk_path)
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

        if self.cfg.load_da3_depth:
            # DA3 depth + npz cameras, lazily loaded per scene. Stored on the
            # dataset instance keyed by scene basename so subsequent samples
            # from the same chunk don't reload. mini_npz keeps frame order in
            # the same indexing as transforms.json (sorted by file_path).
            # `chunk_path` may have been re-rooted into `<scene>/nerfstudio`
            # above; DA3 outputs live at scene root regardless, so re-derive.
            depth_root = (self.root / chunk_name) / "da3" / "exports" / "mini_npz" / "results.npz"
            if not depth_root.exists():
                # Drop scenes without DA3 outputs — `first_frame_depth`
                # needs them. `safe_collate` filters None out of the batch.
                return None
            else:
                cache = getattr(self, "_da3_cache", None)
                if cache is None or cache.get("__scene__") != scene:
                    npz = np.load(depth_root)
                    cache = {
                        "__scene__": scene,
                        "depth": torch.from_numpy(npz["depth"]).float(),         # (N, Hd, Wd)
                        "extr34": torch.from_numpy(npz["extrinsics"]).float(),    # (N, 3, 4) w2c
                        "intr": torch.from_numpy(npz["intrinsics"]).float(),      # (N, 3, 3) at (Hd, Wd)
                    }
                    self._da3_cache = cache
                depths = cache["depth"]
                extr34 = cache["extr34"]
                intr_npz = cache["intr"]
                _Hd, _Wd = depths.shape[-2], depths.shape[-1]

                def _select_npz(idx_long, target_hw):
                    """Resize depth + scale intrinsics to target_hw, build (V, 4, 4) w2c."""
                    th, tw = target_hw
                    d = depths[idx_long].unsqueeze(1)                              # (V, 1, Hd, Wd)
                    d = torch.nn.functional.interpolate(d, size=(th, tw), mode="bilinear", align_corners=False)
                    K = intr_npz[idx_long].clone()                                  # (V, 3, 3)
                    sx, sy = tw / _Wd, th / _Hd
                    K[..., 0, 0] *= sx
                    K[..., 1, 1] *= sy
                    K[..., 0, 2] *= sx
                    K[..., 1, 2] *= sy
                    E = torch.eye(4).unsqueeze(0).repeat(idx_long.shape[0], 1, 1)   # (V, 4, 4)
                    E[:, :3] = extr34[idx_long]
                    return d, K, E

                # Context: at context_shape (matches sample["context"]["latent"])
                ctx_indices_long = sample["context"]["index"].long()
                d_ctx, K_ctx, W_ctx = _select_npz(ctx_indices_long, tuple(self.cfg.context_shape))
                sample["context"]["depth"] = d_ctx.contiguous()
                sample["context"]["da3_intrinsics"] = K_ctx.contiguous()
                sample["context"]["da3_w2c"] = W_ctx.contiguous()

                # Target first frame: at target_shape (matches sample["target"]["latent"])
                tgt_indices_long = sample["target"]["index"].long()
                tgt_first_idx = tgt_indices_long[:1]
                d_tgt, K_tgt, W_tgt = _select_npz(tgt_first_idx, tuple(self.cfg.target_shape))
                # And context at target shape (for use by the warp module at target res)
                d_ctx_t, K_ctx_t, W_ctx_t = _select_npz(ctx_indices_long, tuple(self.cfg.target_shape))
                sample["context"]["depth_at_target_shape"] = d_ctx_t.contiguous()
                sample["context"]["da3_intrinsics_at_target_shape"] = K_ctx_t.contiguous()
                # Same extrinsics, but include for symmetry
                sample["context"]["da3_w2c_at_target_shape"] = W_ctx_t.contiguous()
                sample["target"]["da3_w2c_first"] = W_tgt[0].contiguous()           # (4, 4)
                sample["target"]["da3_intrinsics_first"] = K_tgt[0].contiguous()    # (3, 3)

        if self.cfg.load_prompts:
            sample["text"] = self._lookup_prompt_for_target(
                self.root / chunk_name,
                sample["target"]["index"],
            )

        return sample
    
        
    @property
    def data_stage(self) -> Stage:

        if self.data_stage_override == "val":
            return "test"
        return self.data_stage_override

    def __len__(self) -> int:
        return len(self.chunks)
   
