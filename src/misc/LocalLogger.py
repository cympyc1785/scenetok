import json
from pathlib import Path
import shutil
from typing import Any, Optional
import torch
import numpy as np
from PIL import Image
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only
# from torchvision.io import write_video
import imageio.v3 as iio

LOG_PATH = Path("outputs/local")


class LocalLogger(Logger):
    def __init__(self, log_dir: str | Path = LOG_PATH, clean: bool = True) -> None:
        super().__init__()
        self.experiment = None
        self._log_dir = Path(log_dir)
        if clean:
            shutil.rmtree(self._log_dir, ignore_errors=True)

    @property
    def name(self):
        return "LocalLogger"

    @property
    def version(self):
        return 0

    @property
    def log_dir(self):
        return self._log_dir

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        path = self._log_dir / "metrics.jsonl"
        path.parent.mkdir(exist_ok=True, parents=True)
        record = {"step": step}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().item() if v.numel() == 1 else v.detach().cpu().tolist()
            elif isinstance(v, np.ndarray):
                v = v.item() if v.size == 1 else v.tolist()
            record[k] = v
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")

    @rank_zero_only
    def log_image(
        self,
        key: str,
        images: list[Any],
        step: Optional[int] = None,
        **kwargs,
    ):
        # The function signature is the same as the wandb logger's, but the step is
        # actually required.
        assert step is not None
        for index, image in enumerate(images):
            path = self._log_dir / f"{key}/{index:0>2}_{step:0>6}.png"
            path.parent.mkdir(exist_ok=True, parents=True)
            Image.fromarray(image).save(path)

    
    @rank_zero_only
    def log_video(
        self,
        key: str,
        videos: list[Any],
        fps: list[int],
        caption: list[str],
        step: Optional[int],
        format: list[str],
        **kwargs,
    ):
        assert step is not None

        for index, video in enumerate(videos):
            fmat = format[index]
            cap = caption[index]
            path = self._log_dir / f"{key}/{cap}_{step:0>6}.{fmat}"
            path.parent.mkdir(exist_ok=True, parents=True)

            # (T, C, H, W) → (T, H, W, C)
            if isinstance(video, np.ndarray):
                video = torch.from_numpy(video)

            video = video.permute(0, 2, 3, 1).cpu().numpy()

            # dtype 처리 (imageio는 uint8 권장)
            if video.dtype != np.uint8:
                video = np.clip(video, 0, 255)
                video = video.astype(np.uint8)

            print(video.shape, video.dtype)
            print(path)
            print(cap)
            print(fmat)
            print(fps[index])

            iio.imwrite(
                path,
                video,
                fps=fps[index],
            )
            
