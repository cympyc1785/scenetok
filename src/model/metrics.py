import gc

import torch
from lpips import LPIPS

from torchmetrics.functional.image import structural_similarity_index_measure
from torch import nn
from skimage.metrics import structural_similarity
from cleanfid import fid

import numpy as np
from scipy import linalg
from tqdm import tqdm
from einops import rearrange, reduce
import torch
from torch.nn.functional import interpolate
from torchmetrics.image.fid import FrechetInceptionDistance
from ..misc.torch_utils import convert_to_buffer
from submodules.fvd.frechet_video_distance import frechet_video_distance

def freeze(m: torch.nn.Module) -> None:
    for param in m.parameters():
        param.requires_grad = False
    m.eval()


class Metric(nn.Module):
    def __init__(self):
        """
        Args:
            config (MultiValConfig): Hydra config with validation parameters.
            dataloader_map (dict): Mapping from name to an actual DataLoader.
        """
        super().__init__()
        self.setup()

    def setup(self):

        self.lpips = LPIPS(net="vgg").eval().to(torch.float16)
        # self.fid = FrechetInceptionDistance(normalize=True, feature_extractor_weights_path="checkpoints/pt_inception-2015-12-05-6726825d.pth").eval()
        self.fid = FrechetInceptionDistance(
            feature=2048,
            sync_on_compute=False,
            dist_sync_on_step=False,
            compute_on_cpu=True).eval()
        self.pred_list = []
        self.gt_list = []
        # freeze(self.lpips)
        # convert_to_buffer(self.lpips, persistent=False)

        # freeze(self.fid)
        # convert_to_buffer(self.fid, persistent=False)
        # self.fid = FrechetInceptionDistance(normalize=True).eval().to(torch.bfloat16)


    def compute_mse(self, pred, gt, num_views: int=16):
        return ((pred - gt)**2).mean()
    
    def compute_psnr(self, pred, gt, num_views: int=16):

        gt = gt.clip(min=0, max=1)
        pred = pred.clip(min=0, max=1)
        mse = reduce((gt - pred) ** 2, "b c h w -> b", "mean")
        return (-10 * mse.log10()).mean()

    @torch.no_grad()
    def compute_lpips(self, pred, gt, num_views: int=16):
        if pred.ndim == 3:
            pred = pred[None]
        if gt.ndim == 3:
            gt = gt[None]
        
        l = 0
        for p, g in zip(pred, gt):
            l += self.lpips(p[None].to(torch.float16), g[None].to(torch.float16), normalize=True).to(pred.dtype)
        return (l / pred.shape[0]).mean()
    
    def compute_ssim(self, pred, gt, num_views: int=16, chunk: int=8):
        if pred.ndim == 3:
            pred = pred[None]
        if gt.ndim == 3:
            gt = gt[None]
        # Chunk along N to bound peak memory. torchmetrics SSIM creates several
        # tensors the same size as input internally (mu, sigma², sigma_xy, ...).
        # On val end the model is still resident on GPU so a single full-batch
        # call OOMs (e.g. 320 frames × 5–10× ≈ 10–15 GB intermediate).
        N = pred.shape[0]
        if N == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        total = 0.0
        count = 0
        for i in range(0, N, chunk):
            p = pred[i : i + chunk].float()
            g = gt[i : i + chunk].float()
            s = structural_similarity_index_measure(
                p,
                g,
                gaussian_kernel=True,
                kernel_size=11,
                sigma=1.5,
                data_range=1.0,
            )
            total = total + s.item() * p.shape[0]
            count += p.shape[0]
        return torch.tensor(total / count, device=pred.device, dtype=pred.dtype)
    
    def update_fid(self, pred, gt, num_views: int=16):
        pred = rearrange(pred, "(b v) c h w -> b v c h w", v=num_views)
        gt = rearrange(gt, "(b v) c h w -> b v c h w", v=num_views)
        finite = (
            torch.isfinite(pred).reshape(pred.shape[0], -1).all(dim=1)
            & torch.isfinite(gt).reshape(gt.shape[0], -1).all(dim=1)
        )
        if not finite.all():
            print(f"[FID] dropping {(~finite).sum().item()}/{pred.shape[0]} non-finite samples")
            pred, gt = pred[finite], gt[finite]
        if pred.shape[0] == 0:
            return
        pred = (pred * 255).to(torch.uint8).to("cuda:0")
        gt = (gt * 255).to(torch.uint8).to("cuda:0")
        pred_flat = rearrange(pred, "b v c h w -> (b v) c h w")
        gt_flat = rearrange(gt, "b v c h w -> (b v) c h w")
        self.fid.update(gt_flat, real=True)
        self.fid.update(pred_flat, real=False)

    def update_fvd(self, pred, gt, num_views: int=16):

        pred = rearrange(pred, "(b v) c h w -> b v h w c", v=num_views)
        gt = rearrange(gt, "(b v) c h w -> b v h w c", v=num_views)
        finite = (
            torch.isfinite(pred).reshape(pred.shape[0], -1).all(dim=1)
            & torch.isfinite(gt).reshape(gt.shape[0], -1).all(dim=1)
        )
        if not finite.all():
            print(f"[FVD] dropping {(~finite).sum().item()}/{pred.shape[0]} non-finite samples")
            pred, gt = pred[finite], gt[finite]
        if pred.shape[0] == 0:
            return
        pred = (pred * 255).float()
        gt = (gt * 255).float()

        self.pred_list.append(pred.cpu())
        self.gt_list.append(gt.cpu())

    def reset_fid(self):
        self.fid.reset()
    def reset_fvd(self):
        self.pred_list = []
        self.gt_list = []
    @torch.no_grad()
    def compute_fid(self, pred=None, gt=None, update=True, num_views: int=16):
        
        if update:
            self.update_fid(pred, gt, num_views=num_views)
        
        fid = self.fid.compute()

        return fid
    
    @torch.no_grad()
    def compute_fvd(self, pred=None, gt=None, update=True, num_views: int=16):

        if update:
            self.update_fvd(pred, gt, num_views=num_views)

        # Return None (not NaN) so on_validation_end's `if value is not None`
        # check skips logging entirely — wandb sees no metric instead of NaN.
        if not self.pred_list or not self.gt_list:
            return None

        pred_tensor = torch.cat(self.pred_list, dim=0)
        gt_tensor = torch.cat(self.gt_list, dim=0)

        # FVD needs ≥ 2 video clips per set or `np.cov(rowvar=False)` produces a
        # rank-deficient covariance (degrees of freedom <= 0) which makes
        # `scipy.linalg.sqrtm` either return NaN or spin forever on Schur
        # decomposition. CLAUDE.md gotcha — skip rather than hang.
        if pred_tensor.shape[0] < 2 or gt_tensor.shape[0] < 2:
            print(
                f"[FVD] skip: insufficient samples "
                f"pred={pred_tensor.shape[0]}, gt={gt_tensor.shape[0]} (need ≥ 2)"
            )
            return None

        fvd_device = "cuda" if torch.cuda.is_available() else "cpu"
        fvd = frechet_video_distance(pred_tensor, gt_tensor, "submodules/fvd/pytorch_i3d_model/models/rgb_imagenet.pt", device=fvd_device)
        if not np.isfinite(fvd):
            # Final sanity guard: degenerate cov can still produce NaN/Inf
            # through scipy.sqrtm even with ≥ 2 samples. Skip logging.
            print(f"[FVD] skip: non-finite result ({fvd})")
            return None

        return torch.tensor(fvd)

    @torch.no_grad()
    def forward(self, pred, gt, num_views: int=16, **kwargs):
        _dict = {}

        for key in list(kwargs.keys()):
            if kwargs.get(key, False):
                value = getattr(self, f"compute_{key}")(pred, gt, num_views=num_views)
                # Convert non-finite scalar metrics to None so on_validation_end's
                # `if value is not None: log_metrics(...)` skips them — keeps
                # wandb clean of NaN entries when pred/gt is degenerate
                # (e.g. all-NaN model output in early training).
                if torch.is_tensor(value) and value.numel() == 1 and not torch.isfinite(value).item():
                    print(f"[Metric] skip {key}: non-finite value ({value.item()})")
                    value = None
                _dict[key] = value
                gc.collect()
                torch.cuda.empty_cache()

        return _dict



