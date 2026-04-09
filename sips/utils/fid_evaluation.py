import os
import math
import numpy as np
import torch
from einops import rearrange
from torch.nn.functional import adaptive_avg_pool2d
from tqdm.auto import tqdm
from torchvision.models import resnet18, ResNet18_Weights
from pytorch_fid.fid_score import calculate_frechet_distance

def num_to_groups(n, divisor):
    groups = n // divisor
    remainder = n % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

class FIDEvaluation:
    """
    FID evaluation using ResNet-18 features, suitable for small images like CIFAR-10.
    """

    def __init__(
        self,
        batch_size: int,
        dl,                     # DataLoader yielding real images in [0,1]
        sampler,                # diffusion sampler
        channels: int = 3,
        stats_dir: str = "./results",
        device: str = "cuda",
        stats_name: str = "dataset_stats_resnet",  # dataset identity
    ):
        self.batch_size = batch_size
        self.dl = dl
        self.sampler = sampler
        self.channels = channels
        self.device = device

        os.makedirs(stats_dir, exist_ok=True)
        self.stats_path = os.path.join(stats_dir, f"{stats_name}_c{channels}.npz")

        # ResNet-18 as feature extractor
        self.feature_net = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.feature_net.fc = torch.nn.Identity()  # remove final classifier
        self.feature_net.eval().to(device)
        for p in self.feature_net.parameters():
            p.requires_grad = False

        # ImageNet normalization
        self.norm = torch.nn.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )

        self.real_mu = None
        self.real_sigma = None
        self.dataset_stats_loaded = False

    @torch.no_grad()
    def _extract_features(self, x: torch.Tensor, chunk_size: int = 32):
        """
        Extract features from ResNet-18 in chunks to avoid OOM.
        Input x: [B,3,H,W] in [0,1]
        """
        all_feats = []
        for i in range(0, x.shape[0], chunk_size):
            chunk = x[i:i+chunk_size].to(self.device)
            # Upscale to 224x224
            chunk = torch.nn.functional.interpolate(chunk, size=(224,224), mode='bilinear', align_corners=False)
            # Normalize
            chunk = torch.stack([self.norm(img) for img in chunk])
            feats = self.feature_net(chunk)
            all_feats.append(feats)
        return torch.cat(all_feats, dim=0)

    @torch.no_grad()
    def compute_or_load_real_stats(self, force_recompute: bool = False):
        if os.path.exists(self.stats_path) and not force_recompute:
            data = np.load(self.stats_path)
            self.real_mu = data["mu"]
            self.real_sigma = data["sigma"]
            self.dataset_stats_loaded = True
            print("Loaded cached real dataset statistics (ResNet).")
            return

        print("Computing real dataset statistics (ResNet features)...")
        feats = []
        for batch in tqdm(self.dl):
            real = batch[0] if isinstance(batch, (list, tuple)) else batch
            real = real.clamp(0,1)
            feats.append(self._extract_features(real))
        feats = torch.cat(feats, dim=0).cpu().numpy()

        self.real_mu = np.mean(feats, axis=0)
        self.real_sigma = np.cov(feats, rowvar=False)

        np.savez_compressed(self.stats_path, mu=self.real_mu, sigma=self.real_sigma)
        self.dataset_stats_loaded = True
        print(f"Real dataset stats cached to {self.stats_path}")

    @torch.no_grad()
    def fid_score(self, num_samples: int, cfg_scale=None, sample_steps=None) -> float:
        assert self.dataset_stats_loaded, "Call compute_or_load_real_stats() first."

        batches = num_to_groups(num_samples, self.batch_size)
        fake_feats = []

        print(f"Generating {num_samples} samples for FID (ResNet features)...")
        for bs in tqdm(batches):
            fake = self.sampler.sample(batch_size=bs, cfg_scale=cfg_scale, sample_steps=sample_steps)
            fake = ((fake * 0.5) + 0.5).clamp(0,1)  # [-1,1] → [0,1]
            fake_feats.append(self._extract_features(fake))

        fake_feats = torch.cat(fake_feats, dim=0).cpu().numpy()
        mu_fake = np.mean(fake_feats, axis=0)
        sigma_fake = np.cov(fake_feats, rowvar=False)

        fid = calculate_frechet_distance(mu_fake, sigma_fake, self.real_mu, self.real_sigma)
        return float(fid)
