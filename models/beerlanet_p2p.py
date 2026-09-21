from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from compat.beerlanet_autograd import BeerLaNetAutogradCompat
from models.detr import build_model as build_p2p_model


class BeerLaNetP2P(nn.Module):
    """BeerLaNet front end followed by the unchanged P2P detector."""

    def __init__(
        self,
        p2p,
        mean,
        std,
        r=8,
        n_iter=10,
        train_beerlanet=False,
        gradient_checkpointing=False,
    ):
        super().__init__()
        if r <= 0 or n_iter <= 0:
            raise ValueError("r and n_iter must be positive")
        self.p2p = p2p
        self.r = int(r)
        self.n_iter = int(n_iter)
        self.train_beerlanet = bool(train_beerlanet)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.beerlanet = BeerLaNetAutogradCompat(
            r=self.r,
            c=3,
            learn_S_init=True,
            calc_tau=True,
        )
        self.adapt_conv = nn.Conv2d(self.r, 3, kernel_size=1, bias=True)

        mean = torch.as_tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.as_tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        if mean.numel() != 3 or std.numel() != 3 or torch.any(std <= 0):
            raise ValueError("mean/std must contain three channels and std must be positive")
        # Dataset-specific preprocessing metadata must not be restored from a checkpoint.
        self.register_buffer("input_mean", mean, persistent=False)
        self.register_buffer("input_std", std, persistent=False)
        self.set_beerlanet_trainable(self.train_beerlanet)

    def set_beerlanet_trainable(self, trainable):
        self.train_beerlanet = bool(trainable)
        for parameter in self.beerlanet.parameters():
            parameter.requires_grad_(self.train_beerlanet)

    def restore_rgb(self, standardized_images):
        return (standardized_images * self.input_std + self.input_mean).clamp(0.0, 1.0)

    def _beer_forward(self, rgb):
        return self.beerlanet(rgb, n_iter=self.n_iter)

    def decompose(self, rgb):
        if not self.train_beerlanet:
            with torch.no_grad():
                x0, stain_matrix, density = self._beer_forward(rgb)
            return x0.detach(), stain_matrix.detach(), density.detach()

        use_checkpoint = (
            self.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
        )
        if use_checkpoint:
            if not rgb.requires_grad:
                rgb = rgb.detach().requires_grad_(True)
            return checkpoint(self._beer_forward, rgb)
        return self._beer_forward(rgb)

    def forward_components(self, standardized_images):
        rgb = self.restore_rgb(standardized_images)
        x0, stain_matrix, density = self.decompose(rgb)
        adapted = self.adapt_conv(density)
        return rgb, x0, stain_matrix, density, adapted

    def forward(self, standardized_images):
        _, _, _, _, adapted = self.forward_components(standardized_images)
        return self.p2p(adapted)


def load_mean_std(path):
    values = np.load(str(path), allow_pickle=False)
    if len(values) != 2:
        raise ValueError(f"expected mean/std pair in {path}")
    mean, std = values
    return np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)


def load_p2p_initialization(p2p, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"P2P initialization checkpoint not found: {checkpoint_path}")
    checkpoint_data = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint_data.get("model", checkpoint_data)
    incompatible = p2p.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "P2P initialization was not strict: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    return checkpoint_data.get("epoch")


def build_beerlanet_p2p(args, initialize_p2p=True):
    p2p = build_p2p_model(args)
    initialized_epoch = None
    if initialize_p2p:
        initialized_epoch = load_p2p_initialization(p2p, args.init_checkpoint)
    mean, std = load_mean_std(args.mean_std_path)
    model = BeerLaNetP2P(
        p2p=p2p,
        mean=mean,
        std=std,
        r=args.beerlanet_r,
        n_iter=args.beerlanet_n_iter,
        train_beerlanet=args.beerlanet_mode == "joint",
        gradient_checkpointing=args.beerlanet_gradient_checkpointing,
    )
    model.initialized_p2p_epoch = initialized_epoch
    return model
