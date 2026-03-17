#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight SSIM / MS-SSIM for 2D tensors (PyTorch)

Adapted from common implementations; designed for mel spectrogram images.
Usage:
    ms = MS_SSIM(data_range=1.0, channel=1, levels=4)
    loss = ms(x, y).mean()  # returns 1 - MS-SSIM
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _create_window_1d(window_size: int, sigma: float, channel: int) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    g = g.reshape(1, 1, 1, -1).repeat(channel, 1, 1, 1)
    return g


def _gaussian_filter(x: torch.Tensor, window_1d: torch.Tensor, use_padding: bool) -> torch.Tensor:
    """Apply separable Gaussian blur channel‑wise.

    This implementation is robust to the first dimension of ``window_1d``.
    We always interpret the kernel as a *single* 1D filter and broadcast it
    across all channels, then use depthwise conv (``groups=C``) so that the
    number of channels never changes.

    This avoids shape mismatches such as
        Given groups=1, weight of size [16, 1, 11, 1], expected input[...] to
        have 1 channels, but got 16 channels instead
    which can occur if the kernel accidentally carries ``out_channels > C``
    from a previous stage.
    """

    B, C, H, W = x.shape
    padding = (window_1d.shape[3] // 2) if use_padding else 0

    # Use a single 1D kernel and broadcast over channels.
    # ``window_1d`` may already have shape [C, 1, 1, K] or [1, 1, 1, K];
    # in either case we take the first kernel as the canonical filter.
    base = window_1d[:1].to(device=x.device, dtype=x.dtype)  # [1, 1, 1, K]
    kernel = base.expand(C, 1, base.shape[2], base.shape[3])  # [C, 1, 1, K]

    out = F.conv2d(x, kernel, stride=1, padding=(0, padding), groups=C)
    out = F.conv2d(out, kernel.transpose(2, 3), stride=1, padding=(padding, 0), groups=C)
    return out


def _ssim(X: torch.Tensor, Y: torch.Tensor, window: torch.Tensor, data_range: float, use_padding: bool = False):
    K1 = 0.01
    K2 = 0.03
    compensation = 1.0
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    mu1 = _gaussian_filter(X, window, use_padding)
    mu2 = _gaussian_filter(Y, window, use_padding)
    sigma1_sq = _gaussian_filter(X * X, window, use_padding)
    sigma2_sq = _gaussian_filter(Y * Y, window, use_padding)
    sigma12 = _gaussian_filter(X * Y, window, use_padding)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (sigma1_sq - mu1_sq)
    sigma2_sq = compensation * (sigma2_sq - mu2_sq)
    sigma12 = compensation * (sigma12 - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    cs_map = F.relu(cs_map)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_val = ssim_map.mean(dim=(1, 2, 3))
    cs = cs_map.mean(dim=(1, 2, 3))
    return ssim_val, cs


def _ms_ssim(X: torch.Tensor, Y: torch.Tensor, window: torch.Tensor, data_range: float, weights: torch.Tensor,
             use_padding: bool = False, eps: float = 1e-8) -> torch.Tensor:
    # Ensure the MS-SSIM weights live on the same device/dtype as inputs.
    # In many training scripts the MS_SSIM module is instantiated on CPU
    # while X/Y are on CUDA, which would otherwise trigger a device
    # mismatch in ``vals ** weights`` below.
    weights = weights.to(device=X.device, dtype=X.dtype)
    weights = weights[:, None]
    levels = weights.shape[0]
    vals = []
    for i in range(levels):
        # 对齐当前尺度下的 X/Y 空间尺寸，防御性裁剪
        if X.shape != Y.shape:
            try:
                B = min(X.size(0), Y.size(0))
                C = min(X.size(1), Y.size(1))
                H = min(X.size(2), Y.size(2))
                W = min(X.size(3), Y.size(3))
                X = X[:B, :C, :H, :W]
                Y = Y[:B, :C, :H, :W]
            except Exception:
                # 保持原状，让上层调用在必要时 fallback 到 L1
                pass

        # 动态调整窗口尺寸，确保不超过当前特征图的最小边
        _, C, H, W = X.shape
        base_k = int(window.shape[-1])
        k_eff = min(base_k, H, W)
        if k_eff % 2 == 0:
            k_eff = max(1, k_eff - 1)
        # 至少使用 1x1（退化为均值），更稳健可用3x3
        if k_eff < 3:
            k_eff = 3 if min(H, W) >= 3 else 1
        level_window = _create_window_1d(k_eff, 1.5, C).to(device=X.device, dtype=X.dtype)

        ss, cs = _ssim(X, Y, window=level_window, data_range=data_range, use_padding=use_padding)
        if i < levels - 1:
            vals.append(cs)
            X = F.avg_pool2d(X, kernel_size=2, stride=2, ceil_mode=True)
            Y = F.avg_pool2d(Y, kernel_size=2, stride=2, ceil_mode=True)
        else:
            vals.append(ss)
    vals = torch.stack(vals, dim=0).clamp_min(eps)
    ms_ssim_val = torch.prod(vals ** weights, dim=0)
    return ms_ssim_val


class MS_SSIM(torch.nn.Module):
    def __init__(self, window_size: int = 11, window_sigma: float = 1.5, data_range: float = 1.0, channel: int = 1,
                 use_padding: bool = False, weights: list[float] | None = None, levels: int | None = 4, eps: float = 1e-8):
        super().__init__()
        assert window_size % 2 == 1, 'Window size must be odd.'
        self.data_range = data_range
        self.use_padding = use_padding
        self.eps = eps
        window = _create_window_1d(window_size, window_sigma, channel)
        self.register_buffer('window', window)
        if weights is None:
            weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        w = torch.tensor(weights, dtype=torch.float32)
        if levels is not None:
            w = w[:levels]
            w = w / w.sum()
        self.register_buffer('weights', w)

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        # returns 1 - MS-SSIM
        # Ensure inputs are 4D [B, C, H, W]
        if X.dim() == 5:
            X = X.squeeze(1)
        if Y.dim() == 5:
            Y = Y.squeeze(1)
        if X.dim() == 3:
            X = X.unsqueeze(1)
        if Y.dim() == 3:
            Y = Y.unsqueeze(1)
        # Use buffers directly (they should already be on correct device after .to() call)
        ms = _ms_ssim(X, Y, window=self.window, data_range=self.data_range, weights=self.weights,
                      use_padding=self.use_padding, eps=self.eps)
        return 1.0 - ms
