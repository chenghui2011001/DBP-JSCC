"""Metrics helpers (PSNR etc.) for Aether-lite.

This module intentionally reuses the PSNR formulation from the
MambaJSCC project (see ``utils/metrics.py`` there), and adds a
BFCC/Bark-log spectrogram variant that is robust to NaNs/Infs and
shape mismatches.

The goal is to provide a stable BFCC PSNR that can be logged to CSV
during bit-only evaluation without sporadic failures.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def eval_matrix(recon_image: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PSNR (dB) between two images in [0, 1].

    This is a direct adaptation of ``eval_matrix`` from
    ``/home/bluestar/MambaJSCC/utils/metrics.py`` so that Aether-lite
    can share the same PSNR convention.
    """

    recon_image = torch.clamp(recon_image, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)

    mse = F.mse_loss(recon_image, target)
    if float(mse.item()) == 0.0:
        return float("inf")

    max_pixel = 1.0
    psnr = 20.0 * torch.log10(torch.tensor(max_pixel, device=mse.device) / torch.sqrt(mse))
    return float(psnr.item())


def _align_2d(ref: np.ndarray, deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Align two 2D (or higher) arrays on their last two dims.

    The BFCC/Bark features are typically shaped as [F, T] when passed
    in here, but this helper is robust to extra leading dimensions.
    """

    if ref.shape == deg.shape:
        return ref, deg

    if ref.ndim < 2 or deg.ndim < 2:
        return ref, deg

    f = min(ref.shape[-2], deg.shape[-2])
    t = min(ref.shape[-1], deg.shape[-1])
    ref_aligned = ref[..., :f, :t]
    deg_aligned = deg[..., :f, :t]
    return ref_aligned, deg_aligned


def bfcc_psnr(ref: np.ndarray, deg: np.ndarray) -> float:
    """Compute PSNR (dB) between two BFCC/Bark-log spectrograms.

    The implementation mirrors MambaJSCC's ``eval_matrix`` PSNR
    formulation but adds:

    - Alignment of the last two dimensions.
    - NaN/Inf sanitisation on the inputs.
    - Per-sample linear normalisation into [0, 1] using the reference
      BFCC dynamic range before calling :func:`eval_matrix`.
    """

    ref_np = np.asarray(ref, dtype=np.float32)
    deg_np = np.asarray(deg, dtype=np.float32)

    if ref_np.size == 0 or deg_np.size == 0:
        return 0.0

    ref_np, deg_np = _align_2d(ref_np, deg_np)

    # Replace NaN/Inf with finite values to keep PSNR well-defined.
    ref_np = np.nan_to_num(ref_np, nan=0.0, posinf=0.0, neginf=0.0)
    deg_np = np.nan_to_num(deg_np, nan=0.0, posinf=0.0, neginf=0.0)

    data_min = float(ref_np.min())
    data_max = float(ref_np.max())
    denom = max(data_max - data_min, 1e-6)

    ref_norm = (ref_np - data_min) / denom
    deg_norm = (deg_np - data_min) / denom

    ref_t = torch.from_numpy(ref_norm)
    deg_t = torch.from_numpy(deg_norm)

    return eval_matrix(deg_t, ref_t)


__all__ = ["eval_matrix", "bfcc_psnr"]

