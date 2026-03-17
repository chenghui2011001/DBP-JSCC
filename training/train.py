#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DBP-JSCC training entrypoint ("Lite Loss, Heavy Prior").

This entrypoint wraps the existing :class:`DualBranchBarkJSCC` model but
uses a minimal, structurally-aligned loss bundle and a single
"train-as-deploy" forward path through the RVQ/hash bottlenecks.

Key design choices
------------------
- Forward always goes through ``forward_with_hash`` (RVQ/Hash + BPSK+AWGN).
- Losses are reduced to a small, interpretable set:

  * ``L_wave``   – multi-resolution STFT spectral convergence
                   (+ optional mag L1 term).
  * ``L_mel``    – Bark/BFCC structural similarity (historical
                   ``mel`` name kept for CLI compatibility) with
                   optional L1 brightness anchor.
  * ``L_ceps``   – cepstrum L1 reconstruction.
  * ``L_f0``     – F0 MSE, computed only on GT voiced frames.
  * ``L_vuv``    – BCE-with-logits on V/UV (using ``vuv_logits``).
  * ``L_vq``     – RVQ commitment/embedding loss from the bottlenecks.

- All higher-order F0/VUV regularisers, HF texture losses, teacher
  distillation, PHC-style constraints etc. are intentionally omitted
  here to keep optimisation focused and let architectural priors do
  the heavy lifting.

Usage (example)
---------------

.. code-block:: bash

    python training/train.py \
        --data_root ./data_cn \
        --num_epochs 5 --batch_size 8 --sequence_length 200 \
        --with_hash --quantizer_type rvq --hash_bits_content 16

This script is intentionally lean and reuses the proven data loading
and model-construction helpers from
``train_support.py``.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, List

from collections import deque
import numpy as np
import torch
import torch.nn.functional as F

# Optional wandb logging (kept lightweight; training works without it)
try:  # pragma: no cover - wandb is optional
    import wandb  # type: ignore
except ImportError:  # pragma: no cover - wandb is optional
    wandb = None  # type: ignore[assignment]


current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from models.dual_branch_bark_jscc import DualBranchBarkJSCC, WaveToBFCC
from training.spectral_losses import multi_resolution_sc_loss, multi_resolution_stft_loss
from utils.channel_sim import ChannelSimulator
from models.hifi_discriminators import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    BarkHFDiscriminator,
    F0PeriodDiscriminator,
    feature_loss as hifi_feature_loss,
    generator_loss as hifi_generator_loss,
    discriminator_loss as hifi_discriminator_loss,
)
from torch.cuda.amp import autocast

# Reuse robust dataloader/model wrappers from the v3 script
from training.train_support import (  # type: ignore
    build_dataloader,
    build_model,
    model_forward,
    _compute_content_only_losses as v3_compute_content_only_losses,
)
from utils.audio_visualizer import (
    create_batch_comparison_plots,
    create_source_control_plot,
    create_vocoder_internal_plot,
    save_comparison_audio_samples,
)
from utils.metrics import bfcc_psnr
import matplotlib.pyplot as plt


_JSCC_FSK_BER_TABLE: Optional[Dict[str, np.ndarray]] = None


def _lookup_jscc_fsk_ber(snr_db: float) -> Optional[float]:
    """Lookup JSCC+FSK BER from a precomputed table.

    When the environment variable ``JSCC_FSK_BER_TABLE`` is set to a JSON
    file produced from bit_only_metrics.csv (fields: snr_db, ber_mean),
    this helper returns an interpolated ``ber(snr_db)`` value in [0,0.5].

    On any error (missing env, bad file, etc.), it returns ``None`` so
    that callers can fall back to the analytic BPSK+AWGN formula.
    """

    import json as _json

    global _JSCC_FSK_BER_TABLE
    if _JSCC_FSK_BER_TABLE is None:
        table_path = os.environ.get("JSCC_FSK_BER_TABLE", "")
        if not table_path:
            return None
        try:
            with open(table_path, "r", encoding="utf-8") as _f:
                data = _json.load(_f)
            snr_arr = np.asarray(data.get("snr_db", []), dtype=np.float32)
            ber_mean_arr = np.asarray(data.get("ber_mean", []), dtype=np.float32)
            ber_std_arr = np.asarray(data.get("ber_std", []), dtype=np.float32)
            if ber_std_arr.size != snr_arr.size:
                ber_std_arr = np.zeros_like(snr_arr, dtype=np.float32)

            if snr_arr.size == 0 or ber_mean_arr.size == 0 or snr_arr.size != ber_mean_arr.size:
                return None

            order = np.argsort(snr_arr)
            snr_sorted = snr_arr[order]
            mean_sorted = ber_mean_arr[order]
            std_sorted = ber_std_arr[order]
            _JSCC_FSK_BER_TABLE = {"snr": snr_sorted, "mean": mean_sorted, "std": std_sorted}
        except Exception:
            # Cache an empty dict to avoid repeatedly trying to load
            _JSCC_FSK_BER_TABLE = {}
            return None

    if not _JSCC_FSK_BER_TABLE:
        return None

    snrs = _JSCC_FSK_BER_TABLE["snr"]
    means = _JSCC_FSK_BER_TABLE["mean"]
    stds = _JSCC_FSK_BER_TABLE["std"]
    x = float(snr_db)

    # Clamp to table range for extrapolation
    if x <= float(snrs[0]):
        mean_x = float(means[0])
        std_x = float(stds[0])
    elif x >= float(snrs[-1]):
        mean_x = float(means[-1])
        std_x = float(stds[-1])
    else:
        # Linear interpolation between nearest neighbours
        idx = int(np.searchsorted(snrs, x))
        x0 = float(snrs[idx - 1])
        x1 = float(snrs[idx])
        m0 = float(means[idx - 1])
        m1 = float(means[idx])
        s0 = float(stds[idx - 1])
        s1 = float(stds[idx])
        t = (x - x0) / (x1 - x0 + 1e-8)
        mean_x = m0 + t * (m1 - m0)
        std_x = s0 + t * (s1 - s0)

    std_x = max(0.0, std_x)
    mode = os.environ.get("JSCC_FSK_BER_MODE", "gaussian").lower()
    try:
        k_std = float(os.environ.get("JSCC_FSK_BER_STD_K", "1.0"))
    except Exception:
        k_std = 1.0

    if mode == "gaussian":
        eps = float(np.random.randn())
        val = mean_x + k_std * std_x * eps
    else:
        val = mean_x + k_std * std_x

    return float(max(0.0, min(val, 0.5)))


class MetricCorrWindow:
    """Maintain a sliding window of recent metrics and log a correlation heatmap.

    This is intentionally lightweight: only the last ``window`` steps are kept
    in memory, and correlation is computed on-demand every ``log_every`` steps.
    """

    def __init__(self, window: int = 2000, log_every: int = 500) -> None:
        self.window = int(window)
        self.log_every = int(log_every)
        self.buffers: Dict[str, deque[float]] = {}
        self.last_log_step: int = -1

    def update(self, metrics: Dict[str, float]) -> None:
        """Update metric buffers with the latest scalar values.

        Only finite scalar values are kept; everything else is ignored to keep
        the implementation safe and unobtrusive.
        """

        for key, value in metrics.items():
            val: Optional[float]
            if isinstance(value, (int, float)):
                val = float(value)
            else:
                # Best-effort support for zero-dim tensors (e.g. torch scalar)
                try:
                    import torch as _torch  # local import to avoid global dependency

                    if isinstance(value, _torch.Tensor) and value.dim() == 0:
                        val = float(value.item())
                    else:
                        continue
                except Exception:
                    continue

            if not np.isfinite(val):
                continue

            buf = self.buffers.get(key)
            if buf is None:
                buf = deque(maxlen=self.window)
                self.buffers[key] = buf
            buf.append(val)

    def maybe_log(self, step: int) -> None:
        """Compute and log a correlation heatmap for recent metrics.

        The heatmap is logged to wandb under the key ``metric_corr_window``
        when wandb logging is enabled. When wandb is absent or disabled, this
        method is a no-op.
        """

        if wandb is None:
            return
        if step - self.last_log_step < self.log_every:
            return
        self.last_log_step = step

        # Require at least a small buffer of samples to avoid noisy estimates.
        valid = {
            k: np.asarray(buf, dtype=np.float32)
            for k, buf in self.buffers.items()
            if len(buf) >= 32
        }
        if len(valid) < 2:
            return

        names = list(valid.keys())
        min_len = min(len(v) for v in valid.values())
        if min_len < 2:
            return

        data = np.stack([v[-min_len:] for v in valid.values()], axis=0)  # [M, T]
        x = data - data.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True) + 1e-8
        x_norm = x / std
        corr = (x_norm @ x_norm.T) / float(max(min_len - 1, 1))

        try:
            fig, ax = plt.subplots(
                figsize=(0.35 * len(names) + 2.0, 0.35 * len(names) + 2.0)
            )
            im = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap="coolwarm")
            ax.set_xticks(range(len(names)))
            ax.set_yticks(range(len(names)))
            ax.set_xticklabels(names, rotation=90, fontsize=6)
            ax.set_yticklabels(names, fontsize=6)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"Metric corr (last {min_len} steps)")
            fig.tight_layout()
            wandb.log({"metric_corr_window": wandb.Image(fig)}, step=int(step))
        except Exception:
            # Correlation diagnostics should never break training.
            pass
        finally:
            plt.close(fig)


def _build_fixed_period_patches(y: torch.Tensor, period: int) -> torch.Tensor:
    """Slice 1D waveforms into fixed-length periodic patches for F0PeriodDiscriminator.

    Args:
        y:      [B, L] waveform tensor.
        period: Target period in samples (e.g., 128).

    Returns:
        patches: [B, 1, N_frames, period] tensor. When the input length is
        shorter than ``period``, an empty tensor with N_frames=0 is returned.
    """

    if not isinstance(y, torch.Tensor):
        raise TypeError("_build_fixed_period_patches expects a Tensor")
    B, L = y.shape
    if period <= 0:
        raise ValueError("period must be positive")
    n_frames = L // period
    if n_frames <= 0:
        return y.new_zeros(B, 1, 0, int(period))
    y_seg = y[:, : n_frames * period]
    y_seg = y_seg.view(B, n_frames, period)
    return y_seg.unsqueeze(1)


def _build_period_patches_from_dnn_pitch(
    y: torch.Tensor,
    dnn_pitch: torch.Tensor,
    start_idx: int,
    end_idx: int,
    hop: int = 160,
    target_period: int = 128,
    frame_corr: Optional[torch.Tensor] = None,
    vuv_threshold: float = 0.3,
) -> torch.Tensor:
    """Build variable-period patches from waveform using FARGAN dnn_pitch.

    Args:
        y:          [B, L_crop] waveform (cropped segment).
        dnn_pitch:  [B, T, 1] FARGAN dnn_pitch sequence over the *full* audio.
        start_idx:  Crop start (sample index in full audio).
        end_idx:    Crop end (exclusive, sample index in full audio).
        hop:        Frame hop size in samples (default 160).
        target_period: Resampled period length for each patch.

    Returns:
        patches: [B, 1, N_frames, target_period]. If no valid frames
        overlap the crop, returns a zero tensor with N_frames=0.
    """

    if not isinstance(y, torch.Tensor) or not isinstance(dnn_pitch, torch.Tensor):
        raise TypeError("_build_period_patches_from_dnn_pitch expects Tensor inputs")

    B, L_crop = y.shape
    dp = dnn_pitch.detach().to(y.device).to(torch.float32).squeeze(-1)  # [B,T]
    T_all = dp.size(1)
    if T_all <= 0 or L_crop <= 0:
        return y.new_zeros(B, 1, 0, int(target_period))

    # Frame indices whose centers fall inside [start_idx, end_idx).
    start_frame = max(0, int(np.ceil(start_idx / float(hop))))
    end_frame = min(T_all - 1, int(np.floor((end_idx - 1) / float(hop))))
    if end_frame < start_frame:
        return y.new_zeros(B, 1, 0, int(target_period))

    frame_idx = torch.arange(
        start_frame,
        end_frame + 1,
        device=y.device,
        dtype=torch.long,
    )  # [Nf]
    Nf = frame_idx.numel()
    if Nf <= 0:
        return y.new_zeros(B, 1, 0, int(target_period))

    # Optional voiced-frame gate based on frame_corr (GT V/UV mask).
    voiced_mask: Optional[torch.Tensor] = None
    if isinstance(frame_corr, torch.Tensor):
        try:
            fc = frame_corr.detach().to(y.device).to(torch.float32).squeeze(-1)  # [B,T_fc]
            T_fc = fc.size(1)
            T_align = min(T_all, T_fc)
            if T_align <= 0:
                return y.new_zeros(B, 1, 0, int(target_period))

            # Restrict both dnn_pitch and frame indices to the common time span.
            if T_align < T_all:
                dp = dp[:, :T_align]
                T_all = T_align

            valid_mask = frame_idx < T_align
            if not bool(valid_mask.any()):
                return y.new_zeros(B, 1, 0, int(target_period))
            frame_idx = frame_idx[valid_mask]
            Nf = frame_idx.numel()
            if Nf <= 0:
                return y.new_zeros(B, 1, 0, int(target_period))

            voiced_mask = fc[:, frame_idx] > float(vuv_threshold)  # [B,Nf]
        except Exception:
            voiced_mask = None

    # Convert dnn_pitch logits to periods in samples (clamped to [32,255]).
    dp_frames = dp[:, frame_idx]  # [B,Nf]
    period = 256.0 / torch.pow(2.0, dp_frames + 1.5)
    period = period.clamp(32.0, 255.0)  # [B,Nf]

    patches_per_batch: list[torch.Tensor] = []
    for b in range(B):
        rows: list[torch.Tensor] = []
        for j in range(Nf):
            if voiced_mask is not None:
                # Skip unvoiced frames for this sample.
                if not bool(voiced_mask[b, j].item()):
                    continue
            p = float(period[b, j].item())
            if not np.isfinite(p) or p <= 4.0:
                continue
            center_full = int(frame_idx[j].item() * hop)
            center = center_full - int(start_idx)
            half = int(round(p / 2.0))
            s = center - half
            e = center + half
            if e <= 0 or s >= L_crop:
                continue
            s_clamp = max(0, s)
            e_clamp = min(L_crop - 1, e)
            if e_clamp - s_clamp + 1 < 4:
                continue
            seg = y[b, s_clamp : e_clamp + 1]  # [L_seg]
            seg = seg.unsqueeze(0).unsqueeze(0)  # [1,1,L_seg]
            seg_res = F.interpolate(
                seg,
                size=int(target_period),
                mode="linear",
                align_corners=False,
            ).squeeze(0).squeeze(0)  # [target_period]
            rows.append(seg_res)
        if rows:
            patches_per_batch.append(torch.stack(rows, dim=0))  # [Ni,P]

    # If no sample in the batch has any valid voiced frame within the crop,
    # return an empty tensor so that the caller can skip F0Period D/G.
    if not patches_per_batch:
        return y.new_zeros(B, 1, 0, int(target_period))

    # Use the minimum number of frames over *non-empty* samples so that at
    # least one sample contributes, without forcing all-B alignment.
    frame_sizes = [p.size(0) for p in patches_per_batch if p.size(0) > 0]
    if not frame_sizes:
        return y.new_zeros(B, 1, 0, int(target_period))
    min_frames = min(frame_sizes)
    if min_frames <= 0:
        return y.new_zeros(B, 1, 0, int(target_period))

    valid = [p[:min_frames] for p in patches_per_batch if p.size(0) >= min_frames]
    if not valid:
        return y.new_zeros(B, 1, 0, int(target_period))

    patch_tensor = torch.stack(valid, dim=0)  # [B_valid, N, target_period]
    return patch_tensor.unsqueeze(1)  # [B_valid,1,N,target_period]


def _build_period_patches_pair_from_dnn_pitch(
    y_real: torch.Tensor,
    y_fake: torch.Tensor,
    dnn_pitch_real: torch.Tensor,
    dnn_pitch_fake: torch.Tensor,
    start_idx: int,
    end_idx: int,
    hop: int = 160,
    target_period: int = 128,
    frame_corr: Optional[torch.Tensor] = None,
    vuv_threshold: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build aligned variable-period patches for (real, fake) waveforms.

    This helper constructs F0-aligned patches for the real and generated
    waveforms so that :class:`F0PeriodDiscriminator` sees matching
    [B_valid, 1, N_frames, target_period] tensors for both sides.
    Periods come from ``dnn_pitch_real`` / ``dnn_pitch_fake``; an
    optional ``frame_corr`` gate restricts patches to voiced frames.
    """

    if not (
        isinstance(y_real, torch.Tensor)
        and isinstance(y_fake, torch.Tensor)
        and isinstance(dnn_pitch_real, torch.Tensor)
        and isinstance(dnn_pitch_fake, torch.Tensor)
    ):
        raise TypeError("_build_period_patches_pair_from_dnn_pitch expects Tensor inputs")

    # Align batch and crop length.
    B_r, L_r = y_real.shape
    B_f, L_f = y_fake.shape
    B = min(B_r, B_f)
    if B <= 0:
        return (
            y_real.new_zeros(0, 1, 0, int(target_period)),
            y_real.new_zeros(0, 1, 0, int(target_period)),
        )
    L_crop = min(L_r, L_f)
    if L_crop <= 0:
        return (
            y_real.new_zeros(B, 1, 0, int(target_period)),
            y_real.new_zeros(B, 1, 0, int(target_period)),
        )
    y_r = y_real[:B, :L_crop]
    y_g = y_fake[:B, :L_crop]

    dp_r = dnn_pitch_real.detach().to(y_r.device).to(torch.float32).squeeze(-1)
    dp_g = dnn_pitch_fake.detach().to(y_r.device).to(torch.float32).squeeze(-1)
    T_r = dp_r.size(1)
    T_g = dp_g.size(1)
    if T_r <= 0 or T_g <= 0:
        return (
            y_r.new_zeros(B, 1, 0, int(target_period)),
            y_r.new_zeros(B, 1, 0, int(target_period)),
        )

    fc: Optional[torch.Tensor]
    if isinstance(frame_corr, torch.Tensor):
        fc = frame_corr.detach().to(y_r.device).to(torch.float32).squeeze(-1)
        T_fc = fc.size(1)
        T_all = min(T_r, T_g, T_fc)
    else:
        fc = None
        T_all = min(T_r, T_g)

    if T_all <= 0:
        return (
            y_r.new_zeros(B, 1, 0, int(target_period)),
            y_r.new_zeros(B, 1, 0, int(target_period)),
        )

    dp_r = dp_r[:, :T_all]
    dp_g = dp_g[:, :T_all]
    if fc is not None:
        fc = fc[:, :T_all]

    # Frame indices whose centers fall inside [start_idx, end_idx).
    start_frame = max(0, int(np.ceil(start_idx / float(hop))))
    end_frame = min(T_all - 1, int(np.floor((end_idx - 1) / float(hop))))
    if end_frame < start_frame:
        return (
            y_r.new_zeros(B, 1, 0, int(target_period)),
            y_r.new_zeros(B, 1, 0, int(target_period)),
        )

    frame_idx = torch.arange(
        start_frame,
        end_frame + 1,
        device=y_r.device,
        dtype=torch.long,
    )  # [Nf]
    Nf = frame_idx.numel()
    if Nf <= 0:
        return (
            y_r.new_zeros(B, 1, 0, int(target_period)),
            y_r.new_zeros(B, 1, 0, int(target_period)),
        )

    # Optional voiced-frame mask shared for real/fake to keep shapes aligned.
    voiced_mask: Optional[torch.Tensor] = None
    if fc is not None:
        try:
            voiced_mask = fc[:, frame_idx] > float(vuv_threshold)  # [B,Nf]
        except Exception:
            voiced_mask = None

    # Periods from dnn_pitch (clamped to [32,255]).
    dp_r_f = dp_r[:, frame_idx]
    dp_g_f = dp_g[:, frame_idx]
    period_r = 256.0 / torch.pow(2.0, dp_r_f + 1.5)
    period_g = 256.0 / torch.pow(2.0, dp_g_f + 1.5)
    period_r = period_r.clamp(32.0, 255.0)
    period_g = period_g.clamp(32.0, 255.0)

    patches_r: list[torch.Tensor] = []
    patches_g: list[torch.Tensor] = []
    for b in range(B):
        rows_r: list[torch.Tensor] = []
        rows_g: list[torch.Tensor] = []
        for j in range(Nf):
            if voiced_mask is not None and not bool(voiced_mask[b, j].item()):
                continue

            p_r = float(period_r[b, j].item())
            p_g = float(period_g[b, j].item())
            if not (np.isfinite(p_r) and np.isfinite(p_g)):
                continue
            if p_r <= 4.0 or p_g <= 4.0:
                continue

            center_full = int(frame_idx[j].item() * hop)
            center = center_full - int(start_idx)

            # Real patch window
            half_r = int(round(p_r / 2.0))
            s_r = center - half_r
            e_r = center + half_r
            # Fake patch window
            half_g = int(round(p_g / 2.0))
            s_g = center - half_g
            e_g = center + half_g

            if e_r <= 0 or s_r >= L_crop:
                continue
            if e_g <= 0 or s_g >= L_crop:
                continue

            s_r_c = max(0, s_r)
            e_r_c = min(L_crop - 1, e_r)
            s_g_c = max(0, s_g)
            e_g_c = min(L_crop - 1, e_g)

            if e_r_c - s_r_c + 1 < 4 or e_g_c - s_g_c + 1 < 4:
                continue

            seg_r = y_r[b, s_r_c : e_r_c + 1].unsqueeze(0).unsqueeze(0)
            seg_g = y_g[b, s_g_c : e_g_c + 1].unsqueeze(0).unsqueeze(0)

            seg_r_res = F.interpolate(
                seg_r,
                size=int(target_period),
                mode="linear",
                align_corners=False,
            ).squeeze(0).squeeze(0)
            seg_g_res = F.interpolate(
                seg_g,
                size=int(target_period),
                mode="linear",
                align_corners=False,
            ).squeeze(0).squeeze(0)

            rows_r.append(seg_r_res)
            rows_g.append(seg_g_res)

        if rows_r and rows_g:
            patches_r.append(torch.stack(rows_r, dim=0))
            patches_g.append(torch.stack(rows_g, dim=0))

    if not patches_r or not patches_g:
        return (
            y_r.new_zeros(B, 1, 0, int(target_period)),
            y_r.new_zeros(B, 1, 0, int(target_period)),
        )

    frame_sizes = [min(pr.size(0), pg.size(0)) for pr, pg in zip(patches_r, patches_g) if pr.size(0) > 0 and pg.size(0) > 0]
    if not frame_sizes:
        return (
            y_r.new_zeros(B, 1, 0, int(target_period)),
            y_r.new_zeros(B, 1, 0, int(target_period)),
        )

    min_frames = min(frame_sizes)
    if min_frames <= 0:
        return (
            y_r.new_zeros(B, 1, 0, int(target_period)),
            y_r.new_zeros(B, 1, 0, int(target_period)),
        )

    real_list: list[torch.Tensor] = []
    fake_list: list[torch.Tensor] = []
    for pr, pg in zip(patches_r, patches_g):
        if pr.size(0) >= min_frames and pg.size(0) >= min_frames:
            real_list.append(pr[:min_frames])
            fake_list.append(pg[:min_frames])

    if not real_list or not fake_list:
        return (
            y_r.new_zeros(B, 1, 0, int(target_period)),
            y_r.new_zeros(B, 1, 0, int(target_period)),
        )

    x_r = torch.stack(real_list, dim=0)  # [B_valid, N, P]
    x_g = torch.stack(fake_list, dim=0)
    return x_r.unsqueeze(1), x_g.unsqueeze(1)


def _compute_bark_hf_maps(
    wave_to_bfcc: WaveToBFCC,
    y_real: torch.Tensor,
    y_fake: torch.Tensor,
    hf_start_band: int = 16,
    frame_corr: Optional[torch.Tensor] = None,
    vuv_threshold: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute high-frequency Bark/BFCC maps for BarkHFDiscriminator.

    这里在原有基础上增加了两点：

    1) 基于 frame_corr 的 V/UV gating：仅在 GT 判为有声的帧上保留
       Bark HF 图，其余帧置零，从而避免在静音或本应噪声的
       区域上强行制造高频纹理；
    2) 对每帧 HF Bark 向量做 z-norm（减均值/除以标准差），让
       判别器更关注频带之间的相对图案，而不是单纯依赖整体
       亮度差异。

    Args:
        wave_to_bfcc: Shared WaveToBFCC front-end (kept in eval mode).
        y_real:       [B, L] reference waveform.
        y_fake:       [B, L] generated waveform.
        hf_start_band:Index of the first Bark band to treat as "high freq".
        frame_corr:   Optional GT V/UV sequence [B,T,1] or [B,T] used for
                      voiced gating.
        vuv_threshold:Threshold on frame_corr used to build voiced mask.

    Returns:
        (x_real, x_fake): both [B, 1, T, F_hf] tensors ready for 2D CNN.
    """

    if not isinstance(y_real, torch.Tensor) or not isinstance(y_fake, torch.Tensor):
        raise TypeError("_compute_bark_hf_maps expects Tensor inputs")

    bfcc_r = wave_to_bfcc(y_real)  # [B,T,32]
    bfcc_g = wave_to_bfcc(y_fake)

    B, T, Fm = bfcc_r.shape

    voiced_mask: Optional[torch.Tensor] = None
    if isinstance(frame_corr, torch.Tensor):
        try:
            fc = frame_corr.detach().to(y_real.device).to(torch.float32)
            if fc.dim() == 3:
                fc = fc.squeeze(-1)
            T_fc = fc.size(1)
            T_use = min(T, T_fc)
            if T_use > 0:
                bfcc_r = bfcc_r[:, :T_use, :]
                bfcc_g = bfcc_g[:, :T_use, :]
                fc = fc[:, :T_use]
                voiced_mask = (fc > float(vuv_threshold)).to(bfcc_r.dtype).unsqueeze(-1)
        except Exception:
            voiced_mask = None

    B, T, Fm = bfcc_r.shape
    hf = max(1, min(Fm - hf_start_band, Fm))
    if hf_start_band >= Fm:
        bfcc_hf_r = bfcc_r[:, :, -1:].contiguous()
        bfcc_hf_g = bfcc_g[:, :, -1:].contiguous()
    else:
        bfcc_hf_r = bfcc_r[:, :, hf_start_band : hf_start_band + hf].contiguous()
        bfcc_hf_g = bfcc_g[:, :, hf_start_band : hf_start_band + hf].contiguous()

    if voiced_mask is not None:
        vm = voiced_mask
        if vm.size(1) > T:
            vm = vm[:, :T, :]
        elif vm.size(1) < T:
            pad_t = T - vm.size(1)
            vm = torch.nn.functional.pad(vm, (0, 0, 0, pad_t))
        bfcc_hf_r = bfcc_hf_r * vm
        bfcc_hf_g = bfcc_hf_g * vm

    # Per-frame z-normalisation over HF Bark bands to emphasise relative
    # patterns rather than absolute brightness. Avoid in-place ops here:
    # the fake branch stays on the generator graph during GAN G update.
    mean_r = bfcc_hf_r.mean(dim=-1, keepdim=True)
    std_r = bfcc_hf_r.std(dim=-1, keepdim=True) + 1e-6
    bfcc_hf_r = (bfcc_hf_r - mean_r) / std_r

    mean_g = bfcc_hf_g.mean(dim=-1, keepdim=True)
    std_g = bfcc_hf_g.std(dim=-1, keepdim=True) + 1e-6
    bfcc_hf_g = (bfcc_hf_g - mean_g) / std_g

    x_real = bfcc_hf_r.unsqueeze(1)  # [B,1,T,F_hf]
    x_fake = bfcc_hf_g.unsqueeze(1)
    return x_real, x_fake


def _debug_log_gan_disc_stats(
    y_df_hat_r,
    y_df_hat_g,
    y_ds_hat_r,
    y_ds_hat_g,
    y_bark_r=None,
    y_bark_g=None,
    y_f0p_r=None,
    y_f0p_g=None,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
) -> None:
    """Print basic discriminator output statistics for MPD/MSD/Bark/F0D.

    Enabled only when ``DBG_GAN_STATS=1`` to keep training logs clean.
    """

    import os as _os_dbg

    if _os_dbg.environ.get("DBG_GAN_STATS", "0") != "1":
        return

    def _s(x: torch.Tensor) -> str:
        x = x.detach().to(torch.float32).view(-1)
        if x.numel() == 0:
            return "empty"
        return (
            f"mean={x.mean().item():+.3f} std={x.std().item():.3f} "
            f"min={x.min().item():+.3f} max={x.max().item():+.3f}"
        )

    head = "[GAN-DISC]"
    if epoch is not None and step is not None:
        head += f" epoch={epoch} step={step}"

    try:
        print(head)
        print("  MPD real:", _s(y_df_hat_r[0]))
        print("  MPD fake:", _s(y_df_hat_g[0]))
        print("  MSD real:", _s(y_ds_hat_r[0]))
        print("  MSD fake:", _s(y_ds_hat_g[0]))
        if isinstance(y_bark_r, torch.Tensor) and isinstance(y_bark_g, torch.Tensor):
            print("  Bark real:", _s(y_bark_r))
            print("  Bark fake:", _s(y_bark_g))
        if isinstance(y_f0p_r, torch.Tensor) and isinstance(y_f0p_g, torch.Tensor):
            print("  F0P  real:", _s(y_f0p_r))
            print("  F0P  fake:", _s(y_f0p_g))
    except Exception as _e_dbg:
        print(f"[GAN-DISC] stat logging failed: {_e_dbg}")


def _debug_log_gan_grads(
    model: torch.nn.Module,
    global_step: int,
    prefixes: Optional[Tuple[str, ...]] = None,
    tag: str = "default",
) -> None:
    """Print gradient norms for selected modules when DBG_GAN_GRAD=1.

    By default this keeps the previous behaviour (vocoder/F0-related
    modules). Callers can override ``prefixes`` to inspect a narrower
    subset, such as the scoped adversarial path used by
    ``adv_scope='vocoder_l2h_ceps'``.
    """

    import os as _os_dbg

    if _os_dbg.environ.get("DBG_GAN_GRAD", "0") != "1":
        return

    try:
        focus_prefixes = prefixes or (
            "vocoder",
            "f0vuv_dec",
            "hash_f0vuv",
        )
        print(f"[GAN-GRAD][{tag}] step={global_step} grad norms:")
        for name, param in model.named_parameters():
            if not isinstance(param, torch.nn.Parameter):
                continue
            if param.grad is None:
                continue
            if not any(name.startswith(pfx) for pfx in focus_prefixes):
                continue
            g = param.grad.detach().to(torch.float32)
            gnorm = float(g.norm().item())
            print(f"  {name}: {gnorm:.4e}")
    except Exception as _e_grad:
        print(f"[GAN-GRAD] grad logging failed: {_e_grad}")


def _accumulate_adv_grads_for_scope(
    loss_adv_total: torch.Tensor,
    model: torch.nn.Module,
    cfg: "TrainingConfig",
) -> None:
    """Accumulate GAN gradients only on a scoped subset of parameters.

    When ``cfg.adv_scope == 'vocoder_l2h_ceps'``, adversarial gradients are
    restricted to the vocoder, L2H stack and mel→ceps mapping modules. This
    is used to mimic a FARGAN-style adversarial fine-tuning regime where
    high-level conditioning (content/F0 branches) is treated as fixed input
    and only the excitation / refinement stack receives GAN pressure.
    """

    scope = str(getattr(cfg, "adv_scope", "full"))
    if scope != "vocoder_l2h_ceps":
        return

    if not isinstance(loss_adv_total, torch.Tensor) or loss_adv_total.grad_fn is None:
        return

    prefixes = (
        "vocoder",
        "deco_l2h_refiner",
        "l2h_flow",
        "mel18_to_ceps",
        "hf2ceps",
    )

    params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not isinstance(param, torch.nn.Parameter):
            continue
        if not param.requires_grad:
            continue
        if any(name.startswith(pfx) for pfx in prefixes):
            params.append(param)

    if not params:
        return

    grads = torch.autograd.grad(
        loss_adv_total,
        params,
        retain_graph=True,
        allow_unused=True,
    )

    for param, g in zip(params, grads):
        if g is None:
            continue
        g_det = g.detach()
        if param.grad is None:
            param.grad = g_det
        else:
            param.grad = param.grad + g_det


def _debug_log_crepe_f0_and_bark_mse(
    audio_real: torch.Tensor,
    audio_hat: torch.Tensor,
    device: torch.device,
    global_step: int,
) -> None:
    """Compute and log CREPE-based F0 MSE and Bark HF MSE for a small batch.

    This helper is only active when the environment variable
    ``DBG_F0_BARK=1`` is set, and is intended for A/B experiments
    rather than everyday training.
    """

    import os as _os_dbg

    if _os_dbg.environ.get("DBG_F0_BARK", "0") != "1":
        return

    try:
        import torchcrepe  # type: ignore
    except Exception:
        print("[DBG_METRIC] torchcrepe not available; skip CREPE F0 MSE")
        torchcrepe = None  # type: ignore

    B = min(audio_real.size(0), audio_hat.size(0), 2)  # limit to first 2 samples
    if B <= 0:
        return
    y_ref = audio_real[:B].to(device)
    y_hat = audio_hat[:B].to(device)

    sr = 16000
    hop = 160

    f0_mse_hz2: Optional[float] = None
    if torchcrepe is not None:
        try:
            f0_ref_list = []
            f0_hat_list = []
            for b in range(B):
                wav_r = y_ref[b:b+1]
                wav_h = y_hat[b:b+1]
                with torch.no_grad():
                    f0_r, p_r = torchcrepe.predict(
                        wav_r,
                        sr,
                        hop,
                        50.0,
                        500.0,
                        "tiny",
                        batch_size=512,
                        device=device,
                        return_periodicity=True,
                    )
                    f0_g, p_g = torchcrepe.predict(
                        wav_h,
                        sr,
                        hop,
                        50.0,
                        500.0,
                        "tiny",
                        batch_size=512,
                        device=device,
                        return_periodicity=True,
                    )
                f0_ref_list.append(f0_r[0].cpu())
                f0_hat_list.append(f0_g[0].cpu())

            f0_ref = torch.stack(f0_ref_list, dim=0)  # [B,T]
            f0_hat = torch.stack(f0_hat_list, dim=0)  # [B,T]
            Tm = min(f0_ref.size(1), f0_hat.size(1))
            f0_ref = f0_ref[:, :Tm]
            f0_hat = f0_hat[:, :Tm]
            mask = (f0_ref > 1.0) & (f0_hat > 1.0)
            if mask.any():
                diff2 = (f0_hat - f0_ref) ** 2
                f0_mse_hz2 = float(diff2[mask].mean().item())
        except Exception as _e_crepe:
            print(f"[DBG_METRIC] CREPE F0 MSE failed: {_e_crepe}")

    # Bark HF MSE using a lightweight WaveToBFCC front-end.
    bark_mse: Optional[float] = None
    try:
        wave_to_bfcc_dbg = WaveToBFCC(sample_rate=sr, n_fft=400, hop_length=hop, n_bands=32).to(device)
        wave_to_bfcc_dbg.eval()
        for p in wave_to_bfcc_dbg.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            bfcc_r = wave_to_bfcc_dbg(y_ref)  # [B,T,32]
            bfcc_g = wave_to_bfcc_dbg(y_hat)
        hf_start = 16
        Bm, Tm_b, Fm = bfcc_r.shape
        T_use = min(Tm_b, bfcc_g.size(1))
        bfcc_r = bfcc_r[:, :T_use, hf_start:]
        bfcc_g = bfcc_g[:, :T_use, hf_start:]
        if bfcc_r.numel() > 0:
            diff2_bark = (bfcc_g - bfcc_r) ** 2
            bark_mse = float(diff2_bark.mean().item())
    except Exception as _e_barkm:
        print(f"[DBG_METRIC] Bark HF MSE failed: {_e_barkm}")

    msg = f"[DBG_METRIC] step={global_step}"
    if f0_mse_hz2 is not None:
        msg += f" CREPE_F0_MSE={f0_mse_hz2:.2f} Hz^2"
    if bark_mse is not None:
        msg += f" Bark_HF_MSE={bark_mse:.4f}"
    print(msg)


def _save_source_control_plots_for_batch(
    out: Dict[str, torch.Tensor],
    save_dir: str,
    step: int,
    max_samples: int = 2,
    sr: int = 16000,
    hop_length: int = 160,
) -> None:
    """Export per-sample vocoder source-control diagnostics.

    This focuses on the four waveform-side controls most likely to explain
    sample-dependent voiced collapse: dnn_pitch, frame_corr, period and gain.
    """

    needed = (
        "dnn_pitch",
        "dnn_pitch_hat",
        "frame_corr",
        "frame_corr_hat",
        "period_vocoder",
        "ceps",
        "ceps_hat",
    )
    if not all(isinstance(out.get(k), torch.Tensor) for k in needed):
        return

    dnn_pitch_ref = out["dnn_pitch"].detach().cpu()
    dnn_pitch_hat = out.get("dnn_pitch_vocoder", out["dnn_pitch_hat"]).detach().cpu()
    frame_corr_ref = out["frame_corr"].detach().cpu()
    frame_corr_hat = out.get("frame_corr_vocoder", out["frame_corr_hat"]).detach().cpu()
    period_vocoder = out["period_vocoder"].detach().cpu()
    ceps_ref = out["ceps"].detach().cpu()
    ceps_hat = out.get("ceps_vocoder", out["ceps_hat"]).detach().cpu()

    batch_size = min(
        int(max_samples),
        dnn_pitch_ref.size(0),
        dnn_pitch_hat.size(0),
        frame_corr_ref.size(0),
        frame_corr_hat.size(0),
        period_vocoder.size(0),
        ceps_ref.size(0),
        ceps_hat.size(0),
    )
    if batch_size <= 0:
        return

    gain_ref = 0.03 * torch.pow(10.0, 0.5 * ceps_ref[..., 0:1] / np.sqrt(18.0))
    gain_hat = 0.03 * torch.pow(10.0, 0.5 * ceps_hat[..., 0:1] / np.sqrt(18.0))

    for i in range(batch_size):
        save_path = os.path.join(
            save_dir,
            f"source_ctrl_step_{step:06d}_sample_{i:02d}.png",
        )
        title = f"Source Controls - Step {step} - Sample {i}"
        try:
            create_source_control_plot(
                dnn_pitch_ref=dnn_pitch_ref[i],
                dnn_pitch_hat=dnn_pitch_hat[i],
                frame_corr_ref=frame_corr_ref[i],
                frame_corr_hat=frame_corr_hat[i],
                period_vocoder=period_vocoder[i],
                gain_ref=gain_ref[i],
                gain_hat=gain_hat[i],
                save_path=save_path,
                sr=sr,
                hop_length=hop_length,
                title=title,
            )
        except Exception as _e_sc:
            print(f"[Simplified] source-control plot failed for sample {i}: {_e_sc}")


def _save_vocoder_internal_plots_for_batch(
    out: Dict[str, torch.Tensor],
    save_dir: str,
    step: int,
    max_samples: int = 2,
    sr: int = 16000,
    hop_length: int = 160,
) -> None:
    """Export per-sample vocoder internal traces."""

    needed = (
        "vocoder_pitch_gain_mean",
        "vocoder_fwc0_rms",
        "vocoder_skip_rms",
        "vocoder_sig_core_rms",
        "vocoder_sig_out_rms",
    )
    if not all(isinstance(out.get(k), torch.Tensor) for k in needed):
        return

    pg = out["vocoder_pitch_gain_mean"].detach().cpu()
    fwc = out["vocoder_fwc0_rms"].detach().cpu()
    sk = out["vocoder_skip_rms"].detach().cpu()
    sc = out["vocoder_sig_core_rms"].detach().cpu()
    so = out["vocoder_sig_out_rms"].detach().cpu()

    batch_size = min(int(max_samples), pg.size(0), fwc.size(0), sk.size(0), sc.size(0), so.size(0))
    if batch_size <= 0:
        return

    for i in range(batch_size):
        save_path = os.path.join(
            save_dir,
            f"vocoder_internal_step_{step:06d}_sample_{i:02d}.png",
        )
        title = f"Vocoder Internals - Step {step} - Sample {i}"
        try:
            create_vocoder_internal_plot(
                pitch_gain_mean=pg[i],
                fwc0_rms=fwc[i],
                skip_rms=sk[i],
                sig_core_rms=sc[i],
                sig_out_rms=so[i],
                save_path=save_path,
                sr=sr,
                hop_length=hop_length,
                title=title,
            )
        except Exception as _e_vi:
            print(f"[Simplified] vocoder-internal plot failed for sample {i}: {_e_vi}")


@dataclass
class TrainingConfig:
    """Minimal configuration for public DBP-JSCC training.

    This config mirrors the key knobs from the main Stage2.5 script so
    that VMamba/JSCC/quantizer structure and bit allocation can be kept
    identical when desired (e.g., when resuming from an existing v3
    checkpoint).
    """

    # Data / training loop
    data_root: str = os.path.join(root_dir, "data_cn")
    batch_size: int = 8
    sequence_length: int = 200
    num_epochs: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Optional AMP for selected paths (e.g., HiFi-GAN adversarial loss)
    use_amp: bool = False
    lr: float = 1e-4

    # Channel SNR range (train-time JSCC robustness envelope)
    snr_min_db: float = -5.0
    snr_max_db: float = 15.0
    # Optional: discrete SNR grid step (dB) when using ChannelSimulator.
    # When >0, base SNR is drawn from [snr_min_db, snr_min_db+step, ...].
    snr_step_db: float = 1.0

    # Quantiser / codec options (delegated to DualBranchBarkJSCC)
    with_hash: bool = True
    quantizer_type: str = "rvq"  # "rvq" recommended for this branch
    hash_bits_content: int = 16
    hash_bits_f0: Optional[int] = None
    rvq_nq_content: int = 2
    rvq_nq_f0: Optional[int] = None
    rvq_beta: float = 0.25
    # Latent symbol dims (must stay aligned with the main Stage2.5 config so
    # that bit-rate and JSCC structure are identical when using the same CLI).
    d_s_content: int = 8
    freq_downsample_stages: int = 2
    eq_fading: bool = False
    # VMamba content branch channel/depth config; if left as ``None`` the
    # defaults inside :class:`DualBranchBarkJSCC` are used, exactly as in the
    # main Stage2.5 script.
    vm_channels: Optional[List[int]] = None
    vm_depths: Optional[List[int]] = None
    vm_channel_adaptive: str = "no"           # "no" | "ca" | "attn" | "ssm"
    vm_lightweight_config: str = "all_native"  # same choices as v3 script

    # Content-only mode: skip F0/VUV and vocoder; train only the
    # content JSCC front-end (wave -> Bark/BFCC image -> VMamba/JSCC -> bark_hat)
    # with Bark-domain losses. When ``with_l2h`` is enabled the DeCo
    # L2H refiner is also included in the trainable path so that
    # content-only runs can jointly optimise Bark+L2H. Forward path
    # mirrors v3's content_only routing via forward_content_only /
    # forward_content_only_no_hash.
    content_only: bool = False

    # Ceps-only fine-tuning mode: when True, freeze all model parameters
    # except the mel→ceps mapping (mel18_to_ceps / hf2ceps) and optimise
    # primarily ceps-related losses (lambda_ceps / lambda_ceps_hi). This
    # is intended for short fine-tuning runs to realign the ceps space
    # without disturbing the rest of the JSCC/vocoder stack.
    ceps_only: bool = False
    ceps_only_lr: float = 1e-4

    # Optional: enable a self-contained learned energy calibration head
    # inside DualBranchBarkJSCC. When True, the model will predict a
    # per-sample c0 offset from its own mel_used statistics (no GT
    # FARGAN ceps), so that energy alignment is learned and shared
    # between training and inference.
    use_learned_energy_calib: bool = False

    # L2H (low→high mel refinement) configuration。
    #
    # 为了与 v3 版 Stage2.5 脚本中常用的 L2H 配置完全对齐，
    # 这里将默认的 low_bins 设为 18。这样在使用 --with_l2h 且
    # 从同一 v3 checkpoint resume 时，DeCoL2HRefiner 的参数形状
    # 将与原训练一致，不会出现权重无法恢复的情况。
    with_l2h: bool = False
    l2h_low_bins: int = 18
    use_l2h_flow: bool = False
    l2h_flow_hidden: int = 128
    l2h_flow_n_flows: int = 4
    deco_l2h: bool = False
    deco_l2h_hidden: int = 64
    deco_l2h_blocks: int = 3

    # Core loss weights (recommended starting point)
    lambda_wave: float = 0.7   # MR-STFT spectral convergence
    # Optional MR-STFT log-magnitude L1 term (0 to disable).  Exposed via
    # --lambda_wave_mag so that SC + log-mag can be used together in the
    # standard ParallelWaveGAN / HiFi-GAN fashion when desired.
    lambda_wave_mag: float = 0.0
    # Optional waveform frame-RMS envelope loss. When >0, we add an L1
    # loss between log10 frame RMS envelopes of (audio, audio_hat), which
    # acts as a relatively hard anchor on the coarse loudness/energy
    # trajectory in the time domain.
    lambda_wave_rms: float = 0.0
    lambda_mel: float = 0.3    # mel structural similarity (MS-SSIM)
    lambda_mel_l1: float = 0.3 # optional mel L1 anchor
    lambda_mel_energy: float = 0.0  # global mel energy/brightness anchor (v3-style)
    # 专门约束 bits_stats 重建的 mel_mean_hat/mel_std_hat 与 GT mel_mean/mel_std
    # 之间的偏差，补充 lambda_mel_energy 对整体亮度的软约束。仅在 with_hash
    # 路径上生效（forward_with_hash / bits decode）。
    lambda_mel_stats: float = 0.0
    lambda_ceps: float = 0.3   # cepstrum L1
    lambda_f0: float = 0.8     # F0 MSE (voiced-only)
    lambda_f0_env: float = 0.0 # optional F0 envelope hinge loss
    lambda_f0_smooth: float = 0.0  # F0 二阶平滑损失
    f0_env_margin_cents: float = 80.0
    f0_env_window: int = 3
    f0_env_alpha: float = 0.5
    f0_presence_gamma: float = 0.2
    lambda_vuv: float = 0.5    # V/UV BCE-with-logits (vuv_logits vs label)
    lambda_vuv_bce: float = 0.0  # VUV prob BCE (frame_corr_hat vs label)
    # Cepstrum high-order supervision（如 c10..），抑制有声段高频抹平
    lambda_ceps_hi: float = 0.0
    ceps_hi_start: int = 10
    # Mel 高频 patch 结构（高频区域 patch MS-SSIM）
    lambda_mel_hp: float = 0.0
    mel_hp_low_bins: int = 16
    # Mel 高频能量 L1 约束（直接对高频振幅差做 L1），
    # 以补充 mel_hp 在接近噪声底区域对绝对能量不敏感的问题。
    lambda_mel_hf_l1: float = 0.0
    # 静音帧上的高频 BFCC 抑制：仅作用于内容 BFCC
    lambda_silence_mel: float = 0.0
    silence_hf_low_bins: int = 16
    silence_mel_thr_db: float = -35.0
    lambda_vq_c: float = 0.05  # content RVQ loss
    lambda_vq_f: float = 0.01  # F0 RVQ loss
    lambda_vq_stats: float = 0.0  # stats RVQ loss (hash_content_stats)
    # Optional L2H high-frequency residual constraint（仅在 enable_l2h=True 时生效）。
    # 建议从 0.02~0.05 小范围尝试，避免对简化版训练施加过强约束。
    lambda_l2h: float = 0.0

    # JSCC bit entropy / balance 正则（仅在 with_hash 或 RVQ 时生效）
    lambda_c_entropy: float = 0.0
    content_entropy_target_frac: float = 0.5
    lambda_f0_entropy: float = 0.0
    f0_entropy_target_frac: float = 0.5
    lambda_bit_balance_c: float = 0.0

    # 高频 STFT / 纹理保护损失（音频域）
    lambda_hf_stft: float = 0.0
    lambda_texture_protect: float = 0.0
    texture_hf_start: int = 40
    # Optional frequency-weighted STFT knobs. By default we keep
    # ``stft_hf_start_hz=0`` and ``stft_hf_weight=1.0`` so that MR-STFT
    # behaves like the standard, unweighted spectral convergence / mag
    # loss used in PWG/HiFi-GAN. Only advanced experiments need to
    # change these.
    stft_hf_start_hz: float = 0.0
    stft_hf_weight: float = 1.0
    # Additional BarkHF-domain HF energy clamp: only penalise frames
    # where fake HF Bark energy exceeds real + margin.
    lambda_bark_hf_energy: float = 0.0
    bark_hf_energy_margin_db: float = 3.0
    # Optional HF Bark-domain L1 anchor on the (gated + normalised)
    # high-frequency Bark maps used by BarkHFDiscriminator. When >0, this
    # encourages the generator's HF Bark texture to stay close to the
    # reference even under adversarial training.
    lambda_bark_hf_l1: float = 0.0
    # Bit-only BFCC silence loss（bits→audio→BFCC 路径上的静音抑制）。
    lambda_bit_only_silence: float = 0.0

    # Adversarial loss on raw waveform (MPD + MSD + BarkHF + F0Period).
    # 仅在 lambda_gan_adv 或 lambda_gan_fm > 0 时启用，对 audio_hat 引入
    # LSGAN + feature matching 约束，以增强周期性与高频纹理。
    lambda_gan_adv: float = 0.0
    lambda_gan_fm: float = 0.0
    gan_adv_warmup_steps: int = 0
    gan_disc_lr: float = 1e-4
    gan_adv_crop_len: int = 16000
    # Optional discriminator LR decay factor applied once after the
    # GAN warm-up window. When set to a value != 1.0 and
    # ``gan_adv_warmup_steps > 0``, the discriminator optimizer learning
    # rate is multiplied by this factor when
    # ``gan_steps_since_resume >= gan_adv_warmup_steps``.
    gan_disc_lr_decay: float = 1.0

    # Adversarial gradient scope: controls which generator sub-modules
    # receive gradients from GAN losses. When set to "full" (default),
    # adversarial gradients are applied to all trainable parameters
    # (previous behaviour). When set to "vocoder_l2h_ceps", only the
    # vocoder, L2H stack (deco_l2h_refiner / l2h_flow) and mel→ceps
    # mapping (mel18_to_ceps / hf2ceps) receive GAN gradients; other
    # modules are updated only by reconstruction losses.
    adv_scope: str = "full"

    # V/UV threshold used to build GT voiced mask from frame_corr
    vuv_threshold: float = 0.3

    # Optional: pipeline probe mode. When enabled, the script will not
    # run the training loop but instead feed a small, fixed number of
    # audio segments through both train/eval forward paths and print
    # component-level feature statistics via DBG_PIPELINE-style helpers.
    pipeline_probe: bool = False
    pipeline_probe_num_samples: int = 10

    # Checkpoint / logging
    ckpt_dir: str = os.path.join(root_dir, "outputs", "checkpoints")
    save_every_steps: int = 1000
    # Optional pretrained FARGAN checkpoint (kept for API parity with v3
    # script; default None means "do not load").
    vocoder_ckpt: Optional[str] = None
    # If True, re-load FARGAN weights from ``fargan_ckpt`` *after* resuming
    # from a Stage2.5 checkpoint, overriding the student's vocoder core.
    # Mirrors the behaviour in the full Stage2.5 script.
    reload_vocoder_after_resume: bool = False
    # Optional content-branch checkpoint: when provided together with
    # ``reload_content_after_resume=True``, a second Stage2.5 checkpoint is
    # loaded *after* the main resume, but only for the content JSCC branch
    # (e.g., ``content_vmamba`` / ``hash_content`` / ``hash_content_stats``).
    # This mirrors the FARGAN reload behaviour and is useful when you want
    # to re-use a particular content encoder/decoder from another run while
    # keeping F0/vocoder/etc. from the main ``--resume`` checkpoint.
    content_ckpt: Optional[str] = None
    reload_content_after_resume: bool = False
    # Visualization
    viz_dir: str = os.path.join(root_dir, "outputs", "visualizations")
    viz_every_steps: int = 1000
    viz_max_samples: int = 2
    # Convenience: if set, overrides ``ckpt_dir`` and ``viz_dir`` with
    # ``out_dir/checkpoints`` and ``out_dir/viz``.
    out_dir: Optional[str] = None
    # Optional: resume from an existing checkpoint produced either by this
    # simplified script or by the main Stage2.5 script. When provided, the
    # training loop will restore model/optimizer/global_step/epoch and
    # continue from there.
    resume: Optional[str] = None
    # When True, keep the vocoder fully frozen for all training steps. This
    # mirrors the ``freeze_vocoder_all`` flag in the full Stage2.5 script and
    # is useful when focusing purely on the JSCC front-end.
    freeze_vocoder_all: bool = False
    # When True, keep the *entire* content branch (VMamba encoder/decoder,
    # content VQ/hash, content predictors, etc.) frozen for all training
    # steps, including the decoder head and hash_content_stats. This is
    # analogous to ``freeze_vocoder_all`` but for the content JSCC path.
    freeze_content_all: bool = False
    # Selective overrides for ``freeze_content_all``. These flags are kept
    # in the dataclass so they survive ``asdict(cfg)`` checkpoint exports.
    unfreeze_ceps_map: bool = False
    unfreeze_l2h: bool = False
    unfreeze_stats: bool = False
    # FARGAN runtime gating controls. These are CLI-visible counterparts of
    # the previous environment-only knobs and are synced onto
    # ``model.vocoder.fargan_core`` after construction.
    vocoder_strict_vuv_gate: bool = True
    vocoder_final_voicing_gate: bool = True
    vocoder_final_voicing_gate_floor: float = 0.0
    vocoder_final_voicing_gate_gamma: float = 1.0
    vocoder_silence_gate: bool = True
    vocoder_silence_gate_floor: float = 0.0
    vocoder_silence_energy_thr_db: float = -40.0
    vocoder_silence_gate_width_db: float = 6.0
    vocoder_pitch_gain_scale: float = 1.0
    vocoder_sig_core_scale: float = 1.0
    viz_source_controls: bool = True
    viz_vocoder_internals: bool = True
    oracle_swap_source_controls: str = "none"
    # Logging frequency (stdout); controls how often a one-line summary of
    # losses is printed during training.
    log_every_steps: int = 100
    # Bit-only evaluation: when enabled, periodically run a decode_from_bits
    # path (JSCC bits -> audio_hat_bits) on a few samples and export
    # comparison plots/audio for debugging.
    bit_only_eval: bool = False
    bit_only_eval_max_samples: int = 2
    # Optional: BFCC/Bark-domain validation on the *forward* path (audio →
    # WaveToBFCC) at visualization steps, independent of bit-only eval / RVQ.
    # When enabled, we compute BFCC for (audio, audio_hat) using
    # model.wave_to_mel and save GT/pred images + PSNR.
    bfcc_forward_eval: bool = False
    bfcc_forward_max_samples: int = 2
    # Eval-only mode: when True, skip training and run a single
    # pass over the dataloader to populate bit_only_metrics.csv
    # without producing plots or updating model parameters.
    only_eval: bool = False
    # wandb logging (mirrors main Stage2.5 script semantics)
    use_wandb: bool = False
    wandb_project: str = "DBP-JSCC"
    wandb_run_name: Optional[str] = None
    wandb_log_freq: int = 10
    # Optional path to a JSCC+FSK BER(SNR) JSON table; when set, this
    # overrides the JSCC_FSK_BER_TABLE environment variable for the
    # current training/eval run.
    fsk_ber_table: Optional[str] = None
    # Optional second Stage2.5 checkpoint for the content branch.
    content_ckpt: Optional[str] = None
    reload_content_after_resume: bool = False
    # Warm-up steps before unfreezing content JSCC parameters. When >0,
    # content-related modules (content_vmamba / hash_content /
    # hash_content_stats / content_pred_*) are kept frozen until
    # global_step >= content_warmup_steps, which can be useful when
    # reloading content from another checkpoint and only wanting to
    # fine-tune it after stabilising F0/vocoder/etc.
    content_warmup_steps: int = 0
    # VQ-only training mode: when True, freeze all parameters except the
    # RVQ/Hash bottlenecks in the content/F0 branches (hash_content and
    # hash_f0vuv) so that only codebooks are updated. This is useful for
    # refining quantisers on top of a fixed encoder/decoder.
    vq_only_train: bool = False
    # When True, ignore stats bits (hash_content_stats) when reconstructing
    # mel in forward_with_hash / decode_from_bits_offline: mel_hat is taken
    # directly from content_vmamba decoder output (mel_hat_norm) without
    # applying mean/std from stats bits. This is mainly for comparison
    # experiments where you want to rely purely on the content branch to
    # recover mel energy.
    ignore_stats_in_mel: bool = False

    # Stats-only fine-tuning mode: when True, freeze all modules except
    # ``hash_content_stats`` and train it to reconstruct (mel_mean, mel_std)
    # in the normalised (mean_norm, log_std_norm) space via an L1 loss.
    # This is useful to pull the stats bottleneck out of a collapsed
    # constant solution before running full JSCC training.
    stats_only: bool = False
    stats_only_lr: float = 1e-4
    stats_only_max_steps: int = 10000


def _compute_l2h_resid_and_decor_losses(
    out: Dict[str, torch.Tensor],
    cfg: TrainingConfig,
    device: torch.device,
    model: DualBranchBarkJSCC,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute L2H residual + decorrelation losses from model outputs.

    This helper mirrors the v3-style L2H supervision used in the full
    Stage2.5 script. It is factored out so that both the main
    ``compute_losses_simplified`` path and the ``content_only`` path
    can attach identical L2H losses when ``with_l2h`` is enabled.
    """

    lam_l2h_resid = float(getattr(cfg, "lambda_l2h_resid", 0.0))
    lam_l2h_decor = float(getattr(cfg, "lambda_l2h_decor", 0.0))
    improve_margin = float(getattr(cfg, "l2h_improve_margin", 0.0))

    loss_l2h = torch.zeros((), device=device)
    loss_dict_l2h: Dict[str, float] = {}

    if lam_l2h_resid <= 0.0 and lam_l2h_decor <= 0.0:
        return loss_l2h, loss_dict_l2h

    pred_resid = out.get("l2h_resid")
    vuv_prob = out.get("l2h_vuv_prob")
    mask_harm = out.get("l2h_mask_harm")
    mel_base = out.get("mel_hat")
    mel_gt = out.get("mel")

    if not (
        isinstance(pred_resid, torch.Tensor)
        and isinstance(vuv_prob, torch.Tensor)
        and isinstance(mel_base, torch.Tensor)
        and isinstance(mel_gt, torch.Tensor)
    ):
        return loss_l2h, loss_dict_l2h

    pred_resid = pred_resid.to(device)
    vuv_prob = vuv_prob.to(device)
    mel_base = mel_base.to(device)
    mel_gt = mel_gt.to(device)

    # High-frequency split point for L2H supervision
    lb = int(getattr(cfg, "l2h_low_bins", 10))
    lb = max(1, min(lb, mel_base.size(-1) - 1))

    base_high = mel_base[:, :, lb:]
    tgt_high = mel_gt[:, :, lb:]

    T_use = min(pred_resid.size(1), base_high.size(1), tgt_high.size(1))
    H_use = min(pred_resid.size(2), base_high.size(2), tgt_high.size(2))
    if T_use <= 0 or H_use <= 0:
        return loss_l2h, loss_dict_l2h

    pred_resid = pred_resid[:, :T_use, :H_use]
    base_high = base_high[:, :T_use, :H_use]
    tgt_high = tgt_high[:, :T_use, :H_use]
    vuv_prob = vuv_prob[:, :T_use, :1]

    # target_resid: (GT_high - base_high_det) – only let L2H learn
    # residuals in the HF band; improvement-style loss should still
    # compare *absolute* HF spectra (baseline vs refined) against
    # the GT HF target.
    base_high_det = base_high.detach()
    target_resid = tgt_high - base_high_det

    # Optional limiter to prevent very large residuals in early training
    resid_clip = float(getattr(cfg, "l2h_resid_clip", 0.0))
    if resid_clip > 0.0:
        pred_resid = pred_resid.clamp(-resid_clip, resid_clip)

    # Silence mask based on log-mel energy (log10 domain)
    mel_energy = mel_gt[:, :T_use, :].mean(dim=-1)
    thr_db = float(getattr(cfg, "silence_rms_thr_db", -35.0))
    thr_log = thr_db / 10.0
    silence_mask = mel_energy <= thr_log  # [B,T] bool

    # Gate: blend * voiced_prob * non-silence (matches v3 structure)
    blend = float(getattr(model, "l2h_blend", 1.0))
    gate = (blend * vuv_prob).clamp(0.0, 1.0)
    gate = gate * (~silence_mask[:, :T_use]).to(gate.dtype).unsqueeze(-1)

    # Harmonic/noise masks alignment
    if not isinstance(mask_harm, torch.Tensor):
        mask_harm_t = torch.zeros(1, 1, H_use, device=device, dtype=pred_resid.dtype)
    else:
        mh = mask_harm
        if mh.dim() == 3:
            mask_harm_t = mh[:, :, :H_use]
        elif mh.dim() == 2:
            mask_harm_t = mh.unsqueeze(0).unsqueeze(0)[:, :, :H_use]
        else:
            mask_harm_t = mh.reshape(1, 1, -1)[:, :, :H_use]
        mask_harm_t = mask_harm_t.to(device=device, dtype=pred_resid.dtype)

    harm_mask = mask_harm_t
    noise_mask = 1.0 - mask_harm_t

    # Harmonic path uses full gate; noise path only uses non-silence
    gate_h = gate
    gate_n = torch.ones_like(gate)
    gate_n = gate_n * (~silence_mask[:, :T_use]).to(gate.dtype).unsqueeze(-1)

    resid_h = pred_resid * harm_mask
    resid_n = pred_resid * noise_mask

    # Absolute HF targets for harmonic / noise bands.
    tgt_h_abs = tgt_high * harm_mask
    tgt_n_abs = tgt_high * noise_mask

    # 1) Improvement-style residual loss (compare refined HF vs baseline HF)
    if lam_l2h_resid > 0.0:
        eps = 1e-8
        margin = float(improve_margin)

        base_h = base_high_det * harm_mask
        base_n = base_high_det * noise_mask

        # Refined HF = baseline HF + predicted residual.
        ref_h = base_h + resid_h
        ref_n = base_n + resid_n

        # Improvement-style errors are measured against the
        # absolute HF target, not the residual target.
        err_base_h = (base_h - tgt_h_abs).abs()
        err_ref_h = (ref_h - tgt_h_abs).abs()
        err_base_n = (base_n - tgt_n_abs).abs()
        err_ref_n = (ref_n - tgt_n_abs).abs()

        gate_h_full = gate_h
        gate_n_full = gate_n

        mask_h_full = harm_mask * gate_h_full
        mask_n_full = noise_mask * gate_n_full

        improv_h = torch.relu(err_ref_h - (err_base_h - margin))
        improv_n = torch.relu(err_ref_n - (err_base_n - margin))

        l_h = (improv_h * mask_h_full).sum() / (mask_h_full.sum() + eps)
        l_n = (improv_n * mask_n_full).sum() / (mask_n_full.sum() + eps)
        l2h_resid_l1 = l_h + l_n

        loss_l2h = loss_l2h + lam_l2h_resid * l2h_resid_l1
        loss_dict_l2h["l2h_resid"] = float(l2h_resid_l1.item())
        loss_dict_l2h["l2h_resid_harm"] = float(l_h.item())
        loss_dict_l2h["l2h_resid_noise"] = float(l_n.item())

    # 2) Decor: only decorrelate noise residual from low-band baseline
    if lam_l2h_decor > 0.0:
        resid_noise = resid_n * gate.detach()

        # Low-band reference: baseline mel low frequencies
        mel_low = mel_base[:, :T_use, :lb].detach()

        def decorrelation_loss(low_btL: torch.Tensor, high_btH: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
            X = low_btL - low_btL.mean(dim=1, keepdim=True)
            Y = high_btH - high_btH.mean(dim=1, keepdim=True)

            cov = torch.einsum("btl,bth->blh", X, Y) / (X.size(1) - 1 + eps)
            stdX = (X.pow(2).mean(dim=1) + eps).sqrt().unsqueeze(-1)
            stdY = (Y.pow(2).mean(dim=1) + eps).sqrt().unsqueeze(-2)
            corr = cov / (stdX * stdY + eps)
            return (corr ** 2).mean()

        l2h_decor = decorrelation_loss(mel_low, resid_noise)
        loss_l2h = loss_l2h + lam_l2h_decor * l2h_decor
        loss_dict_l2h["l2h_decor"] = float(l2h_decor.item())

    return loss_l2h, loss_dict_l2h


def compute_losses_simplified(
    out: Dict[str, torch.Tensor],
    cfg: TrainingConfig,
    device: torch.device,
    model: DualBranchBarkJSCC,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Core loss bundle used in the simplified Stage2.5 branch."""

    loss_dict: Dict[str, float] = {}

    audio_real = out["audio"].to(device)
    audio_hat = out["audio_hat"].to(device)

    total = torch.zeros((), device=device, dtype=audio_hat.dtype)

    # Local helper: STFT magnitude for 1D waveform batch, with simple
    # frame-count correction to avoid off-by-one issues at the end.
    def _stft_mag_main(x: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
        x32 = x.to(torch.float32)
        win_t = torch.hann_window(win, device=x32.device, dtype=torch.float32)
        X = torch.stft(
            x32,
            n_fft=n_fft,
            hop_length=hop,
            win_length=win,
            window=win_t,
            center=False,
            return_complex=True,
        )
        mag = X.abs()  # [B,F,T]
        B_, F_, T_ = mag.shape
        expected_frames = x32.size(-1) // hop
        if T_ > expected_frames:
            mag = mag[:, :, :expected_frames]
        return mag

    # Local helper: frequency-weighted multi-resolution spectral
    # convergence loss. This mirrors multi_resolution_sc_loss but
    # allows us to down-weight high-frequency bins so that the main
    # STFT loss focuses more on low/mid bands while Bark/L2H handle
    # high-frequency texture.
    def _mr_sc_loss_weighted(
        x: torch.Tensor,
        y: torch.Tensor,
        fft_sizes: List[int],
        hop_sizes: List[int],
        win_lengths: List[int],
        hf_start_hz: float,
        hf_weight: float,
    ) -> torch.Tensor:
        # Fallback to the original unweighted loss when no
        # high-frequency de-emphasis is requested.
        if hf_start_hz <= 0.0 or hf_weight >= 1.0:
            return multi_resolution_sc_loss(
                x,
                y,
                device=device,
                fft_sizes=fft_sizes,
                hop_sizes=hop_sizes,
                win_lengths=win_lengths,
            )

        min_len = min(x.size(-1), y.size(-1))
        x = x[..., :min_len]
        y = y[..., :min_len]

        sr = 16000.0
        hf_w = max(0.0, min(float(hf_weight), 1.0))
        sc_total = torch.zeros((), device=device, dtype=x.dtype)
        n_scales = 0
        for fs, hs, wl in zip(fft_sizes, hop_sizes, win_lengths):
            win_t = torch.hann_window(wl, device=x.device, dtype=torch.float32)
            X = torch.stft(
                x.to(torch.float32),
                n_fft=fs,
                hop_length=hs,
                win_length=wl,
                window=win_t,
                center=False,
                return_complex=True,
            )
            Y = torch.stft(
                y.to(torch.float32),
                n_fft=fs,
                hop_length=hs,
                win_length=wl,
                window=win_t,
                center=False,
                return_complex=True,
            )
            X_mag = X.abs().clamp_min(1e-7)
            Y_mag = Y.abs().clamp_min(1e-7)

            # 原版谱收敛中先开方再计算；这里在频率维加权。
            X_s = torch.sqrt(X_mag)
            Y_s = torch.sqrt(Y_mag)

            Bm, Fbins, Tm = X_s.shape
            if Fbins <= 0 or Tm <= 0:
                continue

            freqs = torch.linspace(
                0.0,
                sr / 2.0,
                Fbins,
                device=X_s.device,
                dtype=X_s.dtype,
            )
            w = torch.ones_like(freqs)
            w = torch.where(freqs >= float(hf_start_hz), w * hf_w, w)
            w_f = w.view(1, Fbins, 1)

            diff = torch.abs((Y_s - X_s) * w_f)
            num = diff.sum()
            den = (torch.abs(Y_s * w_f)).sum() + 1e-12
            sc_scale = num / den

            sc_total = sc_total + sc_scale
            n_scales += 1

        if n_scales <= 0:
            return torch.zeros((), device=device, dtype=x.dtype)
        return sc_total / float(n_scales)

    # ---- 1) Multi-resolution STFT: spectral convergence (+ optional mag-L1) ----
    lam_wave = float(cfg.lambda_wave)
    lam_wave_mag = float(cfg.lambda_wave_mag)
    if lam_wave > 0.0 or lam_wave_mag > 0.0:
        if getattr(cfg, "stft_preset", "aether") == "fargan":
            fs = [2560, 1280, 640, 320, 160, 80]
            hs = [640, 320, 160, 80, 40, 20]
            wl = [2560, 1280, 640, 320, 160, 80]
        else:
            fs = [1024, 512, 256, 128]
            hs = [256, 128, 64, 32]
            wl = [1024, 512, 256, 128]

        # Frequency-weighted MR-STFT spectral convergence: emphasise
        # low/mid bands while optionally relaxing HF bins.
        sr = 16000.0
        hf_start_sc = float(getattr(cfg, "stft_hf_start_hz", 0.0))
        hf_weight_sc = float(getattr(cfg, "stft_hf_weight", 1.0))

        loss_stft = _mr_sc_loss_weighted(
            audio_hat,
            audio_real,
            fft_sizes=fs,
            hop_sizes=hs,
            win_lengths=wl,
            hf_start_hz=hf_start_sc,
            hf_weight=hf_weight_sc,
        )
        total = total + lam_wave * loss_stft
        loss_dict["stft"] = float(loss_stft.item())

        if lam_wave_mag > 0.0:
            # Band-weighted MR-STFT magnitude L1: emphasise low/mid
            # frequencies while relaxing HF bins so that GAN/HF losses
            # have room to shape high-frequency texture.
            sr = 16000.0
            hf_start = float(getattr(cfg, "stft_hf_start_hz", 4000.0))
            hf_weight = float(getattr(cfg, "stft_hf_weight", 0.5))
            hf_weight = max(0.0, min(hf_weight, 1.0))

            loss_stft_mag = torch.zeros((), device=device, dtype=audio_hat.dtype)
            n_scales = 0
            for n_fft, hop, win in zip(fs, hs, wl):
                Mag_h = _stft_mag_main(audio_hat, n_fft=n_fft, hop=hop, win=win)
                Mag_r = _stft_mag_main(audio_real, n_fft=n_fft, hop=hop, win=win)
                Tm_use = min(Mag_h.size(-1), Mag_r.size(-1))
                if Tm_use <= 0:
                    continue
                Mag_h = Mag_h[:, :, :Tm_use]
                Mag_r = Mag_r[:, :, :Tm_use]
                Bm, Fbins, Tm = Mag_h.shape
                if Fbins <= 0 or Tm <= 0:
                    continue
                freqs = torch.linspace(
                    0.0,
                    sr / 2.0,
                    Fbins,
                    device=Mag_h.device,
                    dtype=Mag_h.dtype,
                )
                w = torch.ones_like(freqs)
                if hf_start > 0.0 and hf_weight < 1.0:
                    w = torch.where(freqs >= hf_start, w * hf_weight, w)
                w_f = w.view(1, Fbins, 1)
                diff = torch.abs(Mag_h - Mag_r)
                loss_scale = (diff * w_f).mean()
                loss_stft_mag = loss_stft_mag + loss_scale
                n_scales += 1

            if n_scales > 0:
                loss_stft_mag = loss_stft_mag / float(n_scales)
                total = total + lam_wave_mag * loss_stft_mag
                loss_dict["stft_mag"] = float(loss_stft_mag.item())

    # ---- 1.b) Silence / voiced masks (ported from v3) ----
    def _frame_rms(x: torch.Tensor, frame_len: int = 160, hop: int = 160) -> torch.Tensor:
        B, L = x.shape
        if L < frame_len:
            pad = frame_len - L
            x = F.pad(x, (0, pad))
            L = frame_len
        x_frames = x.unfold(dimension=1, size=frame_len, step=hop)
        rms = torch.sqrt(x_frames.pow(2).mean(dim=-1) + 1e-8)
        return rms

    rms_real = _frame_rms(audio_real)
    eps = 1e-8
    rms_max = rms_real.max(dim=1, keepdim=True).values.clamp_min(eps)
    rms_norm = rms_real / rms_max
    rms_db = 20.0 * torch.log10(rms_norm + eps)

    # Optional waveform RMS envelope anchor: encourage the per-frame
    # loudness trajectory of audio_hat to match the reference. This
    # acts as a relatively hard constraint on coarse energy while
    # remaining agnostic to fine spectral detail.
    try:
        lam_rms = float(getattr(cfg, "lambda_wave_rms", 0.0))
    except Exception:
        lam_rms = 0.0
    if lam_rms > 0.0:
        try:
            rms_hat = _frame_rms(audio_hat)
            # Use log10 RMS to reduce scale sensitivity.
            log_r = torch.log10(rms_real + eps)
            log_h = torch.log10(rms_hat + eps)
            # Align time axis to the shorter of the two.
            T_env = min(log_r.size(1), log_h.size(1))
            if T_env > 0:
                log_r = log_r[:, :T_env]
                log_h = log_h[:, :T_env]
                loss_rms_env = torch.mean(torch.abs(log_h - log_r))
                total = total + lam_rms * loss_rms_env
                loss_dict["wave_rms"] = float(loss_rms_env.item())
        except Exception:
            pass

    silence_mask: Optional[torch.Tensor] = None
    voiced_mask: Optional[torch.Tensor] = None
    mel_for_mask = out.get("mel", None)
    if isinstance(mel_for_mask, torch.Tensor) and mel_for_mask.dim() == 3:
        try:
            Bm, Tm, Fm = mel_for_mask.shape
            hf_low = int(
                getattr(
                    cfg,
                    "silence_hf_low_bins",
                    int(getattr(cfg, "mel_hp_low_bins", 16)),
                )
            )
            hf_low = max(0, min(hf_low, Fm - 1))
            mel_hf = mel_for_mask[:, :, hf_low:] if hf_low < Fm else mel_for_mask
            mel_energy = mel_hf.mean(dim=-1)  # [B,Tm]

            Tr = rms_db.size(1)
            T_use = min(Tm, Tr)
            mel_energy_use = mel_energy[:, :T_use]
            rms_db_use = rms_db[:, :T_use]

            sil_thr_db_hf = float(
                getattr(
                    cfg,
                    "silence_energy_thr_db",
                    getattr(cfg, "silence_mel_thr_db", -35.0),
                )
            )
            sil_thr_log = sil_thr_db_hf / 10.0
            energy_sil = mel_energy_use <= sil_thr_log

            rms_thr_db = float(getattr(cfg, "silence_rms_thr_db", -35.0))
            rms_sil = rms_db_use <= rms_thr_db

            base_sil = energy_sil | rms_sil

            use_var = bool(getattr(cfg, "silence_use_hf_var", False))
            if use_var:
                hf_var = mel_hf.var(dim=-1)
                hf_var_use = hf_var[:, :T_use]
                var_thr = float(getattr(cfg, "silence_hf_var_thr", 0.02))
                texture_quiet = hf_var_use <= var_thr
                silence_mask = base_sil & texture_quiet
            else:
                silence_mask = base_sil

            voiced_mask = ~silence_mask
        except Exception:
            silence_mask = None
            voiced_mask = None

    try:
        dilate = int(getattr(cfg, "silence_dilate_frames", 0))
    except Exception:
        dilate = 0
    if silence_mask is not None and dilate > 0:
        try:
            ns = (~silence_mask).to(torch.float32).unsqueeze(1)
            ns = F.max_pool1d(ns, kernel_size=2 * dilate + 1, stride=1, padding=dilate)
            voiced_mask = ns.squeeze(1) > 0.5
            silence_mask = ~voiced_mask
        except Exception:
            pass

    lam_sil_mel = float(getattr(cfg, "lambda_silence_mel", 0.0))

    # ---- 2) Bark/BFCC reconstruction and historical "mel"-domain losses ----
    mel = out.get("mel")
    mel_hat = out.get("mel_hat")
    if isinstance(mel, torch.Tensor) and isinstance(mel_hat, torch.Tensor):
        mel = mel.to(device)
        if mel_hat.dim() == 4:
            mel_hat = mel_hat.squeeze(1)

        # Align time/freq and optional cropping (kept simple: no extra crop cfg)
        Tm = min(mel.size(1), mel_hat.size(1))
        Fm = min(mel.size(2), mel_hat.size(2))
        mel = mel[:, :Tm, :Fm]
        mel_hat = mel_hat[:, :Tm, :Fm]

        mel_hat_refined = out.get("mel_hat_refined", mel_hat)
        if isinstance(mel_hat_refined, torch.Tensor):
            if mel_hat_refined.dim() == 4:
                mel_hat_refined = mel_hat_refined.squeeze(1)
            mel_hat_refined = mel_hat_refined[:, :Tm, :Fm]
        else:
            mel_hat_refined = mel_hat

        mel_hat_for_loss = mel_hat_refined
        try:
            Bm = min(mel.size(0), mel_hat_for_loss.size(0))
            Tm2 = min(mel.size(1), mel_hat_for_loss.size(1))
            Fm2 = min(mel.size(2), mel_hat_for_loss.size(2))
            mel = mel[:Bm, :Tm2, :Fm2]
            mel_hat_for_loss = mel_hat_for_loss[:Bm, :Tm2, :Fm2]
        except Exception:
            pass

        def _soft_unit(x: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid((x + 8.0) / 2.0)

        mel_n = _soft_unit(mel)
        mel_hat_n = _soft_unit(mel_hat_for_loss)
        if mel_n.shape != mel_hat_n.shape:
            try:
                Bb = min(mel_n.size(0), mel_hat_n.size(0))
                Tb = min(mel_n.size(1), mel_hat_n.size(1))
                Fb = min(mel_n.size(2), mel_hat_n.size(2))
                mel_n = mel_n[:Bb, :Tb, :Fb]
                mel_hat_n = mel_hat_n[:Bb, :Tb, :Fb]
            except Exception:
                pass

        min_spatial = 16
        use_ms_ssim = (
            mel_n.dim() == 3
            and mel_n.size(1) >= min_spatial
            and mel_n.size(2) >= min_spatial
        )
        if use_ms_ssim:
            try:
                mel_n_img = mel_n.unsqueeze(1)
                mel_hat_n_img = mel_hat_n.unsqueeze(1)
                from utils.ssim import MS_SSIM

                ms = MS_SSIM(data_range=1.0, channel=1, levels=4)
                loss_mel_struct = ms(mel_hat_n_img, mel_n_img).mean()
            except Exception:
                if mel_hat_n.shape == mel_n.shape:
                    loss_mel_struct = F.l1_loss(mel_hat_n, mel_n)
                else:
                    a = mel_hat_n.reshape(-1)
                    b = mel_n.reshape(-1)
                    n = min(a.numel(), b.numel())
                    loss_mel_struct = F.l1_loss(a[:n], b[:n])
        else:
            if mel_n.shape == mel_hat_n.shape:
                loss_mel_struct = F.l1_loss(mel_hat_n, mel_n)
            else:
                a = mel_hat_n.reshape(-1)
                b = mel_n.reshape(-1)
                n = min(a.numel(), b.numel())
                loss_mel_struct = F.l1_loss(a[:n], b[:n])

        lam_mel = float(getattr(cfg, "lambda_mel", 0.0))
        if lam_mel > 0.0:
            total = total + lam_mel * loss_mel_struct
            loss_dict["mel_ms_ssim"] = float(loss_mel_struct.item())

        lam_mel_l1 = float(getattr(cfg, "lambda_mel_l1", 0.0))
        if lam_mel_l1 > 0.0:
            mel_l1_ref = mel
            mel_l1_hat = mel_hat_for_loss
            if mel_l1_ref.shape != mel_l1_hat.shape:
                try:
                    Bx = min(mel_l1_ref.size(0), mel_l1_hat.size(0))
                    Tx = min(mel_l1_ref.size(1), mel_l1_hat.size(1))
                    Fx = min(mel_l1_ref.size(2), mel_l1_hat.size(2))
                    mel_l1_ref = mel_l1_ref[:Bx, :Tx, :Fx]
                    mel_l1_hat = mel_l1_hat[:Bx, :Tx, :Fx]
                except Exception:
                    pass
            if mel_l1_ref.shape == mel_l1_hat.shape:
                l_mel_l1 = torch.mean(torch.abs(mel_l1_hat - mel_l1_ref))
            else:
                a = mel_l1_hat.reshape(-1)
                b = mel_l1_ref.reshape(-1)
                n = min(a.numel(), b.numel())
                l_mel_l1 = F.l1_loss(a[:n], b[:n])

            total = total + lam_mel_l1 * l_mel_l1
            loss_dict["mel_l1"] = float(l_mel_l1.item())

        # Global mel energy / brightness anchor (v3-style lambda_mel_energy)
        try:
            lam_energy = float(getattr(cfg, "lambda_mel_energy", 0.0))
            if lam_energy > 0.0:
                mel_mean_hat = mel_hat_for_loss.mean(dim=(1, 2))  # [B]
                mel_mean = mel.mean(dim=(1, 2))                    # [B]
                loss_mel_energy = torch.mean(torch.abs(mel_mean_hat - mel_mean))
                total = total + lam_energy * loss_mel_energy
                loss_dict["mel_energy"] = float(loss_mel_energy.item())
        except Exception:
            pass

        # Stats-bits reconstruction loss: encourage mel_mean_hat/mel_std_hat
        # (decoded from hash_content_stats) to match GT mel_mean/mel_std.
        # 仅在 with_hash 路径上有意义；若 out 中不存在 mel_mean_hat/mel_std_hat，
        # 或者 lambda_mel_stats 为 0，则跳过。
        try:
            lam_stats = float(getattr(cfg, "lambda_mel_stats", 0.0))
            if lam_stats > 0.0:
                mm_hat_any = out.get("mel_mean_hat", None)
                ms_hat_any = out.get("mel_std_hat", None)
                if isinstance(mm_hat_any, torch.Tensor) and isinstance(ms_hat_any, torch.Tensor):
                    mm_hat = mm_hat_any.to(device).view(mm_hat_any.size(0), -1).mean(dim=1)  # [B]
                    ms_hat = ms_hat_any.to(device).view(ms_hat_any.size(0), -1).mean(dim=1)  # [B]

                    mm_ref = mel.mean(dim=(1, 2))                         # [B]
                    ms_ref = mel.std(dim=(1, 2)).clamp_min(0.1)            # [B]

                    loss_stats_mean = torch.mean(torch.abs(mm_hat - mm_ref))
                    loss_stats_std = torch.mean(torch.abs(ms_hat - ms_ref))
                    loss_stats = loss_stats_mean + loss_stats_std

                    total = total + lam_stats * loss_stats
                    loss_dict["mel_stats"] = float(loss_stats.item())

                    # 可选：若模型 forward 暴露了 latent 级 stats L1（在
                    # 归一化 (mean_norm, std_norm) 空间下直接约束
                    # hb_stats["reconstructed"] 与 GT stats），则将其
                    # 作为同一个 lambda_mel_stats 项的一部分，一并加入
                    # 总 loss 中，以便更强地驱动 hash_content_stats。
                    stats_latent_any = out.get("stats_latent_l1", None)
                    if isinstance(stats_latent_any, torch.Tensor):
                        total = total + lam_stats * stats_latent_any
                        loss_dict["stats_latent"] = float(stats_latent_any.item())
        except Exception:
            pass

        # Silence-frame HF mel suppression (v3-style)
        if lam_sil_mel > 0.0 and silence_mask is not None:
            try:
                Bm, Tm_s, Fm_s = mel_hat_for_loss.shape
                hf_low = int(
                    getattr(
                        cfg,
                        "silence_hf_low_bins",
                        int(getattr(cfg, "mel_hp_low_bins", 16)),
                    )
                )
                hf_low = max(0, min(hf_low, Fm_s - 1))

                mel_hat_hf = (
                    mel_hat_for_loss[:, :, hf_low:]
                    if hf_low < Fm_s
                    else mel_hat_for_loss
                )
                mel_ref_hf = mel[:, :, hf_low:] if hf_low < Fm_s else mel

                Ts = silence_mask.size(1)
                T_use_s = min(Tm_s, Ts)
                if T_use_s > 0:
                    sm = silence_mask[:, :T_use_s].to(mel_hat_hf.dtype)
                    mel_hat_hf = mel_hat_hf[:, :T_use_s, :]
                    mel_ref_hf = mel_ref_hf[:, :T_use_s, :]

                    diff = torch.abs(mel_hat_hf - mel_ref_hf) * sm.unsqueeze(-1)
                    denom = sm.sum() * max(mel_hat_hf.size(-1), 1) + 1e-6
                    loss_sil_mel = diff.sum() / denom
                    total = total + lam_sil_mel * loss_sil_mel
                    loss_dict["sil_mel_hf"] = float(loss_sil_mel.item())
            except Exception:
                pass

        # High-frequency mel patch structural loss (mel_hp, v3-style)
        lam_mel_hp = float(getattr(cfg, "lambda_mel_hp", 0.0))
        try:
            if lam_mel_hp > 0.0:
                Bm_hp, Tm_hp, Fm_hp = mel_hat_for_loss.shape
                hf_start = int(getattr(cfg, "mel_hp_low_bins", 16))
                hf_start = max(4, min(hf_start, Fm_hp - 4))
                F_hf = Fm_hp - hf_start
                if F_hf >= 4 and Tm_hp >= 4:
                    mel_h = mel[:, :Tm_hp, hf_start:]
                    mel_hat_h = mel_hat_for_loss[:, :Tm_hp, hf_start:]

                    if silence_mask is not None:
                        Tf = silence_mask.size(1)
                        T_use_hp = min(Tm_hp, Tf)
                        ns_mask = (~silence_mask[:, :T_use_hp]).to(mel_h.dtype)
                        mel_h = mel_h[:, :T_use_hp, :] * ns_mask.unsqueeze(-1)
                        mel_hat_h = mel_hat_h[:, :T_use_hp, :] * ns_mask.unsqueeze(-1)

                    mel_h_n = _soft_unit(mel_h).unsqueeze(1)
                    mel_hat_h_n = _soft_unit(mel_hat_h).unsqueeze(1)

                    from utils.ssim import MS_SSIM as _MS_SSIM_hp

                    ms_hp = _MS_SSIM_hp(data_range=1.0, channel=1, levels=4)
                    loss_hf_ssim = ms_hp(mel_hat_h_n, mel_h_n).mean()
                    total = total + lam_mel_hp * loss_hf_ssim
                    loss_dict["mel_hp"] = float(loss_hf_ssim.item())
        except Exception:
            pass

        # High-frequency mel energy L1 (direct amplitude matching in HF band).
        lam_hf_l1 = float(getattr(cfg, "lambda_mel_hf_l1", 0.0))
        try:
            if lam_hf_l1 > 0.0:
                Bm_hf, Tm_hf, Fm_hf = mel_hat_for_loss.shape
                hf_start_l1 = int(getattr(cfg, "mel_hp_low_bins", 16))
                hf_start_l1 = max(4, min(hf_start_l1, Fm_hf - 4))
                F_hf_l1 = Fm_hf - hf_start_l1
                if F_hf_l1 >= 4 and Tm_hf >= 1:
                    mel_h_l1 = mel[:, :Tm_hf, hf_start_l1:]
                    mel_hat_h_l1 = mel_hat_for_loss[:, :Tm_hf, hf_start_l1:]

                    # 仅在非静音帧上对齐高频能量，避免推动静音段噪声。
                    if silence_mask is not None:
                        Tf_l1 = silence_mask.size(1)
                        T_use_l1 = min(Tm_hf, Tf_l1)
                        ns_mask_l1 = (~silence_mask[:, :T_use_l1]).to(mel_h_l1.dtype)
                        mel_h_l1 = mel_h_l1[:, :T_use_l1, :] * ns_mask_l1.unsqueeze(-1)
                        mel_hat_h_l1 = mel_hat_h_l1[:, :T_use_l1, :] * ns_mask_l1.unsqueeze(-1)

                    hf_l1 = torch.mean(torch.abs(mel_hat_h_l1 - mel_h_l1))
                    total = total + lam_hf_l1 * hf_l1
                    loss_dict["mel_hf_l1"] = float(hf_l1.item())
        except Exception:
            pass

    # ---- 3) Cepstrum L1 reconstruction ----
    ceps_t = out.get("ceps")
    ceps_h = out.get("ceps_hat")
    if isinstance(ceps_t, torch.Tensor) and isinstance(ceps_h, torch.Tensor):
        ceps_t = ceps_t.to(device)
        ceps_h = ceps_h.to(device)
        Tm = min(ceps_t.size(1), ceps_h.size(1))
        Dm = min(ceps_t.size(2), ceps_h.size(2))
        ceps_t = ceps_t[:, :Tm, :Dm]
        ceps_h = ceps_h[:, :Tm, :Dm]
        loss_ceps = F.l1_loss(ceps_h, ceps_t)
        total = total + float(cfg.lambda_ceps) * loss_ceps
        loss_dict["ceps"] = float(loss_ceps.item())

        # 3.b) High-order cepstrum supervision (ceps_hi)
        lam_ceps_hi = float(getattr(cfg, "lambda_ceps_hi", 0.0))
        if lam_ceps_hi > 0.0 and Dm > 1:
            s0 = int(getattr(cfg, "ceps_hi_start", 10))
            s0 = max(1, min(s0, Dm - 1))
            try:
                l_chi = F.l1_loss(ceps_h[:, :Tm, s0:Dm], ceps_t[:, :Tm, s0:Dm])
                total = total + lam_ceps_hi * l_chi
                loss_dict["ceps_hi"] = float(l_chi.item())
            except Exception:
                pass

    # Helper: map dnn_pitch logits to F0 Hz (matches v3 mapping)
    def _dp_to_f0(dp: torch.Tensor) -> torch.Tensor:
        period = 256.0 / torch.pow(2.0, dp + 1.5)
        period = period.clamp(32.0, 255.0)
        f0 = 16000.0 / period
        return f0.squeeze(-1)

    # ---- 4) F0 MSE on voiced frames (GT mask) ----
    # 监督 decoder 输出 + 标定后的 F0（dnn_pitch_calib），
    # 使 JSCC F0 分支与原始 FARGAN dnn_pitch 在同一刻度。若
    # 标定键缺失，则回退到 raw / hat 以兼容旧 checkpoint。
    dp = out.get("dnn_pitch_calib")
    if not isinstance(dp, torch.Tensor):
        dp = out.get("dnn_pitch_raw")
    if not isinstance(dp, torch.Tensor):
        dp = out.get("dnn_pitch_hat")
    dp_ref = out.get("dnn_pitch")
    fc_ref = out.get("frame_corr")
    if isinstance(dp, torch.Tensor) and isinstance(dp_ref, torch.Tensor):
        dp = dp.to(device)
        dp_ref = dp_ref.to(device)
        if isinstance(fc_ref, torch.Tensor):
            fc_ref = fc_ref.to(device)
            th = float(cfg.vuv_threshold)
            mask = (fc_ref > th).to(dp.dtype)  # [B,T,1]
            err2 = (dp - dp_ref) ** 2 * mask
            denom = mask.sum() + 1e-6
            loss_f0 = err2.sum() / denom
        else:
            loss_f0 = F.mse_loss(dp, dp_ref)
        total = total + float(cfg.lambda_f0) * loss_f0
        loss_dict["f0"] = float(loss_f0.item())

    # 4.b) F0 smoothness loss (second-order difference on raw pitch)
    lam_f0_smooth = float(getattr(cfg, "lambda_f0_smooth", 0.0))
    dp_for_smooth = out.get("dnn_pitch_calib")
    if not isinstance(dp_for_smooth, torch.Tensor):
        dp_for_smooth = out.get("dnn_pitch_raw")
    if not isinstance(dp_for_smooth, torch.Tensor):
        dp_for_smooth = out.get("dnn_pitch_hat")
    if lam_f0_smooth > 0.0 and isinstance(dp_for_smooth, torch.Tensor) and isinstance(fc_ref, torch.Tensor):
        try:
            dp_s = dp_for_smooth.to(device)
            fc_ref_s = fc_ref.to(device)
            if dp_s.size(1) > 2:
                th = float(cfg.vuv_threshold)
                d2_f0 = dp_s[:, 2:, :] - 2.0 * dp_s[:, 1:-1, :] + dp_s[:, :-2, :]
                mask_smooth = (fc_ref_s[:, 1:-1, :] > th).to(dp_s.dtype)
                err_smooth = torch.abs(d2_f0)
                denom_smooth = mask_smooth.sum() + 1e-6
                loss_f0_smooth = (err_smooth * mask_smooth).sum() / denom_smooth
                total = total + lam_f0_smooth * loss_f0_smooth
                loss_dict["f0_smooth"] = float(loss_f0_smooth.item())
        except Exception:
            pass

    # ---- 5) VUV BCE-with-logits on logits (not frame_corr) ----
    vuv_logits = out.get("vuv_logits")
    if isinstance(vuv_logits, torch.Tensor) and isinstance(fc_ref, torch.Tensor):
        vuv_logits = vuv_logits.to(device)
        fc_ref = fc_ref.to(device)
        th = float(cfg.vuv_threshold)
        vuv_label = (fc_ref > th).to(vuv_logits.dtype)  # [B,T,1]
        loss_vuv = F.binary_cross_entropy_with_logits(vuv_logits, vuv_label)
        total = total + float(cfg.lambda_vuv) * loss_vuv
        loss_dict["vuv_bce"] = float(loss_vuv.item())

    # 5.b) Optional VUV prob BCE using frame_corr_hat (v3-style)
    lam_vuv_bce = float(getattr(cfg, "lambda_vuv_bce", 0.0))
    if lam_vuv_bce > 0.0:
        try:
            fc_hat = out.get("frame_corr_hat", None)
            fc_ref2 = out.get("frame_corr", None)
            if isinstance(fc_hat, torch.Tensor) and isinstance(fc_ref2, torch.Tensor):
                fc_hat = fc_hat.to(device)
                fc_ref2 = fc_ref2.to(device)
                th = float(cfg.vuv_threshold)
                v_tgt = (fc_ref2 > th).float().squeeze(-1)
                k = 10.0
                logits = (fc_hat.squeeze(-1) - th) * k
                v_prob = torch.sigmoid(logits)
                bce = F.binary_cross_entropy(v_prob, v_tgt)
                total = total + lam_vuv_bce * bce
                loss_dict["vuv_ce"] = float(bce.item())
        except Exception:
            pass

    # ---- 6) RVQ VQ losses (content + F0 + stats) ----
    vq_loss_c = out.get("vq_loss_content")
    vq_loss_f = out.get("vq_loss_f0")
    vq_loss_stats = out.get("vq_loss_stats")
    if isinstance(vq_loss_c, torch.Tensor):
        loss_vq_c = vq_loss_c.to(device)
        total = total + float(cfg.lambda_vq_c) * loss_vq_c
        loss_dict["vq_c"] = float(loss_vq_c.item())
    if isinstance(vq_loss_f, torch.Tensor):
        loss_vq_f = vq_loss_f.to(device)
        total = total + float(cfg.lambda_vq_f) * loss_vq_f
        loss_dict["vq_f"] = float(loss_vq_f.item())
    if isinstance(vq_loss_stats, torch.Tensor):
        loss_vq_s = vq_loss_stats.to(device)
        total = total + float(getattr(cfg, "lambda_vq_stats", 0.0)) * loss_vq_s
        loss_dict["vq_stats"] = float(loss_vq_s.item())

    # ---- 7) CREPE-guided F0 envelope loss (copied from v3) ----
    lam_f0_env = float(getattr(cfg, "lambda_f0_env", 0.0))

    # Helper functions for F0 envelope (adapted from the training support module)
    def _extract_f0_batch(y: torch.Tensor, sr: int = 16000, hop: int = 160,
                          estimator: str = "auto", model: str = "tiny"):
        """Return (f0_hz, periodicity, f0_fb, fb_mask) for y: [B,L].

        Uses torchcrepe if available; falls back to librosa.pyin + yin.
        """

        B, L = y.shape
        f0_list, p_list, fb_list, fbmask_list = [], [], [], []
        try:
            import librosa  # type: ignore
        except Exception:  # pragma: no cover - optional dependency
            librosa = None  # type: ignore
        try:
            import torchcrepe  # type: ignore
            has_crepe = True
        except Exception:  # pragma: no cover - optional dependency
            torchcrepe = None  # type: ignore
            has_crepe = False

        use_crepe = (estimator in ("auto", "crepe")) and has_crepe

        for b in range(B):
            wav = y[b].detach()
            if use_crepe:
                try:
                    dev = wav.device if wav.is_cuda else (
                        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                    )
                    x = wav.unsqueeze(0).to(dev)
                    with torch.no_grad():
                        f0_t, p_t = torchcrepe.predict(
                            x,
                            sr,
                            hop,
                            50.0,
                            500.0,
                            model,
                            batch_size=512,
                            device=dev,
                            return_periodicity=True,
                        )
                        # median/mean smoothing (window=3~5)
                        try:
                            f0_t = torchcrepe.filter.median(f0_t, 3)
                            p_t = torchcrepe.filter.median(p_t, 3)
                            f0_t = torchcrepe.filter.mean(f0_t, 3)
                            p_t = torchcrepe.filter.mean(p_t, 3)
                        except Exception:
                            pass
                    f0 = f0_t.squeeze(0).float().cpu().numpy()
                    p = p_t.squeeze(0).float().cpu().numpy()
                except Exception:
                    use_crepe = False  # fall through to pyin
            if not use_crepe:
                # pYIN fallback only
                if librosa is not None:
                    try:
                        f0, vflag, _ = librosa.pyin(
                            wav.detach().cpu().numpy(),
                            fmin=50,
                            fmax=500,
                            sr=sr,
                            hop_length=hop,
                            frame_length=hop * 4,
                            fill_na=0.0,
                        )
                        f0 = np.where((f0 > 0) & (f0 >= 50) & (f0 <= 500), f0, 0.0)
                        p = vflag.astype(np.float32)
                    except Exception:
                        T = int(np.ceil(L / hop))
                        f0 = np.zeros((T,), dtype=np.float32)
                        p = np.zeros((T,), dtype=np.float32)
                else:
                    T = int(np.ceil(L / hop))
                    f0 = np.zeros((T,), dtype=np.float32)
                    p = np.zeros((T,), dtype=np.float32)

            # Fallback (YIN) based on librosa if available
            if librosa is not None:
                try:
                    fb = librosa.yin(
                        wav.detach().cpu().numpy(),
                        fmin=50,
                        fmax=500,
                        sr=sr,
                        frame_length=hop * 4,
                        hop_length=hop,
                    )
                    fb = np.where((fb > 0) & (fb >= 50) & (fb <= 500), fb, 0.0).astype(np.float32)
                except Exception:
                    fb = np.zeros_like(f0, dtype=np.float32)
            else:
                fb = np.zeros_like(f0, dtype=np.float32)

            fb_mask = fb > 0
            f0_list.append(torch.from_numpy(f0))
            p_list.append(torch.from_numpy(p))
            fb_list.append(torch.from_numpy(fb))
            fbmask_list.append(torch.from_numpy(fb_mask.astype(np.float32)))

        # pad to same length
        T = min(min(t.size(0) for t in f0_list), y.size(-1) // hop)
        f0 = torch.stack([t[:T] for t in f0_list], 0)
        p = torch.stack([t[:T] for t in p_list], 0)
        fb = torch.stack([t[:T] for t in fb_list], 0)
        fb_mask = torch.stack([t[:T] for t in fbmask_list], 0)
        return (
            f0.to(y.device),
            p.to(y.device),
            fb.to(y.device),
            fb_mask.to(y.device),
        )

    def _to_cents(f_hz: torch.Tensor) -> torch.Tensor:
        return 1200.0 * torch.log2(torch.clamp(f_hz, min=1e-3) / 55.0)

    def _smooth_median(x: torch.Tensor, k: int) -> torch.Tensor:
        # quick median smoothing using unfold (odd k)
        if k <= 1 or x.size(1) < k:
            return x
        pad = k // 2
        xp = F.pad(x.unsqueeze(1), (pad, pad), mode="replicate").squeeze(1)  # [B,T+2p]
        xs = xp.unfold(1, k, 1)  # [B, T, k]
        return xs.median(dim=-1).values

    def _erode_mask(m: torch.Tensor, k: int = 3) -> torch.Tensor:
        """1D morphological erosion over time axis for [B,T] boolean mask."""
        if k <= 1:
            return m
        m_f = m.to(torch.float32)
        pad = k // 2
        mp = F.pad(m_f.unsqueeze(1), (pad, pad), mode="constant", value=0.0)  # [B,1,T+2p]
        w = torch.ones(1, 1, k, device=m.device, dtype=m_f.dtype)
        conv = F.conv1d(mp, w, stride=1)
        core = conv.squeeze(1) >= float(k) - 1e-6
        if core.shape != m.shape:
            core = core[:, : m.size(1)]
        return core & m

    # Only compute envelope when explicitly enabled
    if lam_f0_env > 0.0:
        B, L = audio_hat.shape

        # Extract F0 from target and prediction (no_grad)
        with torch.no_grad():
            f0_t, p_t, f0_fb_t, fb_mask_t = _extract_f0_batch(
                audio_real,
                sr=16000,
                hop=160,
                estimator=getattr(cfg, "f0_estimator", "auto"),
                model=getattr(cfg, "f0_estimator_model", "tiny"),
            )
            f0_h, p_h, f0_fb_h, fb_mask_h = _extract_f0_batch(
                audio_hat,
                sr=16000,
                hop=160,
                estimator=getattr(cfg, "f0_estimator", "auto"),
                model=getattr(cfg, "f0_estimator_model", "tiny"),
            )

        # ----- Simplified envelope/center (same as v3) -----
        thr = 0.35
        # Align reference sequence lengths, including f0_h
        T_ref = min(f0_t.size(1), f0_fb_t.size(1), f0_h.size(1))
        f0_t = f0_t[:, :T_ref]
        f0_h = f0_h[:, :T_ref]
        p_t = p_t[:, :T_ref]
        p_h = p_h[:, :T_ref]

        core_full = _erode_mask((p_t > thr) & (f0_t > 60.0) & (f0_t < 450.0), k=2)
        w_env_win = int(getattr(cfg, "f0_env_window", 5))
        k_env = max(3, w_env_win if (w_env_win % 2 == 1) else (w_env_win + 1))
        f0_smooth = _smooth_median(f0_t, k_env)
        cent_ref = _to_cents(torch.clamp(f0_smooth, min=1e-3))
        base_m = float(getattr(cfg, "f0_env_margin_cents", 80.0))
        margin_full = cent_ref.new_full(cent_ref.shape, base_m)

        # Predicted f0 from dnn_pitch_hat / dnn_pitch (aligned to Tm)
        dp_any = out.get("dnn_pitch_hat", None)
        if not isinstance(dp_any, torch.Tensor):
            dp_any = out.get("dnn_pitch", None)
        if isinstance(dp_any, torch.Tensor):
            Tm = min(cent_ref.size(1), core_full.size(1), p_t.size(1), dp_any.size(1), L // 160)
            cent_env = cent_ref[:, :Tm]
            core_t = core_full[:, :Tm]
            p_conf = torch.clamp(p_t[:, :Tm], 0.0, 1.0)
            margin_t = margin_full[:, :Tm]
            dp = dp_any[:, :Tm, :]

            # Predicted F0 → cents
            f0_pred = (16000.0 / torch.clamp(256.0 / torch.pow(2.0, dp + 1.5), 32.0, 255.0)).squeeze(-1)
            if f0_pred.size(1) != Tm:
                f0_pred = f0_pred[:, :Tm]
            cent_pred = _to_cents(f0_pred)
            if cent_pred.size(1) != Tm:
                cent_pred = cent_pred[:, :Tm]

            # Soft envelope gate: CREPE core + predicted V/UV
            m_env = core_t.to(f0_pred.dtype)
            fc_hat_any = out.get("frame_corr_hat", None)
            if isinstance(fc_hat_any, torch.Tensor):
                fc_hat_t = fc_hat_any[:, :Tm, 0]
                tau = float(getattr(cfg, "vuv_ce_tau", 0.15))
                th_vuv = float(getattr(cfg, "vuv_threshold", 0.3))
                q_hat = torch.sigmoid((fc_hat_t - th_vuv) / max(tau, 1e-6))
                alpha = float(getattr(cfg, "vuv_ce_alpha", 0.7))
                m_env = alpha * m_env + (1.0 - alpha) * q_hat.to(m_env.dtype)

            w_env = torch.sqrt(p_conf) * m_env
            err_c = torch.abs(cent_pred - cent_env)
            margin_t_f = margin_t.to(err_c.dtype)
            norm_h = torch.relu(err_c - margin_t_f) / (margin_t_f + 1e-6)

            active = w_env > 0
            if active.any():
                q = torch.quantile(norm_h[active], 0.90)
                norm_h = torch.minimum(norm_h, q)

            denom_env = w_env.sum() + 1e-6
            l_env = (norm_h * w_env).sum() / denom_env

            total = total + lam_f0_env * l_env
            loss_dict["f0_env"] = float(l_env.item())

    # ---- 8) Full L2H residual + decorrelation losses (v3-style) ----
    loss_l2h, loss_dict_l2h = _compute_l2h_resid_and_decor_losses(out, cfg, device, model)
    if loss_l2h.requires_grad or float(loss_l2h.item()) != 0.0:
        total = total + loss_l2h
        loss_dict.update(loss_dict_l2h)

    # ---- 9) High-frequency STFT loss (audio-domain, v3-style) ----
    lam_hf = float(getattr(cfg, "lambda_hf_stft", 0.0))
    if lam_hf > 0.0 and isinstance(out.get("audio_hat"), torch.Tensor) and isinstance(out.get("audio"), torch.Tensor):
        try:
            def _stft_mag(x: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
                x32 = x.to(torch.float32)
                win_t = torch.hann_window(win, device=x32.device, dtype=torch.float32)
                X = torch.stft(x32, n_fft=n_fft, hop_length=hop, win_length=win,
                               window=win_t, center=False, return_complex=True)
                mag = X.abs()
                B_, F_, T_ = mag.shape
                expected_frames = x32.size(-1) // hop
                if T_ > expected_frames:
                    mag = mag[:, :, :expected_frames]
                return mag

            y_hat_hf = out["audio_hat"].to(device)
            y_ref_hf = out["audio"].to(device)
            n_fft = 1024
            hop = 160
            Mag_h = _stft_mag(y_hat_hf, n_fft=n_fft, hop=hop, win=n_fft)
            Mag_r = _stft_mag(y_ref_hf, n_fft=n_fft, hop=hop, win=n_fft)
            Bm, Fbins, Tm = Mag_h.shape
            sr = 16000.0
            freqs = torch.linspace(0, sr / 2.0, Fbins, device=Mag_h.device, dtype=Mag_h.dtype)
            f0_hf = float(getattr(cfg, "hf_start_hz", 4000.0))
            p = float(getattr(cfg, "hf_power", 2.0))
            w = torch.clamp(freqs / max(f0_hf, 1.0), min=0.0, max=10.0).pow(p)
            w = w * (freqs >= f0_hf).to(w.dtype)
            wn = w / (w.sum() + 1e-6)

            diff = torch.abs(Mag_h - Mag_r)
            w_f = wn.view(1, Fbins, 1)
            w_t = torch.ones(Bm, 1, Tm, device=diff.device, dtype=diff.dtype)
            wt = w_f * w_t
            denom_hf = wt.sum() + 1e-6
            l_hf = (diff * wt).sum() / denom_hf
            total = total + lam_hf * l_hf
            loss_dict["hf_stft"] = float(l_hf.item())
        except Exception:
            pass

    # ---- 9.b) Residual texture protection loss (audio-domain, simplified) ----
    lam_texture = float(getattr(cfg, "lambda_texture_protect", 0.0))
    if lam_texture > 0.0:
        try:
            y_hat = out.get("audio_hat")
            y_ref = out.get("audio")
            fc_ref = out.get("frame_corr")
            dp_ref = out.get("dnn_pitch")
            if not (
                isinstance(y_hat, torch.Tensor)
                and isinstance(y_ref, torch.Tensor)
                and isinstance(fc_ref, torch.Tensor)
                and isinstance(dp_ref, torch.Tensor)
            ):
                raise RuntimeError("texture_protect requires audio_hat/audio/frame_corr/dnn_pitch")

            y_hat = y_hat.to(device)
            y_ref = y_ref.to(device)
            fc_ref = fc_ref.to(device)
            dp_ref = dp_ref.to(device)

            # Shared STFT mag helper
            def _stft_mag_tex(x: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
                x32 = x.to(torch.float32)
                win_t = torch.hann_window(win, device=x32.device, dtype=torch.float32)
                X = torch.stft(
                    x32,
                    n_fft=n_fft,
                    hop_length=hop,
                    win_length=win,
                    window=win_t,
                    center=False,
                    return_complex=True,
                )
                mag = X.abs()
                B_, F_, T_ = mag.shape
                expected_frames = x32.size(-1) // hop
                if T_ > expected_frames:
                    mag = mag[:, :, :expected_frames]
                return mag

            n_fft = 1024
            hop = 160
            sr = 16000
            Mag_g = _stft_mag_tex(y_hat, n_fft=n_fft, hop=hop, win=n_fft)
            Mag_r = _stft_mag_tex(y_ref, n_fft=n_fft, hop=hop, win=n_fft)
            Bm, Fbins, Ts = Mag_g.shape

            # Build mel filterbank (HTK-like, simplified)
            def _hz_to_mel(f: torch.Tensor) -> torch.Tensor:
                return 2595.0 * torch.log10(1.0 + f / 700.0)

            def _mel_to_hz(m: torch.Tensor) -> torch.Tensor:
                return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

            def _mel_fb(n_freqs: int, sr_: int, n_mels: int) -> torch.Tensor:
                f_min = 0.0
                f_max = sr_ / 2.0
                m_min = _hz_to_mel(torch.tensor(f_min, device=Mag_g.device, dtype=Mag_g.dtype))
                m_max = _hz_to_mel(torch.tensor(f_max, device=Mag_g.device, dtype=Mag_g.dtype))
                m_pts = torch.linspace(m_min, m_max, n_mels + 2, device=Mag_g.device, dtype=Mag_g.dtype)
                f_pts = _mel_to_hz(m_pts)
                freqs = torch.linspace(0.0, f_max, n_freqs, device=Mag_g.device, dtype=Mag_g.dtype)
                fb = torch.zeros(n_freqs, n_mels, device=Mag_g.device, dtype=Mag_g.dtype)
                for i in range(n_mels):
                    f_l, f_c, f_r = f_pts[i], f_pts[i + 1], f_pts[i + 2]
                    left = (freqs >= f_l) & (freqs <= f_c)
                    right = (freqs >= f_c) & (freqs <= f_r)
                    fb[left, i] = (freqs[left] - f_l) / (f_c - f_l + 1e-9)
                    fb[right, i] = (f_r - freqs[right]) / (f_r - f_c + 1e-9)
                fb = fb / (fb.sum(dim=0, keepdim=True) + 1e-9)
                return fb

            n_mels = 80
            fb = _mel_fb(Fbins, sr, n_mels)  # [F,M]

            # Voiced/unvoiced masks
            T = min(Mag_g.size(2), fc_ref.size(1), dp_ref.size(1))
            Mag_g = Mag_g[:, :, :T]
            Mag_r = Mag_r[:, :, :T]
            thr = float(getattr(cfg, "vuv_threshold", 0.3))
            voiced = (fc_ref[:, :T, :].squeeze(-1) > thr)
            unvoiced = (~voiced).to(Mag_g.dtype)  # [B,T]

            # Harmonic mask H [B,F,T]
            Ffreqs = torch.linspace(0.0, sr / 2.0, Fbins, device=Mag_g.device, dtype=Mag_g.dtype)
            K = int(getattr(cfg, "harmonics_max", 5))
            ks = torch.arange(1, K + 1, device=Mag_g.device, dtype=Mag_g.dtype).view(1, 1, K)
            period = torch.clamp(256.0 / torch.pow(2.0, dp_ref[:, :T, :].squeeze(-1) + 1.5), 32.0, 255.0)
            f0_hz = 16000.0 / period
            centers = f0_hz.unsqueeze(-1) * ks
            bw = float(getattr(cfg, "harmonic_bandwidth_hz", 30.0))
            diff_f = Ffreqs.view(1, 1, 1, Fbins) - centers.unsqueeze(-1)
            Hk = torch.exp(-0.5 * (diff_f / bw) ** 2)
            Hk = Hk / (Hk.sum(dim=-1, keepdim=True) + 1e-6)
            H = Hk.sum(dim=2)
            H = H * voiced.to(H.dtype).unsqueeze(-1)
            H = H.permute(0, 2, 1).contiguous()  # [B,F,T]

            Mag_g_res = Mag_g * (1.0 - H)
            Mag_r_res = Mag_r * (1.0 - H)

            eps_tex = float(getattr(cfg, "texture_eps", 1e-4))
            mel_res_g = torch.log10(torch.matmul((Mag_g_res ** 2).transpose(1, 2), fb) + eps_tex)
            mel_res_r = torch.log10(torch.matmul((Mag_r_res ** 2).transpose(1, 2), fb) + eps_tex)

            hf_start = int(getattr(cfg, "texture_hf_start", 40))
            Xg = mel_res_g[:, :, hf_start:]
            Xr = mel_res_r[:, :, hf_start:]
            if Xg.numel() > 0:
                Bq = Xr.size(0)
                tau = torch.quantile(Xr.detach().reshape(Bq, -1), 0.75, dim=1, keepdim=True).view(Bq, 1, 1)
                occupy_ref = (Xr > tau).to(Xg.dtype)
                logit_pred = (Xg - tau) / 0.5
                prob_pred = torch.sigmoid(logit_pred)
                L_occ_all = F.binary_cross_entropy(prob_pred, occupy_ref, reduction="none")
                L_occ_t = L_occ_all.mean(dim=-1)

                mu_g = Xg.mean(dim=-1)
                mu_r = Xr.mean(dim=-1)
                if mu_g.size(1) > 1:
                    diff_g = torch.diff(mu_g, dim=1)
                    diff_r = torch.diff(mu_r, dim=1)
                    L_dyn_t = (diff_g - diff_r).abs()
                else:
                    L_dyn_t = torch.zeros(mu_g.size(0), 0, device=mu_g.device, dtype=mu_g.dtype)

                vm = voiced.to(Xg.dtype)
                unv = (1.0 - vm)
                boundary = torch.zeros_like(vm)
                if vm.size(1) > 1:
                    boundary[:, 1:] = (vm[:, 1:] != vm[:, :-1]).to(vm.dtype)

                if silence_mask is not None and silence_mask.dim() == 2:
                    if silence_mask.size(1) >= T:
                        sil_tex = silence_mask[:, :T].to(unv.dtype)
                    else:
                        sil_tex = torch.zeros_like(unv, dtype=unv.dtype, device=unv.device)
                        sil_tex[:, : silence_mask.size(1)] = silence_mask.to(unv.dtype)
                    unv_nonsil = unv * (1.0 - sil_tex)
                else:
                    unv_nonsil = unv

                mask_tex = unv + 2.0 * boundary
                mask_tex = mask_tex + unv_nonsil
                mask_tex = torch.clamp(mask_tex, max=3.0)

                denom_occ = mask_tex.sum() + 1e-6
                L_occ = (L_occ_t * mask_tex).sum() / denom_occ
                if L_dyn_t.numel() > 0:
                    mask_dyn = mask_tex[:, 1:]
                    denom_dyn = mask_dyn.sum() + 1e-6
                    L_dyn = (L_dyn_t * mask_dyn).sum() / denom_dyn
                else:
                    L_dyn = torch.tensor(0.0, device=Xg.device, dtype=Xg.dtype)

                tex_grad_w = float(getattr(cfg, "texture_grad_weight", 0.5))
                L_tex_res = L_occ + tex_grad_w * L_dyn
                total = total + lam_texture * L_tex_res
                loss_dict["texture_residual"] = float(L_tex_res.detach().item())
            else:
                loss_dict["texture_residual"] = 0.0
        except Exception as _e:
            loss_dict["texture_residual"] = 0.0
            if os.environ.get("DBG_TEXTURE", "0") == "1":
                try:
                    print(f"[TEXTURE_RES] skipped due to error: {_e}")
                except Exception:
                    pass

    # ---- 10) Content/F0 entropy & bit-balance regularizers (v3-style) ----
    try:
        qtype = str(getattr(cfg, "quantizer_type", "hash"))
        hb_c = out.get("hash_bits_clean", None)
        frame_rate = 16000.0 / 160.0  # 100 Hz
        if isinstance(hb_c, torch.Tensor) and hb_c.numel() > 0:
            Bc, Tc, Kc = hb_c.shape
            Tc_eff = max(1, int(Tc))
            Tm_eff = max(1, int(out.get("ceps_hat", ceps_h).size(1))) if isinstance(ceps_h, torch.Tensor) else Tc_eff
            tokens_per_frame_c = float(Tc_eff) / float(Tm_eff)
            bits_c = hb_c.detach()
            p1_c = (bits_c > 0).float().mean(dim=(0, 1))
            p1_c = torch.clamp(p1_c, 1e-6, 1.0 - 1e-6)
            Hc = -(p1_c * torch.log2(p1_c) + (1.0 - p1_c) * torch.log2(1.0 - p1_c))
            Hc_total = float(Hc.sum().item())
            entropy_rate_c_bps = frame_rate * tokens_per_frame_c * Hc_total

            lam_c_entropy = float(getattr(cfg, "lambda_c_entropy", 0.0))
            if lam_c_entropy > 0.0:
                try:
                    alpha_c = float(getattr(cfg, "content_entropy_target_frac", 0.5))
                    alpha_c = max(0.0, min(alpha_c, 1.0))
                    H_target_c = alpha_c * float(Kc)
                    if qtype == "rvq" and "rvq_c_H" in out:
                        Hc_eff_val = out["rvq_c_H"]
                        Hc_eff = float(Hc_eff_val.detach().item()) if isinstance(Hc_eff_val, torch.Tensor) else float(Hc_eff_val)
                    else:
                        Hc_eff = float(Hc_total)
                    L_c_entropy = torch.nn.functional.relu(
                        torch.tensor(H_target_c - Hc_eff, device=hb_c.device, dtype=hb_c.dtype)
                    )
                    total = total + lam_c_entropy * L_c_entropy
                    loss_dict["content_entropy"] = float(L_c_entropy.item())
                except Exception:
                    pass

            lam_bit_balance_c = float(getattr(cfg, "lambda_bit_balance_c", 0.0))
            if lam_bit_balance_c > 0.0:
                try:
                    L_balance_c = torch.mean((p1_c - 0.5) ** 2)
                    total = total + lam_bit_balance_c * L_balance_c
                    loss_dict["bit_balance_c"] = float(L_balance_c.item())
                except Exception:
                    pass

        hb_f = out.get("f0_hash_bits_clean", None)
        if isinstance(hb_f, torch.Tensor) and isinstance(ceps_h, torch.Tensor):
            Bf, Tf, Kf = hb_f.shape
            Tf_eff = max(1, int(Tf))
            Tc_eff = max(1, int(ceps_h.size(1)))
            tokens_per_frame_f = float(Tf_eff) / float(Tc_eff)
            bits_f = hb_f.detach()
            p1_f = (bits_f > 0).float().mean(dim=(0, 1))
            p1_f = torch.clamp(p1_f, 1e-6, 1.0 - 1e-6)
            Hf = -(p1_f * torch.log2(p1_f) + (1.0 - p1_f) * torch.log2(1.0 - p1_f))
            Hf_total = float(Hf.sum().item())

            lam_f0_entropy = float(getattr(cfg, "lambda_f0_entropy", 0.0))
            if lam_f0_entropy > 0.0:
                try:
                    alpha_f = float(getattr(cfg, "f0_entropy_target_frac", 0.5))
                    alpha_f = max(0.0, min(alpha_f, 1.0))
                    H_target_f = alpha_f * float(Kf)
                    if qtype == "rvq" and "rvq_f_H" in out:
                        H_eff_val = out["rvq_f_H"]
                        H_eff = float(H_eff_val.detach().item()) if isinstance(H_eff_val, torch.Tensor) else float(H_eff_val)
                    else:
                        H_eff = float(Hf_total)
                    L_f0_entropy = torch.nn.functional.relu(
                        torch.tensor(H_target_f - H_eff, device=hb_f.device, dtype=hb_f.dtype)
                    )
                    total = total + lam_f0_entropy * L_f0_entropy
                    loss_dict["f0_entropy"] = float(L_f0_entropy.item())
                except Exception:
                    pass
    except Exception:
        pass

    return total, loss_dict


def _compute_pesq_stoi(
    ref: np.ndarray,
    deg: np.ndarray,
    sample_rate: int = 16000,
) -> Tuple[Optional[float], Optional[float]]:
    """Compute PESQ and STOI between reference and degraded audio.

    Best-effort helper adapted from the main Stage2.5 script. If third-party
    packages (pesq, pystoi) are missing, it prints a warning and returns
    ``None`` for the corresponding metric instead of raising.
    """

    pesq_score: Optional[float] = None
    stoi_score: Optional[float] = None

    ref_f = np.asarray(ref, dtype=np.float32).reshape(-1)
    deg_f = np.asarray(deg, dtype=np.float32).reshape(-1)
    if ref_f.size == 0 or deg_f.size == 0:
        return pesq_score, stoi_score

    n = min(ref_f.size, deg_f.size)
    ref_f = ref_f[:n]
    deg_f = deg_f[:n]
    ref_f = np.nan_to_num(ref_f, nan=0.0).clip(-1.0, 1.0)
    deg_f = np.nan_to_num(deg_f, nan=0.0).clip(-1.0, 1.0)

    try:
        from pesq import pesq as pesq_fn  # type: ignore

        pesq_score = float(pesq_fn(sample_rate, ref_f, deg_f, "wb"))
    except Exception as exc:
        print(f"[bit_only_eval] WARNING: PESQ computation failed or pesq package missing: {exc}")

    try:
        from pystoi import stoi as stoi_fn  # type: ignore

        stoi_val = stoi_fn(ref_f, deg_f, sample_rate, extended=False)
        try:
            stoi_score = float(stoi_val)
        except Exception:
            stoi_score = None
    except Exception as exc:
        print(f"[bit_only_eval] WARNING: STOI computation failed or pystoi package missing: {exc}")

    return pesq_score, stoi_score


def _compute_visqol(
    ref: np.ndarray,
    deg: np.ndarray,
    sample_rate: int = 16000,
) -> Optional[float]:
    """Best-effort VISQOL (MOS-LQO) between reference and degraded audio.

    实现基于 ``pyvisqol`` 包的封装：依赖缺失或运行失败时返回 ``None``，
    只打印警告，不中断训练。
    """

    try:
        import soundfile as sf  # type: ignore
        from pyvisqol.pyvisqol import Visqol  # type: ignore
    except Exception as exc:
        print(f"[bit_only_eval] WARNING: VISQOL import failed or pyvisqol missing: {exc}")
        return None

    ref_f = np.asarray(ref, dtype=np.float32).reshape(-1)
    deg_f = np.asarray(deg, dtype=np.float32).reshape(-1)
    if ref_f.size == 0 or deg_f.size == 0:
        return None

    n = min(ref_f.size, deg_f.size)
    ref_f = np.nan_to_num(ref_f[:n], nan=0.0).clip(-1.0, 1.0)
    deg_f = np.nan_to_num(deg_f[:n], nan=0.0).clip(-1.0, 1.0)

    import tempfile
    import os as _os

    tmp_dir = tempfile.mkdtemp(prefix="visqol_tmp_")
    ref_path = _os.path.join(tmp_dir, "ref.wav")
    deg_path = _os.path.join(tmp_dir, "deg.wav")

    try:
        sf.write(ref_path, ref_f, sample_rate)
        sf.write(deg_path, deg_f, sample_rate)

        try:
            engine = Visqol(mode="speech" if sample_rate == 16000 else "audio")
        except Exception as exc:
            print(f"[bit_only_eval] WARNING: VISQOL engine init failed: {exc}")
            return None

        try:
            score = float(engine.measure(ref_path, deg_path))
        except Exception as exc:
            print(f"[bit_only_eval] WARNING: VISQOL computation failed: {exc}")
            return None

        return score
    finally:
        try:
            for p in (ref_path, deg_path):
                if _os.path.isfile(p):
                    _os.remove(p)
            _os.rmdir(tmp_dir)
        except Exception:
            pass


def _simulate_fsk_bit_errors(bits: Optional[torch.Tensor], snr_db: Optional[float], device: torch.device) -> Tuple[Optional[torch.Tensor], Optional[float], Optional[float]]:
    """2-FSK 风格的 bit 级信道仿真（仅用于 bit-only eval / 诊断）。

    优先使用 Fargan_sim 中的 2-FSK 波形信道（FSKEncoder +
    add_noise_to_fsk_gpu + decode_fsk_signal），在波形上叠加噪声再
    解调回比特；若相关模块不可用或出现异常，则退回到 JSCC+FSK
    感知的 BSC 近似：

    - 若设置了 JSCC_FSK_BER_TABLE，则按 snr_db 从表中查找源比特
      BER，并用其作为翻转概率；
    - 否则回退到 BPSK+AWGN 理论 BER
      (0.5 * erfc(sqrt(SNR_lin)))。

    返回:
      bits_noisy: 注入 bit 错误后的 ±1 比特（或原 bits），
      ber_th:     理论/查表 BER，若走波形路径则为 None，
      ber_emp:    实际误码率（经验 BER）。

    注意：仅用于 bit-only eval / 诊断，不参与反向传播。
    """

    import math as _math
    import os as _os

    if bits is None or snr_db is None:
        return bits, None, None
    b = torch.as_tensor(bits, device=device, dtype=torch.float32)
    if b.numel() == 0:
        return b, None, None

    # 允许 {0,1} 或 {-1,+1} 或软值；统一映射到 ±1
    bmin = float(b.min().item())
    bmax = float(b.max().item())
    if bmin >= 0.0 and bmax <= 1.0:
        b = b * 2.0 - 1.0

    # --- 优先尝试使用 FSK 波形信道 ---
    # 默认启用波形级 2-FSK 信道仿真；若环境变量 USE_WAVE_FSK_CHANNEL
    # 显式设置为 "0"，则退回到纯 BSC 近似。在你的默认训练/评估环境
    # 下，若 Fargan_sim 与噪声 CSV 路径存在，就会自动使用真实噪声叠加。
    use_wave_fsk = _os.environ.get("USE_WAVE_FSK_CHANNEL", "1") != "0"
    if use_wave_fsk:
        try:
            import numpy as _np
            import sys as _sys

            fargan_sim_root = _os.environ.get("FARGAN_SIM_ROOT", "")
            if not fargan_sim_root:
                try:
                    base = _os.path.dirname(root_dir)  # root_dir points to the repository root
                    fargan_sim_root = _os.path.abspath(
                        _os.path.join(base, "..", "Fargan_sim")
                    )
                except Exception:
                    fargan_sim_root = ""

            if not fargan_sim_root or not _os.path.isdir(fargan_sim_root):
                raise RuntimeError(
                    "Fargan_sim root not found; set FARGAN_SIM_ROOT to /home/bluestar/FARGAN/opus/dnn/Fargan_sim"
                )

            if fargan_sim_root not in _sys.path:
                _sys.path.append(fargan_sim_root)

            from fsk_encoder import FSKEncoder  # type: ignore
            from main import (  # type: ignore
                add_noise_to_fsk_gpu,
                decode_fsk_signal,
            )

            # 展平成 1D bit 序列（0/1 表示）
            b_cpu = b.detach().cpu().numpy().astype(_np.float32)
            bits01 = (b_cpu > 0.0).astype(_np.uint8).reshape(-1)

            # FSK 参数从环境变量读取
            fsk_fs = float(_os.environ.get("FSK_FS", "48000"))
            f0 = float(_os.environ.get("FSK_F0", "18500"))
            f1 = float(_os.environ.get("FSK_F1", "19500"))
            symbol_duration = float(_os.environ.get("FSK_SYMBOL", "0.002"))
            amplitude = float(_os.environ.get("FSK_AMPL", "0.2"))

            modem = FSKEncoder(
                sample_rate=fsk_fs,
                f0=f0,
                f1=f1,
                symbol_duration=symbol_duration,
                amplitude=amplitude,
            )
            fsk_signal = modem.generate_fsk_signal(bits01)

            noise_csv = _os.environ.get(
                "FSK_NOISE_CSV",
                "/home/bluestar/FARGAN/opus/dnn/Fargan_sim/noise_voltage_50s_300s.csv",
            )
            if not noise_csv or not _os.path.isfile(noise_csv):
                raise RuntimeError(
                    f"FSK_NOISE_CSV not set or file missing: {noise_csv}"
                )

            fsk_bitrate = float(_os.environ.get("FSK_BITRATE", "2000"))
            noise_fs = float(_os.environ.get("FSK_NOISE_FS", "48000"))

            noisy_signal, _ = add_noise_to_fsk_gpu(
                fsk_signal,
                noise_csv,
                float(snr_db),
                fsk_bitrate=fsk_bitrate,
                noise_fs=noise_fs,
                fsk_fs=fsk_fs,
            )

            decoded_bits = decode_fsk_signal(
                noisy_signal,
                sample_rate=fsk_fs,
                f0=f0,
                f1=f1,
                symbol_duration=symbol_duration,
            )

            decoded_bits = _np.asarray(decoded_bits, dtype=_np.uint8)
            if decoded_bits.size < bits01.size:
                # pad 0s if decode shorter
                pad = _np.zeros(bits01.size - decoded_bits.size, dtype=_np.uint8)
                decoded_bits = _np.concatenate([decoded_bits, pad], axis=0)
            decoded_bits = decoded_bits[: bits01.size]

            # 回到原来的张量形状，并映射回 ±1
            decoded_bits = decoded_bits.reshape(b_cpu.shape)
            b_noisy_np = decoded_bits.astype(_np.float32) * 2.0 - 1.0
            b_noisy = torch.from_numpy(b_noisy_np).to(device=device, dtype=b.dtype)

            # 估计一个经验 BER
            ber_emp = float((decoded_bits.flatten() != bits01.flatten()).mean())
            return b_noisy, None, ber_emp
        except Exception as _e_fsk:
            print(f"[_simulate_fsk_bit_errors] waveform FSK sim failed, fallback to BSC: {_e_fsk}")

    # --- BSC 近似 ---
    # 优先尝试从 JSCC_FSK_BER_TABLE 查表，以匹配实际 JSCC+FSK
    # bit_only_metrics.csv 中观察到的 BER–SNR 行为；若表缺失或
    # 格式错误，则退回到理论 BPSK+AWGN 公式。
    ber_th = _lookup_jscc_fsk_ber(float(snr_db))
    if ber_th is None:
        snr_lin = 10.0 ** (float(snr_db) / 10.0)
        ber_th = 0.5 * _math.erfc(_math.sqrt(max(snr_lin, 1e-8)))
        ber_th = max(0.0, min(ber_th, 0.5))

    if ber_th <= 0.0:
        return b, ber_th, 0.0
    flip = (torch.rand_like(b) < ber_th)
    b_noisy = torch.where(flip, -b, b)
    ber_emp = float(flip.float().mean().item())
    return b_noisy, ber_th, ber_emp


def _compute_jscc_fsk_ber_via_schemes(
    bits_c: Optional[torch.Tensor],
    bits_f: Optional[torch.Tensor],
    bits_s: Optional[torch.Tensor],
    snr_db: float,
) -> Optional[float]:
    """Compute source-bit BER via Fargan_sim's ``run_jscc_hash_fsk``.

    This helper flattens ``content+F0+stats`` hash bits into a single
    0/1 numpy vector, writes it to a temporary ``.npy`` file under the
    repository root, and calls ``Fargan_sim.schemes.run_jscc_hash_fsk``
    with ``FSKConfig`` defaults and real-noise channel parameters.

    It returns the *source-bit* BER as defined in Fargan_sim (Scheme C):
    errors between the original source bits and the bits recovered after
    2-FSK modulation/demodulation (and LDPC decoding when enabled).

    On any failure (missing Fargan_sim, noise CSV, import error, etc.),
    the function returns ``None`` and prints a debug message only when
    ``DBG_FSK_BER=1`` is set in the environment.
    """

    # Collect all non-empty bit tensors for this sample.
    src_tensors: List[torch.Tensor] = []
    for t in (bits_c, bits_f, bits_s):
        if isinstance(t, torch.Tensor) and t.numel() > 0:
            src_tensors.append(t.detach().cpu().reshape(-1))

    if not src_tensors:
        return None

    bits_flat = torch.cat(src_tensors, dim=0)
    if bits_flat.numel() == 0:
        return None

    bits_np = bits_flat.numpy().astype(np.float32)
    bits01 = (bits_np > 0.0).astype(np.uint8)
    if bits01.size == 0:
        return None

    dbg = os.environ.get("DBG_FSK_BER", "0") == "1"

    # Locate Fargan_sim root: prefer explicit env, otherwise infer from
    # the shared ``dnn`` layout used on your machine.
    fargan_sim_root = os.environ.get("FARGAN_SIM_ROOT", "")
    if not fargan_sim_root:
        try:
            base = os.path.dirname(root_dir)  # root_dir -> repository root
            fargan_sim_root = os.path.abspath(os.path.join(base, "..", "Fargan_sim"))
        except Exception:
            fargan_sim_root = ""

    if not (fargan_sim_root and os.path.isdir(fargan_sim_root)):
        if dbg:
            print(
                f"[JSCC-Simplified] Fargan_sim root not found at '{fargan_sim_root}', "
                "skip JSCC FSK BER via schemes.run_jscc_hash_fsk",
            )
        return None

    if fargan_sim_root not in sys.path:
        sys.path.append(fargan_sim_root)

    try:
        from schemes import run_jscc_hash_fsk, FSKConfig  # type: ignore
    except Exception as exc:  # pragma: no cover - external dependency
        if dbg:
            print(f"[JSCC-Simplified] Import Fargan_sim.schemes failed: {exc}")
        return None

    noise_csv = os.environ.get(
        "FSK_NOISE_CSV",
        os.path.join(fargan_sim_root, "noise_voltage_50s_300s.csv"),
    )
    if not noise_csv or not os.path.isfile(noise_csv):
        if dbg:
            print(f"[JSCC-Simplified] Noise CSV for FSK BER not found: {noise_csv}")
        return None

    # Temporary bit files stay under the repository root so that
    # they are always writable under the current workspace sandbox.
    bits_tmp = os.path.join(root_dir, "jscc_bits_tmp.npy")
    bits_rx_tmp = os.path.join(root_dir, "jscc_bits_rx_tmp.npy")

    try:
        np.save(bits_tmp, bits01)
        fsk_cfg = FSKConfig()
        res = run_jscc_hash_fsk(
            bits_file=bits_tmp,
            bits_rx_out=bits_rx_tmp,
            noise_csv=noise_csv,
            snr_db=float(snr_db),
            fsk_cfg=fsk_cfg,
            use_ldpc=False,
            debug_timing=dbg,
        )
        ber_val = float(res.get("ber", 0.0))
        return ber_val
    except Exception as exc:  # pragma: no cover - external dependency
        if dbg:
            print(f"[JSCC-Simplified] run_jscc_hash_fsk failed: {exc}")
        return None
    finally:
        for p in (bits_tmp, bits_rx_tmp):
            try:
                if os.path.isfile(p):
                    os.remove(p)
            except Exception:
                pass


def _compute_bit_only_silence_loss(
    model: DualBranchBarkJSCC,
    batch: Dict[str, torch.Tensor],
    cfg: TrainingConfig,
    channel_sim: ChannelSimulator,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Bit-only BFCC 静音约束：直接在 bits→audio→BFCC 路径上压低静音底噪。

    - 仅当 ``lambda_bit_only_silence>0`` 且模型提供 encode/decode 接口时启用；
    - 为了控制开销，只使用当前 batch 的前 ``bit_only_eval_max_samples`` 条样本；
    - 静音判定与主 BFCC 静音 loss 共享 ``silence_mel_thr_db`` 与
      ``silence_hf_low_bins``。
    """

    lam = float(getattr(cfg, "lambda_bit_only_silence", 0.0) or 0.0)
    if lam <= 0.0:
        return torch.zeros((), device=device), {}

    if not (getattr(model, "with_hash", False) and hasattr(model, "encode_hash_codec") and hasattr(model, "decode_from_bits_offline") and hasattr(model, "wave_to_mel")):
        return torch.zeros((), device=device), {}

    try:
        audio = batch.get("audio")
        if audio is None:
            return torch.zeros((), device=device), {}
        audio = audio.to(device)

        feats_src = None
        if "x" in batch:
            feats_src = batch["x"]
        elif "features" in batch:
            feats_src = batch["features"]
        if feats_src is None:
            return torch.zeros((), device=device), {}
        feats = feats_src.to(device)

        B_cur = int(audio.size(0))
        max_samples = int(getattr(cfg, "bit_only_eval_max_samples", 2) or 2)
        B_eval = min(B_cur, max_samples)
        if B_eval <= 0:
            return torch.zeros((), device=device), {}

        audio_eval = audio[:B_eval]
        feats_eval = feats[:B_eval]

        bits_c, bits_f, bits_s, meta = model.encode_hash_codec(
            audio=audio_eval,
            fargan_feats=feats_eval,
            channel_sim=channel_sim,
            snr_min_db=float(getattr(cfg, "snr_min_db", -5.0)),
            snr_max_db=float(getattr(cfg, "snr_max_db", 15.0)),
            use_noisy_bits=False,
        )

        csi_vec = None
        if isinstance(meta, dict) and "csi_vec" in meta:
            try:
                csi_val = torch.as_tensor(meta["csi_vec"], device=device, dtype=torch.float32)
                if csi_val.dim() == 2:
                    csi_vec = csi_val
            except Exception:
                csi_vec = None

        out_bits = model.decode_from_bits_offline(
            bits_content=bits_c,
            bits_f0=bits_f,
            bits_stats=bits_s,
            f0_T=int(meta.get("T", feats_eval.size(1))) if isinstance(meta, dict) else feats_eval.size(1),
            target_len=int(audio_eval.size(1)),
            csi_vec=csi_vec,
            snr_db=None,
            content_hw=meta.get("hw", None) if isinstance(meta, dict) else None,
        )

        if not (isinstance(out_bits, dict) and "audio_hat" in out_bits):
            return torch.zeros((), device=device), {}

        audio_bits = out_bits["audio_hat"]  # [B_eval,L]
        mel_real = model.wave_to_mel(audio_eval)  # [B,T,F]
        mel_bits = model.wave_to_mel(audio_bits)

        Bm, Tm, Fm = mel_real.shape
        Tb = mel_bits.size(1)
        Fb = mel_bits.size(2)
        T_use = min(Tm, Tb)
        F_use = min(Fm, Fb)
        mel_real = mel_real[:, :T_use, :F_use]
        mel_bits = mel_bits[:, :T_use, :F_use]

        hf_low = int(getattr(cfg, "silence_hf_low_bins", 16))
        hf_low = max(0, min(hf_low, F_use - 1))
        mel_real_hf = mel_real[:, :, hf_low:] if hf_low < F_use else mel_real
        mel_bits_hf = mel_bits[:, :, hf_low:] if hf_low < F_use else mel_bits

        mel_energy = mel_real_hf.mean(dim=-1)  # [B,T]
        thr_db = float(getattr(cfg, "silence_mel_thr_db", -35.0))
        thr_log = thr_db / 10.0
        silence_mask = (mel_energy <= thr_log)  # [B,T]
        if not silence_mask.any():
            return torch.zeros((), device=device), {}

        sm = silence_mask.to(mel_bits_hf.dtype)
        diff = torch.abs(mel_bits_hf - mel_real_hf) * sm.unsqueeze(-1)
        denom = sm.sum() * max(mel_bits_hf.size(-1), 1) + 1e-6
        loss_core = diff.sum() / denom

        loss = lam * loss_core
        return loss, {"bit_sil_mel_hf": float(loss_core.detach().item())}
    except Exception as _e_bit:
        print(f"[bit_only_silence] WARNING: skipped due to error: {_e_bit}")
        return torch.zeros((), device=device), {}


def _run_bit_only_eval_for_batch(
    model: DualBranchBarkJSCC,
    batch: Dict[str, torch.Tensor],
    out: Dict[str, torch.Tensor],
    cfg: TrainingConfig,
    channel_sim: ChannelSimulator,
    device: torch.device,
    global_step_val: int,
    do_visualization: bool,
    snr_grid_all: Optional[List[float]] = None,
) -> None:
    """Run bit-only JSCC eval for a single batch and append CSV rows.

    This helper encapsulates the logic that was previously inlined in the
    training loop so that it can be reused both during training (with
    ``do_visualization=True``) and in ``only_eval`` mode (with
    ``do_visualization=False``).
    """

    # Guard: require hash/RVQ codec support.
    if not (
        getattr(model, "with_hash", False)
        and hasattr(model, "encode_hash_codec")
        and hasattr(model, "decode_from_bits_offline")
    ):
        return

    try:
        audio_real_b = out["audio"].detach().cpu()
        max_samples = int(getattr(cfg, "bit_only_eval_max_samples", 2))
        B_cur = int(audio_real_b.size(0))

        # SNR 栅格（dB）。若调用方未显式传入，则使用训练时的默认
        # 稀疏栅格 [-5, -3, 0, 3, 5, 10]；在 only_eval 模式下，调用方
        # 可传入更密集的 [-5, -4, ..., 10]。
        if snr_grid_all is None:
            snr_grid_all = [-5.0, -3.0, 0.0, 3.0, 5.0, 10.0]
        B_eval = min(B_cur, max_samples, len(snr_grid_all))
        if B_eval <= 0:
            return

        audio_eval = out["audio"][:B_eval].to(device)

        # FARGAN 特征：优先 batch['x']，其次 batch['features']。
        feats_src = None
        if isinstance(batch, dict):
            if "x" in batch:
                feats_src = batch["x"]
            elif "features" in batch:
                feats_src = batch["features"]
        if feats_src is None:
            raise KeyError("bit_only_eval requires batch['x'] or batch['features']")
        feats_eval = feats_src[:B_eval].to(device)

        # 1) 导出 clean bits 与 CSI meta（不经 FSK/信道扰动）。
        bits_c, bits_f, bits_s, meta = model.encode_hash_codec(
            audio=audio_eval,
            fargan_feats=feats_eval,
            channel_sim=channel_sim,
            snr_min_db=float(getattr(cfg, "snr_min_db", -5.0)),
            snr_max_db=float(getattr(cfg, "snr_max_db", 15.0)),
            use_noisy_bits=False,
        )

        bits_c_clean = bits_c
        bits_f_clean = bits_f
        bits_s_clean = bits_s

        # 2) 提取 CSI（仅作为 decode_from_bits_offline 的辅助），不再
        #    把其均值当作唯一评测 SNR。
        csi_vec = None
        if isinstance(meta, dict) and "csi_vec" in meta:
            try:
                csi_val = torch.as_tensor(meta["csi_vec"], device=device, dtype=torch.float32)
                if csi_val.dim() == 2 and csi_val.size(1) >= 1:
                    csi_vec = csi_val
            except Exception:
                csi_vec = None

        # 3) 使用固定 SNR 栅格：[-5,-3,0,3,5,10]。在上面已将 B_eval 限制为
        #    不超过栅格长度，这里直接取前 B_eval 个 SNR，每个样本 i
        #    对应 snr_grid[i]。
        snr_grid = snr_grid_all[:B_eval]

        # 为了让 decoder 侧的 CSI 中的 SNR 与当前 FSK bit 翻转所使用的
        # snr_eval 对齐，这里基于 meta 中的 csi_vec 构造一个新的
        # csi_vec_eval：除第 0 维 SNR 外，其余 CSI 分量保持不变，只将每个
        # 样本的 snr_proxy 覆盖为对应的 snr_grid[i]。
        csi_vec_eval = csi_vec
        if isinstance(csi_vec, torch.Tensor):
            try:
                if csi_vec.dim() == 2 and csi_vec.size(0) >= B_eval and csi_vec.size(1) >= 1:
                    csi_vec_eval = csi_vec.clone()
                    snr_tensor = torch.as_tensor(
                        snr_grid,
                        device=csi_vec_eval.device,
                        dtype=csi_vec_eval.dtype,
                    )
                    csi_vec_eval[:B_eval, 0] = snr_tensor
            except Exception:
                csi_vec_eval = csi_vec

        # Basic F0/Bark MSE placeholders（占位，便于后续扩展）。
        f0_mse = 0.0
        mel_mse = 0.0

        from pathlib import Path
        import csv as _csv

        bit_csv_path = Path(cfg.ckpt_dir).expanduser().resolve() / "bit_only_metrics.csv"
        bit_csv_path.parent.mkdir(parents=True, exist_ok=True)
        is_new_csv = not bit_csv_path.is_file()
        fieldnames = [
            "step",
            "stoi",
            "pesq",
            "f0_mse",
            "mel_mse",
            "bfcc_psnr_db",
            "snr_db",
            "ber",  # JSCC+FSK 源比特 BER（对齐 Fargan_sim 定义）
            "visqol",
            "rate_c_kbps",
            "rate_f_kbps",
            "rate_total_kbps",
        ]

        # Bitrate 估计对所有样本/SNR 相同，先算一次即可。
        rate_c_kbps_str = ""
        rate_f_kbps_str = ""
        rate_total_kbps_str = ""
        try:
            frame_rate = 16000.0 / 160.0  # 100 Hz
            rate_c_bps = 0.0
            rate_f_bps = 0.0
            if isinstance(bits_c_clean, torch.Tensor):
                _, Lc, Kc = bits_c_clean.shape
                T = int(meta.get("T", feats_eval.size(1))) if isinstance(meta, dict) else int(feats_eval.size(1))
                if T > 0:
                    tokens_per_frame_c = float(Lc) / float(T)
                    rate_c_bps = frame_rate * tokens_per_frame_c * float(Kc)
            if isinstance(bits_f_clean, torch.Tensor):
                _, Tf, Kf = bits_f_clean.shape
                T = int(meta.get("T", feats_eval.size(1))) if isinstance(meta, dict) else int(feats_eval.size(1))
                if T > 0:
                    tokens_per_frame_f = float(Tf) / float(T)
                    rate_f_bps = frame_rate * tokens_per_frame_f * float(Kf)
            rate_c_kbps_str = f"{rate_c_bps/1000.0:.4f}"
            rate_f_kbps_str = f"{rate_f_bps/1000.0:.4f}"
            rate_total_kbps_str = f"{(rate_c_bps+rate_f_bps)/1000.0:.4f}"
        except Exception:
            pass

        bits_c_list: List[torch.Tensor] = []
        bits_f_list: List[torch.Tensor] = []
        bits_s_list: List[torch.Tensor] = []
        ber_src_list: List[Optional[float]] = []

        # 4) 针对每个样本 i：使用 snr_grid[i] 注入一次 FSK 风格的 bit
        #    错误用于本地 decode_from_bits_offline 的音频评估；同时通过
        #    Fargan_sim 的 run_jscc_hash_fsk 计算“源比特 BER”，并记录到 CSV。
        for i_s, snr_eval in enumerate(snr_grid):
            try:
                # 4a) 使用 Fargan_sim.schemes 计算源比特 BER。
                ber_i: Optional[float] = None
                try:
                    ber_i = _compute_jscc_fsk_ber_via_schemes(
                        bits_c=bits_c_clean[i_s : i_s + 1]
                        if isinstance(bits_c_clean, torch.Tensor)
                        else None,
                        bits_f=bits_f_clean[i_s : i_s + 1]
                        if isinstance(bits_f_clean, torch.Tensor)
                        else None,
                        bits_s=bits_s_clean[i_s : i_s + 1]
                        if isinstance(bits_s_clean, torch.Tensor)
                        else None,
                        snr_db=float(snr_eval),
                    )
                except Exception as _e_ber:
                    if os.environ.get("DBG_FSK_BER", "0") == "1":
                        print(
                            f"[JSCC-Simplified] _compute_jscc_fsk_ber_via_schemes "
                            f"failed for sample {i_s} at SNR={snr_eval}: {_e_ber}"
                        )

                # 4b) 本地 FSK/BSC 仿真仅用于为 decode_from_bits_offline
                #     提供 noisy bits，不再用于 BER 统计。
                if isinstance(bits_c_clean, torch.Tensor):
                    bc_src = bits_c_clean[i_s : i_s + 1]
                    bc_i, _, _ = _simulate_fsk_bit_errors(bc_src, snr_eval, device)
                else:
                    bc_i = bits_c_clean

                if isinstance(bits_f_clean, torch.Tensor):
                    bf_src = bits_f_clean[i_s : i_s + 1]
                    bf_i, _, _ = _simulate_fsk_bit_errors(bf_src, snr_eval, device)
                else:
                    bf_i = bits_f_clean

                if isinstance(bits_s_clean, torch.Tensor):
                    bs_src = bits_s_clean[i_s : i_s + 1]
                    bs_i, _, _ = _simulate_fsk_bit_errors(bs_src, snr_eval, device)
                else:
                    bs_i = bits_s_clean

            except Exception:
                bc_i, bf_i, bs_i = bits_c_clean, bits_f_clean, bits_s_clean
                ber_i = None

            if isinstance(bc_i, torch.Tensor):
                bits_c_list.append(bc_i)
            if isinstance(bf_i, torch.Tensor):
                bits_f_list.append(bf_i)
            if isinstance(bs_i, torch.Tensor):
                bits_s_list.append(bs_i)
            ber_src_list.append(ber_i)

        bits_c_snr = torch.cat(bits_c_list, dim=0) if bits_c_list else bits_c_clean
        bits_f_snr = torch.cat(bits_f_list, dim=0) if bits_f_list else bits_f_clean
        bits_s_snr = torch.cat(bits_s_list, dim=0) if bits_s_list else bits_s_clean

        # Optional: print per-SNR BER summary when DBG_FSK_BER=1。
        if os.environ.get("DBG_FSK_BER", "0") == "1":
            valid_bers = [b for b in ber_src_list if b is not None]
            if valid_bers:
                mean_ber = sum(valid_bers) / float(len(valid_bers))
            else:
                mean_ber = None
            print(f"[bit_only_eval][FSK-BER] step={int(global_step_val)}")
            for i_s, snr_eval in enumerate(snr_grid):
                ber_i = ber_src_list[i_s] if i_s < len(ber_src_list) else None
                if ber_i is None:
                    print(f"  sample={i_s} SNR={snr_eval:.1f} dB: BER=None")
                else:
                    print(f"  sample={i_s} SNR={snr_eval:.1f} dB: BER={ber_i:.6e}")
            if mean_ber is not None:
                print(
                    f"[bit_only_eval][FSK-BER] step={int(global_step_val)} "
                    f"mean_BER={mean_ber:.6e} over {len(valid_bers)} samples"
                )

        # Debug path: optionally decode *clean* bits as well, to compare
        # with the FSK-perturbed bits in terms of mel/audio distribution.
        f0_T_use = int(meta.get("T", feats_eval.size(1))) if isinstance(meta, dict) else feats_eval.size(1)
        target_len_use = int(audio_eval.size(1))

        out_bits_clean: Optional[Dict[str, torch.Tensor]] = None
        if os.environ.get("DBG_FSK_DECODE", "0") == "1":
            try:
                out_bits_clean = model.decode_from_bits_offline(
                    bits_content=bits_c_clean[:B_eval] if isinstance(bits_c_clean, torch.Tensor) else None,
                    bits_f0=bits_f_clean[:B_eval] if isinstance(bits_f_clean, torch.Tensor) else None,
                    bits_stats=bits_s_clean[:B_eval] if isinstance(bits_s_clean, torch.Tensor) else None,
                    f0_T=f0_T_use,
                    target_len=target_len_use,
                    csi_vec=csi_vec_eval,
                    snr_db=None,
                    content_hw=meta.get("hw", None) if isinstance(meta, dict) else None,
                )
            except Exception as _e_dec_clean:
                print(f"[DBG_FSK_DECODE] clean decode failed at step {global_step_val}: {_e_dec_clean}")

        out_bits = model.decode_from_bits_offline(
            bits_content=bits_c_snr,
            bits_f0=bits_f_snr,
            bits_stats=bits_s_snr,
            f0_T=f0_T_use,
            target_len=target_len_use,
            csi_vec=csi_vec_eval,
            snr_db=None,
            content_hw=meta.get("hw", None) if isinstance(meta, dict) else None,
        )

        if not (isinstance(out_bits, dict) and "audio_hat" in out_bits):
            return

        audio_gen_bits = out_bits["audio_hat"].detach().cpu()

        if os.environ.get("DBG_FSK_DECODE", "0") == "1":
            # 打印 clean bits vs FSK bits offline decode 的 mel/audio 统计，
            # 用于验证“信道本身”对编码特征分布的影响是不是过于极端。
            def _stat_np(x: Optional[torch.Tensor]) -> str:
                if not isinstance(x, torch.Tensor) or x.numel() == 0:
                    return "mean=N/A std=N/A min=N/A max=N/A"
                x_np = x.detach().to(torch.float32).view(-1).cpu().numpy()
                return (
                    f"mean={float(x_np.mean()):+.3e} std={float(x_np.std()):+.3e} "
                    f"min={float(x_np.min()):+.3e} max={float(x_np.max()):+.3e}"
                )

            mel_clean = None
            audio_clean = None
            if isinstance(out_bits_clean, dict):
                mel_clean = out_bits_clean.get("mel_hat", None)
                audio_clean = out_bits_clean.get("audio_hat", None)

            mel_fsk = out_bits.get("mel_hat", None)
            audio_fsk = out_bits.get("audio_hat", None)

            print(f"[FSK-DECODE] step={global_step_val} SNR_grid={snr_grid_all[:B_eval]}")
            print(f"  mel_clean: {_stat_np(mel_clean)}")
            print(f"  mel_fsk:   {_stat_np(mel_fsk)}")
            print(f"  audio_clean: {_stat_np(audio_clean)}")
            print(f"  audio_fsk:   {_stat_np(audio_fsk)}")

        # 可视化：在训练过程中可以额外导出 bit-only 波形对比图和音频；
        # 在 only_eval 模式下（do_visualization=False）则跳过。
        if do_visualization:
            viz_dir_bits = os.path.join(cfg.viz_dir, "bit_only")
            os.makedirs(viz_dir_bits, exist_ok=True)
            create_batch_comparison_plots(
                audio_real_batch=audio_real_b[:B_eval],
                audio_gen_batch=audio_gen_bits,
                save_dir=viz_dir_bits,
                step=global_step_val,
                max_samples=B_eval,
                sr=16000,
            )
            save_comparison_audio_samples(
                audio_real_batch=audio_real_b[:B_eval],
                audio_gen_batch=audio_gen_bits,
                save_dir=viz_dir_bits,
                step=global_step_val,
                max_samples=B_eval,
                sr=16000,
            )

        # 额外：为 bit-only 路径计算对应的 BFCC PSNR（GT vs bits-only）。
        # 在训练模式下若启用可视化，还会保存 BFCC 图像；only_eval 下
        # 仅计算数值并写入 CSV，不生成图片。此外，若主干 forward 输出
        # 中包含内容分支的 mel 解码特征（mel_hat 或 mel_hat_refined），
        # 也一并导出其可视化，方便对比“内容分支 decode vs bit-only”。
        bfcc_psnr_list: List[Optional[float]] = [None] * B_eval
        try:
            if hasattr(model, "wave_to_mel"):
                bfcc_dir_bits = os.path.join(cfg.viz_dir, "bit_only_bfcc")
                if do_visualization:
                    os.makedirs(bfcc_dir_bits, exist_ok=True)

                hop = 160
                sr_vis = 16000

                def _save_bfcc_img_bits(arr_T: np.ndarray, path: str, title: str) -> None:
                    if not do_visualization:
                        return
                    if arr_T.ndim > 2:
                        arr_T = arr_T.mean(axis=-1)
                    elif arr_T.ndim < 2:
                        arr_T = np.reshape(arr_T, (1, -1))
                    vmin = float(np.percentile(arr_T, 1))
                    vmax = float(np.percentile(arr_T, 99))
                    plt.figure(figsize=(8, 3))
                    n_frames = arr_T.shape[1]
                    duration_sec = n_frames * hop / float(sr_vis)
                    extent = [0.0, duration_sec, 0, arr_T.shape[0]]
                    img = plt.imshow(
                        arr_T,
                        origin="lower",
                        aspect="auto",
                        interpolation="nearest",
                        cmap="magma",
                        vmin=vmin,
                        vmax=vmax,
                        extent=extent,
                    )
                    plt.colorbar(img, label="log10 energy")
                    plt.xlabel("Time (s)")
                    plt.ylabel("Mel/Bark band index")
                    plt.title(title)
                    plt.tight_layout()
                    plt.savefig(path, dpi=150)
                    plt.close()

                mel_content_b: Optional[torch.Tensor] = None
                with torch.no_grad():
                    audio_real_eval = audio_real_b[:B_eval].to(device)
                    audio_bits_eval = audio_gen_bits.to(device)
                    mel_real_b = model.wave_to_mel(audio_real_eval)  # [B,T,F]
                    mel_bits_b = model.wave_to_mel(audio_bits_eval)

                    mel_content_src = out.get("mel_hat_refined", out.get("mel_hat", None))
                    if isinstance(mel_content_src, torch.Tensor):
                        try:
                            mel_content_b = mel_content_src[:B_eval]
                            if mel_content_b.dim() == 4:
                                mel_content_b = mel_content_b.squeeze(1)
                        except Exception:
                            mel_content_b = None

                for i in range(B_eval):
                    mel_real_i = mel_real_b[i].detach().cpu().numpy().T  # [F,T]
                    mel_bits_i = mel_bits_b[i].detach().cpu().numpy().T

                    mel_content_i: Optional[np.ndarray] = None
                    if isinstance(mel_content_b, torch.Tensor) and mel_content_b.size(0) > i:
                        try:
                            mel_content_i = (
                                mel_content_b[i]
                                .detach()
                                .to(torch.float32)
                                .cpu()
                                .numpy()
                                .T
                            )
                        except Exception:
                            mel_content_i = None

                    try:
                        psnr_val = bfcc_psnr(mel_real_i, mel_bits_i)
                        bfcc_psnr_list[i] = float(psnr_val)
                    except Exception as _psnr_err:
                        print(
                            f"[BFCC-PSNR] failed for sample {i} at step "
                            f"{global_step_val}: {_psnr_err}"
                        )
                        bfcc_psnr_list[i] = None

                    if do_visualization:
                        base_name = f"step{global_step_val:06d}_sample{i:02d}"
                        real_path = os.path.join(
                            bfcc_dir_bits,
                            base_name + "_bfcc_gt.png",
                        )
                        bits_path = os.path.join(
                            bfcc_dir_bits,
                            base_name + "_bfcc_bits.png",
                        )
                        _save_bfcc_img_bits(
                            mel_real_i,
                            real_path,
                            "BFCC (GT, bit-only eval)",
                        )
                        _save_bfcc_img_bits(
                            mel_bits_i,
                            bits_path,
                            "BFCC (bits-only decoded)",
                        )

                        if mel_content_i is not None:
                            content_path = os.path.join(
                                bfcc_dir_bits,
                                base_name + "_content_mel.png",
                            )
                            _save_bfcc_img_bits(
                                mel_content_i,
                                content_path,
                                "Content feature (mel_hat_refined)",
                            )
        except Exception as _e_bfcc:
            print(
                f"[Simplified] bit_only BFCC export failed at step {global_step_val}: {_e_bfcc}"
            )

        # 最终将每个样本 / SNR 对应的一行指标写入 CSV。
        with bit_csv_path.open("a", newline="", encoding="utf-8") as _fcsv:
            _writer = _csv.DictWriter(_fcsv, fieldnames=fieldnames)
            if is_new_csv:
                _writer.writeheader()

            for i in range(B_eval):
                ref_np = audio_real_b[i].detach().cpu().numpy()
                deg_np = audio_gen_bits[i].detach().cpu().numpy()
                pesq_i, stoi_i = _compute_pesq_stoi(ref_np, deg_np, sample_rate=16000)
                visq_i = _compute_visqol(ref_np, deg_np, sample_rate=16000)

                snr_eval = snr_grid[i]
                ber_val = ber_src_list[i] if i < len(ber_src_list) else None

                psnr_val = None
                if bfcc_psnr_list[i] is not None:
                    psnr_val = float(bfcc_psnr_list[i])

                row = {
                    "step": int(global_step_val),
                    "stoi": f"{stoi_i:.6f}" if stoi_i is not None else "",
                    "pesq": f"{pesq_i:.6f}" if pesq_i is not None else "",
                    "f0_mse": f"{f0_mse:.6f}",
                    "mel_mse": f"{mel_mse:.6f}",
                    "bfcc_psnr_db": f"{psnr_val:.4f}" if psnr_val is not None else "",
                    "snr_db": f"{snr_eval:.4f}",
                    # 统一 BER 定义：源比特 BER（若可用），否则留空。
                    "ber": f"{ber_val:.6e}" if ber_val is not None else "",
                    "visqol": f"{visq_i:.6f}" if visq_i is not None else "",
                    "rate_c_kbps": rate_c_kbps_str,
                    "rate_f_kbps": rate_f_kbps_str,
                    "rate_total_kbps": rate_total_kbps_str,
                }
                _writer.writerow(row)
    except Exception as e:
        print(f"[Simplified] bit_only_eval failed at step {global_step_val}: {e}")


def _run_forward_bfcc_eval_for_batch(
    model: DualBranchBarkJSCC,
    out: Dict[str, torch.Tensor],
    cfg: TrainingConfig,
    device: torch.device,
    global_step_val: int,
) -> None:
    """BFCC/Bark-domain validation for *content branch* outputs.

    与 bit-only eval 的 BFCC 不同，这里直接对内容分支的 mel 输出
    进行比较：优先使用 ``mel_hat_refined``，否则退回 ``mel_hat``，并
    与 GT ``mel``（通常是 WaveToBFCC 输出的 Bark log-energy 图）对齐，
    计算 BFCC PSNR 并导出 GT / pred 图像。

    - 完全基于内容分支特征，不依赖 RVQ/hash 或 decode_from_bits_offline；
    - 仅在 ``bfcc_forward_eval=True`` 时从训练主循环中调用；
    - 为了节省开销，只处理前 ``bfcc_forward_max_samples`` 条样本。
    """

    mel_ref = out.get("mel")
    mel_hat_any = out.get("mel_hat_refined", out.get("mel_hat", None))
    if not (
        isinstance(mel_ref, torch.Tensor)
        and isinstance(mel_hat_any, torch.Tensor)
    ):
        return

    try:
        mel_ref = mel_ref.to(device)
        mel_hat_any = mel_hat_any.to(device)

        B_cur = int(mel_ref.size(0))
        max_samples = int(getattr(cfg, "bfcc_forward_max_samples", 2) or 2)
        B_eval = min(B_cur, max_samples)
        if B_eval <= 0:
            return

        mel_real_b = mel_ref[:B_eval]
        mel_hat_b = mel_hat_any[:B_eval]

        Bm, Tm, Fm = mel_real_b.shape
        Tb = mel_hat_b.size(1)
        Fb = mel_hat_b.size(2)
        T_use = min(Tm, Tb)
        F_use = min(Fm, Fb)
        mel_real_b = mel_real_b[:, :T_use, :F_use]
        mel_hat_b = mel_hat_b[:, :T_use, :F_use]

        bfcc_dir = os.path.join(cfg.viz_dir, "bfcc_forward")
        os.makedirs(bfcc_dir, exist_ok=True)

        def _save_bfcc_img(arr_T: np.ndarray, path: str, title: str) -> None:
            # arr_T: [F,T]
            if arr_T.ndim > 2:
                arr_T = arr_T.mean(axis=-1)
            elif arr_T.ndim < 2:
                arr_T = np.reshape(arr_T, (1, -1))
            vmin = float(np.percentile(arr_T, 1))
            vmax = float(np.percentile(arr_T, 99))
            plt.figure(figsize=(8, 3))
            hop = 160
            sr_vis = 16000
            n_frames = arr_T.shape[1]
            duration_sec = n_frames * hop / float(sr_vis)
            extent = [0.0, duration_sec, 0, arr_T.shape[0]]
            img = plt.imshow(
                arr_T,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                cmap="magma",
                vmin=vmin,
                vmax=vmax,
                extent=extent,
            )
            plt.colorbar(img, label="log10 energy")
            plt.xlabel("Time (s)")
            plt.ylabel("Bark/BFCC band index")
            plt.title(title)
            plt.tight_layout()
            plt.savefig(path, dpi=150)
            plt.close()

        for i in range(B_eval):
            mel_real_i = mel_real_b[i].detach().cpu().numpy().T  # [F,T]
            mel_hat_i = mel_hat_b[i].detach().cpu().numpy().T

            try:
                psnr_val = bfcc_psnr(mel_real_i, mel_hat_i)
            except Exception as _psnr_err:
                print(
                    f"[BFCC-Forward] PSNR failed for sample {i} at step "
                    f"{global_step_val}: {_psnr_err}"
                )
                psnr_val = None

            base_name = f"step{global_step_val:06d}_sample{i:02d}"
            real_path = os.path.join(
                bfcc_dir,
                base_name + "_bfcc_gt.png",
            )
            hat_path = os.path.join(
                bfcc_dir,
                base_name + "_bfcc_hat.png",
            )
            _save_bfcc_img(
                mel_real_i,
                real_path,
                "Content Bark/BFCC (GT, forward path)",
            )
            _save_bfcc_img(
                mel_hat_i,
                hat_path,
                "Content Bark/BFCC (hat, forward path)",
            )

            if psnr_val is not None:
                print(
                    f"[BFCC-Forward] step={global_step_val} sample={i} "
                    f"PSNR={float(psnr_val):.4f} dB",
                )
    except Exception as _e_bfcc_fwd:
        print(f"[BFCC-Forward] WARNING: forward BFCC eval failed at step {global_step_val}: {_e_bfcc_fwd}")


def _log_train_sample_snr_ber(
    out: Dict[str, torch.Tensor],
    epoch_val: int,
    step_val: int,
    max_samples: int = 10,
) -> None:
    """Print per-sample SNR/BER summary for the JSCC hash channel.

    This is a lightweight, training-time diagnostic that mirrors the
    ``[bit_only_eval][FSK-BER]`` formatting but uses the *internal*
    clean/noisy hash bits from the current batch together with
    ``csi_vec[..., 0]`` as the SNR proxy.

    The function is defensive by design: on any shape/type mismatch it
    simply returns without affecting the training loop.
    """

    try:
        bits_c_clean = out.get("hash_bits_clean", None)
        bits_c_noisy = out.get("hash_bits_noisy", out.get("bits_noisy", None))
        bits_f_clean = out.get("f0_hash_bits_clean", None)
        bits_f_noisy = out.get("f0_hash_bits_noisy", out.get("f0_bits_noisy", None))
        bits_s_clean = out.get("hash_bits_stats", None)
        bits_s_noisy = out.get("hash_bits_stats_noisy", None)

        clean_list: List[torch.Tensor] = []
        noisy_list: List[torch.Tensor] = []

        for b_clean, b_noisy in (
            (bits_c_clean, bits_c_noisy),
            (bits_f_clean, bits_f_noisy),
            (bits_s_clean, bits_s_noisy),
        ):
            if not (
                isinstance(b_clean, torch.Tensor)
                and isinstance(b_noisy, torch.Tensor)
            ):
                continue
            if b_clean.size(0) != b_noisy.size(0):
                continue

            B = int(b_clean.size(0))
            bc = b_clean.detach().reshape(B, -1).to(torch.float32)
            bn = b_noisy.detach().reshape(B, -1).to(torch.float32)
            n = int(min(bc.size(1), bn.size(1)))
            if n <= 0:
                continue

            clean_list.append(bc[:, :n])
            noisy_list.append(bn[:, :n])

        if not clean_list or not noisy_list:
            return

        clean_cat = torch.cat(clean_list, dim=1)
        noisy_cat = torch.cat(noisy_list, dim=1)
        if clean_cat.numel() == 0 or noisy_cat.numel() == 0:
            return

        # Map to {0,1} via sign and compute per-sample BER over all bits.
        xb = (clean_cat > 0).to(torch.int32)
        yb = (noisy_cat > 0).to(torch.int32)
        err_mat = (xb != yb).float()
        ber_per_sample = err_mat.mean(dim=1)  # [B]
        B = int(ber_per_sample.size(0))

        # Optional SNR per sample from csi_vec[..., 0].
        snr_vec: Optional[torch.Tensor] = None
        csi_val = out.get("csi_vec", None)
        if isinstance(csi_val, torch.Tensor):
            csi_t = csi_val.detach().to(torch.float32)
            if csi_t.dim() == 2 and csi_t.size(1) >= 1:
                snr_vec = csi_t[:, 0]

        print(f"[train][FSK-BER] epoch={epoch_val} step={int(step_val)}")
        n_print = min(B, int(max_samples))
        for i in range(n_print):
            ber_i = float(ber_per_sample[i].item())
            if snr_vec is not None and i < int(snr_vec.size(0)):
                snr_i = float(snr_vec[i].item())
                print(f"  sample={i} SNR={snr_i:.1f} dB: BER={ber_i:.6e}")
            else:
                print(f"  sample={i} SNR=N/A dB: BER={ber_i:.6e}")
    except Exception:
        # Diagnostics must never break training.
        return


def run_training(cfg: TrainingConfig) -> None:
    """Main training loop for the public training entrypoint."""

    device = torch.device(cfg.device)
    use_amp = bool(getattr(cfg, "use_amp", False)) and torch.cuda.is_available()

    # Optional: configure JSCC_FSK_BER_TABLE from CLI so that both
    # internal BSC channels (Hash/RVQ bottlenecks) and external
    # bit_only_eval use the same BER(SNR) JSON table.
    fsk_tbl = getattr(cfg, "fsk_ber_table", None)
    if fsk_tbl:
        try:
            os.environ["JSCC_FSK_BER_TABLE"] = os.path.abspath(fsk_tbl)
            print(f"[JSCC] Using BER table JSON from cfg.fsk_ber_table={os.environ['JSCC_FSK_BER_TABLE']}")
        except Exception as _e_tbl:
            print(f"[JSCC] WARNING: failed to set JSCC_FSK_BER_TABLE from fsk_ber_table: {_e_tbl}")

    # Base output dirs: allow ``out_dir`` to override both checkpoint and
    # visualization locations (mirrors v3 script behaviour).
    if cfg.out_dir is not None:
        base = os.path.abspath(cfg.out_dir)
        ckpt_dir = os.path.join(base, "checkpoints")
        viz_dir = os.path.join(base, "viz")
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(viz_dir, exist_ok=True)
        cfg.ckpt_dir = ckpt_dir
        cfg.viz_dir = viz_dir
    else:
        os.makedirs(cfg.ckpt_dir, exist_ok=True)
        os.makedirs(cfg.viz_dir, exist_ok=True)

    dataloader = build_dataloader(cfg)  # type: ignore[arg-type]
    channel_sim = ChannelSimulator(
        sample_rate=16000,
        frame_hz=100,
        snr_step_db=float(getattr(cfg, "snr_step_db", 1.0)),
    )

    # Reuse the robust builder but override a few fields to match our config type
    # by constructing a lightweight view with the attributes it expects.
    class _ShimCfg:
        """Lightweight shim that exposes support-config-like attributes.

        We simply attach attributes from the simplified config so that
        ``build_model`` and ``model_forward`` can operate unchanged.
        """

        pass

    shim = _ShimCfg()
    for k, v in asdict(cfg).items():
        setattr(shim, k, v)

    model: DualBranchBarkJSCC = build_model(shim)  # type: ignore[arg-type]
    model.to(device)

    # Ensure the shim config's with_hash flag matches the actually
    # constructed model so that helper utilities such as
    # ``model_forward`` (used in --only_eval / pipeline_probe
    # modes) pick the correct forward path (with or without
    # hash/RVQ) even if CLI flags differ from the checkpoint
    # configuration.
    try:
        setattr(shim, "with_hash", bool(getattr(model, "with_hash", getattr(shim, "with_hash", False))))
    except Exception:
        pass

    # Optional: enable the learned energy calibration head inside
    # DualBranchBarkJSCC when requested via CLI/config. This head
    # predicts a per-sample c0 offset from mel_used statistics and
    # does not rely on GT FARGAN ceps, so training and inference
    # share the same energy calibration path.
    if bool(getattr(cfg, "use_learned_energy_calib", False)) and hasattr(
        model, "use_learned_energy_calib"
    ):
        model.use_learned_energy_calib = True

    # Optional: ignore stats bits when reconstructing mel energy in
    # forward_with_hash / decode_from_bits_offline. This flag is wired
    # directly onto the model so that both training and offline decode
    # paths see the same behaviour.
    if bool(getattr(cfg, "ignore_stats_in_mel", False)):
        setattr(model, "ignore_stats_in_mel", True)

    # Sync FARGAN runtime gating knobs from CLI/config onto the actual
    # vocoder instance so this script does not need environment variables
    # for strict VUV / final voicing / silence gating experiments.
    try:
        core = getattr(getattr(model, "vocoder", None), "fargan_core", None)
        sig_net = getattr(core, "sig_net", None)
        if core is not None:
            core.strict_vuv_gate = bool(getattr(cfg, "vocoder_strict_vuv_gate", True))
            core._strict_vuv_gate_set_from_cli = True
            core.collect_internal_tracks = bool(getattr(cfg, "viz_vocoder_internals", True))
            core.silence_energy_thr_db = float(
                getattr(cfg, "vocoder_silence_energy_thr_db", -40.0)
            )
            core.silence_gate_width_db = float(
                getattr(cfg, "vocoder_silence_gate_width_db", 6.0)
            )
        if sig_net is not None:
            sig_net.final_voicing_gate = bool(
                getattr(cfg, "vocoder_final_voicing_gate", True)
            )
            sig_net.final_voicing_gate_floor = float(
                getattr(cfg, "vocoder_final_voicing_gate_floor", 0.0)
            )
            sig_net.final_voicing_gate_gamma = float(
                getattr(cfg, "vocoder_final_voicing_gate_gamma", 1.0)
            )
            sig_net.silence_gate_enabled = bool(
                getattr(cfg, "vocoder_silence_gate", True)
            )
            sig_net.silence_gate_floor = float(
                getattr(cfg, "vocoder_silence_gate_floor", 0.0)
            )
            sig_net.pitch_gain_scale = float(
                getattr(cfg, "vocoder_pitch_gain_scale", 1.0)
            )
            sig_net.sig_core_scale = float(
                getattr(cfg, "vocoder_sig_core_scale", 1.0)
            )
        if hasattr(model, "oracle_swap_source_controls"):
            model.oracle_swap_source_controls = str(
                getattr(cfg, "oracle_swap_source_controls", "none")
            )
        else:
            setattr(
                model,
                "oracle_swap_source_controls",
                str(getattr(cfg, "oracle_swap_source_controls", "none")),
            )
        if core is not None:
            print(
                "[FARGAN] Gates "
                f"strict_vuv={bool(getattr(core, 'strict_vuv_gate', True))} "
                f"final_voicing={bool(getattr(sig_net, 'final_voicing_gate', True)) if sig_net is not None else 'n/a'} "
                f"silence_gate={bool(getattr(sig_net, 'silence_gate_enabled', True)) if sig_net is not None else 'n/a'} "
                f"pitch_gain_scale={float(getattr(sig_net, 'pitch_gain_scale', 1.0)) if sig_net is not None else float('nan'):.3f} "
                f"sig_core_scale={float(getattr(sig_net, 'sig_core_scale', 1.0)) if sig_net is not None else float('nan'):.3f} "
                f"sil_thr_db={float(getattr(core, 'silence_energy_thr_db', -40.0)):.1f} "
                f"sil_width_db={float(getattr(core, 'silence_gate_width_db', 6.0)):.1f} "
                f"oracle_swap={str(getattr(model, 'oracle_swap_source_controls', 'none'))}"
            )
    except Exception as _e_fgcfg:
        print(f"[FARGAN] WARNING: failed to sync gate config: {_e_fgcfg}")

    # Optionally freeze vocoder parameters for the entire training run. This
    # mirrors the ``freeze_vocoder_all`` behaviour in the full Stage2.5
    # script and is useful when focusing purely on the JSCC front-end.
    if bool(getattr(cfg, "freeze_vocoder_all", False)):
        try:
            if hasattr(model, "vocoder"):
                for _p in model.vocoder.parameters():
                    _p.requires_grad = False
                print("[Vocoder] Freezing vocoder for ALL steps (freeze_vocoder_all=True)")
        except Exception as _e_vf:
            print(f"[Vocoder] WARNING: failed to freeze vocoder: {_e_vf}")

    # Optionally freeze the entire content JSCC branch for all training
    # steps, including VMamba encoder/decoder, content VQ/hash, content
    # predictors, L2H stack and stats bottleneck.  This is intended for
    # the second-stage training where you want to lock in everything
    # learned during the content-only phase (content_vmamba + RVQ +
    # L2H + hash_content_stats) and only update F0/vocoder or other
    # non-content components.
    if bool(getattr(cfg, "freeze_content_all", False)):
        try:
            prefixes = (
                "content_",
                "hash_content",
                "content_pred_",
                "deco_l2h_refiner",
                "l2h_flow",
                "hash_content_stats",
                "mel18_to_ceps",
                "hf2ceps",
            )
            n_frozen = 0
            for name, param in model.named_parameters():
                if not isinstance(param, torch.nn.Parameter):
                    continue
                if any(name.startswith(pfx) for pfx in prefixes):
                    param.requires_grad = False
                    n_frozen += 1
            print(f"[Content] Freezing entire content branch (freeze_content_all=True); frozen_params={n_frozen}")
        except Exception as _e_cf:
            print(f"[Content] WARNING: failed to freeze content branch: {_e_cf}")

        # Optional: keep BFCC→ceps mapping trainable while freezing the
        # rest of the content branch. When ``--unfreeze_ceps_map`` is set,
        # we selectively re-enable gradients for ``mel18_to_ceps`` and
        # ``hf2ceps`` so that the mel/BFCC→ceps adapter can adapt to the
        # current GAN/F0 regime without touching JSCC content bits.
        if bool(getattr(cfg, "unfreeze_ceps_map", False)):
            try:
                n_unfrozen = 0
                for name, param in model.named_parameters():
                    if not isinstance(param, torch.nn.Parameter):
                        continue
                    if name.startswith("mel18_to_ceps") or name.startswith("hf2ceps"):
                        param.requires_grad = True
                        n_unfrozen += 1
                print(
                    "[Content] Unfreezing BFCC→ceps mapping "
                    f"(mel18_to_ceps/hf2ceps); params={n_unfrozen}"
                )
            except Exception as _e_unf:
                print(f"[Content] WARNING: failed to unfreeze ceps mapping: {_e_unf}")

        # Optional: keep the L2H stack trainable while freezing the
        # rest of the content branch. When ``--unfreeze_l2h`` is set,
        # we selectively re-enable gradients for ``deco_l2h_refiner``
        # and ``l2h_flow`` so that the harmonic/noise L2H refiner can
        # continue to adapt to the current GAN/F0 regime without
        # changing the JSCC content encoder/decoder.
        if bool(getattr(cfg, "unfreeze_l2h", False)):
            try:
                n_unfrozen_l2h = 0
                for name, param in model.named_parameters():
                    if not isinstance(param, torch.nn.Parameter):
                        continue
                    if name.startswith("deco_l2h_refiner") or name.startswith("l2h_flow"):
                        param.requires_grad = True
                        n_unfrozen_l2h += 1
                print(
                    "[Content] Unfreezing L2H stack "
                    f"(deco_l2h_refiner/l2h_flow); params={n_unfrozen_l2h}"
                )
            except Exception as _e_unf_l2h:
                print(f"[Content] WARNING: failed to unfreeze L2H stack: {_e_unf_l2h}")

        # Optional: keep the stats bottleneck trainable while freezing the
        # rest of the content branch. When ``--unfreeze_stats`` is set,
        # we selectively re-enable gradients for ``hash_content_stats`` so
        # that the mean/std codec can continue to adapt without touching
        # content_vmamba/hash_content or the BFCC->ceps/L2H stacks.
        if bool(getattr(cfg, "unfreeze_stats", False)):
            try:
                n_unfrozen_stats = 0
                for name, param in model.named_parameters():
                    if not isinstance(param, torch.nn.Parameter):
                        continue
                    if name.startswith("hash_content_stats"):
                        param.requires_grad = True
                        n_unfrozen_stats += 1
                print(
                    "[Content] Unfreezing stats bottleneck "
                    f"(hash_content_stats); params={n_unfrozen_stats}"
                )
            except Exception as _e_unf_stats:
                print(f"[Content] WARNING: failed to unfreeze stats bottleneck: {_e_unf_stats}")

    # In content-only runs (without freeze_content_all), when L2H is
    # explicitly enabled via ``--with_l2h``, treat the L2H stack as
    # part of the "content" branch and ensure its parameters remain
    # trainable.  When freeze_content_all=True we intentionally do not
    # override the freeze behaviour so that previously learned L2H
    # weights stay fixed.
    if (
        bool(getattr(cfg, "content_only", False))
        and bool(getattr(cfg, "with_l2h", False))
        and not bool(getattr(cfg, "freeze_content_all", False))
    ):
        try:
            for name, param in model.named_parameters():
                if not isinstance(param, torch.nn.Parameter):
                    continue
                if name.startswith("deco_l2h_refiner") or name.startswith("l2h_flow"):
                    param.requires_grad = True
        except Exception as _e_l2h:
            print(f"[ContentOnly] WARNING: failed to ensure L2H params trainable: {_e_l2h}")

    # Debug: 打印 hash_content_stats 子模块的 requires_grad 标志，便于确认在
    # freeze_content_all 模式下 stats 瓶颈是否仍然可训练。仅当 DBG_STATS=1
    # 时启用，避免在正常训练中产生过多日志。
    try:
        import os as _os_dbg

        if _os_dbg.environ.get("DBG_STATS", "0") == "1":
            try:
                print("[DBG] hash_content_stats parameters (name, requires_grad):")
                for name, param in model.named_parameters():
                    if "hash_content_stats" in name:
                        print(f"  {name}: {getattr(param, 'requires_grad', None)}")
            except Exception as _e_dbg:
                print(f"[DBG] WARNING: failed to inspect hash_content_stats params: {_e_dbg}")
    except Exception:
        pass

    # Default optimizer for full JSCC training
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)

    # Stats-only fine-tuning mode: only train hash_content_stats to
    # reconstruct (mel_mean, mel_std) in the normalised space, then exit.
    if bool(getattr(cfg, "stats_only", False)):
        try:
            # Freeze all parameters except hash_content_stats.*
            stats_params: list[torch.nn.Parameter] = []
            for name, param in model.named_parameters():
                if not isinstance(param, torch.nn.Parameter):
                    continue
                if name.startswith("hash_content_stats"):
                    param.requires_grad = True
                    stats_params.append(param)
                else:
                    param.requires_grad = False

            if not stats_params:
                print("[StatsOnly] WARNING: no hash_content_stats parameters found; aborting stats-only mode")
                return

            stats_lr = float(getattr(cfg, "stats_only_lr", cfg.lr))
            optimizer = torch.optim.AdamW(stats_params, lr=stats_lr, weight_decay=0.0)
            print(
                f"[StatsOnly] Enabled stats-only fine-tuning (params={len(stats_params)}, "
                f"lr={stats_lr}, max_steps={int(getattr(cfg, 'stats_only_max_steps', 10000))})"
            )

            model.train()
            max_steps = int(getattr(cfg, "stats_only_max_steps", 10000) or 10000)
            global_step = 0
            num_epochs = int(getattr(cfg, "num_epochs", 1))
            for epoch in range(num_epochs):
                for batch in dataloader:
                    if global_step >= max_steps:
                        print(f"[StatsOnly] Reached max_steps={max_steps}; exiting stats-only mode")
                        # Save a stats-only checkpoint before exiting.
                        try:
                            ckpt_path = os.path.join(
                                cfg.ckpt_dir,
                                f"stats_only_step_{global_step:06d}_epoch_{epoch:02d}.pth",
                            )
                            state = {
                                "model_state_dict": model.state_dict(),
                                "cfg": asdict(cfg),
                                "global_step": global_step,
                                "epoch": epoch,
                            }
                            torch.save(state, ckpt_path)
                            print(f"[StatsOnly] Saved checkpoint to {ckpt_path}")
                        except Exception as _e_ck:
                            print(f"[StatsOnly] WARNING: failed to save checkpoint: {_e_ck}")
                        return

                    audio = batch.get("audio")
                    if not isinstance(audio, torch.Tensor):
                        continue
                    audio = audio.to(device)

                    optimizer.zero_grad(set_to_none=True)

                    # wave -> Bark/BFCC -> (mel_mean, mel_std)
                    mel = model.wave_to_mel(audio)  # [B,T,F]
                    Bm, Tm, Fm = mel.shape
                    mel_mean = mel.mean(dim=(1, 2), keepdim=True)  # [B,1,1]
                    mel_std = mel.std(dim=(1, 2), keepdim=True).clamp(min=0.1)  # [B,1,1]

                    # 归一化到 stats 空间：mean_norm / log_std_norm
                    mean_center = -5.0
                    mean_scale = 2.0
                    std_log = torch.log(mel_std)
                    std_center = 0.8
                    std_scale = 0.8

                    mean_norm = (mel_mean - mean_center) / mean_scale
                    std_norm = (std_log - std_center) / std_scale
                    stats_target = torch.cat([mean_norm, std_norm], dim=-1).view(Bm, 1, 2)  # [B,1,2]

                    hb_stats = model.hash_content_stats(stats_target, channel_params=None, mask=None)
                    stats_hat = hb_stats["reconstructed"].view(Bm, 1, 2)

                    # L1 loss in normalised (mean_norm, log_std_norm) space
                    loss_stats = torch.mean(torch.abs(stats_hat - stats_target))
                    loss_stats.backward()
                    optimizer.step()

                    global_step += 1

                    if global_step % int(getattr(cfg, "log_every_steps", 10) or 10) == 0:
                        print(
                            f"[StatsOnly] epoch={epoch} step={global_step} "
                            f"loss_stats={float(loss_stats.detach().item()):.4f}"
                        )

            # Normal completion without hitting max_steps: save a final stats-only checkpoint.
            try:
                ckpt_path = os.path.join(
                    cfg.ckpt_dir,
                    f"stats_only_step_{global_step:06d}_epoch_{(num_epochs-1):02d}.pth",
                )
                state = {
                    "model_state_dict": model.state_dict(),
                    "cfg": asdict(cfg),
                    "global_step": global_step,
                    "epoch": num_epochs - 1,
                }
                torch.save(state, ckpt_path)
                print(f"[StatsOnly] Saved checkpoint to {ckpt_path}")
            except Exception as _e_ck:
                print(f"[StatsOnly] WARNING: failed to save checkpoint: {_e_ck}")
            return
        except Exception as _e_stats_only:
            print(f"[StatsOnly] WARNING: stats-only mode failed: {_e_stats_only}")
            # Fall through to full JSCC training if stats-only fails

    # Optional: keep content JSCC branch frozen for a warm-up period so that
    # only non-content components (F0/vocoder/etc.) are updated in the early
    # steps. This is particularly useful when reloading content from a
    # separate checkpoint via --content_ckpt/--reload_content_after_resume.
    content_warmup_steps = int(getattr(cfg, "content_warmup_steps", 0) or 0)
    freeze_content_all_enabled = bool(getattr(cfg, "freeze_content_all", False))
    # Interpret content_warmup_steps as "steps since (re)start" rather than
    # absolute global_step, so that after a resume it always counts from 0.
    # We track a local counter content_steps_since_resume starting at 0.
    content_steps_since_resume = 0
    # ``freeze_content_all`` means "freeze for the entire run", so it must
    # take precedence over the temporary content warm-up logic.
    content_frozen = (content_warmup_steps > 0) and not freeze_content_all_enabled
    if freeze_content_all_enabled and content_warmup_steps > 0:
        print(
            "[Content-Warmup] Ignoring content_warmup_steps because "
            "freeze_content_all=True"
        )

    # Local GAN warm-up counter: counts steps since the current training
    # command started or since the last resume, so that
    # ``gan_adv_warmup_steps`` behaves like "N steps after this run
    # begins" rather than comparing directly against the absolute
    # global_step stored in the checkpoint.
    gan_steps_since_resume = 0

    def _set_content_requires_grad(flag: bool) -> None:
        if not hasattr(model, "named_parameters"):
            return
        prefixes = (
            "content_",
            "hash_content",
            "hash_content_stats",
            "content_pred_",
        )
        for name, param in model.named_parameters():
            if not isinstance(param, torch.nn.Parameter):
                continue
            if any(name.startswith(pfx) for pfx in prefixes):
                # Keep the very last projection head of the content branch
                # (content_vmamba.decoder.head.*) trainable even during
                # warm-up. In simplified Stage2.5 this head often uses a
                # different shape from older checkpoints, so its weights
                # start from the current architecture initialisation rather
                # than from a prior run. Freezing it would effectively lock
                # in a random projection for the duration of the warm-up,
                # which is undesirable.
                if name.startswith("content_vmamba.decoder.head."):
                    continue
                param.requires_grad = flag

    # Optional VQ-only training mode: freeze all parameters except the
    # RVQ/Hash bottlenecks (hash_content/hash_f0vuv) so that only codebooks
    # are updated. When enabled, this overrides content_warmup behaviour.
    vq_only_mode = bool(getattr(cfg, "vq_only_train", False)) and not bool(getattr(cfg, "freeze_content_all", False))

    def _apply_vq_only_freeze() -> None:
        if not hasattr(model, "named_parameters"):
            return
        for name, param in model.named_parameters():
            if not isinstance(param, torch.nn.Parameter):
                continue
            if name.startswith("hash_content") or name.startswith("hash_f0vuv"):
                param.requires_grad = True
            else:
                param.requires_grad = False

    if vq_only_mode:
        print("[VQ-Only] Freezing all parameters except hash_content/hash_f0vuv (VQ bottlenecks)")
        _apply_vq_only_freeze()
        # In VQ-only mode we effectively disable content warm-up, since
        # non-VQ content parameters are permanently frozen.
        content_frozen = False
        content_steps_since_resume = 0
    elif content_frozen:
        print(
            f"[Content-Warmup] Freezing content branch for the first "
            f"{content_warmup_steps} step(s) after start/resume"
        )
        _set_content_requires_grad(False)

    # Optional adversarial raw-waveform discriminators
    # (MPD + MSD + BarkHF + F0Period).
    mpd: Optional[torch.nn.Module] = None
    msd: Optional[torch.nn.Module] = None
    bark_disc: Optional[torch.nn.Module] = None
    f0p_disc: Optional[torch.nn.Module] = None
    wave_to_bfcc_adv: Optional[WaveToBFCC] = None
    optimizer_hifi_disc: Optional[torch.optim.Optimizer] = None
    try:
        lam_hifi_adv = float(getattr(cfg, "lambda_gan_adv", 0.0))
        lam_hifi_fm = float(getattr(cfg, "lambda_gan_fm", 0.0))
        if lam_hifi_adv > 0.0 or lam_hifi_fm > 0.0:
            mpd = MultiPeriodDiscriminator().to(device)
            msd = MultiScaleDiscriminator().to(device)
            bark_disc = BarkHFDiscriminator().to(device)
            f0p_disc = F0PeriodDiscriminator().to(device)

            # Shared BFCC/Bark front-end for adversarial training.
            wave_to_bfcc_adv = WaveToBFCC(
                sample_rate=16000,
                n_fft=400,
                hop_length=160,
                n_bands=32,
            ).to(device)
            wave_to_bfcc_adv.eval()
            for p in wave_to_bfcc_adv.parameters():
                p.requires_grad_(False)

            lr_hifi = float(getattr(cfg, "gan_disc_lr", cfg.lr))
            optimizer_hifi_disc = torch.optim.Adam(
                list(mpd.parameters())
                + list(msd.parameters())
                + list(bark_disc.parameters())
                + list(f0p_disc.parameters()),
                lr=lr_hifi,
                betas=(0.8, 0.99),
            )
            print(f"[GAN-ADV] Enabled MPD+MSD+BarkHF+F0Period discriminators (lr={lr_hifi})")
    except Exception as _e_hifi_init:
        print(f"[GAN-ADV] WARNING: failed to init discriminators: {_e_hifi_init}")
        mpd = None
        msd = None
        bark_disc = None
        f0p_disc = None
        wave_to_bfcc_adv = None
        optimizer_hifi_disc = None

    # Optional resume: restore model/optimizer/global_step/epoch when a
    # checkpoint path is provided. We support checkpoints produced by both
    # this simplified script and the main Stage2.5 script as long as the
    # underlying model structure matches.
    start_epoch = 0
    global_step = 0
    if cfg.resume is not None and cfg.resume != "":
        if not os.path.isfile(cfg.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {cfg.resume}")
        print(f"[Simplified-Resume] Loading checkpoint from {cfg.resume}")
        ckpt = torch.load(cfg.resume, map_location=device)

        # Model state: try several common keys for compatibility
        sd = None
        for key in ("model_state_dict", "model", "state_dict"):
            if key in ckpt:
                sd = ckpt[key]
                break
        shape_mismatch_on_resume = False
        if sd is not None:
            # Shape-safe load: drop keys whose shapes do not match.
            cur_sd = model.state_dict()
            safe_sd = {}
            dropped = []
            for k, v in sd.items():
                if k in cur_sd and isinstance(v, torch.Tensor) and v.shape == cur_sd[k].shape:
                    safe_sd[k] = v
                else:
                    dropped.append(k)
            if dropped:
                shape_mismatch_on_resume = True
                print(f"[Simplified-Resume] Dropped {len(dropped)} key(s) due to shape mismatch (e.g. {dropped[:5]})")
            load_ret = model.load_state_dict(safe_sd, strict=False)
            try:
                missing = list(getattr(load_ret, "missing_keys", []))
                unexpected = list(getattr(load_ret, "unexpected_keys", []))
            except Exception:
                missing, unexpected = [], []
            if missing or unexpected:
                print(f"[Simplified-Resume] Non-strict load: missing={len(missing)}, unexpected={len(unexpected)}")
        else:
            print("[Simplified-Resume] WARNING: no compatible model key found in checkpoint")

        # Optimizer state (optional)。若检测到模型形状有变（例如启用了
        # 新的 L2H 分支或调整了其维度），为避免 Adam 内部 exp_avg 与
        # grads 的数量不一致，直接放弃旧的 optimizer state，使用新
        # optimizer；否则尝试正常加载旧 state。
        try:
            opt_sd = ckpt.get("optimizer_state_dict", ckpt.get("optimizer", None))
            if opt_sd is not None and not shape_mismatch_on_resume:
                optimizer.load_state_dict(opt_sd)
            elif opt_sd is not None and shape_mismatch_on_resume:
                print("[Simplified-Resume] Skipping optimizer state load due to shape mismatch; using fresh optimizer")
                optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)
        except Exception as e:
            print(f"[Simplified-Resume] WARNING: failed to load optimizer state: {e}; using fresh optimizer")
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)

        # When resuming, prefer the checkpoint's own config for
        # structural flags such as ``with_hash`` so that eval
        # modes (e.g., --only_eval) run a forward path that
        # matches how the checkpoint was trained.
        try:
            cfg_saved = ckpt.get("cfg", None)
            if isinstance(cfg_saved, dict) and "with_hash" in cfg_saved:
                saved_with_hash = bool(cfg_saved.get("with_hash"))
                cfg.with_hash = saved_with_hash
                try:
                    setattr(shim, "with_hash", saved_with_hash)
                except Exception:
                    pass
                try:
                    if hasattr(model, "with_hash"):
                        setattr(model, "with_hash", saved_with_hash)
                except Exception:
                    pass
        except Exception:
            # Config alignment should never break resume.
            pass

        global_step = int(ckpt.get("global_step", 0))
        start_epoch = int(ckpt.get("epoch", 0))
        print(f"[Simplified-Resume] start_epoch={start_epoch}, global_step={global_step}")

        # Optional: resume adversarial discriminators when enabled and present.
        try:
            if mpd is not None and "mpd" in ckpt:
                mpd.load_state_dict(ckpt["mpd"])
            if msd is not None and "msd" in ckpt:
                msd.load_state_dict(ckpt["msd"])
            if bark_disc is not None and "bark_disc" in ckpt:
                bark_disc.load_state_dict(ckpt["bark_disc"])
            if f0p_disc is not None and "f0p_disc" in ckpt:
                f0p_disc.load_state_dict(ckpt["f0p_disc"])
            if optimizer_hifi_disc is not None and "optimizer_hifi_disc" in ckpt:
                optimizer_hifi_disc.load_state_dict(ckpt["optimizer_hifi_disc"])
        except Exception as _e_hifi_resume:
            print(f"[GAN-ADV] WARNING: failed to load discriminator states at resume: {_e_hifi_resume}")

        # Optionally re-load FARGAN vocoder weights after resume, overriding
        # the student's vocoder core. This mirrors the behaviour of the full
        # Stage2.5 script and is useful when resuming from a JSCC checkpoint
        # but still wanting to keep a fixed FARGAN baseline.
        if cfg.vocoder_ckpt and bool(getattr(cfg, "reload_vocoder_after_resume", False)):
            try:
                if os.path.isfile(cfg.vocoder_ckpt):
                    print(f"[Vocoder] Re-loading vocoder ckpt after resume: {cfg.vocoder_ckpt}")
                    try:
                        _ck = torch.load(cfg.vocoder_ckpt, map_location="cpu", weights_only=True)
                    except TypeError:
                        _ck = torch.load(cfg.vocoder_ckpt, map_location="cpu")
                    _sd = _ck["state_dict"] if isinstance(_ck, dict) and "state_dict" in _ck else _ck
                    load_ret = model.vocoder.fargan_core.load_state_dict(_sd, strict=False)
                    _missing = list(getattr(load_ret, "missing_keys", []))
                    _unexpected = list(getattr(load_ret, "unexpected_keys", []))
                    print(
                        f"[Vocoder] FARGAN weights re-loaded (strict=False). "
                        f"missing={len(_missing)}, unexpected={len(_unexpected)}"
                    )
                else:
                    print(f"[Vocoder] WARNING: vocoder_ckpt not found: {cfg.vocoder_ckpt}")
            except Exception as _e_fargan:
                print(f"[Vocoder] WARNING: failed re-load FARGAN after resume: {_e_fargan}")

        # Optionally re-load *content* JSCC branch (VMamba + content bottlenecks)
        # from a separate Stage2.5 checkpoint after the main resume. This is
        # useful when you want to pick a particular content encoder/decoder
        # from another run (e.g., stage3_freq8_ds4_fresh) while keeping F0,
        # vocoder, etc. from the primary --resume checkpoint.
        if bool(getattr(cfg, "reload_content_after_resume", False)):
            ckpt_content_path = getattr(cfg, "content_ckpt", None)
            if not ckpt_content_path:
                print("[Content-Reload] WARNING: reload_content_after_resume=True but cfg.content_ckpt is empty; skipping content reload")
            elif not os.path.isfile(ckpt_content_path):
                print(f"[Content-Reload] WARNING: content_ckpt not found: {ckpt_content_path}; skipping content reload")
            else:
                try:
                    print(f"[Content-Reload] Loading content branch from {ckpt_content_path}")
                    _ck2 = torch.load(ckpt_content_path, map_location=device)
                    sd2 = None
                    for key in ("model_state_dict", "model", "state_dict"):
                        if isinstance(_ck2, dict) and key in _ck2:
                            sd2 = _ck2[key]
                            break
                    if sd2 is None and isinstance(_ck2, dict):
                        # Fallback: treat dict itself as state_dict when values are tensors.
                        if all(isinstance(v, torch.Tensor) for v in _ck2.values()):
                            sd2 = _ck2

                    if sd2 is None:
                        print("[Content-Reload] WARNING: no model state_dict found in content_ckpt; skipping")
                    else:
                        cur_sd = model.state_dict()
                        safe_sd = {}
                        dropped_c: list[str] = []
                        for k, v in sd2.items():
                            if not isinstance(v, torch.Tensor):
                                continue
                            # Restrict reload to clearly content-related modules.
                            if not (
                                k.startswith("content_")
                                or k.startswith("hash_content")
                                or k.startswith("content_pred_")
                                or k.startswith("hash_content_stats")
                            ):
                                continue
                            if k in cur_sd and v.shape == cur_sd[k].shape:
                                safe_sd[k] = v
                            else:
                                dropped_c.append(k)

                        if not safe_sd:
                            print("[Content-Reload] WARNING: no matching content keys to reload (all mismatched or absent)")
                        else:
                            load_ret_c = model.load_state_dict(safe_sd, strict=False)
                            missing_c = list(getattr(load_ret_c, "missing_keys", []))
                            unexpected_c = list(getattr(load_ret_c, "unexpected_keys", []))
                            print(
                                f"[Content-Reload] Re-loaded {len(safe_sd)} content key(s) from {ckpt_content_path}; "
                                f"dropped={len(dropped_c)}, missing={len(missing_c)}, unexpected={len(unexpected_c)}"
                            )
                except Exception as _e_cont:
                    print(f"[Content-Reload] WARNING: failed to reload content branch from {ckpt_content_path}: {_e_cont}")

    # Optional: initialize wandb logging
    use_wandb = False
    corr_window: Optional[MetricCorrWindow] = None
    try:
        if bool(getattr(cfg, "use_wandb", False)) and wandb is not None:
            project = getattr(cfg, "wandb_project", None) or "DBP-JSCC"
            run_name = getattr(cfg, "wandb_run_name", None)
            wandb.init(project=project, name=run_name, config=asdict(cfg))
            use_wandb = True
            # Only allocate the correlation window when wandb is enabled.
            corr_window = MetricCorrWindow(window=2000, log_every=max(100, int(cfg.wandb_log_freq)))
            print(f"[wandb] Initialized: project='{project}', run='{run_name}'")
        elif bool(getattr(cfg, "use_wandb", False)) and wandb is None:
            print("[wandb] WARNING: wandb is not installed, disabling wandb logging")
    except Exception as _w:
        print(f"[wandb] WARNING: failed to initialize wandb: {_w}")
        use_wandb = False

    print("[Simplified] Starting Stage2.5 training with config:")
    print(asdict(cfg))

    def _debug_rvq_latent_stats(
        out_dict: Dict[str, torch.Tensor],
        epoch_val: int,
        step_val: int,
        prefix: str,
    ) -> None:
        """Print RVQ latent before/after-channel stats when DBG_RVQ_LATENT=1.

        主要用于快速检查 RVQ 信道前后的连续特征分布是否异常：
        - 内容分支:  z_content vs z_content_hat
        - F0 分支:   z_f0      vs z_f0_hat
        """

        if os.environ.get("DBG_RVQ_LATENT", "0") != "1":
            return

        def _stat(x: Optional[torch.Tensor]) -> Optional[Dict[str, float]]:
            if not isinstance(x, torch.Tensor) or x.numel() == 0:
                return None
            x_det = x.detach().to(torch.float32)
            return {
                "mean": float(x_det.mean().item()),
                "std": float(x_det.std().item()),
                "min": float(x_det.min().item()),
                "max": float(x_det.max().item()),
            }

        z_c = out_dict.get("z_content", None)
        z_ch = out_dict.get("z_content_hat", None)
        z_f = out_dict.get("z_f0", None)
        z_fh = out_dict.get("z_f0_hat", None)

        s_c = _stat(z_c)
        s_ch = _stat(z_ch)
        s_cd = _stat((z_ch - z_c) if isinstance(z_c, torch.Tensor) and isinstance(z_ch, torch.Tensor) else None)
        s_f = _stat(z_f)
        s_fh = _stat(z_fh)
        s_fd = _stat((z_fh - z_f) if isinstance(z_f, torch.Tensor) and isinstance(z_fh, torch.Tensor) else None)

        csi_val = out_dict.get("csi_vec", None)
        snr_mean: Optional[float] = None
        if isinstance(csi_val, torch.Tensor) and csi_val.dim() >= 2 and csi_val.size(-1) >= 1:
            try:
                # 允许 [B,T,4] 或 [B,4]；统一在 batch 和时间维上取均值
                c_det = csi_val.detach().to(torch.float32)
                if c_det.dim() == 3:
                    snr_mean = float(c_det[..., 0].mean().item())
                elif c_det.dim() == 2:
                    snr_mean = float(c_det[:, 0].mean().item())
            except Exception:
                snr_mean = None

        def _fmt(d: Optional[Dict[str, float]]) -> str:
            if d is None:
                return "mean=N/A std=N/A min=N/A max=N/A"
            return (
                f"mean={d['mean']:+.3e} std={d['std']:+.3e} "
                f"min={d['min']:+.3e} max={d['max']:+.3e}"
            )

        snr_str = f" snr_db~{snr_mean:.2f}" if snr_mean is not None else " snr_db~N/A"
        print(f"[RVQ-LATENT]{prefix} epoch={epoch_val} step={step_val}{snr_str} content_clean:  {_fmt(s_c)}")
        print(f"[RVQ-LATENT]{prefix} epoch={epoch_val} step={step_val}{snr_str} content_noisy:  {_fmt(s_ch)}")

    def _debug_stats_hash(
        model_stats: DualBranchBarkJSCC,
        out_dict: Dict[str, torch.Tensor],
        epoch_val: int,
        step_val: int,
        prefix: str,
    ) -> None:
        """Lightweight stats-bits collapse diagnostics when DBG_STATS=1.

        打印三类信息，方便快速判断 hash_content_stats 是否塌缩：

        - [STATS][gt]   : GT mel_mean/mel_std 在 batch 内的分布；
        - [STATS][bits] : stats bits 在 batch 内的 unique code 数与 bit 均值；
        - [STATS][rand] : 随机 bits 经 hash_content_stats.decode_hash_codec 解码后
                          的输出在 batch 维度上的 std，用于粗判 decoder 是否依赖 bits。

        所有逻辑都包在 try/except 中，确保在任何异常情况下都不会打断训练。
        """

        if os.environ.get("DBG_STATS", "0") != "1":
            return

        try:
            import torch as _torch  # local alias for type checks

            B: Optional[int] = None

            # 1) GT stats: 优先使用 out_dict 中的 mel_mean/mel_std，若缺失则从 mel 现算。
            mel_any = out_dict.get("mel", None)
            mm_any = out_dict.get("mel_mean", None)
            ms_any = out_dict.get("mel_std", None)

            if isinstance(mm_any, _torch.Tensor) and isinstance(ms_any, _torch.Tensor):
                mm = mm_any.detach().view(mm_any.size(0), -1)
                ms = ms_any.detach().view(ms_any.size(0), -1)
                B = int(mm.size(0))
            elif isinstance(mel_any, _torch.Tensor) and mel_any.dim() == 3:
                mel_t = mel_any.detach()
                B = int(mel_t.size(0))
                mm = mel_t.mean(dim=(1, 2), keepdim=True).view(B, -1)
                ms = mel_t.std(dim=(1, 2), keepdim=True).view(B, -1)
            else:
                mm = None
                ms = None

            def _s(x: Optional[_torch.Tensor]) -> str:
                if not isinstance(x, _torch.Tensor) or x.numel() == 0:
                    return "mean=N/A std=N/A min=N/A max=N/A"
                x_det = x.detach().to(_torch.float32)
                return (
                    f"mean={float(x_det.mean().item()):+.4f} "
                    f"std={float(x_det.std().item()):+.4f} "
                    f"min={float(x_det.min().item()):+.4f} "
                    f"max={float(x_det.max().item()):+.4f}"
                )

            if mm is not None and ms is not None:
                print(
                    f"[STATS][gt]{prefix} epoch={epoch_val} step={step_val} "
                    f"mel_mean={_s(mm)} | mel_std={_s(ms)}"
                )

            # 2) stats bits 使用情况：batch 内 unique code 数量 + bit 均值。
            bits_s = out_dict.get("hash_bits_stats", None)
            if isinstance(bits_s, _torch.Tensor) and bits_s.numel() > 0:
                # 兼容 [B,T,K] / [B,1,K] / [B,K]
                if bits_s.dim() == 3:
                    B = int(bits_s.size(0))
                    bits_flat = bits_s.detach().reshape(B, -1)
                elif bits_s.dim() == 2:
                    B = int(bits_s.size(0))
                    bits_flat = bits_s.detach()
                else:
                    bits_flat = bits_s.detach().view(1, -1)
                    B = int(bits_flat.size(0))

                unique_codes = int(_torch.unique(bits_flat, dim=0).shape[0])
                bit_mean = bits_flat.to(_torch.float32).mean(dim=0).cpu().tolist()
                print(
                    f"[STATS][bits]{prefix} epoch={epoch_val} step={step_val} "
                    f"unique_codes={unique_codes} bit_mean={bit_mean}"
                )

                # 3) 随机 bits 解码：检查 decoder 是否真的对 bits 敏感。
                hb_stats_mod = getattr(model_stats, "hash_content_stats", None)
                if hb_stats_mod is not None:
                    stats_hat_rand = None
                    # HashBottleneck / GroupedHashBottleneck: decode_hash_codec(bits)
                    if hasattr(hb_stats_mod, "decode_hash_codec") and callable(getattr(hb_stats_mod, "decode_hash_codec")):
                        bits_rand_01 = _torch.randint(0, 2, bits_flat.shape, device=bits_flat.device)
                        stats_hat_rand = hb_stats_mod.decode_hash_codec(bits_rand_01)
                    # RVQBottleneck: decode_bits(bits_sign) 接受 ±1 比特
                    elif hasattr(hb_stats_mod, "decode_bits") and callable(getattr(hb_stats_mod, "decode_bits")):
                        bits_rand_sign = _torch.randint(0, 2, bits_flat.shape, device=bits_flat.device)
                        bits_rand_sign = bits_rand_sign.to(_torch.float32) * 2.0 - 1.0
                        if bits_rand_sign.dim() == 2:
                            bits_rand_sign = bits_rand_sign.unsqueeze(1)
                        stats_hat_rand = hb_stats_mod.decode_bits(bits_rand_sign)

                    if isinstance(stats_hat_rand, _torch.Tensor) and stats_hat_rand.numel() > 0:
                        stats_hat_rand_flat = stats_hat_rand.detach().view(stats_hat_rand.size(0), -1)
                        rand_std_over_B = stats_hat_rand_flat.std(dim=0).cpu().tolist()
                        print(
                            f"[STATS][rand]{prefix} epoch={epoch_val} step={step_val} "
                            f"std_over_B={rand_std_over_B}"
                        )
        except Exception:
            # 统计诊断不得中断主训练逻辑。
            return

    def _debug_f0_vuv(
        cfg_f0: TrainingConfig,
        out_dict: Dict[str, torch.Tensor],
        epoch_val: int,
        step_val: int,
        prefix: str,
    ) -> None:
        """Lightweight diagnostics for F0/VUV branch when DBG_F0=1.

        打印以下信息，便于观察韵律分支是否塌缩：

        - GT / 预测 dnn_pitch 映射到 Hz 后，在 voiced 掩码下的 mean/std/min/max；
        - frame_corr / vuv_logits 的分布；
        - voiced_ratio_gt / voiced_ratio_hat（基于 VUV 概率阈值）。
        """

        import os as _os_f0

        if _os_f0.environ.get("DBG_F0", "0") != "1":
            return

        try:
            import torch as _torch

            dp_hat_any = out_dict.get("dnn_pitch_hat", None)
            dp_ref_any = out_dict.get("dnn_pitch", None)
            fc_ref_any = out_dict.get("frame_corr", None)
            fc_hat_any = out_dict.get("frame_corr_hat", None)
            vuv_logits_any = out_dict.get("vuv_logits", None)

            if not isinstance(dp_ref_any, _torch.Tensor):
                return

            dp_ref = dp_ref_any.detach().to(_torch.float32)
            dp_hat = (
                dp_hat_any.detach().to(_torch.float32)
                if isinstance(dp_hat_any, _torch.Tensor)
                else None
            )

            fc_ref = (
                fc_ref_any.detach().to(_torch.float32)
                if isinstance(fc_ref_any, _torch.Tensor)
                else None
            )
            fc_hat = (
                fc_hat_any.detach().to(_torch.float32)
                if isinstance(fc_hat_any, _torch.Tensor)
                else None
            )
            vuv_logits = (
                vuv_logits_any.detach().to(_torch.float32)
                if isinstance(vuv_logits_any, _torch.Tensor)
                else None
            )

            def _dp_to_f0_local(dp: _torch.Tensor) -> _torch.Tensor:
                period = 256.0 / _torch.pow(dp + 1.5, 1.0)
                # 与 compute_losses_simplified 中保持一致：period=256/2^(dp+1.5)
                period = 256.0 / _torch.pow(2.0, dp + 1.5)
                period = _torch.clamp(period, 32.0, 255.0)
                f0 = 16000.0 / period
                return f0.squeeze(-1)

            f0_ref = _dp_to_f0_local(dp_ref)
            f0_hat = _dp_to_f0_local(dp_hat) if isinstance(dp_hat, _torch.Tensor) else None

            vuv_thr = float(getattr(cfg_f0, "vuv_threshold", 0.3))

            mask_gt: _torch.Tensor
            if isinstance(fc_ref, _torch.Tensor):
                mask_gt = (fc_ref > vuv_thr)
            else:
                mask_gt = _torch.ones_like(dp_ref, dtype=_torch.bool)

            def _summ(x: _torch.Tensor, mask: Optional[_torch.Tensor] = None) -> str:
                if x.numel() == 0:
                    return "mean=N/A std=N/A min=N/A max=N/A"
                x_det = x.detach().to(_torch.float32)
                if mask is not None:
                    if mask.dtype != _torch.bool:
                        mask = mask > 0.5
                    # broadcast to last dim if needed
                    while mask.dim() < x_det.dim():
                        mask = mask.expand_as(x_det)
                    if not mask.any():
                        return "mean=N/A std=N/A min=N/A max=N/A"
                    x_det = x_det[mask]
                return (
                    f"mean={float(x_det.mean().item()):+.2f} "
                    f"std={float(x_det.std().item()):+.2f} "
                    f"min={float(x_det.min().item()):+.2f} "
                    f"max={float(x_det.max().item()):+.2f}"
                )

            # voiced_ratio 基于 GT 掩码和 VUV 概率
            voiced_ratio_gt: Optional[float] = None
            try:
                voiced_ratio_gt = float(mask_gt.to(_torch.float32).mean().item())
            except Exception:
                voiced_ratio_gt = None

            voiced_ratio_hat: Optional[float] = None
            vuv_prob = None
            if isinstance(vuv_logits, _torch.Tensor):
                vuv_prob = _torch.sigmoid(vuv_logits)
                try:
                    voiced_ratio_hat = float((vuv_prob > 0.5).to(_torch.float32).mean().item())
                except Exception:
                    voiced_ratio_hat = None

            def _fmt_ratio(x: Optional[float]) -> str:
                return f"{x:.3f}" if x is not None else "N/A"

            # 1) F0 Hz (voiced-only by GT mask)
            f0_ref_stats = _summ(f0_ref, mask_gt)
            f0_hat_stats = (
                _summ(f0_hat, mask_gt) if isinstance(f0_hat, _torch.Tensor) else "mean=N/A std=N/A min=N/A max=N/A"
            )
            print(
                f"[F0-DBG]{prefix} epoch={epoch_val} step={step_val} "
                f"F0_ref_voiced: {f0_ref_stats} | F0_hat_voiced: {f0_hat_stats}"
            )

            # 2) VUV / frame_corr 分布
            if isinstance(fc_ref, _torch.Tensor) or isinstance(fc_hat, _torch.Tensor):
                fc_ref_stats = _summ(fc_ref) if isinstance(fc_ref, _torch.Tensor) else "mean=N/A std=N/A min=N/A max=N/A"
                fc_hat_stats = _summ(fc_hat) if isinstance(fc_hat, _torch.Tensor) else "mean=N/A std=N/A min=N/A max=N/A"
                print(
                    f"[VUV-DBG]{prefix} epoch={epoch_val} step={step_val} "
                    f"frame_corr_ref: {fc_ref_stats} | frame_corr_hat: {fc_hat_stats} "
                    f"voiced_ratio_gt={_fmt_ratio(voiced_ratio_gt)} "
                    f"voiced_ratio_hat={_fmt_ratio(voiced_ratio_hat)}"
                )

        except Exception as _e_f0:
            try:
                print(f"[F0-DBG]{prefix} WARNING: F0/VUV debug failed at step {step_val}: {_e_f0}")
            except Exception:
                pass

    def _debug_pipeline_main_vs_offline(
        model_pipe: DualBranchBarkJSCC,
        out_main: Dict[str, torch.Tensor],
        epoch_val: int,
        step_val: int,
        prefix: str,
        device_pipe: torch.device,
        max_samples: Optional[int] = None,
        force: bool = False,
    ) -> None:
        """Compare main forward_with_hash vs offline decode on the same bits.

        当 ``DBG_PIPELINE=1`` 时，使用同一批 *信道后* bits（content/F0/stats）
        走一遍 ``decode_from_bits_offline``，并与当前 batch 的主干输出
        做分布级对比，覆盖：

        - 内容分支:  mel_hat_refined（若存在，否则 mel_hat）
        - 倒谱:      ceps_hat
        - F0/VUV:    dnn_pitch_hat / frame_corr_hat
        - 声码器输出: audio_hat

        这样可以在不引入 FSK 的前提下，精确观察「训练主干 vs 推理解码」
        在各个组件上的分布差异。

        当 ``force=True`` 时忽略环境变量开关，始终执行对比；
        否则仅在 ``DBG_PIPELINE=1`` 时生效。
        """

        if not force and os.environ.get("DBG_PIPELINE", "0") != "1":
            return

        # Guard: require bits and decode API.
        if not hasattr(model_pipe, "decode_from_bits_offline"):
            return

        # 允许通过环境变量选择使用 clean bits 还是 noisy bits：
        #   DBG_PIPELINE_USE_CLEAN_BITS=1 时使用 *_clean；否则默认使用 *_noisy。
        use_clean_bits = os.environ.get("DBG_PIPELINE_USE_CLEAN_BITS", "0") == "1"

        if use_clean_bits:
            bits_c = out_main.get("hash_bits_clean", out_main.get("bits_clean", None))
            bits_f = out_main.get("f0_hash_bits_clean", out_main.get("f0_bits_clean", None))
            bits_s = out_main.get("hash_bits_stats", None)
        else:
            bits_c = out_main.get("hash_bits_noisy", out_main.get("bits_noisy", None))
            bits_f = out_main.get("f0_hash_bits_noisy", out_main.get("f0_bits_noisy", None))
            bits_s = out_main.get("hash_bits_stats_noisy", out_main.get("hash_bits_stats", None))

        if not (
            isinstance(bits_c, torch.Tensor)
            or isinstance(bits_f, torch.Tensor)
            or isinstance(bits_s, torch.Tensor)
        ):
            return

        audio_main = out_main.get("audio_hat", None)
        if not isinstance(audio_main, torch.Tensor) or audio_main.numel() == 0:
            return

        # Limit debug cost to a small number of samples per batch.
        if max_samples is not None:
            try:
                max_b = int(max_samples)
            except Exception:
                max_b = 2
        else:
            try:
                max_b = int(os.environ.get("DBG_PIPELINE_MAX_SAMPLES", "2"))
            except Exception:
                max_b = 2
        max_b = max(1, max_b)

        B_cur = int(audio_main.size(0))
        B_use = min(B_cur, max_b)

        def _slice_first(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if not isinstance(x, torch.Tensor):
                return x
            if x.size(0) <= B_use:
                return x
            return x[:B_use]

        audio_main_s = _slice_first(audio_main)
        bits_c_s = _slice_first(bits_c) if isinstance(bits_c, torch.Tensor) else None
        bits_f_s = _slice_first(bits_f) if isinstance(bits_f, torch.Tensor) else None
        bits_s_s = _slice_first(bits_s) if isinstance(bits_s, torch.Tensor) else None

        csi_vec = out_main.get("csi_vec", None)
        if isinstance(csi_vec, torch.Tensor):
            csi_vec = _slice_first(csi_vec)

        dp_main = out_main.get("dnn_pitch_hat", None)
        if isinstance(dp_main, torch.Tensor):
            dp_main_s = _slice_first(dp_main)
            f0_T_use = int(dp_main_s.size(1))
        else:
            f0_T_use = None

        target_len = int(audio_main_s.size(1))

        # content_hw: 若主干 forward_with_hash 暴露了 hw，则优先使用，以确保
        # VMamba 解码的 token 网格形状与训练路径完全一致；否则退回到
        # decode_from_bits_offline 内部的形状推断逻辑。
        content_hw = None
        hw_main = out_main.get("hw", None)
        if isinstance(hw_main, (tuple, list)) and len(hw_main) == 2:
            try:
                content_hw = (int(hw_main[0]), int(hw_main[1]))
            except Exception:
                content_hw = None

        try:
            with torch.no_grad():
                out_off = model_pipe.decode_from_bits_offline(
                    bits_content=bits_c_s,
                    bits_f0=bits_f_s,
                    bits_stats=bits_s_s,
                    f0_T=f0_T_use,
                    target_len=target_len,
                    csi_vec=csi_vec,
                    snr_db=None,
                    content_hw=content_hw,
                )
        except Exception as _e_pipe:
            try:
                print(f"[DBG_PIPELINE]{prefix} epoch={epoch_val} step={step_val} offline decode failed: {_e_pipe}")
            except Exception:
                pass
            return

        def _stat(x: Optional[torch.Tensor]) -> Optional[Dict[str, float]]:
            if not isinstance(x, torch.Tensor) or x.numel() == 0:
                return None
            x_det = x.detach().to(torch.float32)
            return {
                "mean": float(x_det.mean().item()),
                "std": float(x_det.std().item()),
                "min": float(x_det.min().item()),
                "max": float(x_det.max().item()),
            }

        def _fmt(d: Optional[Dict[str, float]]) -> str:
            if d is None:
                return "mean=N/A std=N/A min=N/A max=N/A"
            return (
                f"mean={d['mean']:+.3e} std={d['std']:+.3e} "
                f"min={d['min']:+.3e} max={d['max']:+.3e}"
            )

        # Content branch (mel)
        mel_main = out_main.get("mel_hat_refined", out_main.get("mel_hat", None))
        mel_main = _slice_first(mel_main) if isinstance(mel_main, torch.Tensor) else None
        mel_off = out_off.get("mel_hat_refined", out_off.get("mel_hat", None)) if isinstance(out_off, dict) else None

        # Normalised mel (using stats decoded from bits) and L2H-used mel
        mm_main = out_main.get("mel_mean_hat", None)
        mm_main = _slice_first(mm_main) if isinstance(mm_main, torch.Tensor) else None
        ms_main = out_main.get("mel_std_hat", None)
        ms_main = _slice_first(ms_main) if isinstance(ms_main, torch.Tensor) else None

        mm_off = out_off.get("mel_mean_hat", None) if isinstance(out_off, dict) else None
        mm_off = _slice_first(mm_off) if isinstance(mm_off, torch.Tensor) else None
        ms_off = out_off.get("mel_std_hat", None) if isinstance(out_off, dict) else None
        ms_off = _slice_first(ms_off) if isinstance(ms_off, torch.Tensor) else None

        mel_norm_main: Optional[torch.Tensor] = None
        mel_norm_off: Optional[torch.Tensor] = None
        if isinstance(mel_main, torch.Tensor) and isinstance(mm_main, torch.Tensor) and isinstance(ms_main, torch.Tensor):
            eps_n = 1e-6
            mel_norm_main = (mel_main - mm_main) / ms_main.clamp(min=eps_n)
        if isinstance(mel_off, torch.Tensor) and isinstance(mm_off, torch.Tensor) and isinstance(ms_off, torch.Tensor):
            eps_n = 1e-6
            mel_norm_off = (mel_off - mm_off) / ms_off.clamp(min=eps_n)

        mel_used_main = out_main.get("mel_hat_refined", None)
        mel_used_main = _slice_first(mel_used_main) if isinstance(mel_used_main, torch.Tensor) else None
        mel_used_off = mel_off

        # Cepstrum
        ceps_main = out_main.get("ceps_hat", None)
        ceps_main = _slice_first(ceps_main) if isinstance(ceps_main, torch.Tensor) else None
        ceps_off = out_off.get("ceps_hat", None) if isinstance(out_off, dict) else None

        # F0 / VUV
        dp_main_s = _slice_first(dp_main) if isinstance(dp_main, torch.Tensor) else None
        fc_main = out_main.get("frame_corr_hat", None)
        fc_main = _slice_first(fc_main) if isinstance(fc_main, torch.Tensor) else None
        dp_off = out_off.get("dnn_pitch_hat", None) if isinstance(out_off, dict) else None
        fc_off = out_off.get("frame_corr_hat", None) if isinstance(out_off, dict) else None

        # Audio
        audio_off = out_off.get("audio_hat", None) if isinstance(out_off, dict) else None

        base = f"[PIPE]{prefix} epoch={epoch_val} step={step_val}"
        try:
            print(f"{base} mel_main:  {_fmt(_stat(mel_main))}")
            print(f"{base} mel_off:   {_fmt(_stat(mel_off))}")
            print(f"{base} mel_norm_main:  {_fmt(_stat(mel_norm_main))}")
            print(f"{base} mel_norm_off:   {_fmt(_stat(mel_norm_off))}")
            print(f"{base} mel_used_main:  {_fmt(_stat(mel_used_main))}")
            print(f"{base} mel_used_off:   {_fmt(_stat(mel_used_off))}")
            print(f"{base} mel_mean_hat_main: {_fmt(_stat(mm_main))}")
            print(f"{base} mel_mean_hat_off:  {_fmt(_stat(mm_off))}")
            print(f"{base} mel_std_hat_main:  {_fmt(_stat(ms_main))}")
            print(f"{base} mel_std_hat_off:   {_fmt(_stat(ms_off))}")
            print(f"{base} ceps_main: {_fmt(_stat(ceps_main))}")
            print(f"{base} ceps_off:  {_fmt(_stat(ceps_off))}")
            print(f"{base} dnn_main:  {_fmt(_stat(dp_main_s))}")
            print(f"{base} dnn_off:   {_fmt(_stat(dp_off))}")
            print(f"{base} fc_main:   {_fmt(_stat(fc_main))}")
            print(f"{base} fc_off:    {_fmt(_stat(fc_off))}")
            print(f"{base} audio_main:{_fmt(_stat(audio_main_s))}")
            print(f"{base} audio_off: {_fmt(_stat(audio_off))}")
        except Exception:
            # Debug printing must never break training.
            pass

    def _run_pipeline_probe(
        model_probe: DualBranchBarkJSCC,
        dataloader_probe,
        channel_sim_probe: ChannelSimulator,
        shim_probe,
        device_probe: torch.device,
        cfg_probe: TrainingConfig,
    ) -> None:
        """Run a lightweight pipeline probe on a small number of samples.

        该探针模式不进行训练更新，仅针对前 ``pipeline_probe_num_samples``
        条音频（按 batch 裁剪）分别在 *train* / *eval* 模式下跑一次
        ``model_forward``，并调用 RVQ / PIPELINE 调试打印函数：

        - [probe-train] 前缀：model.train() 下的分布；
        - [probe-eval]  前缀：model.eval()  下的分布；

        打印完指定样本数后立即返回，训练主循环不会执行。
        """

        try:
            max_total = int(getattr(cfg_probe, "pipeline_probe_num_samples", 10) or 10)
        except Exception:
            max_total = 10
        max_total = max(1, max_total)

        processed = 0
        step_local = 0

        with torch.no_grad():
            for batch in dataloader_probe:
                if processed >= max_total:
                    break
                if not isinstance(batch, dict) or "audio" not in batch:
                    continue

                audio_b = batch["audio"]
                if not isinstance(audio_b, torch.Tensor) or audio_b.size(0) <= 0:
                    continue

                B_batch = int(audio_b.size(0))
                remain = max_total - processed
                if remain <= 0:
                    break
                B_use = min(B_batch, remain)

                # Slice batch到前 B_use 个样本，确保总样本数受控。
                batch_slice: Dict[str, torch.Tensor] = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor) and v.size(0) >= B_use:
                        batch_slice[k] = v[:B_use]
                    else:
                        batch_slice[k] = v

                # 1) train 模式 forward（保留 dropout / BN 行为）。
                #    前向路径根据 *模型本身* 是否启用
                #    hash/RVQ（model_probe.with_hash）自动选择
                #    带 hash 的 forward_with_hash 或无 hash 的
                #    常规 forward，避免依赖 CLI 中的 with_hash
                #    标志与 checkpoint 配置不一致。
                model_probe.train()
                out_train = model_forward(  # type: ignore[arg-type]
                    model_probe,
                    batch_slice,
                    channel_sim_probe,
                    shim_probe,
                    device_probe,
                )
                _debug_rvq_latent_stats(
                    out_train,
                    epoch_val=0,
                    step_val=step_local,
                    prefix="[probe-train]",
                )
                _debug_pipeline_main_vs_offline(
                    model_pipe=model_probe,
                    out_main=out_train,
                    epoch_val=0,
                    step_val=step_local,
                    prefix="[probe-train]",
                    device_pipe=device_probe,
                    max_samples=B_use,
                    force=True,
                )

                # 2) eval 模式 forward（固定 BN / 关闭 dropout）。
                model_probe.eval()
                out_eval = model_forward(  # type: ignore[arg-type]
                    model_probe,
                    batch_slice,
                    channel_sim_probe,
                    shim_probe,
                    device_probe,
                )
                _debug_rvq_latent_stats(
                    out_eval,
                    epoch_val=0,
                    step_val=step_local,
                    prefix="[probe-eval]",
                )
                _debug_pipeline_main_vs_offline(
                    model_pipe=model_probe,
                    out_main=out_eval,
                    epoch_val=0,
                    step_val=step_local,
                    prefix="[probe-eval]",
                    device_pipe=device_probe,
                    max_samples=B_use,
                    force=True,
                )

                processed += B_use
                step_local += 1
                if processed >= max_total:
                    break

        try:
            print(
                f"[PipelineProbe] Completed pipeline probe on {processed} sample(s); "
                "skipping training loop."
            )
        except Exception:
            pass

    # Pipeline probe mode: only run a small number of samples through
    # train/eval forward paths for feature distribution debugging, then exit.
    if bool(getattr(cfg, "pipeline_probe", False)):
        _run_pipeline_probe(
            model_probe=model,
            dataloader_probe=dataloader,
            channel_sim_probe=channel_sim,
            shim_probe=shim,
            device_probe=device,
            cfg_probe=cfg,
        )
        return

    # Eval-only mode: run a single pass over the dataloader using a
    # forward path that matches the *checkpoint*'s hash setting
    # (model.with_hash). We still skip optimizer/gradient updates and
    # plotting, but populate bit_only_metrics.csv when
    # bit_only_eval=True. This is useful for quickly regenerating CSV
    # metrics from a fixed checkpoint.
    if bool(getattr(cfg, "only_eval", False)):
        model.eval()
        global_step_eval = int(global_step)
        with torch.no_grad():
            for batch in dataloader:
                out = model_forward(model, batch, channel_sim, shim, device)  # type: ignore[arg-type]

                # 可选：在 only_eval 模式下同样打印 RVQ latent 分布，方便
                # 直接比较训练/推理端 RVQ→信道→RVQ 的行为。
                _debug_rvq_latent_stats(out, epoch_val=start_epoch, step_val=global_step_eval, prefix="[only_eval]")

                # 可选：only_eval 模式下进行 stats bits 诊断，观察 hash_content_stats
                # 在纯推理场景下的行为（与训练阶段对齐）。
                _debug_stats_hash(model, out, epoch_val=start_epoch, step_val=global_step_eval, prefix="[only_eval]")

                # 可选：F0/VUV 分支诊断（only_eval 路径）。
                _debug_f0_vuv(cfg, out, epoch_val=start_epoch, step_val=global_step_eval, prefix="[only_eval]")

                # 可选：主干 forward_with_hash vs decode_from_bits_offline
                # 的组件级对比（mel / ceps / F0 / audio）。
                _debug_pipeline_main_vs_offline(
                    model_pipe=model,
                    out_main=out,
                    epoch_val=start_epoch,
                    step_val=global_step_eval,
                    prefix="[only_eval]",
                    device_pipe=device,
                )

                # 仅在 bit_only_eval=True 时触发 bit-only 评估路径；
                # 跳过训练损失和所有可视化，只写 CSV。
                if bool(getattr(cfg, "bit_only_eval", False)):
                    # only_eval 模式下使用致密的 SNR 栅格：-5dB 到 10dB，步长 1dB。
                    snr_grid_dense: List[float] = [float(s) for s in range(-5, 11)]
                    _run_bit_only_eval_for_batch(
                        model=model,
                        batch=batch,
                        out=out,
                        cfg=cfg,
                        channel_sim=channel_sim,
                        device=device,
                        global_step_val=global_step_eval,
                        do_visualization=False,
                        snr_grid_all=snr_grid_dense,
                    )

                global_step_eval += 1
        return

    for epoch in range(start_epoch, cfg.num_epochs):
        model.train()
        for batch in dataloader:
            optimizer.zero_grad(set_to_none=True)

            # Unfreeze content branch after the warm-up window has passed.
            # content_warmup_steps is counted from 0 *since start/resume*,
            # not in absolute global_step, so we compare against the local
            # counter content_steps_since_resume.
            if content_frozen and content_steps_since_resume >= content_warmup_steps:
                _set_content_requires_grad(True)
                content_frozen = False
                print(
                    "[Content-Warmup] Unfroze content branch after "
                    f"{content_steps_since_resume} step(s) since start/resume "
                    f"(global_step={global_step})"
                )

            out = model_forward(model, batch, channel_sim, shim, device)  # type: ignore[arg-type]

            # Debug: RVQ latent stats before/after channel (训练模式)
            _debug_rvq_latent_stats(out, epoch_val=epoch, step_val=global_step, prefix="[train]")

            # Debug: stats bits collapse / utilisation diagnostics（训练模式）
            _debug_stats_hash(model, out, epoch_val=epoch, step_val=global_step, prefix="[train]")

            # Debug: F0/VUV branch diagnostics（训练模式）。
            _debug_f0_vuv(cfg, out, epoch_val=epoch, step_val=global_step, prefix="[train]")

            # Debug: 主干 forward_with_hash vs offline decode（组件级）
            _debug_pipeline_main_vs_offline(
                model_pipe=model,
                out_main=out,
                epoch_val=epoch,
                step_val=global_step,
                prefix="[train]",
                device_pipe=device,
            )
            # Content-only mode: use v3-style mel / ceps / hash losses,
            # skip F0/vocoder/bit-only/HiFi-GAN extras, and optionally
            # attach L2H residual/decor losses when --with_l2h is enabled.
            if bool(getattr(cfg, "content_only", False)):
                loss, loss_dict_content, _ = v3_compute_content_only_losses(
                    model, out, shim, device
                )
                # v3 content-only losses already include lambda_mel,
                # lambda_mel_l1, lambda_mel_energy 以及可选 hash/HashReg
                # 项。对于 RVQ 量化器，为了在 content_only 模式下也
                # 训练 codebook，这里额外叠加简化版的 VQ 损失
                # (lambda_vq_c, lambda_vq_stats)。
                loss_dict = dict(loss_dict_content)

                try:
                    lam_vq_c = float(getattr(cfg, "lambda_vq_c", 0.0))
                    lam_vq_stats = float(getattr(cfg, "lambda_vq_stats", 0.0))

                    vq_loss_c = out.get("vq_loss_content", None)
                    vq_loss_stats = out.get("vq_loss_stats", None)

                    if isinstance(vq_loss_c, torch.Tensor) and lam_vq_c > 0.0:
                        loss = loss + lam_vq_c * vq_loss_c.to(device)
                        loss_dict["vq_c"] = float(vq_loss_c.item())

                    if isinstance(vq_loss_stats, torch.Tensor) and lam_vq_stats > 0.0:
                        loss = loss + lam_vq_stats * vq_loss_stats.to(device)
                        loss_dict["vq_stats"] = float(vq_loss_stats.item())
                except Exception as _e_vq_co:
                    # VQ 诊断/附加项绝不能打断主训练流程。
                    print(f"[ContentOnly] WARNING: VQ losses skipped due to error: {_e_vq_co}")

                # Optional: when --with_l2h is enabled and corresponding
                # lambda_l2h_* > 0, also attach the same L2H residual /
                # decorrelation losses as in the full simplified loss
                # bundle so that content_only+L2H jointly trains the
                # DeCoL2HRefiner.
                try:
                    if bool(getattr(cfg, "with_l2h", False)) and (
                        float(getattr(cfg, "lambda_l2h_resid", 0.0)) > 0.0
                        or float(getattr(cfg, "lambda_l2h_decor", 0.0)) > 0.0
                    ):
                        loss_l2h, loss_dict_l2h = _compute_l2h_resid_and_decor_losses(
                            out, cfg, device, model
                        )
                        loss = loss + loss_l2h
                        loss_dict.update(loss_dict_l2h)
                except Exception as _e_l2h_co:
                    print(f"[ContentOnly] WARNING: L2H losses skipped due to error: {_e_l2h_co}")

                loss.backward()
                # Debug: print hash_content_stats grad norms when DBG_STATS=1
                try:
                    import os as _os_dbg
                    if _os_dbg.environ.get("DBG_STATS", "0") == "1" and global_step % int(getattr(cfg, "log_every_steps", 10) or 10) == 0:
                        try:
                            print("[STATS-GRAD][train] hash_content_stats parameter grad norms:")
                            for name, param in model.named_parameters():
                                if "hash_content_stats" in name and isinstance(param, torch.nn.Parameter):
                                    g = param.grad
                                    if g is None:
                                        gnorm = None
                                    else:
                                        gnorm = float(g.detach().norm().item())
                                    print(f"  {name}: grad_norm={gnorm}")
                        except Exception as _e_grad:
                            print(f"[STATS-GRAD][train] WARNING: grad inspection failed: {_e_grad}")
                except Exception:
                    pass

                optimizer.step()
            else:
                loss, loss_dict = compute_losses_simplified(out, cfg, device, model)

                # Optional: training-time BER diagnostics for the internal
                # BPSK+AWGN channel, controlled by DBG_TRAIN_BER=1.
                if os.environ.get("DBG_TRAIN_BER", "0") == "1":
                    try:
                        def _ber_from_pair(
                            b_clean: Optional[torch.Tensor],
                            b_noisy: Optional[torch.Tensor],
                        ) -> Optional[float]:
                            if not (
                                isinstance(b_clean, torch.Tensor)
                                and isinstance(b_noisy, torch.Tensor)
                            ):
                                return None
                            x = b_clean.detach().reshape(-1)
                            y = b_noisy.detach().reshape(-1)
                            n = int(min(x.numel(), y.numel()))
                            if n <= 0:
                                return None
                            xb = (x[:n] > 0).to(torch.int32)
                            yb = (y[:n] > 0).to(torch.int32)
                            err = int((xb != yb).sum().item())
                            return float(err / max(1, n))

                        bits_c_clean = out.get("hash_bits_clean", None)
                        bits_c_noisy = out.get(
                            "hash_bits_noisy", out.get("bits_noisy", None)
                        )
                        bits_f_clean = out.get("f0_hash_bits_clean", None)
                        bits_f_noisy = out.get(
                            "f0_hash_bits_noisy", out.get("f0_bits_noisy", None)
                        )
                        bits_s_clean = out.get("hash_bits_stats", None)
                        bits_s_noisy = out.get("hash_bits_stats_noisy", None)

                        ber_c = _ber_from_pair(bits_c_clean, bits_c_noisy)
                        ber_f = _ber_from_pair(bits_f_clean, bits_f_noisy)
                        ber_s = _ber_from_pair(bits_s_clean, bits_s_noisy)

                        all_clean: list[torch.Tensor] = []
                        all_noisy: list[torch.Tensor] = []
                        for b_cln, b_nsy in (
                            (bits_c_clean, bits_c_noisy),
                            (bits_f_clean, bits_f_noisy),
                            (bits_s_clean, bits_s_noisy),
                        ):
                            if isinstance(b_cln, torch.Tensor) and isinstance(
                                b_nsy, torch.Tensor
                            ):
                                all_clean.append(b_cln.detach().reshape(-1))
                                all_noisy.append(b_nsy.detach().reshape(-1))

                        ber_all: Optional[float] = None
                        if all_clean and all_noisy:
                            x_all = torch.cat(all_clean, dim=0)
                            y_all = torch.cat(all_noisy, dim=0)
                            n_all = int(min(x_all.numel(), y_all.numel()))
                            if n_all > 0:
                                xb_all = (x_all[:n_all] > 0).to(torch.int32)
                                yb_all = (y_all[:n_all] > 0).to(torch.int32)
                                err_all = int((xb_all != yb_all).sum().item())
                                ber_all = float(err_all / max(1, n_all))

                        snr_str = ""
                        csi_val = out.get("csi_vec", None)
                        if (
                            isinstance(csi_val, torch.Tensor)
                            and csi_val.dim() == 2
                            and csi_val.size(1) >= 1
                        ):
                            try:
                                snr_mean = float(csi_val[:, 0].mean().item())
                                snr_str = f" snr_db~{snr_mean:.2f}"
                            except Exception:
                                snr_str = ""

                        def _fmt_ber(v: Optional[float]) -> str:
                            return f"{v:.6e}" if v is not None else "N/A"

                        if snr_str == "":
                            snr_str = " snr_db~N/A"

                        msg_ber = (
                            f"[train][BPSK-BER] epoch={epoch} step={global_step}{snr_str} "
                            f"ber_all={_fmt_ber(ber_all)} "
                            f"ber_c={_fmt_ber(ber_c)} "
                            f"ber_f={_fmt_ber(ber_f)} "
                            f"ber_s={_fmt_ber(ber_s)}"
                        )
                        print(msg_ber)
                    except Exception as _e_ber_train:
                        if os.environ.get("DBG_TRAIN_BER", "0") == "1":
                            print(
                                f"[train][BPSK-BER] WARNING: BER calc failed at step {global_step}: {_e_ber_train}"
                            )

                # Optional: bit-only BFCC 静音约束
                bit_sil_loss, bit_sil_dict = _compute_bit_only_silence_loss(
                    model, batch, cfg, channel_sim, device
                )
                if bit_sil_loss.requires_grad:
                    loss = loss + bit_sil_loss
                    loss_dict.update(bit_sil_dict)

                # Adversarial waveform training (only in full mode).  We
                # always update the discriminators when GAN is enabled, but
                # the generator only receives adversarial gradients once the
                # local warm-up counter ``gan_steps_since_resume`` exceeds
                # ``gan_adv_warmup_steps`` so that D can stabilise first
                # (FARGAN-style warm-up).
                if (
                    mpd is not None
                    and msd is not None
                    and optimizer_hifi_disc is not None
                ):
                    lam_adv_hifi = float(getattr(cfg, "lambda_gan_adv", 0.0))
                    lam_fm_hifi = float(getattr(cfg, "lambda_gan_fm", 0.0))
                    warm_hifi = int(getattr(cfg, "gan_adv_warmup_steps", 0))
                    if lam_adv_hifi > 0.0 or lam_fm_hifi > 0.0:
                        # Shared crop for D/G updates: random segment when
                        # crop_len > 0, otherwise fall back to min-length.
                        try:
                            y_ref_full = out["audio"]
                            y_hat_full = out["audio_hat"]
                            min_len = min(
                                y_ref_full.size(-1),
                                y_hat_full.size(-1),
                            )
                            crop_len_cfg = int(
                                getattr(cfg, "gan_adv_crop_len", 0)
                            )
                            if crop_len_cfg > 0 and min_len > crop_len_cfg:
                                max_start = max(1, min_len - crop_len_cfg)
                                start_idx = int(
                                    torch.randint(
                                        0,
                                        max_start,
                                        (1,),
                                        device=y_ref_full.device,
                                    ).item()
                                )
                                end_idx = start_idx + crop_len_cfg
                            else:
                                start_idx = 0
                                end_idx = min_len
                        except Exception:
                            # Fallback: use full aligned waveforms
                            y_ref_full = out["audio"]
                            y_hat_full = out["audio_hat"]
                            min_len = min(
                                y_ref_full.size(-1),
                                y_hat_full.size(-1),
                            )
                            start_idx = 0
                            end_idx = min_len

                        # D update (always applied when GAN is enabled)
                        try:
                            y_real = y_ref_full.detach()[:, start_idx:end_idx]
                            y_fake = y_hat_full.detach()[:, start_idx:end_idx]

                            y_real_d = y_real.unsqueeze(1)
                            y_fake_d = y_fake.unsqueeze(1)

                            for p in mpd.parameters():
                                p.requires_grad_(True)
                            for p in msd.parameters():
                                p.requires_grad_(True)
                            if bark_disc is not None:
                                for p in bark_disc.parameters():
                                    p.requires_grad_(True)
                            if f0p_disc is not None:
                                for p in f0p_disc.parameters():
                                    p.requires_grad_(True)

                            optimizer_hifi_disc.zero_grad(set_to_none=True)
                            with autocast(enabled=use_amp):
                                # Waveform MPD/MSD
                                y_df_hat_r, y_df_hat_g, _, _ = mpd(y_real_d, y_fake_d)
                                y_ds_hat_r, y_ds_hat_g, _, _ = msd(y_real_d, y_fake_d)
                                loss_disc_f, _, _ = hifi_discriminator_loss(
                                    y_df_hat_r, y_df_hat_g
                                )
                                loss_disc_s, _, _ = hifi_discriminator_loss(
                                    y_ds_hat_r, y_ds_hat_g
                                )
                                loss_disc_all = loss_disc_f + loss_disc_s

                                # BarkHF discriminator（高频 Bark/BFCC 纹理），
                                # 仅在有声帧上评估经 z-norm 归一化的 HF Bark 图，
                                # 避免静音段噪声主导判别结果。
                                if wave_to_bfcc_adv is not None and bark_disc is not None:
                                    try:
                                        x_bark_r, x_bark_g = _compute_bark_hf_maps(
                                            wave_to_bfcc_adv,
                                            y_real,
                                            y_fake,
                                            frame_corr=out.get("frame_corr", None),
                                            vuv_threshold=float(getattr(cfg, "vuv_threshold", 0.3)),
                                        )
                                        y_bark_r, _ = bark_disc(x_bark_r)
                                        y_bark_g, _ = bark_disc(x_bark_g)
                                        loss_disc_bark, _, _ = hifi_discriminator_loss(
                                            [y_bark_r], [y_bark_g]
                                        )
                                        loss_disc_all = loss_disc_all + loss_disc_bark
                                    except Exception as _e_bark:
                                        if os.environ.get("DBG_HIFI_ADV", "0") == "1":
                                            print(f"[HiFi-ADV] BarkHF D skipped: {_e_bark}")

                                # F0Period discriminator：real 使用 GT dnn_pitch/frame_corr，
                                # fake 使用 dnn_pitch_hat，并确保 patch 形状对齐。
                                if f0p_disc is not None:
                                    try:
                                        P0 = 128
                                        dp_real = out.get("dnn_pitch")
                                        dp_fake = out.get("dnn_pitch_hat")
                                        # 回退：若预测不存在，则退回到 GT，以兼容旧 checkpoint。
                                        if not isinstance(dp_fake, torch.Tensor):
                                            dp_fake = dp_real

                                        fc_ref_adv = out.get("frame_corr")

                                        if isinstance(dp_real, torch.Tensor) and isinstance(dp_fake, torch.Tensor):
                                            x_f0p_r, x_f0p_g = _build_period_patches_pair_from_dnn_pitch(
                                                y_real,
                                                y_fake,
                                                dp_real,
                                                dp_fake,
                                                start_idx=start_idx,
                                                end_idx=end_idx,
                                                hop=160,
                                                target_period=P0,
                                                frame_corr=fc_ref_adv,
                                                vuv_threshold=float(getattr(cfg, "vuv_threshold", 0.3)),
                                            )

                                            if x_f0p_r.size(2) > 0 and x_f0p_g.size(2) > 0:
                                                y_f0p_r, _ = f0p_disc(x_f0p_r)
                                                y_f0p_g, _ = f0p_disc(x_f0p_g)
                                                loss_disc_f0p, _, _ = hifi_discriminator_loss(
                                                    [y_f0p_r], [y_f0p_g]
                                                )
                                                loss_disc_all = loss_disc_all + loss_disc_f0p
                                    except Exception as _e_f0p:
                                        if os.environ.get("DBG_HIFI_ADV", "0") == "1":
                                            print(f"[HiFi-ADV] F0Period D skipped: {_e_f0p}")

                            # Optional debug: discriminator output stats
                            _debug_log_gan_disc_stats(
                                y_df_hat_r,
                                y_df_hat_g,
                                y_ds_hat_r,
                                y_ds_hat_g,
                                y_bark_r if "y_bark_r" in locals() else None,
                                y_bark_g if "y_bark_g" in locals() else None,
                                y_f0p_r if "y_f0p_r" in locals() else None,
                                y_f0p_g if "y_f0p_g" in locals() else None,
                                epoch=epoch,
                                step=global_step,
                            )

                            loss_disc_all.backward()
                            optimizer_hifi_disc.step()
                            loss_dict["gan_disc"] = float(loss_disc_all.detach().item())
                        except Exception as _e_hifi_d:
                            loss_dict["gan_disc"] = 0.0
                            _dbg = os.environ.get("DBG_GAN_ADV", os.environ.get("DBG_HIFI_ADV", "0"))
                            if _dbg == "1":
                                print(f"[GAN-ADV] D update skipped: {_e_hifi_d}")

                        # G update: only apply adversarial gradients after
                        # the warm-up period so that D can learn to
                        # discriminate around the current generator first.
                        if gan_steps_since_resume < warm_hifi:
                            lam_adv_hifi = 0.0
                            lam_fm_hifi = 0.0

                        try:
                            for p in mpd.parameters():
                                p.requires_grad_(False)
                            for p in msd.parameters():
                                p.requires_grad_(False)
                            if bark_disc is not None:
                                for p in bark_disc.parameters():
                                    p.requires_grad_(False)
                            if f0p_disc is not None:
                                for p in f0p_disc.parameters():
                                    p.requires_grad_(False)

                            # Use the same crop [start_idx:end_idx] as the
                            # discriminator update so that D/G see the same
                            # time region.
                            y_real_g = out["audio"][:, start_idx:end_idx]
                            y_fake_g = out["audio_hat"][:, start_idx:end_idx]

                            y_real_g1 = y_real_g.unsqueeze(1)
                            y_fake_g1 = y_fake_g.unsqueeze(1)

                            with autocast(enabled=use_amp):
                                # Waveform MPD/MSD
                                y_df_hat_r_g, y_df_hat_g_g, fmap_f_r, fmap_f_g = mpd(
                                    y_real_g1, y_fake_g1
                                )
                                y_ds_hat_r_g, y_ds_hat_g_g, fmap_s_r, fmap_s_g = msd(
                                    y_real_g1, y_fake_g1
                                )

                                loss_gen_f, _ = hifi_generator_loss(y_df_hat_g_g)
                                loss_gen_s, _ = hifi_generator_loss(y_ds_hat_g_g)
                                loss_g_adv = loss_gen_f + loss_gen_s

                                loss_fm_f = hifi_feature_loss(fmap_f_r, fmap_f_g)
                                loss_fm_s = hifi_feature_loss(fmap_s_r, fmap_s_g)
                                loss_g_fm = loss_fm_f + loss_fm_s

                                # BarkHF G loss：在 voiced 帧上对经 z-norm 的 HF
                                # Bark 图施加 LSGAN + FM 约束，并可选地叠加简单
                                # L1 锚点，避免 GAN 单纯通过高频噪声取巧。
                                if wave_to_bfcc_adv is not None and bark_disc is not None:
                                    try:
                                        x_bark_r_g, x_bark_g_g = _compute_bark_hf_maps(
                                            wave_to_bfcc_adv,
                                            y_real_g,
                                            y_fake_g,
                                            frame_corr=out.get("frame_corr", None),
                                            vuv_threshold=float(getattr(cfg, "vuv_threshold", 0.3)),
                                        )
                                        y_bark_r_g, fmap_bark_r = bark_disc(x_bark_r_g)
                                        y_bark_g_g, fmap_bark_g = bark_disc(x_bark_g_g)
                                        loss_gen_bark, _ = hifi_generator_loss([y_bark_g_g])
                                        loss_fm_bark = hifi_feature_loss(
                                            [fmap_bark_r], [fmap_bark_g]
                                        )
                                        loss_g_adv = loss_g_adv + loss_gen_bark
                                        loss_g_fm = loss_g_fm + loss_fm_bark

                                        # Optional HF Bark L1 anchor on the
                                        # (gated + normalised) HF Bark maps
                                        # to keep adversarial updates close
                                        # to the real Bark texture.
                                        lam_bark_l1 = float(
                                            getattr(cfg, "lambda_bark_hf_l1", 0.0)
                                        )
                                        if lam_bark_l1 > 0.0:
                                            try:
                                                if (
                                                    x_bark_r_g.shape == x_bark_g_g.shape
                                                    and x_bark_r_g.numel() > 0
                                                ):
                                                    l_bark_l1 = torch.mean(
                                                        torch.abs(x_bark_g_g - x_bark_r_g)
                                                    )
                                                    loss_g_adv = (
                                                        loss_g_adv
                                                        + lam_bark_l1 * l_bark_l1
                                                    )
                                                    loss_dict["bark_hf_l1"] = float(
                                                        l_bark_l1.detach().item()
                                                    )
                                            except Exception:
                                                loss_dict.setdefault("bark_hf_l1", 0.0)

                                        # Optional HF Bark energy clamp: only penalise
                                        # cases where fake HF Bark energy significantly
                                        # exceeds real (helps avoid over-bright, noisy HF
                                        # while keeping some freedom for GAN to shape
                                        # high-frequency texture).
                                        lam_bark_e = float(
                                            getattr(cfg, "lambda_bark_hf_energy", 0.0)
                                        )
                                        if lam_bark_e > 0.0:
                                            try:
                                                # x_bark_* are log10 Bark energies.
                                                E_r = x_bark_r_g.mean(dim=(1, 2, 3))
                                                E_g = x_bark_g_g.mean(dim=(1, 2, 3))
                                                margin_db = float(
                                                    getattr(
                                                        cfg,
                                                        "bark_hf_energy_margin_db",
                                                        3.0,
                                                    )
                                                )
                                                margin_log = margin_db / 10.0
                                                # Only penalise when fake > real + margin.
                                                diff_e = torch.relu(E_g - (E_r + margin_log))
                                                if diff_e.numel() > 0:
                                                    l_bark_e = diff_e.mean()
                                                    loss_g_adv = loss_g_adv + lam_bark_e * l_bark_e
                                                    loss_dict["bark_hf_energy"] = float(
                                                        l_bark_e.detach().item()
                                                    )
                                            except Exception:
                                                loss_dict.setdefault("bark_hf_energy", 0.0)
                                    except Exception as _e_bark_g:
                                        if os.environ.get("DBG_HIFI_ADV", "0") == "1":
                                            print(f"[HiFi-ADV] BarkHF G skipped: {_e_bark_g}")

                                # F0Period G loss：real patch 用 GT dnn_pitch，
                                # fake patch 用 dnn_pitch_hat，对齐到各自预测 F0。
                                if f0p_disc is not None:
                                    try:
                                        P0 = 128
                                        dp_real = out.get("dnn_pitch")
                                        dp_fake = out.get("dnn_pitch_hat")
                                        if not isinstance(dp_fake, torch.Tensor):
                                            dp_fake = dp_real

                                        fc_ref_adv = out.get("frame_corr")

                                        if isinstance(dp_real, torch.Tensor) and isinstance(dp_fake, torch.Tensor):
                                            x_f0p_r_g, x_f0p_g_g = _build_period_patches_pair_from_dnn_pitch(
                                                y_real_g,
                                                y_fake_g,
                                                dp_real,
                                                dp_fake,
                                                start_idx=start_idx,
                                                end_idx=end_idx,
                                                hop=160,
                                                target_period=P0,
                                                frame_corr=fc_ref_adv,
                                                vuv_threshold=float(getattr(cfg, "vuv_threshold", 0.3)),
                                            )

                                            if x_f0p_r_g.size(2) > 0 and x_f0p_g_g.size(2) > 0:
                                                y_f0p_r_g, fmap_f0p_r = f0p_disc(x_f0p_r_g)
                                                y_f0p_g_g, fmap_f0p_g = f0p_disc(x_f0p_g_g)
                                                loss_gen_f0p, _ = hifi_generator_loss([y_f0p_g_g])
                                                loss_fm_f0p = hifi_feature_loss(
                                                    [fmap_f0p_r], [fmap_f0p_g]
                                                )
                                                loss_g_adv = loss_g_adv + loss_gen_f0p
                                                loss_g_fm = loss_g_fm + loss_fm_f0p
                                    except Exception as _e_f0p_g:
                                        if os.environ.get("DBG_HIFI_ADV", "0") == "1":
                                            print(f"[HiFi-ADV] F0Period G skipped: {_e_f0p_g}")

                            adv_total = lam_adv_hifi * loss_g_adv + lam_fm_hifi * loss_g_fm

                            # When adv_scope is set to "vocoder_l2h_ceps",
                            # restrict adversarial gradients to vocoder/L2H/
                            # mel→ceps modules; other parameters will only
                            # see reconstruction losses. In the default
                            # "full" scope, fall back to the original
                            # behaviour and add GAN losses to the global
                            # objective.
                            if str(getattr(cfg, "adv_scope", "full")) == "vocoder_l2h_ceps":
                                _accumulate_adv_grads_for_scope(
                                    adv_total,
                                    model,
                                    cfg,
                                )
                                _debug_log_gan_grads(
                                    model,
                                    global_step=global_step,
                                    prefixes=(
                                        "vocoder",
                                        "deco_l2h_refiner",
                                        "l2h_flow",
                                        "mel18_to_ceps",
                                        "hf2ceps",
                                    ),
                                    tag="scoped-adv",
                                )
                            else:
                                loss = loss + adv_total

                            if lam_adv_hifi > 0.0:
                                loss_dict["gan_adv_g"] = float((lam_adv_hifi * loss_g_adv).detach().item())
                            if lam_fm_hifi > 0.0:
                                loss_dict["gan_adv_fm"] = float((lam_fm_hifi * loss_g_fm).detach().item())
                        except Exception as _e_hifi_g:
                            _dbg = (
                                os.environ.get("DBG_GAN_ADV", "0") == "1"
                                or os.environ.get("DBG_HIFI_ADV", "0") == "1"
                                or os.environ.get("DBG_GAN_GRAD", "0") == "1"
                            )
                            if _dbg:
                                print(f"[GAN-ADV] G update skipped: {_e_hifi_g}")
                            loss_dict.setdefault("gan_adv_g", 0.0)
                            loss_dict.setdefault("gan_adv_fm", 0.0)

                loss.backward()
                optimizer.step()

            log_every = int(getattr(cfg, "log_every_steps", 100) or 100)
            if log_every > 0 and global_step % log_every == 0:
                # Before the aggregated loss line, print per-sample
                # SNR/BER for up to the first 10 audio segments in the
                # current batch, using the internal JSCC hash bits.
                _log_train_sample_snr_ber(
                    out=out,
                    epoch_val=epoch,
                    step_val=global_step,
                    max_samples=10,
                )

                msg = f"epoch={epoch} step={global_step} loss={float(loss.item()):.4f}"
                for k, v in loss_dict.items():
                    msg += f" {k}={v:.4f}"
                print(msg)

            # Optional wandb scalar logging
            if use_wandb and wandb is not None:
                try:
                    log_freq = int(getattr(cfg, "wandb_log_freq", 10) or 10)
                    if log_freq > 0 and global_step % log_freq == 0:
                        log_data: Dict[str, float] = {"total": float(loss.detach().item())}
                        for k, v in loss_dict.items():
                            try:
                                log_data[k] = float(v)
                            except Exception:
                                continue
                        wandb.log(log_data, step=int(global_step))
                        # Update sliding-window correlation diagnostics.
                        if corr_window is not None:
                            corr_window.update(log_data)
                            corr_window.maybe_log(int(global_step))
                except Exception as _we:
                    print(f"[wandb] WARNING: log failed at step {global_step}: {_we}")

            # Periodic visualization: waveform comparisons for a few
            # samples from the current batch, using the same utilities as
            # the main Stage2.5 script.
            if (
                getattr(cfg, "viz_every_steps", 0) > 0
                and global_step % int(getattr(cfg, "viz_every_steps", 0)) == 0
                and "audio_hat" in out
            ):
                try:
                    audio_real_b = out["audio"].detach().cpu()
                    audio_gen_b = out["audio_hat"].detach().cpu()
                    create_batch_comparison_plots(
                        audio_real_batch=audio_real_b,
                        audio_gen_batch=audio_gen_b,
                        save_dir=cfg.viz_dir,
                        step=global_step,
                        max_samples=int(getattr(cfg, "viz_max_samples", 2)),
                        sr=16000,
                    )
                    if bool(getattr(cfg, "viz_source_controls", True)):
                        _save_source_control_plots_for_batch(
                            out=out,
                            save_dir=cfg.viz_dir,
                            step=global_step,
                            max_samples=int(getattr(cfg, "viz_max_samples", 2)),
                            sr=16000,
                            hop_length=160,
                        )
                    if bool(getattr(cfg, "viz_vocoder_internals", True)):
                        _save_vocoder_internal_plots_for_batch(
                            out=out,
                            save_dir=cfg.viz_dir,
                            step=global_step,
                            max_samples=int(getattr(cfg, "viz_max_samples", 2)),
                            sr=16000,
                            hop_length=160,
                        )
                    save_comparison_audio_samples(
                        audio_real_batch=audio_real_b,
                        audio_gen_batch=audio_gen_b,
                        save_dir=cfg.viz_dir,
                        step=global_step,
                        max_samples=int(getattr(cfg, "viz_max_samples", 2)),
                        sr=16000,
                            )
                except Exception as e:
                    print(f"[Simplified] visualization failed at step {global_step}: {e}")

                # Optional: debug CREPE F0 MSE and Bark HF MSE on a
                # small subset of the current batch, primarily for
                # A/B experiments when tuning adversarial components.
                try:
                    _debug_log_crepe_f0_and_bark_mse(
                        audio_real_b.to(device=device),
                        audio_gen_b.to(device=device),
                        device=device,
                        global_step=global_step,
                    )
                except Exception as _e_dbg_metric:
                    if os.environ.get("DBG_F0_BARK", "0") == "1":
                        print(f"[DBG_METRIC] F0/Bark debug failed: {_e_dbg_metric}")

                # Optional: BFCC/Bark-domain validation directly on the
                # forward path (audio/audio_hat → WaveToBFCC), independent
                # of bit-only eval / RVQ.
                if bool(getattr(cfg, "bfcc_forward_eval", False)):
                    _run_forward_bfcc_eval_for_batch(
                        model=model,
                        out=out,
                        cfg=cfg,
                        device=device,
                        global_step_val=global_step,
                    )

                # Optional: bit-only eval path (encode_hash_codec +
                # decode_from_bits_offline) on a few samples from the
                # same batch, to inspect the pure bits -> audio quality.
                if bool(getattr(cfg, "bit_only_eval", False)):
                    _run_bit_only_eval_for_batch(
                        model=model,
                        batch=batch,
                        out=out,
                        cfg=cfg,
                        channel_sim=channel_sim,
                        device=device,
                        global_step_val=global_step,
                        do_visualization=True,
                        snr_grid_all=None,
                    )

                # Debug: print hash_content_stats grad norms when DBG_STATS=1
                try:
                    import os as _os_dbg2
                    if _os_dbg2.environ.get("DBG_STATS", "0") == "1" and global_step % int(getattr(cfg, "log_every_steps", 10) or 10) == 0:
                        try:
                            print("[STATS-GRAD][train] hash_content_stats parameter grad norms:")
                            for name, param in model.named_parameters():
                                if "hash_content_stats" in name and isinstance(param, torch.nn.Parameter):
                                    g = param.grad
                                    if g is None:
                                        gnorm = None
                                    else:
                                        gnorm = float(g.detach().norm().item())
                                    print(f"  {name}: grad_norm={gnorm}")
                        except Exception as _e_grad2:
                            print(f"[STATS-GRAD][train] WARNING: grad inspection failed: {_e_grad2}")
                except Exception:
                    pass

                # Checkpoint saving: mirror v3 naming (step+epoch) while keeping
            # a clear suffix for the simplified branch.
            if cfg.save_every_steps > 0 and global_step > 0 and global_step % cfg.save_every_steps == 0:
                ckpt_path = os.path.join(
                    cfg.ckpt_dir,
                    f"checkpoint_step_{global_step:06d}_epoch_{epoch:02d}.pth",
                )
                state = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "cfg": asdict(cfg),
                    "global_step": global_step,
                    "epoch": epoch,
                }
                # Optionally include HiFi-GAN discriminator states when enabled.
                if mpd is not None:
                    state["mpd"] = mpd.state_dict()
                if msd is not None:
                    state["msd"] = msd.state_dict()
                if bark_disc is not None:
                    state["bark_disc"] = bark_disc.state_dict()
                if f0p_disc is not None:
                    state["f0p_disc"] = f0p_disc.state_dict()
                if optimizer_hifi_disc is not None:
                    state["optimizer_hifi_disc"] = optimizer_hifi_disc.state_dict()

                torch.save(state, ckpt_path)
                print(f"[Simplified] Saved checkpoint to {ckpt_path}")

            global_step += 1
            if content_frozen:
                content_steps_since_resume += 1
            # GAN warm-up counter increases with each optimisation step in
            # this run, independent of the absolute global_step restored
            # from the checkpoint.
            if (
                mpd is not None
                and msd is not None
                and (float(getattr(cfg, "lambda_gan_adv", 0.0)) > 0.0
                     or float(getattr(cfg, "lambda_gan_fm", 0.0)) > 0.0)
            ):
                gan_steps_since_resume += 1


def _parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="DBP-JSCC training")
    parser.add_argument("--data_root", type=str, default=None, help="Training data root directory")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--sequence_length", type=int, default=200)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Enable AMP autocast for selected paths (e.g., HiFi-GAN adversarial loss)",
    )

    parser.add_argument("--snr_min_db", type=float, default=-5.0)
    parser.add_argument("--snr_max_db", type=float, default=15.0)
    parser.add_argument("--snr_step_db", type=float, default=1.0,
                        help="SNR grid step (dB) for ChannelSimulator; <=0 uses legacy discrete grid")

    parser.add_argument("--with_hash", action="store_true", help="Enable hash/RVQ bottlenecks (recommended)")
    parser.add_argument("--quantizer_type", type=str, default="rvq", choices=["hash", "rvq"])
    parser.add_argument("--hash_bits_content", type=int, default=16)
    parser.add_argument("--hash_bits_f0", type=int, default=None)
    parser.add_argument("--rvq_nq_content", type=int, default=2)
    parser.add_argument("--rvq_nq_f0", type=int, default=None)
    parser.add_argument("--rvq_beta", type=float, default=0.25)
    parser.add_argument("--d_s_content", type=int, default=8)
    parser.add_argument("--freq_downsample_stages", type=int, default=2)
    parser.add_argument("--eq_fading", action="store_true")
    # VMamba content branch configuration
    parser.add_argument("--vm_channels", type=str, default=None,
                        help="Comma-separated VMamba channels, e.g. '16,24,32,48'")
    parser.add_argument("--vm_depths", type=str, default=None,
                        help="Comma-separated VMamba depths, e.g. '2,2,3,2'")
    parser.add_argument("--vm_channel_adaptive", type=str, default="no",
                        choices=["no", "ca", "attn", "ssm"],
                        help="Channel adaptation mode for VMamba content branch")
    parser.add_argument("--vm_lightweight_config", type=str, default="all_native",
                        help="VMamba lightweight config preset (mirrors v3 script choices)")

    parser.add_argument(
        "--content_only",
        action="store_true",
        help=(
            "Content-only mode: skip F0/VUV, ceps, L2H and vocoder; "
            "only train the content branch with Bark/BFCC-domain losses "
            "(mirrors v3 content_only routing)"
        ),
    )

    # L2H (low→high mel refinement) configuration
    parser.add_argument("--with_l2h", action="store_true",
                        help="Enable DeCo-style L2H mel refinement")
    parser.add_argument("--l2h_low_bins", type=int, default=18,
                        help="Number of low-frequency mel bins used as L2H input (default 18 to match v3 L2H config)")
    parser.add_argument("--use_l2h_flow", action="store_true",
                        help="Enable optional conditional flow for L2H NLL regularization")
    parser.add_argument("--l2h_flow_hidden", type=int, default=128,
                        help="Hidden size for L2H conditional flow")
    parser.add_argument("--l2h_flow_n_flows", type=int, default=4,
                        help="Number of flow steps in L2H conditional flow")
    parser.add_argument("--deco_l2h", action="store_true",
                        help="Use DeCo-style L2H refiner (mel_low + F0/VUV → HF) instead of legacy L2H")
    parser.add_argument("--deco_l2h_hidden", type=int, default=64,
                        help="Hidden size for DeCoL2HRefiner")
    parser.add_argument("--deco_l2h_blocks", type=int, default=3,
                        help="Number of AdaLN blocks inside DeCoL2HRefiner")

    # L2H residual/decorrelation knobs（主要供主 Stage2.5 模型使用）；
    # 在简化版中，当启用 --with_l2h 且这些值保持默认时，会在
    # _parse_args 中自动设置为经验值。
    parser.add_argument("--lambda_l2h_resid", type=float, default=0.0,
                        help="Weight for L2H high-frequency residual loss (full v3 path)")
    parser.add_argument("--lambda_l2h_decor", type=float, default=0.0,
                        help="Weight for L2H decorrelation loss between low/high bands")
    parser.add_argument("--l2h_improve_margin", type=float, default=0.0,
                        help="Improvement margin for L2H residual vs baseline HF error")

    parser.add_argument("--lambda_wave", type=float, default=0.7)
    parser.add_argument(
        "--lambda_wave_mag",
        type=float,
        default=0.0,
        help=(
            "Multi-resolution STFT log-magnitude L1 weight; "
            "set >0 to use SC+log-mag instead of spectral convergence only",
        ),
    )
    parser.add_argument(
        "--lambda_wave_rms",
        type=float,
        default=0.0,
        help=(
            "Waveform frame-RMS envelope L1 loss weight (log10 RMS over "
            "short frames); acts as a coarse loudness anchor.",
        ),
    )
    parser.add_argument("--lambda_mel", type=float, default=0.3)
    parser.add_argument("--lambda_mel_l1", type=float, default=0.3,
                        help="Mel L1 brightness/detail anchor weight")
    parser.add_argument(
        "--lambda_mel_energy",
        type=float,
        default=0.0,
        help="Global mel energy/brightness anchor weight (matches v3 lambda_mel_energy)",
    )
    parser.add_argument(
        "--lambda_mel_stats",
        type=float,
        default=0.0,
        help=(
            "Weight for stats-bits reconstruction loss on mel_mean/mel_std "
            "(mel_mean_hat/mel_std_hat vs GT mel_mean/mel_std; only used when with_hash=True)",
        ),
    )
    parser.add_argument("--lambda_silence_mel", type=float, default=0.0,
                        help="Silence-frame HF mel suppression weight (content-only)")
    parser.add_argument("--lambda_ceps", type=float, default=0.3)
    parser.add_argument("--lambda_f0", type=float, default=0.8)
    parser.add_argument("--lambda_f0_env", type=float, default=0.0,
                        help="F0 envelope hinge loss weight")
    parser.add_argument("--f0_env_margin_cents", type=float, default=80.0,
                        help="Allowed F0 envelope deviation (cents) before penalty")
    parser.add_argument("--f0_env_window", type=int, default=3,
                        help="Smoothing window (frames) for F0 envelope")
    parser.add_argument("--f0_env_alpha", type=float, default=0.5,
                        help="Reserved: env blend factor (kept for parity)")
    parser.add_argument("--f0_presence_gamma", type=float, default=0.2,
                        help="Reserved: F0 presence loss strength (kept for parity)")
    parser.add_argument("--lambda_vuv", type=float, default=0.5)
    parser.add_argument("--lambda_vuv_bce", type=float, default=0.0,
                        help="VUV prob BCE loss weight (frame_corr_hat vs label)")
    parser.add_argument("--lambda_vq_c", type=float, default=0.05)
    parser.add_argument("--lambda_vq_f", type=float, default=0.01)
    parser.add_argument("--lambda_vq_stats", type=float, default=0.0,
                        help="Stats RVQ VQ loss weight (hash_content_stats)")
    parser.add_argument("--lambda_f0_smooth", type=float, default=0.0,
                        help="F0 second-order smoothness loss weight")
    parser.add_argument("--lambda_ceps_hi", type=float, default=0.0,
                        help="High-order cepstrum supervision weight")
    parser.add_argument("--ceps_hi_start", type=int, default=10,
                        help="Start index for high-order cepstra supervision")
    parser.add_argument("--lambda_mel_hp", type=float, default=0.0,
                        help="HF mel patch MS-SSIM loss weight")
    parser.add_argument("--lambda_mel_hf_l1", type=float, default=0.0,
                        help="HF mel energy L1 loss weight (amplitude matching in HF band)")
    parser.add_argument("--mel_hp_low_bins", type=int, default=16,
                        help="Low-bin cutoff for mel HP loss (only bins >= cutoff are used)")
    parser.add_argument("--silence_hf_low_bins", type=int, default=16,
                        help="Low-bin cutoff for silence HF mel suppression (bins >= cutoff are used)")
    parser.add_argument("--silence_mel_thr_db", type=float, default=-35.0,
                        help="Silence threshold on high-band log10 mel energy (dB domain)")
    parser.add_argument("--lambda_c_entropy", type=float, default=0.0,
                        help="Content bit entropy regularization weight")
    parser.add_argument("--content_entropy_target_frac", type=float, default=0.5,
                        help="Target entropy fraction for content bits")
    parser.add_argument("--lambda_f0_entropy", type=float, default=0.0,
                        help="F0 bit entropy regularization weight")
    parser.add_argument("--f0_entropy_target_frac", type=float, default=0.5,
                        help="Target entropy fraction for F0 bits")
    parser.add_argument("--lambda_bit_balance_c", type=float, default=0.0,
                        help="Bit-balance regularizer weight for content bits")
    parser.add_argument("--lambda_hf_stft", type=float, default=0.0,
                        help="High-frequency STFT magnitude loss weight")
    parser.add_argument("--lambda_texture_protect", type=float, default=0.0,
                        help="Residual texture protection loss weight")
    parser.add_argument("--texture_hf_start", type=int, default=40,
                        help="Start mel bin for HF residual texture loss")
    parser.add_argument(
        "--stft_hf_start_hz",
        type=float,
        default=0.0,
        help=(
            "Advanced: HF boundary for optional MR-STFT re-weighting. "
            "Default 0 means standard, unweighted STFT loss.",
        ),
    )
    parser.add_argument(
        "--stft_hf_weight",
        type=float,
        default=1.0,
        help=(
            "Advanced: relative MR-STFT mag weight for f>=stft_hf_start_hz "
            "(0..1). Default 1.0 keeps standard STFT behaviour.",
        ),
    )
    parser.add_argument(
        "--lambda_bark_hf_energy",
        type=float,
        default=0.0,
        help=(
            "Extra HF energy clamp weight on BarkHF maps: only penalises "
            "frames where fake HF Bark energy exceeds real+margin.",
        ),
    )
    parser.add_argument(
        "--lambda_bark_hf_l1",
        type=float,
        default=0.0,
        help=(
            "HF Bark-domain L1 anchor weight on (gated + normalised) HF Bark "
            "maps used by BarkHF discriminator.",
        ),
    )
    parser.add_argument(
        "--bark_hf_energy_margin_db",
        type=float,
        default=3.0,
        help=(
            "Margin (dB) before BarkHF energy clamp activates; "
            "3 dB ≈ 0.3 in log10-energy units.",
        ),
    )
    parser.add_argument(
        "--adv_scope",
        type=str,
        default="full",
        help=(
            "Adversarial gradient scope: 'full' (default) applies GAN "
            "losses to all trainable parameters; 'vocoder_l2h_ceps' "
            "restricts GAN gradients to vocoder, L2H and mel→ceps "
            "mapping modules.",
        ),
    )
    # Adversarial loss on raw waveform (MPD + MSD + BarkHF + F0Period)
    parser.add_argument(
        "--lambda_gan_adv",
        type=float,
        default=0.0,
        help="Generator adversarial loss weight for waveform GAN",
    )
    parser.add_argument(
        "--lambda_gan_fm",
        type=float,
        default=0.0,
        help="Feature-matching loss weight for waveform GAN",
    )
    parser.add_argument(
        "--gan_adv_warmup_steps",
        type=int,
        default=0,
        help="Steps before enabling GAN discriminator updates",
    )
    parser.add_argument(
        "--gan_disc_lr",
        type=float,
        default=1e-4,
        help="Learning rate for waveform GAN discriminators",
    )
    parser.add_argument(
        "--gan_disc_lr_decay",
        type=float,
        default=1.0,
        help=(
            "Multiplicative LR decay factor for discriminator after "
            "gan_adv_warmup_steps (1.0 = no decay).",
        ),
    )
    parser.add_argument(
        "--gan_adv_crop_len",
        type=int,
        default=16000,
        help="Crop length (samples) for GAN discriminators (0=use full waveform)",
    )
    parser.add_argument("--lambda_l2h", type=float, default=0.0,
                        help="Optional L2H high-frequency residual loss weight")
    parser.add_argument(
        "--lambda_bit_only_silence",
        type=float,
        default=0.0,
        help="Bit-only BFCC silence loss weight (bits→audio→BFCC path)",
    )

    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--vocoder_ckpt", type=str, default=None,
                        help="Optional pretrained vocoder checkpoint path")
    parser.add_argument(
        "--reload_vocoder_after_resume",
        action="store_true",
        help=(
            "Re-load vocoder weights from --vocoder_ckpt after resume, "
            "overriding vocoder core (mirrors full Stage2.5 script)"
        ),
    )
    parser.add_argument(
        "--content_ckpt",
        type=str,
        default=None,
        help=(
            "Optional checkpoint path whose content JSCC branch "
            "(content_vmamba/hash_content/...) will be re-loaded "
            "after --resume when --reload_content_after_resume is set."
        ),
    )
    parser.add_argument(
        "--reload_content_after_resume",
        action="store_true",
        help=(
            "Re-load content JSCC branch from --content_ckpt after resume, "
            "overriding content_vmamba/hash_content/... while keeping other "
            "modules from --resume."
        ),
    )
    parser.add_argument(
        "--content_warmup_steps",
        type=int,
        default=0,
        help=(
            "Number of global steps to keep content JSCC branch frozen "
            "before unfreezing it for fine-tuning",
        ),
    )
    parser.add_argument(
        "--freeze_vocoder_all",
        action="store_true",
        help="Freeze vocoder parameters for all training steps",
    )
    parser.add_argument(
        "--freeze_content_all",
        action="store_true",
        help="Freeze the entire content JSCC branch for all training steps",
    )
    parser.add_argument(
        "--unfreeze_ceps_map",
        action="store_true",
        help=(
            "When used together with --freeze_content_all, keep mel18_to_ceps/"
            "hf2ceps (BFCC→ceps mapping) trainable while freezing other "
            "content-branch modules."
        ),
    )
    parser.add_argument(
        "--unfreeze_l2h",
        action="store_true",
        help=(
            "When used together with --freeze_content_all, keep the L2H "
            "stack (deco_l2h_refiner/l2h_flow) trainable while freezing "
            "other content-branch modules."
        ),
    )
    parser.add_argument(
        "--unfreeze_stats",
        action="store_true",
        help=(
            "When used together with --freeze_content_all, keep "
            "hash_content_stats (stats bottleneck) trainable while "
            "freezing other content-branch modules.",
        ),
    )
    parser.add_argument(
        "--vocoder_strict_vuv_gate",
        dest="vocoder_strict_vuv_gate",
        action="store_true",
        default=None,
        help="Enable strict VUV gate mapping inside FARGAN vocoder.",
    )
    parser.add_argument(
        "--no_vocoder_strict_vuv_gate",
        dest="vocoder_strict_vuv_gate",
        action="store_false",
        help="Disable strict VUV gate mapping inside FARGAN vocoder.",
    )
    parser.add_argument(
        "--vocoder_final_voicing_gate",
        dest="vocoder_final_voicing_gate",
        action="store_true",
        default=None,
        help="Enable final output voicing gate before FARGAN sig_out.",
    )
    parser.add_argument(
        "--no_vocoder_final_voicing_gate",
        dest="vocoder_final_voicing_gate",
        action="store_false",
        help="Disable final output voicing gate before FARGAN sig_out.",
    )
    parser.add_argument(
        "--vocoder_final_voicing_gate_floor",
        type=float,
        default=None,
        help="Floor for FARGAN final voicing gate (0..1).",
    )
    parser.add_argument(
        "--vocoder_final_voicing_gate_gamma",
        type=float,
        default=None,
        help="Gamma shaping for FARGAN final voicing gate (>0).",
    )
    parser.add_argument(
        "--vocoder_silence_gate",
        dest="vocoder_silence_gate",
        action="store_true",
        default=None,
        help="Enable c0-based silence gate on final FARGAN output.",
    )
    parser.add_argument(
        "--no_vocoder_silence_gate",
        dest="vocoder_silence_gate",
        action="store_false",
        help="Disable c0-based silence gate on final FARGAN output.",
    )
    parser.add_argument(
        "--vocoder_silence_gate_floor",
        type=float,
        default=None,
        help="Floor for FARGAN silence gate (0..1).",
    )
    parser.add_argument(
        "--vocoder_silence_energy_thr_db",
        type=float,
        default=None,
        help="c0-derived silence threshold in dB for FARGAN silence gate.",
    )
    parser.add_argument(
        "--vocoder_silence_gate_width_db",
        type=float,
        default=None,
        help="Transition width in dB for FARGAN silence gate sigmoid.",
    )
    parser.add_argument(
        "--vocoder_pitch_gain_scale",
        type=float,
        default=None,
        help="Experimental multiplicative scale on FARGAN pitch_gain after sigmoid.",
    )
    parser.add_argument(
        "--vocoder_sig_core_scale",
        type=float,
        default=None,
        help="Experimental multiplicative scale on FARGAN sig_core before final gain.",
    )
    parser.add_argument(
        "--viz_source_controls",
        dest="viz_source_controls",
        action="store_true",
        default=None,
        help="Export extra source-control plots at visualization steps.",
    )
    parser.add_argument(
        "--no_viz_source_controls",
        dest="viz_source_controls",
        action="store_false",
        help="Disable extra source-control plots at visualization steps.",
    )
    parser.add_argument(
        "--oracle_swap_source_controls",
        type=str,
        default=None,
        help=(
            "Diagnostic oracle swap for waveform-side controls. "
            "Supported tokens: none, pitch, period, frame_corr, vuv, gain, c0, ceps, all. "
            "Multiple tokens can be comma-separated."
        ),
    )
    parser.add_argument(
        "--viz_vocoder_internals",
        dest="viz_vocoder_internals",
        action="store_true",
        default=None,
        help="Export extra vocoder-internal plots at visualization steps.",
    )
    parser.add_argument(
        "--no_viz_vocoder_internals",
        dest="viz_vocoder_internals",
        action="store_false",
        help="Disable extra vocoder-internal plots at visualization steps.",
    )
    parser.add_argument("--viz_dir", type=str, default=None)
    parser.add_argument("--viz_every_steps", type=int, default=1000)
    parser.add_argument("--viz_max_samples", type=int, default=2)
    parser.add_argument("--out_dir", type=str, default=None,
                        help="If set, override ckpt_dir/viz_dir with out_dir/checkpoints and out_dir/viz")
    parser.add_argument("--save_every_steps", type=int, default=1000)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--log_every_steps", type=int, default=100,
                        help="Print training loss every N steps")
    parser.add_argument("--bit_only_eval", action="store_true",
                        help="Enable bit-only decode eval at viz steps")
    parser.add_argument("--bit_only_eval_max_samples", type=int, default=2,
                        help="Max samples per batch for bit-only eval")
    parser.add_argument(
        "--bfcc_forward_eval",
        action="store_true",
        help=(
            "Enable BFCC/Bark-domain validation on the forward path at "
            "viz steps (audio/audio_hat → WaveToBFCC images)",
        ),
    )
    parser.add_argument(
        "--bfcc_forward_max_samples",
        type=int,
        default=2,
        help="Max samples per batch for forward BFCC evaluation",
    )
    parser.add_argument(
        "--ignore_stats_in_mel",
        action="store_true",
        help=(
            "Ignore stats bits when reconstructing mel in forward_with_hash/"
            "decode_from_bits_offline (mel_hat uses content_vmamba output "
            "directly without applying mean/std from hash_content_stats)",
        ),
    )
    parser.add_argument(
        "--stats_only",
        action="store_true",
        help=(
            "Enable stats-only fine-tuning mode: freeze all modules except "
            "hash_content_stats and train it to reconstruct (mel_mean, mel_std) "
            "in the normalised space via an L1 loss.",
        ),
    )
    parser.add_argument(
        "--stats_only_lr",
        type=float,
        default=1e-4,
        help="Learning rate for stats-only fine-tuning (hash_content_stats)",
    )
    parser.add_argument(
        "--stats_only_max_steps",
        type=int,
        default=10000,
        help="Maximum optimisation steps in stats-only fine-tuning mode",
    )
    parser.add_argument(
        "--vq_only_train",
        action="store_true",
        help=(
            "Freeze all parameters except RVQ/Hash bottlenecks "
            "(hash_content/hash_f0vuv) so that only VQ codebooks are "
            "updated",
        ),
    )
    parser.add_argument("--only_eval", action="store_true",
                        help="Eval-only mode: run one pass to populate bit_only_metrics.csv without training")
    parser.add_argument(
        "--pipeline_probe",
        action="store_true",
        help=(
            "Enable pipeline probe mode: run a small number of samples "
            "through train/eval forward paths, print component-wise "
            "feature statistics, then exit without training"
        ),
    )
    parser.add_argument(
        "--pipeline_probe_num_samples",
        type=int,
        default=10,
        help="Number of audio segments to use in pipeline probe mode",
    )
    parser.add_argument(
        "--use_learned_energy_calib",
        action="store_true",
        help=(
            "Enable learned energy calibration head inside DualBranchBarkJSCC "
            "(predict per-sample c0 offset from mel_used; no GT FARGAN ceps)"
        ),
    )
    parser.add_argument(
        "--fsk_ber_table",
        type=str,
        default=None,
        help=(
            "Optional path to JSCC+FSK BER(SNR) JSON table; "
            "overrides JSCC_FSK_BER_TABLE for this run"
        ),
    )
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable wandb logging (if wandb is installed)")
    parser.add_argument("--wandb_project", type=str, default="DBP-JSCC",
                        help="wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="wandb run name")
    parser.add_argument("--wandb_log_freq", type=int, default=10,
                        help="Log to wandb every N steps")

    args = parser.parse_args()
    cfg = TrainingConfig()

    def _parse_int_list(val: Optional[str]) -> Optional[List[int]]:
        if val is None:
            return None
        parts = [p.strip() for p in val.split(",")]
        ints: List[int] = []
        for p in parts:
            if not p:
                continue
            try:
                ints.append(int(p))
            except ValueError:
                pass
        return ints or None

    if args.data_root is not None:
        cfg.data_root = args.data_root
    cfg.batch_size = args.batch_size
    cfg.sequence_length = args.sequence_length
    cfg.num_epochs = args.num_epochs
    cfg.lr = args.lr
    cfg.use_amp = bool(getattr(args, "use_amp", False))
    cfg.snr_min_db = args.snr_min_db
    cfg.snr_max_db = args.snr_max_db
    cfg.with_hash = bool(args.with_hash)
    cfg.quantizer_type = args.quantizer_type
    cfg.hash_bits_content = args.hash_bits_content
    cfg.hash_bits_f0 = args.hash_bits_f0
    cfg.rvq_nq_content = args.rvq_nq_content
    cfg.rvq_nq_f0 = args.rvq_nq_f0
    cfg.rvq_beta = args.rvq_beta
    cfg.d_s_content = args.d_s_content
    cfg.freq_downsample_stages = args.freq_downsample_stages
    cfg.eq_fading = bool(args.eq_fading)
    cfg.content_only = bool(getattr(args, "content_only", False))
    cfg.use_learned_energy_calib = bool(getattr(args, "use_learned_energy_calib", False))
    cfg.vm_channels = _parse_int_list(args.vm_channels)
    cfg.vm_depths = _parse_int_list(args.vm_depths)
    cfg.vm_channel_adaptive = args.vm_channel_adaptive
    cfg.vm_lightweight_config = args.vm_lightweight_config
    cfg.with_l2h = bool(args.with_l2h)
    cfg.l2h_low_bins = args.l2h_low_bins
    cfg.use_l2h_flow = bool(args.use_l2h_flow)
    cfg.l2h_flow_hidden = args.l2h_flow_hidden
    cfg.l2h_flow_n_flows = args.l2h_flow_n_flows
    cfg.deco_l2h = bool(args.deco_l2h)
    cfg.deco_l2h_hidden = args.deco_l2h_hidden
    cfg.deco_l2h_blocks = args.deco_l2h_blocks

    cfg.lambda_wave = args.lambda_wave
    cfg.lambda_wave_rms = float(getattr(args, "lambda_wave_rms", 0.0))
    # MR-STFT log-magnitude term; default 0.0 keeps previous behaviour
    # (SC-only) unless the user explicitly enables it via CLI.
    cfg.lambda_wave_mag = float(getattr(args, "lambda_wave_mag", 0.0))
    cfg.lambda_mel = args.lambda_mel
    cfg.lambda_mel_l1 = args.lambda_mel_l1
    cfg.lambda_mel_energy = float(getattr(args, "lambda_mel_energy", 0.0))
    cfg.lambda_mel_stats = float(getattr(args, "lambda_mel_stats", 0.0))
    cfg.lambda_silence_mel = args.lambda_silence_mel
    cfg.lambda_ceps = args.lambda_ceps
    cfg.lambda_f0 = args.lambda_f0
    cfg.lambda_f0_env = args.lambda_f0_env
    cfg.lambda_f0_smooth = args.lambda_f0_smooth
    cfg.f0_env_margin_cents = args.f0_env_margin_cents
    cfg.f0_env_window = args.f0_env_window
    cfg.f0_env_alpha = args.f0_env_alpha
    cfg.f0_presence_gamma = args.f0_presence_gamma
    cfg.lambda_vuv = args.lambda_vuv
    cfg.lambda_vuv_bce = args.lambda_vuv_bce
    cfg.lambda_vq_c = args.lambda_vq_c
    cfg.lambda_vq_f = args.lambda_vq_f
    cfg.lambda_vq_stats = float(getattr(args, "lambda_vq_stats", 0.0))
    cfg.lambda_ceps_hi = args.lambda_ceps_hi
    cfg.ceps_hi_start = args.ceps_hi_start
    cfg.lambda_mel_hp = args.lambda_mel_hp
    cfg.lambda_mel_hf_l1 = args.lambda_mel_hf_l1
    cfg.mel_hp_low_bins = args.mel_hp_low_bins
    cfg.silence_hf_low_bins = args.silence_hf_low_bins
    cfg.silence_mel_thr_db = args.silence_mel_thr_db
    cfg.lambda_c_entropy = args.lambda_c_entropy
    cfg.content_entropy_target_frac = args.content_entropy_target_frac
    cfg.lambda_f0_entropy = args.lambda_f0_entropy
    cfg.f0_entropy_target_frac = args.f0_entropy_target_frac
    cfg.lambda_bit_balance_c = args.lambda_bit_balance_c
    cfg.lambda_hf_stft = args.lambda_hf_stft
    cfg.lambda_texture_protect = args.lambda_texture_protect
    cfg.texture_hf_start = args.texture_hf_start
    cfg.stft_hf_start_hz = float(getattr(args, "stft_hf_start_hz", 0.0))
    cfg.stft_hf_weight = float(getattr(args, "stft_hf_weight", 0.5))
    cfg.lambda_bark_hf_energy = float(getattr(args, "lambda_bark_hf_energy", 0.0))
    cfg.bark_hf_energy_margin_db = float(getattr(args, "bark_hf_energy_margin_db", 3.0))
    cfg.lambda_bark_hf_l1 = float(getattr(args, "lambda_bark_hf_l1", 0.0))
    cfg.adv_scope = str(getattr(args, "adv_scope", "full"))
    cfg.lambda_gan_adv = float(getattr(args, "lambda_gan_adv", 0.0))
    cfg.lambda_gan_fm = float(getattr(args, "lambda_gan_fm", 0.0))
    cfg.gan_adv_warmup_steps = int(getattr(args, "gan_adv_warmup_steps", 0) or 0)
    cfg.gan_disc_lr = float(getattr(args, "gan_disc_lr", 1e-4))
    cfg.gan_disc_lr_decay = float(getattr(args, "gan_disc_lr_decay", 1.0))
    cfg.gan_adv_crop_len = int(getattr(args, "gan_adv_crop_len", 16000))
    cfg.lambda_gan_adv = float(getattr(args, "lambda_gan_adv", 0.0))
    cfg.lambda_gan_fm = float(getattr(args, "lambda_gan_fm", 0.0))
    cfg.gan_adv_warmup_steps = int(getattr(args, "gan_adv_warmup_steps", 0))
    cfg.gan_disc_lr = float(getattr(args, "gan_disc_lr", 1e-4))
    cfg.gan_adv_crop_len = int(getattr(args, "gan_adv_crop_len", 16000))
    cfg.lambda_bit_only_silence = args.lambda_bit_only_silence
    cfg.lambda_l2h = args.lambda_l2h
    cfg.lambda_l2h_resid = args.lambda_l2h_resid
    cfg.lambda_l2h_decor = args.lambda_l2h_decor
    cfg.l2h_improve_margin = args.l2h_improve_margin

    if args.ckpt_dir is not None:
        cfg.ckpt_dir = args.ckpt_dir
    if args.vocoder_ckpt is not None:
        cfg.vocoder_ckpt = args.vocoder_ckpt
    cfg.reload_vocoder_after_resume = bool(getattr(args, "reload_vocoder_after_resume", False))
    if getattr(args, "content_ckpt", None) is not None:
        cfg.content_ckpt = args.content_ckpt
    cfg.reload_content_after_resume = bool(getattr(args, "reload_content_after_resume", False))
    cfg.content_warmup_steps = int(getattr(args, "content_warmup_steps", 0) or 0)
    cfg.freeze_vocoder_all = bool(getattr(args, "freeze_vocoder_all", False))
    cfg.freeze_content_all = bool(getattr(args, "freeze_content_all", False))
    cfg.unfreeze_ceps_map = bool(getattr(args, "unfreeze_ceps_map", False))
    cfg.unfreeze_l2h = bool(getattr(args, "unfreeze_l2h", False))
    cfg.unfreeze_stats = bool(getattr(args, "unfreeze_stats", False))
    if getattr(args, "vocoder_strict_vuv_gate", None) is not None:
        cfg.vocoder_strict_vuv_gate = bool(args.vocoder_strict_vuv_gate)
    if getattr(args, "vocoder_final_voicing_gate", None) is not None:
        cfg.vocoder_final_voicing_gate = bool(args.vocoder_final_voicing_gate)
    if getattr(args, "vocoder_final_voicing_gate_floor", None) is not None:
        cfg.vocoder_final_voicing_gate_floor = float(args.vocoder_final_voicing_gate_floor)
    if getattr(args, "vocoder_final_voicing_gate_gamma", None) is not None:
        cfg.vocoder_final_voicing_gate_gamma = float(args.vocoder_final_voicing_gate_gamma)
    if getattr(args, "vocoder_silence_gate", None) is not None:
        cfg.vocoder_silence_gate = bool(args.vocoder_silence_gate)
    if getattr(args, "vocoder_silence_gate_floor", None) is not None:
        cfg.vocoder_silence_gate_floor = float(args.vocoder_silence_gate_floor)
    if getattr(args, "vocoder_silence_energy_thr_db", None) is not None:
        cfg.vocoder_silence_energy_thr_db = float(args.vocoder_silence_energy_thr_db)
    if getattr(args, "vocoder_silence_gate_width_db", None) is not None:
        cfg.vocoder_silence_gate_width_db = float(args.vocoder_silence_gate_width_db)
    if getattr(args, "vocoder_pitch_gain_scale", None) is not None:
        cfg.vocoder_pitch_gain_scale = float(args.vocoder_pitch_gain_scale)
    if getattr(args, "vocoder_sig_core_scale", None) is not None:
        cfg.vocoder_sig_core_scale = float(args.vocoder_sig_core_scale)
    if getattr(args, "viz_source_controls", None) is not None:
        cfg.viz_source_controls = bool(args.viz_source_controls)
    if getattr(args, "oracle_swap_source_controls", None) is not None:
        cfg.oracle_swap_source_controls = str(args.oracle_swap_source_controls)
    if getattr(args, "viz_vocoder_internals", None) is not None:
        cfg.viz_vocoder_internals = bool(args.viz_vocoder_internals)
    if args.viz_dir is not None:
        cfg.viz_dir = args.viz_dir
    cfg.viz_every_steps = args.viz_every_steps
    cfg.viz_max_samples = args.viz_max_samples
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
    cfg.save_every_steps = args.save_every_steps
    if args.resume is not None:
        cfg.resume = args.resume
    cfg.log_every_steps = args.log_every_steps
    cfg.bit_only_eval = bool(args.bit_only_eval)
    cfg.bit_only_eval_max_samples = args.bit_only_eval_max_samples
    cfg.bfcc_forward_eval = bool(getattr(args, "bfcc_forward_eval", False))
    cfg.bfcc_forward_max_samples = int(getattr(args, "bfcc_forward_max_samples", 2))
    cfg.vq_only_train = bool(getattr(args, "vq_only_train", False))
    cfg.ignore_stats_in_mel = bool(getattr(args, "ignore_stats_in_mel", False))
    cfg.stats_only = bool(getattr(args, "stats_only", False))
    cfg.stats_only_lr = float(getattr(args, "stats_only_lr", 1e-4))
    cfg.stats_only_max_steps = int(getattr(args, "stats_only_max_steps", 10000))
    cfg.only_eval = bool(getattr(args, "only_eval", False))
    # Pipeline probe: only run a few segments for debugging.
    cfg.pipeline_probe = bool(getattr(args, "pipeline_probe", False))
    cfg.pipeline_probe_num_samples = int(getattr(args, "pipeline_probe_num_samples", 10))
    cfg.use_wandb = bool(args.use_wandb)
    cfg.wandb_project = args.wandb_project
    cfg.wandb_run_name = args.wandb_run_name
    cfg.wandb_log_freq = args.wandb_log_freq
    if getattr(args, "fsk_ber_table", None) is not None:
        cfg.fsk_ber_table = args.fsk_ber_table

    # Convenience: when L2H is enabled but细粒度超参数保持默认值时，
    # 自动采用与 v3 版训练脚本一致的一组经验默认项，避免每次命令行
    # 都手动补齐：
    #   lambda_l2h_resid   = 0.08
    #   lambda_l2h_decor   = 0.20
    #   l2h_improve_margin = 0.20
    #   l2h_low_bins       默认 18（见上方 dataclass / ArgumentParser 默认）
    # 若用户在 CLI 中显式传入这些值（非 0 / 非默认），则保留其设置。
    if cfg.with_l2h:
        if float(getattr(cfg, "lambda_l2h_resid", 0.0)) == 0.0:
            cfg.lambda_l2h_resid = 0.08
        if float(getattr(cfg, "lambda_l2h_decor", 0.0)) == 0.0:
            cfg.lambda_l2h_decor = 0.20
        if float(getattr(cfg, "l2h_improve_margin", 0.0)) == 0.0:
            cfg.l2h_improve_margin = 0.20

    return cfg


if __name__ == "__main__":
    cfg = _parse_args()
    run_training(cfg)
