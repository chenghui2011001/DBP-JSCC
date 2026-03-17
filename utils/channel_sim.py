#!/usr/bin/env python3
"""
Channel simulation and CSI synthesis utilities for Stage4 FiLM testing.

Design goals
- Provide lightweight, physically-inspired channel factors with a few stable scalars
- Return per-batch CSI dict for FiLM (aggregated), and per-frame factors to perturb z
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import numpy as np


class ChannelSimulator:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_hz: int = 100,
        snr_step_db: float | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_hz = frame_hz
        # Optional: discrete SNR grid step (in dB). When set to a
        # positive value, ``sample_csi`` will draw the base SNR from
        # a uniform grid ``[snr_min_db, snr_min_db+step, ...]`` up to
        # ``snr_max_db`` (inclusive when possible). If ``None`` or
        # non-positive, we fall back to the legacy behaviour.
        self.snr_step_db = snr_step_db

        # Optional explicit device override. When ``device`` is None
        # we keep the original behaviour and pick ``cuda`` when
        # available; when set (e.g. CPU-only inference scripts), all
        # CSI tensors will be created on this device to avoid
        # cross-device mismatches with the JSCC model.
        self._device_override: torch.device | None
        if device is None:
            self._device_override = None
        else:
            self._device_override = torch.device(device)

    @staticmethod
    def _lowpass_noise(shape, alpha: float, device, dtype, kernel_len: int = 256):
        """First-order IIR low-pass colored noise with pole alpha in (0,1).

        Vectorized approximation using FIR of length ``kernel_len`` with impulse
        response h[k] = (1-alpha) * alpha^k. This avoids Python loops over T and
        dramatically reduces per-batch overhead.
        """
        B, T = shape
        eps = torch.randn(B, T, device=device, dtype=dtype)
        # Build FIR kernel once per call; length sufficient for alpha≈0.98 (tau≈50)
        K = max(16, int(kernel_len))
        k = torch.arange(K, device=device, dtype=dtype)
        h = (1.0 - float(alpha)) * torch.pow(torch.tensor(float(alpha), device=device, dtype=dtype), k)
        # Convolve along time with padding to keep length T
        eps3 = eps.unsqueeze(1)  # [B,1,T]
        h3 = h.view(1, 1, -1)    # [1,1,K]
        y = torch.nn.functional.conv1d(eps3, h3, padding=K - 1).squeeze(1)  # [B, T+K-1]
        return y[:, :T]

    def sample_csi(self, B: int, T: int, channel: str = "fading",
                   snr_min_db: float | None = None,
                   snr_max_db: float | None = None) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Create CSI dict (aggregated scalars) and per-frame factors for z-perturbation.

        Universal4 proxy (pilot-free friendly) outputs in ``csi``:
          - snr_proxy         (dB): proxy of SNR, derived from simulated SNR trajectory
          - time_selectivity  (0..1): 1 - corr(amp_t[t-1], amp_t[t])
          - freq_selectivity  (0..1): normalised RMS delay spread proxy
          - los_ratio         (0..1): K/(K+1) from Rician K-factor

        Also returns:
          amp_t: [B, T] multiplicative fading envelope (≈1.0 ±)
          snr_db_t: [B, T] per-frame SNR in dB
        """
        if self._device_override is not None:
            device = self._device_override
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dtype = torch.float32

        # SNR (dB): time-varying around a base level.

        lo = -5.0 if snr_min_db is None else float(snr_min_db)
        hi = 15.0 if snr_max_db is None else float(snr_max_db)
        if hi < lo:
            lo, hi = hi, lo

        step = getattr(self, "snr_step_db", None)
        if step is not None and step > 0.0:
            # Discrete grid with user-specified step (e.g., 1 dB).
            # We include the upper bound when (hi-lo)/step is integer.
            span = max(hi - lo, 0.0)
            n_steps = int(math.floor(span / float(step))) + 1
            n_steps = max(1, n_steps)
            hi_eff = lo + float(step) * float(n_steps - 1)
            snr_grid = torch.linspace(lo, hi_eff, steps=n_steps, device=device, dtype=dtype)
        else:
            # Legacy behaviour: small fixed grid for the canonical
            # [-5,10] range, otherwise a 6-point linspace between
            # [lo,hi].
            snr_grid = torch.tensor(
                [-5.0, -3.0, 0.0, 3.0, 5.0, 10.0], device=device, dtype=dtype
            )
            if not (abs(lo + 5.0) < 1e-3 and abs(hi - 10.0) < 1e-3):
                snr_grid = torch.linspace(lo, hi, steps=snr_grid.numel(), device=device, dtype=dtype)

        idx = torch.randint(0, snr_grid.numel(), (B,), device=device)
        base_snr = snr_grid[idx]
        # doppler_norm ~ [0.0, 0.1]
        doppler_norm = torch.rand(B, device=device, dtype=dtype) * 0.1
        # low-pass color: alpha close to 1.0 for slow variation
        alpha = torch.clamp(1.0 - doppler_norm, 0.85, 0.999)
        alphas = alpha.view(B, 1).expand(B, T)
        # colored noise for snr fluctuation (~ ±3 dB)
        snr_delta = self._lowpass_noise((B, T), alpha=0.98, device=device, dtype=dtype) * 1.0
        snr_db_t = base_snr.view(B, 1) + snr_delta

        # Rician K-factor (dB): [-3, +10] (Rayleigh≈-inf → use small negative)
        k_factor_db = -3.0 + torch.rand(B, device=device, dtype=dtype) * 13.0
        K_lin = torch.pow(10.0, k_factor_db / 10.0)  # [B]

        # Fading amplitude envelope (Rician approx):
        # a(t) ≈ sqrt( (sqrt(K/(K+1)) + n)^2 + n2^2 ), simplified as mean 1.0 +/- colored fluctuation
        fade_lp = self._lowpass_noise((B, T), alpha=0.98, device=device, dtype=dtype) * 0.1
        amp_t = (1.0 + fade_lp).clamp(0.7, 1.3)

        # RMS delay spread (ms) and coherence time (frames)
        tau_rms_ms = 0.1 + torch.rand(B, device=device, dtype=dtype) * 3.0  # 0.1..3.1 ms
        # Coherence time ~ 1/(2 f_D), use doppler_norm ≈ f_D/f_s, frames at 100 Hz
        coh_time_s = (1.0 / (2.0 * (doppler_norm * self.sample_rate + 1e-3))).clamp(0.01, 1.0)
        coherence_frames = (coh_time_s * self.frame_hz).clamp(1.0, 100.0)

        # Burst/loss model (for completeness)
        loss_prob = torch.rand(B, device=device, dtype=dtype) * 0.02  # ≤2%
        burst_len_mean = 1.0 + torch.rand(B, device=device, dtype=dtype) * 4.0  # 1..5 frames

        # System-related auxiliaries (optional)
        rate_margin = torch.rand(B, device=device, dtype=dtype) * 0.5  # 0..0.5
        buffer_level = 0.2 + torch.rand(B, device=device, dtype=dtype) * 0.6  # 0.2..0.8

        # Aggregate CSI scalars (use means)
        snr_db = snr_db_t.mean(dim=1)

        # Universal4 proxies
        # 1) snr_proxy (dB)
        snr_proxy = snr_db

        # 2) time_selectivity (0..1): 1 - corr(amp_t[t-1], amp_t[t])
        if T > 1:
            x = amp_t[:, :-1]
            y = amp_t[:, 1:]
            x_c = x - x.mean(dim=1, keepdim=True)
            y_c = y - y.mean(dim=1, keepdim=True)
            num = (x_c * y_c).mean(dim=1)
            # 只在有足够样本时计算std，避免自由度警告
            if x.size(1) > 0:
                den = (x_c.std(dim=1, unbiased=False) * y_c.std(dim=1, unbiased=False)).clamp_min(1e-6)
                corr = (num / den).clamp(-1.0, 1.0)
            else:
                corr = torch.zeros_like(num)
            time_selectivity = (1.0 - corr).clamp(0.0, 1.0)
        else:
            # 时间维度太小，无法计算时间选择性
            time_selectivity = torch.zeros(B, device=amp_t.device, dtype=amp_t.dtype)

        # 3) freq_selectivity (0..1): normalised RMS delay spread proxy
        # tau_rms_ms in [0.1, 3.1] → normalise to [0,1]
        freq_selectivity = ((tau_rms_ms - 0.1) / (3.1 - 0.1)).clamp(0.0, 1.0)

        # 4) los_ratio (0..1): K_lin/(K_lin+1)
        los_ratio = (K_lin / (K_lin + 1.0)).clamp(0.0, 1.0)

        csi = {
            'snr_proxy': snr_proxy,
            'time_selectivity': time_selectivity,
            'freq_selectivity': freq_selectivity,
            'los_ratio': los_ratio,
        }
        return csi, amp_t, snr_db_t

    def apply(
        self,
        z: torch.Tensor,
        amp_t: torch.Tensor,
        snr_db_t: torch.Tensor,
        *,
        use_interleaver: bool = True,
        dim_jitter_std: float = 0.05,
    ) -> torch.Tensor:
        """Apply channel fading/noise to latent ``z``.

        设计目标（较原版做了两点增强）：
        1) 在符号序列维度上加入“交织 + 反交织”，将慢变/突发衰落打散到
           整个序列上，避免对 2D token 图像形成大块条纹灾难；
        2) 在特征维度上加入轻微的 per-dim 乘性抖动，缓解“整条向量共衰落”。

        Args:
            z:         [B,T,D] latent sequence.
            amp_t:     [B,T] fading envelope（由 ``sample_csi`` 给出）。
            snr_db_t:  [B,T] per-frame SNR in dB.
            use_interleaver: 若为 True，则在 T 维上做一次随机置换并在输出前反置换，
                             相当于“交织器 + 反交织器”，对上层网络透明；
            dim_jitter_std:  特征维度上的轻微乘性抖动标准差（高斯，均值 1.0），
                             例如 0.05 表示 ≈±5% 级别的 per-dim 衰落差异。
        """
        B, T, D = z.shape
        device = z.device
        dtype = z.dtype

        # --- 可选：在 T 维上做交织（perm）+ 反交织（inv_perm），对上层网络透明 ---
        if use_interleaver and T > 1:
            perm = torch.randperm(T, device=device)
            inv_perm = torch.empty_like(perm)
            inv_perm[perm] = torch.arange(T, device=device)

            z_seq = z[:, perm, :]
            amp_seq = amp_t.to(device=device, dtype=dtype)[:, perm]
            snr_seq = snr_db_t.to(device=device, dtype=dtype)[:, perm]
        else:
            perm = None
            inv_perm = None
            z_seq = z
            amp_seq = amp_t.to(device=device, dtype=dtype)
            snr_seq = snr_db_t.to(device=device, dtype=dtype)

        # Expand amp to [B,T,1]
        amp = amp_seq.unsqueeze(-1)

        # 在特征维度上加入轻微的乘性抖动，避免整条向量完全共衰落
        if D > 1 and dim_jitter_std > 0.0:
            # per-batch、per-dim 抖动，沿 T 维广播
            jitter = 1.0 + dim_jitter_std * torch.randn(B, 1, D, device=device, dtype=dtype)
            amp = amp * jitter

        z_amp = amp * z_seq

        # Per-frame noise std from SNR(dB): snr = 20*log10(sig/noise), noise = sig / 10^(snr/20)
        # Estimate signal std per-batch (avoid zero):
        sig_std = (z_amp.detach().float().pow(2).mean(dim=(1, 2)).sqrt() + 1e-3).to(dtype)
        # Broadcast to [B,T]
        snr_lin = torch.pow(10.0, (snr_seq / 20.0))
        noise_std_bt = (sig_std.view(B, 1) / (snr_lin + 1e-3)).clamp(1e-6, 1e3)
        noise = torch.randn_like(z_amp) * noise_std_bt.unsqueeze(-1).to(dtype)
        z_out = (z_amp + noise).to(dtype)

        # 反交织回原来的时间顺序（若启用了交织）
        if inv_perm is not None:
            z_out = z_out[:, inv_perm, :]

        return z_out


class RealNoiseChannelSimulator(ChannelSimulator):
    """ChannelSimulator variant driven by an external noise CSV.

    This adapter uses the temporal structure of a recorded noise
    voltage trace (e.g. ``noise_voltage_50s_300s.csv``) to modulate the
    SNR trajectory ``snr_db_t`` while keeping the overall interface
    identical to ``ChannelSimulator``.

    Design:
      - The mean SNR is still controlled by ``snr_min_db``/``snr_max_db``
        (as in the base simulator).
      - Frame-wise SNR fluctuations are obtained by sampling segments
        from the normalised noise waveform and scaling them to a
        ±3 dB dynamic range.
      - Fading envelope ``amp_t`` and the "Universal4" CSI proxies are
        computed in the same way as the base class so that the JSCC
        model sees familiar ranges.

    This is intended for *inference* only; training continues to use
    the original synthetic ``ChannelSimulator``.
    """

    def __init__(
        self,
        noise_csv: str,
        sample_rate: int = 16000,
        frame_hz: int = 100,
    ) -> None:
        super().__init__(sample_rate=sample_rate, frame_hz=frame_hz)

        try:
            try:
                noise_data = np.loadtxt(
                    noise_csv, delimiter=",", skiprows=16, usecols=1
                )
            except Exception:
                noise_data = np.loadtxt(noise_csv, delimiter=",", skiprows=16)
        except Exception as exc:  # pragma: no cover - best-effort IO
            raise FileNotFoundError(
                f"Failed to load noise CSV for RealNoiseChannelSimulator: {noise_csv} ({exc})"
            ) from exc

        if noise_data.size == 0:
            raise RuntimeError(
                f"Noise CSV appears empty for RealNoiseChannelSimulator: {noise_csv}"
            )

        noise = noise_data.astype(np.float32)
        noise = noise - float(noise.mean())
        std = float(noise.std())
        if std > 1e-8:
            noise = noise / std

        # Store as 1D Tensor on CPU; segments will be moved to the
        # appropriate device inside ``sample_csi``.
        self._noise = torch.from_numpy(noise)

    def sample_csi(
        self,
        B: int,
        T: int,
        channel: str = "fading",
        snr_min_db: float | None = None,
        snr_max_db: float | None = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Create CSI and per-frame factors using recorded noise shape.

        The overall structure mirrors ``ChannelSimulator.sample_csi``
        but replaces the synthetic SNR fluctuation with a segment drawn
        from the normalised noise trace.
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32

        # Base SNR around which fluctuations will vary
        lo = -5.0 if snr_min_db is None else float(snr_min_db)
        hi = 15.0 if snr_max_db is None else float(snr_max_db)
        if hi < lo:
            lo, hi = hi, lo
        base_snr = lo + torch.rand(B, device=device, dtype=dtype) * max(hi - lo, 1e-3)

        # Draw frame-wise SNR perturbations from the recorded noise trace
        noise = self._noise
        N = int(noise.numel())
        if N <= 0:
            snr_delta = torch.zeros(B, T, device=device, dtype=dtype)
        else:
            snr_delta = torch.empty(B, T, device=device, dtype=dtype)
            for b in range(B):
                if N >= T:
                    start = int(torch.randint(0, N - T + 1, (1,)).item())
                    seg = noise[start : start + T]
                else:
                    reps = (T + N - 1) // N
                    seg = noise.repeat(reps)[:T]
                snr_delta[b] = seg.to(device=device, dtype=dtype) * 3.0  # ≈ ±3 dB

        snr_db_t = base_snr.view(B, 1) + snr_delta

        # Fading envelope: reuse base implementation (slow-varying around 1.0)
        fade_lp = self._lowpass_noise((B, T), alpha=0.98, device=device, dtype=dtype) * 0.1
        amp_t = (1.0 + fade_lp).clamp(0.7, 1.3)

        # Universal4 proxies
        snr_db = snr_db_t.mean(dim=1)
        snr_proxy = snr_db

        if T > 1:
            x = amp_t[:, :-1]
            y = amp_t[:, 1:]
            x_c = x - x.mean(dim=1, keepdim=True)
            y_c = y - y.mean(dim=1, keepdim=True)
            num = (x_c * y_c).mean(dim=1)
            if x.size(1) > 0:
                den = (x_c.std(dim=1, unbiased=False) * y_c.std(dim=1, unbiased=False)).clamp_min(1e-6)
                corr = (num / den).clamp(-1.0, 1.0)
            else:
                corr = torch.zeros_like(num)
            time_selectivity = (1.0 - corr).clamp(0.0, 1.0)
        else:
            time_selectivity = torch.zeros(B, device=amp_t.device, dtype=amp_t.dtype)

        # Keep freq_selectivity/los_ratio in familiar ranges
        tau_rms_ms = 0.1 + torch.rand(B, device=device, dtype=dtype) * 3.0  # 0.1..3.1 ms
        freq_selectivity = ((tau_rms_ms - 0.1) / (3.1 - 0.1)).clamp(0.0, 1.0)

        k_factor_db = -3.0 + torch.rand(B, device=device, dtype=dtype) * 13.0
        K_lin = torch.pow(10.0, k_factor_db / 10.0)
        los_ratio = (K_lin / (K_lin + 1.0)).clamp(0.0, 1.0)

        csi = {
            "snr_proxy": snr_proxy,
            "time_selectivity": time_selectivity,
            "freq_selectivity": freq_selectivity,
            "los_ratio": los_ratio,
        }
        return csi, amp_t, snr_db_t
