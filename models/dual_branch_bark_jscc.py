#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dual-branch JSCC (Bark/BFCC-vSSM content branch).

内容：
- 内容分支：
    原始音频 audio[B,L] → Bark/BFCC 频谱 [B,T_bark,32]
    → 展平为 2D 图像 token 序列 → vSSMJSCCEncoder/Decoder (CSI→H0)
    → bark_hat[B,T_bark,32] → DCT → ceps_hat[B,T_bark,18]

- F0/voicing 分支：
    从 FARGAN 特征中提取 dnn_pitch/frame_corr，走轻量 1D JSCC，与 Stage2 DualBranch 一致。

- 最终：
    拼成 vocoder 20D 特征 (ceps_hat + dnn_pitch_hat + frame_corr_hat)，喂解码器生成音频。

说明：
- 这是在 Stage2 双支路基础上对内容分支的升级版本（Stage2.5），
  使用 Bark/BFCC 频谱图像 + vSSM 进行 JSCC，而非在 1D ceps 上编码。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any

import math
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.cuda.amp import autocast as amp_autocast

from .vmamba_jscc2d import VMambaJSCC2D
from .lite_speech_jscc import JSCCEncoder, JSCCDecoder
from .vocoder_decoder import FARGANDecoder
from .bfcc_vocoder import BFCCVocoder
from .feature_adapter import get_fargan_feature_spec
from .hash_bottleneck import HashBottleneck, GroupedHashBottleneck, TwoStageHashBottleneck
from .rvq_bottleneck import RVQBottleneck
DEBUG = bool(int(os.environ.get("DBG_STAGE25", "0")))
def pstats(name: str, t: torch.Tensor) -> None:
    if not DEBUG:
        return
    with torch.no_grad():
        nan = torch.isnan(t).any().item() if t.numel() else 0
        inf = torch.isinf(t).any().item() if t.numel() else 0
        tmin = float(t.min().item()) if t.numel() else 0.0
        tmax = float(t.max().item()) if t.numel() else 0.0
        tmean = float(t.mean().item()) if t.numel() else 0.0
        tstd = float(t.std().item()) if t.numel() else 0.0
        print(f"[DBG] {name}: shape={tuple(t.shape)} min={tmin:.4f} max={tmax:.4f} mean={tmean:.4f} std={tstd:.4f} nan={nan} inf={inf}")
        if nan or inf:
            raise RuntimeError(f"NaN/Inf detected at {name}")

def assert_finite(name: str, t: torch.Tensor) -> None:
    if not torch.isfinite(t).all():
        raise RuntimeError(f"[NaNGuard] non-finite detected at {name}: shape={tuple(t.shape)}")

import os, torch
def _pstats(name, x):
    if x is None:
        print(f"[{name}] None"); return
    x = torch.as_tensor(x)
    print(f"[{name}] shape={tuple(x.shape)} dtype={x.dtype} "
          f"min={x.min().item():+.4f} max={x.max().item():+.4f} "
          f"mean={x.mean().item():+.4f} std={x.std().item():+.4f}")
def _rvq_code_hist(tag, hb, codes):
    # hb: RVQBottleneck, codes: [B,T,nq]
    if codes is None: 
        print(f"[{tag}] codes=None"); return
    codes = codes.detach()
    B,T,nq = codes.shape
    n_levels = getattr(hb, "n_levels", None)
    print(f"[{tag}] codes shape={tuple(codes.shape)} n_levels={n_levels}")
    for qi in range(nq):
        lv = int(n_levels[qi]) if n_levels is not None else int(codes[...,qi].max().item()+1)
        h = torch.bincount(codes[...,qi].reshape(-1), minlength=lv).float()
        p = h / h.sum().clamp_min(1.0)
        H = float((-(p * (p+1e-12).log2())).sum().item())
        perp = 2.0 ** H
        usage = float((h>0).float().mean().item())
        top = int(h.argmax().item())
        ptop = float(p.max().item())
        print(f"  [q{qi}] lv={lv} usage={usage:.3f} H={H:.3f} perp={perp:.3f} top={top} p={ptop:.3f}")
def _ber_pm1(clean, noisy):
    # clean/noisy in {-1,+1} (or 0 included)
    clean = torch.as_tensor(clean)
    noisy = torch.as_tensor(noisy)
    return float((clean*noisy < 0).float().mean().item())

def opus_band_log_smooth(logE18: torch.Tensor) -> torch.Tensor:
    """Opus/LPCNet 风格的 log-域逐带平滑/跟随钳位（18带）。

    规则（参见 freq.c / lpcnet_enc.c）：
      Ly[i] = log10(1e-2 + E[i]) 之后，
      Ly[i] = max(logMax-8, max(follow-2.5, Ly[i]));
      logMax = max(logMax, Ly[i]);
      follow = max(follow-2.5, Ly[i]);
    这里直接对 logE18 应用该钳位（视作已做 log10），以减少 mel/BFCC→ceps 的系统偏差。
    """
    if logE18.dim() != 3 or logE18.size(-1) != 18:
        return logE18
    B, T, N = logE18.shape
    y = torch.empty_like(logE18)
    # 起始值与原实现一致
    logMax = logE18.new_full((B, T, 1), -2.0)
    follow = logE18.new_full((B, T, 1), -2.0)
    for i in range(N):
        cur = logE18[:, :, i:i+1]
        yi = torch.maximum(logMax - 8.0, torch.maximum(follow - 2.5, cur))
        y[:, :, i:i+1] = yi
        logMax = torch.maximum(logMax, yi)
        follow = torch.maximum(follow - 2.5, yi)
    return y


class BFCCContentEncoder(nn.Module):
    """Simple 2D CNN encoder for BFCC content branch.

    输入:  [B,1,32,T]  (freq x time)
    输出: [B,C_lat,H_lat,W_lat]  压缩后的 latent 特征。
    """

    def __init__(self, latent_channels: int = 1) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),   # [B,32,16,T/2]
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [B,64,8,T/4]
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=(2, 4), padding=1),  # [B,32,4,T/16]
            nn.ReLU(inplace=True),
            nn.Conv2d(32, self.latent_channels, kernel_size=1, stride=1),  # [B,C_lat,4,W_lat]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BFCCContentDecoder(nn.Module):
    """Simple 2D CNN decoder for BFCC content branch.

    输入:  [B,C_lat,H_lat,W_lat]
    输出: [B,1,32,T]  (freq x time)  与原始 BFCC 对齐后再裁剪时间维。
    """

    def __init__(self, latent_channels: int = 1) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                self.latent_channels,
                32,
                kernel_size=(2, 4),
                stride=(2, 4),
                padding=1,
            ),  # 约 [B,32,8,T/4]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),  # [B,32,16,T/2]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # [B,16,32,T]
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),  # [B,1,32,T]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

class WaveToBFCC(nn.Module):
    """波形 → Bark 滤波器组对数能量图 [B,T,n_bands]。"""

    def __init__(self,
                 sample_rate: int = 16000,
                 n_fft: int = 400,
                 hop_length: int = 160,
                 n_bands: int = 32) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop = hop_length
        self.n_bands = n_bands

        win = torch.hann_window(n_fft, periodic=True)
        self.register_buffer("win", win)

        # 构建 Bark 三角滤波器组权重 [Fbins, n_bands]
        Fbins = n_fft // 2 + 1
        freqs = torch.linspace(0, sample_rate / 2, Fbins)

        def hz_to_bark(f: torch.Tensor) -> torch.Tensor:
            return 13.0 * torch.atan(0.00076 * f) + 3.5 * torch.atan((f / 7500.0) ** 2)

        z = hz_to_bark(freqs)
        z_min, z_max = float(z.min()), float(z.max())
        centers = torch.linspace(z_min, z_max, steps=n_bands)
        # 边界使用相邻中心的中点，首尾用全域边界
        edges = torch.zeros(n_bands + 1)
        if n_bands > 1:
            edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
        edges[0] = z_min
        edges[-1] = z_max
        W = torch.zeros(Fbins, n_bands)
        for j in range(n_bands):
            left = edges[j]
            c = centers[j]
            right = edges[j + 1]
            lj = (z >= left) & (z <= c)
            rj = (z > c) & (z <= right)
            if (c - left) > 1e-6:
                W[lj, j] = (z[lj] - left) / (c - left)
            if (right - c) > 1e-6:
                W[rj, j] = (right - z[rj]) / (right - c)
        W = W / (W.sum(dim=0, keepdim=True) + 1e-8)
        self.register_buffer("bark_w", W)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Waveform → Bark log-energy [B,T,n_bands] in float32.

        在 AMP/F16 下若直接在半精度中计算 ``log10(E+eps)``，极小能量帧
        容易因为 eps 下溢为 0 而得到 ``log10(0)=-inf``。为避免这一点，
        这里显式禁用 autocast，并在 float32 中完成 STFT 与能量计算。
        """
        # 禁用 AMP autocast，强制使用 float32 计算 STFT/log10
        with amp_autocast(enabled=False):
            audio_f32 = audio.to(dtype=torch.float32)
            # 去除直流分量，避免 DC/极低频漂移在第 0 Bark 带上持续堆积
            if audio_f32.dim() >= 2:
                audio_f32 = audio_f32 - audio_f32.mean(dim=-1, keepdim=True)
            win = self.win.to(device=audio_f32.device, dtype=torch.float32)
            bark_w = self.bark_w.to(device=audio_f32.device, dtype=torch.float32)

            # STFT: [B,F,T]
            X = torch.stft(
                audio_f32,
                n_fft=self.n_fft,
                hop_length=self.hop,
                win_length=self.n_fft,
                window=win,
                center=False,
                return_complex=True,
            )
            Mag2 = X.real.pow(2) + X.imag.pow(2)  # [B,F,T]
            E = torch.matmul(Mag2.transpose(1, 2), bark_w)  # [B,T,n_bands]
            logE = torch.log10(E + 1e-10)
        return logE


        


class LearnableBandAgg(nn.Module):
    """可学习的能量域频带聚合：N_src → N_dst（仿滤波器组，非负、列归一）。

    - 使用 softplus 保持非负，再按列归一使每个目标带的权重和为 1。
    - 初始权重用现有三角聚合矩阵进行初始化，便于平滑替换。
    """

    def __init__(self, init_W: torch.Tensor) -> None:
        super().__init__()
        W0 = init_W.clone().detach()  # [N_src, N_dst]
        self.N_src, self.N_dst = W0.shape
        # 反参数化以保证非负
        self.raw_W = nn.Parameter(torch.log(torch.exp(W0) - 1.0 + 1e-8))

    def forward(self, E_src: torch.Tensor) -> torch.Tensor:
        # E_src: [B,T,N_src]
        W_pos = F.softplus(self.raw_W)  # [N_src,N_dst] >= 0
        W = W_pos / (W_pos.sum(dim=0, keepdim=True) + 1e-8)
        return torch.matmul(E_src, W)  # [B,T,N_dst]


class LearnableCepsMap(nn.Module):
    """
    log 能量谱 → 倒谱：在固定 DCT-II 的基础上加入极轻的可学习校正，专门减少 mel→ceps 的系统误差。

    y = (logE * scale + bias)
    ceps_base = y @ DCT
    ceps = ceps_base ⊙ lifter + tanh(alpha) · V(U y)
    ceps[...,0] += c0_bias
    """

    def __init__(self, n_bands: int, res_rank: int = 6) -> None:
        super().__init__()
        self.n_bands = n_bands
        # 固定 DCT-II 矩阵
        N = n_bands
        n = torch.arange(N, dtype=torch.float32).unsqueeze(1)
        k = torch.arange(N, dtype=torch.float32).unsqueeze(0)
        mat = torch.cos((n + 0.5) * k * math.pi / N)
        mat[:, 0] *= math.sqrt(0.5)
        mat = mat * math.sqrt(2.0 / N)
        self.register_buffer("dct_mat", mat)
        # 可学习带级仿射与 c0 偏置
        self.scale = nn.Parameter(torch.ones(N))
        self.bias = nn.Parameter(torch.zeros(N))
        self.c0_bias = nn.Parameter(torch.tensor(0.0))
        # learnable lifter（c0 固定为1，其余维轻微可调）
        lifter = torch.ones(N)
        lifter[0] = 1.0
        self.lifter = nn.Parameter(lifter)
        # 低秩残差校正（初值为0），极小参数量
        r = max(1, int(res_rank))
        self.res_u = nn.Linear(N, r, bias=False)
        self.res_v = nn.Linear(r, N, bias=False)
        nn.init.zeros_(self.res_u.weight)
        nn.init.zeros_(self.res_v.weight)
        self.res_alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, logE: torch.Tensor) -> torch.Tensor:
        # logE: [B,T,N]
        y = logE * self.scale.view(1, 1, -1) + self.bias.view(1, 1, -1)
        ceps_base = torch.matmul(y, self.dct_mat)  # [B,T,N]
        ceps_base = ceps_base * self.lifter.view(1, 1, -1)
        # 低秩残差（若保持0初始化将逐步学习，不会破坏初始 DCT 行为）
        if (self.res_u.weight.abs().sum() + self.res_v.weight.abs().sum()) > 0:
            res = self.res_v(self.res_u(y))
            ceps_base = ceps_base + torch.tanh(self.res_alpha) * res
        ceps = ceps_base.clone()
        ceps[..., 0] = ceps[..., 0] + self.c0_bias
        return ceps




class AdaLNBlock(nn.Module):
    """简洁的 AdaLN 调制块，用于 DeCo 风格的高频生成。

    x: [B,T,F] 为需要调制的特征；
    cond: [B,T,C] 为语义条件（由低频 mel + F0 + VUV 编码得到）。
    """

    def __init__(self, feat_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feat_dim)
        self.modulation = nn.Linear(cond_dim, 3 * feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.GELU(),
            nn.Linear(feat_dim * 2, feat_dim),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """AdaLN 调制 + 残差。

        Args:
            x: [B,T,F]
            cond: [B,T,C]
        """
        shift, scale, gate = self.modulation(cond).chunk(3, dim=-1)
        h = self.norm(x)
        h = scale * h + shift
        h = self.mlp(h)
        return x + gate.sigmoid() * h


class DeCoL2HRefiner(nn.Module):
    """DeCo 启发的高频生成器：从低频 + F0/VUV 条件生成 **高频残差**，并用 limiter 控制幅度。

    目标：解决 “L2H 把低频纹理复制到中高频”：
    - 输出 residual(Δhigh)，最终 high = high_base + Δhigh
    - tanh limiter + resid_scale，避免 L2H 过强重绘
    - harmonic/noise 双头：有声/无声高频分工（vuv_prob 门控）
    - decorrelation loss 在训练脚本里做（需要 out 里拿 resid/mask）
    """

    def __init__(
        self,
        n_mels: int = 32,
        low_bins: int = 10,
        hidden: int = 64,
        cond_dim: int = 32,
        n_blocks: int = 3,
        resid_scale: float = 0.35,
        limiter: str = "tanh",
        dual_head: bool = True,
        harmonic_cutoff_hz: float = 4200.0,
        band_centers_hz: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        assert 1 <= low_bins < n_mels
        self.n_mels = int(n_mels)
        self.low_bins = int(low_bins)
        self.high_bins = int(n_mels - low_bins)

        self.resid_scale = float(resid_scale)
        self.limiter = str(limiter)
        self.dual_head = bool(dual_head)
        self.harmonic_cutoff_hz = float(harmonic_cutoff_hz)

        if band_centers_hz is None:
            band_centers_hz = torch.linspace(0.0, 8000.0, steps=self.n_mels, dtype=torch.float32)
        else:
            band_centers_hz = band_centers_hz.detach().float().view(-1)
            if band_centers_hz.numel() != self.n_mels:
                raise ValueError(f"band_centers_hz must have shape [{self.n_mels}], got {tuple(band_centers_hz.shape)}")
        self.register_buffer("band_centers_hz", band_centers_hz)

        self.cond_in = nn.Linear(self.low_bins + 2, cond_dim)  # low + pitch + frame_corr
        self.in_proj = nn.Linear(self.high_bins, hidden)
        self.hf_blocks = nn.ModuleList([AdaLNBlock(hidden, cond_dim) for _ in range(int(n_blocks))])

        self.out_harm = nn.Linear(hidden, self.high_bins)
        self.out_noise = nn.Linear(hidden, self.high_bins) if self.dual_head else None

        self.last_vuv_prob: Optional[torch.Tensor] = None
        self.last_resid: Optional[torch.Tensor] = None
        self.last_mask_harm: Optional[torch.Tensor] = None

    @staticmethod
    def _to_vuv(frame_corr_logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(frame_corr_logits)

    def _limit(self, x: torch.Tensor) -> torch.Tensor:
        if self.limiter == "tanh":
            return torch.tanh(x)
        raise ValueError(f"Unsupported limiter: {self.limiter}")

    def _static_harm_mask(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        centers = self.band_centers_hz.to(device=device, dtype=dtype)[self.low_bins:]  # [high_bins]
        mask = (centers <= self.harmonic_cutoff_hz).to(dtype=dtype)
        return mask.view(1, 1, -1)

    def forward(
        self,
        mel_low: torch.Tensor,
        dnn_pitch: torch.Tensor,
        frame_corr: torch.Tensor,
        mel_high_base: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        device = mel_low.device
        dtype = mel_low.dtype
        B, T, _ = mel_low.shape

        cond = torch.cat([mel_low, dnn_pitch, frame_corr], dim=-1)
        cond_embed = self.cond_in(cond)

        if mel_high_base is None:
            h0 = torch.zeros(B, T, self.high_bins, device=device, dtype=dtype)
        else:
            h0 = mel_high_base
            if h0.size(-1) != self.high_bins:
                if h0.size(-1) < self.high_bins:
                    h0 = F.pad(h0, (0, self.high_bins - h0.size(-1)), value=0.0)
                else:
                    h0 = h0[..., : self.high_bins]

        h = self.in_proj(h0)
        for block in self.hf_blocks:
            h = block(h, cond_embed)

        r_harm = self.out_harm(h)
        r_noise = self.out_noise(h) if (self.dual_head and self.out_noise is not None) else r_harm

        r_harm = self.resid_scale * self._limit(r_harm)
        r_noise = self.resid_scale * self._limit(r_noise)

        vuv_prob = self._to_vuv(frame_corr)            # [B,T,1]
        mask_harm = self._static_harm_mask(device, dtype)  # [1,1,H]

        resid_voiced = mask_harm * r_harm + (1.0 - mask_harm) * r_noise
        resid_unvoiced = r_noise
        resid = vuv_prob * resid_voiced + (1.0 - vuv_prob) * resid_unvoiced

        mel_high_ref = h0 + resid

        with torch.no_grad():
            self.last_vuv_prob = vuv_prob.detach()
            self.last_resid = resid.detach()
            self.last_mask_harm = mask_harm.detach()

        return mel_high_ref.to(device=device, dtype=dtype), None



class AffineCouplingFlow(nn.Module):
    """简单的一维 affine coupling flow（RealNVP 风格）。

    在最后一维做通道划分：x = [x1, x2]，条件为 h[B,T,H]。
    forward:  x2 -> y2 = x2 * exp(s) + t
    inverse:  y2 -> x2 = (y2 - t) * exp(-s)
    """

    def __init__(self, num_channels: int, cond_dim: int) -> None:
        super().__init__()
        if num_channels < 2:
            raise ValueError("num_channels must be >= 2 for coupling flow")
        self.num_channels = num_channels
        self.cond_dim = cond_dim
        self.split = num_channels // 2
        in_dim = self.split + cond_dim
        hidden = max(cond_dim, 32)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * (num_channels - self.split)),
        )

    def _coupling(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 (log_s, t)，形状与 x2 相同。"""
        x1 = x[..., : self.split]
        x2 = x[..., self.split :]
        cond = torch.cat([x1, h], dim=-1)
        st = self.net(cond)
        s, t = torch.chunk(st, 2, dim=-1)
        # 将 log_s 控制在合理范围内，避免数值爆炸
        log_s = torch.tanh(s)  # [-1,1]
        return log_s, t

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x,h -> (y, log_det)，log_det 形状为 [B,T]。"""
        log_s, t = self._coupling(x, h)
        x1 = x[..., : self.split]
        x2 = x[..., self.split :]
        y2 = x2 * torch.exp(log_s) + t
        y = torch.cat([x1, y2], dim=-1)
        log_det = log_s.sum(dim=-1)
        return y, log_det

    def inverse(self, y: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """y,h -> (x, log_det)，log_det 为 log|det(dz/dx)|，形状 [B,T]。"""
        log_s, t = self._coupling(y, h)
        y1 = y[..., : self.split]
        y2 = y[..., self.split :]
        x2 = (y2 - t) * torch.exp(-log_s)
        x = torch.cat([y1, x2], dim=-1)
        log_det = -log_s.sum(dim=-1)
        return x, log_det


class ConditionalHFGenerator(nn.Module):
    """基于条件 flow 的高频 mel 生成器。

    训练：输入 GT 高频 mel，反向求 latent z 及 log_det，用于 NLL 损失；
    推理：从 N(0,I) 采样 latent，经 flow 正向生成高频 mel 样本。
    当前仅在训练中用于 NLL 正则，主前向路径使用 DeCoL2HRefiner 输出。
    """

    def __init__(
        self,
        n_mels: int = 32,
        low_bins: int = 10,
        hidden: int = 128,
        n_flows: int = 4,
    ) -> None:
        super().__init__()
        self.low_bins = int(low_bins)
        self.high_bins = int(n_mels - low_bins)
        if self.high_bins <= 0:
            raise ValueError("n_mels must be greater than low_bins for HF generator")

        self.cond_enc = nn.Sequential(
            nn.Linear(self.low_bins + 2, hidden),  # low_mel + pitch + vuv
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

        self.flows = nn.ModuleList(
            AffineCouplingFlow(self.high_bins, hidden) for _ in range(max(1, n_flows))
        )

    def forward(
        self,
        mel_low: torch.Tensor,
        pitch: torch.Tensor,
        vuv: torch.Tensor,
        mel_high_base: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Flow 前向接口。

        训练：输入 GT 高频 mel (mel_high_base)，输出 latent z 与 log_det(dz/dx)。
        推理：从标准高斯采样 latent，经 flow 正向生成高频 mel 样本。
        """
        B, T, _ = mel_low.shape
        cond = torch.cat([mel_low, pitch, vuv], dim=-1)  # [B,T,low_bins+2]
        h = self.cond_enc(cond)  # [B,T,hidden]

        if self.training:
            if mel_high_base is None:
                raise ValueError("mel_high_base is required in training mode for ConditionalHFGenerator")
            if mel_high_base.shape[:2] != (B, T):
                mel_high_base = mel_high_base[:, :T, : self.high_bins]
            else:
                mel_high_base = mel_high_base[..., : self.high_bins]
            z = mel_high_base
            log_det_total: Optional[torch.Tensor] = None
            for flow in reversed(self.flows):
                z, log_det = flow.inverse(z, h)
                if log_det_total is None:
                    log_det_total = log_det
                else:
                    log_det_total = log_det_total + log_det
            if log_det_total is None:
                log_det_total = torch.zeros(B, T, device=mel_low.device, dtype=mel_low.dtype)
            return z, log_det_total

        # 推理：从 N(0,I) 采样 latent 并正向生成高频 mel
        z = torch.randn(B, T, self.high_bins, device=mel_low.device, dtype=mel_low.dtype)
        for flow in self.flows:
            z, _ = flow.forward(z, h)
        return z, None
class HarmNoiseResidualHead(nn.Module):
    """
    Predict HF residual as harmonic + noise.
    light GRU + linear, used in L2H residual mode.
    """
    def __init__(self, in_dim: int, high_bins: int, hidden: int = 96):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(2 * hidden, 2 * high_bins)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h, _ = self.gru(x)
        y = self.proj(h)
        H = y.size(-1) // 2
        return y[..., :H], y[..., H:]

class F0VUVEncoder(nn.Module):
    """F0/voicing 分支编码器：f0vuv[2] → latent z_fv

    加强版：在原始 2 维 f0+vuv 上先通过一小段 1D 卷积
    堆栈提取局部时序模式，再送入双向 2 层 GRU，最后
    投影到 JSCC/RVQ 使用的 latent 维度 d_zf。

    这样可以在保持整体结构不变的情况下，增加 F0 分支
    对局部抖动、边界毛刺以及单个 bit 错误的鲁棒性。
    """

    def __init__(self, in_dim: int = 2, d_zf: int = 16, hidden: int = 64) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)

        # 轻量多尺度卷积前端：捕获 20–80ms 左右的局部模式
        # 输入形状 [B,T,in_dim]，在 time 维做 1D conv
        self.conv = nn.Sequential(
            nn.Conv1d(self.in_dim, self.hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden, self.hidden, kernel_size=5, padding=2, dilation=1),
            nn.ReLU(inplace=True),
        )

        # 双向 GRU：在卷积特征上建模更长时序上下文
        self.gru = nn.GRU(
            input_size=self.hidden,
            hidden_size=self.hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.proj = nn.Linear(self.hidden * 2, d_zf)
        self.norm = nn.LayerNorm(d_zf)

    def forward(self, f0vuv: torch.Tensor) -> torch.Tensor:
        """Encode F0+VUV into latent z_fv with basic NaNGuard.

        - 输入先在 [-5,5] 做裁剪并清理 NaN/Inf；
        - 经过 Conv1d 堆栈获得 [B,T,hidden] 局部特征；
        - 再送入 BiGRU + 线性投影得到 z_fv。
        """

        # 防御性处理：清理输入中的 NaN/Inf，并裁剪极端值
        if not torch.isfinite(f0vuv).all():
            if os.environ.get("DBG_F0_ENC", "0") == "1":
                print("[NaNGuard-F0] non-finite input to F0VUVEncoder; cleaning")
            f0vuv = torch.nan_to_num(f0vuv, nan=0.0, posinf=0.0, neginf=0.0)
        f0vuv = f0vuv.clamp(min=-5.0, max=5.0)

        # [B,T,2] -> [B,2,T] -> Conv1d -> [B,hidden,T] -> [B,T,hidden]
        x = f0vuv.transpose(1, 2)  # [B,in_dim,T]
        x = self.conv(x)
        x = x.transpose(1, 2)      # [B,T,hidden]

        h, _ = self.gru(x)  # [B,T,hidden*2]
        if not torch.isfinite(h).all():
            if os.environ.get("DBG_F0_ENC", "0") == "1":
                print("[NaNGuard-F0] non-finite GRU output in F0VUVEncoder; cleaning")
            h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)

        z = self.proj(h)
        if not torch.isfinite(z).all():
            if os.environ.get("DBG_F0_ENC", "0") == "1":
                print("[NaNGuard-F0] non-finite proj output in F0VUVEncoder; cleaning")
            z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

        z = self.norm(z)
        z = torch.tanh(z)
        return z


class F0VUVDecoder(nn.Module):
    """F0/voicing 分支解码器：latent z_fv → dnn_pitch_hat, frame_corr_hat

    小改版：在原有双向 GRU 基础上，
    可选地引入 *content-conditioned* 交叉注意力：

    - 主干仍然是 BiGRU(z_fv_hat) 提取 F0 时序特征；
    - 若提供 mel 条件（通常为 mel_hat_norm[B,T,F_mel]），
      则通过 MultiheadAttention 让 F0 在每一帧看到
      同步的 Mel 上下文，以提升 VUV 判决与句内
      F0 连续性的鲁棒性；
    - 若未提供 mel 条件（例如 F0-only JSCC 解码）、
      则退化为原来的纯 BiGRU+MLP 解码器。
    """

    def __init__(
        self,
        d_zf: int = 16,
        hidden: int = 64,
        cond_dim: int = 0,
        num_heads: int = 4,
    ) -> None:
        super().__init__()

        # 双向 GRU 捕获上下文信息（±50 帧 ≈ ±500ms）
        self.gru = nn.GRU(
            input_size=d_zf,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )

        self.hidden = int(hidden)
        self.cond_dim = int(cond_dim)
        self.use_cond = self.cond_dim > 0

        if self.use_cond:
            # 将 mel 条件投影到与 GRU 输出相同的维度，并
            # 使用多头注意力在时间维上做简单的 cross-attention。
            self.cond_proj = nn.Linear(self.cond_dim, hidden * 2)
            self.attn = nn.MultiheadAttention(
                embed_dim=hidden * 2,
                num_heads=num_heads,
                batch_first=True,
            )
            self.attn_norm = nn.LayerNorm(hidden * 2)
            # 可学习 gate：通过 sigmoid(attn_gate) 控制 cross-attn 强度。
            # 额外的标量 attn_alpha 由训练脚本按步数调度，实现可控 warmup。
            self.attn_gate = nn.Parameter(torch.zeros(1))
            self.attn_alpha: float = 0.0

        # 非线性投影层
        self.proj = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 2),
        )

    def forward(
        self,
        z_fv_hat: torch.Tensor,
        mel_cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode latent z_fv_hat into (dnn_pitch_hat, vuv_logits).

        Args:
            z_fv_hat: [B,T,d_zf] latent sequence after JSCC decode / RVQ。
            mel_cond: 可选的 Mel 条件特征 [B,T,F_mel]；若提供
                且在初始化时 cond_dim>0，则在 GRU 输出上
                施加一层 cross-attention，提升对内容与
                能量模式的感知。
        """

        h, _ = self.gru(z_fv_hat)  # [B,T,hidden*2]

        if self.use_cond and mel_cond is not None:
            # 对 mel_cond 做简单的时间对齐与线性投影，
            # 然后以 GRU 输出为 query、mel 为 key/value
            # 做一次 cross-attention，并加残差。
            B, T, _ = h.shape
            if mel_cond.dim() != 3:
                mel_cond = mel_cond.view(B, -1, self.cond_dim)

            if mel_cond.size(1) != T:
                # [B,Tm,F] -> [B,F,Tm] -> interpolate -> [B,F,T]
                mel_cond = F.interpolate(
                    mel_cond.transpose(1, 2),
                    size=T,
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)

            cond = self.cond_proj(mel_cond)  # [B,T,hidden*2]
            # 对于 T≈400 的序列，完整的 attention 权重矩阵
            # [B,num_heads,T,T] 占用显存较大，这里将
            # need_weights=False 以避免保存权重，仅保留
            # attn_out 参与后向传播。
            attn_out, _ = self.attn(query=h, key=cond, value=cond, need_weights=False)

            # gate = sigmoid(attn_gate) ∈ (0,1)，alpha 来自训练脚本调度；
            # 两者相乘得到最终的 cross-attention 强度。
            gate = torch.sigmoid(self.attn_gate)
            alpha = float(getattr(self, "attn_alpha", 1.0))
            scale = alpha * gate
            h = self.attn_norm(h + scale * attn_out)

        out = self.proj(h)  # [B,T,2]
        dnn_pitch_hat = out[..., :1].clamp(-3.0, 3.0)
        # Raw logits for BCE / sigmoid-based VUV losses
        vuv_logits = out[..., 1:2]
        return dnn_pitch_hat, vuv_logits




class DualBranchBarkJSCC(nn.Module):
    """Stage2.5 双支路 JSCC 模型（Bark/BFCC + VMamba 内容分支）。

    内容分支：
        audio[B,L] → 32-bin Bark/BFCC 图像 [B,T_bark,32]
        → 2D VMamba JSCC 编码/解码 → bark_hat → DCT → ceps_hat

    F0/voicing 分支与 DualBranchJSCC 相同，仍从 FARGAN 特征中取 dnn_pitch/frame_corr 做 JSCC。
    """

    def __init__(
        self,
        d_csi: int = 4,
        d_zc: int = 32,
        d_s_content: int = 8,
        d_zf: int = 16,
        d_s_f0: int = 16,
        hidden_f0: int = 32,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 32,
        use_hash: Optional[bool] = None,
        with_hash: Optional[bool] = None,
        hash_bits_content: int = 16,
        hash_bits_f0: Optional[int] = None,
        vm_channels: Optional[List[int]] = None,
        vm_depths: Optional[List[int]] = None,
        freq_downsample_stages: int = 2,
        content_time_downsample: int = 1,
        vm_channel_adaptive: str = "no",
        vm_lightweight_config: str = "all_native",
        eq_fading: bool = False,
        device: Optional[torch.device] = None,
        with_l2h: bool = False,
        l2h_low_bins: int = 10,
        use_l2h_resid: bool = True,
        l2h_resid_hidden: int = 96,
        l2h_resid_scale: float = 0.25,
        l2h_harm_bins: Optional[int] = None,
        l2h_silence_thr: float = -7.0,
        l2h_dual_head: bool = True,
        l2h_harmonic_cutoff_hz: float = 4200.0,
        # 可选：使用条件 flow 对 GT 高频 Mel 做生成式建模（NLL 正则）
        use_l2h_flow: bool = False,
        l2h_flow_hidden: int = 128,
        l2h_flow_n_flows: int = 4,
        # DeCo 风格 L2H：条件生成 Mel 高频，而非对基线 Mel 高频做残差补全
        deco_l2h: bool = False,
        deco_l2h_hidden: int = 64,
        deco_l2h_blocks: int = 3,
        # 内容分支 hash 细化：是否在通道维上做分组 hash
        group_hash_content: bool = True,
        # HF 侧通道：将高频残差特征直接传给 FARGAN，绕过 32->18 聚合的信息损失
        with_hf_sideband: bool = False,
        hf_sideband_dim: int = 6,  # 侧通道维度 (4-8 维通常足够)
        hf_sideband_type: str = "learnable",  # "learnable" | "dct" | "pca"
        hf2ceps_dim: int = 8,
        hf2ceps_scale: float = 0.5,
        # 可选：使用简单 CNN BFCC JSCC 作为内容分支 baseline（替代 VMamba）
        content_cnn_baseline: bool = False,
        content_cnn_latent_channels: int = 1,
        use_bfcc_vocoder_debug: bool = False,
        # Hash / RVQ 量化器选择与 RVQ 配置
        quantizer_type: str = "hash",
        rvq_nq_content: int = 2,
        rvq_nq_f0: Optional[int] = None,
        rvq_beta: float = 0.25,
        # F0/VUV 分支：SR + 时序冗余配置
        # f0_sr_k: 前 k 个符号维度作为“base 层”承载稳定骨架，其余维度承载细节。
        f0_sr_k: int = 4,
    ) -> None:
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # F0 / frame_corr 标定参数：在 F0 decoder 输出后做简单仿射变换，
        # 用于校准 JSCC F0 分支与原始 FARGAN 特征空间之间的全局
        # scale / bias 差异。默认初始化为恒等映射。
        self.f0_calib_scale = nn.Parameter(torch.tensor(1.0))
        self.f0_calib_bias = nn.Parameter(torch.tensor(0.0))
        self.fc_calib_scale = nn.Parameter(torch.tensor(1.0))
        self.fc_calib_bias = nn.Parameter(torch.tensor(0.0))

        # 是否在内容分支上使用简单 CNN BFCC JSCC baseline（替代 VMamba 内容 JSCC）
        self.use_cnn_content: bool = bool(content_cnn_baseline)
        self.content_cnn_latent_channels: int = int(content_cnn_latent_channels)

        # 量化器类型：hash 或 rvq
        self.quantizer_type = str(quantizer_type or "hash")
        if self.quantizer_type not in ("hash", "rvq"):
            raise ValueError(f"Unsupported quantizer_type={self.quantizer_type}; expected 'hash' or 'rvq'")

        if with_hash is None:
            if use_hash is None:
                with_hash = False
            else:
                with_hash = use_hash
        self.with_hash = bool(with_hash)
        self.use_hash = self.with_hash  # 保持向后兼容
        self.hash_bits_content = int(hash_bits_content)
        self.hash_bits_f0 = None if hash_bits_f0 is None else int(hash_bits_f0)
        self.group_hash_content = bool(group_hash_content)

        # RVQ 配置
        self.rvq_nq_content = int(rvq_nq_content)
        self.rvq_nq_f0 = None if rvq_nq_f0 is None else int(rvq_nq_f0)
        self.rvq_beta = float(rvq_beta)

        # F0/VUV SR 配置：前 f0_sr_k 维作为 base 层
        self.f0_sr_k: int = max(0, int(f0_sr_k))

        # 调整架构以达到64-128倍压缩率
        # 目标：200×32=6400 -> 压缩到28-56个元素
        # 方案：增加更多下采样层，减少通道数
        # 修正：遵循MambaJSCC通道递增/递减模式
        # 编码器：通道递增 16→24→32→48，解码器：通道递减 48→32→24→16
        vm_channels = vm_channels or [16, 24, 32, 48]  # 修正为4层对称架构
        vm_depths = vm_depths or [2, 2, 3, 2]         # 调整深度匹配
        if len(vm_depths) != len(vm_channels):
            raise ValueError("vm_depths must match vm_channels length")
        self.vm_channels = vm_channels
        self.vm_depths = vm_depths
        self.freq_downsample_stages = max(0, int(freq_downsample_stages))
        self.d_s_content = d_s_content
        # 内容分支时间下采样因子（1 表示不下采样）
        self.content_time_downsample = max(1, int(content_time_downsample))

        env_eq = False
        try:
            env_eq = bool(int(os.environ.get("EQ_FADING", "0")))
        except Exception:
            env_eq = False
        self.eq_fading = bool(eq_fading or env_eq)

        self.fargan_spec = get_fargan_feature_spec()

        # Optional learned energy calibration head:
        # maps per-sample mel statistics to a small c0 offset so that
        # the model can self-calibrate overall energy without relying
        # on GT FARGAN ceps. Disabled by default; can be enabled from
        # training scripts via ``model.use_learned_energy_calib = True``.
        self.use_learned_energy_calib: bool = False
        self.energy_calib_head = nn.Sequential(
            nn.Linear(n_mels, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

        # 内容分支：wave -> BFCC(Bark对数能量图)
        # 后续 JSCC/解码可选：VMambaJSCC2D 或 BFCCContentEncoder/Decoder baseline
        self.wave_to_mel = WaveToBFCC(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_bands=n_mels,
        )
        # 预计算 Bark 32 带中心频率（Hz），供 L2H harmonic/noise 静态切分使用。
        with torch.no_grad():
            _Fbins = int(n_fft // 2 + 1)
            _freqs = torch.linspace(0.0, float(sample_rate) / 2.0, steps=_Fbins, dtype=torch.float32)
            _bark_w = self.wave_to_mel.bark_w.detach().float().cpu()  # [Fbins, n_mels], col-sum=1
            _centers_hz = (_freqs.view(-1, 1) * _bark_w).sum(dim=0)   # [n_mels]
        self.register_buffer("bark_centers_hz", _centers_hz.float())
        # 18带版本 DCT 压缩（与 FARGAN 一致维度），改为可学习映射
        self.mel18_to_ceps = LearnableCepsMap(n_bands=18)
        # 32(Bark) → 18(Opus eband) 线性最小二乘投影：E18 ≈ E32 @ M
        # 以 STFT 频点为公共基底：W_bark(F×32) 已在 WaveToBFCC 中构建；构造 W_eband(F×18) 后，
        # 取 M = pinv(W_bark) @ W_eband，使得 W_bark @ M ≈ W_eband（Frobenius 最小二乘）。
        class FixedLinearProject(nn.Module):
            def __init__(self, M: torch.Tensor) -> None:
                super().__init__()
                self.register_buffer("M", M)
            def forward(self, E_src: torch.Tensor) -> torch.Tensor:
                return torch.matmul(E_src, self.M)


        def _build_eband_w(Fbins: int, sr: int) -> torch.Tensor:
            # 参照 freq.c 注释：边界约为 [0,200,400,600,800,1k,1.2k,1.4k,1.6k,2k,2.4k,2.8k,3.2k,4k,4.8k,5.6k,6.8k,8k]
            edges_hz = torch.tensor([0, 200, 400, 600, 800, 1000, 1200, 1400, 1600,
                                     2000, 2400, 2800, 3200, 4000, 4800, 5600, 6800, 8000], dtype=torch.float32)
            hz = torch.linspace(0.0, float(sr) / 2.0, Fbins, dtype=torch.float32)
            N = 18
            W = torch.zeros(Fbins, N, dtype=torch.float32)
            for i in range(N - 1):
                f0 = edges_hz[i]; f1 = edges_hz[i + 1]
                if f1 <= f0 + 1e-6:
                    continue
                m = (hz >= f0) & (hz <= f1)
                if m.any():
                    frac = (hz[m] - f0) / (f1 - f0)
                    W[m, i] += (1.0 - frac)
                    W[m, i + 1] += frac
            # 首末两带乘2（与原实现一致）
            W[:, 0] *= 2.0
            W[:, -1] *= 2.0
            return W

        try:
            W_bark = self.wave_to_mel.bark_w  # [F,32]
            Fbins = W_bark.size(0)
            W_eband = _build_eband_w(Fbins, sample_rate)
            # 最小二乘 M: 32×18。为提升数值稳定性，在 float64
            # 中计算 pinv，再将结果裁剪为非负并按列归一，使其
            # 更像能量聚合滤波器组，而非任意线性变换。
            Wb64 = W_bark.detach().double()
            We64 = W_eband.detach().double()
            M64 = torch.linalg.pinv(Wb64) @ We64
            M64 = M64.clamp(min=0.0)
            M64 = M64 / (M64.sum(dim=0, keepdim=True) + 1e-8)
            M = M64.float()
            self.band_agg_32_to_18 = FixedLinearProject(M)
        except Exception:
            # 回退到简易三角聚合（保底）
            def _build_agg(n_src: int, n_dst: int) -> torch.Tensor:
                idx_src = torch.arange(n_src, dtype=torch.float32)
                centers = torch.linspace(0, n_src - 1, n_dst)
                W = []
                for c in centers:
                    step = (n_src - 1) / max(n_dst - 1, 1)
                    left = c - step
                    right = c + step
                    w = torch.clamp(1.0 - (idx_src - c).abs() / max(step, 1e-6), min=0.0)
                    W.append(w / (w.sum() + 1e-8))
                return torch.stack(W, dim=1)
            agg = _build_agg(n_mels, 18)
            self.band_agg_32_to_18 = LearnableBandAgg(agg)

        # 2D VMamba JSCC 内容分支（默认）
        # channel_adaptive / lightweight_config 由上层配置驱动，便于在训练/推理
        # 时通过 CLI 切换 SNR 融合与 CSI 门控策略。
        self.content_vmamba = VMambaJSCC2D(
            in_ch=1,
            out_ch=1,
            channels=self.vm_channels,
            depths=self.vm_depths,
            d_s=d_s_content,
            csi_dim=d_csi,
            freq_downsample_stages=self.freq_downsample_stages,
            channel_adaptive=str(vm_channel_adaptive),
            lightweight_config=str(vm_lightweight_config),
        )

        # 简单 CNN BFCC JSCC baseline（可选；在 forward_content_only_no_hash 中使用）
        self.content_cnn_encoder = BFCCContentEncoder(
            latent_channels=self.content_cnn_latent_channels
        )
        self.content_cnn_decoder = BFCCContentDecoder(
            latent_channels=self.content_cnn_latent_channels
        )

        if self.with_hash:
            # 内容分支瓶颈：根据 quantizer_type 选择 Hash 或 RVQ。
            if self.quantizer_type == "hash":
                # Two-stage residual HashBottleneck：先粗量化再对残差做第二级量化；
                # 若启用 group_hash_content，则每一级内部采用 GroupedHashBottleneck。
                if self.group_hash_content:
                    num_groups = 4 if d_s_content >= 4 else 1
                    self.hash_content = TwoStageHashBottleneck(
                        input_dim=d_s_content,
                        hash_bits=self.hash_bits_content,
                        decoder_hidden=128,
                        output_dim=d_s_content,
                        hash_method="bihalf",   # train: BiHalf surrogate; eval: greedy/sign
                        channel_type="bsc",
                        use_grouped=True,
                        num_groups=num_groups,
                    )
                else:
                    self.hash_content = TwoStageHashBottleneck(
                        input_dim=d_s_content,
                        hash_bits=self.hash_bits_content,
                        decoder_hidden=128,
                        output_dim=d_s_content,
                        hash_method="bihalf",   # train: BiHalf surrogate; eval: greedy/sign
                        channel_type="bsc",
                        use_grouped=False,
                        num_groups=1,
                    )
            elif self.quantizer_type == "rvq":
                # RVQ bottleneck：残差向量量化，bits_total 与 hash_bits_content 对齐。
                self.hash_content = RVQBottleneck(
                    dim=d_s_content,
                    bits_total=self.hash_bits_content,
                    num_codebooks=max(1, int(self.rvq_nq_content)),
                    commitment=float(self.rvq_beta),
                    channel_type="bsc",
                    use_interleaver=True,
                )
            else:  # defensive, should be validated in __init__
                raise ValueError(f"Unsupported quantizer_type={self.quantizer_type}")

            # 内容分支统计量瓶颈：在 Hash 模式下使用 HashBottleneck，在 RVQ
            # 模式下改用 RVQBottleneck，使三条分支（内容/F0/统计量）在
            # quantizer_type 上保持一致，同时保持总 bit 数不变。
            if self.quantizer_type == "hash":
                self.hash_content_stats = HashBottleneck(
                    input_dim=2,
                    hash_bits=8,              # 2个标量使用更多比特以提升表达能力
                    decoder_hidden=64,         # 提升解码容量，缓解常数模板塌缩
                    output_dim=2,
                    hash_method="bihalf",    # 训练期BiHalf，推理期sign/greedy
                    channel_type="bsc",
                )
            elif self.quantizer_type == "rvq":
                self.hash_content_stats = RVQBottleneck(
                    dim=2,
                    bits_total=8,
                    num_codebooks=max(1, int(self.rvq_nq_content)),
                    commitment=float(self.rvq_beta),
                    channel_type="bsc",
                    use_interleaver=False,
                )

            # F0/VUV 分支的哈希瓶颈（将 z_fv 压缩为二值 / RVQ 码字，经过比特级信道再重建）。
            # F0/VUV 分支哈希位数可通过 hash_bits_f0 独立控制；
            # 若未显式指定，则回退为 max(4, hash_bits_content // 2)，
            # 保持与旧版模型近似的码率分配策略。
            hash_bits_f0_actual = (
                self.hash_bits_f0
                if self.hash_bits_f0 is not None
                else max(4, self.hash_bits_content // 2)
            )

            if self.quantizer_type == "hash":
                self.hash_f0vuv = HashBottleneck(
                    input_dim=d_zf,
                    hash_bits=hash_bits_f0_actual,
                    decoder_hidden=64,
                    output_dim=d_zf,
                    hash_method="greedy",
                    channel_type="bsc",
                )
            else:  # rvq
                n_q_f0 = int(self.rvq_nq_f0) if self.rvq_nq_f0 is not None else int(self.rvq_nq_content)
                self.hash_f0vuv = RVQBottleneck(
                    dim=d_zf,
                    bits_total=hash_bits_f0_actual,
                    num_codebooks=max(1, n_q_f0),
                    commitment=float(self.rvq_beta),
                    channel_type="bsc",
                    use_interleaver=True,
                )

            # ---- 轻量时序预测器：content / F0 残差 hash ----
            # 内容分支：s_t = VMamba token，g 只看 s_{<t} 预测 s_pred_t，
            # hash 只编码 r_t = s_t - s_pred_t。
            self.content_pred_gru = nn.GRU(
                input_size=d_s_content,
                hidden_size=d_s_content,
                num_layers=1,
                batch_first=True,
            )
            self.content_pred_out = nn.Linear(d_s_content, d_s_content)
            self.content_pred_norm = nn.LayerNorm(d_s_content)

            # F0/VUV 分支：同理在 z_fv 上做残差预测，hash 只编码残差，
            # 并在时间维用 frame_corr 做简单的 voiced-aware gate。
            self.f0_pred_gru = nn.GRU(
                input_size=d_zf,
                hidden_size=d_zf,
                num_layers=1,
                batch_first=True,
            )
            self.f0_pred_out = nn.Linear(d_zf, d_zf)
            self.f0_pred_norm = nn.LayerNorm(d_zf)

        # F0/voicing 分支
        self.f0vuv_enc = F0VUVEncoder(in_dim=2, d_zf=d_zf, hidden=hidden_f0)
        self.f0vuv_jscc_enc = JSCCEncoder(d_z=d_zf, d_s=d_s_f0, d_csi=d_csi, hidden=hidden_f0)
        self.f0vuv_jscc_dec = JSCCDecoder(d_z=d_zf, d_s=d_s_f0, d_csi=d_csi, hidden=hidden_f0)
        # 在解码端启用 content-conditioned F0 decoder：使用 n_mels 作为
        # 条件维度，以 mel_hat_norm[B,T,n_mels] 作为 default 上下文。
        self.f0vuv_dec = F0VUVDecoder(d_zf=d_zf, hidden=hidden_f0, cond_dim=n_mels)
        # 训练期的亮度/能量校准（减少 mel → ceps 的整体偏移）
        self.enable_energy_calib = False
        self.energy_calib_alpha = 0.8  # 每步校正比例（0~1）
        # L2H 融合系数（由训练脚本按步数渐进设定，0: 仅用基线高频；1: 完全采用细化高频）
        self.l2h_blend: float = 1.0
        # Cepstral delta quality schedule (cosine ramp, updated by training script)
        self.ceps_delta_alpha: float = 0.0

        # Vocoder
        self.vocoder = FARGANDecoder(
            fargan_subframe_size=40,
            fargan_nb_subframes=4,
            frame_rate_hz=100.0,
        )

        # 低→高 mel 细化头（可选，默认关闭），统一采用 DeCo 风格 L2H。
        self.enable_l2h = bool(with_l2h)
        self.l2h_low_bins = int(l2h_low_bins)
        self.l2h_resid_scale = float(l2h_resid_scale)
        self.l2h_dual_head = bool(l2h_dual_head)
        self.l2h_harmonic_cutoff_hz = float(l2h_harmonic_cutoff_hz)

        self.use_l2h_flow = bool(use_l2h_flow)
        self.deco_l2h_hidden = int(deco_l2h_hidden)
        self.deco_l2h_blocks = int(deco_l2h_blocks)

        self.deco_l2h_refiner: Optional[DeCoL2HRefiner] = None
        self.l2h_flow: Optional[ConditionalHFGenerator] = None

        if self.enable_l2h:
            self.deco_l2h_refiner = DeCoL2HRefiner(
                n_mels=n_mels,
                low_bins=self.l2h_low_bins,
                hidden=self.deco_l2h_hidden,
                n_blocks=self.deco_l2h_blocks,
                resid_scale=self.l2h_resid_scale,
                dual_head=self.l2h_dual_head,
                harmonic_cutoff_hz=self.l2h_harmonic_cutoff_hz,
                band_centers_hz=getattr(self, "bark_centers_hz", None),
            )

            if self.use_l2h_flow:
                self.l2h_flow = ConditionalHFGenerator(
                    n_mels=n_mels,
                    low_bins=self.l2h_low_bins,
                    hidden=int(l2h_flow_hidden),
                    n_flows=int(l2h_flow_n_flows),
                )

    @staticmethod
    def _vuv_logits_to_feat(vuv_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert VUV logits to (frame_corr_feat, vuv_prob).

        - ``vuv_prob``: ``sigmoid(logits)`` in ``[0, 1]``.
        - ``frame_corr_feat``: affine map of prob into ``[-0.8, 0.5]``,
          matching FARGAN/F0 branch correlation semantics.
        """

        vuv_prob = torch.sigmoid(vuv_logits)
        frame_corr_feat = vuv_prob * 1.3 - 0.8
        return frame_corr_feat, vuv_prob

        if self.enable_l2h:
            self.deco_l2h_refiner = DeCoL2HRefiner(
                n_mels=n_mels,
                low_bins=self.l2h_low_bins,
                hidden=self.deco_l2h_hidden,
                n_blocks=self.deco_l2h_blocks,
                resid_scale=self.l2h_resid_scale,
                dual_head=self.l2h_dual_head,
                harmonic_cutoff_hz=self.l2h_harmonic_cutoff_hz,
                band_centers_hz=getattr(self, "bark_centers_hz", None),
            )


            if self.use_l2h_flow:
                self.l2h_flow = ConditionalHFGenerator(
                    n_mels=n_mels,
                    low_bins=self.l2h_low_bins,
                    hidden=int(l2h_flow_hidden),
                    n_flows=int(l2h_flow_n_flows),
                )

        # HF 侧通道：将高频残差特征直接传给 FARGAN
        # 解决 32->18 聚合后高频细节丢失的问题
        self.with_hf_sideband = bool(with_hf_sideband)
        self.hf_sideband_dim = int(hf_sideband_dim)
        self.hf_sideband_type = str(hf_sideband_type)
        self.hf2ceps_dim = int(max(1, hf2ceps_dim))
        self.hf2ceps_scale = float(hf2ceps_scale)
        self.hf_sideband_encoder: Optional[nn.Module] = None
        self.hf2ceps: Optional[nn.Module] = None

        if self.with_hf_sideband:
            hf_input_dim = n_mels - self.l2h_low_bins  # 高频 mel bins 数量
            if self.hf_sideband_type == "learnable":
                # 可学习的线性投影：从高频残差提取紧凑表示
                self.hf_sideband_encoder = nn.Sequential(
                    nn.Linear(hf_input_dim, self.hf_sideband_dim * 2),
                    nn.GELU(),
                    nn.Linear(self.hf_sideband_dim * 2, self.hf_sideband_dim),
                )
            elif self.hf_sideband_type == "dct":
                # 固定 DCT 投影：取高频残差的前 K 个 DCT 系数
                dct_mat = self._build_dct_mat(hf_input_dim, self.hf_sideband_dim)
                self.register_buffer("hf_dct_proj", dct_mat)
            else:
                # 简单线性投影
                self.hf_sideband_encoder = nn.Linear(hf_input_dim, self.hf_sideband_dim)

            # HF 侧通道 → 倒谱高阶校正的小 MLP
            self.hf2ceps = nn.Sequential(
                nn.Linear(self.hf_sideband_dim, 32),
                nn.GELU(),
                nn.Linear(32, self.hf2ceps_dim),
            )
            # 初始时弱化校正，以便从“接近恒等”开始训练
            with torch.no_grad():
                last_linear = self.hf2ceps[-1]
                if isinstance(last_linear, nn.Linear):
                    nn.init.zeros_(last_linear.weight)
                    nn.init.zeros_(last_linear.bias)

        # 可选：在 vocoder period override 阶段使用统一的能量掩膜
        # （默认阈值来源于训练配置 silence_energy_thr_db，若未设置则回退到 -40 dB）
        self.silence_energy_thr_db: float = -40.0

        # Optional BFCC-based vocoder for side-by-side debugging.
        # 不改变现有 FARGAN 路径，仅在 DBG_BFCC_VOCODER 或
        # use_bfcc_vocoder_debug=True 时，额外生成一条 BFCC 声码器
        # 输出 audio_hat_bfcc，便于离线比较两条 vocoder 的表现。
        dbg_flag = False
        try:
            dbg_flag = os.environ.get("DBG_BFCC_VOCODER", "0") == "1"
        except Exception:
            dbg_flag = False
        self.use_bfcc_vocoder_debug: bool = bool(use_bfcc_vocoder_debug or dbg_flag)
        self.bfcc_vocoder: Optional[BFCCVocoder] = None
        if self.use_bfcc_vocoder_debug:
            try:
                self.bfcc_vocoder = BFCCVocoder(
                    bfcc_subframe_size=40,
                    bfcc_nb_subframes=4,
                    frame_rate_hz=100.0,
                    soft_period=True,
                ).to(self.device)
                print("[BFCCVocoder] Debug BFCC vocoder enabled inside DualBranchBarkJSCC.")
            except Exception as _e:
                self.bfcc_vocoder = None
                print(f"[BFCCVocoder] WARNING: failed to init BFCCVocoder debug path: {_e}")

        self.to(self.device)

    def _content_pre_downsample(self, mel_img: torch.Tensor) -> torch.Tensor:
        """在进入内容分支 VMamba 前做时间维下采样。

        Args:
            mel_img: [B,1,T,F] 的 Bark/BFCC 图像。

        Returns:
            若 ``content_time_downsample == 1``，原样返回；
            否则按因子 k 在时间维做平均池化，下采样为 T/k。
        """

        if self.content_time_downsample <= 1:
            return mel_img
        k = int(self.content_time_downsample)
        # 仅在时间维做 avg pooling，频率维保持不变
        return F.avg_pool2d(mel_img, kernel_size=(k, 1), stride=(k, 1))

    @staticmethod
    def _build_dct_mat(n_in: int, n_out: int) -> torch.Tensor:
        """构建 DCT-II 投影矩阵，用于 HF 侧通道的固定 DCT 编码。

        Args:
            n_in: 输入维度（高频 mel bins）
            n_out: 输出维度（侧通道维度）

        Returns:
            DCT 矩阵 [n_in, n_out]
        """
        n = torch.arange(n_in, dtype=torch.float32).unsqueeze(1)
        k = torch.arange(n_out, dtype=torch.float32).unsqueeze(0)
        mat = torch.cos((n + 0.5) * k * math.pi / float(n_in))
        mat[:, 0] *= math.sqrt(0.5)
        mat = mat * math.sqrt(2.0 / float(n_in))
        return mat

    # ------------------------------------------------------------------
    # JSCC 编码/解码辅助接口（不改变原有前向逻辑）
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_content_jscc(
        self,
        audio: torch.Tensor,
        fargan_feats: torch.Tensor,
        channel_sim,
        snr_min_db: float,
        snr_max_db: float,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """将波形编码为 VMamba JSCC 内容符号（含信道噪声），返回 [B,L,C]。

        返回值:
            s_tokens_noisy_flat: [B, L_seq, d_s_content]，已经过 channel_sim.apply。
            meta: 包含 T、F_mel、hw、csi_vec，用于后续解码。
        """

        device = self.device
        audio = audio.to(device)
        fargan_feats = fargan_feats.to(device)

        B, L = audio.shape
        B2, T_feat, _ = fargan_feats.shape
        assert B == B2

        mel = self.wave_to_mel(audio)  # [B,T_mel,32]
        Bm, T_mel, F_mel = mel.shape
        T = min(T_mel, T_feat)
        mel = mel[:, :T, :]

        # 显式归一化：编码前归一化，保存统计量供解码端反归一化
        mel_mean = mel.mean(dim=(1, 2), keepdim=True)  # [B,1,1]
        mel_std = mel.std(dim=(1, 2), keepdim=True).clamp(min=0.1)  # [B,1,1]
        mel_norm = (mel - mel_mean) / mel_std  # [B,T,32]

        mel_img = mel_norm.unsqueeze(1)  # [B,1,T,F_mel] 使用归一化后的mel
        mel_img_c = self._content_pre_downsample(mel_img)

        # 全局 CSI → encoder FiLM
        # 编码端 CSI 设置：
        # - 若 snr_min_db == snr_max_db（离线导出场景），则使用与 decode_from_bits_offline
        #   一致的简化 CSI：[snr_db, 0.5, 0.5, 0.5]；
        # - 否则保留原来的 ChannelSimulator.sample_csi 行为（训练/在线推理）。
        if snr_min_db == snr_max_db:
            snr = torch.full((B,), float(snr_min_db), device=device, dtype=mel.dtype)
            time_sel = torch.full((B,), 0.5, device=device, dtype=mel.dtype)
            freq_sel = torch.full((B,), 0.5, device=device, dtype=mel.dtype)
            los_ratio = torch.full((B,), 0.5, device=device, dtype=mel.dtype)
            csi_vec = torch.stack([snr, time_sel, freq_sel, los_ratio], dim=-1)  # [B,4]
        else:
            csi_dict_enc, _, _ = channel_sim.sample_csi(
                B, T, channel="fading", snr_min_db=snr_min_db, snr_max_db=snr_max_db
            )
            csi_vec = torch.stack(
                [
                    csi_dict_enc["snr_proxy"],
                    csi_dict_enc["time_selectivity"],
                    csi_dict_enc["freq_selectivity"],
                    csi_dict_enc["los_ratio"],
                ],
                dim=-1,
            )


        _, s_tokens, hw = self.content_vmamba.encode(mel_img_c, csi_vec)  # [B,C,H,W]
        Bc, Cc, H, W = s_tokens.shape
        L_seq = H * W

        # 为符号序列采样逐 token CSI，并应用信道噪声
        _csi_tmp, amp_t, snr_db_t = channel_sim.sample_csi(
            B, L_seq, channel="fading", snr_min_db=snr_min_db, snr_max_db=snr_max_db
        )
        amp_t = amp_t.to(device=s_tokens.device, dtype=s_tokens.dtype)
        snr_db_t = snr_db_t.to(device=s_tokens.device, dtype=s_tokens.dtype)

        s_tokens_flat = s_tokens.permute(0, 2, 3, 1).contiguous().view(B, L_seq, Cc)  # [B,L,C]
        s_tokens_noisy_flat = channel_sim.apply(s_tokens_flat, amp_t, snr_db_t)       # [B,L,C]
        if self.eq_fading:
            s_tokens_noisy_flat = s_tokens_noisy_flat / (amp_t.unsqueeze(-1) + 1e-3)

        meta: Dict[str, Any] = {
            "T": int(T),
            "F_mel": int(F_mel),
            "hw": (int(H), int(W)),
            "csi_vec": csi_vec.detach(),
            "mel_mean": mel_mean.detach(),  # 用于解码后反归一化
            "mel_std": mel_std.detach(),
        }
        return s_tokens_noisy_flat, meta

    @torch.no_grad()
    def decode_content_jscc(
        self,
        s_tokens_noisy_flat: torch.Tensor,
        meta: Dict[str, Any],
    ) -> torch.Tensor:
        """从 VMamba JSCC 内容符号解码 mel 估计图。

        Args:
            s_tokens_noisy_flat: [B, L_seq, d_s_content]
            meta: encode_content_jscc 返回的元信息（T, F_mel, hw, csi_vec）。

        Returns:
            mel_hat: [B, T, F_mel]
        """

        device = self.device
        s_tokens_noisy_flat = s_tokens_noisy_flat.to(device)

        B, L_seq, Cc = s_tokens_noisy_flat.shape
        H, W = meta["hw"]
        T = int(meta["T"])
        F_mel = int(meta["F_mel"])
        csi_vec = meta["csi_vec"].to(device)

        # [B,L,C] → [B,C,H,W]
        s_tokens_noisy = s_tokens_noisy_flat.view(B, H, W, Cc).permute(0, 3, 1, 2).contiguous()
        mel_hat_img_norm = self.content_vmamba.decode(s_tokens_noisy, csi_vec, meta["hw"])  # [B,1,T',F'] 归一化空间
        mel_hat_img_norm = F.interpolate(mel_hat_img_norm, size=(T, F_mel), mode="bilinear", align_corners=False)
        mel_hat_norm = mel_hat_img_norm.squeeze(1)  # [B,T,F_mel] 归一化空间

        # 反归一化：恢复原始能量水平
        mel_mean = meta["mel_mean"].to(device)
        mel_std = meta["mel_std"].to(device)
        mel_hat = mel_hat_norm * mel_std + mel_mean  # [B,T,F_mel]
        return mel_hat

    @torch.no_grad()
    def encode_f0_jscc(
        self,
        fargan_feats: torch.Tensor,
        channel_sim,
        snr_min_db: float,
        snr_max_db: float,
        csi_vec: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """将 FARGAN 36 维特征中的 F0/VUV 部分编码为 JSCC 符号，并通过信道噪声。

        若提供 ``csi_vec``，则与内容分支共用同一 CSI；否则内部采样一个新的。
        """

        device = self.device
        fargan_feats = fargan_feats.to(device)
        B, T, _ = fargan_feats.shape

        # F0/VUV 分支在 AMP 场景下数值较敏感，
        # 强制在 float32 中计算以避免半精度溢出/下溢。
        with amp_autocast(enabled=False):
            dnn_pitch = self.fargan_spec.extract_feature(fargan_feats, "dnn_pitch")  # [B,T,1]
            frame_corr = self.fargan_spec.extract_feature(fargan_feats, "frame_corr")  # [B,T,1]
            dnn_pitch = torch.nan_to_num(dnn_pitch, nan=0.0, posinf=3.0, neginf=-3.0).clamp(-3.0, 3.0)
            frame_corr = torch.nan_to_num(frame_corr, nan=0.0, posinf=0.5, neginf=-0.8).clamp(-0.8, 0.5)
            f0vuv = torch.cat([dnn_pitch, frame_corr], dim=-1)  # [B,T,2]

            z_fv = self.f0vuv_enc(f0vuv)  # [B,T,d_zf]

        # 若未提供 CSI 向量，则一次性采样全局 CSI + 帧级衰落/SNR；
        # 否则仅根据外部给定的 csi_vec 做 FiLM，噪声仍单独采样。
        amp_t_f: torch.Tensor
        snr_db_t_f: torch.Tensor
        if csi_vec is None:
            csi_dict_enc, amp_t_f, snr_db_t_f = channel_sim.sample_csi(
                B, T, channel="fading", snr_min_db=snr_min_db, snr_max_db=snr_max_db
            )
            csi_vec = torch.stack(
                [
                    csi_dict_enc["snr_proxy"],
                    csi_dict_enc["time_selectivity"],
                    csi_dict_enc["freq_selectivity"],
                    csi_dict_enc["los_ratio"],
                ],
                dim=-1,
            )

        else:
            # 仍需为该分支采样一条具体的衰落轨迹与 SNR 序列
            _csi_tmp, amp_t_f, snr_db_t_f = channel_sim.sample_csi(
                B, T, channel="fading", snr_min_db=snr_min_db, snr_max_db=snr_max_db
            )

        # JSCC 编码也保持在 float32 中运行
        with amp_autocast(enabled=False):
            s_fv = self.f0vuv_jscc_enc(z_fv, csi_vec)  # [B,T,d_s_f0]
        amp_t_f = amp_t_f.to(device=s_fv.device, dtype=s_fv.dtype)
        snr_db_t_f = snr_db_t_f.to(device=s_fv.device, dtype=s_fv.dtype)
        s_fv_noisy = channel_sim.apply(s_fv, amp_t_f, snr_db_t_f)  # [B,T,d_s_f0]

        meta: Dict[str, Any] = {
            "T": int(T),
            "csi_vec": csi_vec.detach(),
        }
        return s_fv_noisy, meta

    @torch.no_grad()
    def decode_f0_jscc(
        self,
        s_fv_noisy: torch.Tensor,
        meta: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """从 F0 JSCC 符号中解码 dnn_pitch_hat 与 VUV 相关特征。

        Args:
            s_fv_noisy: [B,T,d_s_f0]，已经过信道噪声。
            meta: 来自 ``encode_f0_jscc`` 的元信息（T, csi_vec）。

        Returns:
            (dnn_pitch_hat, frame_corr_hat, vuv_prob): 形状均为 [B,T,1]。
        """

        device = self.device
        s_fv_noisy = s_fv_noisy.to(device)
        csi_vec = meta["csi_vec"].to(device)

        z_fv_hat = self.f0vuv_jscc_dec(s_fv_noisy, csi_vec, h_rayleigh=None)
        # decode_f0_jscc 用于 F0-only JSCC 路径，此处不具备 mel 条件信息，
        # 因此保持解码器在无 mel_cond 时的退化行为。
        dnn_pitch_hat, vuv_logits = self.f0vuv_dec(z_fv_hat)
        frame_corr_hat, vuv_prob = self._vuv_logits_to_feat(vuv_logits)
        return dnn_pitch_hat, frame_corr_hat, vuv_prob

    # ---- Hash codec encode/decode helpers ---------------------------------

    @torch.no_grad()
    def encode_hash_codec(
        self,
        audio: torch.Tensor,
        fargan_feats: torch.Tensor,
        channel_sim,
        snr_min_db: float,
        snr_max_db: float,
        *,
        return_meta: bool = True,
        use_noisy_bits: bool = False,
    ):
        """Export hash bitstreams for offline decode (+ meta).

        Args:
            audio:        [B,L] waveform
            fargan_feats: [B,T,36] FARGAN features
            channel_sim:  ChannelSimulator used during training
            snr_min_db:   SNR lower bound used for CSI sampling
            snr_max_db:   SNR upper bound used for CSI sampling
            return_meta:  If True, also return structural/channel meta
            use_noisy_bits: If True, prefer BPSK+AWGN "soft" bits
                (``hash_bits_noisy``) when available; otherwise use
                clean hard bits (``hash_bits_clean``).

        Returns:
            bits_c : [B, Lc, Kc]
            bits_f : [B, Tf, Kf]
            bits_s : [B, Ls, Ks]  (mel mean/std side info)
            meta   : dict with T/F_mel/hw/csi_vec (when ``return_meta``)
        """
        # 强制走与训练一致的 hash_only 分支，避免 encode 复制一套逻辑导致漂移。
        # 其中 forward_with_hash 会同时返回 clean/noisy bits 与必要 meta。
        ret = self.forward_with_hash(
            audio=audio,
            fargan_feats=fargan_feats,
            channel_sim=channel_sim,
            snr_min_db=snr_min_db,
            snr_max_db=snr_max_db,
            hash_only=True,
        )

        if use_noisy_bits:
            bits_c = ret.get("hash_bits_noisy", ret.get("hash_bits_clean", None))
            bits_f = ret.get("f0_hash_bits_noisy", ret.get("f0_hash_bits_clean", None))
            bits_s = ret.get("hash_bits_stats_noisy", ret.get("hash_bits_stats", None))
        else:
            bits_c = ret.get("hash_bits_clean", None)
            bits_f = ret.get("f0_hash_bits_clean", None)
            bits_s = ret.get("hash_bits_stats", None)

        if not return_meta:
            return bits_c, bits_f, bits_s

        meta = {
            "T": int(ret.get("T", fargan_feats.shape[1])),
            "F_mel": int(ret.get("F_mel", 32)),
            "hw": ret.get("hw", None),  # (H,W) token grid
        }
        if "csi_vec" in ret and ret["csi_vec"] is not None:
            # 训练路径中通常使用 numpy sidecar，保持兼容性
            csi_val = ret["csi_vec"]
            if isinstance(csi_val, torch.Tensor):
                meta["csi_vec"] = csi_val.detach().cpu().numpy()
            else:
                meta["csi_vec"] = csi_val

        if bits_c is not None:
            meta["content_shape"] = tuple(bits_c.shape)
        if bits_f is not None:
            meta["f0_shape"] = tuple(bits_f.shape)
        if bits_s is not None:
            meta["stats_shape"] = tuple(bits_s.shape)

        return bits_c, bits_f, bits_s, meta

    # ---- Quantizer-agnostic codec helpers (Hash/RVQ) ----------------------

    @torch.no_grad()
    def encode_quant_codec(
        self,
        audio: torch.Tensor,
        fargan_feats: torch.Tensor,
        channel_sim,
        snr_min_db: float,
        snr_max_db: float,
        *,
        return_meta: bool = True,
        use_noisy_bits: bool = False,
    ):
        """Quantizer-agnostic bitstream export wrapper.

        当前实现直接委托给 ``encode_hash_codec``，但“hash”一词仅
        代表Aether-lite内部沿用的接口命名；具体量化器类型由
        ``self.quantizer_type`` 决定，可为 HashBottleneck 或 RVQ。
        """

        return self.encode_hash_codec(
            audio=audio,
            fargan_feats=fargan_feats,
            channel_sim=channel_sim,
            snr_min_db=snr_min_db,
            snr_max_db=snr_max_db,
            return_meta=return_meta,
            use_noisy_bits=use_noisy_bits,
        )
    @torch.no_grad()
    def decode_hash_codec(
        self,
        bits_content: Optional[torch.Tensor],
        bits_f0: Optional[torch.Tensor],
        audio: torch.Tensor,
        fargan_feats: torch.Tensor,
        channel_sim,
        snr_min_db: float,
        snr_max_db: float,
        target_len: Optional[int] = None,
        *,
        meta: Optional[dict] = None,
        bits_stats: Optional[torch.Tensor] = None,
        f0_T: Optional[int] = None,
        csi_vec: Optional[torch.Tensor] = None,
        snr_db: Optional[float] = None,
        content_hw: Optional[Tuple[int, int]] = None,
        gt_mel_mean: Optional[torch.Tensor] = None,
        gt_mel_std: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Decode audio and intermediate features from external quant bits.

        Compatibility strategy:
          - If ``meta`` is provided (recommended for offline bitstream decode),
            we go *decoder-only* via ``decode_quant_codec`` (works for Hash/RVQ).
          - If ``meta`` is not provided, we try cached encoder meta (``_hash_meta_*``).
          - If neither exists, we fallback to the legacy patch+forward path so
            old callers can still run.

        Notes:
          - RVQ uses ``decode_bits``; Hash uses ``decode_hash_codec`` / ``hash_decoder``.
          - ``bits_stats`` is optional side info (mel mean/std) and will be used when provided.
        """

        device = self.device
        audio = audio.to(device)
        fargan_feats = fargan_feats.to(device)

        # No external bits -> internal forward
        if bits_content is None and bits_f0 is None and bits_stats is None:
            return self.forward_with_hash(
                audio=audio,
                fargan_feats=fargan_feats,
                channel_sim=channel_sim,
                snr_min_db=snr_min_db,
                snr_max_db=snr_max_db,
                target_len=target_len or audio.size(-1),
                hash_only=False,
            )

        def _bits_to_sign(b: torch.Tensor) -> torch.Tensor:
            b = torch.as_tensor(b, device=device)
            if b.dtype == torch.bool:
                b = b.to(torch.float32)
            else:
                b = b.to(torch.float32)
            if b.numel() == 0:
                return b
            bmin = float(b.min().item())
            bmax = float(b.max().item())
            if os.getenv("DBG_BITS_ONLY", "0") == "1":
                print(f"[bits2sign] pre  dtype={b.dtype} bmin={bmin:+.3f} bmax={bmax:+.3f} shape={tuple(b.shape)}")

            # stored as 0/1 -> convert to -1/+1
            if bmin >= 0.0 and bmax <= 1.0:
                b = b * 2.0 - 1.0
            if os.getenv("DBG_BITS_ONLY", "0") == "1":
                print(f"[bits2sign] post dtype={b.dtype} min={float(b.min()):+.3f} max={float(b.max()):+.3f} "
                    f"pos={(b>0).float().mean().item():.3f} neg={(b<0).float().mean().item():.3f}")

            return torch.clamp(b, -3.0, 3.0)

        def _hb_decode_any(hb, bits: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            """Decode bits -> latent for Hash / Grouped / TwoStage / RVQ."""
            if hb is None or bits is None:
                return None
            b = _bits_to_sign(bits)
            if os.getenv("DBG_BITS","0") == "1":
                bp = (b > 0).float().mean(dim=(0,1)).detach().cpu()
                bz = (b == 0).float().mean(dim=(0,1)).detach().cpu()
                print(f"[DEC bits] p(+1) per-bit={bp.numpy()} p(0) per-bit={bz.numpy()}")

            if b.dim() == 2:
                b = b.unsqueeze(1)

            # RVQ: preferred
            if hasattr(hb, "decode_bits") and callable(getattr(hb, "decode_bits")):
                if os.getenv("DBG_BITS","0") == "1" and hasattr(hb, "bits_sign_to_codes"):
                    try:
                        codes_dbg = hb.bits_sign_to_codes(b)
                        _rvq_code_hist("DEC codes(from bits)", hb, codes_dbg)
                    except Exception as e:
                        print("[DEC codes] failed:", e)

                return hb.decode_bits(b)

            # GroupedHashBottleneck: split by groups
            if hasattr(hb, "groups") and hasattr(hb, "hash_bits_per_group"):
                groups = list(getattr(hb, "groups"))
                bits_per = [int(x) for x in list(getattr(hb, "hash_bits_per_group"))]
                xs = []
                start = 0
                for sub_hb, k_g in zip(groups, bits_per):
                    b_g = b[..., start : start + k_g]
                    start += k_g
                    x_g = _hb_decode_any(sub_hb, b_g)
                    if x_g is None:
                        out_dim = int(getattr(sub_hb, "output_dim", 0))
                        x_g = torch.zeros(
                            b.size(0), b.size(1), out_dim,
                            device=device, dtype=torch.float32
                        )
                    xs.append(x_g)
                return torch.cat(xs, dim=-1) if len(xs) else None

            # Two-stage (Hash): stage1 + stage2
            if hasattr(hb, "stage1") and hasattr(hb, "stage2"):
                stage_bits = getattr(hb, "stage_bits", None)
                if stage_bits is not None and len(stage_bits) >= 2:
                    k1, k2 = int(stage_bits[0]), int(stage_bits[1])
                else:
                    k1 = int(getattr(hb, "stage1_bits"))
                    k2 = int(getattr(hb, "stage2_bits"))
                b1 = b[..., :k1]
                b2 = b[..., k1 : k1 + k2]
                x1 = _hb_decode_any(getattr(hb, "stage1"), b1)
                x2 = _hb_decode_any(getattr(hb, "stage2"), b2)
                if x1 is None:
                    return x2
                if x2 is None:
                    return x1
                return x1 + x2

            # HashBottleneck-style
            if hasattr(hb, "decode_hash_codec") and callable(getattr(hb, "decode_hash_codec")):
                return hb.decode_hash_codec(b)
            if hasattr(hb, "hash_decoder") and callable(getattr(hb, "hash_decoder")):
                return hb.hash_decoder(b)

            raise TypeError(f"Unsupported bottleneck type for decode: {type(hb)}")

        # -------------------------------------------------
        # Decoder-only: prefer explicit meta, then cached meta
        # -------------------------------------------------
        meta_use = meta
        if meta_use is None:
            meta_T = getattr(self, "_hash_meta_T", None)
            meta_F_mel = getattr(self, "_hash_meta_F_mel", None)
            meta_H = getattr(self, "_hash_meta_H", None)
            meta_W = getattr(self, "_hash_meta_W", None)
            meta_csi = getattr(self, "_hash_meta_csi", None)
            if all(v is not None for v in (meta_T, meta_F_mel, meta_H, meta_W, meta_csi)):
                meta_use = {
                    "T": int(meta_T),
                    "F_mel": int(meta_F_mel),
                    "hw": (int(meta_H), int(meta_W)),
                    "csi_vec": meta_csi,
                }

        if meta_use is not None or csi_vec is not None or snr_db is not None or content_hw is not None:
            # csi
            csi_in = csi_vec
            if csi_in is None and meta_use is not None:
                csi_in = meta_use.get("csi_vec", None)
            if csi_in is not None and not isinstance(csi_in, torch.Tensor):
                csi_in = torch.as_tensor(csi_in, device=device, dtype=torch.float32)

            # content hw
            hw_in = content_hw
            if hw_in is None and meta_use is not None:
                hw0 = meta_use.get("hw", None)
                # allow batched hw like [[H,W], ...] / [(H,W), ...]
                if isinstance(hw0, (list, tuple)) and len(hw0) > 0 and isinstance(hw0[0], (list, tuple)):
                    hw0 = hw0[0]
                if isinstance(hw0, (list, tuple)) and len(hw0) == 2:
                    hw_in = (int(hw0[0]), int(hw0[1]))

            # f0_T
            f0_T_in = f0_T
            if f0_T_in is None and bits_f0 is not None:
                f0_T_in = int(torch.as_tensor(bits_f0).shape[1])
            if f0_T_in is None and meta_use is not None:
                f0_shape = meta_use.get("f0_shape", None)
                if isinstance(f0_shape, (list, tuple)) and len(f0_shape) >= 2:
                    try:
                        f0_T_in = int(f0_shape[1])
                    except Exception:
                        pass

            return self.decode_quant_codec(
                bits_content=bits_content,
                bits_f0=bits_f0,
                bits_stats=bits_stats,
                f0_T=f0_T_in,
                target_len=target_len or audio.size(-1),
                csi_vec=csi_in,
                snr_db=snr_db,
                content_hw=hw_in,
                gt_mel_mean=gt_mel_mean,
                gt_mel_std=gt_mel_std,
            )

        # -------------------------------------------------
        # Legacy fallback: patch bottleneck.forward and call forward_with_hash
        # (keeps old callers working; now also supports RVQ)
        # -------------------------------------------------
        orig_forward_c = None
        orig_forward_f = None
        hash_mod_c = None
        hash_mod_f = None
        try:
            if bits_content is not None:
                if not hasattr(self, "hash_content"):
                    raise RuntimeError("decode_hash_codec: model has no 'hash_content' module")
                hash_mod_c = self.hash_content

                def _forward_override_c(self_mod, x, channel_params=None, mask=None):  # type: ignore[override]
                    bits = self_mod._external_bits_content.to(device=x.device)
                    reconstructed = _hb_decode_any(self_mod, bits)
                    if reconstructed is None:
                        reconstructed = torch.zeros(x.size(0), x.size(1), int(getattr(self_mod, "output_dim", 0)), device=x.device)
                    zeros = torch.zeros_like(torch.as_tensor(bits, device=x.device))
                    return {
                        "hash_logits": zeros,
                        "hash_bits_clean": bits,
                        "hash_bits_noisy": bits,
                        "reconstructed": reconstructed,
                    }

                orig_forward_c = hash_mod_c.forward
                hash_mod_c._external_bits_content = bits_content.to(device)
                hash_mod_c.forward = _forward_override_c.__get__(hash_mod_c, type(hash_mod_c))  # type: ignore[assignment]

            if bits_f0 is not None:
                if not hasattr(self, "hash_f0vuv"):
                    raise RuntimeError("decode_hash_codec: model has no 'hash_f0vuv' module")
                hash_mod_f = self.hash_f0vuv

                def _forward_override_f(self_mod, x, channel_params=None, mask=None):  # type: ignore[override]
                    bits = self_mod._external_bits_f0.to(device=x.device)
                    reconstructed = _hb_decode_any(self_mod, bits)
                    if reconstructed is None:
                        reconstructed = torch.zeros(x.size(0), x.size(1), int(getattr(self_mod, "output_dim", 0)), device=x.device)
                    zeros = torch.zeros_like(torch.as_tensor(bits, device=x.device))
                    return {
                        "hash_logits": zeros,
                        "hash_bits_clean": bits,
                        "hash_bits_noisy": bits,
                        "reconstructed": reconstructed,
                    }

                orig_forward_f = hash_mod_f.forward
                hash_mod_f._external_bits_f0 = bits_f0.to(device)
                hash_mod_f.forward = _forward_override_f.__get__(hash_mod_f, type(hash_mod_f))  # type: ignore[assignment]

            out = self.forward_with_hash(
                audio=audio,
                fargan_feats=fargan_feats,
                channel_sim=channel_sim,
                snr_min_db=snr_min_db,
                snr_max_db=snr_max_db,
                target_len=target_len or audio.size(-1),
                hash_only=False,
            )
        finally:
            if hash_mod_c is not None and orig_forward_c is not None:
                hash_mod_c.forward = orig_forward_c  # type: ignore[assignment]
                if hasattr(hash_mod_c, "_external_bits_content"):
                    delattr(hash_mod_c, "_external_bits_content")
            if hash_mod_f is not None and orig_forward_f is not None:
                hash_mod_f.forward = orig_forward_f  # type: ignore[assignment]
                if hasattr(hash_mod_f, "_external_bits_f0"):
                    delattr(hash_mod_f, "_external_bits_f0")

        return out

    @torch.no_grad()
    def decode_quant_codec(
        self,
        bits_content: Optional[torch.Tensor],
        bits_f0: Optional[torch.Tensor],
        bits_stats: Optional[torch.Tensor] = None,
        *,
        f0_T: Optional[int] = None,
        target_len: Optional[int] = None,
        csi_vec: Optional[torch.Tensor] = None,
        snr_db: Optional[float] = None,
        content_hw: Optional[Tuple[int, int]] = None,
        gt_mel_mean: Optional[torch.Tensor] = None,
        gt_mel_std: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Quantizer-agnostic offline decode wrapper.

        与 ``decode_from_bits_offline`` 完全共享实现，只是对外名字
        不再绑定到 Hash，实现 Hash / RVQ 统一接口。"""

        return self.decode_from_bits_offline(
            bits_content=bits_content,
            bits_f0=bits_f0,
            bits_stats=bits_stats,
            f0_T=f0_T,
            target_len=target_len,
            csi_vec=csi_vec,
            snr_db=snr_db,
            content_hw=content_hw,
            gt_mel_mean=gt_mel_mean,
            gt_mel_std=gt_mel_std,
        )

    def decode_from_bits_offline(
        self,
        bits_content: Optional[torch.Tensor],
        bits_f0: Optional[torch.Tensor],
        bits_stats: Optional[torch.Tensor] = None,
        f0_T: Optional[int] = None,
        target_len: Optional[int] = None,
        csi_vec: Optional[torch.Tensor] = None,
        snr_db: Optional[float] = None,
        content_hw: Optional[Tuple[int, int]] = None,
        gt_mel_mean: Optional[torch.Tensor] = None,
        gt_mel_std: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Offline decoder: only uses bitstreams (+ optional CSI/header), no GT leakage."""

        device = self.device

        # -------------------------
        # helpers: bits -> sign, and decode hash bottlenecks (Hash/Grouped/TwoStage)
        # -------------------------
        def _bits_to_sign(b: torch.Tensor) -> torch.Tensor:
            """Prepare external bits for hash decoding.

            支持三种常见编码：
            - {0,1}  : 视作硬判决概率，映射到 {-1,+1}
            - {-1,+1}: 视作理想 BPSK bit，不做变换
            - 软值   : 视作 BPSK+AWGN（±1+噪声），仅做轻微裁剪
            """

            b = torch.as_tensor(b, device=device)
            if b.dtype == torch.bool:
                b = b.to(torch.float32)
            else:
                b = b.to(torch.float32)

            if b.numel() == 0:
                return b

            bmin = float(b.min().item())
            bmax = float(b.max().item())

            # stored as 0/1 -> convert to -1/+1
            if bmin >= 0.0 and bmax <= 1.0:
                b = b * 2.0 - 1.0

            # 保留软值幅度信息，仅防御性裁剪极端 outlier
            b = torch.clamp(b, -3.0, 3.0)
            return b

        def _hb_decode(hb, bits: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            """Decode bits -> continuous latent for HashBottleneck / GroupedHashBottleneck /
            TwoStageHashBottleneck / RVQBottleneck.
            """
            if hb is None or bits is None:
                return None

            b = _bits_to_sign(bits)
            if b.dim() == 2:
                b = b.unsqueeze(1)

            # RVQ bottleneck：优先使用 decode_bits 接口
            try:
                if isinstance(hb, RVQBottleneck) or hasattr(hb, "decode_bits"):
                    if os.getenv("DBG_BITS_ONLY", "0") == "1":
                        print(f"[hb_decode] hb={type(hb).__name__} bits shape={tuple(b.shape)} "
                            f"min={float(b.min()):+.3f} max={float(b.max()):+.3f}")

                    x = hb.decode_bits(b)
                    if os.getenv("DBG_BITS_ONLY", "0") == "1":
                        xf = x.detach().flatten()
                        print(f"[hb_decode] -> latent shape={tuple(x.shape)} min={xf.min().item():+.3f} max={xf.max().item():+.3f} mean={xf.mean().item():+.3f} std={xf.std().item():+.3f}")
                    return x

            except Exception:
                pass

            # Prefer explicit decode helper if available (HashBottleneck-style)
            if hasattr(hb, "decode_hash_codec") and callable(getattr(hb, "decode_hash_codec")):
                return hb.decode_hash_codec(b)

            # Standard HashBottleneck/GroupedHashBottleneck usually has hash_decoder
            if hasattr(hb, "hash_decoder") and callable(getattr(hb, "hash_decoder")):
                return hb.hash_decoder(b)
            # GroupedHashBottleneck fallback: split bits by group, decode each group, then concat.
            # (Some implementations expose `groups` + `hash_bits_per_group` but no `hash_decoder`/`decode_hash_codec`.)
            try:
                if isinstance(hb, GroupedHashBottleneck) or (
                    hasattr(hb, "groups") and hasattr(hb, "hash_bits_per_group")
                ):
                    print(f"[DEBUG] HashBottleneck - 进入hash解码分支") 
                    groups = list(getattr(hb, "groups"))
                    bits_per = [int(x) for x in list(getattr(hb, "hash_bits_per_group"))]
                    if len(groups) != len(bits_per):
                        raise ValueError(
                            f"GroupedHashBottleneck mismatch: len(groups)={len(groups)} vs "
                            f"len(hash_bits_per_group)={len(bits_per)}"
                        )
                    if sum(bits_per) != int(b.size(-1)):
                        raise ValueError(
                            f"GroupedHashBottleneck bits last-dim={int(b.size(-1))} != sum(bits_per_group)={sum(bits_per)}"
                        )

                    xs = []
                    start = 0
                    for sub_hb, k_g in zip(groups, bits_per):
                        b_g = b[..., start : start + k_g]
                        start += k_g
                        x_g = _hb_decode(sub_hb, b_g)
                        if x_g is None:
                            out_dim = int(getattr(sub_hb, "output_dim", 0))
                            x_g = torch.zeros(
                                b.size(0), b.size(1), out_dim,
                                device=device, dtype=torch.float32
                            )
                        xs.append(x_g)
                    return torch.cat(xs, dim=-1)
            except Exception:
                # If grouped probing failed, continue to other fallbacks (TwoStage etc.)
                pass
            # Two-stage fallback: stage1 + stage2
            if hasattr(hb, "stage1") and hasattr(hb, "stage2"):
                stage_bits = getattr(hb, "stage_bits", None)
                if stage_bits is not None and len(stage_bits) >= 2:
                    k1, k2 = int(stage_bits[0]), int(stage_bits[1])
                else:
                    k1 = int(getattr(hb, "stage1_bits"))
                    k2 = int(getattr(hb, "stage2_bits"))

                if b.size(-1) < (k1 + k2):
                    raise ValueError(f"TwoStageHashBottleneck bits last-dim={b.size(-1)} < k1+k2={k1+k2}")

                b1 = b[..., :k1]
                b2 = b[..., k1:k1 + k2]

                x1 = _hb_decode(hb.stage1, b1)
                x2 = _hb_decode(hb.stage2, b2)
                if x1 is None:
                    return x2
                if x2 is None:
                    return x1
                return x1 + x2

            raise TypeError(f"Unsupported hash bottleneck type for decode: {type(hb)}")

        def _interp_time(x: torch.Tensor, T_out: int) -> torch.Tensor:
            """x: [B,T,C] -> [B,T_out,C]"""
            if x is None:
                return x
            if x.size(1) == T_out:
                return x
            x_ = x.permute(0, 2, 1)  # [B,C,T]
            x_ = torch.nn.functional.interpolate(x_, size=T_out, mode="linear", align_corners=False)
            return x_.permute(0, 2, 1).contiguous()

        # -------------------------
        # infer B, T
        # -------------------------
        if bits_f0 is not None:
            bits_f0 = torch.as_tensor(bits_f0, device=device)
            B = int(bits_f0.size(0))
            T = int(bits_f0.size(1))
        elif f0_T is not None:
            T = int(f0_T)
            B = int(bits_content.size(0)) if bits_content is not None else 1
        else:
            B = int(bits_content.size(0)) if bits_content is not None else 1
            if target_len is not None:
                T = int((int(target_len) + 159) // 160)
            else:
                T = 400  # fallback

        F_mel = 32  # your model uses 32-bin log-mel

        # -------------------------
        # CSI vector (receiver side info; not GT leakage)
        # -------------------------
        if csi_vec is None:
            if snr_db is None:
                csi_vec = torch.zeros(B, 4, device=device, dtype=torch.float32)
            else:
                csi_vec = torch.tensor([[float(snr_db), 0.0, 0.0, 1.0]] * B, device=device, dtype=torch.float32)
        else:
            csi_vec = torch.as_tensor(csi_vec, device=device, dtype=torch.float32)
            if csi_vec.dim() == 1:
                csi_vec = csi_vec.unsqueeze(0).expand(B, -1)

        # -------------------------
        # decode stats(mean/std) via hash_content_stats if provided
        # 可选调试：若提供 gt_mel_mean/gt_mel_std，则优先使用 GT 统计量，
        # 从而旁路 stats bits，便于对比 rumble 是否来自能量标尺错误。
        # -------------------------
        mel_mean_hat = torch.zeros(B, 1, 1, device=device)
        mel_std_hat = torch.ones(B, 1, 1, device=device)

        # DEBUG 路径：显式传入 GT 统计量时，优先使用 GT 覆盖 hash 重构结果。
        if gt_mel_mean is not None and gt_mel_std is not None:
            mm = torch.as_tensor(gt_mel_mean, device=device, dtype=torch.float32)
            ms = torch.as_tensor(gt_mel_std, device=device, dtype=torch.float32)
            if mm.dim() == 1:
                mm = mm.view(B, 1, 1)
            elif mm.dim() == 2:
                mm = mm.view(B, 1, 1)
            if ms.dim() == 1:
                ms = ms.view(B, 1, 1)
            elif ms.dim() == 2:
                ms = ms.view(B, 1, 1)
            mel_mean_hat = mm
            mel_std_hat = ms.clamp(min=0.1)

        elif bits_stats is not None and hasattr(self, "hash_content_stats") and self.hash_content_stats is not None:
            stats_hat = _hb_decode(self.hash_content_stats, bits_stats)  # expect [B,1,2] or [B,2] etc
            if stats_hat is not None:
                stats_hat = stats_hat.view(B, -1)
                if stats_hat.size(-1) >= 2:
                    # 与训练 forward 保持一致：stats_hat 表示归一化后的
                    # mean_norm/std_norm，需要先反归一化回物理 mean/std。
                    mean_hat_norm = stats_hat[:, 0:1].view(B, 1, 1)
                    std_hat_norm = stats_hat[:, 1:2].view(B, 1, 1)

                    mean_center = -5.0
                    mean_scale = 2.0
                    std_center = 0.8
                    std_scale = 0.8

                    mel_mean_hat = mean_hat_norm * mean_scale + mean_center
                    std_log_hat = std_hat_norm * std_scale + std_center
                    mel_std_hat = std_log_hat.exp().clamp(min=0.1)

                    # Debug: log stats bits reconstruction error in offline
                    # decode when DBG_STATS=1. 这里没有 GT mel_mean/mel_std
                    # (它们只在编码端可用)，因此仅展示 bits_stats 重构
                    # 后 mel_mean_hat/mel_std_hat 的分布，方便与前向路径
                    # 的统计进行对比。
                    if os.environ.get("DBG_STATS", "0") == "1":
                        try:
                            with torch.no_grad():
                                mmh = mel_mean_hat.view(B, -1)
                                msh = mel_std_hat.view(B, -1)

                                def _s(x: torch.Tensor) -> str:
                                    return (
                                        f"mean={float(x.mean().item()):+.4f} "
                                        f"std={float(x.std().item()):+.4f} "
                                        f"min={float(x.min().item()):+.4f} "
                                        f"max={float(x.max().item()):+.4f}"
                                    )

                                print(
                                    "[STATS][bits_decode] "
                                    f"mel_mean_hat={_s(mmh)} | mel_std_hat={_s(msh)}"
                                )
                        except Exception:
                            pass

        # -------------------------
        # decode content -> mel (denorm)
        # -------------------------
        mel_hat = torch.zeros(B, T, F_mel, device=device)

        if bits_content is not None:
            bits_content = torch.as_tensor(bits_content, device=device)
            s_hat_flat = _hb_decode(self.hash_content, bits_content)  # [B, L_seq, D_latent]
            if s_hat_flat is None:
                s_hat_flat = torch.zeros(B, 1, 1, device=device)

            Bc, L_seq, D_latent = s_hat_flat.shape

            # infer (Hc, Wc)
            if content_hw is not None:
                Hc, Wc = int(content_hw[0]), int(content_hw[1])
                if Hc * Wc != L_seq:
                    content_hw = None

            if content_hw is None:
                freq_ds = int(getattr(self, "freq_downsample_stages", 1))
                Wc_try = max(1, F_mel // (2 ** freq_ds))
                if (L_seq % Wc_try) == 0:
                    Wc = Wc_try
                    Hc = max(1, L_seq // Wc)
                else:
                    Wc = max(1, int(round(L_seq ** 0.5)))
                    Hc = max(1, L_seq // Wc)
            else:
                Hc, Wc = int(content_hw[0]), int(content_hw[1])
            if os.getenv("DBG_BITS_ONLY", "0") == "1":
                print(f"[content_reshape] L_seq={L_seq} D_latent={D_latent} "
                    f"content_hw_arg={content_hw} -> using Hc={Hc} Wc={Wc} (Hc*Wc={Hc*Wc})")

            # [B,L,D] -> [B,D,H,W]
            s_hat = s_hat_flat.view(Bc, Hc, Wc, D_latent).permute(0, 3, 1, 2).contiguous()

            # VMamba decode: prefer .decode (与 forward_with_hash 路径保持一致），
            # fallback 到 .decoder；若均不存在则使用零占位。
            mel_hat_img_norm = None
            if hasattr(self, "content_vmamba"):
                if hasattr(self.content_vmamba, "decode") and callable(getattr(self.content_vmamba, "decode")):
                    try:
                        mel_hat_img_norm = self.content_vmamba.decode(s_hat, csi_vec, (Hc, Wc))
                    except TypeError:
                        mel_hat_img_norm = self.content_vmamba.decode(s_hat, csi_vec)
                elif hasattr(self.content_vmamba, "decoder") and callable(getattr(self.content_vmamba, "decoder")):
                    try:
                        mel_hat_img_norm = self.content_vmamba.decoder(s_hat, csi_vec, (Hc, Wc))
                    except TypeError:
                        mel_hat_img_norm = self.content_vmamba.decoder(s_hat, csi_vec)

            if mel_hat_img_norm is None:
                mel_hat_img_norm = torch.zeros(B, 1, T, F_mel, device=device)

            mel_hat_img_norm = torch.nn.functional.interpolate(
                mel_hat_img_norm, size=(T, F_mel), mode="bilinear", align_corners=False
            )

            # decoder output may be multi-channel -> mean over channel
            if mel_hat_img_norm.size(1) == 1:
                mel_hat_norm = mel_hat_img_norm.squeeze(1)  # [B,T,F]
            else:
                mel_hat_norm = mel_hat_img_norm.mean(dim=1)  # [B,T,F]

        mel_hat = mel_hat_norm * mel_std_hat + mel_mean_hat  # denorm

        # Optional energy calibration in offline decode:
        # use only stats bits (mel_mean_hat) as the reference "brightness".
        # This mirrors forward_with_hash, but avoids any GT leakage.
        if getattr(self, "enable_energy_calib", False):
            with torch.no_grad():
                # mu_ref: global mean implied by stats bits
                mu_ref = mel_mean_hat.view(B, 1)
                # mu_hat: global mean of current reconstruction
                mu_hat = mel_hat.view(B, -1).mean(dim=1, keepdim=True)
                d_mu = (mu_hat - mu_ref).view(B, 1, 1)
            mel_hat = mel_hat - float(getattr(self, "energy_calib_alpha", 0.8)) * d_mu

        mel_hat = torch.clamp(mel_hat, -10.0, 2.0)
        mel_used = mel_hat

        # -------------------------
        # decode f0/vuv（mirror forward_with_hash hashed path）
        # -------------------------
        if bits_f0 is not None and hasattr(self, "hash_f0vuv") and self.hash_f0vuv is not None:
            z_fv_hat = _hb_decode(self.hash_f0vuv, bits_f0)  # [B,T,latent]
            if z_fv_hat is None:
                z_fv_hat = torch.zeros(B, T, 1, device=device)
            z_fv_hat = _interp_time(z_fv_hat, T)
            # 与 forward_with_hash 一致：使用 VUV 门控后的 F0 以及由 logits
            # 映射得到的 frame_corr_hat，避免在无声段产生不必要的基频。
            dnn_pitch_raw, vuv_logits = self.f0vuv_dec(
                z_fv_hat,
                mel_cond=mel_hat_norm,
            )  # [B,T,1]x2
            frame_corr_hat_raw, vuv_prob = self._vuv_logits_to_feat(vuv_logits)

            dp_calib = self.f0_calib_scale * dnn_pitch_raw + self.f0_calib_bias
            frame_corr_hat = self.fc_calib_scale * frame_corr_hat_raw + self.fc_calib_bias
            dnn_pitch_hat = vuv_prob * dp_calib
        else:
            dnn_pitch_hat = torch.zeros(B, T, 1, device=device)
            frame_corr_hat = torch.zeros(B, T, 1, device=device)
        if os.getenv("DBG_BITS_ONLY", "0") == "1":
            dp = dnn_pitch_hat.detach().flatten()
            fc = frame_corr_hat.detach().flatten()
            print(f"[f0] dnn_pitch_hat: min={dp.min().item():+.3f} max={dp.max().item():+.3f} mean={dp.mean().item():+.3f} std={dp.std().item():+.3f}")
            print(f"[f0] frame_corr_hat: min={fc.min().item():+.3f} max={fc.max().item():+.3f} mean={fc.mean().item():+.3f} std={fc.std().item():+.3f}")
            vuv_prob = torch.sigmoid(frame_corr_hat)
            print(f"[f0] vuv_prob: min={vuv_prob.min().item():.3f} max={vuv_prob.max().item():.3f} mean={vuv_prob.mean().item():.3f}")

        mel_hat_refined = None

        # -------------------------
        # optional L2H refine（对齐 forward_with_hash hashed 路径）
        # -------------------------
        enable_l2h = bool(getattr(self, "enable_l2h", False)) and hasattr(self, "deco_l2h_refiner")
        if enable_l2h and self.deco_l2h_refiner is not None:
            lb = int(getattr(self, "mel_hp_low_bins", getattr(self, "l2h_low_bins", 16)))
            lb = max(1, min(lb, F_mel - 1))
            mel_low = mel_hat[:, :, :lb]
            mel_high = mel_hat[:, :, lb:]

            # 与 forward_with_hash 一致：使用预测 dnn_pitch_hat 作为 F0 条件，
            # 使用 frame_corr_hat 作为相关性条件，并通过 sigmoid 得到 vuv_prob 门控。
            pitch_cond = torch.nan_to_num(dnn_pitch_hat, nan=0.0).clamp(-3.0, 3.0)
            _frame_corr_cond = torch.nan_to_num(frame_corr_hat, nan=0.0).clamp(-8.0, 8.0)

            # DeCo 风格 L2H：从低频 + F0/VUV 条件生成高频基准
            mel_high_ref = self.deco_l2h_refiner(mel_low, pitch_cond, _frame_corr_cond, mel_high)  # [B,T,high]
            if isinstance(mel_high_ref, tuple):
                mel_high_ref = mel_high_ref[0]

            import numpy as _np  # 局部导入，避免顶层污染

            blend = float(getattr(self, "l2h_blend", 1.0))
            blend = 0.0 if not _np.isfinite(blend) else max(0.0, min(1.0, blend))

            # VUV 概率用于在有声帧上更强地引入 L2H 高频，在无声帧上更保守
            vuv_prob = torch.sigmoid(_frame_corr_cond)
            mel_high_blend = mel_high + (blend * vuv_prob) * (mel_high_ref - mel_high)

            mel_used = torch.cat([mel_low, mel_high_blend], dim=-1)
            mel_hat_refined = mel_used

        # ---- mel -> ceps ----（mirror forward_with_hash；不再额外裁剪到 [-2,2]）
        E = torch.pow(10.0, mel_used)                 # [B,T,32] 线性能量
        E18 = self.band_agg_32_to_18(E)               # [B,T,18] 32->18 聚合（与训练一致）
        logE18 = torch.log10(torch.clamp(E18, min=1e-6))
        logE18 = opus_band_log_smooth(logE18)
        ceps_hat = self.mel18_to_ceps(logE18)

        if os.environ.get("DBG_CEPS_ENERGY", "0") == "1":
            try:
                # 在 bits-only/offline 解码路径中，同样打印 ceps/mel/e18 的统计，
                # 便于与 forward_with_hash 主干路径下的分布进行对比。
                _pstats("DBG(off) ceps_hat_c0", ceps_hat[..., 0:1])
                _pstats("DBG(off) mel_used", mel_used)
                _pstats("DBG(off) dnn_pitch_hat", dnn_pitch_hat)
                _pstats("DBG(off) frame_corr_hat", frame_corr_hat)
                _pstats("DBG(off) e18_energy", E18)
            except Exception as _dbg_e:
                print(f"[DBG_CEPS_ENERGY] offline decode debug print failed: {_dbg_e}")


        # -------------------------
        # optional HF sideband (match training)
        # -------------------------
        hf_sideband_feat_hash = None
        if bool(getattr(self, "with_hf_sideband", False)) and enable_l2h and (mel_hat_refined is not None):
            lb = int(getattr(self, "mel_hp_low_bins", getattr(self, "l2h_low_bins", 16)))
            lb = max(1, min(lb, F_mel - 1))

            mel_high_base = mel_hat[:, :, lb:]
            mel_high_ref  = mel_hat_refined[:, :, lb:]
            hf_resid = mel_high_ref - mel_high_base                      # [B,T,hf_bins]

            # ✅ 和训练一致：Linear/DCT 都输出 [B,T,dim]
            if self.hf_sideband_type == "dct":
                hf_feat = torch.matmul(hf_resid, self.hf_dct_proj)        # [B,T,dim]
            else:
                hf_feat = self.hf_sideband_encoder(hf_resid)              # [B,T,dim]

            # 对齐时间
            if hf_feat.size(1) > T:
                hf_feat = hf_feat[:, :T, :]
            elif hf_feat.size(1) < T:
                pad = torch.zeros(B, T - hf_feat.size(1), hf_feat.size(2), device=device, dtype=hf_feat.dtype)
                hf_feat = torch.cat([hf_feat, pad], dim=1)

            hf_sideband_feat_hash = hf_feat                               # [B,T,dim]

            # ✅ 同训练：加 hf2ceps_scale + 严格 slice_len
            if hasattr(self, "hf2ceps") and self.hf2ceps is not None:
                delta_hi = self.hf2ceps(hf_sideband_feat_hash)            # [B,T,hi]
                hi_start = int(getattr(self, "ceps_hi_start", 10))
                hi_start = max(0, min(hi_start, ceps_hat.size(-1)))
                max_len = max(0, ceps_hat.size(-1) - hi_start)
                slice_len = min(delta_hi.size(-1), max_len)
                if slice_len > 0:
                    ceps_hat = ceps_hat.clone()
                    ceps_hat[..., hi_start:hi_start+slice_len] = (
                        ceps_hat[..., hi_start:hi_start+slice_len]
                        + float(getattr(self, "hf2ceps_scale", 0.5)) * delta_hi[..., :slice_len]
                    )


        # -------------------------
        # build vocoder features (robust pad/truncate to total_dim)
        # -------------------------
        swap_raw = str(getattr(self, "oracle_swap_source_controls", "none") or "none").lower()
        swap_tokens = {tok.strip() for tok in swap_raw.replace("+", ",").split(",") if tok.strip()}
        swap_pitch = bool({"all", "source", "pitch", "period"} & swap_tokens)
        swap_fc = bool({"all", "source", "frame_corr", "vuv"} & swap_tokens)
        swap_gain = bool({"all", "source", "gain", "c0"} & swap_tokens)
        swap_ceps = bool({"all", "vocoder", "ceps"} & swap_tokens)

        ceps_vocoder = ceps_target if swap_ceps else ceps_hat
        if (not swap_ceps) and swap_gain:
            try:
                ceps_vocoder = ceps_hat.clone()
                ceps_vocoder[..., 0:1] = ceps_target[..., 0:1]
            except Exception:
                ceps_vocoder = ceps_hat
        dnn_pitch_vocoder = dnn_pitch if swap_pitch else dnn_pitch_hat
        frame_corr_vocoder = frame_corr if swap_fc else frame_corr_hat
        frame_corr_in = torch.clamp(frame_corr_vocoder, -0.8, 0.5)

        feat_parts = [ceps_vocoder, dnn_pitch_vocoder, frame_corr_in]
        if hf_sideband_feat_hash is not None:
            feat_parts.append(hf_sideband_feat_hash)

        fargan_feats_hat = torch.cat(feat_parts, dim=-1)

        total_dim = int(getattr(self.fargan_spec, "total_dim", fargan_feats_hat.size(-1)))
        cur_dim = int(fargan_feats_hat.size(-1))
        if cur_dim < total_dim:
            pad = torch.zeros(B, T, total_dim - cur_dim, device=device, dtype=fargan_feats_hat.dtype)
            fargan_feats_hat = torch.cat([fargan_feats_hat, pad], dim=-1)
        elif cur_dim > total_dim:
            fargan_feats_hat = fargan_feats_hat[..., :total_dim]

        # -------------------------
        # vocoder（对齐 forward_with_hash 推理分支）
        # 使用 decode 端预测的 dnn_pitch_hat / frame_corr_hat / mel_used
        # 构造 period_override 与 override_mask，仅在 "有声 ∧ 非静音" 段
        # 覆盖周期。整个过程仅依赖 bits 解码得到的特征，不引入编码端先验。
        # -------------------------
        if target_len is None:
            target_len = int(T * 160)

        with torch.no_grad():
            dp_src = dnn_pitch_vocoder
            period_override = 256.0 / torch.pow(2.0, dp_src + 1.5)  # [B,T,1]
            period_override = torch.clamp(period_override.squeeze(-1), 32.0, 255.0)  # [B,T]

            try:
                VUV_THR = 0.25
                fc_for_mask = frame_corr_vocoder
                mel_for_energy = mel_used

                vmask = (fc_for_mask > VUV_THR).squeeze(-1)  # [B,T]
                try:
                    mel_energy = mel_for_energy.mean(dim=-1)  # [B,T]
                    sil_thr_db = float(getattr(self, "silence_energy_thr_db", -40.0))
                    sil_thr_log = sil_thr_db / 10.0
                    vo_mask = mel_energy > sil_thr_log
                    vmask = vmask & vo_mask
                except Exception:
                    pass
            except Exception:
                vmask = None

        # 对齐 period_override / vmask 的时间长度到声码器输入
        T_vocoder = fargan_feats_hat.size(1)
        if period_override.size(1) != T_vocoder:
            if period_override.size(1) > T_vocoder:
                period_override = period_override[:, :T_vocoder]
            else:
                pad = period_override[:, -1:].expand(-1, T_vocoder - period_override.size(1))
                period_override = torch.cat([period_override, pad], dim=1)
        if vmask is not None and vmask.size(1) != T_vocoder:
            if vmask.size(1) > T_vocoder:
                vmask = vmask[:, :T_vocoder]
            else:
                pad = vmask.new_zeros(vmask.size(0), T_vocoder - vmask.size(1))
                vmask = torch.cat([vmask, pad], dim=1)

        if vmask is not None:
            _p, audio_hat = self.vocoder(
                fargan_feats_hat,
                target_len=target_len,
                period_override=period_override,
                override_mask=vmask,
            )
        else:
            _p, audio_hat = self.vocoder(
                fargan_feats_hat,
                target_len=target_len,
                period_override=period_override,
            )

        audio_hat = audio_hat.squeeze(1)
        voc_int = getattr(self.vocoder, "last_internal_tracks", None)

        out_decode = {
            "audio_hat": audio_hat,
            "mel_hat": mel_hat,
            "mel_hat_refined": mel_hat_refined,
            "ceps_hat": ceps_hat,
            "ceps_vocoder": ceps_vocoder,
            "dnn_pitch_hat": dnn_pitch_hat,
            "dnn_pitch_vocoder": dnn_pitch_vocoder,
            "frame_corr_hat": frame_corr_hat,
            "frame_corr_vocoder": frame_corr_vocoder,
            "period_vocoder": _p,
            "period_override_vocoder": period_override,
            # 调试字段：在离线解码/探针脚本中观察 mel 归一化统计量的重构情况。
            # 在线训练/常规推理路径不会使用这些键。
            "mel_mean_hat": mel_mean_hat,
            "mel_std_hat": mel_std_hat,
        }
        if isinstance(voc_int, dict):
            for _k_src, _k_dst in (
                ("pitch_gain_mean", "vocoder_pitch_gain_mean"),
                ("fwc0_rms", "vocoder_fwc0_rms"),
                ("skip_rms", "vocoder_skip_rms"),
                ("sig_core_rms", "vocoder_sig_core_rms"),
                ("sig_out_rms", "vocoder_sig_out_rms"),
            ):
                _v = voc_int.get(_k_src, None)
                if isinstance(_v, torch.Tensor):
                    out_decode[_k_dst] = _v
        return out_decode


        # # decoder-only：利用 meta 中的 T/F_mel/H/W/csi_vec 直接还原 mel_hat 与 F0 latent
        # T = int(meta_T)
        # F_mel = int(meta_F_mel)
        # H = int(meta_H)
        # W = int(meta_W)
        # csi_vec = torch.from_numpy(meta_csi).to(device)

        # # 内容分支：bits_content -> s_hat_flat -> VMamba decoder -> mel_hat
        # if bits_content is not None:
        #     bits_c = bits_content.to(device)
        #     if isinstance(self.hash_content, GroupedHashBottleneck):
        #         # 分组 hash: 按 group bit 数切分后逐 decoder 重建
        #         bits_groups = []
        #         start = 0
        #         for k_g in self.hash_content.hash_bits_per_group:
        #             bits_groups.append(bits_c[..., start : start + k_g])
        #             start += k_g
        #         recon_groups = []
        #         for bits_g, hb in zip(bits_groups, self.hash_content.groups):
        #             recon_g = hb.hash_decoder(bits_g)
        #             recon_groups.append(recon_g)
        #         s_hat_flat = torch.cat(recon_groups, dim=-1)  # [B,L_seq,C]
        #     else:
        #         s_hat_flat = self.hash_content.hash_decoder(bits_c)       # [B,L_seq,C]

        #     Bc, L_seq, Cc = s_hat_flat.shape
        #     s_hat = s_hat_flat.view(Bc, H, W, Cc).permute(0, 3, 1, 2).contiguous()
        #     mel_hat_img = self.content_vmamba.decoder(s_hat, csi_vec)  # [B,1,T',F']
        #     mel_hat_img = F.interpolate(mel_hat_img, size=(T, F_mel), mode="bilinear", align_corners=False)
        #     mel_hat = mel_hat_img.squeeze(1)
        # else:
        #     # 若未提供内容 bits，则退化为基于输入 audio 的 mel（无内容 JSCC）
        #     mel_hat = self.wave_to_mel(audio)[:, :T, :]

        # # F0 分支：bits_f0 -> z_fv_hat -> dnn_pitch_hat/frame_corr_hat
        # if bits_f0 is not None:
        #     bits_f = bits_f0.to(device)
        #     z_fv_hat = self.hash_f0vuv.hash_decoder(bits_f)   # [B,T,d_zf]
        #     dnn_pitch_hat, frame_corr_hat = self.f0vuv_dec(z_fv_hat)
        # else:
        #     # 若未提供 F0 bits，则直接从输入特征提取 F0/VUV 作为退化路径
        #     dnn_pitch_hat = self.fargan_spec.extract_feature(fargan_feats[:, :T, :], "dnn_pitch")
        #     frame_corr_hat = self.fargan_spec.extract_feature(fargan_feats[:, :T, :], "frame_corr")

        # # 后半部分：mel_hat + (dnn_pitch_hat, frame_corr_hat) -> ceps_hat -> vocoder
        # # 这里重用 forward_with_hash 中的逻辑以保持训练/推理一致

        # mel_used = mel_hat
        # if self.enable_energy_calib and self.training:
        #     with torch.no_grad():
        #         mel_ref = self.wave_to_mel(audio)[:, :T, :]
        #         d_mu = (mel_used.mean(dim=(1, 2)) - mel_ref.mean(dim=(1, 2))).view(B, 1, 1)
        #     mel_used = mel_used - float(self.energy_calib_alpha) * d_mu
        # mel_used = torch.clamp(mel_used, -10.0, 2.0)

        # ceps_target = self.fargan_spec.extract_feature(fargan_feats[:, :T, :], "ceps")
        # E = torch.pow(10.0, torch.clamp(mel_used, min=-10.0, max=10.0))
        # e18_energy = self.band_agg_32_to_18(E)
        # e18_log = torch.log10(e18_energy + 1e-10)
        # e18_log = opus_band_log_smooth(e18_log)
        # ceps_hat = self.mel18_to_ceps(e18_log)
        # ceps_target = torch.nan_to_num(ceps_target, nan=0.0)
        # ceps_hat = torch.nan_to_num(ceps_hat, nan=0.0)

        # # 拼回 FARGAN 特征
        # try:
        #     fc_hat_in = torch.clamp(frame_corr_hat, -0.8, 0.5)
        #     frame_corr_in = fc_hat_in
        # except Exception:
        #     frame_corr_in = frame_corr_hat

        # feat_extra = None
        # try:
        #     extra_dim = max(0, fargan_feats.size(-1) - 20)
        #     if extra_dim > 0:
        #         feat_extra = torch.zeros(B, T, extra_dim, device=ceps_hat.device, dtype=ceps_hat.dtype)
        # except Exception:
        #     feat_extra = None

        # if feat_extra is None:
        #     fargan_feats_hat = torch.cat([
        #         ceps_hat,
        #         dnn_pitch_hat,
        #         frame_corr_in,
        #     ], dim=-1)
        # else:
        #     fargan_feats_hat = torch.cat([
        #         ceps_hat,
        #         dnn_pitch_hat,
        #         frame_corr_in,
        #         feat_extra,
        #     ], dim=-1)

        # if target_len is None:
        #     target_len = audio.size(-1)
        # with torch.no_grad():
        #     dp_src = dnn_pitch_hat
        #     period_override = 256.0 / torch.pow(2.0, dp_src + 1.5)
        #     period_override = torch.clamp(period_override.squeeze(-1), 32.0, 255.0)

        #     mel_energy = mel_used.mean(dim=-1)
        #     sil_thr_db = float(getattr(self, "silence_energy_thr_db", -40.0))
        #     sil_thr_log = sil_thr_db / 10.0
        #     vo_mask = (mel_energy > sil_thr_log)
        #     vuv_thr = float(getattr(self, "vuv_threshold", 0.3))
        #     vmask = (frame_corr_hat > vuv_thr).squeeze(-1) & vo_mask

        # if vmask is not None:
        #     _period_hat, audio_hat = self.vocoder(
        #         fargan_feats_hat, target_len=target_len,
        #         period_override=period_override, override_mask=vmask,
        #     )
        # else:
        #     _period_hat, audio_hat = self.vocoder(
        #         fargan_feats_hat, target_len=target_len, period_override=period_override,
        #     )
        # audio_hat = audio_hat.squeeze(1)

        # return {
        #     "audio_hat": audio_hat,
        #     "fargan_feats_hat": fargan_feats_hat,
        #     "period_vocoder": _period_hat,
        #     "ceps": ceps_target,
        #     "ceps_hat": ceps_hat,
        #     "mel": mel_used,
        #     "mel_hat": mel_hat,
        #     "mel_hat_refined": mel_used,
        #     "dnn_pitch": dnn_pitch_hat,
        #     "dnn_pitch_hat": dnn_pitch_hat,
        #     "frame_corr": frame_corr_hat,
        #     "frame_corr_hat": frame_corr_hat,
        # }

    def forward(
        self,
        audio: torch.Tensor,
        fargan_feats: torch.Tensor,
        channel_sim,
        snr_min_db: float,
        snr_max_db: float,
        target_len: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Stage2.5 前向：Mel+vSSM 内容分支 + F0/voicing JSCC。

        Args:
            audio: [B,L] 原始音频
            fargan_feats: [B,T_feat,36] FARGAN 特征（用于 F0 分支与目标 ceps）
            channel_sim: ChannelSimulator
        """

        device = self.device
        audio = audio.to(device)
        fargan_feats = fargan_feats.to(device)

        B, L = audio.shape
        B2, T_feat, _ = fargan_feats.shape
        assert B == B2

        # === 内容分支：wave -> mel 图像 ===
        mel = self.wave_to_mel(audio)                 # [B,T_mel,32]
        pstats("mel", mel)
        Bm, T_mel, F_mel = mel.shape

        # 对齐时间长度（mel 与 FARGAN 特征帧数可能略有不同）
        T = min(T_mel, T_feat)
        mel = mel[:, :T, :]
        fargan_feats = fargan_feats[:, :T, :]

        # 显式归一化：记录 mel 的 mean/std，编码前归一化，解码后反归一化
        mel_mean = mel.mean(dim=(1, 2), keepdim=True)  # [B,1,1]
        mel_std = mel.std(dim=(1, 2), keepdim=True).clamp(min=0.1)  # [B,1,1]
        mel_norm = (mel - mel_mean) / mel_std  # [B,T,32]

        mel_img = mel_norm.unsqueeze(1)               # [B,1,T,F_mel] 使用归一化后的mel
        mel_img_c = self._content_pre_downsample(mel_img)

        # === 信道采样（全局 CSI + 每帧/每符号噪声一次性采样） ===
        # 说明：为避免“编码 FiLM 的 CSI”与“实际施加在 token 上的噪声"来自两次独立采样
        # 而产生错配，这里对时间维度 T 一次性调用 sample_csi，得到：
        #   - csi_dict_enc: 聚合后的全局 CSI（供 FiLM 使用）
        #   - amp_t:        [B,T] 时间相关衰落包络（供 F0 分支直接使用）
        #   - snr_db_t:     [B,T] 每帧 SNR（供 F0 分支使用）
        csi_dict_enc, amp_t_frame, snr_db_t_frame = channel_sim.sample_csi(
            B, T, channel="fading", snr_min_db=snr_min_db, snr_max_db=snr_max_db
        )
        csi_vec = torch.stack(
            [
                csi_dict_enc["snr_proxy"],
                csi_dict_enc["time_selectivity"],
                csi_dict_enc["freq_selectivity"],
                csi_dict_enc["los_ratio"],
            ],
            dim=-1,
        )


        # === 内容分支 VMamba JSCC ===
        tokens, s_tokens, hw = self.content_vmamba.encode(mel_img_c, csi_vec)  # tokens不再使用；s_tokens: [B,C,H,W]
        pstats("s_tokens", s_tokens)

        # 为符号序列长度 L 生成与全局 CSI 一致的逐 token 衰落与 SNR
        # 做法：对 amp_t_frame/snr_db_t_frame 在时间维上做线性插值/拉伸到 L_seq，
        # 从而保证“用于 FiLM 的 CSI”与“token 上的噪声统计”来自同一次采样。
        B, C, H, W = s_tokens.shape
        L_seq = H * W
        amp_t_frame = amp_t_frame.to(device=s_tokens.device, dtype=s_tokens.dtype)
        snr_db_t_frame = snr_db_t_frame.to(device=s_tokens.device, dtype=s_tokens.dtype)

        if L_seq == T:
            amp_t = amp_t_frame
            snr_db_t = snr_db_t_frame
        else:
            # [B,T] -> [B,1,T] -> interpolate -> [B,L_seq]
            amp_t = F.interpolate(
                amp_t_frame.unsqueeze(1), size=L_seq, mode="linear", align_corners=False
            ).squeeze(1)
            snr_db_t = F.interpolate(
                snr_db_t_frame.unsqueeze(1), size=L_seq, mode="linear", align_corners=False
            ).squeeze(1)

        # 将4D符号张量reshape为3D以适配channel simulation
        s_tokens_flat = s_tokens.permute(0, 2, 3, 1).contiguous().view(B, L_seq, C)  # [B,H*W,C]
        s_tokens_noisy_flat = channel_sim.apply(s_tokens_flat, amp_t, snr_db_t)  # [B,L,C]
        if self.eq_fading:
            s_tokens_noisy_flat = s_tokens_noisy_flat / (amp_t.unsqueeze(-1) + 1e-3)

        # 将3D结果reshape回4D
        s_tokens_noisy = s_tokens_noisy_flat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # [B,C,H,W]
        pstats("s_tokens_noisy", s_tokens_noisy)
        mel_hat_img_norm = self.content_vmamba.decode(s_tokens_noisy, csi_vec, hw)  # [B,1,T',F'] 归一化空间
        pstats("mel_hat_img_raw", mel_hat_img_norm)

        # 对齐到原始 mel 尺寸 (T, F_mel)
        mel_hat_img_norm = F.interpolate(mel_hat_img_norm, size=(T, F_mel), mode='bilinear', align_corners=False)
        mel_hat_norm = mel_hat_img_norm.squeeze(1)  # [B,T,F_mel] 归一化空间

        # 反归一化：恢复原始能量水平
        mel_hat = mel_hat_norm * mel_std + mel_mean  # [B,T,F_mel]
        pstats("mel_hat", mel_hat)
        mel_hat = torch.nan_to_num(mel_hat, nan=0.0)  # 清理NaN

        # === F0/voicing 分支 JSCC（沿用 FARGAN 特征） ===
        dnn_pitch = self.fargan_spec.extract_feature(fargan_feats, "dnn_pitch")  # [B,T,1]
        frame_corr = self.fargan_spec.extract_feature(fargan_feats, "frame_corr")# [B,T,1]
        # 数值安全：F0/相关系数偶尔会含有 NaN/Inf（数据缺失或对齐边界），
        # 在进入 RNN 前做去 NaN 与合理裁剪，避免传播到 z_fv。
        dnn_pitch = torch.nan_to_num(dnn_pitch, nan=0.0, posinf=3.0, neginf=-3.0).clamp(-3.0, 3.0)
        frame_corr = torch.nan_to_num(frame_corr, nan=0.0, posinf=0.5, neginf=-0.8).clamp(-0.8, 0.5)
        pstats("dnn_pitch(target)", dnn_pitch); pstats("frame_corr(target)", frame_corr)
        f0vuv = torch.cat([dnn_pitch, frame_corr], dim=-1)                             # [B,T,2]

        z_fv = self.f0vuv_enc(f0vuv)                                                 # [B,T,d_zf]
        assert_finite("z_fv", z_fv)
        s_fv = self.f0vuv_jscc_enc(z_fv, csi_vec)                                    # [B,T,d_s_f0]
        assert_finite("s_fv", s_fv)
        # F0 分支直接复用与内容分支相同的帧级衰落/SNR，确保两条 JSCC 支路看到一致的
        # 渠道统计（只在 token 变长时做插值，F0 本身长度就是 T）。
        amp_t_f = amp_t_frame.to(device=s_fv.device, dtype=s_fv.dtype)
        snr_db_t_f = snr_db_t_frame.to(device=s_fv.device, dtype=s_fv.dtype)
        s_fv_noisy = channel_sim.apply(s_fv, amp_t_f, snr_db_t_f)                     # [B,T,d_s_f0]
        assert_finite("s_fv_noisy", s_fv_noisy)
        # full 路径：使用全部 d_s_f0 维度解码
        z_fv_hat = self.f0vuv_jscc_dec(s_fv_noisy, csi_vec, h_rayleigh=None)         # [B,T,d_zf]
        assert_finite("z_fv_hat", z_fv_hat)
        # 使用归一化后的 mel 作为 F0 decoder 的内容条件，提高 VUV/F0 的鲁棒性。
        dnn_pitch_raw, vuv_logits = self.f0vuv_dec(z_fv_hat, mel_cond=mel_hat_norm)  # [B,T,1]x2 (logits for VUV)
        assert_finite("dnn_pitch_raw", dnn_pitch_raw)
        assert_finite("vuv_logits", vuv_logits)
        frame_corr_hat_raw, vuv_prob_hat = self._vuv_logits_to_feat(vuv_logits)     # corr feature + prob

        # 标定：对 decoder 输出的 F0 / frame_corr 做仿射变换，以对齐
        # JSCC F0 分支与原始 FARGAN 特征空间的尺度。
        dp_calib = self.f0_calib_scale * dnn_pitch_raw + self.f0_calib_bias
        frame_corr_hat = self.fc_calib_scale * frame_corr_hat_raw + self.fc_calib_bias

        # 在结构上强制 F0 由 VUV 门控，使无声段的 F0 自然压到 0 附近。
        dnn_pitch_hat = vuv_prob_hat * dp_calib
        pstats("dnn_pitch_hat", dnn_pitch_hat); pstats("frame_corr_hat", frame_corr_hat)

        # 可选：在 DBG_F0_STATS=1 时打印 GT 与标定后 F0 的分布，便于观察
        # JSCC F0 分支与原始 FARGAN 特征空间之间的偏移情况。
        if os.environ.get("DBG_F0_STATS", "0") == "1":
            try:
                with torch.no_grad():
                    dp_gt = dnn_pitch.detach().to(torch.float32).view(-1)
                    dp_cb = dp_calib.detach().to(torch.float32).view(-1)

                    def _s(x: torch.Tensor) -> str:
                        return (
                            f"mean={float(x.mean().item()):+.4f} "
                            f"std={float(x.std().item()):+.4f} "
                            f"min={float(x.min().item()):+.4f} "
                            f"max={float(x.max().item()):+.4f}"
                        )

                    print(f"[F0_STATS] gt   dp: {_s(dp_gt)}")
                    print(f"[F0_STATS] calibdp: {_s(dp_cb)}")
            except Exception:
                pass

        # base 路径（Successive Refinement）：仅使用前 k 维符号，
        # 将其余维度置零，以在固定符号预算下学习“骨架先、细节后”的鲁棒表示。
        dnn_pitch_hat_base: Optional[torch.Tensor] = None
        frame_corr_hat_base: Optional[torch.Tensor] = None
        k_sr = int(getattr(self, "f0_sr_k", 0))
        if k_sr > 0 and k_sr < s_fv_noisy.size(-1):
            try:
                s_base = s_fv_noisy.clone()
                s_base[..., k_sr:] = 0.0
                z_fv_hat_base = self.f0vuv_jscc_dec(s_base, csi_vec, h_rayleigh=None)  # [B,T,d_zf]
                dnn_pitch_raw_base, vuv_logits_base = self.f0vuv_dec(
                    z_fv_hat_base, mel_cond=mel_hat_norm
                )  # [B,T,1]x2
                frame_corr_hat_base, vuv_prob_base = self._vuv_logits_to_feat(vuv_logits_base)
                dnn_pitch_hat_base = vuv_prob_base * dnn_pitch_raw_base
                pstats("dnn_pitch_hat_base", dnn_pitch_hat_base)
                pstats("frame_corr_hat_base", frame_corr_hat_base)
            except Exception:
                dnn_pitch_hat_base = None
                frame_corr_hat_base = None

        # === 低→高 Mel 细化（可选）+ log-mel -> ceps ===
        mel_used = mel_hat
        mel_used = torch.clamp(mel_used, -10.0, 2.0)

        mel_hat_refined = None
        ceps_delta_high: Optional[torch.Tensor] = None
        if hasattr(self, "enable_l2h") and getattr(self, "enable_l2h") and self.deco_l2h_refiner is not None:
            lb = int(getattr(self, "l2h_low_bins", 10))
            lb = max(1, min(lb, F_mel - 1))
            mel_low = mel_hat[:, :, :lb]
            mel_high = mel_hat[:, :, lb:]

            # 统一使用预测 F0/VUV 作为 L2H 条件，避免训练/推理分布漂移。
            _dnn_pitch_cond = torch.nan_to_num(dnn_pitch_hat, nan=0.0).clamp(-3.0, 3.0)
            # DeCoL2HRefiner 期望 VUV logits 作为条件，这里使用 vuv_logits（裁剪到安全范围）。
            _frame_corr_cond = torch.nan_to_num(vuv_logits, nan=0.0).clamp(-8.0, 8.0)

            out_h = self.deco_l2h_refiner(
                mel_low,
                _dnn_pitch_cond,
                _frame_corr_cond,
                mel_high,
            )

            if isinstance(out_h, tuple):
                mel_high_ref, _dc_high = out_h
            else:
                mel_high_ref, _dc_high = out_h, None

            # 渐进式融合：仅在 vuv_prob 高的帧上引入 L2H 高频，
            # 减少在无声段/擦音上过度重绘高频结构的风险。
            blend = float(getattr(self, "l2h_blend", 1.0))
            blend = 0.0 if not np.isfinite(blend) else max(0.0, min(1.0, blend))
            vuv_prob = torch.sigmoid(_frame_corr_cond)  # [B,T,1]
            mel_high_mix = mel_high + (blend * vuv_prob) * (mel_high_ref - mel_high)
            mel_used = torch.cat([mel_low, mel_high_mix], dim=-1)
            mel_hat_refined = mel_used
            pstats("mel_hat_refined", mel_used)

        # 先将 mel_used（Bark对数能量）转回能量域，聚合到 18 带，再做（可学习）DCT 映射
        ceps_target = self.fargan_spec.extract_feature(fargan_feats, "ceps")
        ceps_target = torch.nan_to_num(ceps_target, nan=0.0)
        # 按能量域聚合（避免在log域加权导致对比度丢失）
        E = torch.pow(10.0, torch.clamp(mel_used, min=-10.0, max=10.0))  # [B,T,32]
        e18_energy = self.band_agg_32_to_18(E)                           # [B,T,18]
        e18_log = torch.log10(e18_energy + 1e-10)
        # Opus风格 log-域逐带平滑/跟随钳位，减少 BFCC→ceps 映射误差
        e18_log = opus_band_log_smooth(e18_log)
        ceps_hat = torch.nan_to_num(self.mel18_to_ceps(e18_log), nan=0.0)
        # 注入 Δceps 高阶（c12..c17），仅在亮区∧（更偏向无声段）门控时生效
        if ceps_delta_high is not None:
            try:
                # 亮区：基于目标 mel 的高频帧内0.8分位
                mel_high_t = mel[:, :, lb:]
                tau = torch.quantile(mel_high_t, 0.8, dim=2, keepdim=True)
                bright = (mel_high_t >= tau).float().mean(dim=2, keepdim=False)  # [B,T]
                # 使用预测的 VUV 概率作为门控
                vuv_prob = vuv_prob_hat.squeeze(-1)                               # [B,T]

                # gate 设计：在“亮且无声”帧上更强地注入 Δceps，
                # 以强化擦音/爆破等无声辅音的高频线索，避免
                # 在有声段叠加过强高阶倒谱而拉出过度谐波纹理。
                gate = (bright * (1.0 - vuv_prob)).unsqueeze(-1)                  # [B,T,1]
                # 注入
                idx0, idx1 = 12, 18
                ceps_hat[..., idx0:idx1] = ceps_hat[..., idx0:idx1] + gate * ceps_delta_high
            except Exception:
                pass
        pstats("ceps_target", ceps_target)
        pstats("ceps_hat", ceps_hat)

        # === 拼回 FARGAN 特征 (前20维) ===
        # 避免 in-place 操作，使用 torch.cat 重新构造。
        # 训练/推理统一：始终使用预测的 frame_corr_hat 作为声码器 gate，
        # 避免在训练阶段依赖 GT frame_corr 进行 teacher-forcing 混合。
        try:
            frame_corr_in = torch.clamp(frame_corr_hat, -0.8, 0.5)
        except Exception:
            frame_corr_in = torch.clamp(frame_corr_hat, -0.8, 0.5)

        # 训练/推理统一：不再使用 FARGAN 后 16 维 GT 特征作为声码器条件，
        # 而是始终使用零占位，避免训练路径与部署路径出现条件不一致。
        try:
            extra_dim = max(0, fargan_feats.size(-1) - 20)
            if extra_dim > 0:
                feat_extra = torch.zeros(B, T, extra_dim, device=ceps_hat.device, dtype=ceps_hat.dtype)
            else:
                feat_extra = None
        except Exception:
            feat_extra = None

        # HF 侧通道：从高频 mel 残差提取紧凑表示，绕过 32->18 聚合的信息损失
        hf_sideband_feat: Optional[torch.Tensor] = None
        if getattr(self, 'with_hf_sideband', False) and mel_hat_refined is not None:
            try:
                lb = int(getattr(self, 'l2h_low_bins', 10))
                # 计算高频残差：mel_refined - mel_baseline (或直接使用 mel_high)
                mel_high_refined = mel_hat_refined[:, :, lb:]  # [B,T,hf_bins]
                mel_high_baseline = mel_hat[:, :, lb:]         # [B,T,hf_bins]
                hf_residual = mel_high_refined - mel_high_baseline  # [B,T,hf_bins]

                # 编码为紧凑表示
                if self.hf_sideband_type == "dct":
                    # 固定 DCT 投影
                    hf_sideband_feat = torch.matmul(hf_residual, self.hf_dct_proj)
                elif self.hf_sideband_encoder is not None:
                    # 可学习编码器
                    hf_sideband_feat = self.hf_sideband_encoder(hf_residual)

                if hf_sideband_feat is not None:
                    # 对齐时间维度
                    T_ceps = ceps_hat.size(1)
                    T_hf = hf_sideband_feat.size(1)
                    if T_hf > T_ceps:
                        hf_sideband_feat = hf_sideband_feat[:, :T_ceps, :]
                    elif T_hf < T_ceps:
                        # 用零填充
                        pad = torch.zeros(B, T_ceps - T_hf, self.hf_sideband_dim,
                                          device=hf_sideband_feat.device, dtype=hf_sideband_feat.dtype)
                        hf_sideband_feat = torch.cat([hf_sideband_feat, pad], dim=1)
                    pstats("hf_sideband_feat", hf_sideband_feat)

                    # 使用 HF 侧通道特征对倒谱高阶分量做轻量校正
                    if self.hf2ceps is not None and self.hf2ceps_dim > 0:
                        try:
                            hi_start = int(getattr(self, 'ceps_hi_start', 10))
                            hi_start = max(0, min(hi_start, ceps_hat.size(-1)))
                            max_len = max(0, ceps_hat.size(-1) - hi_start)
                            slice_len = min(self.hf2ceps_dim, max_len)
                            if slice_len > 0:
                                delta_hi = self.hf2ceps(hf_sideband_feat)  # [B,T,dim]
                                delta_hi = delta_hi[..., :slice_len]
                                ceps_hat[..., hi_start:hi_start + slice_len] = (
                                    ceps_hat[..., hi_start:hi_start + slice_len]
                                    + float(self.hf2ceps_scale) * delta_hi
                                )
                        except Exception:
                            pass
            except Exception:
                hf_sideband_feat = None

        # 拼接 FARGAN 特征
        feat_parts = [ceps_hat, dnn_pitch_hat, frame_corr_in]
        if hf_sideband_feat is not None:
            feat_parts.append(hf_sideband_feat)
        if feat_extra is not None:
            feat_parts.append(feat_extra)
        fargan_feats_hat = torch.cat(feat_parts, dim=-1)
        pstats("fargan_feats_hat", fargan_feats_hat)
        assert_finite("fargan_feats_hat", fargan_feats_hat)

        # 可选：对声码器输入做更细粒度的诊断，便于观察
        # ceps / F0 / VUV 各自的尺度与分布，判断是否存在“某一路通道被挤占”的情况。
        try:
            dbg_voc = os.environ.get("DBG_VOCODER_IO", "0") == "1" or os.environ.get("DBG_F0", "0") == "1"
        except Exception:
            dbg_voc = False
        if dbg_voc:
            with torch.no_grad():
                def _stat(name: str, t: torch.Tensor) -> None:
                    if not isinstance(t, torch.Tensor):
                        return
                    x = t.detach().to(torch.float32)
                    if x.numel() == 0:
                        return
                    # 使用 reshape 而不是 view，以避免在非连续张量（例如经过 permute/slice）上触发 RuntimeError。
                    x_flat = x.reshape(-1)
                    print(
                        f"[voc_in] {name}: min={x_flat.min().item():+.4f} max={x_flat.max().item():+.4f} "
                        f"mean={x_flat.mean().item():+.4f} std={x_flat.std().item():.4f}"
                    )

                _stat("ceps_target", ceps_target)
                _stat("ceps_hat", ceps_hat)
                _stat("dnn_pitch", dnn_pitch)
                _stat("dnn_pitch_hat", dnn_pitch_hat)
                _stat("frame_corr", frame_corr)
                _stat("frame_corr_hat", frame_corr_hat)
                _stat("fargan_feats_hat_all", fargan_feats_hat)

        # === Vocoder ===
        # 训练/推理统一：不再使用 F0 驱动的 period_override，
        # 而是让 FARGAN 内部网络根据输入特征自行决定周期结构。
        if target_len is None:
            target_len = audio.size(-1)
        with torch.no_grad():
            _period_hat, audio_hat = self.vocoder(
                fargan_feats_hat,
                target_len=target_len,
            )
        audio_hat = audio_hat.squeeze(1)

        # Optional: parallel BFCC-based vocoder for debugging. This does NOT
        # affect training loss; it only writes an extra key "audio_hat_bfcc"
        # into out_dict when enabled, so you can compare two vocoders side by
        # side in visualization.
        #
        # 为了兼容较旧的 checkpoint（可能没有 bfcc_vocoder / use_bfcc_vocoder_debug
        # 属性），这里通过 getattr 做防御性访问，确保在这些属性缺失或调试关掉时
        # 完全跳过 BFCC 路径，仅使用 FARGAN 声码器。
        audio_hat_bfcc: Optional[torch.Tensor] = None
        bfcc_vocoder = getattr(self, "bfcc_vocoder", None)
        use_bfcc_dbg = bool(getattr(self, "use_bfcc_vocoder_debug", False))
        if bfcc_vocoder is not None and use_bfcc_dbg:
            try:
                # BFCC vocoder consumes BFCC32 + dnn_pitch + frame_corr.
                # 使用 mel_used 作为 BFCC32，保持与内容分支一致的对齐。
                bfcc32 = mel_used.detach()
                dp_for_v = torch.nan_to_num(dp_src.detach().squeeze(-1), nan=0.0)
                fc_for_v = torch.nan_to_num(frame_corr_hat.detach().squeeze(-1), nan=0.0)
                _period_bfcc, audio_bfcc = bfcc_vocoder(
                    bfcc32, dp_for_v, fc_for_v, target_len=target_len
                )
                audio_hat_bfcc = audio_bfcc.squeeze(1)
            except Exception as _e:
                if os.environ.get("DBG_BFCC_VOCODER", "0") == "1":
                    print(f"[BFCCVocoder] debug synthesis failed: {_e}")

        # 声码器输出诊断（可选）：观察 audio_hat 的整体幅度与能量范围，
        # 帮助判断 F0 / ceps / VUV 输入是否被声码器合理利用。
        try:
            dbg_voc = os.environ.get("DBG_VOCODER_IO", "0") == "1" or os.environ.get("DBG_F0", "0") == "1"
        except Exception:
            dbg_voc = False
        if dbg_voc:
            with torch.no_grad():
                x = audio_hat.detach().to(torch.float32)
                if x.numel() > 0:
                    # 使用 reshape 以在非连续张量上安全地展平。
                    xf = x.reshape(-1)
                    print(
                        f"[voc_out] audio_hat: min={xf.min().item():+.4f} max={xf.max().item():+.4f} "
                        f"mean={xf.mean().item():+.4f} std={xf.std().item():.4f}"
                    )

        out_dict = {
            "audio_hat": audio_hat,
            "fargan_feats_hat": fargan_feats_hat,
            "period_vocoder": _period_hat,
            "ceps": ceps_target,
            "ceps_hat": ceps_hat,
            "mel": mel,
            "mel_hat": mel_hat,
            "mel_hat_refined": mel_hat_refined if mel_hat_refined is not None else mel_hat,
            "dnn_pitch": dnn_pitch,
            # 原始 decoder 输出的 F0（未经标定），便于调试对比
            "dnn_pitch_raw": dnn_pitch_raw,
            # 标定后的 F0，用于 loss 与 vocoder / period_override
            "dnn_pitch_calib": dp_calib,
            "dnn_pitch_hat": dnn_pitch_hat,
            "frame_corr": frame_corr,
            # 原始 frame_corr_hat（由 vuv_logits 派生），仅作调试使用
            "frame_corr_hat_raw": frame_corr_hat_raw,
            # 标定后的 frame_corr_hat，用于 VUV CE / gate / probes
            "frame_corr_hat": frame_corr_hat,
            # Semantic latents for loss (clean vs channel-affected)
            "z_content": s_tokens_flat.detach(),
            "z_content_hat": s_tokens_noisy_flat,
            "z_f0": z_fv.detach(),
            "z_f0_hat": z_fv_hat,
        }

        # 在无 Hash/RVQ 的 forward 路径中，同样暴露 L2H 中间量，
        # 以便简化训练脚本中的 L2H residual / decor loss 使用。
        if getattr(self, "enable_l2h", False) and getattr(self, "deco_l2h_refiner", None) is not None:
            out_dict["l2h_resid"] = getattr(self.deco_l2h_refiner, "last_resid", None)
            out_dict["l2h_vuv_prob"] = getattr(self.deco_l2h_refiner, "last_vuv_prob", None)
            out_dict["l2h_mask_harm"] = getattr(self.deco_l2h_refiner, "last_mask_harm", None)
        # SR base 输出（若可用）：用于在训练中鼓励前 k 维承载稳定骨架，
        # 以及在诊断/可选推理路径中观察 base 分支的退化行为。
        if dnn_pitch_hat_base is not None and frame_corr_hat_base is not None:
            out_dict["dnn_pitch_hat_base"] = dnn_pitch_hat_base
            out_dict["frame_corr_hat_base"] = frame_corr_hat_base
        # HF 侧通道特征（可选）
        if hf_sideband_feat is not None:
            out_dict["hf_sideband"] = hf_sideband_feat
        # BFCC vocoder debug output（仅在 use_bfcc_vocoder_debug 时存在）。
        if audio_hat_bfcc is not None:
            out_dict["audio_hat_bfcc"] = audio_hat_bfcc
        return out_dict

    def forward_content_only_no_hash(
        self,
        audio: torch.Tensor,
        fargan_feats: torch.Tensor,
        channel_sim,
        snr_min_db: float,
        snr_max_db: float,
        target_len: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Content-branch-only forward WITHOUT hash: wave -> Bark/BFCC -> VMamba -> bark_hat.

        This is the simplest content-only path for training the content branch
        (VMamba encoder/decoder) without the HashBottleneck. Skips:
        - F0/VUV branch
        - Hash bottleneck
        - ceps computation
        - L2H refinement
        - FARGAN vocoder

        Returns only mel-related outputs needed for content-only loss computation.
        """
        device = self.device
        audio = audio.to(device)
        fargan_feats = fargan_feats.to(device)

        B, L = audio.shape
        B2, T_feat, _ = fargan_feats.shape
        assert B == B2

        # === Content branch: wave -> BFCC (Bark log energy) ===
        mel = self.wave_to_mel(audio)  # [B,T_mel,32]
        pstats("mel", mel)
        Bm, T_mel, F_mel = mel.shape

        T = min(T_mel, T_feat)
        mel = mel[:, :T, :]  # [B,T,32]

        # 若启用 CNN baseline，则在 BFCC 域上用简单 CNN+JSCC 编解码；否则走 VMamba 路径。
        if getattr(self, "use_cnn_content", False):
            # CNN baseline 采用 [B,1,32,T] 格式
            bfcc_img = mel.transpose(1, 2).unsqueeze(1)  # [B,1,32,T]

            z = self.content_cnn_encoder(bfcc_img)      # [B,C_lat,H,W]
            Bz, Cz, Hz, Wz = z.shape

            # 通过 ChannelSimulator 在 latent 上施加 JSCC 噪声（flatten → apply → reshape）
            L_seq = Hz * Wz
            _csi_tmp, amp_t, snr_db_t = channel_sim.sample_csi(
                Bz, L_seq, channel="fading", snr_min_db=snr_min_db, snr_max_db=snr_max_db
            )
            amp_t = amp_t.to(device=z.device, dtype=z.dtype)
            snr_db_t = snr_db_t.to(device=z.device, dtype=z.dtype)

            z_flat = z.permute(0, 2, 3, 1).contiguous().view(Bz, L_seq, Cz)
            z_noisy_flat = channel_sim.apply(z_flat, amp_t, snr_db_t)
            if self.eq_fading:
                z_noisy_flat = z_noisy_flat / (amp_t.unsqueeze(-1) + 1e-3)

            z_noisy = z_noisy_flat.view(Bz, Hz, Wz, Cz).permute(0, 3, 1, 2).contiguous()
            bfcc_hat_img = self.content_cnn_decoder(z_noisy)   # [B,1,32,T_hat]
            bfcc_hat = bfcc_hat_img.squeeze(1).transpose(1, 2) # [B,T_hat,32]

            # 对齐时间长度
            T_use = min(mel.size(1), bfcc_hat.size(1))
            mel = mel[:, :T_use, :]
            mel_hat = bfcc_hat[:, :T_use, :]
            mel_hat = torch.nan_to_num(mel_hat, nan=0.0)

            return {
                "mel": mel,
                "mel_hat": mel_hat,
                "mel_hat_refined": mel_hat,
            }

        # === 默认 VMamba 内容分支路径 ===
        # 显式归一化：记录 mel 的 mean/std，编码前归一化，解码后反归一化
        # 这样解码器只需学习结构重建，绝对能量由统计量恢复
        mel_mean = mel.mean(dim=(1, 2), keepdim=True)  # [B,1,1]
        mel_std = mel.std(dim=(1, 2), keepdim=True).clamp(min=0.1)  # [B,1,1], 防止除零
        mel_norm = (mel - mel_mean) / mel_std  # [B,T,32] 归一化到 mean~0, std~1

        mel_img = mel_norm.unsqueeze(1)  # [B,1,T,32] 使用归一化后的mel
        mel_img_c = self._content_pre_downsample(mel_img)

        # Channel sampling: 一次性获得全局 CSI + 帧级衰落/SNR，
        # 后续通过插值扩展到 token 序列长度，避免与 FiLM 使用的 CSI 不一致。
        csi_dict_enc, amp_t_frame, snr_db_t_frame = channel_sim.sample_csi(
            B, T, channel="fading", snr_min_db=snr_min_db, snr_max_db=snr_max_db
        )
        csi_vec = torch.stack(
            [
                csi_dict_enc["snr_proxy"],
                csi_dict_enc["time_selectivity"],
                csi_dict_enc["freq_selectivity"],
                csi_dict_enc["los_ratio"],
            ],
            dim=-1,
        )


        # VMamba encode (编码归一化后的mel)。在 AMP 场景下显式禁用 autocast，
        # 防止长序列 SSM 在半精度下数值溢出导致 NaN/Inf。
        with amp_autocast(enabled=False):
            tokens, s_tokens, hw = self.content_vmamba.encode(mel_img_c.float(), csi_vec.float())
        pstats("s_tokens", s_tokens)

        # 由帧级 CSI 插值出逐 token 衰落与 SNR
        Bv, Cv, H, W = s_tokens.shape
        L_seq = H * W
        amp_t_frame = amp_t_frame.to(device=s_tokens.device, dtype=s_tokens.dtype)
        snr_db_t_frame = snr_db_t_frame.to(device=s_tokens.device, dtype=s_tokens.dtype)
        if L_seq == T:
            amp_t = amp_t_frame
            snr_db_t = snr_db_t_frame
        else:
            amp_t = F.interpolate(
                amp_t_frame.unsqueeze(1), size=L_seq, mode="linear", align_corners=False
            ).squeeze(1)
            snr_db_t = F.interpolate(
                snr_db_t_frame.unsqueeze(1), size=L_seq, mode="linear", align_corners=False
            ).squeeze(1)

        # Flatten 4D symbols to 3D for channel simulation
        s_tokens_flat = s_tokens.permute(0, 2, 3, 1).contiguous().view(Bv, L_seq, Cv)
        s_tokens_noisy_flat = channel_sim.apply(s_tokens_flat, amp_t, snr_db_t)
        if self.eq_fading:
            s_tokens_noisy_flat = s_tokens_noisy_flat / (amp_t.unsqueeze(-1) + 1e-3)

        # Reshape back to 4D
        s_tokens_noisy = s_tokens_noisy_flat.view(Bv, H, W, Cv).permute(0, 3, 1, 2).contiguous()
        pstats("s_tokens_noisy", s_tokens_noisy)

        # VMamba decode (输出是归一化空间的重建)，同样在 float32 中执行以增强稳定性。
        with amp_autocast(enabled=False):
            mel_hat_img_norm = self.content_vmamba.decode(s_tokens_noisy.float(), csi_vec.float(), hw)
        pstats("mel_hat_img_raw", mel_hat_img_norm)

        # Align to original mel size (T, F_mel)
        mel_hat_img_norm = F.interpolate(mel_hat_img_norm, size=(T, F_mel), mode="bilinear", align_corners=False)
        mel_hat_norm = mel_hat_img_norm.squeeze(1)  # [B,T,32] 归一化空间

        # 反归一化：恢复原始能量水平
        mel_hat = mel_hat_norm * mel_std + mel_mean  # [B,T,32]
        pstats("mel_hat", mel_hat)
        mel_hat = torch.nan_to_num(mel_hat, nan=0.0)

        return {
            "mel": mel,
            "mel_hat": mel_hat,
            "mel_hat_refined": mel_hat,  # No L2H refinement in content-only mode
            "tokens": s_tokens_flat.detach(),
            "tokens_hat": s_tokens_noisy_flat,
            "tokens_map": s_tokens.detach(),
            "tokens_hat_map": s_tokens_noisy.detach(),
            "z_content": s_tokens_flat.detach(),
            "z_content_hat": s_tokens_noisy_flat,
            "mel_mean": mel_mean.detach(),  # 保存统计量用于诊断
            "mel_std": mel_std.detach(),
        }

    def forward_content_only(
        self,
        audio: torch.Tensor,
        fargan_feats: torch.Tensor,
        channel_sim,
        snr_min_db: float,
        snr_max_db: float,
        target_len: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Content-branch-only forward WITH hash.

        路径：wave -> Bark/BFCC -> VMamba(encode) -> HashBottleneck -> VMamba(decode)
        -> bark_hat -> 18-band + DCT -> ceps_hat。

        与 ``forward_with_hash`` 相比，本函数仍然跳过 F0/VUV 分支和
        FARGAN 声码器，但会额外生成 ceps/ceps_hat，便于在 content-only
        模式下对倒谱域进行监控或增加 ceps 相关损失。

        返回键包括：
        - mel / mel_hat / mel_hat_refined
        - ceps, ceps_hat（从 GT/重建 mel 推导）
        - tokens / tokens_hat / tokens_map / tokens_hat_map
        - hash_logits / hash_bits_clean / hash_reg_terms 等
        """

        assert self.with_hash, "forward_content_only requires with_hash=True"

        device = self.device
        audio = audio.to(device)
        fargan_feats = fargan_feats.to(device)

        B, L = audio.shape
        B2, T_feat, _ = fargan_feats.shape
        assert B == B2

        # === Content branch: wave -> mel ===
        mel = self.wave_to_mel(audio)  # [B,T_mel,32]
        pstats("mel", mel)
        Bm, T_mel, F_mel = mel.shape

        T = min(T_mel, T_feat)
        mel = mel[:, :T, :]
        fargan_feats = fargan_feats[:, :T, :]

        # 显式归一化：记录 mel 的 mean/std，编码前归一化。解码端使用的 mean/std
        # 将通过 hash_content_stats 重构得到，避免直接跨编解码器传递浮点统计量。
        mel_mean = mel.mean(dim=(1, 2), keepdim=True)  # [B,1,1]
        mel_std = mel.std(dim=(1, 2), keepdim=True).clamp(min=0.1)  # [B,1,1]
        mel_norm = (mel - mel_mean) / mel_std  # [B,T,32]

        mel_img = mel_norm.unsqueeze(1)  # [B,1,T,32] 使用归一化后的mel
        mel_img_c = self._content_pre_downsample(mel_img)

        # === Channel sampling (global CSI + frame-level SNR, single draw) ===
        csi_dict_enc, amp_t_frame, snr_db_t_frame = channel_sim.sample_csi(
            B, T, channel="fading", snr_min_db=snr_min_db, snr_max_db=snr_max_db
        )
        csi_vec = torch.stack(
            [
                csi_dict_enc["snr_proxy"],
                csi_dict_enc["time_selectivity"],
                csi_dict_enc["freq_selectivity"],
                csi_dict_enc["los_ratio"],
            ],
            dim=-1,
        )


        # === 统计量哈希：对 (mel_mean, mel_std) 做极低维哈希，并在解码端重构 ===
        # 为了提升 4bit 哈希在 (mean,std) 上的表达效率，这里对 mean/std
        # 进行可逆归一化：
        #   - mean_norm = (mel_mean - mean_center) / mean_scale
        #   - std_norm  = (log(mel_std) - std_center) / std_scale
        # 这样哈希瓶颈主要处理范围大致在 [-2,2] 的近似对称分布，
        # 解码时再映射回物理空间，并对 std_hat 加下界约束，避免数值不稳。
        mean_center = -5.0
        mean_scale = 2.0
        std_log = torch.log(mel_std)
        std_center = 0.8
        std_scale = 0.8

        mean_norm = (mel_mean - mean_center) / mean_scale
        std_norm = (std_log - std_center) / std_scale
        stats = torch.cat([mean_norm, std_norm], dim=-1).view(B, 1, 2)  # [B,1,2]

        # 使用与内容分支相同一次采样得到的帧级 SNR，通过时间均值
        # 得到样本级 SNR 标量，避免 stats 分支在独立 SNR 分布下训练。
        snr_db_s = snr_db_t_frame.mean(dim=1, keepdim=True)  # [B,1]
        snr_db_s = snr_db_s.to(device=stats.device, dtype=stats.dtype)

        hb_stats = self.hash_content_stats(stats, channel_params={"snr_db": snr_db_s}, mask=None)
        stats_hat = hb_stats["reconstructed"].view(B, 2)

        mean_hat_norm = stats_hat[:, 0:1].view(B, 1, 1)
        std_hat_norm = stats_hat[:, 1:2].view(B, 1, 1)

        # 直接 latent 级 stats L1：在归一化空间下对齐 hash 重构与 GT 统计量
        # stats: [B,1,2] → [B,2]
        stats_target = stats.view(B, 2)
        stats_latent_l1 = torch.mean(torch.abs(stats_hat - stats_target))

        mean_hat = mean_hat_norm * mean_scale + mean_center
        std_log_hat = std_hat_norm * std_scale + std_center
        # 为避免 std_hat 接近 0 或为负导致 mel 标尺翻折/爆炸，
        # 在训练路径中同样对 std_hat 做约束，使其与离线解码
        # (decode_from_bits_offline) 的行为一致。
        std_hat = std_log_hat.exp().clamp(min=0.1)

        # Debug: log stats bits reconstruction error when DBG_STATS=1
        if os.environ.get("DBG_STATS", "0") == "1":
            try:
                with torch.no_grad():
                    mm = mel_mean.view(B, -1)
                    ms = mel_std.view(B, -1)
                    mmh = mean_hat.view(B, -1)
                    msh = std_hat.view(B, -1)
                    d_mean = mmh - mm
                    d_std = msh - ms

                    def _s(x: torch.Tensor) -> str:
                        return (
                            f"mean={float(x.mean().item()):+.4f} "
                            f"std={float(x.std().item()):+.4f} "
                            f"min={float(x.min().item()):+.4f} "
                            f"max={float(x.max().item()):+.4f}"
                        )

                    print(
                        "[STATS][forward_with_hash] "
                        f"mel_mean={_s(mm)} | mel_mean_hat={_s(mmh)} | "
                        f"delta_mean={_s(d_mean)}"
                    )
                    print(
                        "[STATS][forward_with_hash] "
                        f"mel_std={_s(ms)} | mel_std_hat={_s(msh)} | "
                        f"delta_std={_s(d_std)}"
                    )

                    # 额外输出与离线 decode 相同格式的 bits_decode 统计，
                    # 便于直接对比训练路径与 decode_from_bits_offline。
                    print(
                        "[STATS][bits_decode_train] "
                        f"mel_mean_hat={_s(mmh)} | mel_std_hat={_s(msh)}"
                    )
            except Exception:
                pass

        # === Content branch: VMamba encode -> HashBottleneck -> VMamba decode ===
        _tok_ign, s_map, hw = self.content_vmamba.encode(mel_img_c, csi_vec)  # s_map: [B,C,H,W]
        pstats("s_map", s_map)
        H, W = hw
        Bc, Cc, Hc, Wc = s_map.shape
        assert Hc == H and Wc == W, "Symbol map spatial dims mismatch with hw"
        L_seq = Hc * Wc

        # 使用帧级 SNR 通过插值为 Hash token 生成逐 token SNR 轨迹，
        # 确保与 FiLM 使用的 CSI 来自同一次采样。
        snr_db_t_frame = snr_db_t_frame.to(device=s_map.device, dtype=s_map.dtype)
        if L_seq == T:
            snr_db_t = snr_db_t_frame
        else:
            snr_db_t = F.interpolate(
                snr_db_t_frame.unsqueeze(1), size=L_seq, mode="linear", align_corners=False
            ).squeeze(1)

        # Flatten symbols -> HashBottleneck compress/reconstruct
        s_flat = s_map.permute(0, 2, 3, 1).contiguous().view(B, L_seq, Cc)  # [B,L,C]
        pstats("s_flat", s_flat)

        # 仅将连续 token 作为瓶颈 encoder 的输入，梯度不再回传到 VMamba encoder，
        # 确保解码端依赖的语义信息只通过 bitstream 重构得到。
        hb_in = s_flat.detach()
        hb_ret_c = self.hash_content(hb_in, channel_params={"snr_db": snr_db_t}, mask=None)
        hash_logits_c = hb_ret_c.get("hash_logits", None)
        if hash_logits_c is not None:
            pstats("hash_logits_content", hash_logits_c)
        s_hat_flat = hb_ret_c["reconstructed"]
        s_hat = s_hat_flat.view(B, Hc, Wc, Cc).permute(0, 3, 1, 2).contiguous()
        pstats("s_hat", s_hat)

        # VMamba decoder -> mel_hat (输出是归一化空间的重建)
        # 优先使用 VMambaJSCC2D.decode（支持 CA / per-stage SNR 调制），
        # 若不存在则回退到底层 decoder 模块。
        mel_hat_img_norm = None
        if hasattr(self.content_vmamba, "decode") and callable(getattr(self.content_vmamba, "decode")):
            try:
                mel_hat_img_norm = self.content_vmamba.decode(s_hat, csi_vec, hw)  # [B,C,T',F']
            except TypeError:
                # 兼容旧签名 decode(x, csi_vec)
                mel_hat_img_norm = self.content_vmamba.decode(s_hat, csi_vec)
        if mel_hat_img_norm is None:
            mel_hat_img_norm = self.content_vmamba.decoder(s_hat, csi_vec)  # [B,C,T',F']
        pstats("mel_hat_img_raw", mel_hat_img_norm)
        mel_hat_img_norm = F.interpolate(
            mel_hat_img_norm,
            size=(T, F_mel),
            mode='bilinear',
            align_corners=False,
        )
        # 若 decoder 输出多通道，则在通道维上取均值以得到单一 [B,T,F] 轨迹；
        # 否则直接 squeeze 掉通道维。
        if mel_hat_img_norm.size(1) == 1:
            mel_hat_norm = mel_hat_img_norm.squeeze(1)  # [B,T,F]
        else:
            mel_hat_norm = mel_hat_img_norm.mean(dim=1)  # [B,T,F]

        # 反归一化：默认使用经 hash 重构的 mean/std 恢复能量水平；
        # 若显式设置 ignore_stats_in_mel=True，则跳过 mean/std 校正，
        # 直接让 content_vmamba 输出的 mel_hat_norm 承担完整的能量恢复，
        # 便于对比“纯 content 分支” vs “content+stats bits” 的效果。
        if bool(getattr(self, "ignore_stats_in_mel", False)):
            mel_hat = mel_hat_norm
        else:
            mel_hat = mel_hat_norm * std_hat + mean_hat  # [B,T,32]
        pstats("mel_hat", mel_hat)

        # 可选：在 content-only 预训练阶段启用 L2H 细化，以 GT F0/VUV
        # 作为条件对高频 mel 做补充。这样可以让 content-only 训练时就
        # 学到更合理的高频结构，而不依赖尚未训练好的 F0 分支。
        mel_hat_refined = mel_hat
        if self.enable_l2h and self.deco_l2h_refiner is not None:
            try:
                lb = int(getattr(self, "l2h_low_bins", 10))
                lb = max(1, min(lb, F_mel - 1))
                mel_low = mel_hat[:, :, :lb]
                mel_high_base = mel_hat[:, :, lb:]

                # 使用 FARGAN GT 特征提取 dnn_pitch / frame_corr 作为 L2H 条件
                dnn_pitch_gt = None
                frame_corr_gt = None
                try:
                    dnn_pitch_gt = self.fargan_spec.extract_feature(fargan_feats, "dnn_pitch")
                    frame_corr_gt = self.fargan_spec.extract_feature(fargan_feats, "frame_corr")
                except Exception:
                    dnn_pitch_gt = None
                    frame_corr_gt = None
                if isinstance(dnn_pitch_gt, torch.Tensor):
                    dnn_pitch_gt = dnn_pitch_gt[:, :T, :]
                else:
                    dnn_pitch_gt = None
                if isinstance(frame_corr_gt, torch.Tensor):
                    frame_corr_gt = frame_corr_gt[:, :T, :]
                else:
                    frame_corr_gt = None

                # 统一调用新版 DeCoL2HRefiner 接口
                out_h = self.deco_l2h_refiner(
                    mel_low,
                    dnn_pitch_gt if isinstance(dnn_pitch_gt, torch.Tensor) else torch.zeros_like(mel_low[..., :1]),
                    frame_corr_gt if isinstance(frame_corr_gt, torch.Tensor) else torch.zeros_like(mel_low[..., :1]),
                    mel_high_base,
                )

                if isinstance(out_h, tuple):
                    mel_high_ref, _ = out_h
                else:
                    mel_high_ref, _ = out_h, None

                blend = float(getattr(self, "l2h_blend", 1.0))
                if not np.isfinite(blend):
                    blend = 1.0
                blend = max(0.0, min(1.0, blend))
                vuv_prob = torch.sigmoid(frame_corr_gt) if isinstance(frame_corr_gt, torch.Tensor) else 1.0
                mel_high_mix = mel_high_base + (blend * vuv_prob) * (mel_high_ref - mel_high_base)
                mel_hat_refined = torch.cat([mel_low, mel_high_mix], dim=-1)
            except Exception:
                mel_hat_refined = mel_hat
        pstats("mel_hat_refined", mel_hat_refined)

        # ---- log-mel -> ceps（聚合到18带后做 DCT）----
        ceps_target: Optional[torch.Tensor]
        try:
            ceps_target = self.fargan_spec.extract_feature(fargan_feats, "ceps")
        except Exception:
            ceps_target = None

        ceps_hat: Optional[torch.Tensor]
        ceps_hat_base: Optional[torch.Tensor]
        try:
            def _mel_to_ceps(mel_src: torch.Tensor) -> torch.Tensor:
                mel_used = torch.clamp(mel_src, -10.0, 2.0)
                E = torch.pow(10.0, torch.clamp(mel_used, min=-10.0, max=10.0))
                e18_energy = self.band_agg_32_to_18(E)
                e18_energy = torch.clamp(e18_energy, min=1e-6)
                e18_log = torch.log10(e18_energy)
                e18_log = opus_band_log_smooth(e18_log)
                return torch.nan_to_num(self.mel18_to_ceps(e18_log), nan=0.0)

            # baseline ceps (不经过 L2H)，用于低阶监督/对比
            ceps_hat_base = _mel_to_ceps(mel_hat)
            # refined ceps（经过 L2H），用于高阶监督
            ceps_hat = _mel_to_ceps(mel_hat_refined)

            if ceps_target is not None:
                ceps_target = torch.nan_to_num(ceps_target, nan=0.0)
            # 可选：训练期使用 GT ceps c0 做轻量校准；若模型上设置了
            # disable_ceps_c0_calib=True，则完全跳过该步骤，避免依赖 GT。
            if (
                self.enable_energy_calib
                and self.training
                and ceps_target is not None
                and not bool(getattr(self, "disable_ceps_c0_calib", False))
            ):
                with torch.no_grad():
                    dc0 = (
                        ceps_target[..., 0].mean(dim=1, keepdim=True)
                        - ceps_hat[..., 0].mean(dim=1, keepdim=True)
                    )
                ceps_hat = ceps_hat.clone()
                ceps_hat[..., 0] = ceps_hat[..., 0] + float(self.energy_calib_alpha) * dc0
            pstats("ceps_hat", ceps_hat)
            if ceps_target is not None:
                pstats("ceps_target", ceps_target)
        except Exception:
            ceps_hat = None
            ceps_hat_base = None

        # Hash/RVQ regularization terms (content branch only)
        hash_reg_terms: Dict[str, torch.Tensor] = {}
        if self.quantizer_type == "hash" and hasattr(self.hash_content, "compute_hash_regularization"):
            try:
                hash_reg_terms_c = self.hash_content.compute_hash_regularization(
                    hash_logits=hb_ret_c.get("hash_logits"),
                    hash_bits=hb_ret_c.get("hash_bits_clean"),
                    mask=None,
                )
                hash_reg_terms.update({f"content_{k}": v for k, v in hash_reg_terms_c.items()})
            except Exception:
                hash_reg_terms = {}

        # 统一提取 bitstream（兼容 HashBottleneck 与 RVQBottleneck）。
        # 为了在训练时统计各分支 BER，这里同时保留 clean/noisy 版本。
        if "hash_bits_clean" in hb_ret_c:
            bits_c_clean = hb_ret_c["hash_bits_clean"]
            bits_c_noisy = hb_ret_c.get("hash_bits_noisy", bits_c_clean)
        else:
            bits_c_clean = hb_ret_c.get("bits_clean")
            bits_c_noisy = hb_ret_c.get("bits_noisy", bits_c_clean)

        # 统计量分支同样兼容 HashBottleneck 与 RVQBottleneck：
        # 前者返回 hash_bits_*，后者返回 bits_*。
        if "hash_bits_clean" in hb_stats:
            bits_s_clean = hb_stats["hash_bits_clean"]
            bits_s_noisy = hb_stats.get("hash_bits_noisy", bits_s_clean)
        else:
            bits_s_clean = hb_stats.get("bits_clean")
            bits_s_noisy = hb_stats.get("bits_noisy", bits_s_clean)

        out: Dict[str, torch.Tensor] = {
            "mel": mel,
            "mel_hat": mel_hat,
            "mel_hat_refined": mel_hat_refined,
            "tokens": s_flat.detach(),
            "tokens_hat": s_hat_flat,
            "tokens_map": s_map.detach(),
            "tokens_hat_map": s_hat.detach(),
            "hash_bits_clean": bits_c_clean,
            "hash_bits_noisy": bits_c_noisy,
            # 对于 RVQ stats，hash_logits_stats 可能不存在，此时保持键缺失即可。
            "hash_logits_stats": hb_stats.get("hash_logits"),
            "hash_bits_stats": bits_s_clean,
            "hash_bits_stats_noisy": bits_s_noisy,
            "hash_reg_terms": hash_reg_terms,
            # 归一化 stats 空间下的直接 L1 损失（仅在 quantizer_type='hash' 时非空），
            # 用于在训练中加强对 hash_content_stats 的监督，防止其塌缩为常数解。
            "stats_latent_l1": stats_latent_l1,
            "z_content": s_flat.detach(),
            "z_content_hat": s_hat_flat,
            "mel_mean": mel_mean.detach(),
            "mel_std": mel_std.detach(),
            "csi_vec": csi_vec.detach(),
        }

        # 在 content-only 路径中同样暴露 L2H 中间量，便于在
        # 简化版 Stage2.5 脚本中复用与 full 模式一致的 L2H
        # residual / decor loss 公式。
        if getattr(self, "enable_l2h", False) and getattr(self, "deco_l2h_refiner", None) is not None:
            l2h_resid = getattr(self.deco_l2h_refiner, "last_resid", None)
            l2h_vuv_prob = getattr(self.deco_l2h_refiner, "last_vuv_prob", None)
            l2h_mask_harm = getattr(self.deco_l2h_refiner, "last_mask_harm", None)
            if isinstance(l2h_resid, torch.Tensor):
                out["l2h_resid"] = l2h_resid
            if isinstance(l2h_vuv_prob, torch.Tensor):
                out["l2h_vuv_prob"] = l2h_vuv_prob
            if isinstance(l2h_mask_harm, torch.Tensor):
                out["l2h_mask_harm"] = l2h_mask_harm

        if hash_logits_c is not None:
            out["hash_logits"] = hash_logits_c

        # RVQ VQ loss（仅在 quantizer_type=="rvq" 时存在）
        vq_loss_c = hb_ret_c.get("vq_loss", None)
        if isinstance(vq_loss_c, torch.Tensor):
            out["vq_loss"] = vq_loss_c
            out["vq_loss_content"] = vq_loss_c

        # Stats RVQ VQ loss（仅在 quantizer_type=="rvq" 时非空）。
        vq_loss_s = hb_stats.get("vq_loss", None)
        if isinstance(vq_loss_s, torch.Tensor):
            out["vq_loss_stats"] = vq_loss_s

        if ceps_hat is not None:
            out["ceps_hat"] = ceps_hat
        if ceps_hat_base is not None:
            out["ceps_hat_base"] = ceps_hat_base
        if ceps_target is not None:
            out["ceps"] = ceps_target

        return out

    def forward_with_hash(
        self,
        audio: torch.Tensor,
        fargan_feats: torch.Tensor,
        channel_sim,
        snr_min_db: float,
        snr_max_db: float,
        target_len: Optional[int] = None,
        hash_only: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Stage3.5 前向：在 Mel+VMamba 内容分支上加入 HashBottleneck。

        当 ``hash_only=True`` 时，仅进行 HashBottleneck 编码/解码并返回
        ``hash_bits_clean`` / ``f0_hash_bits_clean``，跳过 mel→ceps 与
        FARGAN vocoder 的合成路径，以便在离线导出比特时降低计算量。
        """

        assert self.with_hash, "forward_with_hash 仅在 with_hash/use_hash=True 时使用"

        device = self.device
        audio = audio.to(device)
        fargan_feats = fargan_feats.to(device)

        B, L = audio.shape
        B2, T_feat, _ = fargan_feats.shape
        assert B == B2

        # === 内容分支：wave -> mel ===
        mel = self.wave_to_mel(audio)                 # [B,T_mel,32]
        pstats("mel", mel)
        Bm, T_mel, F_mel = mel.shape

        T = min(T_mel, T_feat)
        mel = mel[:, :T, :]
        fargan_feats = fargan_feats[:, :T, :]

        # 显式归一化：记录 mel 的 mean/std，编码前归一化，解码后反归一化
        mel_mean = mel.mean(dim=(1, 2), keepdim=True)  # [B,1,1]
        mel_std = mel.std(dim=(1, 2), keepdim=True).clamp(min=0.1)  # [B,1,1]
        mel_norm = (mel - mel_mean) / mel_std  # [B,T,32]

        mel_img = mel_norm.unsqueeze(1)                # [B,1,T,32] 使用归一化后的mel
        mel_img_c = self._content_pre_downsample(mel_img)

        # === 信道采样（全局 CSI + 帧级 SNR，一次性采样） ===
        csi_dict_enc, _amp_t_frame_h, snr_db_t_frame_h = channel_sim.sample_csi(
            B, T, channel="fading", snr_min_db=snr_min_db, snr_max_db=snr_max_db
        )
        csi_vec = torch.stack(
            [
                csi_dict_enc["snr_proxy"],
                csi_dict_enc["time_selectivity"],
                csi_dict_enc["freq_selectivity"],
                csi_dict_enc["los_ratio"],
            ],
            dim=-1,
        )

        # NOTE: simplified Stage2.5 训练中暂时关闭 SNR-aware 的 L2H blend，
        # 仅使用由调度器/配置直接设定的 self.l2h_blend。这样可以避免在
        # resume 旧 checkpoint 时，由于 SNR 量纲差异导致 L2H 有效强度不
        # 易对齐。若后续需要恢复 SNR-aware 行为，可在此处重新引入
        # snr_proxy 相关逻辑。


        # === 统计量哈希：对 (mel_mean, mel_std) 做极低维哈希，并在解码端重构 ===
        # 使用与 forward_content_only 相同的归一化策略：
        #   mean_norm / std_norm (log-std) → 4bit hash → 反归一化。
        mean_center = -5.0
        mean_scale = 2.0
        std_log = torch.log(mel_std)
        std_center = 0.8
        std_scale = 0.8

        mean_norm = (mel_mean - mean_center) / mean_scale
        std_norm = (std_log - std_center) / std_scale
        stats = torch.cat([mean_norm, std_norm], dim=-1).view(B, 1, 2)  # [B,1,2]

        # 复用内容分支同一次 CSI 的帧级 SNR，通过时间均值得到样本级 SNR，
        # 避免 stats 分支在独立 SNR 分布下训练。
        snr_db_s = snr_db_t_frame_h.mean(dim=1, keepdim=True)  # [B,1]
        snr_db_s = snr_db_s.to(device=stats.device, dtype=stats.dtype)

        hb_stats = self.hash_content_stats(stats, channel_params={"snr_db": snr_db_s}, mask=None)
        stats_hat = hb_stats["reconstructed"].view(B, 2)

        mean_hat_norm = stats_hat[:, 0:1].view(B, 1, 1)
        std_hat_norm = stats_hat[:, 1:2].view(B, 1, 1)

        # latent 级 stats L1，与 forward_content_only 保持一致
        stats_target = stats.view(B, 2)
        stats_latent_l1 = torch.mean(torch.abs(stats_hat - stats_target))

        mean_hat = mean_hat_norm * mean_scale + mean_center
        std_log_hat = std_hat_norm * std_scale + std_center
        std_hat = std_log_hat.exp().clamp(min=0.1)

        # Debug: 训练路径下的 bits_decode 统计，与 decode_from_bits_offline
        # 中的 [STATS][bits_decode] 保持相同格式，便于直接对比。
        if os.environ.get("DBG_STATS", "0") == "1":
            try:
                with torch.no_grad():
                    mmh = mean_hat.view(B, -1)
                    msh = std_hat.view(B, -1)

                    def _s_bits(x: torch.Tensor) -> str:
                        return (
                            f"mean={float(x.mean().item()):+.4f} "
                            f"std={float(x.std().item()):+.4f} "
                            f"min={float(x.min().item()):+.4f} "
                            f"max={float(x.max().item()):+.4f}"
                        )

                    print(
                        "[STATS][bits_decode_train] "
                        f"mel_mean_hat={_s_bits(mmh)} | mel_std_hat={_s_bits(msh)}"
                    )
            except Exception:
                pass

        # === 内容分支 Hash + VMamba JSCC（去除残差预测耦合，直接在 token 上做 HashBottleneck） ===
        _tok_ign, s_map, hw = self.content_vmamba.encode(mel_img_c, csi_vec)    # s_map: [B,C,H,W], C=d_s_content
        pstats("s_map", s_map)
        H, W = hw
        Bc, Cc, Hc, Wc = s_map.shape
        assert Hc == H and Wc == W, "Symbol map spatial dims mismatch with hw"
        L_seq = Hc * Wc

        # 使用同一次 CSI 的帧级 SNR 插值到 Hash token 序列上
        snr_db_t_frame_h = snr_db_t_frame_h.to(device=s_map.device, dtype=s_map.dtype)
        if L_seq == T:
            snr_db_t = snr_db_t_frame_h
        else:
            snr_db_t = F.interpolate(
                snr_db_t_frame_h.unsqueeze(1), size=L_seq, mode="linear", align_corners=False
            ).squeeze(1)

        # 展平符号 → 经 Hash/RVQ 瓶颈压缩/重建（不再依赖残差预测），
        # 以获得更清晰的编解码分界，代价由重新训练弥补。
        s_flat = s_map.permute(0, 2, 3, 1).contiguous().view(B, L_seq, Cc)  # [B,L,C]
        pstats("s_flat", s_flat)

        # 仅将连续 token 作为瓶颈 encoder 的输入。默认情况下，不将梯度
        # 回传到 VMamba encoder，以保持“语义只能通过 bitstream + CSI” 的
        # JSCC 设计。若模型上显式设置 ``allow_hash_content_grad=True``，则
        # 放开这一限制，允许某些损失（例如静音相关）直接影响内容分支
        # encoder，以便更强地塑形 BFCC 统计。
        if bool(getattr(self, "allow_hash_content_grad", False)):
            hb_in = s_flat
        else:
            hb_in = s_flat.detach()
        hb_ret_c = self.hash_content(hb_in, channel_params={"snr_db": snr_db_t}, mask=None)
        if os.getenv("DBG_BITS", "0") == "1":
            # 内容分支的比特 BER，用于验证 JSCC 信道在 content 路径上是否生效
            bclean_c = hb_ret_c.get("bits_clean", hb_ret_c.get("hash_bits_clean", None))
            bnoisy_c = hb_ret_c.get("bits_noisy", hb_ret_c.get("hash_bits_noisy", None))
            if bclean_c is not None and bnoisy_c is not None:
                print(
                    f"[ENC content BER] ber={_ber_pm1(bclean_c, bnoisy_c):.4f} "
                    f"p(+1)clean={float((bclean_c>0).float().mean().item()):.3f} "
                    f"p(+1)noisy={float((bnoisy_c>0).float().mean().item()):.3f}"
                )
        hash_logits_c = hb_ret_c.get("hash_logits", None)
        if hash_logits_c is not None:
            pstats("hash_logits_content", hash_logits_c)  # 便于观察幅度与分布
        s_hat_flat = hb_ret_c["reconstructed"]
        s_hat = s_hat_flat.view(B, Hc, Wc, Cc).permute(0, 3, 1, 2).contiguous()
        pstats("s_hat", s_hat)

        # 优先使用 VMambaJSCC2D.decode（支持 CA / per-stage SNR 调制）；
        # 若旧版 VMambaJSCC2D 未实现 decode，则回退到底层 decoder 接口。
        mel_hat_img_norm = None
        if hasattr(self.content_vmamba, "decode") and callable(getattr(self.content_vmamba, "decode")):
            try:
                mel_hat_img_norm = self.content_vmamba.decode(s_hat, csi_vec, hw)  # [B,C,T',F'] 归一化空间
            except TypeError:
                # 兼容旧签名 decode(x, csi_vec)
                mel_hat_img_norm = self.content_vmamba.decode(s_hat, csi_vec)
        if mel_hat_img_norm is None:
            mel_hat_img_norm = self.content_vmamba.decoder(s_hat, csi_vec)   # [B,C,T',F'] 归一化空间

        pstats("mel_hat_img_raw", mel_hat_img_norm)
        mel_hat_img_norm = F.interpolate(mel_hat_img_norm, size=(T, F_mel), mode='bilinear', align_corners=False)
        # 确保输出是 3D [B,T,F]：若 C>1 则取均值，否则 squeeze
        if mel_hat_img_norm.size(1) == 1:
            mel_hat_norm = mel_hat_img_norm.squeeze(1)  # [B,T,F]
        else:
            mel_hat_norm = mel_hat_img_norm.mean(dim=1)  # [B,T,F] 多通道取均值

        # 反归一化：同 forward_with_hash，允许通过 ignore_stats_in_mel
        # 开关完全跳过 mean/std 约束，仅依赖 content 分支自身恢复 mel。
        if bool(getattr(self, "ignore_stats_in_mel", False)):
            mel_hat = mel_hat_norm
        else:
            mel_hat = mel_hat_norm * std_hat + mean_hat  # [B,T,32]
        pstats("mel_hat", mel_hat)

        # === F0/voicing 分支：残差 hash 瓶颈（在 z_fv 上） ===
        dnn_pitch = self.fargan_spec.extract_feature(fargan_feats, "dnn_pitch")  # [B,T,1]
        frame_corr = self.fargan_spec.extract_feature(fargan_feats, "frame_corr")# [B,T,1]
        # 同样进行数值清理，避免 hash 路径下 GRU 接收到非有限值
        dnn_pitch = torch.nan_to_num(dnn_pitch, nan=0.0, posinf=3.0, neginf=-3.0).clamp(-3.0, 3.0)
        frame_corr = torch.nan_to_num(frame_corr, nan=0.0, posinf=0.5, neginf=-0.8).clamp(-0.8, 0.5)
        pstats("dnn_pitch(target)", dnn_pitch)
        pstats("frame_corr(target)", frame_corr)
        f0vuv = torch.cat([dnn_pitch, frame_corr], dim=-1)                           # [B,T,2]

        z_fv = self.f0vuv_enc(f0vuv)                                                 # [B,T,d_zf]

        # 直接对连续 latent z_fv 使用 Hash/RVQ 瓶颈。为了保持与内容分支一致的
        # 信道统计，这里复用前面采样得到的帧级 SNR（或其拷贝），而不再单独
        # 调用一次 sample_csi。同时，允许基于 V/UV 掩膜做一个“帧内
        # rate-shaping”：在有声帧上适当提升 F0 通道的等效 SNR、在无声
        # 帧上略微降低，以在固定 bit 配置下把可靠度更多投向有意义的
        # F0 样本。
        snr_db_t_f = snr_db_t_frame_h.to(device=z_fv.device, dtype=z_fv.dtype)
        try:
            vuv_thr = float(getattr(self, "vuv_threshold", 0.3))
            vuv_mask = (frame_corr > vuv_thr).to(snr_db_t_f.dtype).squeeze(-1)  # [B,T]
            delta_db = float(getattr(self, "f0_snr_delta_db", 0.0))
            if delta_db != 0.0:
                # voiced: +delta_db, unvoiced: -delta_db（近似保持平均 SNR 不变）
                snr_db_t_f = snr_db_t_f + delta_db * (2.0 * vuv_mask - 1.0)
        except Exception:
            pass
        hb_ret = self.hash_f0vuv(
            z_fv,
            channel_params={"snr_db": snr_db_t_f},
            mask=None,
        )
        if os.getenv("DBG_BITS","0") == "1":
            _pstats("ENC z_fv (input to rvq)", z_fv)
            # bits
            bclean = hb_ret.get("bits_clean", hb_ret.get("hash_bits_clean", None))
            bnoisy = hb_ret.get("bits_noisy", hb_ret.get("hash_bits_noisy", None))
            if bclean is not None and bnoisy is not None:
                print(f"[ENC f0 BER] ber={_ber_pm1(bclean, bnoisy):.4f} "
                    f"p(+1)clean={float((bclean>0).float().mean().item()):.3f} "
                    f"p(+1)noisy={float((bnoisy>0).float().mean().item()):.3f}")
            # codes
            if "codes" in hb_ret:
                _rvq_code_hist("ENC f0 codes", self.hash_f0vuv, hb_ret["codes"])

        z_fv_hat = hb_ret["reconstructed"]                                         # [B,T,d_zf]
        if os.getenv("DBG_BITS","0") == "1":
            _pstats("ENC z_fv_hat (rvq recon)", z_fv_hat)

        # full F0/VUV 解码（使用全部 RVQ codebooks 重构），并使用
        # mel_hat_norm 作为内容条件。与 decode_from_bits_offline 保持一致：
        # 先得到原始 dnn_pitch_raw 与 vuv_logits，再通过 vuv_prob 对 F0
        # 做门控，使无声段 F0 自然压到 0 附近。
        dnn_pitch_raw, vuv_logits = self.f0vuv_dec(  # [B,T,1]x2
            z_fv_hat,
            mel_cond=mel_hat_norm,
        )
        frame_corr_hat_raw, vuv_prob_hat = self._vuv_logits_to_feat(vuv_logits)

        dp_calib = self.f0_calib_scale * dnn_pitch_raw + self.f0_calib_bias
        frame_corr_hat = self.fc_calib_scale * frame_corr_hat_raw + self.fc_calib_bias
        dnn_pitch_hat = vuv_prob_hat * dp_calib
        pstats("dnn_pitch_hat", dnn_pitch_hat)
        pstats("frame_corr_hat", frame_corr_hat)

        # SR base 分支（RVQ 版）：仅使用前 N_base 个 codebook，
        # 将其余 codebook 的贡献视为“细节层”，实现 successive refinement。
        dnn_pitch_hat_base: Optional[torch.Tensor] = None
        frame_corr_hat_base: Optional[torch.Tensor] = None
        try:
            # 优先使用经信道扰动后的 noisy codes_hat，使 SR base
            # 真正在“带噪”索引上学习稳定骨架，而非仅在 clean codes 上。
            codes_f = hb_ret.get("codes_hat", hb_ret.get("codes", None))
            codebooks = getattr(self.hash_f0vuv, "codebooks", None)
            if isinstance(codes_f, torch.Tensor) and isinstance(codebooks, torch.nn.ModuleList):
                Bf, Tf, Nq = codes_f.shape
                if Nq > 0:
                    # 选择 base 使用的 codebook 数量：默认为 1；若显式设置了
                    # f0_sr_k，则取 min(f0_sr_k, Nq) 作为 base codebook 数量。
                    sr_levels = int(getattr(self, "f0_sr_k", 1))
                    N_base = max(1, min(sr_levels, Nq))
                    z_base = 0.0
                    for qi in range(N_base):
                        emb = codebooks[qi]
                        idx = codes_f[..., qi].to(torch.long)  # [B,T]
                        z_base = z_base + emb(idx)
                    dnn_pitch_hat_base, vuv_logits_base = self.f0vuv_dec(
                        z_base, mel_cond=mel_hat_norm
                    )  # [B,T,1]x2
                    frame_corr_hat_base, _ = self._vuv_logits_to_feat(vuv_logits_base)
                    pstats("dnn_pitch_hat_base", dnn_pitch_hat_base)
                    pstats("frame_corr_hat_base", frame_corr_hat_base)
        except Exception:
            dnn_pitch_hat_base = None
            frame_corr_hat_base = None

        # 若仅需导出 bitstream，则在此提前返回，避免后续 mel 细化、
        # ceps 计算与 FARGAN 声码器推理带来的额外开销。
        if hash_only:
            # 统一提取 clean/noisy bits（兼容 HashBottleneck 与 RVQBottleneck）。
            if "hash_bits_clean" in hb_ret_c:
                bits_c_clean = hb_ret_c["hash_bits_clean"]
                bits_c_noisy = hb_ret_c.get("hash_bits_noisy", bits_c_clean)
            else:
                bits_c_clean = hb_ret_c["bits_clean"]
                bits_c_noisy = hb_ret_c.get("bits_noisy", bits_c_clean)

            if "hash_bits_clean" in hb_ret:
                bits_f_clean = hb_ret["hash_bits_clean"]
                bits_f_noisy = hb_ret.get("hash_bits_noisy", bits_f_clean)
            else:
                bits_f_clean = hb_ret["bits_clean"]
                bits_f_noisy = hb_ret.get("bits_noisy", bits_f_clean)
            if os.getenv("DBG_BITS","0") == "1":
                _pstats("ENC(hash_only) dnn_pitch_hat", dnn_pitch_hat)
                _pstats("ENC(hash_only) frame_corr_hat", frame_corr_hat)
                # csi/snrs
                _pstats("ENC csi_vec", csi_vec)
                _pstats("ENC snr_db_t_f", snr_db_t_f)

            # 统计量分支同样兼容 Hash / RVQ。
            if "hash_bits_clean" in hb_stats:
                bits_s_clean = hb_stats["hash_bits_clean"]
                bits_s_noisy = hb_stats.get("hash_bits_noisy", bits_s_clean)
            else:
                bits_s_clean = hb_stats.get("bits_clean")
                bits_s_noisy = hb_stats.get("bits_noisy", bits_s_clean)

            return {
                "hash_bits_clean": bits_c_clean,
                "hash_bits_noisy": bits_c_noisy,
                "f0_hash_bits_clean": bits_f_clean,
                "f0_hash_bits_noisy": bits_f_noisy,
                "hash_bits_stats": bits_s_clean,
                "hash_bits_stats_noisy": bits_s_noisy,
                "T": T,
                "F_mel": F_mel,
                "hw": (H, W),
                "csi_vec": csi_vec,
            }

        # 低→高 mel 细化（使用 F0/VUV 条件指导高频生成）

        # 低→高 mel 细化（使用 F0/VUV 条件指导高频生成）
        mel_used = mel_hat
        # 训练期允许使用 GT mel 做能量校准；推理期避免依赖编码端的参考信息。
        # 这里使用展平后的全局均值，避免在 T/F 维度发生变化时形状不匹配。
        if self.enable_energy_calib:
            with torch.no_grad():
                mu_ref = mean_hat.view(B, 1)                     # 来自 bits_stats 解码
                mu_hat = mel_used.view(B, -1).mean(dim=1, keepdim=True)
                d_mu = (mu_hat - mu_ref).view(B, 1, 1)
            mel_used = mel_used - float(self.energy_calib_alpha) * d_mu
        mel_used = torch.clamp(mel_used, -10.0, 2.0)
        if hasattr(self, 'enable_l2h') and getattr(self, 'enable_l2h'):
            lb = int(getattr(self, 'l2h_low_bins', 10))
            lb = max(1, min(lb, F_mel - 1))
            mel_low = mel_hat[:, :, :lb]
            mel_high = mel_hat[:, :, lb:]

            # 获取 F0/VUV 条件：训练和推理统一使用预测值，
            # 避免 L2H 在不同阶段看到不同分布的条件。
            _dnn_pitch_cond = torch.nan_to_num(dnn_pitch_hat, nan=0.0).clamp(-3.0, 3.0)
            # frame_corr_hat 更像 vuv logits：不要过度裁剪，否则 vuv_prob 只能在 0.3~0.6 徘徊。
            _frame_corr_cond = torch.nan_to_num(frame_corr_hat, nan=0.0).clamp(-8.0, 8.0)


            out_h = self.deco_l2h_refiner(
                mel_low,
                _dnn_pitch_cond,
                _frame_corr_cond,
                mel_high,
            )

            if isinstance(out_h, tuple):
                mel_high_ref, _dc_high = out_h
            else:
                mel_high_ref, _dc_high = out_h, None

            # 训练时做渐进 blend：避免 L2H 一上来就“重绘”高频导致语义结构崩。
            blend = float(getattr(self, "l2h_blend", 1.0))
            vuv_prob = torch.sigmoid(_frame_corr_cond)
            mel_high_mix = mel_high + (blend * vuv_prob) * (mel_high_ref - mel_high)
            mel_used = torch.cat([mel_low, mel_high_mix], dim=-1)


        # log-mel -> ceps（聚合到18带后做 DCT）
        ceps_target = self.fargan_spec.extract_feature(fargan_feats, "ceps")
        E = torch.pow(10.0, torch.clamp(mel_used, min=-10.0, max=10.0))
        e18_energy = self.band_agg_32_to_18(E)
        e18_log = torch.log10(e18_energy + 1e-10)
        e18_log = opus_band_log_smooth(e18_log)
        ceps_hat = self.mel18_to_ceps(e18_log)                # [B,T,18]

        # 防御性去除 NaN/Inf
        ceps_target = torch.nan_to_num(ceps_target, nan=0.0)
        ceps_hat = torch.nan_to_num(ceps_hat, nan=0.0)

        # 方案 A：基于 GT FARGAN ceps 的 c0 校准（默认仍关闭，保持与
        # 现有简化版训练行为一致，仅在 enable_energy_calib=True 时启用）。
        if self.enable_energy_calib and self.training and not bool(getattr(self, "disable_ceps_c0_calib", False)):
            with torch.no_grad():
                dc0 = (
                    ceps_target[..., 0].mean(dim=1, keepdim=True)
                    - ceps_hat[..., 0].mean(dim=1, keepdim=True)
                )
            ceps_hat = ceps_hat.clone()
            ceps_hat[..., 0] = ceps_hat[..., 0] + float(self.energy_calib_alpha) * dc0

        # 方案 B：完全自包含的可学习能量校准头，不依赖 GT FARGAN 特征；
        # 仅使用模型自身输出的 mel_used 统计预测一个小的 c0 偏移。
        if getattr(self, "use_learned_energy_calib", False):
            try:
                # 使用时间维上的平均 mel 作为全局能量特征 [B, n_mels]
                mel_mean = mel_used.mean(dim=1)  # [B, F_mel]
                delta_c0 = self.energy_calib_head(mel_mean)  # [B,1]
                # 限制偏移幅度，避免极端发散
                delta_c0 = delta_c0.clamp(-1.0, 1.0)
                ceps_hat = ceps_hat.clone()
                ceps_hat[..., 0] = ceps_hat[..., 0] + delta_c0.view(-1, 1, 1)
            except Exception:
                # 若能量校准头出错，退回未校准的 ceps_hat
                pass

        # 轻量级调试打印（仅在 DBG_STAGE25=1 或 DBG_CEPS_ENERGY=1 时启用），
        # 用于对比 FARGAN 输入前的 ceps_hat / ceps_target 以及 mel_used、
        # F0/VUV 的分布，帮助定位能量地板是否主要来自 ceps 侧。
        pstats("ceps_target", ceps_target)
        pstats("ceps_hat", ceps_hat)
        if os.environ.get("DBG_CEPS_ENERGY", "0") == "1":
            try:
                _pstats("DBG ceps_target_c0", ceps_target[..., 0:1])
                _pstats("DBG ceps_hat_c0", ceps_hat[..., 0:1])
                _pstats("DBG mel_used", mel_used)
                _pstats("DBG dnn_pitch_hat", dnn_pitch_hat)
                _pstats("DBG frame_corr_hat", frame_corr_hat)
                _pstats("DBG e18_energy", e18_energy)
            except Exception as _dbg_e:
                print(f"[DBG_CEPS_ENERGY] debug print failed: {_dbg_e}")

        # === 拼回 FARGAN 特征 (前20维) ===
        # 避免 in-place 操作，使用 torch.cat 重新构造。
        # 训练/推理统一：始终使用预测的 frame_corr_hat 作为声码器 gate，
        # 与主 forward 路径保持一致，不再在训练期混入 GT frame_corr。
        try:
            frame_corr_in = torch.clamp(frame_corr_hat, -0.8, 0.5)
        except Exception:
            frame_corr_in = torch.clamp(frame_corr_hat, -0.8, 0.5)

        # 训练/推理统一：不再在训练期使用 FARGAN 后 16 维 GT 特征，
        # 始终以零占位作为额外通道，使声码器输入分布与离线/部署路径一致。
        feat_extra = None
        try:
            extra_dim = max(0, fargan_feats.size(-1) - 20)
            if extra_dim > 0:
                feat_extra = torch.zeros(B, T, extra_dim, device=ceps_hat.device, dtype=ceps_hat.dtype)
        except Exception:
            feat_extra = None

        # HF 侧通道：从高频 mel 残差提取紧凑表示
        hf_sideband_feat_hash: Optional[torch.Tensor] = None
        if getattr(self, 'with_hf_sideband', False) and getattr(self, 'enable_l2h', False):
            try:
                lb = int(getattr(self, 'l2h_low_bins', 10))
                # L2H 启用时，mel_used 已是细化后的结果
                mel_high_refined = mel_used[:, :, lb:]   # [B,T,hf_bins]
                mel_high_baseline = mel_hat[:, :, lb:]   # [B,T,hf_bins]
                hf_residual = mel_high_refined - mel_high_baseline

                if self.hf_sideband_type == "dct":
                    hf_sideband_feat_hash = torch.matmul(hf_residual, self.hf_dct_proj)
                elif self.hf_sideband_encoder is not None:
                    hf_sideband_feat_hash = self.hf_sideband_encoder(hf_residual)

                if hf_sideband_feat_hash is not None:
                    T_ceps = ceps_hat.size(1)
                    T_hf = hf_sideband_feat_hash.size(1)
                    if T_hf > T_ceps:
                        hf_sideband_feat_hash = hf_sideband_feat_hash[:, :T_ceps, :]
                    elif T_hf < T_ceps:
                        pad = torch.zeros(B, T_ceps - T_hf, self.hf_sideband_dim,
                                          device=hf_sideband_feat_hash.device, dtype=hf_sideband_feat_hash.dtype)
                        hf_sideband_feat_hash = torch.cat([hf_sideband_feat_hash, pad], dim=1)

                    # 利用 HF 侧通道特征对 ceps 高阶做同样的轻量校正
                    if self.hf2ceps is not None and self.hf2ceps_dim > 0:
                        try:
                            hi_start = int(getattr(self, 'ceps_hi_start', 10))
                            hi_start = max(0, min(hi_start, ceps_hat.size(-1)))
                            max_len = max(0, ceps_hat.size(-1) - hi_start)
                            slice_len = min(self.hf2ceps_dim, max_len)
                            if slice_len > 0:
                                delta_hi = self.hf2ceps(hf_sideband_feat_hash)  # [B,T,dim]
                                delta_hi = delta_hi[..., :slice_len]
                                ceps_hat[..., hi_start:hi_start + slice_len] = (
                                    ceps_hat[..., hi_start:hi_start + slice_len]
                                    + float(self.hf2ceps_scale) * delta_hi
                                )
                        except Exception:
                            pass
            except Exception:
                hf_sideband_feat_hash = None

        # 拼接 FARGAN 特征
        # 参考尺寸以 ceps_hat 为准：所有特征对齐到 [B_ref,T_ref,D] 再拼接。
        B_ref, T_ref = ceps_hat.size(0), ceps_hat.size(1)

        def _align_to_ceps(x: torch.Tensor, name: str) -> torch.Tensor:
            # 先确保是 3 维 [B,T,D]
            if x.dim() == 3:
                y = x
            elif x.dim() == 4 and x.size(1) == 1:
                y = x.squeeze(1)          # [B,T,D]
            elif x.dim() == 4 and x.size(-1) == 1:
                y = x.squeeze(-1)         # [B,T,D]
            else:
                Bx, Tx = x.size(0), x.size(1)
                y = x.view(Bx, Tx, -1)

            Bx, Tx, Dx = y.size()

            # 若出现 [T,B,D] 形式（罕见），尝试自动转置
            if Bx == T_ref and Tx == B_ref:
                y = y.permute(1, 0, 2)   # [B_ref,T_ref,D]
                Bx, Tx, Dx = y.size()

            # 对齐时间长度到 T_ref
            if Tx > T_ref:
                y = y[:, :T_ref, :]
            elif Tx < T_ref:
                pad_t = y.new_zeros(Bx, T_ref - Tx, Dx)
                y = torch.cat([y, pad_t], dim=1)

            # 对齐 batch 维度到 B_ref（通常已经一致，仅为安全）
            if Bx > B_ref:
                y = y[:B_ref]
            elif Bx < B_ref:
                pad_b = y.new_zeros(B_ref - Bx, T_ref, Dx)
                y = torch.cat([y, pad_b], dim=0)

            return y

        swap_raw = str(getattr(self, "oracle_swap_source_controls", "none") or "none").lower()
        swap_tokens = {tok.strip() for tok in swap_raw.replace("+", ",").split(",") if tok.strip()}
        swap_pitch = bool({"all", "source", "pitch", "period"} & swap_tokens)
        swap_fc = bool({"all", "source", "frame_corr", "vuv"} & swap_tokens)
        swap_gain = bool({"all", "source", "gain", "c0"} & swap_tokens)
        swap_ceps = bool({"all", "vocoder", "ceps"} & swap_tokens)

        ceps_vocoder = ceps_target if swap_ceps else ceps_hat
        if (not swap_ceps) and swap_gain:
            try:
                ceps_vocoder = ceps_hat.clone()
                ceps_vocoder[..., 0:1] = ceps_target[..., 0:1]
            except Exception:
                ceps_vocoder = ceps_hat
        dnn_pitch_vocoder = dnn_pitch if swap_pitch else dnn_pitch_hat
        frame_corr_vocoder = frame_corr if swap_fc else frame_corr_hat
        frame_corr_in_vocoder = torch.clamp(frame_corr_vocoder, -0.8, 0.5)

        feat_parts = [
            _align_to_ceps(ceps_vocoder, "ceps_vocoder"),
            _align_to_ceps(dnn_pitch_vocoder, "dnn_pitch_vocoder"),
            _align_to_ceps(frame_corr_in_vocoder, "frame_corr_in_vocoder"),
        ]
        if hf_sideband_feat_hash is not None:
            feat_parts.append(_align_to_ceps(hf_sideband_feat_hash, "hf_sideband_feat_hash"))
        if feat_extra is not None:
            feat_parts.append(_align_to_ceps(feat_extra, "feat_extra"))

        fargan_feats_hat = torch.cat(feat_parts, dim=-1)

        # === Vocoder ===
        # 训练/推理统一：周期 override 仅使用预测 dnn_pitch_hat，
        # 避免在训练阶段依赖 GT F0 进行 teacher-forcing 调度。
        if target_len is None:
            target_len = audio.size(-1)
        with torch.no_grad():
            dp_src = dnn_pitch_vocoder
            period_override = 256.0 / torch.pow(2.0, dp_src + 1.5)  # [B,T,1]
            period_override = torch.clamp(period_override.squeeze(-1), 32.0, 255.0)  # [B,T]

            # 仅在"有声 ∧ 非静音"段覆盖周期：
            # 训练/推理统一使用预测 F0/VUV 与生成 mel 能量，避免训练阶段依赖 GT gate。
            try:
                VUV_THR = 0.25
                fc_for_mask = frame_corr_vocoder
                mel_for_energy = mel_used

                vmask = (fc_for_mask > VUV_THR).squeeze(-1)  # [B,T]
                try:
                    mel_energy = mel_for_energy.mean(dim=-1)  # [B,T]
                    sil_thr_db = float(getattr(self, 'silence_energy_thr_db', -40.0))
                    sil_thr_log = sil_thr_db / 10.0
                    vo_mask = (mel_energy > sil_thr_log)
                    vmask = vmask & vo_mask
                except Exception:
                    pass
            except Exception:
                vmask = None
        # Align period_override and vmask to fargan_feats_hat's temporal dimension
        T_vocoder = fargan_feats_hat.size(1)
        if period_override.size(1) != T_vocoder:
            if period_override.size(1) > T_vocoder:
                period_override = period_override[:, :T_vocoder]
            else:
                pad = period_override[:, -1:].expand(-1, T_vocoder - period_override.size(1))
                period_override = torch.cat([period_override, pad], dim=1)
        if vmask is not None and vmask.size(1) != T_vocoder:
            if vmask.size(1) > T_vocoder:
                vmask = vmask[:, :T_vocoder]
            else:
                pad = vmask.new_zeros(vmask.size(0), T_vocoder - vmask.size(1))
                vmask = torch.cat([vmask, pad], dim=1)

        if vmask is not None:
            _period_hat, audio_hat = self.vocoder(
                fargan_feats_hat, target_len=target_len,
                period_override=period_override, override_mask=vmask)
        else:
            _period_hat, audio_hat = self.vocoder(
                fargan_feats_hat, target_len=target_len,
                period_override=period_override)
        audio_hat = audio_hat.squeeze(1)
        voc_int = getattr(self.vocoder, "last_internal_tracks", None)

        # Hash/RVQ 正则项（内容分支 + F0 分支）
        hash_reg_terms: Dict[str, torch.Tensor] = {}
        if self.quantizer_type == "hash":
            if hasattr(self.hash_content, "compute_hash_regularization"):
                try:
                    hash_reg_terms_c = self.hash_content.compute_hash_regularization(
                        hash_logits=hb_ret_c.get("hash_logits"),
                        hash_bits=hb_ret_c.get("hash_bits_clean"),
                        mask=None,
                    )
                    hash_reg_terms.update({f"content_{k}": v for k, v in hash_reg_terms_c.items()})
                except Exception:
                    pass
            if hasattr(self.hash_f0vuv, "compute_hash_regularization"):
                try:
                    hash_reg_terms_f0 = self.hash_f0vuv.compute_hash_regularization(
                        hash_logits=hb_ret.get("hash_logits"),
                        hash_bits=hb_ret.get("hash_bits_clean"),
                        mask=None,
                    )
                    hash_reg_terms.update({f"f0_{k}": v for k, v in hash_reg_terms_f0.items()})
                except Exception:
                    pass

        # 统一提取 Hash/RVQ bitstream。为了训练期 BER 统计，保留
        # clean / noisy 两个版本（如可用）。
        if "hash_bits_clean" in hb_ret_c:
            bits_c_clean = hb_ret_c["hash_bits_clean"]
            bits_c_noisy = hb_ret_c.get("hash_bits_noisy", bits_c_clean)
        else:
            bits_c_clean = hb_ret_c.get("bits_clean")
            bits_c_noisy = hb_ret_c.get("bits_noisy", bits_c_clean)
        if "hash_bits_clean" in hb_ret:
            bits_f_clean = hb_ret["hash_bits_clean"]
            bits_f_noisy = hb_ret.get("hash_bits_noisy", hb_ret.get("bits_noisy", bits_f_clean))
        else:
            bits_f_clean = hb_ret.get("bits_clean")
            bits_f_noisy = hb_ret.get("bits_noisy", bits_f_clean)

        # 统计量分支 bits（Hash / RVQ 兼容）。
        if "hash_bits_clean" in hb_stats:
            bits_s_clean = hb_stats["hash_bits_clean"]
            bits_s_noisy = hb_stats.get("hash_bits_noisy", bits_s_clean)
        else:
            bits_s_clean = hb_stats.get("bits_clean")
            bits_s_noisy = hb_stats.get("bits_noisy", bits_s_clean)

        out_dict = {
            "audio_hat": audio_hat,
            "fargan_feats_hat": fargan_feats_hat,
            "period_vocoder": _period_hat,
            "period_override_vocoder": period_override,
            "ceps": ceps_target,
            "ceps_hat": ceps_hat,
            "ceps_vocoder": ceps_vocoder,
            "mel": mel,
            "mel_hat": mel_hat,
            "mel_hat_refined": mel_used,  # L2H细化后的mel，修复梯度断流
            # 通过 stats bits 解码得到的全局 mel 统计量（brightness/contrast）。
            # 这里保留梯度以便在训练中使用 lambda_mel_stats 等损失直接
            # 约束 hash_content_stats 的输出；调试/离线路径可将其视作
            # 只读张量，不会对梯度造成影响。
            "mel_mean_hat": mean_hat,
            "mel_std_hat": std_hat,
            "l2h_resid": getattr(self.deco_l2h_refiner, "last_resid", None) if self.enable_l2h else None,
            "l2h_vuv_prob": getattr(self.deco_l2h_refiner, "last_vuv_prob", None) if self.enable_l2h else None,
            "l2h_mask_harm": getattr(self.deco_l2h_refiner, "last_mask_harm", None) if self.enable_l2h else None,
            "dnn_pitch": dnn_pitch,
            # 原始 decoder 输出的 F0（未经标定），便于调试对比
            "dnn_pitch_raw": dnn_pitch_raw,
            # 标定后的 F0，用于 loss 与 vocoder / period_override
            "dnn_pitch_calib": dp_calib,
            "dnn_pitch_hat": dnn_pitch_hat,
            "dnn_pitch_vocoder": dnn_pitch_vocoder,
            "frame_corr": frame_corr,
            # 原始 frame_corr_hat（由 vuv_logits 派生），仅作调试使用
            "frame_corr_hat_raw": frame_corr_hat_raw,
            # 标定后的 frame_corr_hat，用于 VUV CE / gate / probes
            "frame_corr_hat": frame_corr_hat,
            "frame_corr_vocoder": frame_corr_vocoder,
            # 原始 VUV logits（供简化版 VUV BCE / VUV 相关 loss 使用）
            "vuv_logits": vuv_logits,
            # 内容分支符号重构（hash路径）：在连续 latent s_flat 上监督 Hash 重建
            "tokens": s_flat.detach(),
            "tokens_hat": s_hat_flat,
            # 便于可视化的空间 token 图：
            #   s_map: Hash 编码前 VMamba encoder 输出 [B,C,H,W]
            #   s_hat: Hash 解码后、送入 VMamba decoder 的特征 [B,C,H,W]
            "tokens_map": s_map.detach(),
            "tokens_hat_map": s_hat.detach(),
            # F0 分支 Hash 重构：直接在连续 latent z_fv 上监督
            "tokens_f0": z_fv.detach(),
            "tokens_f0_hat": z_fv_hat,
            "hash_bits_clean": bits_c_clean,
            "hash_bits_noisy": bits_c_noisy,
            "hash_reg_terms": hash_reg_terms,
            # 统计量哈希中间量（便于调试/分析）。对于 RVQ stats，hash_logits_stats
            # 可能不存在，此时保持键缺失即可。
            "hash_logits_stats": hb_stats.get("hash_logits"),
            "hash_bits_stats": bits_s_clean,
            "hash_bits_stats_noisy": bits_s_noisy,
            # F0 分支 hash 中间量（便于调试/分析）
            "f0_hash_bits_clean": bits_f_clean,
            "f0_hash_bits_noisy": bits_f_noisy,
            # Semantic latents for loss (clean vs channel-affected)
            "z_content": s_flat.detach(),
            "z_content_hat": s_hat_flat,
            "z_f0": z_fv.detach(),
            "z_f0_hat": z_fv_hat,
            # 便于离线 decode/debug 精确复原 VMamba token 网格形状
            "T": T,
            "F_mel": F_mel,
            "hw": (H, W),
            # CSI summary used during training; first dim typically encodes
            # an SNR proxy,便于在 DBG_TRAIN_BER 中打印 batch 级 SNR。
            "csi_vec": csi_vec.detach(),
        }
        if isinstance(voc_int, dict):
            for _k_src, _k_dst in (
                ("pitch_gain_mean", "vocoder_pitch_gain_mean"),
                ("fwc0_rms", "vocoder_fwc0_rms"),
                ("skip_rms", "vocoder_skip_rms"),
                ("sig_core_rms", "vocoder_sig_core_rms"),
                ("sig_out_rms", "vocoder_sig_out_rms"),
            ):
                _v = voc_int.get(_k_src, None)
                if isinstance(_v, torch.Tensor):
                    out_dict[_k_dst] = _v
        # SR base 分支输出（若已计算）：用于在 RVQ 路径上实现 successive refinement，
        # 便于在损失中对“骨架”F0/VUV 单独施加约束。
        if dnn_pitch_hat_base is not None and frame_corr_hat_base is not None:
            out_dict["dnn_pitch_hat_base"] = dnn_pitch_hat_base
            out_dict["frame_corr_hat_base"] = frame_corr_hat_base
        if hash_logits_c is not None:
            out_dict["hash_logits"] = hash_logits_c
        if "hash_logits" in hb_ret:
            out_dict["f0_hash_logits"] = hb_ret["hash_logits"]

        # RVQ VQ loss（仅在 quantizer_type=="rvq" 时存在）
        vq_loss_c = hb_ret_c.get("vq_loss", None)
        vq_loss_f = hb_ret.get("vq_loss", None)
        vq_loss_s = hb_stats.get("vq_loss", None)
        vq_loss_total: Optional[torch.Tensor] = None
        if isinstance(vq_loss_c, torch.Tensor):
            vq_loss_total = vq_loss_c
        if isinstance(vq_loss_f, torch.Tensor):
            vq_loss_total = vq_loss_f if vq_loss_total is None else vq_loss_total + vq_loss_f
        if isinstance(vq_loss_s, torch.Tensor):
            vq_loss_total = vq_loss_s if vq_loss_total is None else vq_loss_total + vq_loss_s
            out_dict["vq_loss_stats"] = vq_loss_s

        if isinstance(vq_loss_total, torch.Tensor):
            out_dict["vq_loss"] = vq_loss_total
            if isinstance(vq_loss_c, torch.Tensor):
                out_dict["vq_loss_content"] = vq_loss_c
            if isinstance(vq_loss_f, torch.Tensor):
                out_dict["vq_loss_f0"] = vq_loss_f

        # RVQ 索引使用率 / 熵 / 困惑度诊断（content 分支 + F0 分支）。
        # 仅在 quantizer_type=="rvq" 且底层瓶颈模块为 RVQBottleneck 时启用。
        if self.quantizer_type == "rvq":
            try:
                def _rvq_diag(codes: torch.Tensor, rvq_mod: RVQBottleneck, prefix: str) -> None:
                    if not isinstance(rvq_mod, RVQBottleneck):
                        return
                    if not isinstance(codes, torch.Tensor) or codes.numel() == 0:
                        return
                    stage_bits = getattr(rvq_mod, "stage_bits", None)
                    if not isinstance(stage_bits, list) or len(stage_bits) != codes.size(-1):
                        return

                    Bc, Tc, Nq = codes.shape
                    H_list: list[torch.Tensor] = []
                    usage_list: list[torch.Tensor] = []
                    perp_ratio_list: list[torch.Tensor] = []

                    for qi, b in enumerate(stage_bits):
                        b_int = int(b)
                        if b_int <= 0:
                            continue
                        codes_stage = codes[..., qi].reshape(-1)
                        if codes_stage.numel() == 0:
                            continue
                        K = int(1 << b_int)
                        counts = torch.bincount(codes_stage, minlength=K).to(torch.float32)
                        total = counts.sum()
                        if total <= 0:
                            continue
                        p = counts / total
                        p = p[p > 0]
                        if p.numel() == 0:
                            continue
                        H = -(p * torch.log2(p)).sum()              # bits
                        usage = (counts > 0).to(torch.float32).mean()  # 0..1
                        perp_ratio = (2.0 ** H) / float(K)          # 0..1，有效困惑度占比
                        H_list.append(H)
                        usage_list.append(usage)
                        perp_ratio_list.append(perp_ratio)

                    if not H_list:
                        return

                    H_total = torch.stack(H_list).sum()
                    usage_mean = torch.stack(usage_list).mean()
                    perp_ratio_mean = torch.stack(perp_ratio_list).mean()

                    try:
                        out_dict[f"{prefix}_H"] = float(H_total.item())
                        out_dict[f"{prefix}_usage"] = float(usage_mean.item())
                        out_dict[f"{prefix}_perp"] = float(perp_ratio_mean.item())
                    except Exception:
                        # 若 item() 失败，则退回为原 tensor（后续 compute_losses 中会再次转换为 float）
                        out_dict[f"{prefix}_H"] = H_total
                        out_dict[f"{prefix}_usage"] = usage_mean
                        out_dict[f"{prefix}_perp"] = perp_ratio_mean

                codes_c = hb_ret_c.get("codes", None)
                if isinstance(codes_c, torch.Tensor):
                    _rvq_diag(codes_c, self.hash_content, "rvq_c")

                codes_f = hb_ret.get("codes", None)
                if isinstance(codes_f, torch.Tensor):
                    _rvq_diag(codes_f, self.hash_f0vuv, "rvq_f")
            except Exception:
                # 诊断失败不应影响主训练路径
                pass

        # HF 侧通道特征（可选）
        if hf_sideband_feat_hash is not None:
            out_dict["hf_sideband"] = hf_sideband_feat_hash
        return out_dict


DualBranchMelJSCC = DualBranchBarkJSCC


if __name__ == "__main__":  # 简单自测
    B, L, T = 2, 16000, 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualBranchBarkJSCC(device=device)
    audio = torch.randn(B, L, device=device)
    fargan_feats = torch.randn(B, T, 36, device=device)

    class DummyChannelSim:
        def sample_csi(self, B, T, channel="fading", snr_min_db=-5.0, snr_max_db=15.0):
            csi = {
                "snr_proxy": torch.zeros(B, device=device),
                "time_selectivity": torch.zeros(B, device=device),
                "freq_selectivity": torch.zeros(B, device=device),
                "los_ratio": torch.zeros(B, device=device),
            }
            amp_t = torch.ones(B, T, device=device)
            snr_db_t = torch.zeros(B, T, device=device)
            return csi, amp_t, snr_db_t

        def apply(self, z, amp_t, snr_db_t):
            return z

    dummy_channel = DummyChannelSim()
    out = model(audio, fargan_feats, dummy_channel, -5.0, 15.0, target_len=L)
    print("audio_hat", out["audio_hat"].shape, "ceps_hat", out["ceps_hat"].shape)
