#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-STFT Discriminator and Adversarial Wave Loss (FIXED VERSION)

修复问题：
1. get_discriminator_parameters() 现在包含 MPD 参数
2. 添加 gate_floor 防止梯度消失
3. 添加梯度诊断工具
4. 改进 feature matching loss 的稳定性
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class STFTSubDiscriminator(nn.Module):
    """单尺度 STFT 判别子网络"""

    def __init__(self, in_channels: int = 1, base_channels: int = 32, use_spectral_norm: bool = True):
        super().__init__()
        c = base_channels
        layers = []

        conv1 = nn.Conv2d(in_channels, c, kernel_size=(3, 9), stride=(1, 1), padding=(1, 4))
        if use_spectral_norm:
            conv1 = nn.utils.spectral_norm(conv1)
        layers.append(conv1)
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        for dilation_t in (1, 2, 4):
            convk = nn.Conv2d(
                c, c,
                kernel_size=(3, 9),
                stride=(2, 1),
                padding=(1, 4 * dilation_t),
                dilation=(1, dilation_t),
            )
            if use_spectral_norm:
                convk = nn.utils.spectral_norm(convk)
            layers.append(convk)
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.feature_layers = nn.ModuleList(layers)
        outc = nn.Conv2d(c, 1, kernel_size=(3, 3), stride=1, padding=1)
        if use_spectral_norm:
            outc = nn.utils.spectral_norm(outc)
        self.out_conv = outc

    def forward(self, mag: torch.Tensor) -> List[torch.Tensor]:
        x = mag.unsqueeze(1)
        feats = []
        h = x

        for layer in self.feature_layers:
            h = layer(h)
            feats.append(h)

        score = self.out_conv(h)
        feats.append(score)
        return feats


class WaveDiscriminator(nn.Module):
    """多尺度 STFT 判别器"""

    def __init__(
        self,
        fft_sizes: Optional[List[int]] = None,
        hop_factors: int = 4,
        base_channels: int = 32,
        use_spectral_norm: bool = True,
        sample_rate: int = 16000,
        roi_low_hz: Optional[float] = None,
        preemph: bool = False,
        preemph_coef: float = 0.97,
    ):
        super().__init__()
        if fft_sizes is None:
            fft_sizes = [1024, 512, 256]

        self.fft_sizes = list(fft_sizes)
        self.hop_factors = hop_factors
        self.sample_rate = sample_rate
        self.roi_low_hz = roi_low_hz
        self.preemph = preemph
        self.preemph_coef = float(preemph_coef)
        self.sub_discriminators = nn.ModuleList(
            [STFTSubDiscriminator(in_channels=1, base_channels=base_channels, 
                                  use_spectral_norm=use_spectral_norm) 
             for _ in self.fft_sizes]
        )

    @staticmethod
    def _stft_mag_for_disc(x: torch.Tensor, fft_size: int, hop_size: int, win_length: int) -> torch.Tensor:
        x32 = x.to(torch.float32)
        window = torch.hann_window(win_length, device=x32.device, dtype=torch.float32)
        spec = torch.stft(
            x32, n_fft=fft_size, hop_length=hop_size, win_length=win_length,
            window=window, return_complex=True,
        )
        mag = torch.abs(spec).clamp_min(1e-6)
        # 放大梯度：使用 sqrt(|X|) 而非 log1p(|X|)，并做 unit-mean 归一化
        mag = torch.sqrt(mag)
        m = mag.mean(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        mag = mag / m
        return mag

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        assert x.dim() == 2, f"WaveDiscriminator expects [B, T] or [B,1,T], got {x.shape}"

        outputs: List[List[torch.Tensor]] = []
        for fs, sub_disc in zip(self.fft_sizes, self.sub_discriminators):
            hop = max(1, fs // self.hop_factors)
            win_len = fs
            if self.preemph and x.size(-1) > 1:
                xn = x
                y = xn.clone()
                y[:, 1:] = xn[:, 1:] - self.preemph_coef * xn[:, :-1]
                x_in = y
            else:
                x_in = x
            mag = self._stft_mag_for_disc(x_in, fs, hop, win_len)
            if isinstance(self.roi_low_hz, (int, float)) and self.roi_low_hz is not None and self.roi_low_hz > 0:
                F = mag.size(1)
                freqs = torch.linspace(0, self.sample_rate/2, F, device=mag.device, dtype=mag.dtype)
                keep = freqs >= float(self.roi_low_hz)
                if keep.any():
                    mag = mag[:, keep, :]
            feats = sub_disc(mag)
            outputs.append(feats)
        return outputs


class PeriodDiscriminator(nn.Module):
    """周期判别器"""
    def __init__(self, period: int, base_channels: int = 16, use_spectral_norm: bool = True) -> None:
        super().__init__()
        assert period >= 2
        self.p = int(period)
        C = base_channels
        def sn(layer):
            return nn.utils.spectral_norm(layer) if use_spectral_norm else layer
        self.layers = nn.ModuleList([
            sn(nn.Conv1d(self.p, C, kernel_size=5, stride=1, padding=2)),
            sn(nn.Conv1d(C, C, kernel_size=5, stride=2, padding=2)),
            sn(nn.Conv1d(C, C*2, kernel_size=5, stride=2, padding=2)),
            sn(nn.Conv1d(C*2, C*2, kernel_size=5, stride=2, padding=2)),
        ])
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.head = sn(nn.Conv1d(C*2, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        B, T = x.shape
        p = self.p
        pad = (p - (T % p)) % p
        if pad > 0:
            x = F.pad(x, (0, pad), mode='reflect')
        T_new = x.size(-1)
        x = x.view(B, T_new // p, p).transpose(1, 2)
        feats: List[torch.Tensor] = []
        h = x
        for layer in self.layers:
            h = self.act(layer(h))
            feats.append(h)
        logit = self.head(h)
        feats.append(logit)
        return feats


class MultiPeriodDiscriminator(nn.Module):
    """多周期判别器"""
    def __init__(self, periods: List[int], base_channels: int = 16, use_spectral_norm: bool = True) -> None:
        super().__init__()
        self.periods = list(periods)
        self.discs = nn.ModuleList([
            PeriodDiscriminator(p, base_channels=base_channels, use_spectral_norm=use_spectral_norm)
            for p in self.periods
        ])

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        outs: List[List[torch.Tensor]] = []
        for D in self.discs:
            outs.append(D(x))
        return outs


def feature_matching_loss(
    scores_real: List[List[torch.Tensor]],
    scores_gen: List[List[torch.Tensor]],
    normalize: bool = True,
) -> torch.Tensor:
    """特征匹配损失（健壮对齐版）

    - 对每个尺度、每一层特征先对齐维度与空间/时间长度（裁剪到最小），
      避免 STFT 帧数舍入和卷积边界差异导致的形状不匹配。
    - 用真实特征的统计量对白化，数值稳定。
    """
    device = scores_gen[0][0].device
    loss_feat = torch.tensor(0.0, device=device)
    num_scales = min(len(scores_real), len(scores_gen))

    for k in range(num_scales):
        feats_g = scores_gen[k]
        feats_r = scores_real[k]
        num_layers = min(len(feats_g), len(feats_r)) - 1  # 排除最后的 score
        if num_layers <= 0:
            continue
        for i in range(num_layers):
            fg = feats_g[i]
            fr = feats_r[i]

            # 对齐维度
            if fg.dim() != fr.dim():
                while fg.dim() < fr.dim():
                    fg = fg.unsqueeze(-1)
                while fr.dim() < fg.dim():
                    fr = fr.unsqueeze(-1)

            # 对齐空间/时间尺寸（裁剪到公共最小）
            if fg.dim() >= 4:
                H = min(fg.size(-2), fr.size(-2))
                W = min(fg.size(-1), fr.size(-1))
                fg = fg[..., :H, :W]
                fr = fr[..., :H, :W]
                reduce_dims = (-2, -1)
            elif fg.dim() == 3:
                L = min(fg.size(-1), fr.size(-1))
                fg = fg[..., :L]
                fr = fr[..., :L]
                reduce_dims = (-1,)
            else:
                reduce_dims = tuple(range(1, fg.dim()))

            if normalize:
                mu = fr.detach().mean(dim=reduce_dims, keepdim=True)
                std = fr.detach().std(dim=reduce_dims, keepdim=True).clamp_min(5e-2)
                fg_n = (fg - mu) / std
                fr_n = (fr - mu) / std
                diff = (fg_n - fr_n).abs()
            else:
                diff = (fg - fr).abs()

            diff = diff.clamp(max=10.0)
            # 高层稍大权重
            layer_weight = 1.0 + 0.5 * (i / max(num_layers - 1, 1))
            loss_feat = loss_feat + layer_weight * diff.mean()

    return loss_feat / max(1, num_scales)


def discriminator_loss(scores_real: List[List[torch.Tensor]],
                      scores_gen: List[List[torch.Tensor]]) -> torch.Tensor:
    """判别器损失（LSGAN）"""
    loss_d = torch.tensor(0.0, device=scores_real[0][-1].device)
    num_discs = len(scores_real)

    for k in range(num_discs):
        real_score = scores_real[k][-1]
        gen_score = scores_gen[k][-1]
        loss_real = F.mse_loss(real_score, torch.ones_like(real_score))
        loss_gen = F.mse_loss(gen_score, torch.zeros_like(gen_score))
        loss_d = loss_d + (loss_real + loss_gen) / num_discs

    return loss_d


def generator_adversarial_loss(scores_gen: List[List[torch.Tensor]]) -> torch.Tensor:
    """生成器对抗损失"""
    loss_g = torch.tensor(0.0, device=scores_gen[0][-1].device)
    num_discs = len(scores_gen)

    for k in range(num_discs):
        gen_score = scores_gen[k][-1]
        loss_g = loss_g + F.mse_loss(gen_score, torch.ones_like(gen_score)) / num_discs

    return loss_g


class AdversarialWaveLoss(nn.Module):
    """
    完整的对抗训练波形损失计算组件（修复版）
    
    修复：
    1. get_discriminator_parameters() 现在包含 MPD 参数
    2. 添加梯度诊断方法
    3. 添加判别器状态监控
    """

    def __init__(self,
                 fft_sizes: Optional[List[int]] = None,
                 hop_factors: int = 4,
                 base_channels: int = 32,
                 use_spectral_norm: bool = True,
                 sample_rate: int = 16000,
                 roi_low_hz: Optional[float] = None,
                 preemph: bool = False,
                 preemph_coef: float = 0.97,
                 mpd_periods: Optional[List[int]] = None,
                 mpd_base_channels: int = 16,
                 mpd_use_spectral_norm: bool = True,
                 feature_match_weight: float = 10.0,
                 adversarial_weight: float = 1.0):
        super().__init__()

        self.discriminator = WaveDiscriminator(
            fft_sizes=fft_sizes,
            hop_factors=hop_factors,
            base_channels=base_channels,
            use_spectral_norm=use_spectral_norm,
            sample_rate=sample_rate,
            roi_low_hz=roi_low_hz,
            preemph=preemph,
            preemph_coef=preemph_coef
        )

        self.mpd: Optional[MultiPeriodDiscriminator] = None
        if mpd_periods is not None and len(mpd_periods) > 0:
            self.mpd = MultiPeriodDiscriminator(
                periods=mpd_periods,
                base_channels=mpd_base_channels,
                use_spectral_norm=mpd_use_spectral_norm,
            )

        self.feature_match_weight = feature_match_weight
        self.adversarial_weight = adversarial_weight
        
        # 训练状态监控
        self.register_buffer('disc_real_mean', torch.tensor(0.5))
        self.register_buffer('disc_fake_mean', torch.tensor(0.5))
        self.register_buffer('update_count', torch.tensor(0))

    def discriminator_step(self,
                          audio_real: torch.Tensor,
                          audio_gen: torch.Tensor) -> Dict[str, torch.Tensor]:
        """判别器训练步骤"""
        scores_gen = self.discriminator(audio_gen.detach())
        scores_real = self.discriminator(audio_real)
        scores_gen_m = scores_gen
        scores_real_m = scores_real
        
        if self.mpd is not None:
            mpd_real = self.mpd(audio_real)
            mpd_gen = self.mpd(audio_gen.detach())  # 确保 detach
            scores_real_m = scores_real + mpd_real
            scores_gen_m = scores_gen + mpd_gen

        loss_d = discriminator_loss(scores_real_m, scores_gen_m)
        
        # 更新监控统计
        with torch.no_grad():
            real_scores = torch.cat([s[-1].flatten() for s in scores_real_m])
            fake_scores = torch.cat([s[-1].flatten() for s in scores_gen_m])
            ema = 0.99
            self.disc_real_mean = ema * self.disc_real_mean + (1 - ema) * real_scores.mean()
            self.disc_fake_mean = ema * self.disc_fake_mean + (1 - ema) * fake_scores.mean()
            self.update_count += 1

        return {
            'discriminator_loss': loss_d,
            'scores_real': scores_real,
            'scores_gen': scores_gen,
            'disc_real_mean': self.disc_real_mean.item(),
            'disc_fake_mean': self.disc_fake_mean.item(),
        }

    def generator_step(self,
                      audio_real: torch.Tensor,
                      audio_gen: torch.Tensor) -> Dict[str, torch.Tensor]:
        """生成器训练步骤"""
        with torch.no_grad():
            scores_real = self.discriminator(audio_real)
            if self.mpd is not None:
                mpd_real = self.mpd(audio_real)

        scores_gen = self.discriminator(audio_gen)
        if self.mpd is not None:
            mpd_gen = self.mpd(audio_gen)
            scores_real = scores_real + mpd_real
            scores_gen = scores_gen + mpd_gen

        loss_fm = feature_matching_loss(scores_real, scores_gen)
        loss_adv = generator_adversarial_loss(scores_gen)
        loss_total = self.feature_match_weight * loss_fm + self.adversarial_weight * loss_adv

        return {
            'total_adversarial_loss': loss_total,
            'feature_matching_loss': loss_fm,
            'adversarial_loss': loss_adv,
            'scores_real': scores_real,
            'scores_gen': scores_gen
        }

    def get_discriminator_parameters(self):
        """
        获取判别器参数（修复版：包含 MPD）
        """
        params = list(self.discriminator.parameters())
        if self.mpd is not None:
            params.extend(self.mpd.parameters())
        return params
    
    def get_discriminator_state(self) -> Dict[str, float]:
        """获取判别器状态（用于诊断）"""
        return {
            'disc_real_mean': self.disc_real_mean.item(),
            'disc_fake_mean': self.disc_fake_mean.item(),
            'update_count': self.update_count.item(),
            'is_collapsed': abs(self.disc_real_mean.item() - self.disc_fake_mean.item()) < 0.1,
        }
    
    def reset_discriminator(self):
        """重置判别器权重（当判别器饱和时使用）"""
        for m in self.discriminator.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if self.mpd is not None:
            for m in self.mpd.modules():
                if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        self.disc_real_mean.fill_(0.5)
        self.disc_fake_mean.fill_(0.5)
        self.update_count.zero_()
        print("[Adv] Discriminator reset!")


def create_adversarial_wave_loss(
    fft_sizes: Optional[List[int]] = None,
    hop_factors: int = 4,
    base_channels: int = 32,
    use_spectral_norm: bool = True,
    sample_rate: int = 16000,
    roi_low_hz: Optional[float] = None,
    preemph: bool = False,
    preemph_coef: float = 0.97,
    mpd_periods: Optional[List[int]] = None,
    mpd_base_channels: int = 16,
    mpd_use_spectral_norm: bool = True,
    feature_match_weight: float = 10.0,
    adversarial_weight: float = 1.0
) -> AdversarialWaveLoss:
    """工厂函数"""
    return AdversarialWaveLoss(
        fft_sizes=fft_sizes,
        hop_factors=hop_factors,
        base_channels=base_channels,
        use_spectral_norm=use_spectral_norm,
        sample_rate=sample_rate,
        roi_low_hz=roi_low_hz,
        preemph=preemph,
        preemph_coef=preemph_coef,
        mpd_periods=mpd_periods,
        mpd_base_channels=mpd_base_channels,
        mpd_use_spectral_norm=mpd_use_spectral_norm,
        feature_match_weight=feature_match_weight,
        adversarial_weight=adversarial_weight
    )


# ==============================================================================
# 训练脚本补丁
# ==============================================================================

def soft_gate_and_crop_fixed(
    y: torch.Tensor,
    fc: Optional[torch.Tensor],
    start: int,
    crop_ratio: float = 0.8,
    vuv_threshold: float = 0.3,
    vuv_k: float = 10.0,
    gate_floor: float = 0.2,  # ★ 关键修复：最小权重
    use_gate: bool = True,
) -> torch.Tensor:
    """
    修复版 voicing gate + crop
    
    关键改进：
    - gate_floor = 0.2：即使 unvoiced 区域也保留 20% 的信号权重
    - 防止 voiced ratio 崩塌导致的梯度消失
    """
    if use_gate and fc is not None:
        th = vuv_threshold
        ksig = vuv_k
        m_f = torch.sigmoid(ksig * (fc.squeeze(-1) - th))  # [B, Tf]
        
        # ★ 关键修复：添加 floor
        m_f = m_f * (1.0 - gate_floor) + gate_floor  # m ∈ [gate_floor, 1.0]
        
        hop = 160
        m = m_f.repeat_interleave(hop, dim=1)
        L = y.size(-1)
        if m.size(1) < L:
            pad = m[:, -1:].expand(-1, L - m.size(1))
            m = torch.cat([m, pad], dim=1)
        m = m[:, :L]
        
        y = y * m.detach()
    
    if 0.0 < crop_ratio < 1.0:
        L = y.size(-1)
        crop_len = max(1, int(L * crop_ratio))
        if crop_len < L:
            y = y[:, start:start + crop_len]
    
    return y


# ==============================================================================
# 使用说明
# ==============================================================================

"""
立即应用的修复步骤：

1. 替换 multi_stft_discriminator.py 中的 get_discriminator_parameters：
   
   def get_discriminator_parameters(self):
       params = list(self.discriminator.parameters())
       if self.mpd is not None:
           params.extend(self.mpd.parameters())
       return params

2. 在训练脚本中添加 gate_floor：
   
   在 _soft_gate_and_crop 函数中，修改：
   
   # 原代码：
   # m_f = torch.sigmoid(ksig * (fc.squeeze(-1) - th))
   # y = y * m.detach()
   
   # 改为：
   gate_floor = 0.2  # 最小 20% 权重
   m_f = torch.sigmoid(ksig * (fc.squeeze(-1) - th))
   m_f = m_f * (1.0 - gate_floor) + gate_floor  # m ∈ [0.2, 1.0]
   y = y * m.detach()

3. 调整损失权重：
   
   --lambda_adv 0.1 --lambda_fm 1.0  # FM 权重更大
   
   或者直接在 AdversarialWaveLoss 初始化时：
   feature_match_weight=10.0, adversarial_weight=1.0

4. 添加判别器状态监控：
   
   if global_step % 1000 == 0 and adv is not None:
       state = adv.get_discriminator_state()
       print(f"[Adv State] real={state['disc_real_mean']:.3f}, "
             f"fake={state['disc_fake_mean']:.3f}, "
             f"collapsed={state['is_collapsed']}")
       if state['is_collapsed']:
           adv.reset_discriminator()
           optimizer_disc = torch.optim.AdamW(adv.get_discriminator_parameters(), lr=cfg.lr)
"""


if __name__ == "__main__":
    # 测试
    B, T = 2, 16000
    audio_real = torch.randn(B, T)
    audio_gen = torch.randn(B, T, requires_grad=True)
    
    adv = create_adversarial_wave_loss(
        fft_sizes=[1024, 512, 256],
        mpd_periods=[2, 3, 5],
        feature_match_weight=10.0,
        adversarial_weight=1.0,
    )
    
    # 测试判别器参数
    params = adv.get_discriminator_parameters()
    print(f"Total discriminator params: {sum(p.numel() for p in params)}")
    
    # 测试前向
    d_out = adv.discriminator_step(audio_real, audio_gen)
    print(f"D loss: {d_out['discriminator_loss'].item():.4f}")
    
    g_out = adv.generator_step(audio_real, audio_gen)
    print(f"G loss: {g_out['total_adversarial_loss'].item():.4f}")
    print(f"  - adv: {g_out['adversarial_loss'].item():.4f}")
    print(f"  - fm: {g_out['feature_matching_loss'].item():.4f}")
    
    # 测试梯度
    g_out['total_adversarial_loss'].backward()
    grad_norm = audio_gen.grad.norm().item()
    print(f"Gradient norm on audio_gen: {grad_norm:.6f}")
    
    # 测试状态
    print(f"Discriminator state: {adv.get_discriminator_state()}")
