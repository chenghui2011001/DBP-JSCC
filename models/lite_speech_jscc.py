#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LiteSpeechJSCC: Lyra2风格轻量级语音JSCC系统

基于用户提供的完整技术方案实现：
- "大老师+小学生+JSCC+Hash"架构
- 参数量控制在2-5M，支持端侧部署
- 1.2-1.5 kbps极低码率语音编解码

Key Features:
- EncoderLite: Conv1d + GRU轻量级编码器
- JSCCEncoder/Decoder: CSI感知的模拟码
- HashBottleneck: bit-level JSCC瓶颈
- DecoderLite: 对称的轻量级解码器
- VocoderLite: 复用FARGAN组件
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Dict, Any
import math

# 使用本地复制的成熟组件
from .hash_bottleneck import HashBottleneck
from .vocoder_decoder import FARGANDecoder


# ===== 信道模拟工具函数 =====
def channel_awgn(s: torch.Tensor, snr_db: float) -> torch.Tensor:
    """AWGN信道模拟"""
    noise_power = 10 ** (-snr_db / 10)
    noise = torch.randn_like(s) * math.sqrt(noise_power)
    return s + noise


def channel_rayleigh(s: torch.Tensor, snr_db: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rayleigh衰落信道模拟"""
    # Rayleigh增益 |h|^2 ~ Exponential(1)
    h_real = torch.randn_like(s)
    h_imag = torch.randn_like(s)
    h = torch.sqrt(h_real**2 + h_imag**2)  # Rayleigh分布

    # 信道输出
    y_signal = h * s

    # AWGN噪声
    noise_power = 10 ** (-snr_db / 10)
    noise = torch.randn_like(s) * math.sqrt(noise_power)
    y = y_signal + noise

    return y, h


def apply_bit_noise(bits: torch.Tensor,
                   flip_prob: float = 0.1,
                   block_drop_prob: float = 0.05) -> torch.Tensor:
    """对bit序列应用噪声"""
    B, T, K = bits.shape
    noisy_bits = bits.clone()

    # 随机bit翻转
    flip_mask = torch.rand(B, T, K, device=bits.device) < flip_prob
    noisy_bits = torch.where(flip_mask, -noisy_bits, noisy_bits)

    # 块擦除（模拟丢包）
    if block_drop_prob > 0:
        for b in range(B):
            if torch.rand(1).item() < block_drop_prob:
                # 随机擦除一个时间块
                start = torch.randint(0, max(1, T-4), (1,)).item()
                end = min(T, start + 4)
                noisy_bits[b, start:end, :] = 0  # 擦除为0

    return noisy_bits


# ===== 核心轻量级组件 =====

class EncoderLite(nn.Module):
    """轻量级编码器：特征+CSI → latent"""

    def __init__(self, feat_dim=36, d_csi=3, d_z=16, hidden=80, conv_ch=64):
        super().__init__()
        in_dim = feat_dim + d_csi

        # Conv1d局部特征提取
        self.conv1 = nn.Conv1d(in_dim, conv_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(conv_ch, conv_ch, kernel_size=3, padding=1)

        # GRU时序建模
        self.gru = nn.GRU(
            input_size=conv_ch,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
        )

        # 输出投影
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_z),
        )

        self.norm = nn.LayerNorm(d_z)

    def forward(self, x_feat, csi):
        B, T, Fdim = x_feat.shape
        csi_t = csi.unsqueeze(1).expand(B, T, -1)  # [B,T,d_csi]
        x = torch.cat([x_feat, csi_t], dim=-1)     # [B,T,F+d_csi]

        # Conv1d特征提取
        x = x.transpose(1, 2)      # [B,C,T]
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.transpose(1, 2)      # [B,T,conv_ch]

        # GRU时序建模
        h, _ = self.gru(x)         # [B,T,hidden]

        # 输出投影
        z = self.proj(h)           # [B,T,d_z]
        z = self.norm(z)
        z = torch.tanh(z)          # 限幅到 [-1,1]
        return z


class TemporalMixBlock(nn.Module):
    """轻量级时序混合模块（TCN 风格 depthwise Conv1d）。

    - 输入/输出形状均为 [B,T,C]，便于直接插在 JSCC 符号序列上。
    - 使用 depthwise Conv1d + pointwise Conv1d 做局部时序卷积，
      receptive field 小但足以在 JSCC 符号上注入时间冗余。
    - 采用残差 + LayerNorm，整体为一个稳定的轻量模块。
    """

    def __init__(self, channels: int, kernel_size: int = 5, dropout: float = 0.0) -> None:
        super().__init__()
        ks = int(kernel_size)
        if ks % 2 == 0:
            ks += 1  # 保证对称 padding
        padding = ks // 2

        self.dw_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=ks,
            padding=padding,
            groups=channels,
        )
        self.pw_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.norm = nn.LayerNorm(channels)
        # 残差缩放系数，避免过度扰动原始符号
        self.res_scale: float = 0.2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,C]
        if x.dim() != 3:
            return x
        residual = x
        y = x.transpose(1, 2)              # [B,C,T]
        y = self.dw_conv(y)
        y = self.act(y)
        y = self.pw_conv(y)
        y = self.dropout(y)
        y = y.transpose(1, 2)              # [B,T,C]
        out = residual + self.res_scale * y
        out = self.norm(out)
        return out


class JSCCEncoder(nn.Module):
    """JSCC编码器：CSI感知的模拟码 + 轻量时序混合"""

    def __init__(self, d_z: int = 16, d_s: int = 16, d_csi: int = 3, hidden: int = 32,
                 use_temporal_mix: bool = True) -> None:
        super().__init__()
        self.z_proj = nn.Linear(d_z, d_s)
        self.csi_mlp = nn.Sequential(
            nn.Linear(d_csi, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * d_s),
        )
        self.use_temporal_mix = bool(use_temporal_mix)
        self.temporal_mix = TemporalMixBlock(d_s) if self.use_temporal_mix else None

    def forward(self, z: torch.Tensor, csi: torch.Tensor) -> torch.Tensor:
        B, T, _ = z.shape
        h = self.z_proj(z)  # [B,T,d_s]

        # CSI感知的scale/bias
        c = self.csi_mlp(csi)                 # [B,2*d_s]
        scale, bias = c.chunk(2, dim=-1)      # [B,d_s]
        scale = torch.tanh(scale).unsqueeze(1)  # [B,1,d_s]
        bias = bias.unsqueeze(1)

        # 调制
        s = (h + bias) * scale                # [B,T,d_s]

        # 轻量时序混合（TCN 风格），在符号域注入邻域冗余
        if self.temporal_mix is not None:
            s = self.temporal_mix(s)

        # 功率归一化
        power = (s ** 2).mean(dim=(1, 2), keepdim=True) + 1e-6
        s = s / power.sqrt()
        return s


class JSCCDecoder(nn.Module):
    """JSCC解码器：CSI感知的去噪器 + 轻量时序混合"""

    def __init__(self, d_z: int = 16, d_s: int = 16, d_csi: int = 3, hidden: int = 32,
                 use_temporal_mix: bool = True) -> None:
        super().__init__()
        self.csi_mlp = nn.Sequential(
            nn.Linear(d_csi, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_s),
        )
        self.s_proj = nn.Linear(d_s, d_z)
        self.use_temporal_mix = bool(use_temporal_mix)
        self.temporal_mix = TemporalMixBlock(d_s) if self.use_temporal_mix else None

    def forward(self, y: torch.Tensor, csi: torch.Tensor, h_rayleigh: Optional[torch.Tensor] = None) -> torch.Tensor:
        # CSI感知的shrinkage
        gamma = torch.tanh(self.csi_mlp(csi)).unsqueeze(1)  # [B,1,d_s]
        y_d = gamma * y

        # Rayleigh均衡
        if h_rayleigh is not None:
            y_d = y_d / (h_rayleigh + 1e-3)

        # 解码侧时序混合：在 CSI 调制和均衡之后，对符号序列做轻量平滑/恢复
        if self.temporal_mix is not None:
            y_d = self.temporal_mix(y_d)

        z_hat = self.s_proj(y_d)
        return z_hat


class DecoderLite(nn.Module):
    """轻量级解码器：latent+CSI → 特征"""

    def __init__(self, feat_dim=36, d_csi=3, d_z=16, hidden=80, conv_ch=64):
        super().__init__()
        in_dim = d_z + d_csi

        # GRU时序建模
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
        )

        # Conv1d特征重建
        self.conv1 = nn.Conv1d(hidden, conv_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(conv_ch, conv_ch, kernel_size=3, padding=1)

        # 输出投影
        self.proj = nn.Sequential(
            nn.Linear(conv_ch, conv_ch),
            nn.GELU(),
            nn.Linear(conv_ch, feat_dim),
        )
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, z_hat, csi):
        B, T, _ = z_hat.shape
        csi_t = csi.unsqueeze(1).expand(B, T, -1)
        x = torch.cat([z_hat, csi_t], dim=-1)   # [B,T,d_z+d_csi]

        # GRU时序建模
        h, _ = self.gru(x)                      # [B,T,hidden]

        # Conv1d特征重建
        h = h.transpose(1, 2)                   # [B,hidden,T]
        h = F.gelu(self.conv1(h))
        h = F.gelu(self.conv2(h))
        h = h.transpose(1, 2)                   # [B,T,conv_ch]

        # 输出投影
        f = self.proj(h)                        # [B,T,feat_dim]
        f = self.norm(f)
        return f


# ===== 主模型 =====

class LiteSpeechJSCC(nn.Module):
    """
    Lyra2风格轻量级语音JSCC系统

    架构：EncoderLite → JSCCEncoder → Channel → JSCCDecoder → (Hash) → DecoderLite → VocoderLite
    """

    def __init__(self,
                 feat_dim=36,
                 d_csi=3,
                 d_z=16,
                 d_s=16,
                 n_bits=16,
                 hidden=80,
                 hash_method='greedy',
                 device=None):
        super().__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # 轻量级编解码器
        self.enc = EncoderLite(feat_dim, d_csi, d_z, hidden)
        self.jscc_enc = JSCCEncoder(d_z, d_s, d_csi, hidden//2)
        self.jscc_dec = JSCCDecoder(d_z, d_s, d_csi, hidden//2)
        self.dec = DecoderLite(feat_dim, d_csi, d_z, hidden)

        # Hash瓶颈（复用final_version）
        self.hash = HashBottleneck(
            input_dim=d_z,
            hash_bits=n_bits,
            decoder_hidden=128,
            output_dim=d_z,
            hash_method=hash_method,
            channel_type='bsc'
        )

        # Vocoder（复用final_version的FARGAN）
        self.vocoder = FARGANDecoder(
            fargan_subframe_size=40,
            fargan_nb_subframes=4,
            frame_rate_hz=100.0
        )

        # 配置
        self.feat_dim = feat_dim
        self.d_z = d_z
        self.n_bits = n_bits
        self.hidden = hidden

        # 移动到目标设备
        self.to(self.device)

    def forward_continuous(self, x_feat, csi, snr_db, channel_mode="awgn"):
        """连续JSCC模式（Stage2训练用）"""
        # 编码
        z = self.enc(x_feat, csi)
        s = self.jscc_enc(z, csi)

        # 信道模拟
        if channel_mode == "awgn":
            y = channel_awgn(s, snr_db)
            h = None
        else:
            y, h = channel_rayleigh(s, snr_db)

        # 解码
        z_hat = self.jscc_dec(y, csi, h)
        feat_hat = self.dec(z_hat, csi)

        return feat_hat, z_hat, z

    def forward_hash(self, x_feat, csi, snr_db, channel_mode="awgn",
                    bit_noise=True, channel_params=None):
        """Hash + bit-level JSCC模式（Stage3训练用）"""
        # 编码
        z = self.enc(x_feat, csi)
        s = self.jscc_enc(z, csi)

        # 信道模拟
        if channel_mode == "awgn":
            y = channel_awgn(s, snr_db)
            h = None
        else:
            y, h = channel_rayleigh(s, snr_db)

        z_hat = self.jscc_dec(y, csi, h)

        # Hash瓶颈
        if channel_params is None:
            channel_params = {'ber': 0.1}  # 默认10% BER

        hash_output = self.hash(z_hat, channel_params if bit_noise else None)
        z_q = hash_output['reconstructed']

        # 额外的bit噪声（训练时）
        if bit_noise and self.training:
            b_clean = hash_output['hash_bits_clean']
            b_noisy = apply_bit_noise(b_clean, flip_prob=0.05, block_drop_prob=0.02)
            z_q = self.hash.hash_decoder(b_noisy)

        # 解码
        feat_hat = self.dec(z_q, csi)

        return feat_hat, z_q, z_hat, hash_output

    def forward_full(self, x_feat, csi, snr_db, channel_mode="awgn",
                    target_len=None, with_hash=True):
        """完整前向传播：特征→波形"""
        if with_hash:
            feat_hat, z_q, z_hat, hash_output = self.forward_hash(
                x_feat, csi, snr_db, channel_mode, bit_noise=self.training
            )
        else:
            feat_hat, z_hat, z = self.forward_continuous(
                x_feat, csi, snr_db, channel_mode
            )
            hash_output = None

        # FARGAN声码器
        period, audio = self.vocoder(feat_hat, target_len=target_len)

        return {
            'audio': audio,
            'period': period,
            'feat_hat': feat_hat,
            'z_hat': z_hat if not with_hash else hash_output['reconstructed'],
            'hash_output': hash_output
        }

    def get_bitrate(self, frame_rate: float = 100.0) -> float:
        """计算标称码率"""
        return self.n_bits * frame_rate / 1000.0  # kbps

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_name': 'LiteSpeechJSCC',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_size_mb': total_params * 4 / (1024 * 1024),  # float32
            'bitrate_kbps': self.get_bitrate(),
            'feature_dim': self.feat_dim,
            'latent_dim': self.d_z,
            'hash_bits': self.n_bits,
            'hidden_size': self.hidden,
            'device': str(self.device),
            'components': {
                'encoder': 'EncoderLite',
                'jscc_enc': 'JSCCEncoder',
                'jscc_dec': 'JSCCDecoder',
                'decoder': 'DecoderLite',
                'hash': 'HashBottleneck',
                'vocoder': 'FARGANDecoder'
            }
        }


def create_lite_speech_jscc(feat_dim=36,
                           n_bits=16,
                           hidden=80,
                           device=None) -> LiteSpeechJSCC:
    """便捷工厂函数"""
    return LiteSpeechJSCC(
        feat_dim=feat_dim,
        d_csi=3,
        d_z=16,
        d_s=16,
        n_bits=n_bits,
        hidden=hidden,
        hash_method='greedy',
        device=device
    )


    def encode_stage(self, x_feat, csi, stage=1, hash_bottleneck=None):
        """
        分阶段编码：根据训练阶段选择编码路径

        Args:
            x_feat: [B, T, 36] 输入特征
            csi: [B, 4] 信道状态信息
            stage: 训练阶段 (1, 2, 3, 4)
            hash_bottleneck: HashBottleneck模块（Stage 3+需要）

        Returns:
            编码结果字典，包含中间输出用于损失计算
        """
        result = {}

        # 1. Feature Encoder: 特征 → 潜码
        z = self.enc(x_feat, csi)  # [B, T, d_z]
        result['latent'] = z

        # 2. Hash Encoder (Stage 3+): 潜码 → hash bits (编码端最末尾)
        if stage >= 3 and hash_bottleneck is not None:
            hash_logits = hash_bottleneck.hash_encoder(z)  # [B, T, K]
            hash_bits_clean = hash_bottleneck.hash_layer(hash_logits)  # [B, T, K]
            result['hash_logits'] = hash_logits
            result['hash_bits_clean'] = hash_bits_clean
            # 用hash bits作为JSCC输入
            jscc_input = hash_bits_clean
        else:
            # Stage 1-2: 直接用连续潜码
            jscc_input = z

        # 3. JSCC Encoder: (hash bits 或 潜码) → 符号
        s = self.jscc_enc(jscc_input, csi)  # [B, T, d_s]
        result['symbols'] = s

        return result

    def decode_stage(self, symbols_received, csi, stage=1, hash_bottleneck=None):
        """
        分阶段解码：根据训练阶段选择解码路径

        Args:
            symbols_received: [B, T, d_s] 接收到的符号
            csi: [B, 4] 信道状态信息
            stage: 训练阶段 (1, 2, 3, 4)
            hash_bottleneck: HashBottleneck模块（Stage 3+需要）

        Returns:
            解码结果字典，包含中间输出用于损失计算
        """
        result = {}

        # 1. JSCC Decoder: 接收符号 → (hash bits 或 潜码)
        jscc_output = self.jscc_dec(symbols_received, csi)  # [B, T, K] or [B, T, d_z]

        # 2. Hash Decoder (Stage 3+): hash bits → 重建潜码 (解码端最开始)
        if stage >= 3 and hash_bottleneck is not None:
            # JSCC输出应该是hash bits
            hash_bits_received = jscc_output  # [B, T, K]
            result['hash_bits_received'] = hash_bits_received
            # Hash decoder: bits → 连续潜码
            z_hat = hash_bottleneck.hash_decoder(hash_bits_received)  # [B, T, d_z]
        else:
            # Stage 1-2: JSCC输出直接是连续潜码
            z_hat = jscc_output

        result['latent_reconstructed'] = z_hat

        # 3. Feature Decoder: 重建潜码 → 重建特征
        features_pred = self.dec(z_hat, csi)  # [B, T, 36]
        result['features_pred'] = features_pred

        return result


if __name__ == "__main__":
    pass
