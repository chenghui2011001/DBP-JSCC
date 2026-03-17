# -*- coding: utf-8 -*-
"""
Vocoder decoder - directly consumes 36-dim vocoder features.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any
import math

try:
    from .vocoder_components import FARGANCore
    from .feature_adapter import FARGANFeatureSpec
except ImportError:
    try:
        from vocoder_components import FARGANCore
        from feature_adapter import FARGANFeatureSpec
    except ImportError:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__)))
        from vocoder_components import FARGANCore
        from feature_adapter import FARGANFeatureSpec


class PeriodEstimator(nn.Module):
    """从FARGAN特征估计周期序列"""

    def __init__(self):
        super().__init__()
        self.period_proj = nn.Sequential(
            nn.Linear(20, 32),  
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, fargan_features: torch.Tensor) -> torch.Tensor:
        """
        从FARGAN特征估计周期

        Args:
            fargan_features: [B, T, 36] FARGAN特征

        Returns:
            period: [B, T] 周期序列 (32-255)
        """
        # 基于DNN pitch估计周期 (第18维是dnn_pitch)
        dnn_pitch = fargan_features[..., 18:19]  # [B, T, 1]

        # FARGAN的周期计算: period = round(clip(256/2^(dnn_pitch+1.5), 32, 255))
        period_raw = 256.0 / torch.pow(2.0, dnn_pitch + 1.5)
        period = torch.round(torch.clamp(period_raw, 32.0, 255.0)).long().squeeze(-1)

        return period


class FARGANDecoder(nn.Module):
    """纯FARGAN解码器

    直接使用36维FARGAN特征进行波形生成
    专注于FARGAN声码器功能，不依赖AETHER架构
    """

    def __init__(
        self,
        fargan_subframe_size: int = 40,
        fargan_nb_subframes: int = 4,
        frame_rate_hz: float = 100.0,
        soft_period: bool = True
    ):
        super().__init__()
        self.frame_rate_hz = frame_rate_hz
        self.fargan_frame_size = fargan_subframe_size * fargan_nb_subframes  # 160
        self.soft_period = bool(soft_period)

        # 周期估计器
        self.period_estimator = PeriodEstimator()

        # FARGAN核心合成器
        self.fargan_core = FARGANCore(
            subframe_size=fargan_subframe_size,
            nb_subframes=fargan_nb_subframes,
            feature_dim=20,  # 使用前20维特征
            cond_size=256
        )

        # 初始化提示移除emoji与冗余输出

    def forward(
        self,
        fargan_features: torch.Tensor,
        target_len: Optional[int] = None,
        pre: Optional[torch.Tensor] = None,
        period_override: Optional[torch.Tensor] = None,
        override_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            fargan_features: [B, T, 36] FARGAN特征
            target_len: 目标音频长度

        Returns:
            period: [B, T] 周期序列
            audio: [B, 1, L] 生成的音频波形
        """
        B, T, _ = fargan_features.shape

        # 断言：输入特征必须有限
        if not torch.isfinite(fargan_features).all():
            raise RuntimeError("[NaNGuard] non-finite fargan_features at FARGANDecoder.forward")
        # 步骤1: 周期估计与可选门控覆盖
        # 软周期：始终可用（用于无覆盖或覆盖掩膜为0的位置）
        if self.soft_period:
            dnn_pitch = fargan_features[..., 18:19]
            period_soft = 256.0 / torch.pow(2.0, dnn_pitch + 1.5)
            period_soft = torch.clamp(period_soft.squeeze(-1), 32.0, 255.0)  # [B,T]
        else:
            period_soft = self.period_estimator(fargan_features).to(fargan_features.device).to(torch.float32)

        if period_override is not None or override_mask is not None:
            # 两者其一存在时，进行门控覆盖：mask=1 用 override，否则用 soft
            if period_override is None:
                period_override = period_soft
            po = period_override
            if override_mask is None:
                # 全部覆盖
                period = torch.clamp(po, 32.0, 255.0)
            else:
                m = override_mask.to(dtype=period_soft.dtype, device=period_soft.device)
                # 保证形状 [B,T]
                if m.dim() == 3 and m.size(-1) == 1:
                    m = m.squeeze(-1)
                m = torch.clamp(m, 0.0, 1.0)
                period = torch.clamp(m * po + (1.0 - m) * period_soft, 32.0, 255.0)
        else:
            period = period_soft
        if not torch.isfinite(period).all():
            raise RuntimeError("[NaNGuard] non-finite period at FARGANDecoder.forward")

        # 步骤2: 生成波形
        audio = self._generate_waveform(fargan_features, period, target_len, pre=pre)
        self.last_internal_tracks = getattr(self.fargan_core, "last_internal_tracks", None)
        
        return period, audio

    def _generate_waveform(
        self,
        fargan_features: torch.Tensor,
        period: torch.Tensor,
        target_len: Optional[int] = None,
        pre: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        使用FARGAN核心生成波形

        Args:
            fargan_features: [B, T, 36] FARGAN特征
            period: [B, T] 周期序列
            target_len: 目标音频长度

        Returns:
            audio: [B, 1, L] 生成的音频波形
        """
        B, T, _ = fargan_features.shape

        # 计算需要生成的帧数
        # 注意: FARGAN的条件网络会减少时间维度: T → T-4
        max_available_frames = T - 4  # FARGAN条件网络的输出帧数

        # 教师强制预热帧数（若提供pre）
        nb_pre_frames = pre.size(1) // self.fargan_frame_size if pre is not None else 0

        # 可用于真正生成的帧数上限
        gen_capacity = max(0, max_available_frames - nb_pre_frames)

        if target_len is not None:
            target_frames_total = (target_len + self.fargan_frame_size - 1) // self.fargan_frame_size
            target_frames_gen = max(0, target_frames_total - nb_pre_frames)
            nb_frames = min(gen_capacity, target_frames_gen)
        else:
            nb_frames = gen_capacity

        # 确保帧数为正（至少生成1帧）
        nb_frames = max(1, nb_frames)

        # 使用前20维特征进行FARGAN合成
        features_20 = fargan_features[..., :20]

        # FARGAN生成
        audio, _ = self.fargan_core(features_20, period, nb_frames, pre=pre)

        # 确保输出形状为 [B, 1, L]
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        # 裁剪到目标长度
        if target_len is not None and audio.size(-1) > target_len:
            audio = audio[..., :target_len]

        return audio

    def get_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = {
            'model_type': 'FARGAN解码器',
            'input_feature_dim': 36,  # FARGAN特征维度
            'fargan_frame_size': self.fargan_frame_size,
            'frame_rate_hz': self.frame_rate_hz,
            'components': {
                'period_estimator': 'PeriodEstimator',
                'fargan_core': 'FARGANCore'
            },
            'total_parameters': sum(p.numel() for p in self.parameters())
        }
        return info


## 移除文件内自测函数与emoji输出，避免干扰训练日志
