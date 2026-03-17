# -*- coding: utf-8 -*-
"""
Feature Adapter: AETHER 48-dim → FARGAN 36-dim
将AETHER的48维特征转换为FARGAN兼容的36维特征
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

try:
    from ..utils.feature_spec import get_default_feature_spec
except ImportError:
    try:
        from utils.feature_spec import get_default_feature_spec
    except ImportError:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from utils.feature_spec import get_default_feature_spec


class Feature48To36Adapter(nn.Module):
    """
    AETHER 48维特征 → FARGAN 36维特征适配器

    AETHER 48维结构:
    - ceps: 20维 (倒谱系数)
    - f0: 1维 (DNN基频)
    - voicing: 1维 (浊音概率)
    - enhanced: 6维 (增强特征)
    - lpc: 16维 (LPC系数)
    - prosodic: 4维 (韵律特征)

    FARGAN 36维结构:
    - ceps: 18维 (倒谱系数) - 从AETHER 20维压缩而来
    - dnn_pitch: 1维 (DNN基频) - 从f0转换
    - frame_corr: 1维 (帧相关系数) - 从voicing转换
    - lpc: 16维 (LPC系数) - 直接复用
    """

    def __init__(self, learn_f0_transform: bool = True, learn_corr_transform: bool = True):
        super().__init__()

        self.learn_f0_transform = learn_f0_transform
        self.learn_corr_transform = learn_corr_transform

        # AETHER特征规范
        self.aether_spec = get_default_feature_spec()

        # 可学习的F0转换参数 (AETHER → FARGAN DNN pitch域)
        if learn_f0_transform:
            self.f0_scale = nn.Parameter(torch.tensor(1.0))
            self.f0_bias = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer('f0_scale', torch.tensor(1.0))
            self.register_buffer('f0_bias', torch.tensor(0.0))

        # 可学习的相关系数转换参数 (voicing概率 → 帧相关系数)
        if learn_corr_transform:
            self.corr_weight = nn.Parameter(torch.tensor(1.0))
            self.corr_bias = nn.Parameter(torch.tensor(-0.5))  # FARGAN期望相关系数-0.5偏移
        else:
            self.register_buffer('corr_weight', torch.tensor(1.0))
            self.register_buffer('corr_bias', torch.tensor(-0.5))

        # 可选: 增强特征→LPC微调的轻量映射
        self.enhanced_to_lpc_proj = nn.Linear(6, 16, bias=False)  # 修正维度: 6→16
        nn.init.zeros_(self.enhanced_to_lpc_proj.weight)  # 初始化为0，不影响原始LPC

        # 可选: 韵律特征→ceps微调的轻量映射
        self.prosodic_to_ceps_proj = nn.Linear(4, 18, bias=False)  # 修正维度: 4→18
        nn.init.zeros_(self.prosodic_to_ceps_proj.weight)  # 初始化为0，不影响原始ceps

        # CEPS维度压缩: 20维→18维 (去掉高频噪声成分)
        self.ceps_compress = nn.Linear(20, 18, bias=False)
        # 初始化为保留低18维的恒等映射
        with torch.no_grad():
            self.ceps_compress.weight.copy_(torch.eye(18, 20))

    def forward(self, aether_features: torch.Tensor) -> torch.Tensor:
        """
        转换AETHER 48维特征为FARGAN 36维特征

        Args:
            aether_features: [B, T, 48] AETHER特征

        Returns:
            fargan_features: [B, T, 36] FARGAN兼容特征
        """
        B, T, _ = aether_features.shape

        # 提取AETHER特征分量
        ceps = self.aether_spec.extract_feature(aether_features, 'ceps')           # [B,T,20]
        f0 = self.aether_spec.extract_feature(aether_features, 'f0')               # [B,T,1]
        voicing = self.aether_spec.extract_feature(aether_features, 'voicing')     # [B,T,1]
        enhanced = self.aether_spec.extract_feature(aether_features, 'enhanced')   # [B,T,6]
        lpc = self.aether_spec.extract_feature(aether_features, 'lpc')             # [B,T,16]
        prosodic = self.aether_spec.extract_feature(aether_features, 'prosodic')   # [B,T,4]

        # 1. 倒谱系数 - 压缩20维→18维 + 可选韵律微调
        ceps_compressed = self.ceps_compress(ceps)  # [B,T,20] → [B,T,18]
        fargan_ceps = ceps_compressed + 0.01 * self.prosodic_to_ceps_proj(prosodic)  # 微调权重很小

        # 2. F0转换: AETHER log2域 → FARGAN DNN pitch域
        # AETHER: f0 ∈ [-3, 3] (大致范围)
        # FARGAN: dnn_pitch = log2(256/period) - 1.5, period ∈ [32, 255]
        fargan_dnn_pitch = self.f0_scale * f0 + self.f0_bias

        # 3. 帧相关系数转换: voicing概率 → 帧相关系数
        # FARGAN期望: frame_corr ∈ [-0.5, 0.5], 中心在-0.5附近
        frame_corr = self.corr_weight * (voicing - 0.5) + self.corr_bias
        frame_corr = torch.clamp(frame_corr, -0.8, 0.5)  # 限制在合理范围

        # 4. LPC系数 - 直接复用 + 可选增强特征微调
        fargan_lpc = lpc + 0.01 * self.enhanced_to_lpc_proj(enhanced)  # 微调权重很小

        # 组装FARGAN 36维特征 [ceps(18) + dnn_pitch(1) + frame_corr(1) + lpc(16)]
        fargan_features = torch.cat([
            fargan_ceps,        # [B,T,18]
            fargan_dnn_pitch,   # [B,T,1]
            frame_corr,         # [B,T,1]
            fargan_lpc          # [B,T,16]
        ], dim=-1)  # [B,T,36]

        return fargan_features

    def get_fargan_feature_slices(self) -> Dict[str, slice]:
        """返回FARGAN 36维特征的切片索引"""
        return {
            'ceps': slice(0, 18),
            'dnn_pitch': slice(18, 19),
            'frame_corr': slice(19, 20),
            'lpc': slice(20, 36)
        }

    def extract_fargan_feature(self, fargan_features: torch.Tensor, name: str) -> torch.Tensor:
        """从FARGAN特征中提取指定分量"""
        slices = self.get_fargan_feature_slices()
        if name not in slices:
            raise ValueError(f"Unknown FARGAN feature: {name}")
        return fargan_features[..., slices[name]]


class FARGANFeatureSpec:
    """FARGAN 36维特征规范，对应原始FARGAN特征布局"""

    @staticmethod
    def get_feature_dims() -> Dict[str, int]:
        return {
            'ceps': 18,      # 倒谱系数 (压缩自原20维)
            'dnn_pitch': 1,  # DNN基频 (对应index 18)
            'frame_corr': 1, # 帧相关系数 (对应index 19)
            'lpc': 16        # LPC系数 (index 20-35)
        }

    @staticmethod
    def get_feature_slices() -> Dict[str, slice]:
        return {
            'ceps': slice(0, 18),
            'dnn_pitch': slice(18, 19),
            'frame_corr': slice(19, 20),
            'lpc': slice(20, 36)
        }

    # 兼容AETHER接口：提供 get_feature_slice，并支持别名
    @staticmethod
    def get_feature_slice(name: str) -> slice:
        """获取特征切片；兼容 AETHER 的接口命名。

        - 'f0' 作为别名映射到 'dnn_pitch'
        - 其余键参见 get_feature_slices()
        """
        mapping = FARGANFeatureSpec.get_feature_slices()
        key = 'dnn_pitch' if name == 'f0' else name
        if key not in mapping:
            raise ValueError(f"Unknown FARGAN feature: {name}")
        return mapping[key]

    @staticmethod
    def extract_feature(features: torch.Tensor, name: str) -> torch.Tensor:
        """从FARGAN特征中提取指定分量；支持 'f0' 别名，'voicing' 派生。

        - 'f0' 别名 → 'dnn_pitch'
        - 'voicing'：优先由 'frame_corr' ∈[-0.5,0.5] 映射到 [0,1]；若不可用，
          退化到从 'dnn_pitch' 通过阈值近似派生。
        - 'ceps'：直接提取前18维倒谱系数
        """
        slices = FARGANFeatureSpec.get_feature_slices()
        if name == 'f0':
            name = 'dnn_pitch'
        if name in slices:
            return features[..., slices[name]]
        if name == 'voicing':
            if 'frame_corr' in slices:
                fc = features[..., slices['frame_corr']]
                return (fc + 0.5).clamp(0.0, 1.0)
            # 回退：基于 dnn_pitch 简单阈值
            dp = features[..., slices['dnn_pitch']]
            return (dp > -1.0).float()
        raise ValueError(f"Unknown FARGAN feature: {name}")

    @staticmethod
    def get_total_dims() -> int:
        return sum(FARGANFeatureSpec.get_feature_dims().values())  # 36

    @property
    def total_dim(self) -> int:
        """兼容性属性，匹配AETHERFeatureSpec接口"""
        return self.get_total_dims()

    def get_feature_info(self) -> Dict[str, any]:
        """获取特征规范信息，兼容AETHERFeatureSpec接口"""
        return {
            'total_dims': self.get_total_dims(),
            'feature_dims': self.get_feature_dims(),
            'feature_slices': self.get_feature_slices(),
            'spec_type': 'fargan'
        }

    @staticmethod
    def get_feature_importance_weights() -> Dict[str, float]:
        """获取各特征的重要性权重"""
        return {
            'ceps': 1.0,        # 倒谱系数最重要
            'dnn_pitch': 1.0,   # 基频最重要
            'frame_corr': 0.8,  # 相关系数次重要
            'lpc': 0.9          # LPC系数重要
        }

    @staticmethod
    def get_perceptually_critical_features() -> list:
        """获取感知上最关键的特征"""
        return ['dnn_pitch', 'frame_corr', 'ceps']

    @staticmethod
    def create_layered_loss_config() -> Dict[str, float]:
        """为分层损失创建配置"""
        return {
            'ceps': 1.0,
            'dnn_pitch': 1.0,
            'frame_corr': 0.8,
            'lpc': 0.9
        }


def get_fargan_feature_spec():
    """获取FARGAN特征规范实例"""
    return FARGANFeatureSpec()


def test_feature_adapter():
    """测试特征适配器"""
    import numpy as np

    # 创建适配器
    adapter = Feature48To36Adapter()

    # 模拟AETHER 48维特征
    B, T = 2, 100
    aether_features = torch.randn(B, T, 48)

    # 转换为FARGAN特征
    fargan_features = adapter(aether_features)

    print(f"输入AETHER特征: {aether_features.shape}")
    print(f"输出FARGAN特征: {fargan_features.shape}")

    # 验证特征分量
    spec = FARGANFeatureSpec()
    ceps = spec.extract_feature(fargan_features, 'ceps')
    dnn_pitch = spec.extract_feature(fargan_features, 'dnn_pitch')
    frame_corr = spec.extract_feature(fargan_features, 'frame_corr')
    lpc = spec.extract_feature(fargan_features, 'lpc')

    print(f"倒谱系数: {ceps.shape}")
    print(f"DNN基频: {dnn_pitch.shape}")
    print(f"帧相关系数: {frame_corr.shape}")
    print(f"LPC系数: {lpc.shape}")

    print(f"DNN基频范围: [{dnn_pitch.min().item():.3f}, {dnn_pitch.max().item():.3f}]")
    print(f"帧相关系数范围: [{frame_corr.min().item():.3f}, {frame_corr.max().item():.3f}]")

    print("✅ 特征适配器测试通过")


if __name__ == "__main__":
    pass
