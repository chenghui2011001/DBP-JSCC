#!/usr/bin/env python3
"""
特征配置规范 (FeatureSpec) - 用于定义48维特征的结构
避免硬编码切片，提供灵活的特征定义和访问
"""

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import torch
import numpy as np


@dataclass
class FeatureSpec:
    """48维特征规范配置 - 支持多层次特征访问"""

    # 说明：F0使用LPCNet风格的dnn-pitch（log2域，约-1.5~+1.5）。
    # 训练中不建议做min-max到[0,1]；如需稳定正则，可参考下列建议量级。
    F0_NORM: str = "dnn"
    F0_DNN_RANGE: Tuple[float, float] = (-1.5, +1.5)
    F0_STD_FLOOR: float = 0.25  # dnn尺度上的方差下限建议

    # 特征总维度
    total_dim: int = 48

    # 各部分特征的范围定义 [start_idx, end_idx) (左闭右开)
    # 与extract_features_48_lpcnet_style.py完全一致
    ceps_range: Tuple[int, int] = (0, 20)      # CEPS特征 (20维，DCT对数频谱) [0:19]
    f0_range: Tuple[int, int] = (20, 21)       # F0特征 (1维，简化版本) [20]
    voicing_range: Tuple[int, int] = (21, 22)  # Voicing特征 (1维，语音/非语音判定) [21]
    enhanced_range: Tuple[int, int] = (22, 28) # 增强频谱特征 (6维) [22:27]
    lpc_range: Tuple[int, int] = (28, 44)      # LPC系数 (16维) [28:43]
    prosodic_range: Tuple[int, int] = (44, 48) # 韵律特征 (4维，F0轨迹、能量等) [44:47]

    # 细分特征范围 - 支持更精细的特征访问
    # CEPS子分量 (基于频率重要性分层)
    ceps_low_freq: Tuple[int, int] = (0, 8)      # 低频成分 [0:7] - 最重要的共振峰信息
    ceps_mid_freq: Tuple[int, int] = (8, 14)     # 中频成分 [8:13] - 频谱细节
    ceps_high_freq: Tuple[int, int] = (14, 20)   # 高频成分 [14:19] - 噪声特征

    # 增强频谱特征子分量
    enhanced_centroid: Tuple[int, int] = (22, 23)   # [22] 频谱质心
    enhanced_bandwidth: Tuple[int, int] = (23, 24)  # [23] 频谱带宽
    enhanced_rolloff: Tuple[int, int] = (24, 25)    # [24] 频谱滚降
    enhanced_flatness: Tuple[int, int] = (25, 26)   # [25] 频谱平坦度
    enhanced_entropy: Tuple[int, int] = (26, 27)    # [26] 频谱熵
    enhanced_zcr: Tuple[int, int] = (27, 28)        # [27] 零交叉率

    # LPC系数子分量 (基于预测阶数重要性)
    lpc_low_order: Tuple[int, int] = (28, 32)     # [28:31] 低阶系数 - 基本共振峰
    lpc_mid_order: Tuple[int, int] = (32, 38)     # [32:37] 中阶系数 - 详细频谱形状
    lpc_high_order: Tuple[int, int] = (38, 44)    # [38:43] 高阶系数 - 精细结构

    # 韵律特征子分量
    prosodic_f0_trajectory: Tuple[int, int] = (44, 45)  # [44] F0轨迹特征
    prosodic_energy: Tuple[int, int] = (45, 46)         # [45] 能量特征
    prosodic_voicing_prob: Tuple[int, int] = (46, 47)   # [46] 语音概率
    prosodic_modulation: Tuple[int, int] = (47, 48)     # [47] 频率调制

    # 特征名称映射
    feature_names: Dict[str, str] = None

    # 归一化参数 (用于LPCNet风格归一化)
    normalization_params: Dict[str, Dict[str, float]] = None

    def __post_init__(self):
        """初始化后处理"""
        if self.feature_names is None:
            self.feature_names = {
                # 主要特征类别
                'ceps': 'CEPS特征 (DCT对数频谱)',
                'f0': 'F0特征 (基频)',
                'voicing': 'Voicing特征 (清浊音判定)',
                'enhanced': '增强频谱特征',
                'lpc': 'LPC系数 (线性预测)',
                'prosodic': '韵律特征 (F0轨迹、能量等)',

                # CEPS子分量
                'ceps_low_freq': 'CEPS低频成分 (共振峰主信息)',
                'ceps_mid_freq': 'CEPS中频成分 (频谱细节)',
                'ceps_high_freq': 'CEPS高频成分 (噪声特征)',

                # 增强频谱子分量
                'enhanced_centroid': '频谱质心',
                'enhanced_bandwidth': '频谱带宽',
                'enhanced_rolloff': '频谱滚降',
                'enhanced_flatness': '频谱平坦度',
                'enhanced_entropy': '频谱熵',
                'enhanced_zcr': '零交叉率',

                # LPC子分量
                'lpc_low_order': 'LPC低阶系数 (基本共振峰)',
                'lpc_mid_order': 'LPC中阶系数 (详细频谱形状)',
                'lpc_high_order': 'LPC高阶系数 (精细结构)',

                # 韵律子分量
                'prosodic_f0_trajectory': 'F0轨迹特征',
                'prosodic_energy': '能量特征',
                'prosodic_voicing_prob': '语音概率',
                'prosodic_modulation': '频率调制'
            }

        if self.normalization_params is None:
            # LPCNet风格归一化参数 - 支持细分特征
            self.normalization_params = {
                # 主要特征类别
                'ceps': {'mean': 0.0, 'std': 1.0, 'method': 'z_norm'},  # 零均值单位方差
                # F0采用dnn-pitch量纲；不做二次归一化，避免动态缩放
                'f0': {'method': 'none'},
                'voicing': {'min': 0.0, 'max': 1.0, 'method': 'none'},  # 已是 [0,1]
                'enhanced': {'min': 0.0, 'max': 1.0, 'method': 'none'}, # 已归一化
                'lpc': {'mean': 0.0, 'std': 1.0, 'method': 'z_norm'},   # 零均值单位方差
                'prosodic': {'mean': 0.0, 'std': 1.0, 'method': 'z_norm'}, # 零均值单位方差

                # CEPS子分量 (不同频段有不同重要性权重)
                'ceps_low_freq': {'mean': 0.0, 'std': 1.0, 'method': 'z_norm', 'weight': 1.0},  # 最重要
                'ceps_mid_freq': {'mean': 0.0, 'std': 1.0, 'method': 'z_norm', 'weight': 0.7},  # 中等重要
                'ceps_high_freq': {'mean': 0.0, 'std': 1.0, 'method': 'z_norm', 'weight': 0.5}, # 较不重要

                # 增强频谱子分量 (各有特定范围和重要性)
                'enhanced_centroid': {'min': 0.0, 'max': 1.0, 'method': 'none', 'weight': 0.8},
                'enhanced_bandwidth': {'min': 0.0, 'max': 1.0, 'method': 'none', 'weight': 0.6},
                'enhanced_rolloff': {'min': 0.0, 'max': 1.0, 'method': 'none', 'weight': 0.6},
                'enhanced_flatness': {'min': 0.0, 'max': 1.0, 'method': 'none', 'weight': 0.7},
                'enhanced_entropy': {'min': 0.0, 'max': 1.0, 'method': 'none', 'weight': 0.7},
                'enhanced_zcr': {'min': 0.0, 'max': 1.0, 'method': 'none', 'weight': 0.5},

                # LPC子分量 (低阶更重要)
                'lpc_low_order': {'mean': 0.0, 'std': 1.0, 'method': 'z_norm', 'weight': 1.0},  # 最重要
                'lpc_mid_order': {'mean': 0.0, 'std': 1.0, 'method': 'z_norm', 'weight': 0.8},  # 重要
                'lpc_high_order': {'mean': 0.0, 'std': 1.0, 'method': 'z_norm', 'weight': 0.6}, # 次要

                # 韵律子分量
                'prosodic_f0_trajectory': {'min': -0.2, 'max': 0.2, 'method': 'minmax', 'weight': 0.9},
                'prosodic_energy': {'min': -8.0, 'max': 0.0, 'method': 'minmax', 'weight': 0.8},
                'prosodic_voicing_prob': {'min': 0.0, 'max': 1.0, 'method': 'none', 'weight': 0.9},
                'prosodic_modulation': {'min': -1.0, 'max': 1.0, 'method': 'minmax', 'weight': 0.6}
            }

        # === 细分切片 ===
        self.layered_ranges = {
            # CEPS 20 -> 8/6/6
            "ceps_low_freq":  slice(self.ceps_range[0], self.ceps_range[0] + 8),
            "ceps_mid_freq":  slice(self.ceps_range[0] + 8, self.ceps_range[0] + 14),
            "ceps_high_freq": slice(self.ceps_range[0] + 14, self.ceps_range[1]),
            # 增强 6 -> 单维拆分
            "enhanced_centroid":  slice(self.enhanced_range[0] + 0, self.enhanced_range[0] + 1),
            "enhanced_bandwidth": slice(self.enhanced_range[0] + 1, self.enhanced_range[0] + 2),
            "enhanced_rolloff":   slice(self.enhanced_range[0] + 2, self.enhanced_range[0] + 3),
            "enhanced_flatness":  slice(self.enhanced_range[0] + 3, self.enhanced_range[0] + 4),
            "enhanced_entropy":   slice(self.enhanced_range[0] + 4, self.enhanced_range[0] + 5),
            "enhanced_zcr":       slice(self.enhanced_range[0] + 5, self.enhanced_range[0] + 6),
            # LPC 16 -> 4/6/6
            "lpc_low_order":  slice(self.lpc_range[0] + 0, self.lpc_range[0] + 4),
            "lpc_mid_order":  slice(self.lpc_range[0] + 4, self.lpc_range[0] + 10),
            "lpc_high_order": slice(self.lpc_range[0] + 10, self.lpc_range[1]),
            # 韵律 4 -> 单维拆分
            "prosodic_f0_trajectory": slice(self.prosodic_range[0] + 0, self.prosodic_range[0] + 1),
            "prosodic_energy":        slice(self.prosodic_range[0] + 1, self.prosodic_range[0] + 2),
            "prosodic_voicing_prob":  slice(self.prosodic_range[0] + 2, self.prosodic_range[0] + 3),
            "prosodic_modulation":    slice(self.prosodic_range[0] + 3, self.prosodic_range[1]),
        }

        # === 感知重要性权重 ===
        self.importance = {
            "ceps": 1.0, "f0": 1.0, "voicing": 1.0, "enhanced": 0.8, "lpc": 0.9, "prosodic": 0.8,
            "ceps_low_freq": 1.0, "ceps_mid_freq": 0.7, "ceps_high_freq": 0.5,
            "enhanced_centroid": 0.8, "enhanced_bandwidth": 0.6, "enhanced_rolloff": 0.6,
            "enhanced_flatness": 0.7, "enhanced_entropy": 0.7, "enhanced_zcr": 0.5,
            "lpc_low_order": 1.0, "lpc_mid_order": 0.8, "lpc_high_order": 0.6,
            "prosodic_f0_trajectory": 0.9, "prosodic_energy": 0.8,
            "prosodic_voicing_prob": 0.9, "prosodic_modulation": 0.6,
        }

        # 特征域定义
        if not hasattr(self, 'feature_domains'):
            self.feature_domains = {
                'f0': 'dnn',        # LPCNet风格的dnn_pitch格式
                'voicing': 'unit',  # [0,1]范围
                'ceps': 'norm',     # 归一化倒谱
                'enhanced': 'unit', # [0,1]范围
                'lpc': 'norm',      # 归一化LPC系数
                'prosodic': 'norm'  # 归一化韵律特征
            }

        # 验证特征范围
        self._validate_ranges()

    def _validate_ranges(self):
        """验证特征范围的一致性"""
        ranges = [
            self.ceps_range,
            self.f0_range,
            self.voicing_range,
            self.enhanced_range,
            self.lpc_range,
            self.prosodic_range
        ]

        # 检查范围是否连续且无重叠
        expected_start = 0
        for i, (start, end) in enumerate(ranges):
            if start != expected_start:
                raise ValueError(f"特征范围不连续: 范围{i} 开始于{start}, 期望{expected_start}")
            if start >= end:
                raise ValueError(f"特征范围无效: 范围{i} [{start}, {end})")
            expected_start = end

        if expected_start != self.total_dim:
            raise ValueError(f"特征总维度不匹配: 计算得{expected_start}, 期望{self.total_dim}")

    def get_feature_slice(self, feature_name: str) -> slice:
        """获取指定特征的切片对象 - 支持细分特征"""
        # 首先检查新的 layered_ranges
        if hasattr(self, 'layered_ranges') and feature_name in self.layered_ranges:
            return self.layered_ranges[feature_name]

        range_map = {
            # 主要特征类别
            'ceps': self.ceps_range,
            'f0': self.f0_range,
            'voicing': self.voicing_range,
            'enhanced': self.enhanced_range,
            'lpc': self.lpc_range,
            'prosodic': self.prosodic_range,

            # CEPS子分量
            'ceps_low_freq': self.ceps_low_freq,
            'ceps_mid_freq': self.ceps_mid_freq,
            'ceps_high_freq': self.ceps_high_freq,

            # 增强频谱子分量
            'enhanced_centroid': self.enhanced_centroid,
            'enhanced_bandwidth': self.enhanced_bandwidth,
            'enhanced_rolloff': self.enhanced_rolloff,
            'enhanced_flatness': self.enhanced_flatness,
            'enhanced_entropy': self.enhanced_entropy,
            'enhanced_zcr': self.enhanced_zcr,

            # LPC子分量
            'lpc_low_order': self.lpc_low_order,
            'lpc_mid_order': self.lpc_mid_order,
            'lpc_high_order': self.lpc_high_order,

            # 韵律子分量
            'prosodic_f0_trajectory': self.prosodic_f0_trajectory,
            'prosodic_energy': self.prosodic_energy,
            'prosodic_voicing_prob': self.prosodic_voicing_prob,
            'prosodic_modulation': self.prosodic_modulation
        }

        if feature_name not in range_map:
            raise ValueError(f"未知特征名称: {feature_name}. 可用特征: {list(range_map.keys())}")

        start, end = range_map[feature_name]
        return slice(start, end)

    def extract_feature(self, features: torch.Tensor, feature_name: str) -> torch.Tensor:
        """从特征张量中提取指定特征"""
        feature_slice = self.get_feature_slice(feature_name)
        return features[..., feature_slice]

    def get_feature_dims(self, feature_name: str) -> int:
        """获取指定特征的维度数"""
        start, end = getattr(self, f"{feature_name}_range")
        return end - start

    def normalize_features(self, features: torch.Tensor, feature_name: str) -> torch.Tensor:
        """对特定特征应用LPCNet风格归一化"""
        if feature_name not in self.normalization_params:
            return features

        params = self.normalization_params[feature_name]
        method = params.get('method', 'none')

        if method == 'z_norm':
            # 零均值单位方差
            mean = params.get('mean', 0.0)
            std = params.get('std', 1.0)
            return (features - mean) / std
        elif method == 'minmax':
            # 线性归一化到 [min, max]
            min_val = params.get('min', 0.0)
            max_val = params.get('max', 1.0)
            feat_min = features.min()
            feat_max = features.max()
            feat_range = feat_max - feat_min
            if feat_range > 1e-8:
                normalized = (features - feat_min) / feat_range
                return normalized * (max_val - min_val) + min_val
            else:
                return torch.full_like(features, (min_val + max_val) / 2)
        else:
            # 'none' - 不应用归一化
            return features

    def apply_lpcnet_normalization(self, features: torch.Tensor) -> torch.Tensor:
        """对整个特征张量应用LPCNet风格归一化"""
        result = features.clone()

        for feature_name in ['ceps', 'f0', 'voicing', 'enhanced', 'lpc', 'prosodic']:
            feature_slice = self.get_feature_slice(feature_name)
            feature_data = features[..., feature_slice]
            normalized_data = self.normalize_features(feature_data, feature_name)
            result[..., feature_slice] = normalized_data

        return result

    def get_legacy_indices(self) -> Dict[str, Tuple[int, int]]:
        """获取传统硬编码索引映射 (用于兼容性)"""
        return {
            'mel_idx': self.ceps_range,      # 传统的mel对应到ceps
            'pitch_idx': self.f0_range,      # 传统的pitch对应到f0
            'vuv_idx': self.voicing_range,   # 传统的vuv对应到voicing
            'lpc_idx': self.lpc_range,       # LPC保持不变
            'enhanced_idx': self.enhanced_range,  # 新增的增强特征
            'prosodic_idx': self.prosodic_range   # 新增的韵律特征
        }

    def create_feature_mask(self, feature_names: List[str]) -> torch.Tensor:
        """创建用于选择特定特征的掩码"""
        mask = torch.zeros(self.total_dim, dtype=torch.bool)

        for feature_name in feature_names:
            feature_slice = self.get_feature_slice(feature_name)
            mask[feature_slice] = True

        return mask

    def get_feature_info(self) -> Dict[str, Dict]:
        """获取所有特征的详细信息 - 包括细分特征"""
        info = {}

        # 主要特征类别
        main_features = ['ceps', 'f0', 'voicing', 'enhanced', 'lpc', 'prosodic']

        # 细分特征类别
        sub_features = [
            'ceps_low_freq', 'ceps_mid_freq', 'ceps_high_freq',
            'enhanced_centroid', 'enhanced_bandwidth', 'enhanced_rolloff',
            'enhanced_flatness', 'enhanced_entropy', 'enhanced_zcr',
            'lpc_low_order', 'lpc_mid_order', 'lpc_high_order',
            'prosodic_f0_trajectory', 'prosodic_energy', 'prosodic_voicing_prob', 'prosodic_modulation'
        ]

        all_features = main_features + sub_features

        for feature_name in all_features:
            try:
                feature_slice = self.get_feature_slice(feature_name)
                start, end = feature_slice.start, feature_slice.stop
                info[feature_name] = {
                    'range': (start, end),
                    'dim': end - start,
                    'description': self.feature_names.get(feature_name, 'Unknown'),
                    'normalization': self.normalization_params.get(feature_name, {}),
                    'weight': self.normalization_params.get(feature_name, {}).get('weight', 1.0)
                }
            except Exception:
                # 跳过无效的特征名称
                continue

        return info

    def get_feature_importance_weights(self) -> Dict[str, float]:
        """获取各特征的重要性权重"""
        if hasattr(self, 'importance'):
            return dict(self.importance)
        # 回退到 normalization_params 中的权重
        weights = {}
        for feature_name, params in self.normalization_params.items():
            weights[feature_name] = params.get('weight', 1.0)
        return weights

    def get_high_priority_features(self, threshold: float = 0.8) -> List[str]:
        """获取高优先级特征（权重 >= threshold）"""
        weights = self.get_feature_importance_weights()
        return [name for name, weight in weights.items() if weight >= threshold]

    def get_perceptually_critical_features(self) -> List[str]:
        """获取感知上最关键的特征组合"""
        return ["f0","voicing","ceps_low_freq","lpc_low_order",
                "enhanced_centroid","prosodic_f0_trajectory","prosodic_energy"]

    def extract_multiple_features(self, features: torch.Tensor, feature_names: List[str]) -> Dict[str, torch.Tensor]:
        """一次提取多个特征"""
        result = {}
        for name in feature_names:
            result[name] = self.extract_feature(features, name)
        return result

    def print_feature_layout(self):
        """打印特征布局信息"""
        print(f"48维特征布局:")
        print(f"{'特征名称':<12} {'范围':<10} {'维度':<4} {'描述'}")
        print("-" * 50)

        for feature_name in ['ceps', 'f0', 'voicing', 'enhanced', 'lpc', 'prosodic']:
            start, end = getattr(self, f"{feature_name}_range")
            dim = end - start
            desc = self.feature_names.get(feature_name, '')
            print(f"{feature_name:<12} [{start:2d}:{end:2d}) {dim:2d}维  {desc}")


# 默认的48维特征规范实例
DEFAULT_FEATURE_SPEC = FeatureSpec()


def get_default_feature_spec() -> FeatureSpec:
    """获取默认的特征规范"""
    return DEFAULT_FEATURE_SPEC


def create_custom_feature_spec(
    ceps_dim: int = 20,
    f0_dim: int = 1,
    voicing_dim: int = 1,
    enhanced_dim: int = 6,
    lpc_dim: int = 16,
    prosodic_dim: int = 4
) -> FeatureSpec:
    """创建自定义的特征规范"""

    # 计算范围
    start = 0
    ceps_range = (start, start + ceps_dim)
    start += ceps_dim

    f0_range = (start, start + f0_dim)
    start += f0_dim

    voicing_range = (start, start + voicing_dim)
    start += voicing_dim

    enhanced_range = (start, start + enhanced_dim)
    start += enhanced_dim

    lpc_range = (start, start + lpc_dim)
    start += lpc_dim

    prosodic_range = (start, start + prosodic_dim)
    start += prosodic_dim

    total_dim = start

    return FeatureSpec(
        total_dim=total_dim,
        ceps_range=ceps_range,
        f0_range=f0_range,
        voicing_range=voicing_range,
        enhanced_range=enhanced_range,
        lpc_range=lpc_range,
        prosodic_range=prosodic_range
    )


def test_feature_spec():
    """测试特征规范功能"""
    print("测试FeatureSpec功能...")

    spec = get_default_feature_spec()
    spec.print_feature_layout()

    # 测试特征提取
    features = torch.randn(2, 100, 48)  # [batch, time, feature]

    ceps = spec.extract_feature(features, 'ceps')
    f0 = spec.extract_feature(features, 'f0')
    lpc = spec.extract_feature(features, 'lpc')

    print(f"\n特征提取测试:")
    print(f"原始特征形状: {features.shape}")
    print(f"CEPS形状: {ceps.shape}")
    print(f"F0形状: {f0.shape}")
    print(f"LPC形状: {lpc.shape}")

    # 测试归一化
    normalized = spec.apply_lpcnet_normalization(features)
    print(f"归一化后形状: {normalized.shape}")

    print("✅ FeatureSpec测试通过")


if __name__ == "__main__":
    pass
