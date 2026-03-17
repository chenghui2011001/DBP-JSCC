#!/usr/bin/env python3
"""
JSCC比特流信道模拟工具，仿照channel_sim.py设计

Design goals:
- 对RVQ量化索引进行真实的比特流级别信道模拟
- 支持不同的错误模式：随机错误、突发错误、擦除错误
- 基于CSI参数估计信道质量，计算相应的比特错误率
- 兼容Stage5的JSCC训练流程
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np


class JSCCChannelSimulator:
    """JSCC比特流信道模拟器"""

    def __init__(self, frame_hz: int = 100):
        self.frame_hz = frame_hz

    def csi_to_ber(self, csi_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        从CSI参数估计比特错误率(BER)

        Args:
            csi_dict: CSI字典，包含snr_proxy, time_selectivity, freq_selectivity, los_ratio

        Returns:
            ber: [B] 每个批次的平均BER
        """
        device = next(iter(csi_dict.values())).device
        dtype = next(iter(csi_dict.values())).dtype
        B = next(iter(csi_dict.values())).shape[0]

        # 从SNR代理计算基础BER
        snr_proxy = csi_dict.get('snr_proxy', torch.zeros(B, device=device, dtype=dtype))
        snr_db = snr_proxy * 20.0  # 假设snr_proxy是归一化的

        # SNR到BER的映射（AWGN信道近似）
        # BER ≈ 0.5 * erfc(sqrt(SNR_linear))，简化为指数衰减
        snr_linear = torch.pow(10.0, snr_db / 10.0)
        base_ber = 0.5 * torch.exp(-snr_linear / 4.0)  # 近似公式

        # 时间选择性增加BER（快衰落）
        time_sel = csi_dict.get('time_selectivity', torch.zeros(B, device=device, dtype=dtype))
        time_penalty = 1.0 + 2.0 * time_sel.clamp(0, 1)  # 最多增加3倍

        # 频率选择性增加BER（多径衰落）
        freq_sel = csi_dict.get('freq_selectivity', torch.zeros(B, device=device, dtype=dtype))
        freq_penalty = 1.0 + 1.5 * freq_sel.clamp(0, 1)  # 最多增加2.5倍

        # LOS比例降低BER（视距传播更可靠）
        los_ratio = csi_dict.get('los_ratio', torch.ones(B, device=device, dtype=dtype))
        los_benefit = 1.0 - 0.5 * los_ratio.clamp(0, 1)  # 最多降低50%

        # 综合BER计算
        total_ber = base_ber * time_penalty * freq_penalty * los_benefit
        return total_ber.clamp(1e-6, 0.5)  # 限制在合理范围内

    def apply_random_errors(self, indices: torch.Tensor, ber: float, codebook_size: int) -> torch.Tensor:
        """
        对量化索引应用随机比特错误

        Args:
            indices: [B, T] 量化索引
            ber: 比特错误率
            codebook_size: 码书大小

        Returns:
            corrupted_indices: [B, T] 损坏后的索引
        """
        B, T = indices.shape
        device = indices.device

        # 计算每个索引需要的比特数
        bits_per_index = math.ceil(math.log2(codebook_size))

        # 生成错误掩码
        error_prob = ber * bits_per_index  # 每个索引的错误概率
        error_mask = torch.rand(B, T, device=device) < error_prob

        # 生成随机错误索引
        error_indices = torch.randint(0, codebook_size, (B, T), device=device)

        # 应用错误
        corrupted = torch.where(error_mask, error_indices, indices)
        return corrupted

    def apply_burst_errors(self, indices: torch.Tensor, ber: float, codebook_size: int,
                          burst_length: int = 5) -> torch.Tensor:
        """
        对量化索引应用突发错误

        Args:
            indices: [B, T] 量化索引
            ber: 平均比特错误率
            codebook_size: 码书大小
            burst_length: 突发错误长度

        Returns:
            corrupted_indices: [B, T] 损坏后的索引
        """
        B, T = indices.shape
        device = indices.device
        corrupted = indices.clone()

        # 突发错误概率调整（更少但更集中的错误）
        burst_prob = ber * 10.0  # 突发开始概率

        for b in range(B):
            t = 0
            while t < T:
                if torch.rand(1).item() < burst_prob:
                    # 开始一个突发错误
                    burst_end = min(t + burst_length, T)
                    error_indices = torch.randint(0, codebook_size, (burst_end - t,), device=device)
                    corrupted[b, t:burst_end] = error_indices
                    t = burst_end
                else:
                    t += 1

        return corrupted

    def apply_erasure_errors(self, indices: torch.Tensor, ber: float,
                           erasure_value: int = -1) -> torch.Tensor:
        """
        对量化索引应用擦除错误（丢失的索引用特殊值标记）

        Args:
            indices: [B, T] 量化索引
            ber: 擦除概率
            erasure_value: 擦除标记值

        Returns:
            corrupted_indices: [B, T] 损坏后的索引
        """
        B, T = indices.shape
        device = indices.device

        # 擦除掩码
        erasure_mask = torch.rand(B, T, device=device) < ber

        # 应用擦除
        corrupted = torch.where(erasure_mask,
                               torch.full_like(indices, erasure_value),
                               indices)
        return corrupted

    def apply_channel_to_rvq_indices(self, stage_indices: List[torch.Tensor],
                                   csi_dict: Optional[Dict[str, torch.Tensor]],
                                   codebook_sizes: List[int],
                                   error_mode: str = "random") -> List[torch.Tensor]:
        """
        对RVQ多阶段索引应用信道模拟

        Args:
            stage_indices: List of [B, T] 每阶段的量化索引
            csi_dict: CSI字典
            codebook_sizes: 每阶段的码书大小
            error_mode: 错误模式 ("random", "burst", "erasure")

        Returns:
            corrupted_stage_indices: 损坏后的多阶段索引
        """
        if csi_dict is None:
            return stage_indices

        try:
            # 计算每个批次的BER
            ber_per_batch = self.csi_to_ber(csi_dict)  # [B]

            corrupted_indices = []

            for stage_idx, indices in enumerate(stage_indices):
                B, T = indices.shape
                codebook_size = codebook_sizes[min(stage_idx, len(codebook_sizes)-1)]

                # 为每个批次应用不同的BER
                stage_corrupted = indices.clone()

                for b in range(B):
                    batch_indices = indices[b:b+1, :]  # [1, T]
                    batch_ber = ber_per_batch[b].item()

                    if error_mode == "random":
                        batch_corrupted = self.apply_random_errors(batch_indices, batch_ber, codebook_size)
                    elif error_mode == "burst":
                        batch_corrupted = self.apply_burst_errors(batch_indices, batch_ber, codebook_size)
                    elif error_mode == "erasure":
                        batch_corrupted = self.apply_erasure_errors(batch_indices, batch_ber)
                    else:
                        batch_corrupted = batch_indices

                    stage_corrupted[b:b+1, :] = batch_corrupted

                corrupted_indices.append(stage_corrupted)

            return corrupted_indices

        except Exception as e:
            print(f"Warning: JSCC channel simulation failed: {e}")
            return stage_indices

    def get_channel_stats(self, original_indices: List[torch.Tensor],
                         corrupted_indices: List[torch.Tensor]) -> Dict[str, float]:
        """
        计算信道损坏统计信息

        Returns:
            stats: 包含错误率、损坏帧数等统计信息
        """
        total_errors = 0
        total_symbols = 0
        corrupted_frames = 0
        total_frames = 0

        for orig_stage, corr_stage in zip(original_indices, corrupted_indices):
            # 计算符号错误
            errors = (orig_stage != corr_stage).sum().item()
            symbols = orig_stage.numel()

            # 计算帧错误
            frame_errors = (orig_stage != corr_stage).any(dim=1).sum().item()
            frames = orig_stage.shape[0]

            total_errors += errors
            total_symbols += symbols
            corrupted_frames += frame_errors
            total_frames += frames

        return {
            'symbol_error_rate': total_errors / max(1, total_symbols),
            'frame_error_rate': corrupted_frames / max(1, total_frames),
            'total_errors': total_errors,
            'total_symbols': total_symbols
        }


def create_jscc_channel_simulator() -> JSCCChannelSimulator:
    """创建JSCC信道模拟器的工厂函数"""
    return JSCCChannelSimulator()