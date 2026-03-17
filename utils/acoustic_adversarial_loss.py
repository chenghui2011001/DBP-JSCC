#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
声学特征对抗损失模块

专门针对20维声学特征设计的轻量级判别器，
解决MSE损失抹平特征分布的问题，保持动态范围和统计特性。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class AcousticFeatureDiscriminator(nn.Module):
    """
    轻量级声学特征判别器

    设计原则：
    1. 轻量级：参数量小，不影响主训练效率
    2. 时序感知：利用卷积捕获时序模式
    3. 多尺度：不同层次的特征判别
    4. 谱感知：针对倒谱、基频、清浊音分别判别
    """

    def __init__(
        self,
        input_dim: int = 20,
        hidden_dim: int = 64,
        num_layers: int = 3,
        use_spectral_norm: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 时序卷积层：捕获时间依赖
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim // 2, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
        )

        # 维度特化分支：针对不同类型特征
        self.cepstral_branch = nn.Sequential(  # 倒谱系数 [0-17]
            nn.Linear(18, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(32, 16)
        )

        self.prosodic_branch = nn.Sequential(  # 基频+清浊音 [18-19]
            nn.Linear(2, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 8)
        )

        # 融合判别器
        fusion_input_dim = hidden_dim + 16 + 8  # temporal + cepstral + prosodic

        layers = []
        current_dim = fusion_input_dim

        for i in range(num_layers):
            next_dim = hidden_dim // (2 ** i)
            if next_dim < 8:
                next_dim = 8

            linear = nn.Linear(current_dim, next_dim)
            if use_spectral_norm:
                linear = nn.utils.spectral_norm(linear)

            layers.extend([
                linear,
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout)
            ])
            current_dim = next_dim

        # 最终判别层
        final_linear = nn.Linear(current_dim, 1)
        if use_spectral_norm:
            final_linear = nn.utils.spectral_norm(final_linear)
        layers.append(final_linear)

        self.discriminator = nn.Sequential(*layers)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Xavier初始化，保持训练稳定性"""
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, T, 20] 声学特征
        Returns:
            scores:   [B, T, 1] 判别分数
            features: List[Tensor] 判别中间特征（用于FM）
        """
        B, T, D = x.shape
        assert D == self.input_dim, f"Expected {self.input_dim}D input, got {D}D"

        # 1. 时序卷积：[B, T, 20] -> [B, 20, T] -> [B, hidden_dim, T]
        x_temporal = self.temporal_conv(x.transpose(1, 2))  # [B, hidden_dim, T]
        x_temporal = x_temporal.transpose(1, 2)  # [B, T, hidden_dim]

        # 2. 维度特化分支
        # 倒谱分支 [B, T, 18]
        x_cepstral = self.cepstral_branch(x[:, :, :18])  # [B, T, 16]

        # 韵律分支 [B, T, 2]
        x_prosodic = self.prosodic_branch(x[:, :, 18:20])  # [B, T, 8]

        # 3. 特征融合
        x_fused = torch.cat([x_temporal, x_cepstral, x_prosodic], dim=-1)  # [B, T, fusion_dim]

        # 4. 最终判别（同时收集中间特征）
        feats = [x_temporal, x_cepstral, x_prosodic, x_fused]
        h = x_fused
        for layer in self.discriminator:
            h = layer(h)
            # 记录激活特征（跳过最后的线性输出，保留非线性后的隐藏层）
            if isinstance(layer, nn.LeakyReLU):
                feats.append(h)
        scores = h  # [B, T, 1]
        return scores, feats


class AcousticAdversarialLoss(nn.Module):
    """
    声学特征对抗损失计算器

    包含生成器损失、判别器损失、以及梯度惩罚等高级技术
    """

    def __init__(
        self,
        discriminator: AcousticFeatureDiscriminator,
        recon_weight: float = 1.0,
        adv_weight: float = 0.1,
        fm_weight: float = 5.0,
        gp_weight: float = 10.0,  # 梯度惩罚权重
        use_gradient_penalty: bool = True,
        use_r1_penalty: bool = False,
        label_smoothing: float = 0.1
    ):
        super().__init__()

        self.discriminator = discriminator
        self.recon_weight = recon_weight
        self.adv_weight = adv_weight
        self.gp_weight = gp_weight
        self.fm_weight = fm_weight
        self.use_gradient_penalty = use_gradient_penalty
        self.use_r1_penalty = use_r1_penalty
        self.label_smoothing = label_smoothing

        # 移动平均跟踪判别器损失，用于自适应权重
        self.register_buffer('disc_loss_ema', torch.tensor(0.0))
        self.register_buffer('gen_loss_ema', torch.tensor(0.0))
        self.ema_decay = 0.99

    def compute_generator_loss(
        self,
        pred_acoustic: torch.Tensor,
        target_acoustic: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算生成器损失（用于双头模块训练）

        Args:
            pred_acoustic: [B, T, 20] 生成的声学特征
            target_acoustic: [B, T, 20] 真实声学特征

        Returns:
            total_loss: 总损失
            metrics: 损失分解字典
        """
        # 1. 基础重建损失（多种组合）
        recon_loss = self._compute_reconstruction_loss(pred_acoustic, target_acoustic)

        # 2. 对抗损失（WGAN-GP 形式）：G 目标最小化 -E[D(fake)]
        fake_scores, fake_feats = self.discriminator(pred_acoustic)
        adv_loss = -fake_scores.mean()

        # 2.1 Feature Matching（声学特征版）
        with torch.no_grad():
            real_scores, real_feats = self.discriminator(target_acoustic)
        fm_loss = 0.0
        n = min(len(fake_feats), len(real_feats))
        if n > 0:
            for i in range(n):
                fm_loss = fm_loss + (fake_feats[i] - real_feats[i]).abs().mean()
            fm_loss = fm_loss / float(n)

        # 3. 自适应权重调整
        # 使用对抗幅度（非符号）做自适应权重基准
        adv_mag = fake_scores.abs().mean().detach()
        adaptive_adv_weight = self._compute_adaptive_weight(adv_mag, recon_loss)

        # 4. 总损失
        total_loss = (
            self.recon_weight * recon_loss +
            adaptive_adv_weight * adv_loss +
            self.fm_weight * fm_loss
        )

        # 更新EMA
        self.gen_loss_ema = self.ema_decay * self.gen_loss_ema + (1 - self.ema_decay) * adv_loss.detach()

        metrics = {
            'recon_loss': recon_loss.item(),
            'adv_loss': float(adv_loss.item()),
            'fm_loss': float(fm_loss if isinstance(fm_loss, float) else fm_loss.item()),
            'adaptive_adv_weight': adaptive_adv_weight,
            'fake_score_mean': fake_scores.mean().item(),
            'fake_score_std': fake_scores.std().item()
        }

        return total_loss, metrics

    def compute_discriminator_loss(
        self,
        pred_acoustic: torch.Tensor,
        target_acoustic: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算判别器损失

        Args:
            pred_acoustic: [B, T, 20] 生成的声学特征（detached）
            target_acoustic: [B, T, 20] 真实声学特征

        Returns:
            total_loss: 判别器损失
            metrics: 损失分解字典
        """
        # 确保生成特征已断开梯度
        pred_acoustic = pred_acoustic.detach()

        # 1-3. Wasserstein 判别器损失：最小化 (E[D(fake)] - E[D(real)])
        real_scores, _ = self.discriminator(target_acoustic)
        fake_scores, _ = self.discriminator(pred_acoustic)
        wasserstein = (fake_scores.mean() - real_scores.mean())
        disc_loss = wasserstein

        # 4. 梯度惩罚（可选）
        gp_loss = torch.tensor(0.0, device=disc_loss.device)
        if self.use_gradient_penalty:
            gp_loss = self._gradient_penalty(target_acoustic, pred_acoustic)

        # 5. R1正则化（可选）
        r1_loss = torch.tensor(0.0, device=disc_loss.device)
        if self.use_r1_penalty:
            r1_loss = self._r1_penalty(target_acoustic)

        # 6. 总损失
        total_loss = disc_loss + self.gp_weight * gp_loss + r1_loss

        # 更新EMA
        self.disc_loss_ema = self.ema_decay * self.disc_loss_ema + (1 - self.ema_decay) * disc_loss.detach()

        metrics = {
            'disc_loss': disc_loss.item(),
            'wasserstein': wasserstein.item(),
            'gp_loss': gp_loss.item(),
            'r1_loss': r1_loss.item(),
            'real_score_mean': real_scores.mean().item(),
            'fake_score_mean': fake_scores.mean().item(),
            'real_score_std': real_scores.std().item(),
            'fake_score_std': fake_scores.std().item()
        }

        return total_loss, metrics

    def _compute_reconstruction_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """组合重建损失：L1 + 标准差保持"""
        # 基础L1损失
        l1_loss = F.l1_loss(pred, target)

        # 标准差保持损失（防止MSE抹平效应）
        pred_std = pred.std(dim=[0, 1])
        target_std = target.std(dim=[0, 1])
        std_loss = F.mse_loss(pred_std, target_std)

        return l1_loss + 2.0 * std_loss  # 增强std保持权重

    def _compute_adaptive_weight(
        self,
        adv_loss: torch.Tensor,
        recon_loss: torch.Tensor
    ) -> float:
        """自适应调整对抗损失权重，平衡重建质量和对抗训练"""
        # 基于损失比例的自适应权重
        ratio = (adv_loss / (recon_loss + 1e-8)).item()

        # 如果对抗损失相对较小，增加权重；否则减少权重
        if ratio < 0.1:
            adaptive_weight = self.adv_weight * 2.0
        elif ratio > 1.0:
            adaptive_weight = self.adv_weight * 0.5
        else:
            adaptive_weight = self.adv_weight

        return max(0.01, min(0.5, adaptive_weight))

    def _gradient_penalty(
        self,
        real_data: torch.Tensor,
        fake_data: torch.Tensor
    ) -> torch.Tensor:
        """WGAN-GP风格的梯度惩罚"""
        B, T, D = real_data.shape

        # 随机插值
        epsilon = torch.rand(B, T, 1, device=real_data.device)
        interpolated = epsilon * real_data + (1 - epsilon) * fake_data
        interpolated.requires_grad_(True)

        # 计算判别器输出
        interpolated_scores, _ = self.discriminator(interpolated)

        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=interpolated_scores,
            inputs=interpolated,
            grad_outputs=torch.ones_like(interpolated_scores),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # 梯度范数惩罚
        gradient_norm = gradients.norm(2, dim=-1)  # [B, T]
        penalty = ((gradient_norm - 1) ** 2).mean()

        return penalty

    def _r1_penalty(self, real_data: torch.Tensor) -> torch.Tensor:
        """R1正则化：在真实数据上的梯度惩罚"""
        real_data.requires_grad_(True)
        real_scores, _ = self.discriminator(real_data)

        gradients = torch.autograd.grad(
            outputs=real_scores.sum(),
            inputs=real_data,
            create_graph=True,
            retain_graph=True
        )[0]

        penalty = (gradients ** 2).sum(dim=-1).mean()
        return penalty * 0.5


def create_acoustic_adversarial_loss(
    input_dim: int = 20,
    device: torch.device = torch.device('cuda'),
    **kwargs
) -> AcousticAdversarialLoss:
    """
    创建声学特征对抗损失的便捷函数

    Args:
        input_dim: 输入特征维度（默认20）
        device: 设备
        **kwargs: 其他配置参数

    Returns:
        配置好的对抗损失模块
    """
    discriminator = AcousticFeatureDiscriminator(
        input_dim=input_dim,
        hidden_dim=kwargs.get('hidden_dim', 64),
        num_layers=kwargs.get('num_layers', 3),
        use_spectral_norm=kwargs.get('use_spectral_norm', True),
        dropout=kwargs.get('dropout', 0.1)
    ).to(device)

    adversarial_loss = AcousticAdversarialLoss(
        discriminator=discriminator,
        recon_weight=kwargs.get('recon_weight', 1.0),
        adv_weight=kwargs.get('adv_weight', 0.1),
        fm_weight=kwargs.get('fm_weight', 5.0),
        gp_weight=kwargs.get('gp_weight', 10.0),
        use_gradient_penalty=kwargs.get('use_gradient_penalty', True),
        use_r1_penalty=kwargs.get('use_r1_penalty', False),
        label_smoothing=kwargs.get('label_smoothing', 0.1)
    )

    return adversarial_loss


if __name__ == "__main__":
    pass
