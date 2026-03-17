#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BFCC-based neural vocoders.

本文件现在同时提供两种声码器实现：

1. :class:`BFCCVocoder`  —— 旧版，直接包装 FARGANCore，自回归 GRU
   结构，保留与原始 FARGAN 完全一致的子帧解码逻辑；
2. :class:`BFCCConvVocoder` —— 新版，**完全卷积式 HiFi-GAN 风格**
   声码器，不再依赖 FARGANCore，自底向上为 32BFCC+F0+VUV 设计。

出于兼容性考虑，我们保留 :class:`BFCCVocoder` 供旧脚本使用；而
新的训练脚本（例如 ``training/bfcc_vocoder_train.py``）默认使用
更轻量、训练更稳定的 :class:`BFCCConvVocoder`。
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from .vocoder_components import FARGANCore


class BFCCVocoder(nn.Module):
    """BFCC-based vocoder using :class:`FARGANCore` as waveform generator.

    This class mirrors :class:`models.vocoder_decoder.FARGANDecoder` but
    changes the feature interface to

    - 32 Bark log-energy bands (BFCC-style, log-domain), plus
    - DNN pitch (``dnn_pitch``) and frame correlation (``frame_corr``).

    Args:
        bfcc_subframe_size: Size of each subframe in samples (default: 40).
        bfcc_nb_subframes:  Number of subframes per frame (default: 4).
        frame_rate_hz:      Frame rate of the conditioning features
                            (default: 100 Hz ≈ 16000/160).
        soft_period:        If True, always compute period from
                            ``dnn_pitch``; otherwise expects explicit
                            ``period_override``.
    """

    def __init__(
        self,
        bfcc_subframe_size: int = 40,
        bfcc_nb_subframes: int = 4,
        frame_rate_hz: float = 100.0,
        soft_period: bool = True,
    ) -> None:
        super().__init__()
        self.frame_rate_hz = frame_rate_hz
        self.frame_size = bfcc_subframe_size * bfcc_nb_subframes  # 160
        self.soft_period = bool(soft_period)

        # FARGAN core with widened feature_dim (32 BFCC + 2 F0/VUV).
        # The internal subframe network is unchanged; only the
        # conditioning network will see the larger feature_dim.
        self.core = FARGANCore(
            subframe_size=bfcc_subframe_size,
            nb_subframes=bfcc_nb_subframes,
            feature_dim=34,  # 32 BFCC + dnn_pitch + frame_corr
            cond_size=256,
        )

    @staticmethod
    def _to_period_from_dnn_pitch(dp: torch.Tensor) -> torch.Tensor:
        """Map DNN pitch to FARGAN-style period in [32,255].

        ``dp`` is assumed to follow the usual DNN pitch convention used in
        FARGAN/Aether: ``period = 256 / 2**(dp + 1.5)`` with clamping.
        """

        # Ensure shape [B,T]
        if dp.dim() == 3 and dp.size(-1) == 1:
            dp = dp.squeeze(-1)
        dp = torch.nan_to_num(dp, nan=0.0).clamp(-4.0, 4.0)
        period = 256.0 / torch.pow(2.0, dp + 1.5)
        period = torch.clamp(period, 32.0, 255.0)
        return period

    def forward(
        self,
        bfcc32: torch.Tensor,
        dnn_pitch: torch.Tensor,
        frame_corr: torch.Tensor,
        *,
        target_len: Optional[int] = None,
        pre: Optional[torch.Tensor] = None,
        period_override: Optional[torch.Tensor] = None,
        override_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Synthesize waveform from BFCC32 + F0/VUV.

        Args:
            bfcc32:     [B,T,32] Bark log-energy (log-BFCC) features.
            dnn_pitch:  [B,T,1] or [B,T] DNN pitch (log-period units).
            frame_corr: [B,T,1] or [B,T] correlation / VUV indicator.
            target_len: Optional target length in samples; if None the
                         decoder will generate as many frames as
                         available from the conditioning network.
            pre:        Optional teacher-forcing prefix signal of shape
                         [B,L_pre].
            period_override: Optional explicit period sequence [B,T].
            override_mask:   Optional mask [B,T] (1=use override).

        Returns:
            period: [B,T] period sequence used for synthesis.
            audio:  [B,1,L] generated waveform.
        """

        if bfcc32.dim() != 3 or bfcc32.size(-1) != 32:
            raise ValueError(
                f"bfcc32 must have shape [B,T,32], got {tuple(bfcc32.shape)}"
            )

        B, T, _ = bfcc32.shape

        # Prepare F0/VUV inputs
        if dnn_pitch.dim() == 2:
            dp = dnn_pitch
        elif dnn_pitch.dim() == 3 and dnn_pitch.size(-1) == 1:
            dp = dnn_pitch.squeeze(-1)
        else:
            raise ValueError(
                f"dnn_pitch must have shape [B,T] or [B,T,1], got {tuple(dnn_pitch.shape)}"
            )

        if frame_corr.dim() == 2:
            fc = frame_corr
        elif frame_corr.dim() == 3 and frame_corr.size(-1) == 1:
            fc = frame_corr.squeeze(-1)
        else:
            raise ValueError(
                f"frame_corr must have shape [B,T] or [B,T,1], got {tuple(frame_corr.shape)}"
            )

        # Clamp / clean F0/VUV to a safe range.
        dp = torch.nan_to_num(dp, nan=0.0).clamp(-3.5, 3.5)
        fc = torch.nan_to_num(fc, nan=0.0).clamp(-0.8, 0.8)

        # Feature stack: [BFCC32, dnn_pitch, frame_corr] -> [B,T,34]
        # Clamp BFCC to a sane dynamic range to avoid exploding logits.
        bfcc32_clamped = torch.clamp(bfcc32, min=-10.0, max=2.0)
        feats = torch.cat(
            [bfcc32_clamped, dp.unsqueeze(-1), fc.unsqueeze(-1)], dim=-1
        )

        if not torch.isfinite(feats).all():
            raise RuntimeError(
                "[NaNGuard] non-finite BFCC/F0/VUV features in BFCCVocoder.forward"
            )

        # Period handling: either from override or from dnn_pitch.
        if period_override is not None or override_mask is not None:
            if period_override is None:
                period_override = self._to_period_from_dnn_pitch(dp)
            po = period_override.to(feats.device)
            if override_mask is None:
                period = torch.clamp(po, 32.0, 255.0)
            else:
                m = override_mask
                if m.dim() == 3 and m.size(-1) == 1:
                    m = m.squeeze(-1)
                if m.shape != po.shape:
                    raise ValueError(
                        f"override_mask shape {tuple(m.shape)} incompatible with period_override {tuple(po.shape)}"
                    )
                m = m.to(dtype=po.dtype, device=po.device).clamp(0.0, 1.0)
                base = self._to_period_from_dnn_pitch(dp).to(po.device)
                period = torch.clamp(m * po + (1.0 - m) * base, 32.0, 255.0)
        else:
            # Default: derive directly from dnn_pitch
            period = self._to_period_from_dnn_pitch(dp).to(feats.device)

        if not torch.isfinite(period).all():
            raise RuntimeError("[NaNGuard] non-finite period in BFCCVocoder.forward")

        # Waveform generation (mirrors FARGANDecoder._generate_waveform).
        audio = self._generate_waveform(feats, period, target_len=target_len, pre=pre)

        return period, audio

    def _generate_waveform(
        self,
        features: torch.Tensor,
        period: torch.Tensor,
        target_len: Optional[int] = None,
        pre: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Internal helper to run :class:`FARGANCore`.

        This is largely copied from ``FARGANDecoder._generate_waveform`` but
        uses the full feature vector instead of slicing to 20 dims.
        """

        B, T, _ = features.shape

        # Maximum number of frames available from the conditioning network:
        max_available_frames = T - 4  # FARGANCond discards the first 4 frames

        # Number of teacher-forcing frames (if ``pre`` is provided).
        nb_pre_frames = pre.size(1) // self.frame_size if pre is not None else 0

        gen_capacity = max(0, max_available_frames - nb_pre_frames)
        if target_len is not None:
            target_frames_total = (target_len + self.frame_size - 1) // self.frame_size
            target_frames_gen = max(0, target_frames_total - nb_pre_frames)
            nb_frames = min(gen_capacity, target_frames_gen)
        else:
            nb_frames = gen_capacity

        nb_frames = max(1, nb_frames)

        # Call FARGANCore with widened feature_dim.
        audio, _states = self.core(features, period, nb_frames, pre=pre)

        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        if target_len is not None and audio.size(-1) > target_len:
            audio = audio[..., : target_len]

        return audio


__all__ = ["BFCCVocoder"]


def load_bfcc_vocoder_from_fargan(
    bfcc_vocoder: BFCCVocoder,
    fargan_state: Dict[str, Any] | str,
) -> BFCCVocoder:
    """Warm-start a :class:`BFCCVocoder` from FARGAN weights.

    支持两种常见的权重布局：

    1. 直接来自 FARGANCore/FARGANDecoder 的 ``state_dict``，键为
       ``cond_net.*`` / ``sig_net.*``；
    2. Stage2.x checkpoint 中嵌套在 ``fargan_core.*`` 下的权重
       （键前缀为 ``fargan_core.cond_net.*`` 等）。

    对于所有在 BFCCVocoder.core 中具有**完全相同形状**的参数，
    复制其值；例如 GRU、skip、GLU 等子模块。输入维度发生变化
    的层（典型如 ``cond_net.fdense1`` 等）会被自动跳过，保持
    BFCCVocoder 自己的初始化。

    Args:
        bfcc_vocoder: 目标 BFCCVocoder 实例。
        fargan_state: FARGANDecoder/FARGANCore 的 state_dict 或
                      checkpoint 路径；若为 checkpoint dict，则
                      优先读取其中的 ``"state_dict"`` 键。

    Returns:
        原始 ``bfcc_vocoder`` 实例，便于链式调用。
    """

    if isinstance(fargan_state, str):
        ck = torch.load(fargan_state, map_location="cpu")
        if isinstance(ck, dict) and "state_dict" in ck:
            fargan_sd = ck["state_dict"]
        else:
            fargan_sd = ck
    else:
        fargan_sd = fargan_state

    bfcc_sd = bfcc_vocoder.state_dict()
    copied, skipped = 0, 0

    # 检查 key 布局：优先处理带 "fargan_core." 前缀的 checkpoint，
    # 否则视作 FARGANCore 本身的 state_dict（以 cond_net./sig_net. 开头）。
    has_fargan_core_prefix = any(k.startswith("fargan_core.") for k in fargan_sd.keys())

    if has_fargan_core_prefix:
        # Stage2.x 风格：fargan_core.* → core.*
        for k_old, v_old in fargan_sd.items():
            if not isinstance(v_old, torch.Tensor):
                continue
            if not k_old.startswith("fargan_core."):
                continue
            subkey = k_old[len("fargan_core.") :]
            k_new = "core." + subkey

            v_new = bfcc_sd.get(k_new, None)
            if v_new is None or not isinstance(v_new, torch.Tensor):
                skipped += 1
                continue
            if v_new.shape != v_old.shape:
                skipped += 1
                continue

            bfcc_sd[k_new] = v_old
            copied += 1
    else:
        # 直接来自 FARGANCore/FARGANDecoder：cond_net.* / sig_net.*
        for k_old, v_old in fargan_sd.items():
            if not isinstance(v_old, torch.Tensor):
                continue
            if not (k_old.startswith("cond_net.") or k_old.startswith("sig_net.")):
                continue

            k_new = "core." + k_old
            v_new = bfcc_sd.get(k_new, None)
            if v_new is None or not isinstance(v_new, torch.Tensor):
                skipped += 1
                continue
            if v_new.shape != v_old.shape:
                skipped += 1
                continue

            bfcc_sd[k_new] = v_old
            copied += 1

    bfcc_vocoder.load_state_dict(bfcc_sd)
    try:
        print(f"[BFCCVocoder] Warm-start from FARGAN: copied={copied}, skipped={skipped}")
    except Exception:
        pass

    return bfcc_vocoder


__all__.append("load_bfcc_vocoder_from_fargan")


# ---------------------------------------------------------------------------
# 新架构：完全卷积式 BFCC 声码器（不依赖 FARGANCore）
# ---------------------------------------------------------------------------


class ResBlock1D(nn.Module):
    """简化版 HiFi-GAN Residual Block。

    使用一组不同膨胀率的 1D 卷积，堆叠残差连接来扩大感受野，捕获
    局部时域结构，但保持计算量相对可控。
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: Tuple[int, int, int] = (1, 3, 5),
    ) -> None:
        super().__init__()
        convs = []
        for d in dilations:
            padding = (kernel_size * d - d) // 2
            conv = weight_norm(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    dilation=d,
                    padding=padding,
                )
            )
            convs.append(conv)
        self.convs = nn.ModuleList(convs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for conv in self.convs:
            y = F.leaky_relu(out, negative_slope=0.1)
            y = conv(y)
            out = out + y
        return out


class BFCCConvVocoder(nn.Module):
    """Fully-convolutional BFCC vocoder (HiFi-GAN 风格)。

    输入：
        - 32 维 BFCC (log Bark energy) [B,T,32]
        - dnn_pitch [B,T] or [B,T,1]
        - frame_corr [B,T] or [B,T,1]

    输出：
        - audio [B,1,L]，其中 L≈T*160（16kHz, 10ms 帧移）。

    结构概览：
        34 维特征 → Conv1d 前端 → 逐级上采样 (8×5×4=160) → 残差块堆叠 →
        输出单通道波形。整个网络不包含 RNN/自回归，完全卷积，训练
        和推理都显著快于 FARGANCore 版本。
    """

    def __init__(
        self,
        feature_dim: int = 34,
        upsample_rates: Tuple[int, int, int] = (8, 5, 4),
        upsample_kernel_sizes: Tuple[int, int, int] = (16, 10, 8),
        hidden_channels: int = 128,
    ) -> None:
        super().__init__()
        assert len(upsample_rates) == len(upsample_kernel_sizes)
        self.feature_dim = feature_dim
        self.upsample_rates = upsample_rates

        # 预处理：将 34 维特征映射到隐藏通道
        self.input_conv = weight_norm(
            nn.Conv1d(feature_dim, hidden_channels, kernel_size=3, padding=1)
        )

        # 上采样 + 残差块
        ups = []
        resblocks = []
        in_ch = hidden_channels
        for r, k in zip(upsample_rates, upsample_kernel_sizes):
            # ConvTranspose1d: stride=r, kernel_size=k，padding 使输出长度≈输入×r
            padding = (k - r) // 2
            conv_t = weight_norm(
                nn.ConvTranspose1d(
                    in_ch,
                    in_ch // 2,
                    kernel_size=k,
                    stride=r,
                    padding=padding,
                )
            )
            ups.append(conv_t)
            resblocks.append(
                nn.ModuleList(
                    [
                        ResBlock1D(in_ch // 2, kernel_size=3, dilations=(1, 3, 5)),
                        ResBlock1D(in_ch // 2, kernel_size=5, dilations=(1, 3, 5)),
                    ]
                )
            )
            in_ch = in_ch // 2

        self.ups = nn.ModuleList(ups)
        self.resblocks = nn.ModuleList(resblocks)

        # 输出层
        self.output_conv = weight_norm(
            nn.Conv1d(in_ch, 1, kernel_size=7, padding=3)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")

    @staticmethod
    def _merge_features(
        bfcc32: torch.Tensor,
        dnn_pitch: torch.Tensor,
        frame_corr: torch.Tensor,
    ) -> torch.Tensor:
        """将 32BFCC + F0 + VUV 合并为 [B,T,34]，并做适度裁剪。"""

        if bfcc32.dim() != 3 or bfcc32.size(-1) != 32:
            raise ValueError(
                f"bfcc32 must have shape [B,T,32], got {tuple(bfcc32.shape)}"
            )

        if dnn_pitch.dim() == 3 and dnn_pitch.size(-1) == 1:
            dnn_pitch = dnn_pitch.squeeze(-1)
        if frame_corr.dim() == 3 and frame_corr.size(-1) == 1:
            frame_corr = frame_corr.squeeze(-1)
        if dnn_pitch.dim() != 2 or frame_corr.dim() != 2:
            raise ValueError("dnn_pitch/frame_corr must be [B,T] or [B,T,1]")

        B, T, _ = bfcc32.shape
        if dnn_pitch.shape != (B, T) or frame_corr.shape != (B, T):
            # 对齐时间维到最小长度
            T_use = min(T, dnn_pitch.size(1), frame_corr.size(1))
            bfcc32 = bfcc32[:, :T_use, :]
            dnn_pitch = dnn_pitch[:, :T_use]
            frame_corr = frame_corr[:, :T_use]
            B, T, _ = bfcc32.shape

        bfcc32 = torch.clamp(bfcc32, min=-10.0, max=2.0)
        dp = torch.nan_to_num(dnn_pitch, nan=0.0).clamp(-3.5, 3.5)
        fc = torch.nan_to_num(frame_corr, nan=0.0).clamp(-0.8, 0.8)

        feats = torch.cat(
            [bfcc32, dp.unsqueeze(-1), fc.unsqueeze(-1)], dim=-1
        )  # [B,T,34]
        return feats

    def forward(
        self,
        bfcc32: torch.Tensor,
        dnn_pitch: torch.Tensor,
        frame_corr: torch.Tensor,
        *,
        target_len: Optional[int] = None,
    ) -> torch.Tensor:
        """前向：BFCC32 + F0/VUV → 波形。

        Args:
            bfcc32: [B,T,32] log-BFCC 特征。
            dnn_pitch: [B,T] 或 [B,T,1] DNN 基频。
            frame_corr: [B,T] 或 [B,T,1] 相关系数 / VUV 指示。
            target_len: 目标长度（采样点数），若为 None 则按照
                        特征长度 * 160 生成后再裁剪。
        Returns:
            audio: [B,1,L] 波形。
        """

        feats = self._merge_features(bfcc32, dnn_pitch, frame_corr)
        x = feats.transpose(1, 2)  # [B,34,T] → [B,34,T]

        x = self.input_conv(x)

        for conv_t, rb_group in zip(self.ups, self.resblocks):
            x = F.leaky_relu(x, negative_slope=0.1)
            x = conv_t(x)
            # 多个 ResBlock 取平均输出（HiFi-GAN 做法）
            rb_out = 0.0
            for rb in rb_group:
                rb_out = rb_out + rb(x)
            x = rb_out / float(len(rb_group))

        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.output_conv(x)
        x = torch.tanh(x)  # [-1,1]

        # [B,1,L]
        audio = x
        if target_len is not None and audio.size(-1) > target_len:
            audio = audio[..., :target_len]
        return audio


__all__.append("BFCCConvVocoder")
