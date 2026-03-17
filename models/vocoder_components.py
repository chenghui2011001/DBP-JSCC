# -*- coding: utf-8 -*-
"""
FARGAN Components for AETHER Integration
改编自原始FARGAN代码，适配AETHER架构
"""

from __future__ import annotations
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# 使用旧版 weight_norm（生成 weight_g / weight_v），以便与 Stage2 权重完全对齐
from torch.nn.utils import weight_norm
from typing import Tuple, Optional, Dict, Any


def add_quantization_noise(x: torch.Tensor, training: bool = True) -> torch.Tensor:
    """添加量化噪声 (对应原始FARGAN的n函数)"""
    if not training:
        return x
    noise = (1.0 / 127.0) * (torch.rand_like(x) - 0.5)
    return torch.clamp(x + noise, min=-1.0, max=1.0)


class GLU(nn.Module):
    """门控线性单元 (Gated Linear Unit)"""

    def __init__(self, feat_size: int):
        super().__init__()
        torch.manual_seed(5)  # 保持与原始FARGAN一致的随机种子
        self.gate = weight_norm(nn.Linear(feat_size, feat_size, bias=False))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear, nn.Embedding)):
                nn.init.orthogonal_(m.weight.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.gate(x))


class FWConv(nn.Module):
    """前向卷积模块 (Frame-wise Convolution)"""

    def __init__(self, in_size: int, out_size: int, kernel_size: int = 2):
        super().__init__()
        torch.manual_seed(5)
        self.in_size = in_size
        self.kernel_size = kernel_size
        self.conv = weight_norm(nn.Linear(in_size * kernel_size, out_size, bias=False))
        self.glu = GLU(out_size)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear, nn.Embedding)):
                nn.init.orthogonal_(m.weight.data)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, in_size] 当前输入
            state: [B, in_size*(kernel_size-1)] 历史状态

        Returns:
            output: [B, out_size] 输出
            new_state: [B, in_size*(kernel_size-1)] 更新的状态
        """
        
        w_dtype = self.conv.weight.dtype
        if x.dtype != w_dtype:      x     = x.to(w_dtype)
        if state.dtype != w_dtype:  state = state.to(w_dtype)
        xcat = torch.cat((state, x), -1)  # [B, in_size*kernel_size]
        out = self.glu(torch.tanh(self.conv(xcat)))
        new_state = xcat[:, self.in_size:]  # 更新状态
        return out, new_state


class FARGANCond(nn.Module):
    """FARGAN条件网络 - 将特征转换为条件信号"""

    def __init__(self, feature_dim: int = 20, cond_size: int = 256, pembed_dims: int = 12):
        super().__init__()
        self.feature_dim = feature_dim
        self.cond_size = cond_size

        # 周期嵌入 (Period Embedding)
        self.pembed = nn.Embedding(224, pembed_dims)  # 支持周期32-255

        # 特征处理网络
        self.fdense1 = nn.Linear(self.feature_dim + pembed_dims, 64, bias=False)
        self.fconv1 = nn.Conv1d(64, 128, kernel_size=3, padding='valid', bias=False)
        self.fdense2 = nn.Linear(128, 80 * 4, bias=False)  # 4个子帧

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.GRU):
                for p in m.named_parameters():
                    if p[0].startswith('weight_hh_'):
                        nn.init.orthogonal_(p[1])

    def forward(self, features: torch.Tensor, period: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, T, feature_dim] 输入特征
            period: [B, T] 周期 (32-255)

        Returns:
            cond: [B, T-2, 320] 条件信号 (80*4维)
        """
        # --- 先做时间维稳健对齐，再丢前2帧 ---
        # 对齐到相同的时间长度，避免 features/period 相差2帧导致 cat 报错
        T = min(features.size(1), period.size(1))
        if features.size(1) != T:
            features = features[:, :T, :]
        if period.size(1) != T:
            period = period[:, :T]
        # 去掉前2帧，保持与原始FARGAN一致
        if T > 2:
            features = features[:, 2:, :]
            period = period[:, 2:]
        else:
            # 不足以丢2帧时，直接返回空时间轴（让上游选择跳过本 batch 的 wave loss）
            return features.new_zeros(features.size(0), 0, 80 * 4)

        # 周期嵌入
        w_dtype = self.fdense1.weight.dtype
        period = (period - 32).clamp(0, 223).to(torch.long)     # 索引必须 long
        p = self.pembed(period).to(w_dtype)                     # 与权重同 dtype
        features = features.to(w_dtype)                         # 输入也对齐
        features = torch.cat((features, p), -1)

        # 网络前向
        tmp = torch.tanh(self.fdense1(features))
        tmp = tmp.permute(0, 2, 1)  # [B, 64, T-2]
        tmp = torch.tanh(self.fconv1(tmp))  # [B, 128, T-4]
        tmp = tmp.permute(0, 2, 1)  # [B, T-4, 128]
        tmp = torch.tanh(self.fdense2(tmp))  # [B, T-4, 320]

        return tmp


class FARGANSub(nn.Module):
    """FARGAN子帧网络 - 逐子帧生成音频。

    当 ``compat_mode=True`` 时，尽量复现原始 ``fargan.py`` 中
    ``FARGANSub`` 的行为：

    - 使用 ``gain = exp(cond_gain_dense(cond))``（无显式下限）。
    - 不使用显式 VUV 门控，仅依赖特征自身学习有声/无声。
    - 始终使用原始的量化噪声函数 ``n(x)``（训练/推理一致）。

    默认 ``compat_mode=False``，保持 Aether-lite 中更稳健的改动：
    有界增益、可选 VUV 门控、仅在训练时注入量化噪声。
    """

    def __init__(
        self,
        subframe_size: int = 40,
        nb_subframes: int = 4,
        cond_size: int = 256,
        compat_mode: bool = False,
    ):
        super().__init__()
        self.subframe_size = subframe_size
        self.nb_subframes = nb_subframes
        self.cond_size = cond_size
        self.compat_mode: bool = bool(compat_mode)

        # 条件增益网络
        self.cond_gain_dense = nn.Linear(80, 1)

        # 前向卷积层
        self.fwc0 = FWConv(2 * self.subframe_size + 80 + 4, 192)

        # 三层GRU网络
        self.gru1 = nn.GRUCell(192 + 2 * self.subframe_size, 160, bias=False)
        self.gru2 = nn.GRUCell(160 + 2 * self.subframe_size, 128, bias=False)
        self.gru3 = nn.GRUCell(128 + 2 * self.subframe_size, 128, bias=False)

        # GLU激活
        self.gru1_glu = GLU(160)
        self.gru2_glu = GLU(128)
        self.gru3_glu = GLU(128)
        self.skip_glu = GLU(128)

        # 输出层
        self.skip_dense = nn.Linear(192 + 160 + 2 * 128 + 2 * self.subframe_size, 128, bias=False)
        self.sig_dense_out = nn.Linear(128, self.subframe_size, bias=False)
        self.gain_dense_out = nn.Linear(192, 4)  # 4个基频增益
        # Runtime knobs: prefer module attributes when the training script
        # sets them; otherwise fall back to environment variables for
        # backward-compatible quick experiments.
        self.final_voicing_gate: Optional[bool] = None
        self.final_voicing_gate_floor: Optional[float] = None
        self.final_voicing_gate_gamma: Optional[float] = None
        self.silence_gate_enabled: Optional[bool] = None
        self.silence_gate_floor: Optional[float] = None
        self.pitch_gain_scale: Optional[float] = None
        self.sig_core_scale: Optional[float] = None
        self.last_debug: Optional[Dict[str, torch.Tensor]] = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.GRU):
                for p in m.named_parameters():
                    if p[0].startswith('weight_hh_'):
                        nn.init.orthogonal_(p[1])

    def forward(
        self,
        cond: torch.Tensor,
        prev_pred: torch.Tensor,
        exc_mem: torch.Tensor,
        period: torch.Tensor,
        states: Tuple[torch.Tensor, ...],
        gain: Optional[torch.Tensor] = None,
        vuv: Optional[torch.Tensor] = None,
        silence_gate: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Args:
            cond: [B, 80] 当前子帧条件
            prev_pred: [B, 256] 前一预测信号
            exc_mem: [B, 256] 激励缓冲区
            period: [B] 当前周期
            states: 4个GRU状态元组
            gain: [B, 1] 可选的外部增益
            silence_gate: [B, 1] 可选的静音能量门控

        Returns:
            sig_out: [B, subframe_size] 输出信号
            exc_mem: [B, 256] 更新的激励缓冲区
            prev_pred: [B, 256] 更新的预测信号
            states: 更新的GRU状态
        """
        use_compat = bool(getattr(self, "compat_mode", False))
        run_dtype = self.fwc0.conv.weight.dtype
        # 兜底：若上游忘记提供 states，这里自举零状态
        if states is None:
            B = cond.size(0)
            states = (
                torch.zeros(B, 160, device=cond.device, dtype=run_dtype),
                torch.zeros(B, 128, device=cond.device, dtype=run_dtype),
                torch.zeros(B, 128, device=cond.device, dtype=run_dtype),
                torch.zeros(B, 2*self.subframe_size + 80 + 4, device=cond.device, dtype=run_dtype),
            )

        if cond.dtype      != run_dtype: cond      = cond.to(run_dtype)
        if prev_pred.dtype != run_dtype: prev_pred = prev_pred.to(run_dtype)
        if exc_mem.dtype   != run_dtype: exc_mem   = exc_mem.to(run_dtype)
        # states 是4个张量的tuple
        states = tuple( s.to(run_dtype) if isinstance(s, torch.Tensor) and s.dtype != run_dtype else s
                        for s in states )
        device = exc_mem.device

        # 基频预测 - 从激励缓冲区中提取周期性信号（支持连续周期的线性插值采样）
        rng = torch.arange(self.subframe_size + 4, device=device, dtype=run_dtype)  # [44]
        # 数值健壮化：确保周期/索引无 NaN/Inf 并严格落在界内
        period = torch.nan_to_num(period, nan=128.0, posinf=128.0, neginf=128.0)
        period = torch.clamp(period, 32.0, 255.0)
        if period.dtype.is_floating_point:
            idxf = 256.0 - period[:, None].to(run_dtype)  # [B,1]
            idxf = idxf + rng[None, :] - 2.0             # [B,44]
            idxf = torch.nan_to_num(idxf, nan=0.0)      # 防止 NaN 传播到 gather
            idx0 = torch.floor(idxf)
            idx1 = idx0 + 1.0
            alpha = (idxf - idx0).clamp(0.0, 1.0)
            idx0c = torch.clamp(idx0, 0.0, 255.0).to(torch.long)
            idx1c = torch.clamp(idx1, 0.0, 255.0).to(torch.long)
            # 绝对安全：模 256 防止极端情况下越界
            W = exc_mem.size(1)
            idx0c = torch.remainder(idx0c, W)
            idx1c = torch.remainder(idx1c, W)
            v0 = torch.gather(exc_mem, 1, idx0c)
            v1 = torch.gather(exc_mem, 1, idx1c)
            pred = (1.0 - alpha) * v0 + alpha * v1  # [B,44]
        else:
            idx = 256 - period[:, None]  # [B, 1]
            idx = idx.to(run_dtype)
            idx = idx + rng[None, :] - 2.0  # [B, 44]
            # 处理周期边界
            mask = idx >= 256.0
            idx = idx - mask * period[:, None].to(run_dtype)
            # 处理负索引 - 将负索引设为0
            idx = torch.nan_to_num(idx, nan=0.0)
            idx = torch.clamp(idx, 0.0, 255.0).to(torch.long)
            idx = torch.remainder(idx, exc_mem.size(1))
            # 提取预测信号
            pred = torch.gather(exc_mem, 1, idx)  # [B, 44]
        # 条件增益与激励预处理：在 compat_mode 与默认模式下分支处理。
        if use_compat:
            # 原始 FARGAN 行为：始终使用 n(x)，gain=exp(linear(cond))，无显式上下界。
            cond_n = add_quantization_noise(cond, True)
            if gain is None:
                gain = torch.exp(self.cond_gain_dense(cond_n))
            pred_n = add_quantization_noise(pred / (1e-5 + gain), True)
            prev = exc_mem[:, -self.subframe_size:]
            prev_n = add_quantization_noise(prev / (1e-5 + gain), True)
            tmp = torch.cat((cond_n, pred_n, prev_n), 1)
            fpitch = pred_n[:, 2:-2]
            vuv_gate: Optional[torch.Tensor] = None
        else:
            # Aether-lite 稳健版：条件增益有界，量化噪声仅在训练时注入，支持外部 VUV 门控。
            cond = add_quantization_noise(cond, self.training)
            if gain is None:
                gain_logits = self.cond_gain_dense(cond)
                gain = 0.2 + 0.8 * torch.sigmoid(gain_logits)
            if gain is not None:
                gain = torch.nan_to_num(gain, nan=1.0, posinf=20.0, neginf=1e-3).clamp_(1e-3, 20.0)
            pred = add_quantization_noise(pred / (1e-5 + gain), self.training)
            prev = exc_mem[:, -self.subframe_size:]
            prev = add_quantization_noise(prev / (1e-5 + gain), self.training)
            tmp = torch.cat((cond, pred, prev), 1)
            fpitch = pred[:, 2:-2]

            # 可选：V/UV 门控
            vuv_gate = None
            if vuv is not None:
                vuv_gate = torch.clamp(vuv, 0.0, 1.0).to(run_dtype)
                try:
                    floor = float(os.environ.get("FARGAN_VUV_GATE_FLOOR", "0.0"))
                except Exception:
                    floor = 0.0
                if not math.isfinite(floor):
                    floor = 0.0
                floor = max(0.0, min(1.0, floor))
                if floor > 0.0:
                    vuv_gate = (1.0 - floor) * vuv_gate + floor
                fpitch = fpitch * vuv_gate

        # 前向卷积
        fwc0_out, fwc0_state = self.fwc0(tmp, states[3])
        if use_compat:
            fwc0_out = add_quantization_noise(fwc0_out, True)
        else:
            fwc0_out = add_quantization_noise(fwc0_out, self.training)

        # 基频增益
        pitch_gain = torch.sigmoid(self.gain_dense_out(fwc0_out))  # [B, 4]
        pg_scale_attr = getattr(self, "pitch_gain_scale", None)
        if pg_scale_attr is not None:
            try:
                pg_scale = float(pg_scale_attr)
            except Exception:
                pg_scale = 1.0
            if not math.isfinite(pg_scale) or pg_scale < 0.0:
                pg_scale = 1.0
            if pg_scale != 1.0:
                pitch_gain = torch.clamp(pitch_gain * pg_scale, 0.0, 1.0)
        # Periodic excitation should be VUV-gated only once. ``fpitch`` has
        # already been gated above, so we do not apply the same gate again on
        # ``pitch_gain``; otherwise the effective voiced drive becomes roughly
        # proportional to vuv_gate^2 (and even lower once final voicing gate is
        # enabled), which over-suppresses voiced segments.

        # GRU层级联
        gru1_state = self.gru1(
            torch.cat([fwc0_out, pitch_gain[:, 0:1] * fpitch, prev], 1),
            states[0],
        )
        if use_compat:
            gru1_out = self.gru1_glu(add_quantization_noise(gru1_state, True))
            gru1_out = add_quantization_noise(gru1_out, True)
        else:
            gru1_out = self.gru1_glu(add_quantization_noise(gru1_state, self.training))

        gru2_state = self.gru2(
            torch.cat([gru1_out, pitch_gain[:, 1:2] * fpitch, prev], 1),
            states[1],
        )
        if use_compat:
            gru2_out = self.gru2_glu(add_quantization_noise(gru2_state, True))
            gru2_out = add_quantization_noise(gru2_out, True)
        else:
            gru2_out = self.gru2_glu(add_quantization_noise(gru2_state, self.training))

        gru3_state = self.gru3(
            torch.cat([gru2_out, pitch_gain[:, 2:3] * fpitch, prev], 1),
            states[2],
        )
        if use_compat:
            gru3_out = self.gru3_glu(add_quantization_noise(gru3_state, True))
            gru3_out = add_quantization_noise(gru3_out, True)
        else:
            gru3_out = self.gru3_glu(add_quantization_noise(gru3_state, self.training))

        # 跳跃连接
        gru_concat = torch.cat([gru1_out, gru2_out, gru3_out, fwc0_out], 1)
        skip_input = torch.cat([gru_concat, pitch_gain[:, 3:4] * fpitch, prev], 1)
        skip_out = torch.tanh(self.skip_dense(skip_input))
        if use_compat:
            skip_out = self.skip_glu(add_quantization_noise(skip_out, True))
            skip_out = add_quantization_noise(skip_out, True)
        else:
            skip_out = self.skip_glu(add_quantization_noise(skip_out, self.training))

        # 最终输出。默认在非 compat 模式下启用更强的末端门控：
        # 1) final_voicing_gate：在最终波形前再次用 voiced 概率压制周期泄漏；
        # 2) silence_gate：基于低频能量/c0 的静音门控，压低静音段横纹底噪。
        sig_core = torch.tanh(self.sig_dense_out(skip_out))  # [B, 40]
        sc_scale_attr = getattr(self, "sig_core_scale", None)
        if sc_scale_attr is not None:
            try:
                sc_scale = float(sc_scale_attr)
            except Exception:
                sc_scale = 1.0
            if not math.isfinite(sc_scale) or sc_scale < 0.0:
                sc_scale = 1.0
            if sc_scale != 1.0:
                sig_core = sig_core * sc_scale

        if (not use_compat) and vuv_gate is not None:
            final_v_gate_attr = getattr(self, "final_voicing_gate", None)
            if final_v_gate_attr is None:
                try:
                    use_final_v_gate = os.environ.get("FARGAN_FINAL_VOICING_GATE", "1") != "0"
                except Exception:
                    use_final_v_gate = True
            else:
                use_final_v_gate = bool(final_v_gate_attr)
            if use_final_v_gate:
                v_floor_attr = getattr(self, "final_voicing_gate_floor", None)
                if v_floor_attr is None:
                    try:
                        v_floor = float(os.environ.get("FARGAN_FINAL_VOICING_GATE_FLOOR", "0.0"))
                    except Exception:
                        v_floor = 0.0
                else:
                    v_floor = float(v_floor_attr)
                v_gamma_attr = getattr(self, "final_voicing_gate_gamma", None)
                if v_gamma_attr is None:
                    try:
                        v_gamma = float(os.environ.get("FARGAN_FINAL_VOICING_GATE_GAMMA", "1.0"))
                    except Exception:
                        v_gamma = 1.0
                else:
                    v_gamma = float(v_gamma_attr)
                v_floor = max(0.0, min(1.0, v_floor))
                if not math.isfinite(v_gamma) or v_gamma <= 0.0:
                    v_gamma = 1.0
                final_v_gate = (1.0 - v_floor) * vuv_gate + v_floor
                final_v_gate = final_v_gate.pow(v_gamma)
                sig_core = sig_core * final_v_gate

        sig_out = sig_core * gain

        if (not use_compat) and silence_gate is not None:
            sil_gate_attr = getattr(self, "silence_gate_enabled", None)
            if sil_gate_attr is None:
                try:
                    use_sil_gate = os.environ.get("FARGAN_SILENCE_GATE", "1") != "0"
                except Exception:
                    use_sil_gate = True
            else:
                use_sil_gate = bool(sil_gate_attr)
            if use_sil_gate:
                s_floor_attr = getattr(self, "silence_gate_floor", None)
                if s_floor_attr is None:
                    try:
                        s_floor = float(os.environ.get("FARGAN_SILENCE_GATE_FLOOR", "0.0"))
                    except Exception:
                        s_floor = 0.0
                else:
                    s_floor = float(s_floor_attr)
                s_floor = max(0.0, min(1.0, s_floor))
                silence_gate = torch.clamp(silence_gate, 0.0, 1.0).to(sig_out.dtype)
                if s_floor > 0.0:
                    silence_gate = (1.0 - s_floor) * silence_gate + s_floor
                sig_out = sig_out * silence_gate

        with torch.no_grad():
            self.last_debug = {
                "pitch_gain_mean": pitch_gain.detach().mean(dim=-1),
                "fwc0_rms": fwc0_out.detach().pow(2).mean(dim=-1).sqrt(),
                "skip_rms": skip_out.detach().pow(2).mean(dim=-1).sqrt(),
                "sig_core_rms": sig_core.detach().pow(2).mean(dim=-1).sqrt(),
                "sig_out_rms": sig_out.detach().pow(2).mean(dim=-1).sqrt(),
            }

        # 更新缓冲区
        exc_mem = torch.cat([exc_mem[:, self.subframe_size:], sig_out], 1)
        prev_pred = torch.cat([prev_pred[:, self.subframe_size:], fpitch], 1)

        new_states = (gru1_state, gru2_state, gru3_state, fwc0_state)

        return sig_out, exc_mem, prev_pred, new_states


class FARGANCore(nn.Module):
    """FARGAN核心模块 - 组合条件网络和子帧网络"""

    def __init__(
        self,
        subframe_size: int = 40,
        nb_subframes: int = 4,
        feature_dim: int = 20,
        cond_size: int = 256,
        strict_vuv_gate: bool = True,
        compat_mode: bool = False,
    ):
        super().__init__()
        self.subframe_size = subframe_size
        self.nb_subframes = nb_subframes
        self.frame_size = self.subframe_size * self.nb_subframes
        self.feature_dim = feature_dim
        self.cond_size = cond_size
        # When True, map frame_corr ∈ [-0.8,0.5] linearly to vuv ∈ [0,1]
        # so that frame_corr≈-0.8 leads to a fully closed gate. Default-on:
        # Aether-lite prioritises suppressing periodic leakage in clearly
        # unvoiced frames. Can still be overridden by env at runtime.
        self.strict_vuv_gate: bool = bool(strict_vuv_gate)
        # When True, ask the subframe network to behave as close as
        # possible to the original FARGAN implementation (no explicit
        # VUV gate, exp-based gain, always-on quantisation noise).
        self.compat_mode: bool = bool(compat_mode)
        # Allow overriding compat_mode via environment for quick
        # experiments: FARGAN_COMPAT_MODE=1 → compat_mode=True
        try:
            if os.environ.get("FARGAN_COMPAT_MODE", "0") == "1":
                self.compat_mode = True
        except Exception:
            pass
        self.collect_internal_tracks: bool = False
        self.last_internal_tracks: Optional[Dict[str, torch.Tensor]] = None
        self.silence_energy_thr_db: Optional[float] = None
        self.silence_gate_width_db: Optional[float] = None

        self.cond_net = FARGANCond(feature_dim=feature_dim, cond_size=cond_size)
        self.sig_net = FARGANSub(
            subframe_size=subframe_size,
            nb_subframes=nb_subframes,
            cond_size=cond_size,
            compat_mode=self.compat_mode,
        )

    def forward(
        self,
        features: torch.Tensor,
        period: torch.Tensor,
        nb_frames: int,
        pre: Optional[torch.Tensor] = None,
        states: Optional[Tuple[torch.Tensor, ...]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Args:
            features: [B, T, feature_dim] 输入特征
            period: [B, T] 周期序列
            nb_frames: 要生成的帧数
            pre: [B, L] 可选的前置信号
            states: 可选的初始状态

        Returns:
            sig: [B, nb_frames*frame_size] 生成的音频
            states: 最终状态
        """

        device = features.device
        batch_size = features.size(0)

        # === 新增：以 cond_net 的权重作为全网运行dtype ===
        model_dtype = self.cond_net.fdense1.weight.dtype
        if features.dtype != model_dtype:
            features = features.to(model_dtype)
        if pre is not None and pre.dtype != model_dtype:
            pre = pre.to(model_dtype)

        # 初始化缓冲区（指定 dtype）
        prev   = torch.zeros(batch_size, 256, device=device, dtype=model_dtype)
        exc_mem= torch.zeros(batch_size, 256, device=device, dtype=model_dtype)
        # —— dtype 对齐、device、batch_size 保持你现有写法 ——
        # 初始化 RNN/FWConv 状态（当 states=None）
        if states is None:
            states = (
                torch.zeros(batch_size, 160, device=device, dtype=model_dtype),         # GRU1
                torch.zeros(batch_size, 128, device=device, dtype=model_dtype),         # GRU2
                torch.zeros(batch_size, 128, device=device, dtype=model_dtype),         # GRU3
                torch.zeros(batch_size, 2*self.subframe_size + 80 + 4,                  # FWConv state = in_size*(k-1)
                            device=device, dtype=model_dtype),                          # 2*40 + 80 + 4 = 164
            )

        # 预热帧数
        nb_pre_frames = pre.size(1) // self.frame_size if pre is not None else 0

        # （推荐）prime 激励缓冲为 pre 的“最后一帧”，更贴近自回归状态
        if pre is not None and pre.size(1) >= self.frame_size:
            exc_mem[:, -self.frame_size:] = pre[:, -self.frame_size:]

        # 生成条件（长度 = T_in - 4 = nb_frames）
        cond = self.cond_net(features, period)  # [B, nb_frames, 320]

        # —— 关键修复：循环严格以 cond 的时间维为准 ——
        # n ∈ [0, nb_frames-1]；这样 subframe_cond = cond[:, n, ...] 永不越界
        sig = torch.zeros((batch_size, 0), device=device, dtype=model_dtype)

        # 可选：健壮性断言（出问题就立刻早停，便于诊断）
        assert cond.size(1) >= nb_frames, f"cond_len={cond.size(1)} < nb_frames={nb_frames}"
        assert period.size(1) >= nb_frames + 3, f"period_len={period.size(1)} < nb_frames+3={nb_frames+3}"
        if not hasattr(self, "_checked_shapes"):
            print(f"[CoreCheck] cond_len={cond.size(1)}, nb_frames={nb_frames}, "
                f"period_len={period.size(1)}, feat_len={features.size(1)}, "
                f"state_shapes={[tuple(s.shape) for s in states]}")
            self._checked_shapes = True

        # Decide whether to use strict VUV gate mapping. When
        # compat_mode=True we always disable strict VUV (原始 FARGAN
        # 不使用显式 VUV 门控)。否则默认开启，并允许环境变量覆盖。
        use_compat = bool(getattr(self, "compat_mode", False))
        collect_internal = bool(getattr(self, "collect_internal_tracks", False))
        internal_tracks: Optional[Dict[str, list[torch.Tensor]]] = None
        if collect_internal:
            internal_tracks = {
                "pitch_gain_mean": [],
                "fwc0_rms": [],
                "skip_rms": [],
                "sig_core_rms": [],
                "sig_out_rms": [],
            }
        use_strict_vuv = False
        if not use_compat:
            use_strict_vuv = bool(getattr(self, "strict_vuv_gate", True))
            if not hasattr(self, "_strict_vuv_gate_set_from_cli"):
                try:
                    _strict_env = os.environ.get("FARGAN_STRICT_VUV_GATE", "")
                    if _strict_env != "":
                        use_strict_vuv = (_strict_env == "1")
                except Exception:
                    pass

        for n in range(0, nb_frames):
            for k in range(self.nb_subframes):
                pos = n * self.frame_size + k * self.subframe_size

                # —— 与conv中心对齐：使用 3 + n，并加上界保护（避免末尾极端边界时 +1 越界）
                per_idx  = min(3 + n, period.size(1)   - 1)
                feat_idx = min(3 + n, features.size(1) - 1)

                pitch = period[:, per_idx]
                gain = 0.03 * torch.pow(10.0, 0.5 * features[:, feat_idx, 0:1] / math.sqrt(18.0))
                gain = torch.nan_to_num(gain, nan=1.0, posinf=20.0, neginf=1e-3).clamp_(1e-3, 20.0)


                subframe_cond = cond[:, n, k * 80:(k + 1) * 80]
                # 可选：从第19维 frame_corr 推出 V/UV 概率并做轻量门控；
                # 在 compat_mode=True 时关闭该路径，保持与原始 FARGAN 一致。
                vuv = None
                silence_gate = None
                if (not use_compat) and features.size(-1) >= 20:
                    frame_corr = features[:, feat_idx, 19:20]
                    if use_strict_vuv:
                        # 严格映射：frame_corr=-0.8 → vuv=0, frame_corr=0.5 → vuv=1
                        vuv = torch.clamp((frame_corr + 0.8) / 1.3, 0.0, 1.0)
                    else:
                        # 兼容原始 FARGAN 行为的松弛映射
                        vuv = torch.clamp(0.5 * (frame_corr + 1.0), 0.0, 1.0)

                    # 调试：在 DBG_VUV_GATE=1 时打印当前子帧的 vuv gate 分布，
                    # 便于验证 strict_vuv_gate 映射是否将无声段有效压到接近 0。
                    if os.environ.get("DBG_VUV_GATE", "0") == "1":
                        try:
                            vmin = float(vuv.min().item())
                            vmax = float(vuv.max().item())
                            vmean = float(vuv.mean().item())
                            vstd = float(vuv.std().item())
                            print(
                                f"[DBG_VUV_GATE] strict={use_strict_vuv} "
                                f"min={vmin:+.4f} max={vmax:+.4f} "
                                f"mean={vmean:+.4f} std={vstd:+.4f}"
                            )
                        except Exception as _vdbg:
                            print(f"[DBG_VUV_GATE] debug print failed: {_vdbg}")

                    # 基于 c0 / sqrt(18) 近似的低频对数能量门控。该量与
                    # 主模型里用于 silence mask 的 mel mean(log10 power)
                    # 同量纲近似一致，用于压制静音段横纹。
                    try:
                        c0 = features[:, feat_idx, 0:1]
                        c0_log_energy = c0 / math.sqrt(18.0)
                        sil_thr_db_attr = getattr(self, "silence_energy_thr_db", None)
                        if sil_thr_db_attr is None:
                            sil_thr_db = float(os.environ.get("FARGAN_SILENCE_ENERGY_THR_DB", "-40.0"))
                        else:
                            sil_thr_db = float(sil_thr_db_attr)
                        sil_thr_log = sil_thr_db / 10.0
                        sil_width_db_attr = getattr(self, "silence_gate_width_db", None)
                        if sil_width_db_attr is None:
                            sil_width_db = float(os.environ.get("FARGAN_SILENCE_GATE_WIDTH_DB", "6.0"))
                        else:
                            sil_width_db = float(sil_width_db_attr)
                        sil_width_log = max(sil_width_db / 10.0, 1e-3)
                        silence_gate = torch.sigmoid((c0_log_energy - sil_thr_log) / sil_width_log)
                    except Exception:
                        silence_gate = None

                out, exc_mem, prev, states = self.sig_net(
                    subframe_cond,
                    prev,
                    exc_mem,
                    pitch,
                    states,
                    gain=gain,
                    vuv=vuv,
                    silence_gate=silence_gate,
                )
                if internal_tracks is not None:
                    dbg = getattr(self.sig_net, "last_debug", None)
                    if isinstance(dbg, dict):
                        for key in internal_tracks.keys():
                            val = dbg.get(key, None)
                            if isinstance(val, torch.Tensor):
                                internal_tracks[key].append(val.detach().unsqueeze(-1))

                if (n < nb_pre_frames) and (pre is not None):
                    # teacher-forcing: 用真实波形覆盖，并把 exc_mem 以真实输出推进
                    out = pre[:, pos:pos + self.subframe_size]
                    exc_mem[:, -self.subframe_size:] = out
                else:
                    # 自回归：累计输出
                    sig = torch.cat([sig, out], dim=1)

        # 分离状态梯度
        if internal_tracks is not None:
            try:
                tracks_out: Dict[str, torch.Tensor] = {}
                for key, seq in internal_tracks.items():
                    if len(seq) <= 0:
                        continue
                    cat = torch.cat(seq, dim=-1)  # [B, S]
                    S = cat.size(-1)
                    if self.nb_subframes > 1 and S >= self.nb_subframes:
                        usable = (S // self.nb_subframes) * self.nb_subframes
                        cat = cat[:, :usable].view(cat.size(0), -1, self.nb_subframes).mean(dim=-1)
                    tracks_out[key] = cat
                self.last_internal_tracks = tracks_out if len(tracks_out) > 0 else None
            except Exception:
                self.last_internal_tracks = None
        else:
            self.last_internal_tracks = None
        states = tuple(s.detach() for s in states)
        return sig, states



def test_fargan_components():
    """测试FARGAN组件"""
    print("🧪 测试FARGAN组件...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, T = 2, 10

    # 测试条件网络
    print("📡 测试FARGANCond...")
    cond_net = FARGANCond(feature_dim=20).to(device)
    features = torch.randn(B, T, 20, device=device)
    period = torch.randint(32, 256, (B, T), device=device)
    cond = cond_net(features, period)
    print(f"条件网络: 输入{features.shape} -> 输出{cond.shape}")

    # 测试子帧网络
    print("🎵 测试FARGANSub...")
    sub_net = FARGANSub().to(device)
    states = (
        torch.zeros(B, 160, device=device),
        torch.zeros(B, 128, device=device),
        torch.zeros(B, 128, device=device),
        torch.zeros(B, 124, device=device)
    )
    exc_mem = torch.randn(B, 256, device=device)
    prev_pred = torch.randn(B, 256, device=device)
    cond_sub = torch.randn(B, 80, device=device)
    period_sub = torch.randint(32, 256, (B,), device=device)

    sig_out, exc_mem_new, prev_pred_new, states_new = sub_net(
        cond_sub, prev_pred, exc_mem, period_sub, states
    )
    print(f"子帧网络: 输出{sig_out.shape}")

    # 测试核心模块
    print("🚀 测试FARGANCore...")
    core = FARGANCore(feature_dim=20).to(device)
    nb_frames = 5
    pre = torch.randn(B, 160, device=device)  # 1帧前置
    sig, final_states = core(features, period, nb_frames, pre=pre)
    print(f"核心模块: 输入{features.shape} -> 输出{sig.shape}")

    print("✅ FARGAN组件测试通过")


if __name__ == "__main__":
    pass
