#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量 2D VMamba 风格的 JSCC 编解码器（支持 mamba-ssm CUDA selective_scan，自动降级）。

设计目标：
- 保留 MambaJSCC 的二维扫描和多层块结构，用 cross-selective-scan 拼接行/列正反方向。
- 采用多 stage 下采样/上采样结构，编解码对称，适配 mel 频谱图像输入。
- CSI/SNR 通过每层的线性映射做加性调制。
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import math
import os
import sys
import importlib.util

import torch
import torch.nn as nn
import torch.nn.functional as F

# 优先强制使用本仓库自带的 mamba_ssm 实现，
# 即使环境中已经通过 pip 安装了其它版本。
_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_local_mamba_pkg_dir = os.path.join(_root_dir, "mamba_ssm")

def _ensure_local_mamba_loaded() -> bool:
    """Ensure that sys.modules['mamba_ssm'] 指向仓库内的 mamba_ssm 包。

    若环境中已经 import 了 site-packages 里的 mamba_ssm，这里会用本地版本覆盖。
    返回 True 表示加载成功且本地包已就绪。
    """
    try:
        if not os.path.isdir(_local_mamba_pkg_dir):
            return False

        init_py = os.path.join(_local_mamba_pkg_dir, "__init__.py")
        if not os.path.isfile(init_py):
            return False

        # 若当前已加载的 mamba_ssm 不是本地路径，则覆盖之
        loaded = sys.modules.get("mamba_ssm", None)
        if loaded is not None:
            loaded_path = getattr(loaded, "__file__", "") or ""
            if _local_mamba_pkg_dir in loaded_path:
                return True

        spec = importlib.util.spec_from_file_location("mamba_ssm", init_py)
        if spec is None or spec.loader is None:
            return False
        module = importlib.util.module_from_spec(spec)
        sys.modules["mamba_ssm"] = module
        spec.loader.exec_module(module)
        return True
    except Exception:
        return False

if _ensure_local_mamba_loaded():
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as selective_scan_cuda
        HAS_MAMBA_SSM = True
        print("[VMambaJSCC2D] mamba_ssm selective_scan_fn loaded from local package: using CUDA selective_scan fast path", flush=True)
    except Exception as _e:
        selective_scan_cuda = None
        HAS_MAMBA_SSM = False
        print(f"[VMambaJSCC2D] local mamba_ssm present but selective_scan_fn import failed, fallback to Python scan. Error: {_e}", flush=True)
else:
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as selective_scan_cuda
        HAS_MAMBA_SSM = True
        print("[VMambaJSCC2D] mamba_ssm selective_scan_fn loaded from environment: using CUDA selective_scan fast path", flush=True)
    except Exception as _e:
        selective_scan_cuda = None
        HAS_MAMBA_SSM = False
        print(f"[VMambaJSCC2D] mamba_ssm selective_scan_fn NOT available, fallback to Python scan. Error: {_e}", flush=True)

# 尝试导入 adaptive_selective_scan CUDA 内核，仅在 SSM 自适应模式下使用。
try:  # pragma: no cover - 环境依赖 CUDA
    import adaptive_selective_scan_cuda_core
    HAS_ADAPTIVE_SSCAN = True
    print("[VMambaJSCC2D] adaptive_selective_scan_cuda_core loaded: SSM-adaptive path enabled", flush=True)
except Exception as _e:  # pragma: no cover - 防御性分支
    adaptive_selective_scan_cuda_core = None  # type: ignore[assignment]
    HAS_ADAPTIVE_SSCAN = False
    print(f"[VMambaJSCC2D] adaptive_selective_scan_cuda_core NOT available, will not use adaptive SSM scan. Error: {_e}", flush=True)


class Swish(nn.Module):
    """Swish激活函数"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class SNR_embedding(nn.Module):
    """SNR嵌入模块，将SNR值转换为嵌入向量"""
    def __init__(self, T: int, d_model: int, dim: int):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.SNRembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, SNR):
        emb1 = self.SNRembedding(SNR)
        return emb1


class AdaptiveModulator(nn.Module):
    """自适应调制器，基于SNR调整特征"""
    def __init__(self, M: int):
        super(AdaptiveModulator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.Sigmoid()
        )

    def forward(self, snr):
        return self.fc(snr)


class LightweightCSIGate(nn.Module):
    """轻量级CSI门控：严格按照LW-JSCC论文实现的信道编码器C和解码器C^{-1}"""
    def __init__(self, channels: int, csi_dim: int = 4, hidden_dim: int = 32):
        super().__init__()
        self.channels = channels

        # 编码端C：C_1 (ReLU) + C_2 (Sigmoid) 计算缩放因子σ_C
        self.encoder_scaling_net = nn.Sequential(
            nn.Linear(channels + 1, hidden_dim),  # P_C = Concat(P_w_s, μ)
            nn.ReLU(),                            # C_1 with ReLU
            nn.Linear(hidden_dim, channels),      # C_2
            nn.Sigmoid()                          # σ_C ∈ [0,1]
        )

        # 编码端C：C_3 + C_4 处理缩放后的特征
        self.encoder_transform_net = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),                           # C_3 with ReLU
            nn.Linear(channels, channels),
            nn.ReLU()                            # C_4 with ReLU
        )

        # 解码端C^{-1}：C_4^{-1} + C_3^{-1} 重建特征 (论文描述：先重建再计算缩放)
        self.decoder_reconstruct_net = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),                           # C_4^{-1} with ReLU
            nn.Linear(channels, channels),
            nn.ReLU()                            # C_3^{-1} with ReLU
        )

        # 解码端C^{-1}：C_2^{-1} + C_1^{-1} 计算缩放因子σ_{C^{-1}}
        self.decoder_scaling_net = nn.Sequential(
            nn.Linear(channels + 1, hidden_dim),  # P_{C^{-1}} = Concat(P_reconstructed, μ)
            nn.ReLU(),                            # C_2^{-1} with ReLU
            nn.Linear(hidden_dim, channels),      # C_1^{-1}
            nn.Sigmoid()                          # σ_{C^{-1}} ∈ [0,1]
        )

    def forward(self, x: torch.Tensor, csi: torch.Tensor, is_encoder: bool = True) -> torch.Tensor:
        """
        严格按照LW-JSCC公式(4)(5)实现
        x: [B, C, H, W] 特征
        csi: [B, csi_dim] CSI信息，取第一个维度作为SNR μ
        is_encoder: True为编码器C，False为解码器C^{-1}
        """
        B, C, H, W = x.shape
        mu = csi[:, 0:1]  # [B, 1] SNR值

        if is_encoder:
            # 编码端C的实现 (论文公式4)
            # Step 1: 计算全局统计特征 P_{w_s}
            P_w_s = x.mean(dim=[2, 3])  # [B, C] 全局特征

            # Step 2: P_C = Concat(P_{w_s}, μ)
            P_C = torch.cat([P_w_s, mu], dim=1)  # [B, C+1]

            # Step 3: 计算缩放因子 σ_C = Sigmoid(C_2(ReLU(C_1(P_C))))
            sigma_C = self.encoder_scaling_net(P_C)  # [B, C] σ_C ∈ [0,1]

            # Step 4: 特征缩放 w_s * σ_C
            sigma_C_expanded = sigma_C.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
            x_scaled = x * sigma_C_expanded

            # Step 5: 通过C_3, C_4处理
            x_flat = x_scaled.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
            x_out_flat = self.encoder_transform_net(x_flat)     # [B, H*W, C]
            x_out = x_out_flat.permute(0, 2, 1).view(B, C, H, W)  # [B, C, H, W]

        else:
            # 解码端C^{-1}的实现 (论文公式5)
            # Step 1: 先通过C_4^{-1}, C_3^{-1}重建
            x_flat = x.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
            x_reconstructed_flat = self.decoder_reconstruct_net(x_flat)  # [B, H*W, C]
            x_reconstructed = x_reconstructed_flat.permute(0, 2, 1).view(B, C, H, W)

            # Step 2: 计算重建后的全局特征 P_{C^{-1}}
            P_reconstructed = x_reconstructed.mean(dim=[2, 3])  # [B, C]
            P_C_inv = torch.cat([P_reconstructed, mu], dim=1)   # [B, C+1]

            # Step 3: 计算解码端缩放因子 σ_{C^{-1}} (论文公式5)
            sigma_C_inv = self.decoder_scaling_net(P_C_inv)  # [B, C] σ_{C^{-1}} ∈ [0,1]

            # Step 4: 应用解码端缩放
            sigma_C_inv_expanded = sigma_C_inv.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
            x_out = x_reconstructed * sigma_C_inv_expanded

        return x_out


def _layer_norm_channel_last(x: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
    return norm(x)


class SelectiveScan2D(nn.Module):
    """二维 cross-selective-scan，带可选的 SNR 级别 SSM 调制。

    - 若 mamba-ssm 可用：拼接行/列正反四个方向，一次调用 CUDA selective_scan，再方向合并；
      可通过 snr_scale 对 A 进行简单缩放，实现 SSM 内核级别的 SNR 调制（轻量版）。
    - 否则：退回 Python 递推（行/列正反），同样使用 snr_scale 对 A 做缩放。
    """

    def __init__(self, channels: int, d_state: int = None) -> None:
        super().__init__()
        self.channels = channels
        # DeLight风格：SSM状态维度变窄，体现"深而窄"的"窄"
        self.d_state = d_state if d_state is not None else max(16, channels // 4)

        # SSM参数：状态维度变窄减少参数
        self.A = nn.Parameter(-0.5 * torch.ones(self.d_state))  # [d_state]而非[channels]
        self.B = nn.Parameter(0.5 * torch.ones(self.d_state))
        self.C = nn.Parameter(0.5 * torch.ones(self.d_state))
        self.D = nn.Parameter(torch.zeros(channels))  # D保持[channels]

        # 状态投影层：channels ↔ d_state (仅在维度不同时添加)
        if self.d_state != channels:
            self.input_proj = nn.Linear(channels, self.d_state, bias=False)
            self.output_proj = nn.Linear(self.d_state, channels, bias=False)
        else:
            self.input_proj = self.output_proj = nn.Identity()

    def _scan_dir_python(self, seq: torch.Tensor, snr_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, C = seq.shape
        # 对输入序列进行归一化，防止长序列累积误差
        seq = F.layer_norm(seq, [C])

        # 投影到状态空间
        u_proj = self.input_proj(seq)  # [B, L, d_state]

        h = u_proj.new_zeros(B, self.d_state)
        A_base = torch.tanh(self.A).view(1, self.d_state)  # [1,D]
        # 简单的 SNR 级别调制：snr_scale∈[0,1] 时，低 SNR → A 较小（更强遗忘），
        # 高 SNR → A 接近原值。若 snr_scale 为 None，则不做调制。
        if snr_scale is not None:
            if isinstance(snr_scale, torch.Tensor):
                s = snr_scale.view(B, 1).clamp(0.0, 1.0)  # [B,1]
            else:
                s_val = max(0.0, min(1.0, float(snr_scale)))
                s = seq.new_full((B, 1), s_val)
            scale_min, scale_max = 0.6, 1.0
            scale = scale_min + (scale_max - scale_min) * s  # [B,1]
            A = A_base * scale                               # [B,D]
        else:
            A = A_base.expand(B, self.d_state)               # [B,D]
        # 数值稳定性约束，防止极值
        A = torch.clamp(A, min=-0.99, max=0.99)
        Bp = self.B.view(1, self.d_state)
        Cp = self.C.view(1, self.d_state)

        outs = []
        for t in range(L):
            u = u_proj[:, t, :]  # [B, d_state]
            h = A * h + Bp * u
            y_state = Cp * h      # [B, d_state]
            # 投影回输出空间
            y_out = self.output_proj(y_state)  # [B, C]
            # 添加跳跃连接
            y = y_out + self.D.view(1, C) * seq[:, t, :]
            outs.append(y)
        return torch.stack(outs, dim=1)

    def _cross_scan_cuda(self, x: torch.Tensor, snr_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, H, W, C = x.shape
        L = H * W
        xf = x.reshape(B, L, C).transpose(1, 2).contiguous()
        xb = torch.flip(xf, dims=[2])
        xc = x.permute(0, 2, 1, 3).reshape(B, L, C).transpose(1, 2).contiguous()
        xcb = torch.flip(xc, dims=[2])

        u = torch.cat([xf, xb, xc, xcb], dim=0)  # [4B,C,L]
        delta = torch.ones(u.shape[0], C, L, device=u.device, dtype=u.dtype)
        A_base = torch.tanh(self.A).view(C, 1)

        # 这里由于 selective_scan_cuda 的接口限制，只能使用 batch 平均 snr_scale
        # 做全局缩放；若传入的是逐样本 [B] 向量，则取其均值。
        if snr_scale is not None:
            if isinstance(snr_scale, torch.Tensor):
                s_val = float(snr_scale.mean().item())
            else:
                s_val = float(snr_scale)
            s_val = max(0.0, min(1.0, s_val))
            scale_min, scale_max = 0.6, 1.0
            scale = scale_min + (scale_max - scale_min) * s_val
            A = A_base * scale
        else:
            A = A_base
        Bp = self.B.view(C, 1)
        Cp = self.C.view(C, 1)
        Dp = self.D.view(C)

        # 目前在 Aether-lite 中，adaptive_selective_scan 在此结构下会触发底层 CUDA 浮点异常，
        # 因此这里保守起见始终使用 mamba_ssm 的 selective_scan_cuda fast path。
        # 若后续在与 MambaJSCC 完全一致的 SSM 结构下使用 adaptive_selective_scan，可在此处切换实现。
        y = selective_scan_cuda(u, delta, A, Bp, Cp, Dp, delta_bias=None, delta_softplus=False)

        yf, yb, yc, ycb = torch.chunk(y, 4, dim=0)
        yf = yf.transpose(1, 2).reshape(B, H, W, C)
        yb = torch.flip(yb, dims=[2]).transpose(1, 2).reshape(B, H, W, C)
        yc = yc.transpose(1, 2).reshape(B, W, H, C).permute(0, 2, 1, 3).contiguous()
        ycb = torch.flip(ycb, dims=[2]).transpose(1, 2).reshape(B, W, H, C).permute(0, 2, 1, 3).contiguous()
        return 0.25 * (yf + yb + yc + ycb)

    def forward(self, x: torch.Tensor, snr_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, H, W, C]  (channel-last)
        返回: [B, H, W, C]
        """
        B, H, W, C = x.shape
        L = H * W

        # 对特别长的序列做一次全局 LN，减少累积误差
        if L > 4000:
            x = F.layer_norm(x, [C])

        # fast path：只有在 d_state == channels 时，才安全走 CUDA selective_scan
        if HAS_MAMBA_SSM and self.d_state == self.channels:
            # 仅在第一次调用时打印一次实际采用的路径，避免刷屏
            if not hasattr(self, "_printed_path"):
                setattr(self, "_printed_path", True)
                print("[VMambaJSCC2D] Using CUDA selective_scan fast path in SelectiveScan2D", flush=True)
            return self._cross_scan_cuda(x, snr_scale=snr_scale)

        # -------- Python fallback: 行/列 + 正反双向 扫描 --------
        # 1) 按行优先 (row-major)：展平成长度 L 的序列
        seq_row = x.reshape(B, L, C)                      # [B, L, C]
        y_row_f = self._scan_dir_python(seq_row, snr_scale=snr_scale)         # 正向
        y_row_b = self._scan_dir_python(
            torch.flip(seq_row, dims=[1]), snr_scale=snr_scale
        )                                                # 反向
        y_row = 0.5 * (y_row_f + torch.flip(y_row_b, dims=[1]))
        y_row = y_row.view(B, H, W, C)                   # 还原 [B,H,W,C]

        # 2) 按列优先 (col-major)：先交换 H/W 再展平
        x_col = x.permute(0, 2, 1, 3).contiguous()       # [B, W, H, C]
        seq_col = x_col.reshape(B, L, C)                 # [B, L, C]
        y_col_f = self._scan_dir_python(seq_col, snr_scale=snr_scale)
        y_col_b = self._scan_dir_python(
            torch.flip(seq_col, dims=[1]), snr_scale=snr_scale
        )
        y_col = 0.5 * (y_col_f + torch.flip(y_col_b, dims=[1]))
        y_col = y_col.view(B, W, H, C).permute(0, 2, 1, 3).contiguous()

        # 3) 行/列两种方向做平均
        return 0.5 * (y_row + y_col)



class VMambaBlock2D(nn.Module):
    def __init__(self, channels: int, csi_dim: int, mlp_ratio: float = 2.0, dropout: float = 0.0,
                 csi_mode: str = "native", is_encoder: bool = True) -> None:
        """
        VMamba Block支持不同的CSI处理模式
        csi_mode: "native" (Mamba内生), "lightweight" (LW-JSCC门控), "hybrid" (混合)
        """
        super().__init__()
        self.channels = channels
        self.csi_mode = csi_mode
        self.is_encoder = is_encoder

        # 核心组件
        self.norm1 = nn.LayerNorm(channels)
        # DeLight风格：变窄的SSM状态空间
        if csi_mode == "native":
            # 完整 Mamba 状态维；支持 CUDA selective_scan，算力充裕时用
            d_state = channels
        elif csi_mode == "lightweight":
            # 轻量状态维：显著小于 C，完全走 Python 版 scan
            d_state = max(16, channels // 4)
        else:  # "hybrid"
            # 折中：状态维适中，仍然只走 Python 版 scan
            d_state = max(16, channels // 2)

        self.scan = SelectiveScan2D(channels, d_state=d_state)


        # CSI处理：根据模式选择不同策略
        if csi_mode == "native":
            # 原始Mamba内生CSI处理
            self.csi_proj = nn.Linear(csi_dim, channels)
        elif csi_mode == "lightweight":
            # LW-JSCC轻量门控
            self.csi_gate = LightweightCSIGate(channels, csi_dim, hidden_dim=max(16, channels//4))
        else:  # hybrid
            # 混合模式：内生 + 轻量门控
            self.csi_proj = nn.Linear(csi_dim, channels)
            self.csi_gate = LightweightCSIGate(channels, csi_dim, hidden_dim=max(16, channels//4))

        # FFN - 按照DeLight思想：深而窄
        if csi_mode == "lightweight":
            # 轻量模式：减小FFN宽度
            hidden = max(channels, int(channels * mlp_ratio * 0.5))  # 减半FFN宽度
        else:
            hidden = int(channels * mlp_ratio)

        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, csi: torch.Tensor, snr_scale: Optional[float] = None) -> torch.Tensor:
        x_cl = x.permute(0, 2, 3, 1).contiguous()

        # CSI处理：根据模式分支
        if self.csi_mode == "native":
            # 原始Mamba内生方式
            csi_bias = self.csi_proj(csi).view(csi.size(0), 1, 1, -1)
            y = _layer_norm_channel_last(x_cl + csi_bias, self.norm1)

        elif self.csi_mode == "lightweight":
            # 纯轻量门控方式
            y = _layer_norm_channel_last(x_cl, self.norm1)

        else:  # hybrid
            # 混合方式：内生bias + 轻量门控
            csi_bias = self.csi_proj(csi).view(csi.size(0), 1, 1, -1)
            y = _layer_norm_channel_last(x_cl + csi_bias, self.norm1)

        # Mamba扫描（可选 SNR 级别调制）
        y = self.scan(y, snr_scale=snr_scale)
        x_cl = x_cl + y

        # 后处理：轻量门控应用
        if self.csi_mode in ["lightweight", "hybrid"]:
            # 将channel-last转为channel-first做门控，再转回来
            x_gated = self.csi_gate(x_cl.permute(0, 3, 1, 2), csi, is_encoder=self.is_encoder)
            x_cl = x_gated.permute(0, 2, 3, 1)

        # FFN分支
        y2 = _layer_norm_channel_last(x_cl, self.norm2)
        y2 = self.ffn(y2)
        x_cl = x_cl + y2

        return x_cl.permute(0, 3, 1, 2).contiguous()


class PatchEmbed2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class Downsample2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride_hw: Tuple[int, int] = (2, 2)) -> None:
        super().__init__()
        self.stride_hw = stride_hw
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride_hw, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class PatchMerging2D(nn.Module):
    """2D Patch Merging层，对应原始MambaJSCC的下采样策略。

    将2x2 patch合并为单个patch，通道数变为4倍，然后通过线性层调整。
    """
    def __init__(self, in_ch: int, out_ch: int, stride_hw: Tuple[int, int] = (2, 2)) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride_hw = stride_hw

        if stride_hw == (2, 2):
            # 标准2x2 patch merging
            self.reduction = nn.Linear(4 * in_ch, out_ch, bias=False)
            self.norm = nn.LayerNorm(4 * in_ch)
        else:
            # 非标准stride，回退到卷积
            kernel_size = max(stride_hw[0], stride_hw[1]) + 1
            self.conv_fallback = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                                         stride=stride_hw, padding=kernel_size//2)

    def _patch_merging_pad(self, x: torch.Tensor) -> torch.Tensor:
        """将[B,C,H,W]转换为patch merging格式"""
        B, C, H, W = x.shape

        # 确保H,W都是偶数
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, W % 2, 0, H % 2))
            _, _, H, W = x.shape

        # 转换为[B,H,W,C]格式
        x = x.permute(0, 2, 3, 1)  # [B,C,H,W] -> [B,H,W,C]

        # 提取2x2 patch
        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]

        # 拼接为[B, H/2, W/2, 4*C]
        x = torch.cat([x0, x1, x2, x3], -1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: [B, C, H, W]
        Output: [B, out_ch, H/stride_h, W/stride_w]
        """
        if self.stride_hw == (2, 2):
            # 使用patch merging
            x = self._patch_merging_pad(x)  # [B, H/2, W/2, 4*C]
            x = self.norm(x)
            x = self.reduction(x)           # [B, H/2, W/2, out_ch]
            x = x.permute(0, 3, 1, 2)       # [B, out_ch, H/2, W/2]
            return x
        else:
            # 非标准stride，使用卷积
            return self.conv_fallback(x)


class Upsample2D(nn.Module):
    """时间维双线性上采样 + 卷积细化。

    约定：仅在时间维做 2 倍上采样（stride_hw=(2,1)），
    这样可以避免 2x2 PixelShuffle 带来的强块效应。
    """

    def __init__(self, in_ch: int, out_ch: int, stride_hw: Tuple[int, int] = (2, 1)) -> None:
        super().__init__()
        self.stride_hw = stride_hw
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W]，其中 H 对应时间维，W 对应频率维
        h_scale, w_scale = self.stride_hw
        if h_scale != 1 or w_scale != 1:
            new_h = x.size(2) * h_scale
            new_w = x.size(3) * w_scale
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        return self.proj(x)


class PatchReverseMerging2D(nn.Module):
    """2D Patch Division层，类似原始MambaJSCC的上采样策略。

    使用LayerNorm + Linear + PixelShuffle实现空间上采样，
    比简单的ConvTranspose2d更稳定。
    """
    def __init__(self, in_ch: int, out_ch: int, stride_hw: Tuple[int, int] = (2, 2)) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride_hw = stride_hw

        # 计算上采样因子
        h_factor, w_factor = stride_hw
        total_factor = h_factor * w_factor

        self.norm = nn.LayerNorm(in_ch)
        # 线性变换：输入 → 输出通道数 * 上采样因子
        self.increment = nn.Linear(in_ch, out_ch * total_factor, bias=False)

        # 用于PixelShuffle重组
        self.h_factor = h_factor
        self.w_factor = w_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: [B, C, H, W]
        Output: [B, out_ch, H*h_factor, W*w_factor]
        """
        B, C, H, W = x.shape

        # 转换为 [B, H, W, C] 格式进行归一化和线性变换
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x = self.norm(x)           # LayerNorm
        x = self.increment(x)      # [B, H, W, C] -> [B, H, W, out_ch * factor]

        # 重新排列为 [B, C, H, W] 并按整数因子进行像素重排
        x = x.permute(0, 3, 1, 2)  # [B, H, W, out_ch * factor] -> [B, out_ch * factor, H, W]

        if self.stride_hw == (2, 2):
            # 标准2x2上采样
            x = F.pixel_shuffle(x, 2)
        else:
            # 通用上采样：将通道维拆分为 [out_ch, h_factor, w_factor]，再重排到空间
            factor = self.h_factor * self.w_factor
            assert x.size(1) == self.out_ch * factor, \
                f"Channel mismatch: got {x.size(1)}, expect {self.out_ch * factor}"
            x = x.view(B, self.out_ch, self.h_factor, self.w_factor, H, W)
            # [B, out_ch, h, w, H, W] -> [B, out_ch, H*h, W*w]
            x = x.permute(0, 1, 4, 2, 5, 3).contiguous().view(B, self.out_ch, H * self.h_factor, W * self.w_factor)

        return x


class VMambaEncoder2D(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,  # 新增：输出通道数参数
        channels: List[int],
        depths: List[int],
        csi_dim: int,
        freq_downsample_stages: int = 2,
        lightweight_config: str = "progressive"  # "progressive", "all_lightweight", "all_native"
    ) -> None:
        super().__init__()
        assert len(channels) == len(depths)
        self.stages = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.patch_embed = PatchEmbed2D(in_ch, channels[0])
        self.lightweight_config = lightweight_config
        self.freq_downsample_stages = max(0, int(freq_downsample_stages))

        # 渐进式CSI处理策略：前几层native，中间层hybrid，后几层lightweight
        def get_csi_mode(stage_idx: int, total_stages: int) -> str:
            if lightweight_config == "all_native":
                return "native"
            elif lightweight_config == "all_lightweight":
                return "lightweight"
            else:  # "progressive"
                if stage_idx == 0:
                    return "native"      # 第一层：保持Mamba内生智能
                elif stage_idx < total_stages - 1:
                    return "hybrid"      # 中间层：混合模式
                else:
                    return "lightweight" # 最后层：纯轻量门控

        num_stages = len(channels)
        for i, (ch, depth) in enumerate(zip(channels, depths)):
            csi_mode = get_csi_mode(i, num_stages)
            blocks = nn.ModuleList([
                VMambaBlock2D(ch, csi_dim, csi_mode=csi_mode, is_encoder=True)
                for _ in range(depth)
            ])
            self.stages.append(blocks)
            if i < num_stages - 1:
                # 仅在时间维做 2 倍下采样，频率维保持不变，使用卷积步幅实现。
                stride_hw = (2, 1)
                self.downs.append(Downsample2D(ch, channels[i + 1], stride_hw=stride_hw))

        # 从 (H×W×C) 大幅压缩到 (C_compressed×H×W)，类似从320→32的10倍压缩
        final_ch = channels[-1]  # 如 48
        mid_ch1  = max(out_ch * 4, final_ch // 2)    # 比 out_ch 大一些
        mid_ch2  = max(out_ch * 2, mid_ch1 // 2)
        mid_ch3  = max(out_ch,     mid_ch2 // 2)

        self.conv_compression = nn.Sequential(
            nn.Conv2d(final_ch, mid_ch1, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch1, mid_ch2, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch2, mid_ch3, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch3, out_ch, kernel_size=1, padding=0, stride=1),
        )


    def forward(
        self,
        x: torch.Tensor,
        csi: torch.Tensor,
        snr_embedding: Optional[torch.Tensor] = None,
        proj_list: Optional[nn.ModuleList] = None,
        snr_scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        import os
        debug = bool(int(os.environ.get("DBG_STAGE25", "0")))
        debug_ca = bool(int(os.environ.get("DBG_CA", "0")))

        x = self.patch_embed(x)
        if debug:
            xmin, xmax = float(x.min().item()), float(x.max().item())
            xmean, xstd = float(x.mean().item()), float(x.std().item())
            print(f"[VMAMBA_ENC] after patch_embed: shape={x.shape}, min={xmin:.4f}, max={xmax:.4f}, mean={xmean:.4f}, std={xstd:.4f}")

        B = x.shape[0]

        # 移除stage级残差，采用简洁的块堆叠设计；在每个 stage 入口按照
        # proj_list[i](snr_embedding) 施加一次通道级 SNR 调制。
        for i, blocks in enumerate(self.stages):
            if snr_embedding is not None and proj_list is not None and i < len(proj_list):
                ch = x.shape[1]
                emb = proj_list[i](snr_embedding)  # [B, ch]
                emb = emb.view(B, ch, 1, 1)
                x = x + emb
                if debug_ca:
                    with torch.no_grad():
                        emb_norm = emb.view(B, ch).norm(dim=1).mean().item()
                        print(f"[CA] enc stage {i}, ch={ch}, emb_norm={emb_norm:.3f}")
            for blk in blocks:
                x = blk(x, csi, snr_scale=snr_scale)  # 只依赖VMambaBlock内部的双残差

            if debug:
                xmin, xmax = float(x.min().item()), float(x.max().item())
                xmean, xstd = float(x.mean().item()), float(x.std().item())
                print(f"[VMAMBA_ENC] after stage {i}: shape={x.shape}, min={xmin:.4f}, max={xmax:.4f}, mean={xmean:.4f}, std={xstd:.4f}")

            if i < len(self.downs):
                x = self.downs[i](x)
                if debug:
                    xmin, xmax = float(x.min().item()), float(x.max().item())
                    xmean, xstd = float(x.mean().item()), float(x.std().item())
                    print(f"[VMAMBA_ENC] after downsample {i}: shape={x.shape}, min={xmin:.4f}, max={xmax:.4f}, mean={xmean:.4f}, std={xstd:.4f}")

        # 参照MambaJSCC原版：conv compression处理
        x = self.conv_compression(x)
        if debug:
            xmin, xmax = float(x.min().item()), float(x.max().item())
            xmean, xstd = float(x.mean().item()), float(x.std().item())
            print(f"[VMAMBA_ENC] final conv_compression to output: shape={x.shape}, min={xmin:.4f}, max={xmax:.4f}, mean={xmean:.4f}, std={xstd:.4f}")

        return x, (x.shape[2], x.shape[3])


class VMambaDecoder2D(nn.Module):
    def __init__(
        self,
        in_ch: int,  # 新增：输入通道数参数
        channels: List[int],
        depths: List[int],
        csi_dim: int,
        out_ch: int,
        freq_downsample_stages: int = 2,
        lightweight_config: str = "progressive"  # 新增：轻量化配置
    ) -> None:
        super().__init__()
        assert len(channels) == len(depths)

        # 保留目标输出通道数（例如 1），避免在下方循环中被局部重用覆盖。
        self.out_ch = out_ch

        # 与编码器对称的下采样/上采样步幅策略
        self.freq_downsample_stages = max(0, int(freq_downsample_stages))

        # 参照MambaJSCC原版：conv expansion层，逆转compression操作
        # 从压缩的符号维度(例如24)扩展回解码器第一层通道数(例如48)
        target_ch = channels[0]   # 例如 48

        mid_ch1 = max(in_ch * 4, target_ch // 4)
        mid_ch2 = max(mid_ch1 * 2, target_ch // 2)
        mid_ch3 = max(mid_ch2 * 2, target_ch)

        # 主分支：从符号维度映射到解码器第一层通道数
        self.conv_expansion = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch1, kernel_size=1, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch1, mid_ch2, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch2, mid_ch3, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch3, target_ch, kernel_size=3, padding=1, stride=1),
        )

        # 残差支路：直接将符号通过 1x1 Conv 投影到 target_ch 通道，
        # 与 conv_expansion 输出同形状，可显式保留符号幅度信息。
        self.residual_proj = nn.Conv2d(in_ch, target_ch, kernel_size=1, padding=0, stride=1)

        # 适度放大 conv_expansion 的输出幅度，缓解初始阶段 std 过小的问题；
        # 该系数简单设为常数，后续由训练自行微调。
        self.expansion_gain: float = 2.0


        self.stages = nn.ModuleList()
        self.ups = nn.ModuleList()
        # 轻量去块卷积：与每个 Upsample2D 配对，缓解上采样后的块状伪影
        self.deblocks = nn.ModuleList()

        # 解码器的渐进式CSI处理策略
        def get_csi_mode_decoder(stage_idx: int, total_stages: int) -> str:
            if lightweight_config == "all_native":
                return "native"
            elif lightweight_config == "all_lightweight":
                return "lightweight"
            else:  # "progressive"
                # 解码器反向渐进：前面lightweight，后面native
                if stage_idx < total_stages // 2:
                    return "lightweight" # 前面层：纯轻量门控
                elif stage_idx < total_stages - 1:
                    return "hybrid"      # 中间层：混合模式
                else:
                    return "native"      # 最后层：保持Mamba内生智能

        num_stages = len(channels)
        num_up = max(0, num_stages - 1)

        # 仅在时间维做 2 倍上采样，与编码器的 Conv(stride=(2,1)) 对称。
        up_strides = [(2, 1) for _ in range(num_stages - 1)]

        for i in range(num_stages):
            ch = channels[i]
            depth = depths[i]
            csi_mode = get_csi_mode_decoder(i, num_stages)
            blocks = nn.ModuleList([
                VMambaBlock2D(ch, csi_dim, csi_mode=csi_mode, is_encoder=False)
                for _ in range(depth)
            ])
            self.stages.append(blocks)
            if i < num_up:
                stride_hw = up_strides[i]
                up_out_ch = channels[i + 1]
                # 上采样层的输出通道仅用于中间特征，不应覆盖最终的 out_ch。
                self.ups.append(Upsample2D(ch, up_out_ch, stride_hw=stride_hw))
                # Deblock: 3x3 Conv + GELU 残差细化，弱化像素级块边界
                self.deblocks.append(
                    nn.Sequential(
                        nn.Conv2d(up_out_ch, up_out_ch, kernel_size=3, stride=1, padding=1),
                        nn.GELU(),
                    )
                )

        # 参照MambaJSCC原版：最后一层 head，映射到期望的输出通道数
        final_ch = channels[-1]
        self.head = nn.Conv2d(final_ch, self.out_ch, kernel_size=1, padding=0, stride=1)

    def forward(
        self,
        x: torch.Tensor,
        csi: torch.Tensor,
        snr_embedding: Optional[torch.Tensor] = None,
        proj_list: Optional[nn.ModuleList] = None,
        snr_scale: Optional[float] = None,
    ) -> torch.Tensor:
        import os
        debug = bool(int(os.environ.get("DBG_STAGE25", "0")))
        debug_ca = bool(int(os.environ.get("DBG_CA", "0")))
        if debug:
            xmin, xmax = float(x.min().item()), float(x.max().item())
            xmean, xstd = float(x.mean().item()), float(x.std().item())
            print(f"[VMAMBA_DEC] input: shape={x.shape}, min={xmin:.4f}, max={xmax:.4f}, mean={xmean:.4f}, std={xstd:.4f}")

        # 参照MambaJSCC原版：conv expansion 处理，并加入符号残差通路
        # 主分支：非线性映射
        main = self.conv_expansion(x)
        if self.expansion_gain != 1.0:
            main = main * float(self.expansion_gain)

        # 残差分支：线性投影符号到 target_ch 通道
        res = self.residual_proj(x)

        x = main + res
        if debug:
            xmin, xmax = float(x.min().item()), float(x.max().item())
            xmean, xstd = float(x.mean().item()), float(x.std().item())
            print(f"[VMAMBA_DEC] after conv_expansion: shape={x.shape}, min={xmin:.4f}, max={xmax:.4f}, mean={xmean:.4f}, std={xstd:.4f}")

        B = x.shape[0]

        # 移除stage级残差，采用简洁的块堆叠设计；在每个 stage 入口施加
        # proj_list[i](snr_embedding) 形式的通道级 SNR 调制。
        for i, blocks in enumerate(self.stages):
            if snr_embedding is not None and proj_list is not None and i < len(proj_list):
                ch = x.shape[1]
                emb = proj_list[i](snr_embedding)  # [B, ch]
                emb = emb.view(B, ch, 1, 1)
                x = x + emb
                if debug_ca:
                    with torch.no_grad():
                        emb_norm = emb.view(B, ch).norm(dim=1).mean().item()
                        print(f"[CA] dec stage {i}, ch={ch}, emb_norm={emb_norm:.3f}")
            for blk in blocks:
                x = blk(x, csi, snr_scale=snr_scale)  # 只依赖VMambaBlock内部的双残差
            if debug:
                xmin, xmax = float(x.min().item()), float(x.max().item())
                xmean, xstd = float(x.mean().item()), float(x.std().item())
                print(f"[VMAMBA_DEC] after stage {i}: shape={x.shape}, min={xmin:.4f}, max={xmax:.4f}, mean={xmean:.4f}, std={xstd:.4f}")

            if i < len(self.ups):
                x = self.ups[i](x)
                # Deblock residual: x + Conv3x3(GELU(x))
                x = x + self.deblocks[i](x)
                if debug:
                    xmin, xmax = float(x.min().item()), float(x.max().item())
                    xmean, xstd = float(x.mean().item()), float(x.std().item())
                    print(f"[VMAMBA_DEC] after Upsample+Deblock {i}: shape={x.shape}, min={xmin:.4f}, max={xmax:.4f}, mean={xmean:.4f}, std={xstd:.4f}")

        # 参照MambaJSCC原版：最后一层head处理
        x = self.head(x)
        if debug:
            xmin, xmax = float(x.min().item()), float(x.max().item())
            xmean, xstd = float(x.mean().item()), float(x.std().item())
            print(f"[VMAMBA_DEC] final head to output: shape={x.shape}, min={xmin:.4f}, max={xmax:.4f}, mean={xmean:.4f}, std={xstd:.4f}")

        return x


class VMambaJSCC2D(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        channels: Optional[List[int]] = None,
        depths: Optional[List[int]] = None,
        d_s: int = 4,  # 降低符号维度：从8/24→4，达到1.6kbps目标
        csi_dim: int = 4,
        freq_downsample_stages: int = 2,
        channel_adaptive: str = "no",  # 默认使用轻量版（'no' 或 'CA'/'ca'）
        lightweight_config: str = "all_native"  # 渐进式CSI处理
    ) -> None:
        super().__init__()
        # DeLight风格：深而窄配置 - 重点是块内结构变窄，层数变深
        if channels is None:
            # 外层通道适度减小：[48, 64, 96, 128] → [32, 48, 64, 64]
            channels = [32, 48, 64, 64]  # 保持表达能力的同时减少冗余
        if depths is None:
            # 关键：增加深度，体现"深而窄"的"深"
            depths = [2, 3, 3, 2]  # 从[2,2,2,2] → [3,4,4,3]，中间层更深

        # 兼容大小写：允许上层传入 "ca" 或 "CA"
        # 统一为小写以便与 CLI choices 对齐（'no' / 'ca' / 'attn'）
        self.channel_adaptive = str(channel_adaptive).lower()
        self.lightweight_config = lightweight_config

        self.encoder = VMambaEncoder2D(
            in_ch=in_ch,
            out_ch=d_s,  # 编码器输出到符号维度
            channels=channels,
            depths=depths,
            csi_dim=csi_dim,
            freq_downsample_stages=freq_downsample_stages,
            lightweight_config=lightweight_config,
        )
        self.decoder = VMambaDecoder2D(
            in_ch=d_s,  # 解码器输入来自符号维度
            channels=list(reversed(channels)),
            depths=list(reversed(depths)),
            csi_dim=csi_dim,
            out_ch=out_ch,
            freq_downsample_stages=freq_downsample_stages,
            lightweight_config=lightweight_config,
        )
        # 注释：sym_proj和sym_proj_dec已经被encoder/decoder的head替代

        # 信道自适应模块（CA：SNR-aware 调制；attn：自适应门控；ssm：可叠加使用CA+SSM遗忘调制）
        if self.channel_adaptive in ("ca", "ssm"):
            # 使用统一的 SNR embedding 维度（取最高通道数 channels[-1]），
            # 再通过每个 stage 的 proj_list_* 投射到各自通道数，实现 per-stage
            # 的 SNR 调制，更贴近原始 MambaJSCC 的设计。
            self.SNR_embedding = SNR_embedding(25, channels[-1], channels[-1])
            self.proj_list_enc = nn.ModuleList()
            self.proj_list_dec = nn.ModuleList()
            # 为编码器和解码器的每一层添加投影
            for ch in channels:
                self.proj_list_enc.append(nn.Linear(channels[-1], ch))
            for ch in reversed(channels):
                self.proj_list_dec.append(nn.Linear(channels[-1], ch))
        elif self.channel_adaptive == "attn":
            self.hidden_dim = int(channels[-1] * 1.5)
            self.layer_num = 7
            self.bm_list = nn.ModuleList()
            self.sm_list = nn.ModuleList()
            self.sm_list.append(nn.Linear(channels[-1], self.hidden_dim))
            for i in range(self.layer_num):
                if i == self.layer_num - 1:
                    outdim = channels[-1]
                else:
                    outdim = self.hidden_dim
                self.bm_list.append(AdaptiveModulator(self.hidden_dim))
                self.sm_list.append(nn.Linear(self.hidden_dim, outdim))
            self.sigmoid = nn.Sigmoid()

        # 添加MambaJSCC风格的权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """MambaJSCC风格的权重初始化"""
        if isinstance(m, nn.Linear):
            # 使用xavier初始化，提供更好的训练稳定性
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            # Conv2d层使用xavier初始化
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def encode(self, x_img: torch.Tensor, csi_vec: torch.Tensor, SNR: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        # 提取 SNR proxy，用于 CA embedding 与/或 SSM 内核级别调制。
        snr_embedding: Optional[torch.Tensor] = None
        snr_scale: Optional[torch.Tensor] = None

        # 先根据配置范围对 snr_proxy 做线性归一化，得到逐样本 snr_scale∈[0,1]，用于 SSM 遗忘调制。
        snr_proxy = csi_vec[:, 0]  # [B]
        snr_min, snr_max = -5.0, 10.0  # 与训练脚本中 snr_min_db/sn_rmax_db 对齐
        with torch.no_grad():
            snr_scale = ((snr_proxy - snr_min) / (snr_max - snr_min)).clamp(0.0, 1.0)  # [B]

        # 若启用 CA 或 SSM 模式，则额外构造离散 SNR embedding；
        # 其中 SSM 模式会同时使用 CA 偏置 + SSM 遗忘调制。
        if self.channel_adaptive in ("ca", "ssm"):
            # 每个样本一个 snr_idx，而不是整批共享
            snr_idx = (snr_proxy + 12.0).clamp(0.0, 24.0).long()  # [B]
            SNR_embed = snr_idx
            snr_embedding = self.SNR_embedding(SNR_embed)  # [B, channels[-1]]
            if bool(int(os.environ.get("DBG_CA", "0"))):
                with torch.no_grad():
                    emb_norm = snr_embedding.norm(dim=1).mean().item()
                    print(f"[CA] encode snr_idx_mean={snr_idx.float().mean().item():.2f}, emb_norm={emb_norm:.3f}")

        # 参照 MambaJSCC：编码器内部各 stage 入口施加 SNR 调制，而非仅
        # 在最终符号图上做一次全局偏置。
        if snr_embedding is not None:
            s, hw = self.encoder(
                x_img,
                csi_vec,
                snr_embedding=snr_embedding,
                proj_list=self.proj_list_enc,
                snr_scale=snr_scale,
            )
        else:
            s, hw = self.encoder(x_img, csi_vec, snr_scale=snr_scale)
        B, C, H, W = s.shape

        # 符号归一化：确保特征在合理范围内
        s = F.layer_norm(s.permute(0, 2, 3, 1), [C]).permute(0, 3, 1, 2)

        # 保留符号幅度，避免过度归一化导致输出塌缩为常数
        return None, s, hw  # tokens已经不需要了

    def decode(self, s_noisy: torch.Tensor, csi_vec: torch.Tensor, hw: Tuple[int, int], SNR: float = 0.0) -> torch.Tensor:
        """解码符号为 2D 图像。

        相比早期版本，这里移除了 decode 端对 ``s_noisy`` 的 LayerNorm，
        直接将符号送入解码器，仅在需要信道自适应（CA）时施加一个轻微的
        加性调制。这样可以保留 encode 端已学习到的幅度信息，避免在
        decode 入口再次抹平符号能量。
        """
        import os
        debug = bool(int(os.environ.get("DBG_STAGE25", "0")))

        # 提取 SNR 值用于解码器内部各 stage 的调制
        snr_embedding: Optional[torch.Tensor] = None
        snr_scale: Optional[torch.Tensor] = None

        # 与编码端一致：先对 snr_proxy 做线性归一化，得到逐样本 snr_scale∈[0,1]
        snr_proxy = csi_vec[:, 0]  # [B]
        snr_min, snr_max = -5.0, 10.0
        with torch.no_grad():
            snr_scale = ((snr_proxy - snr_min) / (snr_max - snr_min)).clamp(0.0, 1.0)  # [B]

        if self.channel_adaptive in ("ca", "ssm"):
            snr_idx = (snr_proxy + 12.0).clamp(0.0, 24.0).long()  # [B]
            SNR_embed = snr_idx
            snr_embedding = self.SNR_embedding(SNR_embed)
            if bool(int(os.environ.get("DBG_CA", "0"))):
                with torch.no_grad():
                    emb_norm = snr_embedding.norm(dim=1).mean().item()
                    print(f"[CA] decode snr_idx_mean={snr_idx.float().mean().item():.2f}, emb_norm={emb_norm:.3f}")

        B, C, H, W = s_noisy.shape
        if debug:
            print(f"[DECODER] s_noisy: {s_noisy.shape}")

        # 直接使用经过信道的符号，不再在 decode 入口做 LayerNorm，
        # 以保留幅度差异；SNR 调制在解码器内部各 stage 入口统一施加。
        s_in = s_noisy

        if snr_embedding is not None:
            output = self.decoder(
                s_in,
                csi_vec,
                snr_embedding=snr_embedding,
                proj_list=self.proj_list_dec,
                snr_scale=snr_scale,
            )
        else:
            output = self.decoder(s_in, csi_vec, snr_scale=snr_scale)

        if debug:
            fmin, fmax = float(output.min().item()), float(output.max().item())
            fmean, fstd = float(output.mean().item()), float(output.std().item())
            print(f"[DECODER] final output: shape={output.shape}, min={fmin:.4f}, max={fmax:.4f}, mean={fmean:.4f}, std={fstd:.4f}")

        return output
