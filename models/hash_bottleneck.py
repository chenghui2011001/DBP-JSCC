import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
import math
import os
from typing import Tuple, Optional, Dict, Any, List


_JSCC_FSK_BER_TABLE: Dict[str, torch.Tensor] | None = None


def _load_jscc_fsk_ber_table() -> Dict[str, torch.Tensor] | None:
    """Load JSCC+FSK BER(SNR) table from JSON if configured.

    The environment variable ``JSCC_FSK_BER_TABLE`` should point to a JSON
    file with at least two fields: ``snr_db`` and ``ber_mean``. An optional
    ``ber_std`` field encodes the empirical standard deviation. Both are
    stored so that callers can either use a deterministic
    ``mean + k * std`` mapping or sample from ``N(mean, std)`` per SNR.
    """

    import json as _json

    global _JSCC_FSK_BER_TABLE
    if _JSCC_FSK_BER_TABLE is not None:
        return _JSCC_FSK_BER_TABLE

    path = os.environ.get("JSCC_FSK_BER_TABLE", "")
    if not path:
        _JSCC_FSK_BER_TABLE = {}
        return None
    try:
        with open(path, "r", encoding="utf-8") as _f:
            data = _json.load(_f)
        snr_arr = np.asarray(data.get("snr_db", []), dtype=np.float32)
        ber_mean_arr = np.asarray(data.get("ber_mean", []), dtype=np.float32)
        ber_std_arr = np.asarray(data.get("ber_std", []), dtype=np.float32)
        if ber_std_arr.size != snr_arr.size:
            ber_std_arr = np.zeros_like(snr_arr, dtype=np.float32)

        if snr_arr.size == 0 or ber_mean_arr.size == 0 or snr_arr.size != ber_mean_arr.size:
            _JSCC_FSK_BER_TABLE = {}
            return None

        order = np.argsort(snr_arr)
        snr_sorted = torch.from_numpy(snr_arr[order].copy()).to(torch.float32)
        mean_sorted = torch.from_numpy(ber_mean_arr[order].copy()).to(torch.float32)
        std_sorted = torch.from_numpy(ber_std_arr[order].copy()).to(torch.float32)
        _JSCC_FSK_BER_TABLE = {"snr": snr_sorted, "mean": mean_sorted, "std": std_sorted}
    except Exception:
        _JSCC_FSK_BER_TABLE = {}
        return None
    return _JSCC_FSK_BER_TABLE


def _lookup_jscc_fsk_ber_torch(snr_db: torch.Tensor) -> Optional[torch.Tensor]:
    """Vectorized BER(SNR) lookup on a torch Tensor.

    Returns a tensor broadcastable to ``snr_db`` or ``None`` if the table
    is unavailable. Linear interpolation is used between nearest SNR points,
    and values are clamped to [0, 0.5].
    """

    table = _load_jscc_fsk_ber_table()
    if not table:
        return None

    snrs = table["snr"].to(snr_db.device)
    ber_mean = table["mean"].to(snr_db.device)
    ber_std = table["std"].to(snr_db.device)
    if snrs.numel() == 0 or ber_mean.numel() == 0:
        return None

    x = snr_db.to(torch.float32)

    x_min = snrs[0]
    x_max = snrs[-1]
    x_clamped = torch.clamp(x, float(x_min), float(x_max))

    idx = torch.bucketize(x_clamped, snrs)
    idx0 = torch.clamp(idx - 1, 0, snrs.numel() - 1)
    idx1 = torch.clamp(idx,     0, snrs.numel() - 1)
    x0 = snrs[idx0]
    x1 = snrs[idx1]
    m0 = ber_mean[idx0]
    m1 = ber_mean[idx1]
    s0 = ber_std[idx0]
    s1 = ber_std[idx1]

    denom = (x1 - x0).clamp_min(1e-6)
    t = (x_clamped - x0) / denom
    mean_x = m0 + t * (m1 - m0)
    std_x = s0 + t * (s1 - s0)
    std_x = std_x.clamp_min(0.0)

    mode = os.environ.get("JSCC_FSK_BER_MODE", "gaussian").lower()
    try:
        k_std = float(os.environ.get("JSCC_FSK_BER_STD_K", "1.0"))
    except Exception:
        k_std = 1.0

    if mode == "gaussian":
        eps = torch.randn_like(mean_x)
        val = mean_x + k_std * std_x * eps
    else:  # deterministic fallback: mean + k * std
        val = mean_x + k_std * std_x

    return torch.clamp(val, 0.0, 0.5)


class GreedyHashFunction(Function):
    """
    GreedyHash with Straight-Through Estimator
    Forward: h = sign(u)
    Backward: gradient passes through unchanged
    """
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


class BiHalfHashFunction(Function):
    """
    Bi-half hash layer for maximum bit entropy
    Ensures each bit dimension has 50% +1 and 50% -1 across the batch
    """
    @staticmethod
    def forward(ctx, U: torch.Tensor, gamma: float = 6.0) -> torch.Tensor:
        # U: [B, T, K] - batch, time, hash_bits
        original_shape = U.shape

        # Flatten to [B*T, K] for processing.
        # NOTE: avoid using ``-1`` when K==0, since ``view(-1, 0)`` is
        # ambiguous for tensors with zero elements and raises a runtime
        # error in PyTorch. Explicitly compute the flattened first
        # dimension instead, which is well-defined even when K==0.
        if U.dim() < 2:
            # Fallback for unexpected shapes; keep original behavior
            U_flat = U.view(U.numel(), U.size(-1))
        else:
            B, T = U.shape[0], U.shape[1]
            U_flat = U.view(B * T, U.size(-1))

        # Sort each bit dimension independently
        _, index = U_flat.sort(0, descending=True)
        N, D = U_flat.shape

        # Create balanced binary assignment: top 50% -> +1, bottom 50% -> -1
        # 保持与 U_flat 相同的 dtype（在 AMP/FP16 下避免 scatter 类型不匹配）。
        B_creat = torch.cat([
            torch.ones([int(N/2), D], device=U.device, dtype=U.dtype),
            -torch.ones([N - int(N/2), D], device=U.device, dtype=U.dtype),
        ], dim=0)

        B_flat = torch.zeros_like(U_flat).scatter_(0, index, B_creat)
        B = B_flat.view(original_shape)

        ctx.save_for_backward(U, B)
        ctx.gamma = gamma

        return B

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        U, B = ctx.saved_tensors
        gamma = ctx.gamma

        # Add regularization gradient to encourage U -> B
        add_g = (U - B) / (B.numel())
        grad = grad_output + gamma * add_g

        return grad, None


class HashBottleneck(nn.Module):
    """
    Binary Hash Bottleneck for Speech JSCC

    Combines:
    1. Linear projection to hash logits
    2. Bi-half + GreedyHash binarization
    3. Bit channel simulation (BSC/BPSK+AWGN)
    4. Hash decoder for reconstruction
    """

    def __init__(self,
                 input_dim: int,
                 hash_bits: int,
                 decoder_hidden: int = 128,
                 output_dim: Optional[int] = None,
                 hash_method: str = 'greedy',
                 gamma: float = 6.0,
                 channel_type: str = 'bsc'):
        super().__init__()

        self.input_dim = input_dim
        self.hash_bits = hash_bits
        self.output_dim = output_dim or input_dim
        self.hash_method = hash_method
        self.gamma = gamma
        self.channel_type = channel_type

        # Hash encoder: continuous -> hash logits
        self.hash_encoder = nn.Linear(input_dim, hash_bits)

        # Hash decoder: bits -> continuous latent
        self.hash_decoder = nn.Sequential(
            nn.Linear(hash_bits, decoder_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(decoder_hidden, self.output_dim),
            nn.LayerNorm(self.output_dim)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def hash_layer(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply hash function based on method.

        - 当 ``hash_method='bihalf'`` 时：
          - 训练期使用 BiHalfHash（batch 内 50%/+1，50%/-1），
            通过梯度中额外的 (U-B) 项和外部正则共同塑造 logits 分布；
          - 推理期自动退化为 GreedyHash/sign，避免在 batch 维度上的
            排序依赖，保证单样本/小 batch 推理时的确定性与可复现性。
        - 当 ``hash_method='greedy'`` 时：训练/推理均使用 GreedyHash。
        - 当 ``hash_method='sign'`` 时：直接返回 torch.sign（无 STE）。
        """

        if (not self.training) and self.hash_method == 'bihalf':
            # Inference: use greedy/sign for deterministic hash codes
            return GreedyHashFunction.apply(logits)

        if self.hash_method == 'greedy':
            return GreedyHashFunction.apply(logits)
        elif self.hash_method == 'bihalf':
            return BiHalfHashFunction.apply(logits, self.gamma)
        elif self.hash_method == 'sign':
            return torch.sign(logits)
        else:
            raise ValueError(f"Unknown hash method: {self.hash_method}")

    def channel_simulation(self,
                          bits: torch.Tensor,
                          channel_params: Dict[str, float],
                          mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Simulate bit-level channel noise

        Args:
            bits: Binary hash codes [B, T, K]
            channel_params: Channel parameters (e.g., {'ber': 0.1, 'snr_db': 10})
            mask: Valid patch mask [B, T], only apply noise to valid patches
        """
        if not self.training:
            return bits

        if self.channel_type == 'bsc':
            # Binary Symmetric Channel：
            #  - 优先使用显式传入的 'ber'；
            #  - 若缺失且提供 'snr_db'，则尝试从 JSCC_FSK_BER_TABLE
            #    查表得到 JSCC+FSK 源比特 BER；
            #  - 否则退化为无噪声通道。
            ber_val = channel_params.get('ber', None)
            if ber_val is not None:
                ber_t = torch.as_tensor(ber_val, device=bits.device, dtype=bits.dtype)
            elif 'snr_db' in channel_params:
                snr_val = channel_params.get('snr_db', 0.0)
                snr_t = torch.as_tensor(snr_val, device=bits.device, dtype=bits.dtype)
                ber_lookup = _lookup_jscc_fsk_ber_torch(snr_t)
                if ber_lookup is None:
                    return bits
                ber_t = ber_lookup.to(bits.dtype)
            else:
                ber_t = torch.zeros((), device=bits.device, dtype=bits.dtype)

            # 对齐维度 [B,T,(1)] 以便与 bits 广播
            while ber_t.dim() < bits.dim():
                ber_t = ber_t.unsqueeze(-1)
            flip_mask = torch.rand_like(bits) < ber_t

            # Only apply noise to valid patches if mask is provided
            if mask is not None:
                # Convert mask to bool and expand to match bits shape
                if mask.dtype != torch.bool:
                    mask_bool = mask > 0.5
                else:
                    mask_bool = mask
                mask_expanded = mask_bool.unsqueeze(-1).expand_as(bits)
                flip_mask = flip_mask & mask_expanded

            return torch.where(flip_mask, -bits, bits)

        elif self.channel_type == 'bpsk_awgn':
            # BPSK + AWGN（支持标量或逐帧Tensor的SNR[dB]）
            snr_db_val = channel_params.get('snr_db', 10.0)
            snr_t = torch.as_tensor(snr_db_val, device=bits.device, dtype=bits.dtype)
            # 噪声标准差 = 10^(-SNR[dB]/20)
            noise_std = torch.pow(torch.tensor(10.0, device=bits.device, dtype=bits.dtype), -snr_t / 20.0)
            # 对齐维度 [B,T,(1)]
            while noise_std.dim() < bits.dim():
                noise_std = noise_std.unsqueeze(-1)
            noise = torch.randn_like(bits) * noise_std

            # Only apply noise to valid patches if mask is provided
            if mask is not None:
                # Convert mask to proper dtype and expand to match bits shape
                if mask.dtype != torch.bool:
                    mask_bool = mask > 0.5
                else:
                    mask_bool = mask
                mask_expanded = mask_bool.unsqueeze(-1).expand_as(bits)
                noise = noise * mask_expanded

            return bits + noise

        else:
            return bits



    def forward(self,
                x: torch.Tensor,
                channel_params: Optional[Dict[str, float]] = None,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hash bottleneck

        Args:
            x: Input latent [B, T, D] or [B, N_patches, D]
            channel_params: Channel simulation parameters
            mask: Valid patch mask [B, T] or [B, N_patches], 1 for valid, 0 for padding

        Returns:
            Dictionary with:
            - hash_logits: Pre-binarization logits [B, T, K]
            - hash_bits_clean: Clean binary hash [B, T, K]
            - hash_bits_noisy: Channel-corrupted hash [B, T, K]
            - reconstructed: Decoded continuous latent [B, T, D_out]
            - mask: Valid patch mask (passed through)
        """
        # Encode to hash logits
        hash_logits = self.hash_encoder(x)  # [B, T, K]

        # Binarize using selected hash method
        hash_bits_clean = self.hash_layer(hash_logits)

        # Channel simulation
        if channel_params is not None:
            hash_bits_noisy = self.channel_simulation(hash_bits_clean, channel_params, mask)
        else:
            hash_bits_noisy = hash_bits_clean

        # Decode back to continuous space
        reconstructed = self.hash_decoder(hash_bits_noisy)

        result = {
            'hash_logits': hash_logits,
            'hash_bits_clean': hash_bits_clean,
            'hash_bits_noisy': hash_bits_noisy,
            'reconstructed': reconstructed
        }

        # Pass through mask for downstream processing
        if mask is not None:
            result['mask'] = mask

        return result

    def compute_hash_regularization(self,
                                   hash_logits: torch.Tensor,
                                   hash_bits: torch.Tensor,
                                   mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute hash-specific regularization losses

        Args:
            hash_logits: Pre-binarization logits [B, T, K]
            hash_bits: Binarized hash [B, T, K]
            mask: Valid patch mask [B, T], only compute losses on valid patches
        """
        losses = {}

        # Apply mask if provided - unified dtype handling
        if mask is not None:
            # Convert mask to proper dtype
            if mask.dtype != torch.bool:
                mask_bool = mask > 0.5
            else:
                mask_bool = mask
            mask_expanded = mask_bool.unsqueeze(-1).expand_as(hash_bits)  # [B, T, K]

            # Zero out invalid patches
            hash_bits_masked = hash_bits * mask_expanded
            hash_logits_masked = hash_logits * mask_expanded
            valid_count = mask_bool.sum()
        else:
            hash_bits_masked = hash_bits
            hash_logits_masked = hash_logits
            mask_expanded = None
            valid_count = hash_bits.numel() / hash_bits.size(-1)  # Total valid elements

        # Bit balance: encourage each bit to be 50% +1, 50% -1 (only on valid patches)
        if mask is not None:
            # Compute mean only over valid patches
            bit_means = hash_bits_masked.sum(dim=[0, 1]) / (valid_count + 1e-8)  # [K]
        else:
            bit_means = hash_bits.mean(dim=[0, 1])  # [K]
        balance_target = torch.zeros_like(bit_means)
        losses['bit_balance'] = F.mse_loss(bit_means, balance_target)

        # Bit decorrelation: minimize correlation between different bits (only valid patches)
        if mask is not None:
            hash_flat = hash_bits_masked[mask_expanded.any(dim=-1)]  # Only valid patches
        else:
            hash_flat = hash_bits.view(-1, hash_bits.size(-1))  # [B*T, K]

        if hash_flat.size(0) > 1:
            correlation_matrix = torch.corrcoef(hash_flat.t())  # [K, K]
            # 防止某些 bit 恒定导致 corrcoef 产生 NaN/Inf
            correlation_matrix = torch.nan_to_num(
                correlation_matrix, nan=0.0, posinf=0.0, neginf=0.0
            )
            # Penalize off-diagonal elements
            corr_mask = ~torch.eye(
                correlation_matrix.size(0),
                dtype=torch.bool,
                device=correlation_matrix.device,
            )
            losses['bit_decorrelation'] = correlation_matrix[corr_mask].abs().mean()
        else:
            losses['bit_decorrelation'] = torch.tensor(0.0, device=hash_bits.device)

        # Quantization loss: encourage logits to be close to ±1 (only valid patches)
        if mask is not None:
            quant_err = torch.abs(torch.abs(hash_logits) - 1.0)
            quant_err_masked = quant_err * mask_expanded
            losses['quantization'] = quant_err_masked.sum() / (mask_expanded.sum() + 1e-8)
        else:
            losses['quantization'] = torch.mean(torch.abs(torch.abs(hash_logits) - 1.0))

        # Entropy regularization: maximum entropy per bit (only valid patches)
        probs = torch.sigmoid(hash_logits_masked)
        # 在熵与 rate_kl 中统一使用裁剪后的概率，避免 0/1 → log(0) 产生 NaN/Inf
        p = torch.clamp(probs, 1e-6, 1.0 - 1e-6)
        entropy = -p * torch.log(p) - (1 - p) * torch.log(1 - p)
        if mask is not None:
            entropy = entropy * mask_expanded
            losses['entropy'] = -entropy.sum() / (mask_expanded.sum() + 1e-8)
        else:
            losses['entropy'] = -entropy.mean()

        # Rate loss: Bernoulli KL(q(b|x) || Bernoulli(0.5)) (only valid patches)
        prior_p = 0.5
        log_prior = math.log(prior_p)
        log_prior_comp = math.log(1.0 - prior_p)
        rate_kl = p * (torch.log(p) - log_prior) + (1.0 - p) * (torch.log(1.0 - p) - log_prior_comp)
        if mask is not None:
            rate_kl = rate_kl * mask_expanded
            losses['rate_kl'] = rate_kl.sum() / (mask_expanded.sum() + 1e-8)
        else:
            losses['rate_kl'] = rate_kl.mean()

        return losses

    @torch.no_grad()
    def decode_hash_codec(self, bits: torch.Tensor) -> torch.Tensor:
        """Decode external hash bits into continuous latent.

        仅用于离线/推理场景：
        - 输入可以是 {0,1} 或 {-1,+1}，也可以是浮点近似；
        - 内部统一转换为 {-1,+1} 之后，经 ``hash_decoder`` 映射回
          连续空间 [B, T, output_dim]。
        该接口与 ``forward`` 中内部使用的解码路径保持一致，但跳过
        hash_encoder 与信道噪声，仅依赖显式给定的 bitstream。
        """

        if not isinstance(bits, torch.Tensor):
            bits = torch.as_tensor(bits, device=self.hash_encoder.weight.device)

        # 保证 float32 并放到正确设备
        b = bits.to(device=self.hash_encoder.weight.device, dtype=torch.float32)

        if b.numel() == 0:
            # 空张量：直接走解码器以保持形状一致
            return self.hash_decoder(b)

        bmin = float(b.min().item())
        bmax = float(b.max().item())

        # 若为 0/1 编码，则转换为 -1/+1；
        # 否则视作已经在 BPSK(-1/+1) 域（可能带 AWGN 噪声）的软值，
        # 直接交给解码器处理，仅做轻微裁剪以防极端数值。
        if bmin >= 0.0 and bmax <= 1.0:
            b = b * 2.0 - 1.0

        b = torch.clamp(b, -3.0, 3.0)

        # 确保形状至少为 [B,T,K]
        if b.dim() == 2:
            b = b.unsqueeze(1)

        return self.hash_decoder(b)

    def get_bitrate(self, frame_rate: float = 50.0, patch_len: Optional[int] = None) -> float:
        """
        Calculate nominal bitrate in bps

        Args:
            frame_rate: Original frame rate (e.g., 50 Hz)
            patch_len: Number of frames per patch (e.g., 8), if using multi-patch

        Returns:
            Bitrate in bps accounting for patch structure
        """
        if patch_len is not None:
            # Multi-patch: effective rate = frame_rate / patch_len
            effective_rate = frame_rate / patch_len
            return self.hash_bits * effective_rate
        else:
            # Traditional single-token calculation
            return self.hash_bits * frame_rate

    @torch.no_grad()
    def analyze_bit_statistics(self, hash_bits: torch.Tensor) -> Dict[str, float]:
        """Analyze hash bit statistics for monitoring"""
        hash_flat = hash_bits.view(-1, hash_bits.size(-1))

        stats = {}
        stats['bit_balance'] = hash_flat.mean(dim=0).abs().mean().item()
        stats['bit_utilization'] = (hash_flat.abs().mean()).item()

        if hash_flat.size(0) > 1:
            corr_matrix = torch.corrcoef(hash_flat.t())
            mask = ~torch.eye(
                corr_matrix.size(0),
                dtype=torch.bool,
                device=corr_matrix.device,
            )
            stats['bit_correlation'] = corr_matrix[mask].abs().mean().item()
        else:
            stats['bit_correlation'] = 0.0

        return stats


class GroupedHashBottleneck(nn.Module):
    """Grouped version of :class:`HashBottleneck`.

    在不增加总 bit 数的前提下，将通道维按组拆分，每组走一个独立的
    HashBottleneck（`input_dim_g -> hash_bits_g`），最后在 bit 维和重构
    latent 维度上拼接回来。

    这样可以让不同子空间学到更稳定的语义（例如低频/高频或不同通道
    子带），同时保留与原始 HashBottleneck 基本一致的接口：

    - ``forward`` 返回与 HashBottleneck 相同的字典结构；
    - ``compute_hash_regularization`` 在拼接后的 logits/bits 上统一计算，
      便于保持损失定义不变；
    - ``get_bitrate`` 与原实现兼容，只依赖总 bit 数。

    默认通过等分的方式对通道和 bit 维度进行分组；分组策略保持确定性
    且只依赖输入维度和总 bit 数，因此旧 checkpoint 在 resume 时会被
    完整跳过（键名不同），新的 GroupedHashBottleneck 将从头训练。
    """

    def __init__(
        self,
        input_dim: int,
        hash_bits: int,
        num_groups: int = 4,
        decoder_hidden: int = 128,
        hash_method: str = "greedy",
        gamma: float = 6.0,
        channel_type: str = "bpsk_awgn",
    ) -> None:
        super().__init__()

        if num_groups <= 1:
            # 退化为单组等价 HashBottleneck，保持接口兼容
            self.num_groups = 1
            self.input_dims: List[int] = [input_dim]
            self.hash_bits_per_group: List[int] = [hash_bits]
        else:
            # 等分通道和 bit，最后一组吸收余数，确保总和一致
            base_c = input_dim // num_groups
            base_k = hash_bits // num_groups
            self.input_dims = [base_c] * num_groups
            self.hash_bits_per_group = [base_k] * num_groups
            self.input_dims[-1] += input_dim - sum(self.input_dims)
            self.hash_bits_per_group[-1] += hash_bits - sum(self.hash_bits_per_group)
            self.num_groups = num_groups

        self.total_input_dim = input_dim
        self.total_hash_bits = hash_bits

        groups: List[HashBottleneck] = []
        for d_g, k_g in zip(self.input_dims, self.hash_bits_per_group):
            groups.append(
                HashBottleneck(
                    input_dim=d_g,
                    hash_bits=k_g,
                    decoder_hidden=decoder_hidden,
                    output_dim=d_g,
                    hash_method=hash_method,
                    gamma=gamma,
                    channel_type=channel_type,
                )
            )
        self.groups = nn.ModuleList(groups)

    def _split(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Split last dim of ``x`` according to ``input_dims``.

        Args:
            x: Tensor of shape ``[B, T, D]`` with ``D == total_input_dim``.
        """
        assert x.size(-1) == self.total_input_dim, "GroupedHashBottleneck: input_dim mismatch"
        outs: List[torch.Tensor] = []
        start = 0
        for d in self.input_dims:
            outs.append(x[..., start:start + d])
            start += d
        return outs

    def _concat_last(self, xs: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(xs, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        channel_params: Optional[Dict[str, float]] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # x: [B, T, D]
        x_groups = self._split(x)

        hash_logits_list: List[torch.Tensor] = []
        bits_clean_list: List[torch.Tensor] = []
        bits_noisy_list: List[torch.Tensor] = []
        recon_list: List[torch.Tensor] = []

        # channel_params / mask 在各组之间共享（逐帧 SNR 或 BER 等）
        for x_g, hb in zip(x_groups, self.groups):
            ret = hb(x_g, channel_params=channel_params, mask=mask)
            hash_logits_list.append(ret["hash_logits"])
            bits_clean_list.append(ret["hash_bits_clean"])
            bits_noisy_list.append(ret["hash_bits_noisy"])
            recon_list.append(ret["reconstructed"])

        hash_logits = self._concat_last(hash_logits_list)
        hash_bits_clean = self._concat_last(bits_clean_list)
        hash_bits_noisy = self._concat_last(bits_noisy_list)
        reconstructed = self._concat_last(recon_list)

        result = {
            "hash_logits": hash_logits,
            "hash_bits_clean": hash_bits_clean,
            "hash_bits_noisy": hash_bits_noisy,
            "reconstructed": reconstructed,
        }
        if mask is not None:
            result["mask"] = mask
        return result

    def compute_hash_regularization(
        self,
        hash_logits: torch.Tensor,
        hash_bits: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Reuse HashBottleneck's regularization on concatenated bits.

        直接拷贝 HashBottleneck 的实现逻辑，以保持损失定义完全兼容；
        这里 ``hash_logits``/``hash_bits`` 已经是按组拼接后的整体张量。
        """

        losses: Dict[str, torch.Tensor] = {}

        if mask is not None:
            if mask.dtype != torch.bool:
                mask_bool = mask > 0.5
            else:
                mask_bool = mask
            mask_expanded = mask_bool.unsqueeze(-1).expand_as(hash_bits)
            hash_bits_masked = hash_bits * mask_expanded
            hash_logits_masked = hash_logits * mask_expanded
            valid_count = mask_bool.sum()
        else:
            hash_bits_masked = hash_bits
            hash_logits_masked = hash_logits
            mask_expanded = None
            valid_count = hash_bits.numel() / hash_bits.size(-1)

        # Bit balance
        if mask is not None:
            bit_means = hash_bits_masked.sum(dim=[0, 1]) / (valid_count + 1e-8)
        else:
            bit_means = hash_bits.mean(dim=[0, 1])
        balance_target = torch.zeros_like(bit_means)
        losses["bit_balance"] = F.mse_loss(bit_means, balance_target)

        # Bit decorrelation
        if mask is not None:
            hash_flat = hash_bits_masked[mask_expanded.any(dim=-1)]
        else:
            hash_flat = hash_bits.view(-1, hash_bits.size(-1))
        if hash_flat.size(0) > 1:
            correlation_matrix = torch.corrcoef(hash_flat.t())
            # 防止某些 bit 恒定导致 corrcoef 产生 NaN/Inf
            correlation_matrix = torch.nan_to_num(
                correlation_matrix, nan=0.0, posinf=0.0, neginf=0.0
            )
            corr_mask = ~torch.eye(
                correlation_matrix.size(0),
                dtype=torch.bool,
                device=correlation_matrix.device,
            )
            losses["bit_decorrelation"] = correlation_matrix[corr_mask].abs().mean()
        else:
            losses["bit_decorrelation"] = torch.tensor(0.0, device=hash_bits.device)

        # Quantization loss
        if mask is not None:
            quant_err = torch.abs(torch.abs(hash_logits) - 1.0)
            quant_err_masked = quant_err * mask_expanded
            losses["quantization"] = quant_err_masked.sum() / (mask_expanded.sum() + 1e-8)
        else:
            losses["quantization"] = torch.mean(torch.abs(torch.abs(hash_logits) - 1.0))

        # Entropy regularization
        probs = torch.sigmoid(hash_logits_masked)
        # 在熵与 rate_kl 中统一使用裁剪后的概率，避免 0/1 → log(0) 产生 NaN/Inf
        p = torch.clamp(probs, 1e-6, 1.0 - 1e-6)
        entropy = -p * torch.log(p) - (1 - p) * torch.log(1 - p)
        if mask is not None:
            entropy = entropy * mask_expanded
            losses["entropy"] = -entropy.sum() / (mask_expanded.sum() + 1e-8)
        else:
            losses["entropy"] = -entropy.mean()

        # Rate loss
        # Rate loss：复用裁剪后的概率 p
        prior_p = 0.5
        log_prior = math.log(prior_p)
        log_prior_comp = math.log(1.0 - prior_p)
        rate_kl = p * (torch.log(p) - log_prior) + (1.0 - p) * (torch.log(1.0 - p) - log_prior_comp)
        if mask is not None:
            rate_kl = rate_kl * mask_expanded
            losses["rate_kl"] = rate_kl.sum() / (mask_expanded.sum() + 1e-8)
        else:
            losses["rate_kl"] = rate_kl.mean()

        return losses

    def get_bitrate(self, frame_rate: float = 50.0, patch_len: Optional[int] = None) -> float:
        hb = HashBottleneck(
            input_dim=self.total_input_dim,
            hash_bits=self.total_hash_bits,
            decoder_hidden=1,
            output_dim=self.total_input_dim,
        )
        return hb.get_bitrate(frame_rate=frame_rate, patch_len=patch_len)

    @torch.no_grad()
    def analyze_bit_statistics(self, hash_bits: torch.Tensor) -> Dict[str, float]:
        hb = HashBottleneck(
            input_dim=self.total_input_dim,
            hash_bits=self.total_hash_bits,
            decoder_hidden=1,
            output_dim=self.total_input_dim,
        )
        return hb.analyze_bit_statistics(hash_bits)


class TwoStageHashBottleneck(nn.Module):
    """Two-stage residual HashBottleneck for more stable low-rate compression.

    实现思路：
    - Stage1: 使用 HashBottleneck 对输入 x 做第一次量化/重建，得到 s_hat1；
    - Stage2: 对残差 r = x - s_hat1 再走一次 HashBottleneck，得到 s_hat2；
    - 最终重建: s_hat = s_hat1 + s_hat2；
    - bit 维度按 [K1, K2] 拼接，总 bit 数保持不变（默认均分）。

    这样在总 bit 数不变的前提下，引入“逐级解释残差”的 RVQ 式结构，
    通常在激进低码率下比一次性 K-bit 量化更稳定、可训练性更好。
    """

    def __init__(
        self,
        input_dim: int,
        hash_bits: int,
        decoder_hidden: int = 128,
        output_dim: Optional[int] = None,
        hash_method: str = "bihalf",
        gamma: float = 6.0,
        channel_type: str = "bpsk_awgn",
        use_grouped: bool = True,
        num_groups: int = 4,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hash_bits = hash_bits
        self.output_dim = output_dim or input_dim
        self.hash_method = hash_method
        self.gamma = gamma
        self.channel_type = channel_type

        # 均分 bit 到两个 stage，第二级吸收余数
        k1 = hash_bits // 2
        k2 = hash_bits - k1
        self.stage_bits = (k1, k2)

        self.use_grouped = bool(use_grouped)
        self.num_groups = int(max(1, num_groups))

        if self.use_grouped and self.num_groups > 1:
            self.stage1 = GroupedHashBottleneck(
                input_dim=input_dim,
                hash_bits=k1,
                num_groups=self.num_groups,
                decoder_hidden=decoder_hidden,
                hash_method=hash_method,
                gamma=gamma,
                channel_type=channel_type,
            )
            self.stage2 = GroupedHashBottleneck(
                input_dim=input_dim,
                hash_bits=k2,
                num_groups=self.num_groups,
                decoder_hidden=decoder_hidden,
                hash_method=hash_method,
                gamma=gamma,
                channel_type=channel_type,
            )
        else:
            self.stage1 = HashBottleneck(
                input_dim=input_dim,
                hash_bits=k1,
                decoder_hidden=decoder_hidden,
                output_dim=self.output_dim,
                hash_method=hash_method,
                gamma=gamma,
                channel_type=channel_type,
            )
            self.stage2 = HashBottleneck(
                input_dim=input_dim,
                hash_bits=k2,
                decoder_hidden=decoder_hidden,
                output_dim=self.output_dim,
                hash_method=hash_method,
                gamma=gamma,
                channel_type=channel_type,
            )

    def forward(
        self,
        x: torch.Tensor,
        channel_params: Optional[Dict[str, float]] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Stage 1: coarse quantization
        out1 = self.stage1(x, channel_params=channel_params, mask=mask)
        s_hat1 = out1["reconstructed"]

        # Stage 2: residual quantization on r = x - s_hat1
        r = x - s_hat1
        out2 = self.stage2(r, channel_params=channel_params, mask=mask)
        s_hat2 = out2["reconstructed"]

        s_hat = s_hat1 + s_hat2

        hash_logits = torch.cat([out1["hash_logits"], out2["hash_logits"]], dim=-1)
        hash_bits_clean = torch.cat(
            [out1["hash_bits_clean"], out2["hash_bits_clean"]], dim=-1
        )
        hash_bits_noisy = torch.cat(
            [out1["hash_bits_noisy"], out2["hash_bits_noisy"]], dim=-1
        )

        result: Dict[str, torch.Tensor] = {
            "hash_logits": hash_logits,
            "hash_bits_clean": hash_bits_clean,
            "hash_bits_noisy": hash_bits_noisy,
            "reconstructed": s_hat,
        }
        if mask is not None:
            result["mask"] = mask
        # 方便调试：保留分 stage 的中间结果（不参与 loss 计算时可忽略）
        result["stage1"] = out1
        result["stage2"] = out2
        return result

    def compute_hash_regularization(
        self,
        hash_logits: torch.Tensor,
        hash_bits: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Reuse HashBottleneck's regularization on concatenated bits.

        TwoStageHashBottleneck 对外暴露的 ``hash_logits``/``hash_bits`` 已经是
        [K1+K2] 维的拼接表示，因此可以直接复用单级 HashBottleneck 的
        正则定义，保持损失项完全兼容。
        """

        losses: Dict[str, torch.Tensor] = {}

        if mask is not None:
            if mask.dtype != torch.bool:
                mask_bool = mask > 0.5
            else:
                mask_bool = mask
            mask_expanded = mask_bool.unsqueeze(-1).expand_as(hash_bits)
            hash_bits_masked = hash_bits * mask_expanded
            hash_logits_masked = hash_logits * mask_expanded
            valid_count = mask_bool.sum()
        else:
            hash_bits_masked = hash_bits
            hash_logits_masked = hash_logits
            mask_expanded = None
            valid_count = hash_bits.numel() / hash_bits.size(-1)

        # Bit balance
        if mask is not None:
            bit_means = hash_bits_masked.sum(dim=[0, 1]) / (valid_count + 1e-8)
        else:
            bit_means = hash_bits.mean(dim=[0, 1])
        balance_target = torch.zeros_like(bit_means)
        losses["bit_balance"] = F.mse_loss(bit_means, balance_target)

        # Bit decorrelation
        if mask is not None:
            hash_flat = hash_bits_masked[mask_expanded.any(dim=-1)]
        else:
            hash_flat = hash_bits.view(-1, hash_bits.size(-1))
        if hash_flat.size(0) > 1:
            correlation_matrix = torch.corrcoef(hash_flat.t())
            correlation_matrix = torch.nan_to_num(
                correlation_matrix, nan=0.0, posinf=0.0, neginf=0.0
            )
            corr_mask = ~torch.eye(
                correlation_matrix.size(0),
                dtype=torch.bool,
                device=correlation_matrix.device,
            )
            losses["bit_decorrelation"] = correlation_matrix[corr_mask].abs().mean()
        else:
            losses["bit_decorrelation"] = torch.tensor(0.0, device=hash_bits.device)

        # Quantization loss
        if mask is not None:
            quant_err = torch.abs(torch.abs(hash_logits) - 1.0)
            quant_err_masked = quant_err * mask_expanded
            losses["quantization"] = quant_err_masked.sum() / (mask_expanded.sum() + 1e-8)
        else:
            losses["quantization"] = torch.mean(torch.abs(torch.abs(hash_logits) - 1.0))

        # Entropy regularization
        probs = torch.sigmoid(hash_logits_masked)
        p = torch.clamp(probs, 1e-6, 1.0 - 1e-6)
        entropy = -p * torch.log(p) - (1 - p) * torch.log(1 - p)
        if mask is not None:
            entropy = entropy * mask_expanded
            losses["entropy"] = -entropy.sum() / (mask_expanded.sum() + 1e-8)
        else:
            losses["entropy"] = -entropy.mean()

        # Rate loss (Bernoulli KL to 0.5)
        prior_p = 0.5
        log_prior = math.log(prior_p)
        log_prior_comp = math.log(1.0 - prior_p)
        rate_kl = p * (torch.log(p) - log_prior) + (1.0 - p) * (torch.log(1.0 - p) - log_prior_comp)
        if mask is not None:
            rate_kl = rate_kl * mask_expanded
            losses["rate_kl"] = rate_kl.sum() / (mask_expanded.sum() + 1e-8)
        else:
            losses["rate_kl"] = rate_kl.mean()

        return losses


class TeacherDistillationModule(nn.Module):
    """
    Teacher distillation for hash bottleneck learning
    Supports various teacher models (StableCodec, HuBERT, etc.)
    """

    def __init__(self,
                 teacher_dim: int,
                 student_dim: int,
                 temperature: float = 1.0):
        super().__init__()

        self.teacher_dim = teacher_dim
        self.student_dim = student_dim
        self.temperature = temperature

        # Alignment projection if dimensions differ
        if teacher_dim != student_dim:
            self.projection = nn.Linear(student_dim, teacher_dim)
        else:
            self.projection = nn.Identity()

    def forward(self,
                student_features: torch.Tensor,
                teacher_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute distillation losses

        Args:
            student_features: Student latent [B, T, D_s]
            teacher_features: Teacher latent [B, T, D_t]
        """
        projected_student = self.projection(student_features)

        losses = {}

        # Feature alignment loss
        losses['feature_mse'] = F.mse_loss(projected_student, teacher_features)

        # Cosine similarity loss
        cos_sim = F.cosine_similarity(projected_student, teacher_features, dim=-1)
        losses['cosine_similarity'] = 1 - cos_sim.mean()

        # InfoNCE-style contrastive loss (optional)
        if teacher_features.size(1) > 1:  # Temporal dimension > 1
            # Positive pairs: same time step
            # Negative pairs: different time steps
            pos_sim = F.cosine_similarity(projected_student, teacher_features, dim=-1)

            # Create negative pairs by shifting
            neg_teacher = torch.roll(teacher_features, shifts=1, dims=1)
            neg_sim = F.cosine_similarity(projected_student, neg_teacher, dim=-1)

            logits = torch.stack([pos_sim, neg_sim], dim=-1) / self.temperature
            targets = torch.zeros(logits.size(0), logits.size(1), dtype=torch.long, device=logits.device)

            losses['contrastive'] = F.cross_entropy(logits.view(-1, 2), targets.view(-1))
        else:
            losses['contrastive'] = torch.tensor(0.0, device=student_features.device)

        return losses


if __name__ == "__main__":
    pass
