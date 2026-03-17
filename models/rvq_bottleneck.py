# rvq_bottleneck.py
import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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

    # Clamp outside range
    x_min = snrs[0]
    x_max = snrs[-1]
    x_clamped = torch.clamp(x, float(x_min), float(x_max))

    # Searchsorted-like interpolation indices
    # idx: index of first snr >= x
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


def _allocate_bits(bits_total: int, num_codebooks: int) -> List[int]:
    """
    把总 bit 分配给多个 codebook，保证 sum(bits_i)=bits_total，
    前面的 codebook 先分到 ceil，再把余数抹平。
    """
    bits_total = int(bits_total)
    if bits_total <= 0 or num_codebooks <= 0:
        return []
    num_codebooks = int(num_codebooks)
    base = bits_total // num_codebooks
    rem = bits_total - base * num_codebooks
    bits = [base] * num_codebooks
    for i in range(rem):
        bits[i] += 1
    # 可能出现 base=0 的情况（bits_total < num_codebooks），允许 0bit 的 stage 直接跳过
    return bits


def _bits01_to_sign(bits01: torch.Tensor) -> torch.Tensor:
    # {0,1} -> {-1,+1}
    return bits01.to(torch.float32) * 2.0 - 1.0


def _sign_to_bits01(bits_sign: torch.Tensor) -> torch.Tensor:
    # {-1,+1} or float -> {0,1}
    return (bits_sign > 0).to(torch.long)


class RVQBottleneck(nn.Module):
    """
    Residual Vector Quantizer (RVQ) bottleneck with:
    - residual quantization: x ≈ sum_i e_i[idx_i]
    - bitstream I/O: indices <-> bits (±1) 以兼容你现有 FSK modem
    - optional bit-level channel simulation (BSC or BPSK+AWGN)
    """

    def __init__(
        self,
        dim: int,
        bits_total: int,
        num_codebooks: int = 2,
        commitment: float = 0.25,
        channel_type: str = "bsc",  # "bsc" or "bpsk_awgn" or "none"
        use_interleaver: bool = False,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.bits_total = int(bits_total)
        self.commitment = float(commitment)
        self.channel_type = str(channel_type)
        # 可选：在比特维度前对时间轴做一次固定交织/反交织，仅对
        # bit-level 信道可见，上下游网络仍看到原始时间顺序。
        self.use_interleaver = bool(use_interleaver)

        self.bits_per_stage = _allocate_bits(self.bits_total, int(num_codebooks))
        # 过滤掉 0bit stage
        self.stage_bits: List[int] = [b for b in self.bits_per_stage if b > 0]

        self.codebooks = nn.ModuleList()
        for b in self.stage_bits:
            size = 2 ** int(b)
            emb = nn.Embedding(size, self.dim)
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
            self.codebooks.append(emb)

        # 懒初始化的时间维交织器（per-T 固定 perm），在第一次看到
        # 给定长度 T 时构造；persistent=False 以避免写入 checkpoint。
        self.register_buffer("_perm_T", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("_inv_perm_T", torch.empty(0, dtype=torch.long), persistent=False)

        # 可选：在初始化时做一次 Gray-aware 码本重排，使相邻 Gray
        # code 对应的 embedding 在欧氏空间尽量接近。通过环境变量
        # AUTO_RVQ_GRAY_ASSIGN 控制，默认关闭以避免改变既有模型。
        if os.environ.get("AUTO_RVQ_GRAY_ASSIGN", "0") == "1":
            try:
                self.optimize_gray_assignment()
            except Exception as _e:
                print(f"[RVQ] WARNING: optimize_gray_assignment failed: {_e}")

    def optimize_gray_assignment(self) -> None:
        """Reorder codebook rows to better align with Gray graph.

        对每个 codebook：
          1) 在 embedding 空间中计算第一主成分方向；
          2) 沿该方向对向量排序（相近向量在排序中相邻）；
          3) 将排序后的向量依次映射到 Gray 序列 g(r)=r^(r>>1) 的
             行号上，使 Gray 邻点对应的 embedding 尽量相近。

        该过程不改变比特接口，仅重排 ``Embedding.weight`` 的行，
        适合作为预训练后的小幅结构性微调，也可在初始化时通过
        环境变量 AUTO_RVQ_GRAY_ASSIGN=1 自动调用。
        """

        for emb in self.codebooks:
            weight = emb.weight.data  # [K,D]
            Kc, D = weight.shape
            if Kc <= 1:
                continue

            with torch.no_grad():
                # 去中心化后做简单的 PCA，取第一主成分作为排序方向
                Wc = weight - weight.mean(dim=0, keepdim=True)
                try:
                    # torch.linalg.svd: Wc = U S Vh，取第一行 Vh[0]
                    _, _, Vh = torch.linalg.svd(Wc, full_matrices=False)
                    direction = Vh[0]
                except Exception:
                    # 回退为随机方向
                    direction = torch.randn(D, device=weight.device, dtype=weight.dtype)
                scores = Wc @ direction  # [K]
                order = torch.argsort(scores)  # 从小到大

                ranks = torch.arange(Kc, device=weight.device, dtype=torch.long)
                gray_labels = ranks ^ (ranks >> 1)
                # 重排：排序后第 r 个向量映射到 Gray label g(r)
                new_weight = weight.clone()
                new_weight[gray_labels] = weight[order]
                emb.weight.detach().copy_(new_weight)

    @property
    def effective_bits(self) -> int:
        return int(sum(self.stage_bits))

    def _quantize_one(self, x: torch.Tensor, emb: nn.Embedding) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B,T,D]
        return:
          - q: [B,T,D] (embedding vectors)
          - idx: [B,T] (long)
        """
        B, T, D = x.shape
        # [B*T, D]
        x_flat = x.reshape(B * T, D)

        # dist = ||x||^2 + ||e||^2 - 2 x e
        e = emb.weight  # [K, D]
        x2 = (x_flat ** 2).sum(dim=1, keepdim=True)       # [B*T,1]
        e2 = (e ** 2).sum(dim=1).unsqueeze(0)             # [1,K]
        xe = x_flat @ e.t()                               # [B*T,K]
        dist = x2 + e2 - 2.0 * xe

        idx = torch.argmin(dist, dim=1)                   # [B*T]
        q = emb(idx).view(B, T, D)                        # [B,T,D]
        idx = idx.view(B, T)
        return q, idx

    def codes_to_bits_sign(self, codes: torch.Tensor) -> torch.Tensor:
        """将 RVQ codebook 索引编码为 Gray code 比特流（±1）。

        Args:
            codes: [B,T,Nq]  每个 stage 一个索引。

        Returns:
            bits_sign: [B,T,sum(bits_i)] in {-1,+1}
        """
        if self.effective_bits == 0:
            B, T = codes.shape[0], codes.shape[1]
            return torch.empty(B, T, 0, device=codes.device, dtype=torch.float32)

        bits_chunks = []
        offset_q = 0
        for b in self.stage_bits:
            idx = codes[..., offset_q].to(torch.long)  # [B,T]
            offset_q += 1
            # 二进制索引 -> Gray code：g = i ^ (i >> 1)
            gray = idx ^ (idx >> 1)
            # 按 MSB -> LSB 顺序拆成比特
            for shift in reversed(range(b)):
                bit = (gray >> shift) & 1
                bits_chunks.append(bit)
        bits01 = torch.stack(bits_chunks, dim=-1)  # [B,T,K]
        return _bits01_to_sign(bits01)

    def bits_sign_to_codes(self, bits_sign: torch.Tensor) -> torch.Tensor:
        """将 Gray code 比特流解码回 RVQ codebook 索引。

        Args:
            bits_sign: [B,T,K] in {-1,+1} (or float)

        Returns:
            codes: [B,T,Nq] long
        """
        B, T, K = bits_sign.shape
        if self.effective_bits == 0:
            return torch.empty(B, T, 0, device=bits_sign.device, dtype=torch.long)

        bits01 = _sign_to_bits01(bits_sign)  # [B,T,K]
        gray_codes: List[torch.Tensor] = []
        off = 0
        for b in self.stage_bits:
            chunk = bits01[..., off : off + b]  # [B,T,b]
            off += b
            # 组装 Gray code：按 MSB->LSB 拼接
            g = torch.zeros(B, T, device=bits_sign.device, dtype=torch.long)
            for i in range(b):
                g = (g << 1) | chunk[..., i]
            gray_codes.append(g)

        codes = []
        for g in gray_codes:
            # Gray -> binary：逐位前缀异或
            b = g.clone()
            shift = 1
            # 根据当前张量的最大值确定所需位数
            max_val = int(b.max().item()) if b.numel() > 0 else 0
            max_bits = max_val.bit_length() if max_val > 0 else 1
            while shift < max_bits:
                b = b ^ (b >> shift)
                shift <<= 1
            codes.append(b)
        return torch.stack(codes, dim=-1)  # [B,T,Nq]

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """
        codes: [B,T,Nq]
        return x_hat: [B,T,D] = sum_i emb_i[codes_i]
        """
        if self.effective_bits == 0:
            raise RuntimeError("RVQ decode_codes called but effective_bits==0")
        x_hat = 0.0
        for i, emb in enumerate(self.codebooks):
            idx = codes[..., i].to(torch.long)
            x_hat = x_hat + emb(idx)
        return x_hat

    def decode_bits(self, bits_sign: torch.Tensor) -> torch.Tensor:
        codes = self.bits_sign_to_codes(bits_sign)
        return self.decode_codes(codes)

    def decode_bits_soft(
        self,
        bits_rx: torch.Tensor,
        channel_params: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        """Soft-decode BPSK+AWGN bits into RVQ embeddings using CSI.

        - bits_rx: [B,T,K] real-valued receive samples (BPSK ±1 + AWGN).
        - channel_params['snr_db']: scalar or tensor SNR in dB.

        For each stage (with b bits, 2^b codewords):
          1) compute per-bit posteriors p(b=1|y, SNR) via a logistic mapping
             using an approximate LLR 2*y/sigma^2, where sigma^2 is inferred
             from SNR;
          2) enumerate all code indices, derive their bit patterns, and form
             a categorical distribution over indices;
          3) take the expectation of the corresponding embeddings under this
             distribution and sum over stages.
        """

        if self.effective_bits == 0:
            raise RuntimeError("RVQ decode_bits_soft called but effective_bits==0")

        if channel_params is None:
            # 没有 CSI 时回退到硬判决解码
            return self.decode_bits(bits_rx)

        B, T, K = bits_rx.shape

        # 提取 SNR（dB）。若为张量，则按元素使用（支持 [B,T] 或 [B,T,1]），
        # 与 HashBottleneck 的 BPSK-AWGN 行为保持一致，而不是取全局均值。
        snr_val = channel_params.get("snr_db", 10.0)
        snr_t = torch.as_tensor(snr_val, device=bits_rx.device, dtype=bits_rx.dtype)
        if snr_t.numel() == 0:
            return self.decode_bits(bits_rx)
        # 对齐维度至 [B,T,(1)]，便于与 bits_rx 广播
        while snr_t.dim() < bits_rx.dim():
            snr_t = snr_t.unsqueeze(-1)

        # BPSK: 噪声方差 sigma^2 = 10^(-SNR/10)
        noise_var = torch.pow(torch.tensor(10.0, device=bits_rx.device, dtype=bits_rx.dtype), -snr_t / 10.0)
        # 近似 LLR 系数：2 / sigma^2
        alpha = 2.0 / noise_var.clamp_min(1e-6)

        # 计算每个 bit 为 1 的后验概率 p1 = sigmoid(alpha * y)
        y = bits_rx.clamp(min=-6.0, max=6.0)
        p1 = torch.sigmoid(alpha * y)  # [B,T,K]
        eps = 1e-6
        p1 = p1.clamp(eps, 1.0 - eps)

        x_hat = 0.0
        off = 0

        for qi, b in enumerate(self.stage_bits):
            b_int = int(b)
            if b_int <= 0:
                continue

            p1_stage = p1[..., off : off + b_int]  # [B,T,b]
            off += b_int

            num_codes = 1 << b_int
            # 生成所有 code index 的 Gray code bit pattern（MSB -> LSB），形状 [num_codes,b]
            # 与 codes_to_bits_sign 中的编码保持一致，避免在软解码时
            # 误将“二进制索引位模式”当作“实际发射的 Gray 比特”。
            idxs = torch.arange(num_codes, device=bits_rx.device, dtype=torch.long)
            gray = idxs ^ (idxs >> 1)
            patterns = []
            for shift in reversed(range(b_int)):
                bit = (gray >> shift) & 1
                patterns.append(bit)
            pattern = torch.stack(patterns, dim=-1).to(p1_stage.dtype)  # [num_codes,b]

            # 计算 log p(code | y)（忽略常数项），再 softmax 得到权重
            # p1_stage: [B,T,b] → [B,T,1,b]
            p1_s = p1_stage.unsqueeze(-2)  # [B,T,1,b]
            pat = pattern.view(1, 1, num_codes, b_int)  # [1,1,num_codes,b]
            log_p = pat * torch.log(p1_s) + (1.0 - pat) * torch.log(1.0 - p1_s)
            log_p = log_p.sum(dim=-1)  # [B,T,num_codes]
            weights = F.softmax(log_p, dim=-1)  # [B,T,num_codes]

            # 期望 embedding：E[e] = sum_j w_j * emb[j]
            emb = self.codebooks[qi].weight[:num_codes]  # [num_codes,D]
            Bt = B * T
            w_flat = weights.view(Bt, num_codes)
            e_flat = torch.matmul(w_flat, emb)  # [Bt,D]
            e_stage = e_flat.view(B, T, self.dim)
            x_hat = x_hat + e_stage

        return x_hat

    def _channel(
        self,
        bits_sign: torch.Tensor,
        channel_params: Optional[Dict[str, float]],
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        debug_jscc = bool(int(os.environ.get("DBG_JSCC", "0")))
        # 推理期默认不注入噪声（与 HashBottleneck 保持一致），但允许
        # 通过 channel_params['force_channel']=True 显式开启以便做 BER 对比。
        force_channel = False
        if channel_params is not None:
            try:
                force_channel = bool(channel_params.get("force_channel", False))
            except Exception:
                force_channel = False

        if ((not self.training) and not force_channel) or self.channel_type == "none" or channel_params is None:
            return bits_sign

        if self.channel_type == "bsc":
            # 优先使用显式传入的 BER；若缺失且提供了 snr_db，则尝试
            # 从 JSCC_FSK_BER_TABLE（bit_only_metrics.csv 拟合结果）中
            # 查表得到 JSCC+FSK 对应的源比特 BER，以便让训练期 BSC
            # 与推理期 2-FSK 链路的 BER–SNR 行为更接近。
            ber_val = channel_params.get("ber", None)
            if ber_val is not None:
                ber_t = torch.as_tensor(ber_val, device=bits_sign.device, dtype=bits_sign.dtype)
            elif "snr_db" in channel_params:
                snr_val = channel_params.get("snr_db", 0.0)
                snr_t = torch.as_tensor(snr_val, device=bits_sign.device, dtype=bits_sign.dtype)
                ber_lookup = _lookup_jscc_fsk_ber_torch(snr_t)
                if ber_lookup is None:
                    return bits_sign
                ber_t = ber_lookup.to(bits_sign.dtype)
            else:
                ber_t = torch.zeros((), device=bits_sign.device, dtype=bits_sign.dtype)
            if ber_t.numel() == 0:
                return bits_sign
            # 若整体 BER<=0，直接退化为无噪声通道
            if ber_t.numel() == 1 and float(ber_t.item()) <= 0.0:
                return bits_sign
            # 对齐维度以便逐 bit 采样翻转掩码，支持 [B,T] / [B,T,1]
            while ber_t.dim() < bits_sign.dim():
                ber_t = ber_t.unsqueeze(-1)
            flip = (torch.rand_like(bits_sign) < ber_t)
            if mask is not None:
                m = (mask > 0.5) if mask.dtype != torch.bool else mask
                flip = flip & m.unsqueeze(-1).expand_as(bits_sign)
            if debug_jscc:
                with torch.no_grad():
                    flip_rate = flip.to(torch.float32).mean().item()
                    try:
                        ber_dbg = float(ber_t.mean().item())
                    except Exception:
                        ber_dbg = 0.0
                    print(f"[JSCC-RVQ] BSC ber≈{ber_dbg:.4f}, flip_rate={flip_rate:.4f}")
            return torch.where(flip, -bits_sign, bits_sign)

        if self.channel_type == "bpsk_awgn":
            snr_val = channel_params.get("snr_db", 10.0)
            snr_t = torch.as_tensor(snr_val, device=bits_sign.device, dtype=bits_sign.dtype)
            if snr_t.numel() == 0:
                return bits_sign
            # 噪声标准差 = 10^(-SNR[dB]/20)，支持 [B,T] / [B,T,1]
            noise_std = torch.pow(
                torch.tensor(10.0, device=bits_sign.device, dtype=bits_sign.dtype),
                -snr_t / 20.0,
            )
            while noise_std.dim() < bits_sign.dim():
                noise_std = noise_std.unsqueeze(-1)
            noise = torch.randn_like(bits_sign) * noise_std
            if mask is not None:
                m = (mask > 0.5) if mask.dtype != torch.bool else mask
                noise = noise * m.unsqueeze(-1).expand_as(bits_sign)
            y = bits_sign + noise
            if debug_jscc:
                with torch.no_grad():
                    try:
                        snr_dbg = float(snr_t.mean().item())
                    except Exception:
                        snr_dbg = 0.0
                    print(f"[JSCC-RVQ] BPSK-AWGN snr_db≈{snr_dbg:.2f}")
            # 返回 soft bits（未做 sign 硬判决）；后续 decode_bits_soft 会利用其幅度和 CSI
            return y

        return bits_sign

    def forward(
        self,
        x: torch.Tensor,  # [B,T,D]
        channel_params: Optional[Dict[str, float]] = None,
        mask: Optional[torch.Tensor] = None,
        use_noisy_bits: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        返回字段尽量对齐 HashBottleneck:
          - bits_clean / bits_noisy: [B,T,K] in {-1,+1}
          - reconstructed: [B,T,D]
          - codes: [B,T,Nq]
          - vq_loss: 标准 VQ-VAE (codebook + commitment)
        """
        if self.effective_bits == 0:
            # 0bit：退化为直通
            B, T, _ = x.shape
            bits_empty = torch.empty(B, T, 0, device=x.device, dtype=torch.float32)
            return {
                "bits_clean": bits_empty,
                "bits_noisy": bits_empty,
                "reconstructed": x,
                "codes": torch.empty(B, T, 0, device=x.device, dtype=torch.long),
                "vq_loss": torch.zeros((), device=x.device),
            }

        residual = x
        q_sum = 0.0
        codes = []

        codebook_loss = 0.0
        commit_loss = 0.0

        for emb in self.codebooks:
            q, idx = self._quantize_one(residual, emb)
            codes.append(idx)
            q_sum = q_sum + q
            residual = residual - q

            # VQ-VAE losses
            codebook_loss = codebook_loss + F.mse_loss(q, residual.detach() + q.detach())  # 近似等价于 mse(q, x.detach())，但更稳
            commit_loss = commit_loss + F.mse_loss(residual + q.detach(), q.detach())     # 近似等价于 mse(x, q.detach())

        codes_t = torch.stack(codes, dim=-1)  # [B,T,Nq]
        bits_clean = self.codes_to_bits_sign(codes_t)  # [B,T,K]

        # 可选：在 bit-level 信道前对时间轴做一次固定交织，仅对
        # 信道“可见”，解码端仍工作在原始时间顺序下。
        bits_for_channel = bits_clean
        mask_for_channel = mask
        perm = None
        inv_perm = None
        if self.use_interleaver and channel_params is not None:
            B, T, _ = bits_clean.shape
            if self._perm_T.numel() != T:
                # 为当前 T 构造一次性 perm / inv_perm
                perm_local = torch.randperm(T, device=bits_clean.device)
                inv_local = torch.empty_like(perm_local)
                inv_local[perm_local] = torch.arange(T, device=bits_clean.device)
                # 保存到 buffer（非持久），便于后续复用
                self._perm_T = perm_local
                self._inv_perm_T = inv_local
            perm = self._perm_T.to(bits_clean.device)
            inv_perm = self._inv_perm_T.to(bits_clean.device)
            bits_for_channel = bits_clean[:, perm, :]
            if mask is not None:
                mask_for_channel = mask[:, perm]

        bits_noisy_ch = self._channel(bits_for_channel, channel_params, mask_for_channel) if use_noisy_bits else bits_for_channel

        if self.use_interleaver and channel_params is not None and inv_perm is not None:
            bits_noisy = bits_noisy_ch[:, inv_perm, :]
        else:
            bits_noisy = bits_noisy_ch

        # noisy bits -> codes_hat / x_hat
        # 对 BPSK+AWGN 使用软解码，其他信道保持硬判决行为。
        if self.channel_type == "bpsk_awgn" and use_noisy_bits:
            x_hat = self.decode_bits_soft(bits_noisy, channel_params)
        else:
            x_hat = self.decode_bits(bits_noisy)

        vq_loss = codebook_loss + self.commitment * commit_loss

        # 额外返回 noisy codes_hat，便于在上层做 SR base / 诊断。
        codes_hat = self.bits_sign_to_codes(bits_noisy)

        return {
            "bits_clean": bits_clean,
            "bits_noisy": bits_noisy,
            "reconstructed": x + (x_hat - x).detach(),
            "reconstructed_hat": x_hat,
            "codes": codes_t,
            "codes_hat": codes_hat,
            "vq_loss": vq_loss,
        }
