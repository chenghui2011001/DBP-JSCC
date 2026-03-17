#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DBP-JSCC training support (dual-branch, Bark/BFCC + vSSM content branch)

目的：
- 在 DualBranchBarkJSCC 上训练：
  - 内容分支：原始音频 → Bark/BFCC 图像 → vSSMJSCCEncoder/Decoder → bark_hat → DCT → ceps_hat。
  - F0/voicing 分支：从 FARGAN 特征中提取 dnn_pitch/frame_corr，走轻量 JSCC（与 Stage2 一致）。
  - 损失：波形 STFT 重建 + ceps/dnn_pitch/frame_corr 特征级重建。

使用方式（示例）：

    python training/train.py \
        --data_root ./data \
        --batch_size 8 --sequence_length 200 --num_epochs 5

"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, asdict, field

from typing import Dict, List, Optional, Tuple
from torch import autograd
import numpy as np

import torch
import torch.nn as nn
import os
import math
import random
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import matplotlib
import csv
import json
import subprocess
import numpy as np
from pathlib import Path

matplotlib.use("Agg")
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from models.dual_branch_bark_jscc import DualBranchBarkJSCC, opus_band_log_smooth
from models.hash_bottleneck import HashBottleneck, GroupedHashBottleneck
from models.rvq_bottleneck import RVQBottleneck
from models.hifi_discriminators import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    feature_loss as hifi_feature_loss,
    generator_loss as hifi_generator_loss,
    discriminator_loss as hifi_discriminator_loss,
)
from utils.channel_sim import ChannelSimulator
from utils.real_data_loader import create_combined_data_loader, create_aether_data_loader
from training.spectral_losses import (
    multi_resolution_stft_loss,
    multi_resolution_sc_loss,
)
from models.vocoder_decoder import FARGANDecoder
from utils.ssim import MS_SSIM
from utils.audio_visualizer import (
    create_batch_comparison_plots,
    save_comparison_audio_samples,
    create_f0_alignment_plot,
    create_ceps_hist_comparison,
)

# Optional: OSCE discriminator for vocoder adversarial training.
# 不在顶层直接 ``import osce``，而是在构建判别器时按需以
# 文件路径方式加载 ``../osce/models/fd_discriminator.py``，并临时
# 调整 ``sys.path``，确保其中的 ``from utils.spec`` 命中 OSCE 仓库
# 自带的 ``utils/spec.py``，而不是当前仓库的 ``utils`` 包。
_OSCE_DIR_GLOBAL: Optional[str]
try:
    _ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
    _OSCE_PARENT = os.path.dirname(_ROOT_DIR)                                # .../torch
    _OSCE_DIR = os.path.join(_OSCE_PARENT, "osce")
    if os.path.isdir(_OSCE_DIR):
        _OSCE_DIR_GLOBAL = _OSCE_DIR
    else:
        print(f"[BFCC-GAN] WARNING: osce directory not found at {_OSCE_DIR}; GAN disabled")
        _OSCE_DIR_GLOBAL = None
except Exception as _osce_e:  # pragma: no cover - defensive
    print(f"[BFCC-GAN] ERROR: unexpected exception while locating osce: {_osce_e}")
    _OSCE_DIR_GLOBAL = None

# Global MS_SSIM instance to avoid repeated creation (stays on GPU)
_ms_ssim_instance: Optional[MS_SSIM] = None


# Simple global cache for SSL models (HuBERT / WavLM / Wav2Vec2 etc.)
_ssl_model_cache_global: Dict[str, torch.nn.Module] = {}

# Optional wandb logging (can be disabled if wandb is not installed)
try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover - wandb is optional
    wandb = None  # type: ignore


def get_ssl_model(model_name: str, device: torch.device) -> Optional[torch.nn.Module]:
    """Load a frozen SSL model (HuBERT/WavLM/Wav2Vec2/AutoModel) with basic caching.

    参考 final_version/models/ssl_utils.py 的映射规则：
    - "hubert-base" / "hubert" → facebook/hubert-base-ls960
    - "hubert-large"           → facebook/hubert-large-ls960-ft
    - "wavlm-base" / "wavlm"  → microsoft/wavlm-base
    - "wavlm-large"            → microsoft/wavlm-large
    - 其它 → 直接交给 AutoModel.from_pretrained(model_name)
    """
    key = f"{model_name}_{str(device)}"
    if key in _ssl_model_cache_global:
        return _ssl_model_cache_global[key]

    try:
        from transformers import (
            HubertModel,
            WavLMModel,
            Wav2Vec2Model,
            AutoModel,
        )  # type: ignore
    except Exception as e:
        try:
            print(f"[SSL] transformers not available, disable SSL loss: {e}")
        except Exception:
            pass
        return None

    name = model_name.lower().replace("_", "-")
    model_id = model_name
    model: Optional[torch.nn.Module]

    try:
        if "hubert" in name:
            if "large" in name:
                model_id = "facebook/hubert-large-ls960-ft"
            else:
                model_id = "facebook/hubert-base-ls960"
            model = HubertModel.from_pretrained(model_id)
        elif "wavlm" in name:
            if "large" in name:
                model_id = "microsoft/wavlm-large"
            elif "base-plus" in name:
                model_id = "microsoft/wavlm-base-plus"
            else:
                model_id = "microsoft/wavlm-base"
            model = WavLMModel.from_pretrained(model_id)
        elif "wav2vec2" in name:
            if "large" in name:
                model_id = "facebook/wav2vec2-large-960h"
            else:
                model_id = "facebook/wav2vec2-base-960h"
            model = Wav2Vec2Model.from_pretrained(model_id)
        else:
            # Fallback: let AutoModel handle arbitrary identifiers
            model = AutoModel.from_pretrained(model_name)
    except Exception as e:
        try:
            print(f"[SSL] Failed to load '{model_name}', disable SSL loss: {e}")
        except Exception:
            pass
        return None

    if model is None:
        return None

    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    _ssl_model_cache_global[key] = model
    try:
        print(f"[SSL] Loaded SSL content model: {model_id}")
    except Exception:
        pass
    return model


@dataclass
class SupportConfig:
    """DBP-JSCC 训练支持配置。"""

    data_root: str = "./data"
    batch_size: int = 8
    sequence_length: int = 200
    num_epochs: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Mixed precision and reproducibility
    use_amp: bool = True  # Enable automatic mixed precision for faster training
    seed: Optional[int] = None  # Random seed for reproducibility (None = no fix)

    snr_min_db: float = -5.0
    snr_max_db: float = 15.0

    lr: float = 1e-4
    # 附加损失权重
    lambda_mel: float = 0.5
    lambda_mel_l1: float = 0.0  # 可选：与 MS-SSIM 并行的 mel L1 稳定项
    lambda_mel_delta: float = 0.0  # mel Δ/ΔΔ 动态约束（边界/节奏保形），默认关闭
    lambda_delta: float = 0.0  # ceps Δ/ΔΔ 约束，默认关闭（可设为0.2启用）
    # 两阶段调度
    use_two_stage: bool = True
    # 延长第一阶段预热，给 hash 与 L2H 更多稳定收敛时间
    stage1_steps: int = 5000

    lambda_wave: float = 1.0   # STFT 谱收敛（SC）
    lambda_wave_mag: float = 0.0  # 可选：MR-STFT 幅度L1（与SC并行）
    lambda_ceps: float = 0.5   # ceps 重建
    lambda_ceps_hi: float = 0.03  # 倒谱高阶监督（如 c12..），抑制有声高频抹平
    # 无声但“非静音”段的高阶倒谱监督（结合 VUV 与静音掩膜），专门保护虚线 F0 区的高频纹理
    lambda_ceps_hi_unv: float = 0.0
    # 特征自一致性（manifold 固定点）：ceps_from(audio_hat) ≈ ceps_hat
    lambda_feature_manifold: float = 0.0
    # 频率感知损失：在 DCT / 倒谱域对频率分量加权，突出感知重要的低频结构
    lambda_freq_aware_mel: float = 0.0
    jpeg_quality_factor: int = 85  # JPEG quality for freq-aware loss weighting (1-100)
    lambda_ceps_weighted: float = 0.0
    # BFCC→ceps 映射损失：使用 GT Bark/BFCC 图像通过 band_agg_32_to_18+mel18_to_ceps
    # 拟合 FARGAN 提供的 ceps 目标，用于在不经过 JSCC 的情况下先校准映射偏差。
    lambda_ceps_map_gt: float = 0.0
    lambda_f0: float = 2.0     # dnn_pitch 重建（增强：0.5 -> 2.0）
    lambda_vuv: float = 2.0    # frame_corr 重建（增强：0.5 -> 2.0）
    lambda_f0_smooth: float = 0.3  # F0二阶平滑损失（惩罚加速度，避免抹平）
    lambda_f0_std: float = 0.0     # F0 标准差匹配损失（Hz 域，voiced 内）
    # F0/VUV SR base 分支：使用仅前 k 维符号解码得到的粗 F0/VUV 作为“骨架”，
    # 通过额外监督鼓励网络将稳定信息压到前 k 维，实现在固定符号预算下的
    # successive refinement（骨架先、细节后）。
    lambda_f0_base: float = 0.0
    lambda_f0_base_smooth: float = 0.0
    lambda_vuv_base: float = 0.0
    # 结构增强/基频对齐（保留核心开关，移除冗余细粒度）
    lambda_harmonic: float = 0.3   # 谐波对齐损失权重（增强：0.1 -> 0.3）
    lambda_f0_slope: float = 0.1   # Δf0 斜率一致性损失权重
    vuv_threshold: float = 0.3    # frame_corr>阈值视为有声
    harmonics_max: int = 5         # 参与对齐的谐波数
    harmonic_bandwidth_hz: float = 30.0  # 谐波带宽（Hz）
    lambda_hash_recon: float = 0.1
    lambda_hash_reg: float = 0.1
    # RVQ VQ 损失权重（仅在 quantizer_type=="rvq" 时生效）
    # 为兼容旧脚本：
    #   - 若 lambda_vq_c/lambda_vq_f 均为 0，则回退为统一 lambda_vq；
    #   - 否则对 content/F0 分支分别使用 lambda_vq_c/lambda_vq_f。
    lambda_vq: float = 0.0
    lambda_vq_c: float = 0.0
    lambda_vq_f: float = 0.0
    # Hash 正则 warmup：训练早期减弱正则，逐步过渡到 lambda_hash_reg
    hash_reg_warmup_steps: int = 0
    hash_reg_start: float = 0.02

    # F0 专用 bit 熵正则：鼓励 F0 hash bits 充分利用 DOF（提高 br_f0_entropy_kbps）
    lambda_f0_entropy: float = 0.0      # 权重（默认关闭）
    f0_entropy_target_frac: float = 0.5 # 目标熵占比：0.5≈期望总熵≈Kf/2 bits per token
    # Content 专用 bit/索引熵正则：鼓励内容分支 RVQ codebook 充分利用 DOF，
    # 在 RVQ 模式下与 rvq_c_H 对齐，在 Hash 模式下退化为 bit 熵下界。
    lambda_c_entropy: float = 0.0
    content_entropy_target_frac: float = 0.5
    # Content bit 平衡正则：鼓励每一比特的 P(bit=1) 接近 0.5，避免比特长期饱和。
    lambda_bit_balance_c: float = 0.0

    # SSL 语音内容一致性（HuBERT / Wav2Vec2 / WavLM 等，自监督 content“保险”）
    lambda_ssl: float = 0.0  # SSL content loss 权重（默认关闭）
    ssl_model_name: Optional[str] = None  # 例如 "facebook/hubert-base-ls960"；需用户显式提供
    ssl_layers: Optional[List[int]] = None  # 使用的中间层索引列表（None 时自动选择中高层）
    ssl_warmup_steps: int = 0  # SSL loss warmup 步数（0=不用渐入）

    # F0 envelope/wavelet (crepe-guided, optional)
    lambda_f0_env: float = 0.0        # hinge-to-envelope in cents
    f0_env_margin_cents: float = 80.0 # base margin (cents)
    f0_env_alpha: float = 0.5         # extra margin factor wrt (crepe vs fallback) disagreement
    f0_env_window: int = 3            # smoothing window (frames, odd); smaller to avoid over-smoothing
    f0_env_mode: str = "band"         # 'band' (lo/hi from crepe&fallback) or 'center' (center+margin)
    f0_env_k_sigma: float = 2.0       # gaussian band gate: keep frames within k*std around local mean (core only)
    # (presence and vuv_crepe removed)
    lambda_f0_tv: float = 0.0         # small second-order TV on cent_pred within core

    # wandb logging（可选）
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_log_freq: int = 10  # 每多少步记录一次 loss_dict
    f0_tv_delta_cents: float = 40.0   # robust TV threshold (cents) for normalized hinge
    # wavelet differential constraint (suppress jitter without flattening)
    lambda_f0_wavelet: float = 0.0
    f0_wav_levels: int = 3
    f0_wav_alphas: Optional[List[float]] = None
    # clipping threshold (in cents) applied to the wavelet error before normalization
    f0_wav_clip_cents: float = 120.0
    # tiny bias penalty to discourage constant offset of F0 relative to reference (in cents)
    lambda_f0_bias: float = 0.0
    f0_estimator: str = "auto"        # 'crepe'|'pyin'|'auto' (for loss-only extraction)
    f0_estimator_model: str = "tiny"  # torchcrepe model: 'tiny' or 'full'

    # F0 decoder content-attention schedule：
    # 通过 cross-attention 让 F0 解码看到 Bark/BFCC 上下文。
    # - f0_cond_attn_warmup_steps: 在 warmup 步数内线性从 0→f0_cond_attn_max_alpha；
    # - f0_cond_attn_max_alpha: cross-attn 的最大缩放因子（与解码器内部门控相乘）。
    f0_cond_attn_warmup_steps: int = 0
    f0_cond_attn_max_alpha: float = 1.0

    # JSCC+FSK offline evaluation（可选）：
    # 若 jscc_fsk_eval=True，则在 save_checkpoint 之后自动调用外部
    # pcm_segment_infer_jscc_fsk.py，对固定 PCM 片段做 JSCC+FSK 仿真，并
    # 将 STOI/PESQ/F0/Bark 等指标追加到 CSV。该评估在训练进程内同步执行，
    # 会略微增加保存 checkpoint 时的耗时。
    jscc_fsk_eval: bool = False
    jscc_fsk_pcm_path: Optional[str] = None
    jscc_fsk_output_root: Optional[str] = None
    jscc_fsk_pcm_infer_script: Optional[str] = None
    jscc_fsk_noise_csv: Optional[str] = None
    jscc_fsk_sample_rate: int = 16000
    jscc_fsk_pcm_dtype: str = "int16"
    jscc_fsk_segment_sec: float = 4.0
    jscc_fsk_num_segments: int = 50
    jscc_fsk_seed: int = 123
    jscc_fsk_snr_db: float = 3.0
    jscc_fsk_metrics_csv: Optional[str] = None

    # Eval/diagnostic controls
    # bit_only_eval: 是否在可视化阶段额外跑一次 “纯 bits→音频” 解码，
    # 使用 encode_hash_codec + decode_from_bits_offline 对同一 batch 的前几条样本
    # 做 bit-only 比较图，帮助观察训练前向与真正部署路径之间的差异。
    bit_only_eval: bool = False
    bit_only_eval_max_samples: int = 2

    # disable_ceps_c0_calib: 若为 True，则关闭训练期基于 GT ceps 的 c0 校准，
    # 使训练/推理均只依赖 bits_stats 提供的 mean/std 信息进行能量对齐。
    disable_ceps_c0_calib: bool = False

    # VMamba/Hash/RVQ 控制
    with_hash: bool = False
    # 量化器类型：
    # - "hash": 使用 HashBottleneck/TwoStageHashBottleneck（默认）
    # - "rvq" : 使用 RVQBottleneck（Residual VQ）
    quantizer_type: str = "hash"
    # Content-only mode: skip F0/VUV branch, ceps, L2H, and vocoder; train only Bark/BFCC reconstruction
    content_only: bool = False
    # F0-only warmup: freeze content/JSCC path, train only F0/VUV branch + vocoder
    f0_only: bool = False
    # Freeze BFCC content JSCC (wave_to_mel + content_vmamba + hash_content)，
    # 保持 mel18_to_ceps / L2H / vocoder 等其余模块可训练。
    freeze_content_jscc: bool = False
    # 可选：使用 BFCC CNN JSCC baseline 替代 VMamba 内容分支（仅在 content_only 模式下使用）
    content_cnn_baseline: bool = False
    content_cnn_latent_channels: int = 1
    hash_bits_content: int = 16
    hash_bits_f0: Optional[int] = None
    # RVQ 配置（在 quantizer_type == "rvq" 时使用）
    rvq_nq_content: int = 2
    rvq_nq_f0: Optional[int] = None
    rvq_beta: float = 0.25
    vm_channels: Optional[List[int]] = None
    vm_depths: Optional[List[int]] = None
    # VMamba 内容分支的 CSI/SNR 融合方式
    # vm_channel_adaptive: 控制定点 SNR 自适应模式：
    #   - 'no'  : 不使用 SNR；仅依赖 CSI（csi_vec）
    #   - 'ca'  : 通道级偏置式 SNR 调制（per-stage bias），对齐原版 MambaJSCC 的 CA 模式
    #   - 'ssm' : 轻量版 SSM 内核级别 SNR 调制（通过 SelectiveScan2D 内部缩放 A），不再额外加通道偏置
    # vm_lightweight_config: 控制 VMambaBlock2D 内部 CSI 门控的轻量策略
    #   - 'all_native'     : 全部使用原生 Mamba 风格 CSI 线性偏置
    #   - 'progressive'    : 编码端 native→hybrid→lightweight 渐进，解码端反向
    #   - 'all_lightweight': 全部使用 LW-JSCC 风格 LightweightCSIGate
    vm_channel_adaptive: str = "no"
    vm_lightweight_config: str = "all_native"
    # 放松频率下采样：仅在前 1 个 stage 上做 (2,2) 下采样，
    # 之后的 stage 仅在时间维下采样 (2,1)，保留更多 Bark 频带 DOF。
    freq_downsample_stages: int = 1
    content_time_downsample: int = 1
    # 内容符号维度从 4 提升到 8，以增加 JSCC DOF。
    d_s_content: int = 8
    # HF 侧通道 → 倒谱高阶校正维度与缩放
    hf2ceps_dim: int = 8
    hf2ceps_scale: float = 0.5
    eq_fading: bool = False
    # 可视化与音频导出
    viz_dir: str = "./outputs/visualizations"
    viz_every_steps: int = 1000
    viz_max_samples: int = 2
    ckpt_dir: str = "./outputs/checkpoints"
    save_every_steps: int = 500
    resume: Optional[str] = None
    # Vocoder / FARGAN
    vocoder_ckpt: Optional[str] = None
    vocoder_eval_every_steps: int = 0  # >0 to print vocoder-only baseline STFT
    freeze_vocoder_all: bool = False   # if True, never unfreeze vocoder during training
    # 完全冻结编解码（VMamba/JSCC/F0分支等）只训练 HashBottleneck
    freeze_codec_all: bool = False
    # OSCE FD-based adversarial training（BFCC-GAN 风格，作用于 audio_hat）
    bfcc_gan: bool = False
    bfcc_gan_lambda: float = 1.0        # GAN generator loss 权重
    bfcc_gan_fmap_weight: float = 1.0   # feature-matching loss 权重
    # Anti-flatten knobs (FARGAN-style)
    stft_preset: str = "aether"       # 'aether' or 'fargan'
    lambda_fargan_sc: float = 0.0      # add original-style multi-res spectral convergence
    lambda_fargan_signal: float = 0.0  # normalized frame signal cosine
    lambda_continuity: float = 0.0     # boundary continuity loss
    lambda_pitch_consistency: float = 0.0  # pitch consistency w.r.t. target period
    lambda_subframe_align: float = 0.0     # subframe alignment loss
    # Mel 亮度/能量锚点（避免 mel → ceps 亮度漂移）
    lambda_mel_energy: float = 0.0
    # 逐帧亮度曲线与对比度/分频带亮度锚点
    lambda_mel_energy_t: float = 0.0
    lambda_mel_contrast: float = 0.0
    lambda_mel_bandE: float = 0.0
    # BFCC（32-Bark log 能量图）图像域能量/纹理约束
    # 仿照 training/bfcc_jscc_cnn_baseline.py 中的 lambda_energy_t / lambda_energy_f / lambda_tex_t，
    # 但默认关闭，由 CLI 显式开启以保持向后兼容。
    lambda_energy_t: float = 0.0   # per-frame mean BFCC energy |E_hat(t)-E(t)|
    lambda_energy_f: float = 0.0   # per-band mean BFCC energy |E_hat(f)-E(f)|
    lambda_tex_t: float = 0.0      # BFCC time-axis gradient consistency
    # Mel/BFCC 损失裁边：在计算 mel/BFCC 相关损失前，
    # 可选择在时间维裁掉首尾若干帧、在 Bark 频带维裁掉上下若干 band，
    # 避免卷积/反卷积在边缘感受野不足时被强行拟合。
    mel_loss_crop_time: int = 0    # 每侧裁掉的时间帧数（单位：BFCC 帧）
    mel_loss_crop_freq: int = 0    # 每侧裁掉的频带数（单位：Bark band）
    # 频向谷值钳制：约束有声帧内的“谷”不能被大幅抬高
    lambda_mel_valley: float = 0.0
    # F0 峰性（防抹平）：在 hash-only 场景下默认关闭，
    # 需要时由 CLI 显式开启。
    lambda_f0_peak: float = 0.0
    # Train-only-hash mode（移除 CLI；保留兼容性默认行为）

    # 旧的 mel 高频先验与细化已移除

    # 自适应高频加权（仅用最后一层锚点的梯度尺度匹配）
    adaptive_hf: bool = False
    adaptive_every: int = 20            # 每多少步更新一次权重
    adaptive_alpha: float = 0.5         # 比例幂（0.25~0.75）
    adaptive_beta: float = 0.1          # EMA 平滑系数
    # 梯度范围调查（仅最后一层锚点）
    grad_survey: bool = False
    # —— 替代性、简洁的高频纹理约束 ——
    # 1) 频率加权 STFT（高频段权重大）——在 hash-only 场景中默认关闭，
    # 避免与其它 HF 项叠加导致过强抹平。
    lambda_hf_stft: float = 0.0
    hf_start_hz: int = 4000
    hf_power: float = 2.0
    # 2) 倒谱高阶直接监督（如 c10..c17），默认适度开启
    lambda_ceps_hi: float = 0.03
    ceps_hi_start: int = 10
    # 3) Mel 频向二阶差分（拉普拉斯）匹配，仅高频，默认关闭
    lambda_mel_hp: float = 0.0
    mel_hp_low_bins: int = 16

    # 高频时间边缘约束（仅高频 Mel 的时间差分），默认关闭
    lambda_hf_time_edge: float = 0.0
    hf_time_edge_start: int = 32
    hf_time_edge_ref_thr: float = 0.03
    hf_time_edge_boundary_boost: float = 2.0
    hf_time_edge_weight_clip: float = 5.0

    # 边界高频倾斜约束：对齐边界/无声帧的高低频能量差（频谱“抬高”边界虚线）
    lambda_hf_tilt: float = 0.0
    hf_tilt_split_bin: int = 16
    # 额外“硬推高”项：在边界/无声帧上，要求高频能量至少比低频高出 margin（log-mel 单位），
    # 例如 0.5≈5dB、1.0≈10dB，用于在可接受范围内主动把边界摩擦拉向高频，即便 GT 较保守。
    hf_tilt_extra_push: float = 0.0

    # Teacher-forcing anneal on F0/VUV (period + frame_corr gate)
    tf_start_step: Optional[int] = None
    tf_end_step: Optional[int] = None

    # 可选：低→高 Mel 细化与频带先验
    with_l2h: bool = False
    # L2H 作为“高频补丁”：学习 GT 与 baseline mel_hat 之间的 envelope 残差，
    # 而非对高频绝对能量做硬重建，避免与 valley/silence 等能量项过度拉扯。
    lambda_l2h: float = 0.05
    # 直接在 L2H 高频输出上做 L1 的硬重建项（默认关闭，仅作旧实验兼容）。
    lambda_l2h_direct: float = 0.0
    l2h_low_bins: int = 10
    lambda_l2h_resid: float = 0.0
    lambda_l2h_decor: float = 0.0
    # L2H improvement margin：在高频残差监督中鼓励 L2H 相对 baseline
    # 有所“改进”（improvement）。当 margin>0 时，如果 L2H refined
    # 高频误差未能比 baseline 更小，则产生惩罚；margin=0 时仅保证
    # 不比 baseline 更差。
    l2h_improve_margin: float = 0.0
    # L2H 穿透损失：让 L2H 优化目标对齐到最终进入 vocoder 的 18-band/ceps 表征
    # 避免 32->18 聚合后细化效果被稀释
    lambda_l2h_band18: float = 0.0   # 在 18 维能量域计算 L2H 穿透损失
    lambda_l2h_ceps: float = 0.0     # 在 18 维倒谱域计算 L2H 穿透损失
    l2h_band18_hi_start: int = 8     # 只对 18-band 的高频部分 (bin 8+) 计算穿透损失
    # 旧版频带先验（实现保留，默认关闭；不再提供 CLI 注入）
    # 高频纹理梯度一致性（亮区、带 V/UV 门控）
    lambda_mel_texture: float = 0.0
    # 可选：冻结 32→18 频带聚合以避免训练期偏置向低频塌陷
    freeze_band_agg: bool = False
    # 能量校准强度（0~1），用于 mel/ceps c0 的均值对齐
    energy_calib_alpha: float = 0.8
    # L2H 渐进融合：预热与过渡步数
    l2h_warmup_steps: int = 400
    l2h_blend_steps: int = 800
    # L2H 调度模式：
    # - "abs":  按绝对 global_step 进行 warmup/blend（多次 resume 时连续）；
    # - "resume_rel": 按本次 resume 后的相对步数重新 warmup。
    l2h_schedule_mode: str = "abs"

    # DeCo 风格 L2H：使用 AdaLN 条件生成高频，不再对基线 mel 高频做显式残差补全
    deco_l2h: bool = False
    deco_l2h_hidden: int = 64
    deco_l2h_blocks: int = 3

    # 条件 flow 高频建模（仅用于 GT mel 高频 NLL 正则）
    use_l2h_flow: bool = False
    l2h_flow_hidden: int = 128
    l2h_flow_n_flows: int = 4
    lambda_l2h_flow_nll: float = 0.0

    # HF 侧通道：将高频残差特征直接传给 FARGAN，绕过 32->18 聚合的信息损失
    with_hf_sideband: bool = False
    hf_sideband_dim: int = 6       # 侧通道维度 (4-8 维通常足够)
    hf_sideband_type: str = "learnable"  # "learnable" | "dct" | "linear"

    # (adversarial regularization removed)

    # ---- High-Frequency Texture Protection Loss (replaces PHC) ----
    lambda_texture_protect: float = 0.0    # weight for texture protection loss
    texture_hf_start: int = 40              # starting mel bin for high-freq region (≈4kHz)
    texture_grad_weight: float = 0.5        # weight for gradient term
    texture_var_weight: float = 0.3         # weight for variance term
    texture_contrast_weight: float = 0.4    # weight for frequency contrast term
    texture_eps: float = 1e-4               # numerical epsilon

    # ---- F0 Pattern Preservation Loss ----
    lambda_f0_pattern: float = 0.0          # weight for F0 pattern preservation loss
    f0_pattern_synergy_weight: float = 0.3  # weight for F0-texture synergy component
    # Encourage solid-line continuity inside voiced segments (center 10%-90%)
    lambda_f0_center: float = 0.0

    # ---- HF distillation from frozen teacher (optional) ----
    lambda_teacher_hf: float = 0.0          # weight for teacher HF distillation (uses GT features vocoder path)
    teacher_hf_norm: str = "bft_mean"      # fixed safe default; no CLI needed
    teacher_hf_log: bool = True             # default: log1p magnitude before diff (stabilizes scale)
    teacher_hf_gain: float = 1.0            # internal gain (independent from lambda)
    teacher_hf_auto_balance: bool = True    # default: auto-scale distill RMS-Grad on audio_hat
    teacher_hf_gn_target: float = 5e-3      # target RMS-Grad on audio_hat for distill term
    teacher_hf_scale_min: float = 0.1       # clamp for auto-balance scale
    teacher_hf_scale_max: float = 10.0      # clamp for auto-balance scale
    # 是否在 resume 之后重新从 fargan_ckpt 覆盖学生 vocoder 核心权重。
    # 默认 False：resume 优先使用 checkpoint 中的学生声码器权重，仅在构建 teacher 时使用 fargan_ckpt。
    reload_vocoder_after_resume: bool = False

    # ---- HiFi-GAN style adversarial loss (raw waveform MPD + MSD) ----
    # 通过 Multi-Period Discriminator + Multi-Scale Discriminator 显式建模多周期/多尺度结构，
    # 并配合 feature matching 稳定训练，提升“咬字结构”和周期性纹理。
    lambda_hifi_adv: float = 0.0           # generator adversarial loss weight (MPD+MSD, LSGAN)
    lambda_hifi_fm: float = 0.0            # feature-matching loss weight from MPD/MSD feature maps
    hifi_adv_warmup_steps: int = 0         # steps before enabling MPD/MSD updates
    hifi_disc_lr: float = 1e-4             # learning rate for MPD/MSD discriminators
    # 为减轻内存开销，仅在 HiFi 判别器上使用较短的音频片段做对抗
    hifi_adv_crop_len: int = 16000         # crop length in samples for MPD/MSD (0=use full)

    # ---- HF adversarial regularization (4–8kHz, audio STFT 判别器) ----
    # 仅在 Stage2.5 中使用 MultiScaleSpecDisc，对 4–8kHz 高频区域做 LSGAN + feature matching，
    # 主要通过 L2H 分支生成的高频残差来“长纹理”。
    lambda_hf_adv: float = 0.01            # adversarial term weight (G-side)
    lambda_hf_fm: float = 0.02             # feature-matching term weight (G-side)
    hf_adv_warmup_steps: int = 10000       # steps before enabling HF discriminator
    hf_adv_disc_lr: float = 1e-4           # discriminator learning rate
    hf_adv_roi_low_hz: int = 4000          # HF ROI lower bound (Hz)
    hf_adv_roi_high_hz: int = 8000         # HF ROI upper bound (Hz)

    # ---- Mel-domain HF adversarial regularization (high mel bins) ----
    lambda_hf_mel_adv: float = 0.0          # adversarial term weight on mel HF (G-side)
    lambda_hf_mel_fm: float = 0.0           # feature-matching weight on mel HF (G-side)
    hf_mel_adv_warmup_steps: int = 5000     # warmup before enabling mel HF discriminator
    hf_mel_low_bins: int = 10               # start bin for mel HF region

    # ---- F0 Presence Loss (强制网络预测有声段) ----
    lambda_f0_presence: float = 0.0         # weight for F0 presence loss (forces voiced prediction)
    f0_presence_gamma: float = 2.0          # focal loss gamma for presence loss

    # Legacy PHC (kept for backward compatibility)
    lambda_pitch_harm: float = 0.0          # DEPRECATED: use lambda_texture_protect

    # ---- Silence shaping（静音能量抑制与高频门控）----
    # 1) 波形域静音能量约束：仅在目标音频静音帧上，惩罚 audio_hat 的帧能量
    lambda_silence_wave: float = 0.0
    # 2) Mel 域静音高频约束（可选，与波形互斥或并行使用）
    lambda_silence_mel: float = 0.0
    # bit-only 路径静音约束：在纯 bits→decode_from_bits_offline 路径上
    # 对静音段波形 RMS 做一个轻量惩罚，鼓励部署路径上的静音更干净。
    lambda_bit_only_silence: float = 0.0
    # bit-only distillation：使用 decode_hash_codec/forward_with_hash 作为 teacher，
    # 约束 decode_from_bits_offline 输出靠近 teacher（只依赖比特，不泄漏 GT）。
    # 典型用法：对 teacher 与 bit-only 音频做轻量 MR-STFT 谱收敛损失。
    lambda_bit_only_distill: float = 0.0
    # 3) 静音判定阈值（对 mel 高频与帧 RMS 使用略宽松的 dB 门限，默认 -35 dB）
    silence_energy_thr_db: float = -35.0
    silence_rms_thr_db: float = -35.0
    # 4) 非静音掩膜时间膨胀半径（帧数，用于保护有声/无声边界不被当作静音抹平）
    silence_dilate_frames: int = 0
    # 5) 高频能量用于静音判定的起始 mel bin（默认与 mel_hp_low_bins 对齐）
    silence_hf_low_bins: int = 16
    # 6) 可选：在静音判定中加入高频方差门控，仅在“能量低且高频几乎无纹理”时视为静音
    silence_use_hf_var: bool = False
    silence_hf_var_thr: float = 0.02

    # ---- VUV extra losses（静音帧约束 + 全局占比约束）----
    # 静音帧抑制项：惩罚静音帧上过高的 frame_corr_hat（增强：0.0 -> 0.5）
    lambda_vuv_sil: float = 0.5
    # 全局 voiced 占比对齐：强制 hat 的有声比例接近 GT（增强：0.0 -> 0.2）
    lambda_vuv_ratio: float = 0.2
    # VUV 概率 BCE：防止 frame_corr_hat 全偏到一边（增强：0.0 -> 0.15）
    lambda_vuv_bce: float = 0.15
    # 若为 True，则冻结除 HashBottleneck/GroupedHashBottleneck 编解码器之外的所有模块，
    # 仅训练哈希瓶颈（内容分支 + F0 分支）。
    freeze_codec_all: bool = False
    


def build_dataloader(cfg: SupportConfig):
    """Auto-detect dataset layout and build the proper DataLoader.

    - If expert-mixed files exist under `data_root` (e.g., harmonic_200k_36.f32
      or *_enhanced_36.f32), use `create_combined_data_loader`.
    - Otherwise, fall back to AETHER single-set loader `create_aether_data_loader`.
    """
    from pathlib import Path

    root = Path(cfg.data_root)

    def has_combined_expert_files(r: Path) -> bool:
        # small-200k layout
        small = [
            r / 'harmonic_200k_36.f32', r / 'transient_200k_36.f32',
            r / 'burst_inpaint_200k_36.f32', r / 'low_snr_200k_36.f32',
        ]
        # enhanced fallback layout
        enh = [
            r / 'harmonic_enhanced_36.f32', r / 'transient_enhanced_36.f32',
            r / 'burst_inpaint_enhanced_36.f32', r / 'low_snr_enhanced_36.f32',
        ]
        return any(p.exists() for p in small + enh)

    def has_aether_layout(r: Path) -> bool:
        return (r / 'lmr_export' / 'features_36_fargan_baseline.f32').exists() \
            or (r / 'lmr_export' / 'features_48_complete.f32').exists() \
            or (r / 'out_speech.pcm').exists()

    if has_combined_expert_files(root):
        print('[DataDetect] Detected expert-mixed dataset layout → using create_combined_data_loader')
        dataloader, _dataset = create_combined_data_loader(
            data_root=cfg.data_root,
            sequence_length=cfg.sequence_length,
            batch_size=cfg.batch_size,
            max_samples=None,
            num_workers=4,
            energy_selection=True,
            feature_dims=36,
        )
        return dataloader

    # Fall back to single AETHER-style dataset
    if has_aether_layout(root):
        print('[DataDetect] Detected AETHER/Single-set layout → using create_aether_data_loader')
    else:
        print('[DataDetect] No expert-mixed markers found; using single-set loader by default')

    dataloader, _dataset = create_aether_data_loader(
        data_dir=cfg.data_root,
        sequence_length=cfg.sequence_length,
        batch_size=cfg.batch_size,
        max_samples=None,
        num_workers=4,
        energy_selection=True,
        feature_spec_type='fargan',  # Stage2.5 consumes 36‑dim FARGAN features
    )
    return dataloader


def build_model(cfg: SupportConfig) -> DualBranchBarkJSCC:
    device = torch.device(cfg.device)
    vm_channels = cfg.vm_channels
    vm_depths = cfg.vm_depths
    model = DualBranchBarkJSCC(
        d_csi=4,
        d_zc=32,
        d_s_content=cfg.d_s_content,
        d_zf=16,
        d_s_f0=16,
        hidden_f0=32,
        sample_rate=16000,
        n_fft=400,
        hop_length=160,
        n_mels=32,
        with_hash=cfg.with_hash,
        hash_bits_content=cfg.hash_bits_content,
        hash_bits_f0=cfg.hash_bits_f0,
        quantizer_type=str(getattr(cfg, 'quantizer_type', 'hash')),
        rvq_nq_content=int(getattr(cfg, 'rvq_nq_content', 2)),
        rvq_nq_f0=getattr(cfg, 'rvq_nq_f0', None),
        rvq_beta=float(getattr(cfg, 'rvq_beta', 0.25)),
        content_time_downsample=int(getattr(cfg, 'content_time_downsample', 1)),
        vm_channel_adaptive=str(getattr(cfg, 'vm_channel_adaptive', 'no')),
        vm_lightweight_config=str(getattr(cfg, 'vm_lightweight_config', 'all_native')),
        vm_channels=vm_channels,
        vm_depths=vm_depths,
        freq_downsample_stages=cfg.freq_downsample_stages,
        eq_fading=cfg.eq_fading,
        device=device,
        with_l2h=bool(getattr(cfg, 'with_l2h', False)),
        l2h_low_bins=int(getattr(cfg, 'l2h_low_bins', 10)),
        use_l2h_flow=bool(getattr(cfg, 'use_l2h_flow', False)),
        l2h_flow_hidden=int(getattr(cfg, 'l2h_flow_hidden', 128)),
        l2h_flow_n_flows=int(getattr(cfg, 'l2h_flow_n_flows', 4)),
        deco_l2h=bool(getattr(cfg, 'deco_l2h', False)),
        deco_l2h_hidden=int(getattr(cfg, 'deco_l2h_hidden', 64)),
        deco_l2h_blocks=int(getattr(cfg, 'deco_l2h_blocks', 3)),
        # HF 侧通道
        with_hf_sideband=bool(getattr(cfg, 'with_hf_sideband', False)),
        hf_sideband_dim=int(getattr(cfg, 'hf_sideband_dim', 6)),
        hf_sideband_type=str(getattr(cfg, 'hf_sideband_type', 'learnable')),
        hf2ceps_dim=int(getattr(cfg, 'hf2ceps_dim', 8)),
        hf2ceps_scale=float(getattr(cfg, 'hf2ceps_scale', 0.5)),
        # CNN BFCC JSCC baseline（内容分支替代 VMamba，用于对比实验）
        content_cnn_baseline=bool(getattr(cfg, 'content_cnn_baseline', False)),
        content_cnn_latent_channels=int(getattr(cfg, 'content_cnn_latent_channels', 1)),
        # BFCC 声码器调试路径：仅在 DBG_BFCC_VOCODER=1 或
        # cfg.use_bfcc_vocoder_debug=True 时，额外生成
        # out['audio_hat_bfcc']，不影响现有 FARGAN 训练流程。
        use_bfcc_vocoder_debug=bool(os.environ.get("DBG_BFCC_VOCODER", "0") == "1"),
    )

    # 显式打印量化器配置，便于确认是否使用 RVQ
    try:
        qtype = str(getattr(cfg, 'quantizer_type', 'hash'))
        bits_content = int(getattr(cfg, 'hash_bits_content', getattr(model, 'hash_bits_content', 0)))
        # 与 DualBranchBarkJSCC 内部逻辑保持一致：若未显式指定 hash_bits_f0，
        # 则回退为 max(4, hash_bits_content//2)。
        if getattr(cfg, 'hash_bits_f0', None) is not None:
            bits_f0 = int(getattr(cfg, 'hash_bits_f0'))
        else:
            bits_f0 = max(4, bits_content // 2) if bits_content > 0 else 4

        if qtype == 'rvq':
            nq_content = int(getattr(cfg, 'rvq_nq_content', getattr(model, 'rvq_nq_content', 2)))
            nq_f0 = getattr(cfg, 'rvq_nq_f0', None)
            if nq_f0 is None:
                nq_f0 = nq_content
            print(
                f"[Build] quantizer_type=rvq (content bits={bits_content}, "
                f"nq_content={nq_content}, f0 bits={bits_f0}, nq_f0={int(nq_f0)})"
            )
        else:
            # Hash 模式下也给出基本 bit 配置，便于对比实验
            print(
                f"[Build] quantizer_type=hash (content bits={bits_content}, f0 bits={bits_f0})"
            )
    except Exception:
        # 打印失败不应影响构建流程
        pass

    # 可选：冻结 32→18 频带聚合，避免训练期权重向低频偏置
    try:
        if bool(getattr(cfg, 'freeze_band_agg', False)) and hasattr(model, 'band_agg_32_to_18'):
            for p in model.band_agg_32_to_18.parameters():
                p.requires_grad = False
            print('[Build] Froze band_agg_32_to_18 parameters')
    except Exception:
        pass
    # Optionally load pretrained FARGAN vocoder weights into decoder
    if cfg.vocoder_ckpt:
        try:
            if os.path.isfile(cfg.vocoder_ckpt):
                print(f"[Vocoder] Loading vocoder checkpoint: {cfg.vocoder_ckpt}")
                try:
                    ck = torch.load(cfg.vocoder_ckpt, map_location='cpu', weights_only=True)
                except TypeError:
                    ck = torch.load(cfg.vocoder_ckpt, map_location='cpu')
                sd = ck['state_dict'] if isinstance(ck, dict) and 'state_dict' in ck else ck
                load_ret = model.vocoder.fargan_core.load_state_dict(sd, strict=False)
                missing = list(getattr(load_ret, 'missing_keys', []))
                unexpected = list(getattr(load_ret, 'unexpected_keys', []))
                print(f"[Vocoder] FARGAN weights loaded (strict=False). missing={len(missing)}, unexpected={len(unexpected)}")
            else:
                print(f"[Vocoder] WARNING: vocoder_ckpt not found: {cfg.vocoder_ckpt}")
        except Exception as e:
            print(f"[Vocoder] WARNING: failed to load FARGAN ckpt: {e}")
    # 同步能量校准强度（若模型支持该属性）
    try:
        if hasattr(model, 'energy_calib_alpha'):
            model.energy_calib_alpha = float(getattr(cfg, 'energy_calib_alpha', getattr(model, 'energy_calib_alpha', 0.8)))
    except Exception:
        pass

    # 将静音能量阈值同步到模型（若支持该属性），用于 vocoder 覆盖掩膜中的统一能量门控
    try:
        if hasattr(model, 'silence_energy_thr_db'):
            model.silence_energy_thr_db = float(getattr(cfg, 'silence_energy_thr_db', getattr(model, 'silence_energy_thr_db', -40.0)))
    except Exception:
        pass

    # 可选：完全冻结编解码权重，仅训练 Hash/RVQ Bottleneck
    if bool(getattr(cfg, 'freeze_codec_all', False)):
        total_params = 0
        for p in model.parameters():
            p.requires_grad = False
            total_params += p.numel()

        trainable = 0
        for name, module in model.named_modules():
            if isinstance(module, (HashBottleneck, GroupedHashBottleneck, RVQBottleneck)):
                for p in module.parameters(recurse=True):
                    if not p.requires_grad:
                        p.requires_grad = True
                        trainable += p.numel()

        frozen = total_params - trainable
        print(f"[Build] Freeze codec (only hash bottlenecks trainable): frozen={frozen} trainable={trainable}")

    # 可选：只冻结 BFCC 内容 JSCC 路径（wave_to_mel + content_vmamba + hash_content[_stats]），
    # 保留 mel18_to_ceps / L2H / vocoder / F0 分支等模块可训练。
    if bool(getattr(cfg, 'freeze_content_jscc', False)) and not bool(getattr(cfg, 'freeze_codec_all', False)):
        print("[Build] Freeze BFCC content JSCC (wave_to_mel + content_vmamba + hash_content[_stats])")

        def _freeze_module(mod: nn.Module | None) -> None:
            if mod is None:
                return
            for p in mod.parameters():
                p.requires_grad = False

        for name in [
            'wave_to_mel',
            'content_vmamba',
            'hash_content',
            'hash_content_stats',
            'content_cnn_encoder',
            'content_cnn_decoder',
        ]:
            if hasattr(model, name):
                _freeze_module(getattr(model, name))

    # 可选：F0-only warmup —— 冻结内容/JSCC 路径，仅训练 F0/VUV 分支 + vocoder
    if bool(getattr(cfg, 'f0_only', False)):
        print("[Build] F0-only warmup: freezing content/JSCC, training F0/VUV branch + vocoder")

        def _freeze_module(mod: nn.Module | None) -> None:
            if mod is None:
                return
            for p in mod.parameters():
                p.requires_grad = False

        def _unfreeze_module(mod: nn.Module | None) -> None:
            if mod is None:
                return
            for p in mod.parameters():
                p.requires_grad = True

        # 冻结内容/JSCC 相关模块
        for name in [
            'wave_to_mel',
            'content_vmamba',
            'hash_content',
            'hash_content_stats',
            'band_agg_32_to_18',
            'mel18_to_ceps',
            'content_cnn_encoder',
            'content_cnn_decoder',
            'deco_l2h_refiner',
            'l2h_flow',
            'hf_sideband_encoder',
            'hf2ceps',
        ]:
            if hasattr(model, name):
                _freeze_module(getattr(model, name))

        # 只训练 F0/VUV 分支 + vocoder
        for name in [
            'f0vuv_enc',
            'f0vuv_jscc_enc',
            'f0vuv_jscc_dec',
            'hash_f0vuv',
            'f0vuv_dec',
            'vocoder',
        ]:
            if hasattr(model, name):
                _unfreeze_module(getattr(model, name))

    return model


def model_forward(
    model: DualBranchBarkJSCC,
    batch: Dict[str, torch.Tensor],
    channel_sim: ChannelSimulator,
    cfg: SupportConfig,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    feats_36 = batch["x"].to(device)
    audio = batch["audio"].to(device)

    # Select forward path based on config:
    #   content_only + with_hash -> forward_content_only (mel+hash, skip F0/ceps/vocoder)
    #   content_only + no_hash -> forward_content_only_no_hash (mel only, skip F0/ceps/vocoder/hash)
    #   with_hash -> forward_with_hash (full pipeline with hash)
    #   else -> regular forward (no hash)
    content_only = getattr(cfg, "content_only", False)
    if content_only:
        if cfg.with_hash:
            forward_fn = model.forward_content_only
        else:
            forward_fn = model.forward_content_only_no_hash
    elif cfg.with_hash:
        forward_fn = model.forward_with_hash
    else:
        forward_fn = model

    out = forward_fn(
        audio=audio,
        fargan_feats=feats_36,
        channel_sim=channel_sim,
        snr_min_db=cfg.snr_min_db,
        snr_max_db=cfg.snr_max_db,
        target_len=audio.size(-1),
    )

    out["audio"] = audio

    # Skip teacher path in content-only mode (no vocoder available)
    if getattr(cfg, "content_only", False):
        return out

    # Optional teacher path (frozen HF distill): generate audio from GT features using teacher vocoder (if present)
    try:
        lam_teacher = float(getattr(cfg, 'lambda_teacher_hf', 0.0))
        if lam_teacher > 0.0 and isinstance(feats_36, torch.Tensor):
            with torch.no_grad():
                if hasattr(model, 'vocoder_teacher'):
                    _per_t, aud_t = model.vocoder_teacher(feats_36, target_len=audio.size(-1))
                else:
                    _per_t, aud_t = model.vocoder(feats_36, target_len=audio.size(-1))
                if aud_t.dim() > 2:
                    aud_t = aud_t.squeeze(1)
                # Align length defensively
                if aud_t.size(-1) != audio.size(-1):
                    min_len = min(aud_t.size(-1), audio.size(-1))
                    aud_t = aud_t[..., :min_len]
                out["audio_teacher"] = aud_t
    except Exception:
        # Silent fail; keep training resilient
        pass
    return out


def _compute_content_only_losses(
    model: DualBranchBarkJSCC,
    out: Dict[str, torch.Tensor],
    cfg: SupportConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, float]]:
    """Compute losses for content-only mode (Bark/BFCC reconstruction only).

    This function computes only mel-related losses and hash regularization,
    skipping audio/ceps/F0/VUV losses for faster training when focusing on
    the content branch (wave -> Bark/BFCC -> VMamba+Hash -> bark_hat).

    Supported losses:
    - mel MS-SSIM (lambda_mel)
    - mel L1 (lambda_mel_l1)
    - mel energy/brightness (lambda_mel_energy, lambda_mel_energy_t)
    - mel texture/valley (lambda_mel_texture, lambda_mel_valley)
    - hash reconstruction (lambda_hash_recon)
    - hash regularization (lambda_hash_reg)
    """
    loss_dict: Dict[str, float] = {}
    grad_info: Dict[str, float] = {}
    total = torch.tensor(0.0, device=device, requires_grad=True)

    mel = out.get("mel")
    mel_hat = out.get("mel_hat")

    if mel is None or mel_hat is None:
        return total, loss_dict, grad_info

    def _ensure_btf(x: torch.Tensor) -> torch.Tensor:
        """确保张量形状为 [B,T,F]，若为更高维则压缩中间维度。

        约定：最后两维始终对应时间和频率；若存在多余通道/层，
        则在第二维上取平均以获得单一 [T,F] 轨迹。
        """
        if x.dim() == 3:
            return x
        if x.dim() < 3:
            raise ValueError(f"mel tensor with dim < 3 not supported: shape={tuple(x.shape)}")
        B = x.size(0)
        T = x.size(-2)
        F = x.size(-1)
        x = x.view(B, -1, T, F)
        return x.mean(dim=1)

    # 统一将 mel / mel_hat 压为 [B,T,F]
    mel = _ensure_btf(mel)
    mel_hat = _ensure_btf(mel_hat)

    # Align time and frequency dimensions defensively, since不同前向路径
    # （VMamba vs CNN baseline）可能产生略有差异的边界尺寸。
    T_mel = mel.size(1)
    T_hat = mel_hat.size(1)
    T = min(T_mel, T_hat)
    F_mel = mel.size(2)
    F_hat = mel_hat.size(2)
    F_dim = min(F_mel, F_hat)

    mel = mel[:, :T, :F_dim]
    mel_hat = mel_hat[:, :T, :F_dim]

    # 可选：在计算 mel/BFCC 相关损失前裁掉边界若干帧/频带，
    # 避免 CNN/转置卷积在感受野不足的区域被强行拟合。
    try:
        crop_t = int(getattr(cfg, 'mel_loss_crop_time', 0))
        crop_f = int(getattr(cfg, 'mel_loss_crop_freq', 0))
    except Exception:
        crop_t = 0
        crop_f = 0

    if crop_t > 0 and T > 2 * crop_t:
        mel = mel[:, crop_t: T - crop_t, :]
        mel_hat = mel_hat[:, crop_t: T - crop_t, :]
        T = mel.size(1)
    if crop_f > 0 and F_dim > 2 * crop_f:
        mel = mel[:, :, crop_f: F_dim - crop_f]
        mel_hat = mel_hat[:, :, crop_f: F_dim - crop_f]
        F_dim = mel.size(2)

    # Use mel_hat_refined if available (after L2H, though L2H is skipped in content-only)
    mel_hat_for_loss = out.get("mel_hat_refined", mel_hat)
    mel_hat_for_loss = _ensure_btf(mel_hat_for_loss)[:, :T, :F_dim]

    # ------------------------------------------------------------------
    # 1) Mel MS-SSIM loss (structural similarity)
    # ------------------------------------------------------------------
    def _soft_unit(x: torch.Tensor) -> torch.Tensor:
        # Map [-8,0] to [~0.12,~0.73], leaving gradient margin on both sides
        return torch.sigmoid((x + 8.0) / 2.0)

    mel_n = _soft_unit(mel)
    mel_hat_n = _soft_unit(mel_hat_for_loss)
    # Convert to [B,1,T,F] for MS-SSIM
    mel_n_img = mel_n.unsqueeze(1)
    mel_hat_n_img = mel_hat_n.unsqueeze(1)

    ms_ssim = MS_SSIM(data_range=1.0, channel=1, levels=4)
    loss_mel_struct = ms_ssim(mel_hat_n_img, mel_n_img).mean()

    lam_mel = float(getattr(cfg, 'lambda_mel', 0.0))
    if lam_mel > 0.0:
        total = total + lam_mel * loss_mel_struct
        loss_dict["mel_ms_ssim"] = float(loss_mel_struct.item())

    # ------------------------------------------------------------------
    # 2) Mel L1 loss (optional brightness/detail stabilizer)
    # ------------------------------------------------------------------
    lam_mel_l1 = float(getattr(cfg, 'lambda_mel_l1', 0.0))
    if lam_mel_l1 > 0.0:
        l_mel_l1 = torch.mean(torch.abs(mel_hat_for_loss - mel))
        total = total + lam_mel_l1 * l_mel_l1
        loss_dict['mel_l1'] = float(l_mel_l1.item())

    # ------------------------------------------------------------------
    # 3) Mel/BFCC energy/brightness anchors
    # ------------------------------------------------------------------
    try:
        lam_energy = float(getattr(cfg, 'lambda_mel_energy', 0.0))
        if lam_energy > 0.0:
            mel_mean_hat = mel_hat_for_loss.mean(dim=(1, 2))
            mel_mean = mel.mean(dim=(1, 2))
            loss_mel_energy = torch.mean(torch.abs(mel_mean_hat - mel_mean))
            total = total + lam_energy * loss_mel_energy
            loss_dict["mel_energy"] = float(loss_mel_energy.item())
    except Exception:
        pass

    # 逐帧亮度曲线（与 BFCC baseline 的 lambda_energy_t 一致，只是这里在 content-only 路径上
    # 仍沿用 mel 命名；mel 实际为 32-Bark log 能量图，即 BFCC 图像域。）
    try:
        lam_energy_t_mel = float(getattr(cfg, 'lambda_mel_energy_t', 0.0))
        if lam_energy_t_mel > 0.0:
            mel_e_tgt = mel.mean(dim=2)
            mel_e_hat = mel_hat_for_loss.mean(dim=2)
            loss_mel_energy_t = torch.mean(torch.abs(mel_e_hat - mel_e_tgt))
            total = total + lam_energy_t_mel * loss_mel_energy_t
            loss_dict["mel_energy_t"] = float(loss_mel_energy_t.item())
    except Exception:
        pass

    # 可选：直接复用 BFCC baseline 风格的图像域约束，对同一个 32×T BFCC 图像施加：
    # - 每帧均值能量 |E_hat(t)-E(t)| （lambda_energy_t）
    # - 每频带均值能量的相对误差 |E_hat(f)-E(f)| / (|E(f)|+eps)（lambda_energy_f）
    # - 时间轴梯度一致性 |∂_t BFCC_hat - ∂_t BFCC| （lambda_tex_t）
    # 这些项仅在对应 lambda_* > 0 时启用，默认保持关闭。
    try:
        lam_bfcc_et = float(getattr(cfg, 'lambda_energy_t', 0.0))
        if lam_bfcc_et > 0.0:
            e_t = mel.mean(dim=2)                # [B,T]
            e_hat_t = mel_hat_for_loss.mean(dim=2)
            loss_energy_t = torch.mean(torch.abs(e_hat_t - e_t))
            total = total + lam_bfcc_et * loss_energy_t
            loss_dict["bfcc_energy_t"] = float(loss_energy_t.item())
    except Exception:
        pass

    # 逐频带：同时对时间均值和时间标准差做相对误差约束，
    # 更直接地拉齐每个 Bark band 的亮度标度与对比度。
    try:
        lam_bfcc_ef = float(getattr(cfg, 'lambda_energy_f', 0.0))
        if lam_bfcc_ef > 0.0:
            mean_f_ref = mel.mean(dim=1)             # [B,F]
            mean_f_hat = mel_hat_for_loss.mean(dim=1)
            std_f_ref = mel.std(dim=1)
            std_f_hat = mel_hat_for_loss.std(dim=1)

            eps = 1e-4
            denom_m = torch.clamp(mean_f_ref.abs(), min=eps)
            denom_s = torch.clamp(std_f_ref.abs(), min=eps)

            rel_m = torch.abs(mean_f_hat - mean_f_ref) / denom_m
            rel_s = torch.abs(std_f_hat - std_f_ref) / denom_s

            loss_energy_f = 0.5 * (rel_m.mean() + rel_s.mean())
            total = total + lam_bfcc_ef * loss_energy_f
            loss_dict["bfcc_energy_f"] = float(loss_energy_f.item())
    except Exception:
        pass

    try:
        lam_bfcc_tex = float(getattr(cfg, 'lambda_tex_t', 0.0))
        if lam_bfcc_tex > 0.0 and mel.size(1) > 1:
            grad_t = mel[:, 1:, :] - mel[:, :-1, :]
            grad_hat_t = mel_hat_for_loss[:, 1:, :] - mel_hat_for_loss[:, :-1, :]
            loss_tex_t = torch.mean(torch.abs(grad_hat_t - grad_t))
            total = total + lam_bfcc_tex * loss_tex_t
            loss_dict["bfcc_tex_t"] = float(loss_tex_t.item())
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 4) Mel texture loss (gradient consistency on bright voiced regions)
    # ------------------------------------------------------------------
    try:
        lam_texture = float(getattr(cfg, 'lambda_mel_texture', 0.0))
        if lam_texture > 0.0:
            # Frequency gradient (mel domain)
            grad_hat_f = mel_hat_for_loss[:, :, 1:] - mel_hat_for_loss[:, :, :-1]
            grad_f = mel[:, :, 1:] - mel[:, :, :-1]
            loss_texture = torch.mean(torch.abs(grad_hat_f - grad_f))
            total = total + lam_texture * loss_texture
            loss_dict["mel_texture"] = float(loss_texture.item())
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 5) Mel valley loss (prevent over-smoothing spectral valleys)
    # ------------------------------------------------------------------
    try:
        lam_valley = float(getattr(cfg, 'lambda_mel_valley', 0.0))
        if lam_valley > 0.0:
            # Valley = frames where mel is below per-frame mean
            mel_mean_f = mel.mean(dim=-1, keepdim=True)
            valley_mask = (mel < mel_mean_f).float()
            diff = torch.abs(mel_hat_for_loss - mel) * valley_mask
            denom = valley_mask.sum() + 1e-6
            loss_valley = diff.sum() / denom
            total = total + lam_valley * loss_valley
            loss_dict["mel_valley"] = float(loss_valley.item())
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 6) Mel band energy loss (per-band energy alignment)
    # ------------------------------------------------------------------
    try:
        lam_bandE = float(getattr(cfg, 'lambda_mel_bandE', 0.0))
        if lam_bandE > 0.0:
            loss_bandE = torch.mean(torch.abs(
                mel_hat_for_loss.mean(dim=1) - mel.mean(dim=1)
            ))
            total = total + lam_bandE * loss_bandE
            loss_dict["mel_bandE"] = float(loss_bandE.item())
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 7) Hash reconstruction / regularization（仅 quantizer_type='hash'）
    # ------------------------------------------------------------------
    qtype = str(getattr(cfg, 'quantizer_type', 'hash'))
    if qtype == 'hash':
        lam_hash_recon = float(getattr(cfg, 'lambda_hash_recon', 0.0))
        if lam_hash_recon > 0.0 and "tokens" in out and "tokens_hat" in out:
            tokens = out["tokens"]
            tokens_hat = out["tokens_hat"]
            loss_hash_recon_c = F.mse_loss(tokens_hat, tokens)
            total = total + lam_hash_recon * loss_hash_recon_c
            loss_dict["hash_recon_c"] = float(loss_hash_recon_c.item())

        lam_hash_reg = float(getattr(cfg, 'lambda_hash_reg', 0.0))
        if lam_hash_reg > 0.0 and "hash_reg_terms" in out:
            hash_reg_terms = out["hash_reg_terms"]
            hash_reg = sum(hash_reg_terms.values())
            total = total + lam_hash_reg * hash_reg
            loss_dict["hash_reg"] = float(hash_reg.item()) if hasattr(hash_reg, 'item') else float(hash_reg)

    # ------------------------------------------------------------------
    # 8b) RVQ VQ loss（当 quantizer_type='rvq' 时使用）
    # ------------------------------------------------------------------
    lam_vq_global = float(getattr(cfg, 'lambda_vq', 0.0))
    lam_vq_c = float(getattr(cfg, 'lambda_vq_c', 0.0))
    lam_vq_f = float(getattr(cfg, 'lambda_vq_f', 0.0))
    vq_loss = out.get("vq_loss", None)
    vq_loss_c = out.get("vq_loss_content", None)
    vq_loss_f0 = out.get("vq_loss_f0", None)

    # 分支权重：若 lambda_vq_c/f > 0，则单独使用；否则回退到全局 lambda_vq。
    lam_c = lam_vq_c if lam_vq_c > 0.0 else lam_vq_global
    lam_f = lam_vq_f if lam_vq_f > 0.0 else lam_vq_global

    if isinstance(vq_loss_c, torch.Tensor) and lam_c > 0.0:
        total = total + lam_c * vq_loss_c
        loss_dict["vq_loss_c"] = float(vq_loss_c.item())

    if isinstance(vq_loss_f0, torch.Tensor) and lam_f > 0.0:
        total = total + lam_f * vq_loss_f0
        loss_dict["vq_loss_f0"] = float(vq_loss_f0.item())

    # 仍然提供一个总的 vq 诊断项（不带权重，仅为便于观察整体量级）。
    if isinstance(vq_loss_c, torch.Tensor) or isinstance(vq_loss_f0, torch.Tensor):
        vq_total = None
        if isinstance(vq_loss_c, torch.Tensor):
            vq_total = vq_loss_c if vq_total is None else vq_total + vq_loss_c
        if isinstance(vq_loss_f0, torch.Tensor):
            vq_total = vq_loss_f0 if vq_total is None else vq_total + vq_total  # type: ignore[assignment]
        try:
            if isinstance(vq_total, torch.Tensor):
                loss_dict["vq"] = float(vq_total.item())
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 9) Ceps 重建相关损失（在 content_only 模式下复用主路径逻辑）
    # ------------------------------------------------------------------
    ceps_hat = out.get("ceps_hat", None)
    ceps_tgt = out.get("ceps", None)
    if isinstance(ceps_hat, torch.Tensor) and isinstance(ceps_tgt, torch.Tensor):
        Tm = min(ceps_hat.size(1), ceps_tgt.size(1))
        Dm = min(ceps_hat.size(2), ceps_tgt.size(2))

        # 主 ceps L1（可选有声门控；content_only 下无 voiced_mask，退化为全局 L1）
        lam_ceps = float(getattr(cfg, 'lambda_ceps', 0.0))
        if lam_ceps > 0.0:
            try:
                voiced_mask: Optional[torch.Tensor] = None
                if voiced_mask is not None and isinstance(voiced_mask, torch.Tensor):
                    vm = voiced_mask[:, :Tm].to(ceps_hat.dtype)  # [B,T]
                    diff_ceps = torch.abs(ceps_hat[:, :Tm, :Dm] - ceps_tgt[:, :Tm, :Dm])  # [B,T,D]
                    denom = vm.sum() * Dm + 1e-6
                    loss_ceps = (diff_ceps * vm.unsqueeze(-1)).sum() / denom
                else:
                    loss_ceps = F.l1_loss(ceps_hat[:, :Tm, :Dm], ceps_tgt[:, :Tm, :Dm])
            except Exception:
                a = ceps_hat.reshape(-1)
                b = ceps_tgt.reshape(-1)
                n = min(a.numel(), b.numel())
                loss_ceps = F.l1_loss(a[:n], b[:n])

            total = total + lam_ceps * loss_ceps
            loss_dict["ceps"] = float(loss_ceps.item())

        # 倒谱加权损失：对不同阶数的 ceps 施加非均匀权重。
        #
        # 调整策略（与原版相比）：
        #   - 对低阶系数（对应整体能量/低频包络）只在误差超过
        #     一定 margin 时才继续优化，保留一定“模糊度”；
        #   - 对中高阶系数不设 margin，并可选放大权重，鼓励
        #     内容分支在中高频 BFCC 上投入更多容量。
        lam_ceps_w = float(getattr(cfg, 'lambda_ceps_weighted', 0.0))
        if lam_ceps_w > 0.0:
            try:
                base_w = torch.tensor(
                    [
                        2.0,
                        1.5,
                        1.5,
                        1.2,
                        1.2,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        0.8,
                        0.8,
                        0.7,
                        0.7,
                        0.5,
                        0.5,
                        0.4,
                        0.4,
                        0.3,
                        0.3,
                    ],
                    device=ceps_hat.device,
                    dtype=ceps_hat.dtype,
                )
                if Dm <= base_w.numel():
                    w = base_w[:Dm]
                else:
                    tail = base_w[-1].expand(Dm - base_w.numel())
                    w = torch.cat([base_w, tail], dim=0)
                # 低阶/高阶分段：低阶只追到一定 margin 即可，高阶允许继续压误差。
                low_stop = int(getattr(cfg, 'ceps_low_stop', 4))
                low_stop = max(0, min(low_stop, Dm))
                low_margin = float(getattr(cfg, 'ceps_low_margin', 0.5))
                hi_boost = float(getattr(cfg, 'ceps_hi_boost', 1.0))

                diff = ceps_hat[:, :Tm, :Dm] - ceps_tgt[:, :Tm, :Dm]
                diff_abs = diff.abs()
                diff2 = diff ** 2

                # 对低阶系数使用 hinge margin：|Δ|<=margin 时梯度≈0，只要模糊到一定程度即可。
                if low_stop > 0 and low_margin > 0.0:
                    diff_low = torch.relu(diff_abs[:, :, :low_stop] - low_margin)
                    diff2_low = diff_low ** 2
                    if low_stop < Dm:
                        diff2 = torch.cat([diff2_low, diff2[:, :, low_stop:]], dim=-1)
                    else:
                        diff2 = diff2_low

                # 可选：放大中高阶权重，鼓励内容分支在高频 BFCC 上花更多“精力”。
                w = w.view(1, 1, Dm)
                if hi_boost != 1.0 and low_stop < Dm:
                    w_hi = w[:, :, low_stop:] * hi_boost
                    w = torch.cat([w[:, :, :low_stop], w_hi], dim=-1)

                diff2 = diff2 * w
                loss_cw = diff2.mean()
                total = total + lam_ceps_w * loss_cw
                loss_dict["ceps_weighted"] = float(loss_cw.item())
            except Exception:
                pass

        # 倒谱高阶监督（如 c10..），抑制高频抹平
        lam_ceps_hi = float(getattr(cfg, 'lambda_ceps_hi', 0.0))
        if lam_ceps_hi > 0.0 and Dm > 1:
            s0 = int(getattr(cfg, "ceps_hi_start", 10))
            s0 = max(1, min(s0, Dm - 1))
            try:
                voiced_mask_hi: Optional[torch.Tensor] = None
                if voiced_mask_hi is not None and isinstance(voiced_mask_hi, torch.Tensor):
                    vm = voiced_mask_hi[:, :Tm].to(ceps_hat.dtype)
                    diff_hi = torch.abs(ceps_hat[:, :Tm, s0:Dm] - ceps_tgt[:, :Tm, s0:Dm])
                    denom = vm.sum() * max(1, Dm - s0) + 1e-6
                    l_chi = (diff_hi * vm.unsqueeze(-1)).sum() / denom
                else:
                    l_chi = F.l1_loss(ceps_hat[:, :Tm, s0:Dm], ceps_tgt[:, :Tm, s0:Dm])
                total = total + lam_ceps_hi * l_chi
                loss_dict["ceps_hi"] = float(l_chi.item())
            except Exception:
                pass

        # BFCC→ceps 映射损失（content-only 模式下，可选校准 BFCC→ceps pipeline）
        lam_ceps_map_gt = float(getattr(cfg, 'lambda_ceps_map_gt', 0.0))
        if lam_ceps_map_gt > 0.0:
            try:
                mel_gt = out.get('mel', None)
                if isinstance(mel_gt, torch.Tensor):
                    E_gt = torch.pow(10.0, torch.clamp(mel_gt, min=-10.0, max=10.0))
                    e18_gt = model.band_agg_32_to_18(E_gt)
                    e18_log_gt = torch.log10(e18_gt + 1e-10)
                    e18_log_gt = opus_band_log_smooth(e18_log_gt)

                    ceps_map = model.mel18_to_ceps(e18_log_gt)
                    ceps_map = torch.nan_to_num(ceps_map, nan=0.0)
                    ceps_t = torch.nan_to_num(ceps_tgt, nan=0.0)

                    Tm_map = min(ceps_map.size(1), ceps_t.size(1))
                    Dm_map = min(ceps_map.size(2), ceps_t.size(2))
                    loss_map = F.l1_loss(ceps_map[:, :Tm_map, :Dm_map], ceps_t[:, :Tm_map, :Dm_map])

                    total = total + lam_ceps_map_gt * loss_map
                    loss_dict['ceps_map_gt'] = float(loss_map.item())
            except Exception:
                pass

    return total, loss_dict, grad_info


def compute_losses(
    model: DualBranchBarkJSCC,
    out: Dict[str, torch.Tensor],
    cfg: SupportConfig,
    device: torch.device,
    global_step: int = 0,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, float]]:
    # -------------------------------------------------------------------
    # Content-only mode: only compute mel-related losses, skip audio/ceps/F0
    # -------------------------------------------------------------------
    if getattr(cfg, "content_only", False):
        return _compute_content_only_losses(model, out, cfg, device)

    audio_real = out["audio"]
    audio_hat = out["audio_hat"]

    def _frame_rms(x: torch.Tensor, frame_len: int = 160, hop: int = 160) -> torch.Tensor:
        """按固定窗口/步长计算每帧 RMS，返回 [B,Tf]。"""
        B, L = x.shape
        if L < frame_len:
            pad = frame_len - L
            x = torch.nn.functional.pad(x, (0, pad))
            L = frame_len
        x_frames = x.unfold(dimension=1, size=frame_len, step=hop)  # [B,Tf,frame_len]
        rms = torch.sqrt(x_frames.pow(2).mean(dim=-1) + 1e-8)
        return rms

    lam_sil_wave = float(getattr(cfg, 'lambda_silence_wave', 0.0))
    lam_sil_mel = float(getattr(cfg, 'lambda_silence_mel', 0.0))

    # 预先计算真实音频的帧 RMS（10 ms@16kHz），用于静音判定
    rms_real = _frame_rms(audio_real)  # [B,Tr]
    eps = 1e-8
    rms_max = rms_real.max(dim=1, keepdim=True).values.clamp_min(eps)
    rms_norm = rms_real / rms_max
    rms_db = 20.0 * torch.log10(rms_norm + eps)  # 归一化 dB，0 为峰值，负值为静音

    # ---- 统一静音/有声掩膜：基于帧 RMS OR 高频 log-Bark 能量 ----
    silence_mask: Optional[torch.Tensor] = None   # [B,T] bool
    voiced_mask: Optional[torch.Tensor] = None    # [B,T] bool
    mel_for_mask = out.get("mel", None)
    if isinstance(mel_for_mask, torch.Tensor) and mel_for_mask.dim() == 3:
        try:
            Bm, Tm, Fm = mel_for_mask.shape
            # 使用偏向高频的能量判定静音：默认从 mel_hp_low_bins 之后算均值，
            # 避免摩擦音/高频过渡被整体平均成“低能量”而误判为静音。
            hf_low = int(getattr(cfg, 'silence_hf_low_bins', int(getattr(cfg, 'mel_hp_low_bins', 16))))
            hf_low = max(0, min(hf_low, Fm - 1))
            mel_hf = mel_for_mask[:, :, hf_low:] if hf_low < Fm else mel_for_mask
            mel_energy = mel_hf.mean(dim=-1)  # [B,Tm]

            # 对齐 RMS 与 mel 时间长度
            Tr = rms_db.size(1)
            T_use = min(Tm, Tr)
            mel_energy_use = mel_energy[:, :T_use]
            rms_db_use = rms_db[:, :T_use]

            # 高频 log-mel 能量阈值（log10 域）
            sil_thr_db_hf = float(getattr(cfg, 'silence_energy_thr_db', -35.0))
            sil_thr_log = sil_thr_db_hf / 10.0
            energy_sil = (mel_energy_use <= sil_thr_log)

            # 帧 RMS 阈值（dB，相对每段峰值）
            rms_thr_db = float(getattr(cfg, 'silence_rms_thr_db', -35.0))
            rms_sil = (rms_db_use <= rms_thr_db)

            # 基础静音：RMS 或 高频能量 低于各自阈值
            base_sil = energy_sil | rms_sil

            # 可选：叠加高频方差门控，仅在“能量低且高频几乎无纹理”时视为静音
            use_var = bool(getattr(cfg, 'silence_use_hf_var', False))
            if use_var:
                # 高频 log 能量的方差，刻画高频纹理强度
                hf_var = mel_hf.var(dim=-1)  # [B,Tm]
                hf_var_use = hf_var[:, :T_use]
                var_thr = float(getattr(cfg, 'silence_hf_var_thr', 0.02))
                texture_quiet = hf_var_use <= var_thr
                silence_mask = base_sil & texture_quiet
            else:
                silence_mask = base_sil

            voiced_mask = ~silence_mask
        except Exception:
            silence_mask = None
            voiced_mask = None

    # --- 边界保护：把“非静音/有声”掩膜做时间膨胀，避免有声-无声边界被当作静音抹平 ---
    # 说明：160 samples@16kHz ≈10ms/帧；radius=3 → 约 30ms 的边界保护带
    try:
        dilate = int(getattr(cfg, 'silence_dilate_frames', 0))
    except Exception:
        dilate = 0
    if (silence_mask is not None) and (dilate > 0):
        try:
            ns = (~silence_mask).to(torch.float32).unsqueeze(1)  # [B,1,T]
            ns = F.max_pool1d(ns, kernel_size=2 * dilate + 1, stride=1, padding=dilate)
            voiced_mask = (ns.squeeze(1) > 0.5)
            silence_mask = ~voiced_mask
        except Exception:
            pass

    # 初始化总损失与记录容器
    total = torch.zeros((), device=audio_hat.device, dtype=audio_hat.dtype)
    loss_dict: Dict[str, float] = {}
    grad_info: Dict[str, float] = {}
    loss_tensors: Dict[str, Tuple[torch.Tensor, str]] = {}

    # 1) STFT → 直接替换为“谱收敛（SC）”形式，避免幅度L1带来的“拉平”偏置
    # 当 lambda_wave 与 lambda_wave_mag 均为 0 时，完全跳过 STFT 相关计算，
    # 以避免在 hash-only 微调阶段额外的 FFT 开销。
    lam_wave = float(getattr(cfg, 'lambda_wave', 0.0))
    lam_wave_mag = float(getattr(cfg, 'lambda_wave_mag', 0.0))
    if (lam_wave > 0.0) or (lam_wave_mag > 0.0):
        if getattr(cfg, 'stft_preset', 'aether') == 'fargan':
            fs = [2560, 1280, 640, 320, 160, 80]
            hs = [640, 320, 160, 80, 40, 20]
            wl = [2560, 1280, 640, 320, 160, 80]
        else:
            fs = [1024, 512, 256, 128]
            hs = [256, 128, 64, 32]
            wl = [1024, 512, 256, 128]

        loss_stft = multi_resolution_sc_loss(
            audio_hat, audio_real, device=device,
            fft_sizes=fs, hop_sizes=hs, win_lengths=wl,
        )
        total = total + lam_wave * loss_stft
        loss_dict["stft"] = float(loss_stft.item())
        loss_tensors['stft'] = (loss_stft, 'audio')

        # 额外：MR-STFT 幅度 L1 作为稳定补充
        loss_stft_mag: Optional[torch.Tensor] = None
        try:
            if lam_wave_mag > 0.0:
                loss_stft_mag = multi_resolution_stft_loss(
                    audio_hat, audio_real, device=device,
                    fft_sizes=fs, hop_sizes=hs, win_lengths=wl,
                )
        except Exception:
            loss_stft_mag = None

        if isinstance(loss_stft_mag, torch.Tensor):
            total = total + lam_wave_mag * loss_stft_mag
            loss_dict["stft_mag"] = float(loss_stft_mag.item())
            loss_tensors['stft_mag'] = (loss_stft_mag, 'audio')

    # 1.1) SSL 语音内容一致性损失（帧级加权版）
    # 说明：
    #   - 仅在 cfg.lambda_ssl>0 且 cfg.ssl_model_name 非空时启用；
    #   - 使用 get_ssl_model() 加载并缓存 SSL 模型；
    #   - 只对 audio_hat 反向传播梯度（SSL 模型本身完全冻结）；
    #   - 为避免 AMP half 精度不稳定，内部强制以 full precision 计算 SSL 特征；
    #   - 在时间维上使用 silence_mask / 边界加权做帧级加权。
    lam_ssl = float(getattr(cfg, 'lambda_ssl', 0.0))
    ssl_name = getattr(cfg, 'ssl_model_name', None)
    if lam_ssl > 0.0 and isinstance(ssl_name, str) and ssl_name:
        ssl_model = get_ssl_model(ssl_name, device)
        if isinstance(ssl_model, torch.nn.Module):
            warm = int(getattr(cfg, 'ssl_warmup_steps', 0))
            if warm > 0:
                p = max(0.0, min(1.0, float(global_step) / float(max(1, warm))))
                lam_ssl_eff = lam_ssl * p
            else:
                lam_ssl_eff = lam_ssl

            if lam_ssl_eff > 0.0:
                try:
                    # 1) 提取 SSL 隐状态（全精度）
                    with autocast(enabled=False):
                        wav_hat = audio_hat
                        wav_ref = audio_real.detach()
                        if wav_hat.dim() > 2:
                            wav_hat = wav_hat.squeeze(1)
                        if wav_ref.dim() > 2:
                            wav_ref = wav_ref.squeeze(1)

                        out_hat = ssl_model(wav_hat, output_hidden_states=True)
                        out_ref = ssl_model(wav_ref, output_hidden_states=True)

                        hs_hat = getattr(out_hat, 'hidden_states', None)
                        hs_ref = getattr(out_ref, 'hidden_states', None)
                        if not isinstance(hs_hat, (list, tuple)) or not isinstance(hs_ref, (list, tuple)):
                            if hasattr(out_hat, 'last_hidden_state') and hasattr(out_ref, 'last_hidden_state'):
                                hs_hat = (out_hat.last_hidden_state,)
                                hs_ref = (out_ref.last_hidden_state,)
                            else:
                                hs_hat = None
                                hs_ref = None

                    loss_ssl = None
                    if isinstance(hs_hat, (list, tuple)) and isinstance(hs_ref, (list, tuple)):
                        n_layers = min(len(hs_hat), len(hs_ref))
                        if n_layers > 0:
                            layer_idxs = getattr(cfg, 'ssl_layers', None)
                            if not layer_idxs:
                                if n_layers >= 13:
                                    layer_idxs = [6, 9, 12]
                                elif n_layers >= 7:
                                    layer_idxs = [2, 4, 6]
                                else:
                                    layer_idxs = [n_layers - 1]

                            selected: List[int] = []
                            for idx in layer_idxs:
                                i = idx if idx >= 0 else (n_layers + idx)
                                if 0 <= i < n_layers:
                                    selected.append(i)
                            if not selected:
                                selected = [n_layers - 1]

                            loss_ssl_val = 0.0
                            n_used = 0

                            for i in selected:
                                f_hat = hs_hat[i]  # [B,T_feat,D]
                                f_ref = hs_ref[i].detach()
                                Bf, Tf, Df = f_hat.shape
                                T_feat = min(f_hat.size(1), f_ref.size(1))
                                f_hat = f_hat[:, :T_feat, :]
                                f_ref = f_ref[:, :T_feat, :]

                                # 2) 特征归一化（LayerNorm），避免纯能量差主导
                                f_hat_n = torch.nn.functional.layer_norm(f_hat, (Df,))
                                f_ref_n = torch.nn.functional.layer_norm(f_ref, (Df,))

                                # 3) 帧级差异 [B,T_feat]
                                frame_diff = torch.mean(torch.abs(f_hat_n - f_ref_n), dim=-1)

                                # 4) 时间权重 w[b,t]
                                w = torch.ones_like(frame_diff)

                                # 4.a 非静音权重：使用 silence_mask 上采样到 SSL 帧长
                                if silence_mask is not None and silence_mask.dim() == 2:
                                    try:
                                        sm = silence_mask.to(frame_diff.dtype)  # [B,T_mel]
                                        sm_up = torch.nn.functional.interpolate(
                                            sm.unsqueeze(1), size=T_feat, mode='nearest'
                                        ).squeeze(1)  # [B,T_feat]
                                        ns = 1.0 - sm_up  # 1=非静音
                                        w = w * ns
                                    except Exception:
                                        pass

                                # 4.b 边界加权：静音↔非静音切换的帧额外加权
                                if silence_mask is not None and silence_mask.dim() == 2:
                                    try:
                                        sm = silence_mask.to(frame_diff.dtype)
                                        sm_up = torch.nn.functional.interpolate(
                                            sm.unsqueeze(1), size=T_feat, mode='nearest'
                                        ).squeeze(1)
                                        ns = 1.0 - sm_up
                                        if T_feat > 1:
                                            bd = torch.zeros_like(ns)
                                            bd[:, 1:] = (ns[:, 1:] - ns[:, :-1]).abs()
                                            w = w + bd
                                    except Exception:
                                        pass

                                denom = w.sum() + 1e-6
                                if denom > 0:
                                    loss_layer = (frame_diff * w).sum() / denom
                                    loss_ssl_val = loss_ssl_val + loss_layer
                                    n_used += 1

                            if n_used > 0:
                                loss_ssl = loss_ssl_val / float(n_used)

                    if isinstance(loss_ssl, torch.Tensor):
                        total = total + lam_ssl_eff * loss_ssl
                        loss_dict['ssl_content'] = float(loss_ssl.item())
                        loss_tensors['ssl_content'] = (loss_ssl, 'audio')
                except Exception as _ssl_run_err:
                    try:
                        print(f"[SSL] content loss skipped due to error: {_ssl_run_err}")
                    except Exception:
                        pass

    # 记录静音相关诊断信息，便于确认 lambda/掩膜是否生效
    # （仅在配置中显式启用静音 shaping 时添加，避免污染默认日志）
    if lam_sil_wave > 0.0:
        loss_dict['lam_sil_wave'] = float(lam_sil_wave)
        try:
            if silence_mask is not None:
                frac = float(silence_mask.float().mean().item())
            else:
                frac = 0.0
        except Exception:
            frac = 0.0
        loss_dict['silence_frac'] = frac

    # 波形域静音能量约束：仅在目标音频静音帧上，惩罚 audio_hat 的帧 RMS 能量
    if lam_sil_wave > 0.0 and silence_mask is not None:
        try:
            rms_hat = _frame_rms(audio_hat)                    # [B,Th]
            sm = silence_mask.to(rms_hat.dtype)                # [B,Tm]，True=静音
            # 对齐帧数，防止由于 vocoder 裁剪导致 Th ≠ Tm
            Th = rms_hat.size(1)
            Tm = sm.size(1)
            T_use = min(Th, Tm)
            rms_hat_use = rms_hat[:, :T_use]
            sm_use = sm[:, :T_use]
            loss_sil_wave = (rms_hat_use * sm_use).sum() / (sm_use.sum() + 1e-6)
            total = total + lam_sil_wave * loss_sil_wave
            loss_dict["sil_wave"] = float(loss_sil_wave.item())
            loss_tensors['sil_wave'] = (loss_sil_wave, 'audio')
        except Exception:
            pass

    # 1.5) mel 结构一致性（用 MS-SSIM 替换原 mel L1）
    # 当所有 mel 相关权重为 0 时，可选择完全跳过该分支以节省计算；
    # 目前保持行为：仅在对应 lambda_* > 0 时才将各项加入 total。
    if "mel_hat" in out and "mel" in out:
        mel = out["mel"]
        mel_hat = out["mel_hat"]
        # 确保 mel 是 3D [B,T,F]
        if mel.dim() == 4:
            mel = mel.squeeze(1)
        if mel_hat.dim() == 4:
            mel_hat = mel_hat.squeeze(1)
        # 先对齐时间/频率维度，防御性裁剪，再根据配置可选裁掉边界若干帧/频带，
        # 避免 CNN/转置卷积在边缘感受野不足时被强行拟合。
        Tm = min(mel.size(1), mel_hat.size(1))
        Fm = min(mel.size(2), mel_hat.size(2))
        mel = mel[:, :Tm, :Fm]
        mel_hat = mel_hat[:, :Tm, :Fm]

        try:
            crop_t = int(getattr(cfg, 'mel_loss_crop_time', 0))
            crop_f = int(getattr(cfg, 'mel_loss_crop_freq', 0))
        except Exception:
            crop_t = 0
            crop_f = 0

        if crop_t > 0 and Tm > 2 * crop_t:
            mel = mel[:, crop_t: Tm - crop_t, :]
            mel_hat = mel_hat[:, crop_t: Tm - crop_t, :]
            Tm = mel.size(1)
        if crop_f > 0 and Fm > 2 * crop_f:
            mel = mel[:, :, crop_f: Fm - crop_f]
            mel_hat = mel_hat[:, :, crop_f: Fm - crop_f]

        # 兼容 v2：若模型返回 refined 版本则优先用之；并与 mel 同步裁剪。
        # 为了让主干 mel 损失也直接约束高频纹理，当存在
        # mel_hat_refined 时改为以其为主；未启用 L2H 时两者相同。
        mel_hat_refined = out.get("mel_hat_refined", mel_hat)
        if mel_hat_refined is not None:
            if mel_hat_refined.dim() == 4:
                mel_hat_refined = mel_hat_refined.squeeze(1)
            mel_hat_refined = mel_hat_refined[:, :Tm, :Fm]
            if crop_t > 0 and Tm > 2 * crop_t:
                mel_hat_refined = mel_hat_refined[:, crop_t: Tm - crop_t, :]
            if crop_f > 0 and Fm > 2 * crop_f:
                mel_hat_refined = mel_hat_refined[:, :, crop_f: Fm - crop_f]
        else:
            mel_hat_refined = mel_hat

        # mel 相关损失统一使用 refined 版本（若存在），
        # 让高频细化结果也受到 MS-SSIM / L1 等主干损失的直接约束。
        mel_hat_for_loss = mel_hat_refined
        # 最终再次对齐 mel 与 mel_hat_for_loss 的 [B,T,F]，防御性裁剪，
        # 以防 refined 分支产生略有不同的边界尺寸。
        try:
            Bm = min(mel.size(0), mel_hat_for_loss.size(0))
            Tm2 = min(mel.size(1), mel_hat_for_loss.size(1))
            Fm2 = min(mel.size(2), mel_hat_for_loss.size(2))
            mel = mel[:Bm, :Tm2, :Fm2]
            mel_hat_for_loss = mel_hat_for_loss[:Bm, :Tm2, :Fm2]
        except Exception:
            # 若出现异常，保持原状，让后续 L1 fallback 再次兜底
            pass
        # 归一化到 [0,1]（原 mel 约在 [-8,0]），改用 soft 映射避免 clamp 饱和导致梯度为0
        def _soft_unit(x: torch.Tensor) -> torch.Tensor:
            # 将 [-8,0] 映射到 [~0.12,~0.73]，给两侧留梯度余量
            return torch.sigmoid((x + 8.0) / 2.0)
        mel_n = _soft_unit(mel)
        mel_hat_n = _soft_unit(mel_hat_for_loss)
        # 确保 mel_n 与 mel_hat_n 形状完全一致，再进入 MS-SSIM / L1
        if mel_n.shape != mel_hat_n.shape:
            try:
                Bb = min(mel_n.size(0), mel_hat_n.size(0))
                Tb = min(mel_n.size(1), mel_hat_n.size(1))
                Fb = min(mel_n.size(2), mel_hat_n.size(2))
                mel_n = mel_n[:Bb, :Tb, :Fb]
                mel_hat_n = mel_hat_n[:Bb, :Tb, :Fb]
            except Exception:
                # 留给后续 L1 fallback 处理
                pass
        # MS-SSIM 需要 4D [B,1,H,W] 输入且空间维度足够大
        # 若维度不符合预期，退化为 L1 loss
        min_spatial = 16  # MS-SSIM with levels=4 needs at least 16 pixels
        use_ms_ssim = (mel_n.dim() == 3 and mel_n.size(1) >= min_spatial and mel_n.size(2) >= min_spatial)
        if use_ms_ssim:
            try:
                mel_n_img = mel_n.unsqueeze(1)  # [B, 1, T, F]
                mel_hat_n_img = mel_hat_n.unsqueeze(1)
                # 使用全局 MS_SSIM 实例
                global _ms_ssim_instance
                if '_ms_ssim_instance' not in globals() or _ms_ssim_instance is None:
                    _ms_ssim_instance = MS_SSIM(data_range=1.0, channel=1, levels=4)
                _ms_ssim_instance = _ms_ssim_instance.to(mel_n_img.device)
                loss_mel_struct = _ms_ssim_instance(mel_hat_n_img, mel_n_img).mean()
            except Exception:
                # 出错时退化为 L1；稳健处理潜在的形状/长度不匹配
                if mel_hat_n.shape == mel_n.shape:
                    loss_mel_struct = F.l1_loss(mel_hat_n, mel_n)
                else:
                    a = mel_hat_n.reshape(-1)
                    b = mel_n.reshape(-1)
                    n = min(a.numel(), b.numel())
                    loss_mel_struct = F.l1_loss(a[:n], b[:n])
        else:
            # 维度不符合预期，使用 L1 替代
            if mel_n.shape == mel_hat_n.shape:
                loss_mel_struct = F.l1_loss(mel_hat_n, mel_n)
            else:
                # 形状不匹配，展平后按最小长度计算
                a = mel_hat_n.reshape(-1)
                b = mel_n.reshape(-1)
                n = min(a.numel(), b.numel())
                loss_mel_struct = F.l1_loss(a[:n], b[:n])
        lam_mel = float(getattr(cfg, 'lambda_mel', 0.0))
        if lam_mel > 0.0:
            total = total + lam_mel * loss_mel_struct
            loss_dict["mel_ms_ssim"] = float(loss_mel_struct.item())
            loss_tensors['mel_ms_ssim'] = (loss_mel_struct, 'mel')
        # 可选：并行加入 mel L1，稳定亮度/细节
        lam_mel_l1 = float(getattr(cfg, 'lambda_mel_l1', 0.0))
        if lam_mel_l1 > 0.0:
            mel_l1_ref = mel
            mel_l1_hat = mel_hat_for_loss
            # 再次对齐 [B,T,F]，防御性处理潜在的边界差异
            if mel_l1_ref.shape != mel_l1_hat.shape:
                try:
                    Bx = min(mel_l1_ref.size(0), mel_l1_hat.size(0))
                    Tx = min(mel_l1_ref.size(1), mel_l1_hat.size(1))
                    Fx = min(mel_l1_ref.size(2), mel_l1_hat.size(2))
                    mel_l1_ref = mel_l1_ref[:Bx, :Tx, :Fx]
                    mel_l1_hat = mel_l1_hat[:Bx, :Tx, :Fx]
                except Exception:
                    pass
            if mel_l1_ref.shape == mel_l1_hat.shape:
                l_mel_l1 = torch.mean(torch.abs(mel_l1_hat - mel_l1_ref))
            else:
                # 仍不匹配时，退化为按最小长度的一维 L1
                a = mel_l1_hat.reshape(-1)
                b = mel_l1_ref.reshape(-1)
                n = min(a.numel(), b.numel())
                l_mel_l1 = F.l1_loss(a[:n], b[:n])

            total = total + lam_mel_l1 * l_mel_l1
            loss_dict['mel_l1'] = float(l_mel_l1.item())
            loss_tensors['mel_l1'] = (l_mel_l1, 'mel')

        # 1.6) 极轻的 mel 能量/亮度锚点（在 logmel 空间的全局均值）
        try:
            lam_energy = float(getattr(cfg, 'lambda_mel_energy', 0.0))
            if lam_energy > 0.0:
                mel_mean_hat = mel_hat_for_loss.mean(dim=(1, 2))  # [B]
                mel_mean = mel.mean(dim=(1, 2))          # [B]
                loss_mel_energy = torch.mean(torch.abs(mel_mean_hat - mel_mean))
                total = total + lam_energy * loss_mel_energy
                loss_dict["mel_energy"] = float(loss_mel_energy.item())
                loss_tensors['mel_energy'] = (loss_mel_energy, 'mel')
        except Exception:
            pass

        # 1.6.1) 逐帧亮度曲线对齐：对齐每一帧在频向的均值（仅非静音帧）
        try:
            lam_energy_t = float(getattr(cfg, 'lambda_mel_energy_t', 0.0))
            if lam_energy_t > 0.0:
                mel_e_tgt = mel.mean(dim=2)                 # [B,T]
                mel_e_hat = mel_hat_for_loss.mean(dim=2)    # [B,T]

                if silence_mask is not None:
                    Bm, Tm = mel_e_hat.shape
                    Ts = silence_mask.size(1)
                    T_use = min(Tm, Ts)
                    mel_e_tgt = mel_e_tgt[:, :T_use]
                    mel_e_hat = mel_e_hat[:, :T_use]
                    ns = (~silence_mask[:, :T_use]).to(mel_e_hat.dtype)  # 1=非静音
                else:
                    ns = torch.ones_like(mel_e_hat)

                loss_mel_energy_t = (torch.abs(mel_e_hat - mel_e_tgt) * ns).sum() / (ns.sum() + 1e-6)
                total = total + lam_energy_t * loss_mel_energy_t
                loss_dict["mel_energy_t"] = float(loss_mel_energy_t.item())
                loss_tensors['mel_energy_t'] = (loss_mel_energy_t, 'mel')
        except Exception:
            pass

        # 1.6.1.b) BFCC 图像域能量/纹理约束（对 32×T Bark log 能量图直接施加），
        # 仿照 training/bfcc_jscc_cnn_baseline.py 中的 lambda_energy_t / lambda_energy_f / lambda_tex_t。
        # 这些项默认关闭，仅在显式设置对应 lambda_* > 0 时启用。
        # 注意：此处 mel 即 BFCC（WaveToBFCC 的 Bark log 能量输出）。
        try:
            lam_bfcc_et = float(getattr(cfg, 'lambda_energy_t', 0.0))
            if lam_bfcc_et > 0.0:
                e_t = mel.mean(dim=2)                # [B,T]
                e_hat_t = mel_hat_for_loss.mean(dim=2)
                loss_energy_t = torch.mean(torch.abs(e_hat_t - e_t))
                total = total + lam_bfcc_et * loss_energy_t
                loss_dict["bfcc_energy_t"] = float(loss_energy_t.item())
                loss_tensors['bfcc_energy_t'] = (loss_energy_t, 'mel')
        except Exception:
            pass

        try:
            lam_bfcc_ef = float(getattr(cfg, 'lambda_energy_f', 0.0))
            if lam_bfcc_ef > 0.0:
                # 每频带时间均值的相对误差：|E_hat(f)-E(f)| / (|E(f)|+eps)
                e_f = mel.mean(dim=1)                # [B,F]
                e_hat_f = mel_hat_for_loss.mean(dim=1)
                denom = torch.clamp(e_f.abs(), min=1e-4)
                rel_err = torch.abs(e_hat_f - e_f) / denom
                loss_energy_f = rel_err.mean()
                total = total + lam_bfcc_ef * loss_energy_f
                loss_dict["bfcc_energy_f"] = float(loss_energy_f.item())
                loss_tensors['bfcc_energy_f'] = (loss_energy_f, 'mel')
        except Exception:
            pass

        try:
            lam_bfcc_tex = float(getattr(cfg, 'lambda_tex_t', 0.0))
            if lam_bfcc_tex > 0.0 and mel.size(1) > 1:
                grad_t = mel[:, 1:, :] - mel[:, :-1, :]
                grad_hat_t = mel_hat_for_loss[:, 1:, :] - mel_hat_for_loss[:, :-1, :]
                loss_tex_t = torch.mean(torch.abs(grad_hat_t - grad_t))
                total = total + lam_bfcc_tex * loss_tex_t
                loss_dict["bfcc_tex_t"] = float(loss_tex_t.item())
                loss_tensors['bfcc_tex_t'] = (loss_tex_t, 'mel')
        except Exception:
            pass

        # 1.6.1.c) mel Δ/ΔΔ 动态约束：在时间维上对齐一阶 / 二阶差分，
        # 减少音节边界被过度抹平或错位（主要作用于整体清晰度与节奏）。
        try:
            lam_mel_delta = float(getattr(cfg, 'lambda_mel_delta', 0.0))
            if lam_mel_delta > 0.0 and mel_hat_for_loss.size(1) > 2:
                md_hat = mel_hat_for_loss
                md_ref = mel
                Tm_d = min(md_hat.size(1), md_ref.size(1))
                Fm_d = min(md_hat.size(2), md_ref.size(2))
                md_hat = md_hat[:, :Tm_d, :Fm_d]
                md_ref = md_ref[:, :Tm_d, :Fm_d]

                d1_hat = md_hat[:, 1:, :] - md_hat[:, :-1, :]
                d1_ref = md_ref[:, 1:, :] - md_ref[:, :-1, :]
                d2_hat = d1_hat[:, 1:, :] - d1_hat[:, :-1, :]
                d2_ref = d1_ref[:, 1:, :] - d1_ref[:, :-1, :]

                loss_mel_delta = torch.nn.functional.l1_loss(d1_hat, d1_ref) + \
                    torch.nn.functional.l1_loss(d2_hat, d2_ref)

                total = total + lam_mel_delta * loss_mel_delta
                loss_dict["mel_delta"] = float(loss_mel_delta.item())
                loss_tensors['mel_delta'] = (loss_mel_delta, 'mel')
        except Exception:
            pass

        # 1.6.1.a) 静音帧上的高频能量正则：
        # 在 silence_mask==True 且频率>hf_low 的区域，
        # 惩罚 mel_hat_for_loss 与 mel 之间的能量差，
        # 专门抑制静音段的高频均匀底噪，而不影响有声段纹理。
        try:
            if lam_sil_mel > 0.0 and silence_mask is not None:
                Bm, Tm, Fm = mel_hat_for_loss.shape

                hf_low = int(getattr(cfg, 'silence_hf_low_bins', int(getattr(cfg, 'mel_hp_low_bins', 16))))
                hf_low = max(0, min(hf_low, Fm - 1))

                mel_hat_hf = mel_hat_for_loss[:, :, hf_low:] if hf_low < Fm else mel_hat_for_loss
                mel_ref_hf = mel[:, :, hf_low:] if hf_low < Fm else mel

                Ts = silence_mask.size(1)
                T_use = min(Tm, Ts)
                if T_use > 0:
                    sm = silence_mask[:, :T_use].to(mel_hat_hf.dtype)  # [B,T_use]
                    mel_hat_hf = mel_hat_hf[:, :T_use, :]
                    mel_ref_hf = mel_ref_hf[:, :T_use, :]

                    diff = torch.abs(mel_hat_hf - mel_ref_hf) * sm.unsqueeze(-1)
                    denom = sm.sum() * max(mel_hat_hf.size(-1), 1) + 1e-6
                    loss_sil_mel = diff.sum() / denom
                    total = total + lam_sil_mel * loss_sil_mel
                    loss_dict["sil_mel_hf"] = float(loss_sil_mel.item())
                    loss_tensors['sil_mel_hf'] = (loss_sil_mel, 'mel')
        except Exception:
            pass

        # 1.6.2) 每帧对比度锚点：对齐频向标准差（仅非静音帧）
        try:
            lam_contrast = float(getattr(cfg, 'lambda_mel_contrast', 0.0))
            if lam_contrast > 0.0 and mel_hat_for_loss.size(-1) > 1:
                mel_c_tgt = mel.std(dim=2)                 # [B,T]
                mel_c_hat = mel_hat_for_loss.std(dim=2)    # [B,T]

                if silence_mask is not None:
                    Bm, Tm = mel_c_hat.shape
                    Ts = silence_mask.size(1)
                    T_use = min(Tm, Ts)
                    mel_c_tgt = mel_c_tgt[:, :T_use]
                    mel_c_hat = mel_c_hat[:, :T_use]
                    ns = (~silence_mask[:, :T_use]).to(mel_c_hat.dtype)
                else:
                    ns = torch.ones_like(mel_c_hat)

                loss_mel_contrast = (torch.abs(mel_c_hat - mel_c_tgt) * ns).sum() / (ns.sum() + 1e-6)
                total = total + lam_contrast * loss_mel_contrast
                loss_dict["mel_contrast"] = float(loss_mel_contrast.item())
                loss_tensors['mel_contrast'] = (loss_mel_contrast, 'mel')
        except Exception:
            pass

        # 1.6.2.a) 频域谷值钳制（Valley clamp）：
        # 在非静音帧中，使用 GT mel 的频向分位数划出“谷区”，
        # 对这些谷区 bin 上预测 mel 的过高能量进行 hinge 惩罚，
        # 防止有声段频谱谷被均匀底噪填平。
        try:
            lam_valley = float(getattr(cfg, 'lambda_mel_valley', 0.0))
            if lam_valley > 0.0 and mel_hat_for_loss.size(-1) > 1:
                Bm, Tm, Fm = mel_hat_for_loss.shape

                # 对齐时间长度与 silence_mask
                if silence_mask is not None:
                    Ts = silence_mask.size(1)
                    T_use = min(Tm, Ts)
                    if T_use <= 0:
                        raise RuntimeError("T_use <= 0 in mel_valley")
                    mel_tgt = mel[:, :T_use, :]
                    mel_hat_v = mel_hat_for_loss[:, :T_use, :]
                    ns = (~silence_mask[:, :T_use]).to(mel_hat_v.dtype)  # 1=非静音
                else:
                    mel_tgt = mel
                    mel_hat_v = mel_hat_for_loss
                    ns = torch.ones(Bm, Tm, device=mel_hat_v.device, dtype=mel_hat_v.dtype)

                # 使用每帧 GT mel 的高分位数作为“峰/谷”阈值
                q = 0.8
                tau = torch.quantile(mel_tgt.detach(), q, dim=2, keepdim=True)  # [B,T,1]

                # 谷区：GT 能量低于 tau 的频点
                valley_mask = (mel_tgt <= tau).to(mel_hat_v.dtype)  # [B,T,F]

                # 仅在非静音帧上生效
                valley_mask = valley_mask * ns.unsqueeze(-1)

                # 允许在谷区上有一个小的 margin（log10 域），避免过度压制
                margin = 0.3  # ≈ +5 dB
                allowed = mel_tgt + margin
                violation = torch.relu(mel_hat_v - allowed)  # 只惩罚预测过高的部分

                num = (violation * valley_mask).sum()
                denom = valley_mask.sum() + 1e-6
                if denom > 0:
                    loss_valley = num / denom
                    total = total + lam_valley * loss_valley
                    loss_dict["mel_valley"] = float(loss_valley.item())
                    loss_tensors['mel_valley'] = (loss_valley, 'mel')
        except Exception:
            pass

        # 1.6.3) 分频带亮度锚点：低/中/高三个 band 的均值（仅非静音帧）
        try:
            lam_bandE = float(getattr(cfg, 'lambda_mel_bandE', 0.0))
            if lam_bandE > 0.0:
                Bm, Tm, Fm = mel_hat_for_loss.shape

                # 默认在 32-bin mel 上使用 [0,10), [10,20), [20,32) 三段，
                # 对其它 Fm 做安全裁剪。
                edges = [0, 10, 20, 32]
                edges = [max(0, min(e, Fm)) for e in edges]

                def _band_mean(x: torch.Tensor, s: int, e: int) -> Optional[torch.Tensor]:
                    if e <= s:
                        return None
                    return x[:, :, s:e].mean(dim=2)  # [B,T]

                band_diffs: List[torch.Tensor] = []

                for (s, e) in zip(edges[:-1], edges[1:]):
                    tgt_band = _band_mean(mel, s, e)
                    hat_band = _band_mean(mel_hat_for_loss, s, e)
                    if tgt_band is None or hat_band is None:
                        continue

                    if silence_mask is not None:
                        Ts = silence_mask.size(1)
                        T_use = min(tgt_band.size(1), Ts)
                        tgt_b = tgt_band[:, :T_use]
                        hat_b = hat_band[:, :T_use]
                    else:
                        tgt_b = tgt_band
                        hat_b = hat_band

                    band_diffs.append(torch.abs(hat_b - tgt_b))  # [B,T_use]

                if band_diffs:
                    diff_sum = torch.zeros_like(band_diffs[0])
                    for d in band_diffs:
                        # 对齐时间长度（理论上应一致，这里防御性裁剪）
                        T_ref = diff_sum.size(1)
                        T_cur = d.size(1)
                        T_use = min(T_ref, T_cur)
                        diff_sum = diff_sum[:, :T_use] + d[:, :T_use]

                    if silence_mask is not None:
                        Ts = silence_mask.size(1)
                        T_use = min(diff_sum.size(1), Ts)
                        ns = (~silence_mask[:, :T_use]).to(diff_sum.dtype)
                        diff_use = diff_sum[:, :T_use]
                    else:
                        ns = torch.ones_like(diff_sum)
                        diff_use = diff_sum

                    loss_bandE = (diff_use * ns).sum() / (ns.sum() + 1e-6)
                    total = total + lam_bandE * loss_bandE
                    loss_dict["mel_bandE"] = float(loss_bandE.item())
                    loss_tensors['mel_bandE'] = (loss_bandE, 'mel')
        except Exception:
            pass

        # 1.6.4) Frequency-aware loss: DeCo-inspired loss on prediction residual
        # Note: 1D freq-block DCT (not 2D spatial 8x8 like DeCo images)
        try:
            lam_freq_mel = float(getattr(cfg, 'lambda_freq_aware_mel', 0.0))
            jpeg_quality = int(getattr(cfg, 'jpeg_quality_factor', 85))
            if lam_freq_mel > 0.0:
                Bm, Tm, Fm = mel_hat_for_loss.shape

                # Compute velocity analog: prediction residual (v ≈ mel_hat - mel_target)
                # DeCo: v_t = x_1 - x_0 (flow-matching residual)
                residual = mel_hat_for_loss - mel  # [B,T,F]

                # 1D frequency-block DCT (8-coeff blocks along freq axis)
                block_size = 8
                F_pad = ((Fm + block_size - 1) // block_size) * block_size
                residual_pad = F.pad(residual, (0, F_pad - Fm))

                # Reshape to blocks: [B, T, num_blocks, block_size]
                num_blocks = F_pad // block_size
                residual_blocks = residual_pad.view(Bm, Tm, num_blocks, block_size)

                # Build DCT-II matrix
                def _build_dct_mat(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
                    n_idx = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)
                    k_idx = torch.arange(n, device=device, dtype=dtype).unsqueeze(0)
                    mat = torch.cos((n_idx + 0.5) * k_idx * math.pi / float(n))
                    mat[:, 0] = mat[:, 0] * math.sqrt(0.5)
                    mat = mat * math.sqrt(2.0 / float(n))
                    return mat

                dct_mat = _build_dct_mat(block_size, residual.device, residual.dtype)  # [8,8]

                # Apply block-wise DCT: [B,T,num_blocks,8] @ [8,8]
                residual_dct = torch.matmul(residual_blocks, dct_mat)

                # JPEG quantization table weights with quality scaling (DeCo Eq.11)
                # Q_cur = max(1, floor((Q_base * (100 - q) + 25) / 50))
                # Using first row of standard JPEG luminance table
                Q_base_8 = torch.tensor([
                    16.0, 11.0, 10.0, 16.0, 24.0, 40.0, 51.0, 61.0
                ], device=residual.device, dtype=residual.dtype)

                # Scale by quality factor
                q = float(max(1, min(100, jpeg_quality)))
                scale = max(1.0, (Q_base_8[0].item() * (100.0 - q) + 25.0) / 50.0)
                Q_cur = torch.clamp(Q_base_8 * scale / Q_base_8[0].item(), min=1.0)

                weights = 1.0 / (Q_cur + 1e-6)  # Reciprocal: emphasize perceptually important freqs
                weights = weights / (weights.mean() + 1e-6)  # Normalize

                # Weighted MSE in frequency domain
                diff2 = residual_dct ** 2
                loss_freq = (diff2 * weights.view(1, 1, 1, -1)).mean()
                total = total + lam_freq_mel * loss_freq
                loss_dict['mel_freq'] = float(loss_freq.item())
                loss_tensors['mel_freq'] = (loss_freq, 'mel')
        except Exception:
            pass

        # 1.7) 高频 patch SSIM（替代 mel_hp 频向 Laplacian）
        # 仅在高频区域（mel_hp_low_bins 以上）对局部 patch 结构做 MS-SSIM，
        # 避免直接在频域二阶差分上施加强平滑约束。
        try:
            lam_hp = float(getattr(cfg, 'lambda_mel_hp', 0.0))
            if lam_hp > 0.0:
                Bm, Tm, Fm = mel_hat_for_loss.shape
                hf_start = int(getattr(cfg, 'mel_hp_low_bins', 16))
                hf_start = max(4, min(hf_start, Fm - 4))  # 至少保留若干低频 bin
                F_hf = Fm - hf_start
                if F_hf >= 4 and Tm >= 4:
                    # 仅取高频区域
                    mel_h     = mel[:, :Tm, hf_start:]              # [B,T,F_hf]
                    mel_hat_h = mel_hat_for_loss[:, :Tm, hf_start:]

                    # 对静音帧做 gating：将静音帧在高频区域置零，避免在静音段拉出纹理
                    if silence_mask is not None:
                        Tf = silence_mask.size(1)
                        T_use = min(Tm, Tf)
                        ns_mask = (~silence_mask[:, :T_use]).to(mel_h.dtype)  # 1=非静音
                        mel_h = mel_h[:, :T_use, :] * ns_mask.unsqueeze(-1)
                        mel_hat_h = mel_hat_h[:, :T_use, :] * ns_mask.unsqueeze(-1)

                    # 映射到 [0,1]，再视作图像做 MS-SSIM
                    mel_h_n     = _soft_unit(mel_h)
                    mel_hat_h_n = _soft_unit(mel_hat_h)
                    mel_h_img     = mel_h_n.unsqueeze(1)     # [B,1,T,F_hf]
                    mel_hat_h_img = mel_hat_h_n.unsqueeze(1)

                    # 复用全频 MS_SSIM 实例，对高频 patch 区域做结构一致性度量
                    loss_hf_ssim = ms_ssim(mel_hat_h_img, mel_h_img).mean()
                    total = total + lam_hp * loss_hf_ssim
                    loss_dict['mel_hp'] = float(loss_hf_ssim.item())
                    loss_tensors['mel_hp'] = (loss_hf_ssim, 'mel')
        except Exception:
            pass

        # 1.6.5) HF Time-Edge + 边界保护（最小改动版）：只在高频 Mel 区域做时间差分 Δt，
        # 并对“有声/静音边界 + 无声高能帧”加权，专门拉齐爆破/摩擦的竖向虚线高度。
        try:
            lam_hf_edge = float(getattr(cfg, "lambda_hf_time_edge", 0.0))
            if lam_hf_edge > 0.0 and mel_hat_for_loss is not None:
                eps = 1e-6
                hf_start = int(getattr(cfg, "hf_time_edge_start", 32))
                ref_thr = float(getattr(cfg, "hf_time_edge_ref_thr", 0.03))
                boundary_boost = float(getattr(cfg, "hf_time_edge_boundary_boost", 2.0))
                w_clip = float(getattr(cfg, "hf_time_edge_weight_clip", 5.0))

                Bm, Tm, Fm = mel_hat_for_loss.shape
                # 至少保留 2 个高频 bin，否则直接跳过（避免 cfg_start 设在最顶端导致 F_hf 太小）
                hf_start = max(0, min(hf_start, Fm - 2))
                F_hf = max(0, Fm - hf_start)
                T = min(Tm, mel.size(1))
                if T >= 3 and F_hf >= 2:
                    mh = mel_hat_for_loss[:, :T, hf_start:]  # [B,T,Fh]
                    mr = mel[:, :T, hf_start:]

                    # 基于 silence_mask + VUV 构建 non-silence 与边界 mask
                    ns = None
                    b_sil = None
                    if silence_mask is not None:
                        sm = silence_mask[:, :T].float()          # 1=silence
                        ns = (1.0 - sm)                           # 1=non-silence
                        b_sil = (ns[:, 1:] - ns[:, :-1]).abs()    # [B,T-1]
                        b_sil = F.pad((b_sil > 0).float(), (1, 0))  # -> [B,T]

                    # VUV 边界：使用 frame_corr ∪ frame_corr_hat 的 OR 结果
                    b_vuv = None
                    try:
                        fc_hat = out.get('frame_corr_hat', None)
                        fc_ref = out.get('frame_corr', None)
                        th_vuv = float(getattr(cfg, 'vuv_threshold', 0.3))

                        vm_ref = (fc_ref[:, :T, 0] > th_vuv) if isinstance(fc_ref, torch.Tensor) else None
                        vm_hat = (fc_hat[:, :T, 0] > th_vuv) if isinstance(fc_hat, torch.Tensor) else None
                        if vm_ref is None and vm_hat is None:
                            vm_mix = None
                        elif vm_ref is None:
                            vm_mix = vm_hat
                        elif vm_hat is None:
                            vm_mix = vm_ref
                        else:
                            vm_mix = (vm_ref | vm_hat)
                        if vm_mix is not None:
                            b_vuv = torch.zeros_like(vm_mix, dtype=torch.float32)
                            if vm_mix.size(1) > 1:
                                b_vuv[:, 1:] = (vm_mix[:, 1:] != vm_mix[:, :-1]).float()
                    except Exception:
                        b_vuv = None

                    if b_sil is not None and b_vuv is not None:
                        b = torch.clamp(b_sil + b_vuv, max=1.0)
                    elif b_sil is not None:
                        b = b_sil
                    else:
                        b = b_vuv

                    # HF 时间差分
                    d_mh = mh[:, 1:] - mh[:, :-1]                 # [B,T-1,Fh]
                    d_mr = mr[:, 1:] - mr[:, :-1]

                    # weight: 参考 HF 边缘强度
                    ref_mag = d_mr.abs().mean(dim=-1)             # [B,T-1]
                    w = (ref_mag / (ref_thr + eps)).clamp(0.0, w_clip)

                    # 边界增强 + 深静音抑制（soft gate）
                    if b is not None:
                        wb = b[:, 1:]                             # [B,T-1]
                        w = w * (1.0 + (boundary_boost - 1.0) * wb)
                    if ns is not None:
                        # 将 hard 0/1 non-silence 转为 soft gate，避免边界帧被完全抹掉梯度
                        ns_soft = 0.2 + 0.8 * ns                   # [B,T], 0.2~1.0
                        w = w * ns_soft[:, 1:]

                    edge_err = (d_mh - d_mr).abs().mean(dim=-1)  # [B,T-1]
                    loss_hf_time_edge = (edge_err * w).sum() / (w.sum() + eps)

                    # 额外：在边界帧上匹配 HF 幅度（避免用极小幅度“骗过”Δt）
                    if b is not None:
                        hf_mag_err = (mh - mr).abs().mean(dim=-1)    # [B,T]
                        if ns is not None:
                            ns_soft = 0.2 + 0.8 * ns
                            wE = b * ns_soft
                        else:
                            wE = b
                        if wE.sum() > 1.0:
                            loss_hf_time_edge = loss_hf_time_edge + 0.25 * (
                                (hf_mag_err * wE).sum() / (wE.sum() + eps)
                            )

                    total = total + lam_hf_edge * loss_hf_time_edge
                    loss_dict["hf_time_edge"] = float(loss_hf_time_edge.item())
                    loss_tensors["hf_time_edge"] = (loss_hf_time_edge, "mel")
        except Exception:
            pass

        # 1.6.6) 边界高频倾斜（HF tilt at boundaries）：
        # 在边界/无声帧上对齐高频 vs 低频的能量差，直接拉齐频谱“重心”，
        # 并可选地施加一个“硬推高” margin，把高频能量主动抬到比低频更亮的位置，
        # 即便 GT 相对保守，也能在可视化上明显抬高边界摩擦区域的高频能量。
        try:
            lam_tilt = float(getattr(cfg, 'lambda_hf_tilt', 0.0))
            if lam_tilt > 0.0:
                Bm, Tm, Fm = mel_hat_for_loss.shape
                split = int(getattr(cfg, 'hf_tilt_split_bin', int(getattr(cfg, 'mel_hp_low_bins', 16))))
                split = max(1, min(split, Fm - 1))

                # 低频 / 高频 band 均值
                low_tgt  = mel[:, :Tm, :split].mean(dim=2)          # [B,T]
                high_tgt = mel[:, :Tm, split:].mean(dim=2)
                low_hat  = mel_hat_for_loss[:, :Tm, :split].mean(dim=2)
                high_hat = mel_hat_for_loss[:, :Tm, split:].mean(dim=2)

                tilt_tgt = high_tgt - low_tgt   # 频谱“抬高”程度
                tilt_hat = high_hat - low_hat

                # 边界 + 无声帧 mask：
                #  - 无声：当前 voiced_mask 为 True → 有声，所以无声 = ~voiced
                #  - 边界：voiced 发生切换的帧
                if voiced_mask is not None and voiced_mask.dtype == torch.bool:
                    vm = voiced_mask[:, :Tm]
                    unv = (~vm).to(low_tgt.dtype)         # 无声
                    bd = torch.zeros_like(unv)
                    if vm.size(1) > 1:
                        bd[:, 1:] = (vm[:, 1:] != vm[:, :-1]).to(unv.dtype)
                    mask = torch.clamp(unv + bd, max=1.0)  # [B,T]
                else:
                    mask = torch.ones_like(low_tgt)

                # 为避免极少边界导致数值不稳定，若覆盖太少则跳过
                if mask.sum() > 1.0:
                    diff_tilt = (tilt_hat - tilt_tgt).abs()  # [B,T]
                    loss_tilt = (diff_tilt * mask).sum() / (mask.sum() + 1e-6)

                    # 可选：在边界/无声帧上施加额外“硬推高”约束，
                    # 要求 high_hat 至少比 low_hat 高出 margin（log-mel），
                    # margin≈0.5 对应约 5dB，1.0 对应约 10dB。
                    extra_push = float(getattr(cfg, 'hf_tilt_extra_push', 0.0))
                    if extra_push > 0.0:
                        push = torch.relu((low_hat + extra_push) - high_hat)  # [B,T]
                        loss_push = (push * mask).sum() / (mask.sum() + 1e-6)
                        loss_tilt = loss_tilt + loss_push

                    total = total + lam_tilt * loss_tilt
                    loss_dict['hf_tilt'] = float(loss_tilt.item())
                    loss_tensors['hf_tilt'] = (loss_tilt, 'mel')
        except Exception:
            pass

        # 保留旧的“高bin mel”先验与细化项开关，如果 CLI/配置未显式赋值将默认保持为 0。

        # 1.7) Mel 低频引导的高频先验（频带能量衰减 + 时间调制一致性）
        # 说明：在 log-mel 域中，每 1.0 约等于 10 dB。
        try:
            lam_band = float(getattr(cfg, 'lambda_mel_band_prior', 0.0))
            lam_mod  = float(getattr(cfg, 'lambda_mel_modulation', 0.0))
            if (lam_band > 0.0) or (lam_mod > 0.0):
                B, Tm, Fm = mel_hat_for_loss.shape
                low_bins = int(getattr(cfg, 'mel_low_bins', 10))
                low_bins = max(1, min(low_bins, Fm))

                # 频带边界（默认：在 32mel 情况下分三段）
                edges = getattr(cfg, 'mel_band_edges', None)
                if not edges:
                    # 以 32mel 为例的默认切分；对其它 Fm 做近似缩放
                    # [10,16,24,32] → 去掉第一个（低频），只保高频段上界
                    scale = Fm / 32.0
                    edges = [int(round(x * scale)) for x in [16, 24, 32]]
                # 目标上/下界（相对低频均值，负值表示衰减）
                decay_up = getattr(cfg, 'mel_decay_upper', None) or [-0.3, -0.6, -0.9]
                decay_lo = getattr(cfg, 'mel_decay_lower', None) or [-1.2, -1.8, -2.4]
                betas = getattr(cfg, 'mel_mod_betas', None) or [0.8, 0.6, 0.4]
                # 对齐长度
                if len(decay_up) < len(edges):
                    decay_up = decay_up + [decay_up[-1]] * (len(edges) - len(decay_up))
                if len(decay_lo) < len(edges):
                    decay_lo = decay_lo + [decay_lo[-1]] * (len(edges) - len(decay_lo))
                if len(betas) < len(edges):
                    betas = betas + [betas[-1]] * (len(edges) - len(betas))

                # 有声门控：使用 target ∪ predicted（更稳），或全时段
                if bool(getattr(cfg, 'mel_prior_voiced_only', True)):
                    fc_hat = out.get('frame_corr_hat', None)
                    fc = out.get('frame_corr', None)
                    th = float(getattr(cfg, 'vuv_threshold', 0.3))
                    if bool(getattr(cfg, 'mel_prior_soft_vuv', False)):
                        # 软有声权重：sigmoid(k*(x-th))，优先用预测，其次目标，最后取并集的最大值
                        ksig = float(getattr(cfg, 'mel_prior_vuv_k', 10.0))
                        def _soft_w(x: torch.Tensor) -> torch.Tensor:
                            return torch.sigmoid(ksig * (x.squeeze(-1) - th))
                        m_pred = _soft_w(fc_hat) if isinstance(fc_hat, torch.Tensor) else None
                        m_tgt  = _soft_w(fc)     if isinstance(fc, torch.Tensor)     else None
                        if m_pred is None and m_tgt is None:
                            voiced_mask = torch.ones(B, Tm, dtype=mel_hat.dtype, device=mel_hat.device)
                        elif m_pred is None:
                            voiced_mask = m_tgt.to(mel_hat.dtype)
                        elif m_tgt is None:
                            voiced_mask = m_pred.to(mel_hat.dtype)
                        else:
                            voiced_mask = torch.maximum(m_pred, m_tgt).to(mel_hat.dtype)
                    else:
                        m_pred = (fc_hat > th).squeeze(-1) if isinstance(fc_hat, torch.Tensor) else None
                        m_tgt  = (fc > th).squeeze(-1) if isinstance(fc, torch.Tensor) else None
                        if m_pred is None and m_tgt is None:
                            voiced_mask = torch.ones(B, Tm, dtype=mel_hat.dtype, device=mel_hat.device)
                        elif m_pred is None:
                            voiced_mask = m_tgt.to(mel_hat.dtype)
                        elif m_tgt is None:
                            voiced_mask = m_pred.to(mel_hat.dtype)
                        else:
                            voiced_mask = (m_pred | m_tgt).to(mel_hat.dtype)
                else:
                    voiced_mask = torch.ones(B, Tm, dtype=mel_hat.dtype, device=mel_hat.device)

                # 计算各频带能量（log-mel 平均）
                def band_mean(x: torch.Tensor, s: int, e: int) -> torch.Tensor:
                    e = max(min(e, Fm), s + 1)
                    return x[:, :, s:e].mean(dim=2)

                # 关键修正：用目标低频作为锚点，而非预测低频。
                # 这样 band_prior/modulation 会对“偏离目标低频包络/起伏”的高频产生稳定非零梯度，
                # 而不是让预测低频一起漂移时损失为0。
                low = band_mean(mel, 0, low_bins)  # [B,T] 使用 GT 低频作为参照
                prev = low_bins
                band_losses = []
                mod_losses = []
                band_active_masks: List[torch.Tensor] = []  # [B,T] per segment
                mod_active_masks: List[torch.Tensor] = []   # [B,T] per segment
                for j, hi in enumerate(edges):
                    if hi <= prev:
                        continue
                    band = band_mean(mel_hat_for_loss, prev, hi)  # [B,T]
                    # 频带能量区间先验：
                    #   low + decay_lo[j] <= band <= low + decay_up[j]
                    up_target = low + float(decay_up[j])
                    lo_target = low + float(decay_lo[j])
                    # 软边界：在阈值附近提前进入罚区，增加边界附近梯度
                    margin = float(getattr(cfg, 'mel_band_margin', 0.0))
                    up_t = up_target - margin
                    lo_t = lo_target + margin
                    # 上越界/下越界：使用 softplus 替代 ReLU，避免硬死区
                    if lam_band > 0.0:
                        over = F.softplus(band - up_t, beta=5.0)
                        under = F.softplus(lo_t - band, beta=5.0)
                        w = voiced_mask
                        denom = w.sum() + 1e-6
                        band_loss = ((over + under) * w).sum() / denom
                        # 带宽/覆盖率补偿：抵消 1/带宽 与 覆盖率 稀释
                        bw = max(1, int(hi - prev))
                        bw_alpha = float(getattr(cfg, 'mel_prior_bw_alpha', 0.0))
                        cov_alpha = float(getattr(cfg, 'mel_prior_cov_alpha', 0.0))
                        if bw_alpha > 0.0:
                            band_loss = band_loss * (float(bw) ** bw_alpha)
                        if cov_alpha > 0.0:
                            coverage = (w.sum() / (B * Tm + 1e-6)).clamp_min(1e-6)
                            band_loss = band_loss * ((1.0 / coverage) ** cov_alpha)
                        band_losses.append(band_loss)
                        # 记录越界区域（仅用于 gn 观测）
                        band_active = ((band > up_t) | (band < lo_t)).to(band.dtype) * w
                        band_active_masks.append(band_active)

                    # 调制一致性：Δband ≈ β·Δlow（与 GT 低频起伏对齐）
                    if lam_mod > 0.0 and Tm > 1:
                        d_low = low[:, 1:] - low[:, :-1]
                        d_hi = band[:, 1:] - band[:, :-1]
                        # 归一化，避免幅度尺度差异
                        eps = 1e-3
                        d_low_n = d_low / (d_low.abs().mean() + eps)
                        d_hi_n = d_hi / (d_hi.abs().mean() + eps)
                        beta = float(betas[j])
                        err = (d_hi_n - beta * d_low_n).abs()
                        w2 = voiced_mask[:, 1:]
                        denom2 = w2.sum() + 1e-6
                        lmod = (err * w2).sum() / denom2
                        # 同步带宽/覆盖率补偿
                        bw = max(1, int(hi - prev))
                        bw_alpha = float(getattr(cfg, 'mel_prior_bw_alpha', 0.0))
                        cov_alpha = float(getattr(cfg, 'mel_prior_cov_alpha', 0.0))
                        if bw_alpha > 0.0:
                            lmod = lmod * (float(bw) ** bw_alpha)
                        if cov_alpha > 0.0:
                            coverage2 = (w2.sum() / (B * max(1, Tm - 1))).clamp_min(1e-6)
                            lmod = lmod * ((1.0 / coverage2) ** cov_alpha)
                        mod_losses.append(lmod)
                        # 记录调制误差显著区域（仅用于 gn 观测）
                        thr_mod = 0.02
                        mod_active = (err > thr_mod).to(err.dtype) * w2
                        mod_mask_full = torch.zeros(B, Tm, dtype=err.dtype, device=err.device)
                        mod_mask_full[:, 1:] = mod_active
                        mod_active_masks.append(mod_mask_full)
                    prev = hi

                if lam_band > 0.0 and len(band_losses) > 0:
                    l_band = torch.stack(band_losses).mean()
                    total = total + lam_band * l_band
                    loss_dict['mel_band'] = float(l_band.item())
                    loss_tensors['mel_band'] = (l_band, 'mel')
                    # 区域化梯度观测：仅统计“越界帧 ∧ 高频”的 RMS-Grad
                    if bool(getattr(cfg, 'grad_survey', False)):
                        try:
                            g = autograd.grad(l_band, mel_hat_refined, retain_graph=True, allow_unused=True)[0]
                            if isinstance(g, torch.Tensor):
                                lb = int(getattr(cfg, 'mel_low_bins', 10))
                                lb = max(1, min(lb, mel_hat_refined.size(-1) - 1))
                                if len(band_active_masks) > 0:
                                    bmask = torch.stack(band_active_masks).max(dim=0).values
                                else:
                                    bmask = voiced_mask
                                M = bmask.detach().to(g.dtype).unsqueeze(-1)
                                gh = torch.nan_to_num(g[:, :, lb:], nan=0.0, posinf=0.0, neginf=0.0)
                                num = (gh.pow(2) * M).sum()
                                denomg = M.sum() * gh.size(-1) + 1e-6
                                grad_info['g_band'] = float(torch.sqrt(num / denomg).item())
                        except Exception:
                            pass
                if lam_mod > 0.0 and len(mod_losses) > 0:
                    l_mod = torch.stack(mod_losses).mean()
                    total = total + lam_mod * l_mod
                    loss_dict['mel_mod'] = float(l_mod.item())
                    loss_tensors['mel_mod'] = (l_mod, 'mel')
                    if bool(getattr(cfg, 'grad_survey', False)):
                        try:
                            g = autograd.grad(l_mod, mel_hat_refined, retain_graph=True, allow_unused=True)[0]
                            if isinstance(g, torch.Tensor):
                                lb = int(getattr(cfg, 'mel_low_bins', 10))
                                lb = max(1, min(lb, mel_hat_refined.size(-1) - 1))
                                if len(mod_active_masks) > 0:
                                    mmask = torch.stack(mod_active_masks).max(dim=0).values
                                else:
                                    mmask = voiced_mask
                                M = mmask.detach().to(g.dtype).unsqueeze(-1)
                                gh = torch.nan_to_num(g[:, :, lb:], nan=0.0, posinf=0.0, neginf=0.0)
                                num = (gh.pow(2) * M).sum()
                                denomg = M.sum() * gh.size(-1) + 1e-6
                                grad_info['g_mod'] = float(torch.sqrt(num / denomg).item())
                        except Exception:
                            pass
        except Exception as _e:
            # 不影响训练主流程
            if os.environ.get('DBG_STAGE25', '0') == '1':
                print('[mel_prior_debug]', _e)

        # # 1.8) 高频重建（Low→High 细化头的包络监督，仅作用高频 bins）
        # # 这里不再直接回归逐 bin 振幅，而是先在时间维做轻量平滑，
        # # 对齐高频亮度/包络，把纹理自由度留给噪声+HF 损失去“雕刻”。
        # lam_l2h = float(getattr(cfg, 'lambda_l2h', 0.0))
        # # L2H 仅学习 GT 与 baseline mel_hat 之间的 envelope 残差：
        # # pred_res ≈ smooth(mel_hat_refined - mel_hat_base),
        # # tgt_res  ≈ smooth(mel_gt         - mel_hat_base)。
        # # 避免把高频绝对能量的硬重建压力都压给 L2H，让其更专注于补充 mid‑HF 纹理。
        # if lam_l2h > 0.0 and mel_hat_refined is not None and 'mel_hat' in out:
        #     try:
        #         low_bins = int(getattr(cfg, 'l2h_low_bins', 10))
        #         low_bins = max(1, min(low_bins, mel_hat_refined.size(-1) - 1))
        #         pred_h = mel_hat_refined[:, :, low_bins:]   # L2H 细化后的高频
        #         base_h = out['mel_hat'][:, :, low_bins:]     # baseline VMamba 高频
        #         tgt_h  = mel[:, :, low_bins:]                # GT 高频

        #         def _smooth_env(x: torch.Tensor, k: int = 5) -> torch.Tensor:
        #             if x.size(1) < 3 or k <= 1:
        #                 return x
        #             pad = k // 2
        #             # 在时间维做平均池化，相当于 1D 平滑核
        #             return F.avg_pool1d(
        #                 x.transpose(1, 2), kernel_size=k, stride=1, padding=pad
        #             ).transpose(1, 2)

        #         base_env = _smooth_env(base_h, k=5)
        #         pred_env = _smooth_env(pred_h, k=5)
        #         tgt_env  = _smooth_env(tgt_h,  k=5)

        #         # Residual envelopes: GT 与 baseline 之间的差异，
        #         # 以及 L2H 细化后相对 baseline 的差异。
        #         tgt_res  = tgt_env  - base_env
        #         pred_res = pred_env - base_env

        #         # 有声门控：target ∪ predicted（并集）
        #         fc_hat = out.get('frame_corr_hat', None)
        #         fc = out.get('frame_corr', None)
        #         th = float(getattr(cfg, 'vuv_threshold', 0.3))
        #         m_pred = (fc_hat > th).squeeze(-1) if isinstance(fc_hat, torch.Tensor) else None
        #         m_tgt = (fc > th).squeeze(-1) if isinstance(fc, torch.Tensor) else None
        #         if m_pred is None and m_tgt is None:
        #             mask = torch.ones(pred_env.size(0), pred_env.size(1), dtype=pred_env.dtype, device=pred_env.device)
        #         elif m_pred is None:
        #             mask = m_tgt.to(pred_env.dtype)
        #         elif m_tgt is None:
        #             mask = m_pred.to(pred_env.dtype)
        #         else:
        #             mask = (m_pred | m_tgt).to(pred_env.dtype)

        #         # 额外的能量门控：仅在非静音帧上监督高频 L2H，防止静音段被强行拉出高频结构。
        #         if silence_mask is not None:
        #             Tf = silence_mask.size(1)
        #             T_use = min(pred_env.size(1), Tf)
        #             ns_mask = (~silence_mask[:, :T_use]).to(mask.dtype)
        #             mask = mask[:, :T_use] * ns_mask
        #             pred_env = pred_env[:, :T_use, :]
        #             tgt_env = tgt_env[:, :T_use, :]
        #         else:
        #             T_use = pred_env.size(1)

        #         diff_env = torch.abs(pred_res - tgt_res)
        #         denom = mask.sum() * diff_env.size(-1) + 1e-6
        #         l_h = (mask.unsqueeze(-1) * diff_env).sum() / denom

        #         # 覆盖/带宽补偿（可选）
        #         l2h_cov_alpha = float(getattr(cfg, 'l2h_cov_alpha', 0.0))
        #         l2h_bw_alpha = float(getattr(cfg, 'l2h_bw_alpha', 0.0))
        #         if l2h_cov_alpha > 0.0:
        #             coverage_h = (mask.sum() / (mask.shape[0] * mask.shape[1] + 1e-6)).clamp_min(1e-6)
        #             l_h = l_h * ((1.0 / coverage_h) ** l2h_cov_alpha)
        #         if l2h_bw_alpha > 0.0:
        #             bw_h = float(pred_env.size(-1))
        #             l_h = l_h * (bw_h ** l2h_bw_alpha)

        #         total = total + lam_l2h * l_h
        #         loss_dict['l2h_high_l1'] = float(l_h.item())
        #         loss_tensors['l2h_high_l1'] = (l_h, 'mel')

        #         # 仅当启用自适应时收集梯度尺度（相对最后层 mel_anchor）
        #         if bool(getattr(cfg, 'adaptive_hf', False)):
        #             try:
        #                 g = autograd.grad(l_h, mel_hat_refined, retain_graph=True, allow_unused=True)[0]
        #                 if isinstance(g, torch.Tensor):
        #                     gv = torch.nan_to_num(g.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        #                     grad_info['g_l2h'] = float(gv.abs().mean().item())
        #             except Exception:
        #                 pass
        #     except Exception as _e:
        #         if os.environ.get('DBG_STAGE25', '0') == '1':
        #             print('[l2h_debug]', _e)

        # # 1.8-bis) 直接高频 L1：不做 envelope 平滑，作为 DeCo-L2H 的稳态监督
        # lam_l2h_direct = float(getattr(cfg, 'lambda_l2h_direct', 0.0))
        # if lam_l2h_direct > 0.0 and mel_hat_refined is not None:
        #     try:
        #         low_bins = int(getattr(cfg, 'l2h_low_bins', 10))
        #         low_bins = max(1, min(low_bins, mel_hat_refined.size(-1) - 1))
        #         pred_h = mel_hat_refined[:, :, low_bins:]
        #         tgt_h = mel[:, :, low_bins:]

        #         if silence_mask is not None:
        #             Tf = silence_mask.size(1)
        #             T_use = min(pred_h.size(1), Tf)
        #             ns_mask = (~silence_mask[:, :T_use]).to(pred_h.dtype)
        #             pred_h = pred_h[:, :T_use, :]
        #             tgt_h = tgt_h[:, :T_use, :]
        #         else:
        #             ns_mask = torch.ones(pred_h.size(0), pred_h.size(1), device=pred_h.device, dtype=pred_h.dtype)

        #         diff = torch.abs(pred_h - tgt_h)
        #         denom = ns_mask.sum() * diff.size(-1) + 1e-6
        #         l_direct = (diff * ns_mask.unsqueeze(-1)).sum() / denom
        #         total = total + lam_l2h_direct * l_direct
        #         loss_dict['l2h_high_l1_direct'] = float(l_direct.item())
        #         loss_tensors['l2h_high_l1_direct'] = (l_direct, 'mel')
        #     except Exception as _e:
        #         if os.environ.get('DBG_STAGE25', '0') == '1':
        #             print('[l2h_direct_debug]', _e)

        # 1.8-ter) L2H 穿透损失：在 18-band / ceps 域计算，确保 L2H 优化目标对齐到 vocoder 输入。
        # 这里仅对 L2H refined 输出的高频部分施加约束，不再管低频，以避免与
        # baseline mel loss 拉扯。
        lam_l2h_band18 = float(getattr(cfg, 'lambda_l2h_band18', 0.0))
        lam_l2h_ceps = float(getattr(cfg, 'lambda_l2h_ceps', 0.0))
        if (lam_l2h_band18 > 0.0 or lam_l2h_ceps > 0.0):
            try:
                mel_ref = out.get('mel', None)
                mel_hat_refined = out.get('mel_hat_refined', out.get('mel_hat', None))
                if isinstance(mel_ref, torch.Tensor) and isinstance(mel_hat_refined, torch.Tensor):
                    # 转到能量域
                    E_pred = torch.pow(10.0, torch.clamp(mel_hat_refined, min=-10.0, max=10.0))
                    E_gt = torch.pow(10.0, torch.clamp(mel_ref, min=-10.0, max=10.0))

                    # 聚合到 18 维
                    e18_pred = model.band_agg_32_to_18(E_pred)  # [B,T,18]
                    e18_gt = model.band_agg_32_to_18(E_gt)      # [B,T,18]

                    # 对齐时间维度
                    Tm = min(e18_pred.size(1), e18_gt.size(1))
                    e18_pred = e18_pred[:, :Tm, :]
                    e18_gt = e18_gt[:, :Tm, :]

                    # 只对高频 band 计算损失
                    hi_start = int(getattr(cfg, 'l2h_band18_hi_start', 8))
                    hi_start = max(0, min(hi_start, 17))

                    # 静音掩膜：仅在非静音帧上约束 HF
                    if silence_mask is not None:
                        Tf = silence_mask.size(1)
                        T_use = min(Tm, Tf)
                        ns_mask = (~silence_mask[:, :T_use]).to(e18_pred.dtype)  # [B,T]
                        e18_pred = e18_pred[:, :T_use, :]
                        e18_gt = e18_gt[:, :T_use, :]
                    else:
                        ns_mask = torch.ones(e18_pred.size(0), e18_pred.size(1),
                                             device=e18_pred.device, dtype=e18_pred.dtype)
                        T_use = Tm

                    # 1) 18-band 能量域穿透损失（仅高频部分）
                    if lam_l2h_band18 > 0.0:
                        e18_hi_pred = e18_pred[:, :, hi_start:]
                        e18_hi_gt = e18_gt[:, :, hi_start:]
                        log_pred = torch.log10(e18_hi_pred + 1e-10)
                        log_gt = torch.log10(e18_hi_gt + 1e-10)
                        diff_band = torch.abs(log_pred - log_gt)
                        denom_band = ns_mask.sum() * diff_band.size(-1) + 1e-6
                        l_band18 = (diff_band * ns_mask.unsqueeze(-1)).sum() / denom_band
                        total = total + lam_l2h_band18 * l_band18
                        loss_dict['l2h_band18'] = float(l_band18.item())
                        loss_tensors['l2h_band18'] = (l_band18, 'mel')

                    # 2) 18-ceps 倒谱域穿透损失（仅高阶倒谱，对应 HF）
                    if lam_l2h_ceps > 0.0:
                        e18_log_pred = torch.log10(e18_pred + 1e-10)
                        e18_log_gt = torch.log10(e18_gt + 1e-10)
                        e18_log_pred = opus_band_log_smooth(e18_log_pred)
                        e18_log_gt = opus_band_log_smooth(e18_log_gt)
                        ceps_pred = model.mel18_to_ceps(e18_log_pred)  # [B,T,18]
                        ceps_gt = model.mel18_to_ceps(e18_log_gt)      # [B,T,18]
                        ceps_hi_pred = ceps_pred[:, :T_use, hi_start:]
                        ceps_hi_gt = ceps_gt[:, :T_use, hi_start:]
                        diff_ceps = torch.abs(ceps_hi_pred - ceps_hi_gt)
                        denom_ceps = ns_mask.sum() * diff_ceps.size(-1) + 1e-6
                        l_ceps18 = (diff_ceps * ns_mask.unsqueeze(-1)).sum() / denom_ceps
                        total = total + lam_l2h_ceps * l_ceps18
                        loss_dict['l2h_ceps18'] = float(l_ceps18.item())
                        loss_tensors['l2h_ceps18'] = (l_ceps18, 'ceps')
            except Exception as _e:
                if os.environ.get('DBG_STAGE25', '0') == '1':
                    print('[l2h_passthrough_debug]', _e)


        # 1.8) L2H：只学习 high-bin residual（避免“低频纹理填满高频”）
        # - pred_resid:  预测的高频残差（可带 limiter）
        # - target_resid: (GT_high - base_high) 只让 L2H 补 baseline 补不出的部分
        # - gate: voiced(vuv_prob) * blend * (non-silence) 让 L2H 不在静音/清辅音乱补
        # - decor: 只对 noise 分量做 decor，逼它别学“低频走势复制到高频”
        def masked_l1(pred, target, mask, eps=1e-8):
            # pred/target: [B,T,H], mask: [B,T,1] or [B,T,H]
            if mask is None:
                return F.l1_loss(pred, target)
            if mask.dim() == 3 and mask.size(-1) == 1:
                mask = mask.expand_as(pred)
            w = mask.clamp_min(0.0)
            return (w * (pred - target).abs()).sum() / (w.sum() + eps)

        def decorrelation_loss(low_btL, high_btH, eps=1e-6):
            """
            惩罚 low 与 high 在 time 维度上的线性相关（相关矩阵能量）
            low:  [B,T,L]
            high: [B,T,H]
            """
            # 去均值
            X = low_btL - low_btL.mean(dim=1, keepdim=True)
            Y = high_btH - high_btH.mean(dim=1, keepdim=True)

            cov = torch.einsum("btl,bth->blh", X, Y) / (X.size(1) - 1 + eps)  # [B,L,H]
            stdX = (X.pow(2).mean(dim=1) + eps).sqrt().unsqueeze(-1)          # [B,L,1]
            stdY = (Y.pow(2).mean(dim=1) + eps).sqrt().unsqueeze(-2)          # [B,1,H]
            corr = cov / (stdX * stdY + eps)                                  # [B,L,H]
            return (corr ** 2).mean()

        lam_l2h_resid = float(getattr(cfg, "lambda_l2h_resid", 0.0))
        lam_l2h_decor = float(getattr(cfg, "lambda_l2h_decor", 0.0))
        improve_margin = float(getattr(cfg, "l2h_improve_margin", 0.0))

        if (lam_l2h_resid > 0.0 or lam_l2h_decor > 0.0):
            lb = int(getattr(cfg, "l2h_low_bins", 10))
            lb = max(1, min(lb, out["mel_hat"].size(-1) - 1))

            pred_resid = out.get("l2h_resid", None)        # [B,T,H]
            vuv_prob   = out.get("l2h_vuv_prob", None)     # [B,T,1]（建议这里已经是 sigmoid 后的 prob）
            mask_harm  = out.get("l2h_mask_harm", None)    # [1,1,H] or [B,T,H]（harmonic mask）

            if isinstance(pred_resid, torch.Tensor) and isinstance(vuv_prob, torch.Tensor):
                mel_hat = out["mel_hat"]                   # baseline mel [B,T,32]
                mel_gt  = mel                              # GT mel [B,T,32]

                base_high = mel_hat[:, :, lb:]             # [B,T,H0]
                tgt_high  = mel_gt[:, :, lb:]              # [B,T,H0]

                # ---- 对齐时间/频率维，避免隐式 broadcast/错位 ----
                T_use = min(pred_resid.size(1), base_high.size(1), tgt_high.size(1))
                H_use = min(pred_resid.size(2), base_high.size(2), tgt_high.size(2))
                pred_resid = pred_resid[:, :T_use, :H_use]
                base_high  = base_high[:,  :T_use, :H_use]
                tgt_high   = tgt_high[:,   :T_use, :H_use]
                vuv_prob   = vuv_prob[:,   :T_use, :]

                # target_resid = (GT_high - base_high) 让 L2H 只补差值
                # detach baseline HF so L2H 专属 loss 的梯度只通过 L2H 分支
                base_high_det = base_high.detach()
                target_resid = (tgt_high - base_high_det)

                # ---- limiter（可选）：防止一开始 resid 特别大直接把高频“糊满/轰炸” ----
                resid_clip = float(getattr(cfg, "l2h_resid_clip", 0.0))  # 建议 1.0~2.0（log-mel域）
                if resid_clip > 0.0:
                    pred_resid = pred_resid.clamp(-resid_clip, resid_clip)

                # ---- gate：blend * voiced_prob * non-silence ----
                blend = float(getattr(model, "l2h_blend", 1.0))
                gate = (blend * vuv_prob).clamp(0.0, 1.0)  # [B,T,1]

                if silence_mask is not None:
                    # silence_mask: [B,T] bool，True=静音
                    gate = gate * (~silence_mask[:, :T_use]).to(gate.dtype).unsqueeze(-1)

                # ---- split harm/noise supervision ----
                if not isinstance(mask_harm, torch.Tensor):
                    mask_harm = torch.zeros(1, 1, H_use, device=pred_resid.device, dtype=pred_resid.dtype)
                else:
                    # 对齐到 [*,*,H_use]
                    if mask_harm.dim() == 3:
                        mask_harm = mask_harm[..., :H_use]
                    elif mask_harm.dim() == 2:
                        mask_harm = mask_harm.unsqueeze(0).unsqueeze(0)[..., :H_use]
                    else:
                        mask_harm = mask_harm.reshape(1, 1, -1)[..., :H_use]
                    mask_harm = mask_harm.to(pred_resid.device, pred_resid.dtype)

                harm_mask = mask_harm
                noise_mask = (1.0 - mask_harm)

                # harm: gated by vuv_prob（gate 已含 blend 与非静音）
                gate_h = gate
                # noise: 只按非静音门控，不依赖 vuv_prob，避免无声辅音完全不训练
                gate_n = torch.ones_like(gate)
                if silence_mask is not None:
                    gate_n = gate_n * (~silence_mask[:, :T_use]).to(gate.dtype).unsqueeze(-1)

                resid_h = pred_resid * harm_mask
                resid_n = pred_resid * noise_mask
                tgt_h = target_resid * harm_mask
                tgt_n = target_resid * noise_mask

                # 1) improvement-style residual loss：对比 baseline HF 与
                #    L2H refined HF 的误差，鼓励 refined 在高频上优于 baseline。
                if lam_l2h_resid > 0.0:
                    eps = 1e-8
                    margin = float(improve_margin)

                    # baseline 高频（仅供误差对比，不参与反向）
                    base_h = base_high_det * harm_mask
                    base_n = base_high_det * noise_mask

                    # refined 高频 = baseline.detach() + residual
                    ref_h = base_h + resid_h
                    ref_n = base_n + resid_n

                    # 逐点 HF 误差
                    err_base_h = (base_h - tgt_h).abs()
                    err_ref_h  = (ref_h  - tgt_h).abs()
                    err_base_n = (base_n - tgt_n).abs()
                    err_ref_n  = (ref_n  - tgt_n).abs()

                    # gate 展开到 [B,T,H]
                    gate_h_full = gate_h
                    gate_n_full = gate_n

                    mask_h_full = harm_mask * gate_h_full
                    mask_n_full = noise_mask * gate_n_full

                    # 仅在 refined 误差未优于 baseline−margin 时产生惩罚。
                    improv_h = torch.relu(err_ref_h - (err_base_h - margin))
                    improv_n = torch.relu(err_ref_n - (err_base_n - margin))

                    l_h = (improv_h * mask_h_full).sum() / (mask_h_full.sum() + eps)
                    l_n = (improv_n * mask_n_full).sum() / (mask_n_full.sum() + eps)
                    l2h_resid_l1 = l_h + l_n

                    total = total + lam_l2h_resid * l2h_resid_l1
                    loss_dict["l2h_resid"] = float(l2h_resid_l1.item())
                    loss_dict["l2h_resid_harm"] = float(l_h.item())
                    loss_dict["l2h_resid_noise"] = float(l_n.item())
                    loss_tensors["l2h_resid"] = (l2h_resid_l1, "mel")

                # 2) decor：只对 noise 分量做（避免把“谐波结构”也去相关导致发虚）
                if lam_l2h_decor > 0.0:
                    resid_noise = resid_n
                    # decor 只在 gate>0 的帧上强一点（不要让 unvoiced/silence 的 decor 主导梯度）
                    resid_noise = resid_noise * gate.detach()

                    # low 参照：用 baseline 的 low（或你也可以试试 GT low，但 baseline 更“自洽”）
                    mel_low = mel_hat[:, :T_use, :lb].detach()   # [B,T,lb]

                    l2h_decor = decorrelation_loss(mel_low, resid_noise)
                    total = total + lam_l2h_decor * l2h_decor
                    loss_dict["l2h_decor"] = float(l2h_decor.item())
                    loss_tensors["l2h_decor"] = (l2h_decor, "mel")

                # ---- 3) 诊断统计：直接塞进 loss_dict，跟随你每10step日志输出 ----
                if (global_step % 10) == 0:
                    with torch.no_grad():
                        vp = vuv_prob.detach()
                        loss_dict["l2h_vuv_min"]  = float(vp.min().item())
                        loss_dict["l2h_vuv_max"]  = float(vp.max().item())
                        loss_dict["l2h_vuv_mean"] = float(vp.mean().item())
                        loss_dict["l2h_resid_std"] = float(pred_resid.detach().std().item())
                        loss_dict["l2h_blend"] = float(blend)



        # 1.9) 高频亮区纹理一致性（梯度-结构损失）
        lam_tex = float(getattr(cfg, 'lambda_mel_texture', 0.0))
        if lam_tex > 0.0:
            try:
                low_bins = int(getattr(cfg, 'l2h_low_bins', 10))
                Xp = mel_hat_refined[:, :, low_bins:]
                Xt = mel[:, :, low_bins:]
                # 亮区：把阈值从0.8降到0.6，并做两次3x3膨胀
                tau = torch.quantile(Xt, 0.6, dim=2, keepdim=True)
                M = (Xt >= tau).to(Xt.dtype)
                for _ in range(2):
                    Mp = F.avg_pool2d(M.unsqueeze(1), kernel_size=(3, 3), stride=1, padding=1).squeeze(1)
                    M = (Mp > 0.3).to(Xt.dtype)
                # 有声门控：使用 target ∪ predicted
                fc_hat = out.get('frame_corr_hat', None)
                fc = out.get('frame_corr', None)
                th = float(getattr(cfg, 'vuv_threshold', 0.3))
                m_pred = (fc_hat > th).squeeze(-1) if isinstance(fc_hat, torch.Tensor) else None
                m_tgt  = (fc > th).squeeze(-1) if isinstance(fc, torch.Tensor) else None
                if m_pred is None and m_tgt is None:
                    V = torch.ones_like(M)
                elif m_pred is None:
                    V = m_tgt.to(M.dtype)
                elif m_tgt is None:
                    V = m_pred.to(M.dtype)
                else:
                    V = (m_pred | m_tgt).to(M.dtype)
                M = M * V
                # 时/频向梯度
                def _grad_t(x):
                    return x[:, 1:, :] - x[:, :-1, :]
                def _grad_f(x):
                    return x[:, :, 1:] - x[:, :, :-1]
                Gt_p, Gf_p = _grad_t(Xp), _grad_f(Xp)
                Gt_t, Gf_t = _grad_t(Xt), _grad_f(Xt)
                Mt = M[:, 1:, :]
                Mf = M[:, :, 1:]
                denom_t = Mt.sum() + 1e-6
                denom_f = Mf.sum() + 1e-6
                l_tex = 0.0
                if denom_t > 1e-6:
                    l_tex = l_tex + (torch.abs(Gt_p - Gt_t) * Mt).sum() / denom_t
                if denom_f > 1e-6:
                    l_tex = l_tex + 0.5 * (torch.abs(Gf_p - Gf_t) * Mf).sum() / denom_f
                total = total + lam_tex * l_tex
                loss_dict['mel_tex'] = float(l_tex.item())
                loss_tensors['mel_tex'] = (l_tex, 'mel')
                if bool(getattr(cfg, 'adaptive_hf', False)):
                    try:
                        g = autograd.grad(l_tex, mel_hat_refined, retain_graph=True, allow_unused=True)[0]
                        if isinstance(g, torch.Tensor):
                            gv = torch.nan_to_num(g.detach(), nan=0.0, posinf=0.0, neginf=0.0)
                            grad_info['g_tex'] = float(gv.abs().mean().item())
                    except Exception:
                        pass
            except Exception as _e:
                if os.environ.get('DBG_STAGE25', '0') == '1':
                    print('[mel_tex_debug]', _e)

    # 2) ceps 重建（对齐时间与维度）
    # 当 lambda_ceps 为 0 时，只保留 ceps_hi / Δceps 等附加项，跳过主 ceps L1。
    # 若模型提供 ceps_hat_base（未经过 L2H 的 ceps），则可在 content_only 模式
    # 下将低阶维度的监督绑定到 baseline，避免 L2H 被低阶 loss 牵制，只在高阶
    # 倒谱上施加强约束（尤其是与 ``ceps_hi_start`` 搭配使用时）。
    ceps_hat = out["ceps_hat"]; ceps_tgt = out["ceps"]
    ceps_hat_base = out.get("ceps_hat_base", None)
    Tm = min(ceps_hat.size(1), ceps_tgt.size(1))
    Dm = min(ceps_hat.size(2), ceps_tgt.size(2))
    lam_ceps = float(getattr(cfg, 'lambda_ceps', 0.0))
    if lam_ceps > 0.0:
        try:
            # 若提供 baseline ceps，并处于 content_only 模式，则构造一个
            # 混合 ceps_hat_main：低阶维度来自 ceps_hat_base（与 L2H 解耦），
            # 高阶维度仍使用 ceps_hat（L2H 可自由优化高阶 ceps）。
            ceps_hat_main = ceps_hat[:, :Tm, :Dm]
            if ceps_hat_base is not None and getattr(cfg, "content_only", False):
                s0_lo = int(getattr(cfg, "ceps_hi_start", 10))
                s0_lo = max(0, min(s0_lo, Dm))
                base_slice = ceps_hat_base[:, :Tm, :Dm].detach()
                if s0_lo > 0:
                    ceps_hat_main = ceps_hat_main.clone()
                    ceps_hat_main[:, :, :s0_lo] = base_slice[:, :, :s0_lo]

            if voiced_mask is not None and isinstance(voiced_mask, torch.Tensor):
                # 使用统一的 voiced_mask（基于 mel 能量）只在有声帧上监督倒谱
                vm = voiced_mask[:, :Tm].to(ceps_hat.dtype)  # [B,T]
                diff_ceps = torch.abs(ceps_hat_main - ceps_tgt[:, :Tm, :Dm])  # [B,T,D]
                denom = vm.sum() * Dm + 1e-6
                loss_ceps = (diff_ceps * vm.unsqueeze(-1)).sum() / denom
            else:
                loss_ceps = F.l1_loss(ceps_hat_main, ceps_tgt[:, :Tm, :Dm])
        except Exception:
            # 形状不匹配等异常时，退化为按最小长度的一维 L1
            a = ceps_hat.reshape(-1)
            b = ceps_tgt.reshape(-1)
            n = min(a.numel(), b.numel())
            loss_ceps = F.l1_loss(a[:n], b[:n])

        total = total + lam_ceps * loss_ceps
        loss_dict["ceps"] = float(loss_ceps.item())
        loss_tensors['ceps'] = (loss_ceps, 'ceps')

    # 2.0-bis) 倒谱加权损失：对低阶系数赋予更高权重（能量/包络更重要）
    lam_ceps_w = float(getattr(cfg, 'lambda_ceps_weighted', 0.0))
    if lam_ceps_w > 0.0:
        try:
            # 基于一个固定表，对各维倒谱给出手工权重；
            # 若 Dm 不同，则按需要裁剪或在尾部延用最后一个权重。
            base_w = torch.tensor([
                2.0,
                1.5, 1.5, 1.2, 1.2,
                1.0, 1.0, 1.0, 1.0,
                0.8, 0.8, 0.7, 0.7,
                0.5, 0.5, 0.4, 0.4,
                0.3, 0.3,
            ], device=ceps_hat.device, dtype=ceps_hat.dtype)
            if Dm <= base_w.numel():
                w = base_w[:Dm]
            else:
                tail = base_w[-1].expand(Dm - base_w.numel())
                w = torch.cat([base_w, tail], dim=0)
            w = w.view(1, 1, Dm)
            # 与主 ceps L1 一致，若存在 baseline ceps，则低阶使用 ceps_hat_base，
            # 高阶使用 L2H 细化后的 ceps_hat，以避免 L2H 被低阶损失牵制。
            ceps_hat_for_w = ceps_hat[:, :Tm, :Dm]
            if ceps_hat_base is not None and getattr(cfg, "content_only", False):
                s0_lo = int(getattr(cfg, "ceps_hi_start", 10))
                s0_lo = max(0, min(s0_lo, Dm))
                base_slice = ceps_hat_base[:, :Tm, :Dm].detach()
                if s0_lo > 0:
                    ceps_hat_for_w = ceps_hat_for_w.clone()
                    ceps_hat_for_w[:, :, :s0_lo] = base_slice[:, :, :s0_lo]

            diff2 = (ceps_hat_for_w - ceps_tgt[:, :Tm, :Dm]) ** 2
            loss_cw = (diff2 * w).mean()
            total = total + lam_ceps_w * loss_cw
            loss_dict['ceps_weighted'] = float(loss_cw.item())
            loss_tensors['ceps_weighted'] = (loss_cw, 'ceps')
        except Exception:
            pass

    # 2.0-ter) BFCC→ceps 映射损失（使用 GT mel/BFCC 校准 band_agg_32_to_18+mel18_to_ceps）
    lam_ceps_map_gt = float(getattr(cfg, 'lambda_ceps_map_gt', 0.0))
    if lam_ceps_map_gt > 0.0:
        try:
            mel_gt = out.get('mel', None)
            ceps_target_gt = out.get('ceps', None)
            if isinstance(mel_gt, torch.Tensor) and isinstance(ceps_target_gt, torch.Tensor):
                # 转到能量域并聚合到 18 带
                E_gt = torch.pow(10.0, torch.clamp(mel_gt, min=-10.0, max=10.0))  # [B,T,32]
                e18_gt = model.band_agg_32_to_18(E_gt)                            # [B,T,18]
                e18_log_gt = torch.log10(e18_gt + 1e-10)
                e18_log_gt = opus_band_log_smooth(e18_log_gt)

                ceps_map = model.mel18_to_ceps(e18_log_gt)
                ceps_map = torch.nan_to_num(ceps_map, nan=0.0)
                ceps_t = torch.nan_to_num(ceps_target_gt, nan=0.0)

                Tm_map = min(ceps_map.size(1), ceps_t.size(1))
                Dm_map = min(ceps_map.size(2), ceps_t.size(2))
                loss_map = F.l1_loss(ceps_map[:, :Tm_map, :Dm_map], ceps_t[:, :Tm_map, :Dm_map])

                total = total + lam_ceps_map_gt * loss_map
                loss_dict['ceps_map_gt'] = float(loss_map.item())
                loss_tensors['ceps_map_gt'] = (loss_map, 'ceps')
        except Exception:
            pass

    # 2.1) 倒谱高阶监督（如 c10..），抑制高频抹平
    lam_ceps_hi = float(getattr(cfg, 'lambda_ceps_hi', 0.0))
    if lam_ceps_hi > 0.0:
        s0 = int(getattr(cfg, 'ceps_hi_start', 10))
        s0 = max(1, min(s0, Dm - 1))
        if voiced_mask is not None and isinstance(voiced_mask, torch.Tensor):
            vm = voiced_mask[:, :Tm].to(ceps_hat.dtype)
            diff_hi = torch.abs(ceps_hat[:, :Tm, s0:Dm] - ceps_tgt[:, :Tm, s0:Dm])
            denom = vm.sum() * max(1, Dm - s0) + 1e-6
            l_chi = (diff_hi * vm.unsqueeze(-1)).sum() / denom
        else:
            l_chi = F.l1_loss(ceps_hat[:, :Tm, s0:Dm], ceps_tgt[:, :Tm, s0:Dm])
        total = total + lam_ceps_hi * l_chi
        loss_dict['ceps_hi'] = float(l_chi.item())
        loss_tensors['ceps_hi'] = (l_chi, 'ceps')

    # 2.1-bis) 倒谱高阶监督（无声但“非静音”帧），专门保护虚线 F0 区域的高频纹理。
    # 使用 frame_corr (VUV) + 静音掩膜联合确定目标帧：
    #   - frame_corr <= vuv_threshold 视为“无声”；
    #   - silence_mask=False 视为“非极静音”，通常对应有气声/擦音纹理的无声段。
    lam_ceps_hi_unv = float(getattr(cfg, 'lambda_ceps_hi_unv', 0.0))
    if lam_ceps_hi_unv > 0.0:
        try:
            fc_ref = out.get("frame_corr", None)  # [B,T,1]
            if isinstance(fc_ref, torch.Tensor) and (silence_mask is not None):
                th_vuv = float(getattr(cfg, 'vuv_threshold', 0.3))
                # 基于 GT VUV 的无声掩膜
                v_mask = (fc_ref[:, :Tm, :] > th_vuv).squeeze(-1)   # [B,Tm] True=voiced
                unv_mask = (~v_mask)                                # [B,Tm]

                # 将静音掩膜对齐到 ceps 时间轴
                if silence_mask.size(1) >= Tm:
                    sil_m = silence_mask[:, :Tm]
                else:
                    sil_m = torch.zeros_like(unv_mask, dtype=unv_mask.dtype, device=unv_mask.device)
                    sil_m[:, :silence_mask.size(1)] = silence_mask

                # 目标区域：无声且非静音（通常对应擦音/气声等虚线 F0 区域）
                mask_unv = unv_mask & (~sil_m)                      # [B,Tm]

                if mask_unv.any():
                    s0 = int(getattr(cfg, 'ceps_hi_start', 10))

                    s0 = max(1, min(s0, Dm - 1))
                    Dh = max(1, Dm - s0)
                    ceps_hi_hat = ceps_hat[:, :Tm, s0:Dm]
                    ceps_hi_tgt = ceps_tgt[:, :Tm, s0:Dm]
                    diff_hi_unv = torch.abs(ceps_hi_hat - ceps_hi_tgt)
                    m_unv = mask_unv.to(ceps_hi_hat.dtype).unsqueeze(-1)  # [B,Tm,1]
                    denom_unv = m_unv.sum() * Dh + 1e-6
                    l_chi_unv = (diff_hi_unv * m_unv).sum() / denom_unv
                    total = total + lam_ceps_hi_unv * l_chi_unv
                    loss_dict['ceps_hi_unv'] = float(l_chi_unv.item())
                    loss_tensors['ceps_hi_unv'] = (l_chi_unv, 'ceps')
        except Exception:
            pass

    # 2.1-ter) 条件 flow 高频 Mel NLL（可选）：在 GT 高频 mel 上建模 p(mel_high | mel_low, f0, vuv)
    lam_l2h_flow_nll = float(getattr(cfg, 'lambda_l2h_flow_nll', 0.0))
    if lam_l2h_flow_nll > 0.0 and bool(getattr(cfg, 'use_l2h_flow', False)) and hasattr(model, 'l2h_flow') and model.l2h_flow is not None:
        try:
            mel = out.get("mel", None)
            dp_ref = out.get("dnn_pitch", None)
            fc_ref = out.get("frame_corr", None)
            if isinstance(mel, torch.Tensor) and isinstance(dp_ref, torch.Tensor) and isinstance(fc_ref, torch.Tensor):
                Bm, Tm, Fm = mel.shape
                lb = int(getattr(cfg, 'l2h_low_bins', 10))
                lb = max(1, min(lb, Fm - 1))
                mel_low = mel[:, :, :lb]
                mel_high_gt = mel[:, :, lb:]
                T_use = min(Tm, dp_ref.size(1), fc_ref.size(1))
                mel_low = mel_low[:, :T_use, :]
                mel_high_gt = mel_high_gt[:, :T_use, :]
                dp = dp_ref[:, :T_use, :]
                fc = fc_ref[:, :T_use, :]
                # pitch: 直接使用归一化前的 dnn_pitch；vuv: 将 frame_corr 映射到 [0,1]
                pitch = dp
                vuv = torch.clamp(0.5 * (fc + 1.0), 0.0, 1.0)
                z_flow, log_det = model.l2h_flow(mel_low, pitch, vuv, mel_high_base=mel_high_gt)
                # z ~ N(0,I) prior；忽略常数项，NLL ≈ 0.5*||z||^2 - log_det
                z2 = z_flow.pow(2).sum(dim=-1)  # [B,T]
                nll = 0.5 * z2 - log_det
                loss_flow = nll.mean()
                total = total + lam_l2h_flow_nll * loss_flow
                loss_dict['l2h_flow_nll'] = float(loss_flow.item())
        except Exception:
            pass

    # 3) dnn_pitch 重建（仅在有声段计算）
    lam_f0_main = float(getattr(cfg, 'lambda_f0', 0.0))
    if lam_f0_main > 0.0:
        try:
            dp_hat = out["dnn_pitch_hat"]  # [B,T,1]
            dp_ref = out["dnn_pitch"]      # [B,T,1]
            fc = out.get("frame_corr", None)  # [B,T,1]
            if isinstance(dp_hat, torch.Tensor) and isinstance(dp_ref, torch.Tensor) and isinstance(fc, torch.Tensor):
                th = float(getattr(cfg, 'vuv_threshold', 0.3))
                mask = (fc > th).to(dp_hat.dtype)  # [B,T,1]
                err2 = (dp_hat - dp_ref) ** 2
                denom = mask.sum() + 1e-6
                loss_f0 = (err2 * mask).sum() / denom
            else:
                loss_f0 = F.mse_loss(out["dnn_pitch_hat"], out["dnn_pitch"])
        except Exception:
            loss_f0 = F.mse_loss(out["dnn_pitch_hat"], out["dnn_pitch"])
        total = total + lam_f0_main * loss_f0
        loss_dict["f0"] = float(loss_f0.item())
        loss_tensors['f0'] = (loss_f0, 'f0')

    # 3.b) F0 base 分支重建（仅使用前 k 维符号解码得到的粗 F0），
    #      用于 successive refinement：骨架在 base 上，细节在 full 上。
    lam_f0_base = float(getattr(cfg, 'lambda_f0_base', 0.0))
    if lam_f0_base > 0.0 and "dnn_pitch_hat_base" in out:
        try:
            dp_hat_b = out["dnn_pitch_hat_base"]  # [B,T,1]
            dp_ref = out["dnn_pitch"]            # [B,T,1]
            fc = out.get("frame_corr", None)     # [B,T,1]
            if isinstance(dp_hat_b, torch.Tensor) and isinstance(dp_ref, torch.Tensor) and isinstance(fc, torch.Tensor):
                th = float(getattr(cfg, 'vuv_threshold', 0.3))
                mask = (fc > th).to(dp_hat_b.dtype)  # [B,T,1]
                err2 = (dp_hat_b - dp_ref) ** 2
                denom = mask.sum() + 1e-6
                loss_f0_base = (err2 * mask).sum() / denom
            else:
                loss_f0_base = F.mse_loss(dp_hat_b, dp_ref)
        except Exception:
            loss_f0_base = F.mse_loss(out["dnn_pitch_hat_base"], out["dnn_pitch"])
        total = total + lam_f0_base * loss_f0_base
        loss_dict["f0_base"] = float(loss_f0_base.item())
        loss_tensors['f0_base'] = (loss_f0_base, 'f0')

    # 3.5) F0平滑损失：惩罚二阶差分（加速度），避免F0抹平
    lam_f0_smooth = float(getattr(cfg, 'lambda_f0_smooth', 0.0))
    if lam_f0_smooth > 0.0:
        try:
            dp_hat = out["dnn_pitch_hat"]  # [B,T,1]
            fc_ref = out.get("frame_corr", None)  # [B,T,1]
            if isinstance(dp_hat, torch.Tensor) and isinstance(fc_ref, torch.Tensor) and dp_hat.size(1) > 2:
                th = float(getattr(cfg, 'vuv_threshold', 0.3))
                # 二阶差分：加速度 = f0[t+1] - 2*f0[t] + f0[t-1]
                d2_f0 = dp_hat[:, 2:, :] - 2.0 * dp_hat[:, 1:-1, :] + dp_hat[:, :-2, :]  # [B,T-2,1]
                # 有声段mask（取中间帧）
                mask_smooth = (fc_ref[:, 1:-1, :] > th).to(dp_hat.dtype)  # [B,T-2,1]
                # 使用L1 norm（robust，不过度惩罚正常变化）
                err_smooth = torch.abs(d2_f0)
                denom_smooth = mask_smooth.sum() + 1e-6
                loss_f0_smooth = (err_smooth * mask_smooth).sum() / denom_smooth
                total = total + lam_f0_smooth * loss_f0_smooth
                loss_dict["f0_smooth"] = float(loss_f0_smooth.item())
                loss_tensors['f0_smooth'] = (loss_f0_smooth, 'f0')
        except Exception:
            pass

    # 3.5.b) F0 base 平滑损失：在 base 分支上单独约束二阶差分，
    #        通常可设置为略大于 full 分支，以鼓励骨架更平滑。
    lam_f0_base_smooth = float(getattr(cfg, 'lambda_f0_base_smooth', 0.0))
    if lam_f0_base_smooth > 0.0 and "dnn_pitch_hat_base" in out:
        try:
            dp_hat_b = out["dnn_pitch_hat_base"]  # [B,T,1]
            fc_ref = out.get("frame_corr", None)  # [B,T,1]
            if isinstance(dp_hat_b, torch.Tensor) and isinstance(fc_ref, torch.Tensor) and dp_hat_b.size(1) > 2:
                th = float(getattr(cfg, 'vuv_threshold', 0.3))
                d2_f0_b = dp_hat_b[:, 2:, :] - 2.0 * dp_hat_b[:, 1:-1, :] + dp_hat_b[:, :-2, :]  # [B,T-2,1]
                mask_smooth_b = (fc_ref[:, 1:-1, :] > th).to(dp_hat_b.dtype)  # [B,T-2,1]
                err_smooth_b = torch.abs(d2_f0_b)
                denom_smooth_b = mask_smooth_b.sum() + 1e-6
                loss_f0_base_smooth = (err_smooth_b * mask_smooth_b).sum() / denom_smooth_b
                total = total + lam_f0_base_smooth * loss_f0_base_smooth
                loss_dict["f0_base_smooth"] = float(loss_f0_base_smooth.item())
                loss_tensors['f0_base_smooth'] = (loss_f0_base_smooth, 'f0')
        except Exception:
            pass

    # 3.6) F0 标准差匹配（Hz 域）：在 voiced 帧上对齐预测与 GT 的 pitch std，
    #      避免网络收敛到“平均音高但几乎无起伏”的平坦解。
    lam_f0_std = float(getattr(cfg, 'lambda_f0_std', 0.0))
    if lam_f0_std > 0.0:
        try:
            dp_hat = out.get("dnn_pitch_hat", None)  # [B,T,1]
            dp_ref = out.get("dnn_pitch", None)      # [B,T,1]
            fc_ref = out.get("frame_corr", None)     # [B,T,1]
            if isinstance(dp_hat, torch.Tensor) and isinstance(dp_ref, torch.Tensor):
                Bf, Tf, _ = dp_hat.shape
                T_use = min(Tf, dp_ref.size(1))
                dp_hat_t = dp_hat[:, :T_use, 0]
                dp_ref_t = dp_ref[:, :T_use, 0]

                # 映射到 Hz 域，采用与其它 F0 相关代码一致的公式
                def _dp_to_hz(dp: torch.Tensor) -> torch.Tensor:
                    period = torch.clamp(256.0 / torch.pow(2.0, dp + 1.5), 32.0, 255.0)
                    return 16000.0 / period

                f0_hat_hz = _dp_to_hz(dp_hat_t)  # [B,T_use]
                f0_ref_hz = _dp_to_hz(dp_ref_t)

                # 仅在 voiced 帧上统计 mean/std；若无 frame_corr 则退化为全帧。
                if isinstance(fc_ref, torch.Tensor):
                    fc_t = fc_ref[:, :T_use, 0]
                    th = float(getattr(cfg, 'vuv_threshold', 0.3))
                    voiced = (fc_t > th).to(f0_hat_hz.dtype)  # [B,T_use]
                else:
                    voiced = torch.ones_like(f0_hat_hz)

                # 至少需要若干 voiced 样本才能稳定估计 std；否则跳过。
                voiced_counts = voiced.sum(dim=1, keepdim=True)  # [B,1]
                valid = voiced_counts > 4.0
                if valid.any():
                    eps = 1e-6
                    # 逐样本 voiced 内 mean/std
                    vc = voiced_counts.clamp_min(1.0)
                    m_hat = (f0_hat_hz * voiced).sum(dim=1, keepdim=True) / vc
                    m_ref = (f0_ref_hz * voiced).sum(dim=1, keepdim=True) / vc
                    v_hat = ((f0_hat_hz - m_hat) ** 2 * voiced).sum(dim=1, keepdim=True) / vc
                    v_ref = ((f0_ref_hz - m_ref) ** 2 * voiced).sum(dim=1, keepdim=True) / vc
                    std_hat = torch.sqrt(v_hat + eps)
                    std_ref = torch.sqrt(v_ref + eps)

                    # 仅在有效样本上计算 L1 差异
                    diff_std = torch.abs(std_hat - std_ref)[valid]
                    loss_f0_std = diff_std.mean()

                    total = total + lam_f0_std * loss_f0_std
                    loss_dict["f0_std"] = float(loss_f0_std.item())
                    loss_tensors['f0_std'] = (loss_f0_std, 'f0')
        except Exception:
            pass

    # 4) frame_corr 重建（仅在有声段计算，避免无声被强压为常值）
    lam_vuv_main = float(getattr(cfg, 'lambda_vuv', 0.0))
    if lam_vuv_main > 0.0:
        try:
            fc_hat = out["frame_corr_hat"]; fc_ref = out["frame_corr"]  # [B,T,1]
            th = float(getattr(cfg, 'vuv_threshold', 0.3))
            mask_v = (fc_ref > th).to(fc_hat.dtype)
            err2_v = (fc_hat - fc_ref) ** 2
            denom_v = mask_v.sum() + 1e-6
            loss_vuv = (err2_v * mask_v).sum() / denom_v
        except Exception:
            loss_vuv = F.mse_loss(out["frame_corr_hat"], out["frame_corr"])
        total = total + lam_vuv_main * loss_vuv
        loss_dict["vuv"] = float(loss_vuv.item())
        loss_tensors['vuv'] = (loss_vuv, 'vuv')

    # 4.bis) frame_corr base 重建：在 SR base 分支上对粗 VUV/相关性做重建，
    #         鼓励“骨架”在 base 层稳定存在，减轻 full 层错误对整体 gate 的影响。
    lam_vuv_base = float(getattr(cfg, 'lambda_vuv_base', 0.0))
    if lam_vuv_base > 0.0 and "frame_corr_hat_base" in out:
        try:
            fc_hat_b = out["frame_corr_hat_base"]
            fc_ref = out["frame_corr"]
            th = float(getattr(cfg, 'vuv_threshold', 0.3))
            mask_vb = (fc_ref > th).to(fc_hat_b.dtype)
            err2_vb = (fc_hat_b - fc_ref) ** 2
            denom_vb = mask_vb.sum() + 1e-6
            loss_vuv_base = (err2_vb * mask_vb).sum() / denom_vb
        except Exception:
            loss_vuv_base = F.mse_loss(out["frame_corr_hat_base"], out["frame_corr"])
        total = total + lam_vuv_base * loss_vuv_base
        loss_dict["vuv_base"] = float(loss_vuv_base.item())
        loss_tensors['vuv_base'] = (loss_vuv_base, 'vuv')

    # 4.a) 静音帧抑制项：在 GT 无声段内惩罚 frame_corr_hat 过高（鼓励接近 0/负值）
    lam_vuv_sil = float(getattr(cfg, 'lambda_vuv_sil', 0.0))
    if lam_vuv_sil > 0.0:
        try:
            fc_hat = out["frame_corr_hat"]; fc_ref = out["frame_corr"]  # [B,T,1]
            th = float(getattr(cfg, 'vuv_threshold', 0.3))
            mask_sil = (fc_ref <= th).to(fc_hat.dtype)        # GT 认为无声的帧
            # 惩罚静音帧上正向偏移的相关系数（过高的“伪浊音”）
            penal = (fc_hat.clamp(min=0.0) ** 2) * mask_sil
            denom_sil = mask_sil.sum() + 1e-6
            loss_vuv_sil = penal.sum() / denom_sil
            total = total + lam_vuv_sil * loss_vuv_sil
            loss_dict['vuv_sil'] = float(loss_vuv_sil.item())
            loss_tensors['vuv_sil'] = (loss_vuv_sil, 'vuv')
        except Exception:
            pass

    # 4.b) 全局 voiced 占比对齐 + 概率 BCE（基于 frame_corr_hat）
    lam_vuv_ratio = float(getattr(cfg, 'lambda_vuv_ratio', 0.0))
    lam_vuv_bce = float(getattr(cfg, 'lambda_vuv_bce', 0.0))
    if (lam_vuv_ratio > 0.0) or (lam_vuv_bce > 0.0):
        try:
            fc_hat = out["frame_corr_hat"]; fc_ref = out["frame_corr"]  # [B,T,1]
            th = float(getattr(cfg, 'vuv_threshold', 0.3))
            v_tgt = (fc_ref > th).float().squeeze(-1)             # [B,T]
            # 通过温和 sigmoid 将 frame_corr_hat 映射为有声概率
            k = 10.0
            logits = (fc_hat.squeeze(-1) - th) * k               # [B,T]
            v_prob = torch.sigmoid(logits)

            if lam_vuv_ratio > 0.0:
                ratio_tgt = v_tgt.mean(dim=1)                    # [B]
                ratio_hat = (v_prob > 0.5).float().mean(dim=1)   # [B]
                loss_ratio = torch.mean(torch.abs(ratio_hat - ratio_tgt))
                total = total + lam_vuv_ratio * loss_ratio
                loss_dict['vuv_ratio'] = float(loss_ratio.item())
                loss_tensors['vuv_ratio'] = (loss_ratio, 'vuv')

            if lam_vuv_bce > 0.0:
                bce = torch.nn.functional.binary_cross_entropy(v_prob, v_tgt)
                total = total + lam_vuv_bce * bce
                loss_dict['vuv_bce'] = float(bce.item())
                loss_tensors['vuv_bce'] = (bce, 'vuv')
        except Exception:
            pass

    # 2.5) Δ/ΔΔ 约束（可选）
    if cfg.lambda_delta > 0.0:
        # 复用已对齐的 ceps_hat / ceps_tgt（[Bc,Tm,Dm]）
        try:
            ceps_hat_d = ceps_hat
            ceps_tgt_d = ceps_tgt
            T = min(ceps_hat_d.size(1), ceps_tgt_d.size(1))
            D = min(ceps_hat_d.size(2), ceps_tgt_d.size(2))
            ceps_hat_d = ceps_hat_d[:, :T, :D]
            ceps_tgt_d = ceps_tgt_d[:, :T, :D]
            d1_hat = ceps_hat_d[:, 1:, :] - ceps_hat_d[:, :-1, :]
            d1_tgt = ceps_tgt_d[:, 1:, :] - ceps_tgt_d[:, :-1, :]
            d2_hat = d1_hat[:, 1:, :] - d1_hat[:, :-1, :]
            d2_tgt = d1_tgt[:, 1:, :] - d1_tgt[:, :-1, :]
            loss_delta = F.l1_loss(d1_hat, d1_tgt) + F.l1_loss(d2_hat, d2_tgt)
            total = total + cfg.lambda_delta * loss_delta
            loss_dict["delta"] = float(loss_delta.item())
            loss_tensors['delta'] = (loss_delta, 'ceps')
        except Exception:
            pass

    # 2.6) 特征自一致性（manifold 固定点）：
    #      使用模型内部的 wave_to_mel / band_agg_32_to_18 / mel18_to_ceps
    #      从 audio_hat 重新提取 ceps_recon，与 ceps_hat 对齐，避免“落在声码器流形之外”的畸形特征。
    lam_manifold = float(getattr(cfg, 'lambda_feature_manifold', 0.0))
    if lam_manifold > 0.0:
        try:
            audio_hat = out.get("audio_hat", None)
            ceps_hat = out.get("ceps_hat", None)
            if isinstance(audio_hat, torch.Tensor) and isinstance(ceps_hat, torch.Tensor):
                # 使用与前向一致的 BFCC → 32→18 → ceps pipeline
                # 1) wave → Bark log 能量图
                mel_from_audio = model.wave_to_mel(audio_hat.to(device))  # [B,Tm,32]

                # 2) 聚合到 18 带并做 Opus 风格 log-domain smoothing
                E = torch.pow(10.0, torch.clamp(mel_from_audio, min=-10.0, max=10.0))
                e18_energy = model.band_agg_32_to_18(E)                     # [B,Tm,18]
                e18_log = torch.log10(e18_energy + 1e-10)
                e18_log = opus_band_log_smooth(e18_log)

                ceps_recon = model.mel18_to_ceps(e18_log)                  # [B,Tm,18]

                # 3) 与 ceps_hat 对齐时间和维度
                Tm = min(ceps_recon.size(1), ceps_hat.size(1))
                Dm = min(ceps_recon.size(2), ceps_hat.size(2))
                ceps_r = ceps_recon[:, :Tm, :Dm]
                ceps_h = ceps_hat[:, :Tm, :Dm]

                if voiced_mask is not None and isinstance(voiced_mask, torch.Tensor):
                    vm = voiced_mask[:, :Tm].to(ceps_h.dtype)
                    diff = torch.abs(ceps_r - ceps_h)
                    denom = vm.sum() * Dm + 1e-6
                    loss_manifold = (diff * vm.unsqueeze(-1)).sum() / denom
                else:
                    loss_manifold = F.l1_loss(ceps_r, ceps_h)

                total = total + lam_manifold * loss_manifold
                loss_dict["ceps_manifold"] = float(loss_manifold.item())
                loss_tensors["ceps_manifold"] = (loss_manifold, "ceps")
        except Exception:
            # 若特征提取出现问题（例如 STFT 维度不匹配），跳过该项，不影响主流程
            pass

    # 2.6)（已精简）移除了 FARGAN 风格的附加波形项，保留核心 STFT/mel/ceps/F0 约束

    qtype = str(getattr(cfg, 'quantizer_type', 'hash'))

    # Hash 模式下的重构与正则（仅在 quantizer_type='hash' 时启用）
    if qtype == 'hash':
        # Hash重构（内容分支）
        if cfg.lambda_hash_recon > 0.0 and "tokens" in out and "tokens_hat" in out:
            loss_hash_recon_c = F.mse_loss(out["tokens_hat"], out["tokens"].detach())
            total = total + cfg.lambda_hash_recon * loss_hash_recon_c
            loss_dict["hash_recon_c"] = float(loss_hash_recon_c.item())
        # Hash重构（F0分支）
        if cfg.lambda_hash_recon > 0.0 and "tokens_f0" in out and "tokens_f0_hat" in out:
            loss_hash_recon_f0 = F.mse_loss(out["tokens_f0_hat"], out["tokens_f0"].detach())
            total = total + cfg.lambda_hash_recon * loss_hash_recon_f0
            loss_dict["hash_recon_f0"] = float(loss_hash_recon_f0.item())

        if cfg.lambda_hash_reg > 0.0 and "hash_reg_terms" in out:
            hash_reg_terms = out["hash_reg_terms"]
            if isinstance(hash_reg_terms, dict) and hash_reg_terms:
                hash_reg = sum(hash_reg_terms.values())
                total = total + cfg.lambda_hash_reg * hash_reg
                # hash_reg 可能是 tensor 也可能是标量，统一转换为 float
                if hasattr(hash_reg, "item"):
                    loss_dict["hash_reg"] = float(hash_reg.item())
                else:
                    loss_dict["hash_reg"] = float(hash_reg)

    # RVQ VQ 损失：在完整路径下，支持对 content/F0 分支分别使用
    # lambda_vq_c / lambda_vq_f 控制；当二者均为 0 时，回退到统一的
    # lambda_vq，以保持向后兼容。
    lam_vq_global = float(getattr(cfg, 'lambda_vq', 0.0))
    lam_vq_c = float(getattr(cfg, 'lambda_vq_c', 0.0))
    lam_vq_f = float(getattr(cfg, 'lambda_vq_f', 0.0))

    vq_loss_total = out.get("vq_loss", None)
    vq_loss_c = out.get("vq_loss_content", None)
    vq_loss_f0 = out.get("vq_loss_f0", None)

    lam_c = lam_vq_c if lam_vq_c > 0.0 else lam_vq_global
    lam_f = lam_vq_f if lam_vq_f > 0.0 else lam_vq_global

    used_branch = False
    if isinstance(vq_loss_c, torch.Tensor) and lam_c > 0.0:
        total = total + lam_c * vq_loss_c
        loss_dict["vq_loss_c"] = float(vq_loss_c.item())
        used_branch = True

    if isinstance(vq_loss_f0, torch.Tensor) and lam_f > 0.0:
        total = total + lam_f * vq_loss_f0
        loss_dict["vq_loss_f0"] = float(vq_loss_f0.item())
        used_branch = True

    # 若未能分别加权分支（例如旧 checkpoint 中缺少 vq_loss_*），
    # 则回退到统一的 lambda_vq * vq_loss_total。
    if (not used_branch) and isinstance(vq_loss_total, torch.Tensor) and lam_vq_global > 0.0:
        total = total + lam_vq_global * vq_loss_total
        loss_dict["vq"] = float(vq_loss_total.item())
    else:
        # 仍然提供一个总的 vq 诊断项（不带权重，仅便于观察整体量级）。
        vq_total = None
        if isinstance(vq_loss_c, torch.Tensor):
            vq_total = vq_loss_c if vq_total is None else vq_total + vq_loss_c
        if isinstance(vq_loss_f0, torch.Tensor):
            vq_total = vq_loss_f0 if vq_total is None else vq_total + vq_loss_f0
        if isinstance(vq_total, torch.Tensor):
            try:
                loss_dict["vq"] = float(vq_total.item())
            except Exception:
                pass

    # ---- Bit-level bitrate diagnostics（Hash 模式下的 bit 熵/码率）----
    # 对 Hash / RVQ 统一计算 bit 级统计，以便：
    #   - 在 Hash 模式下直接作为第一性指标打印；
    #   - 在 RVQ 模式下仅作为 F0 熵正则的辅助，不再打印 br_* / hash_*。
    try:
        frame_rate = 16000.0 / 160.0  # 100 Hz

        # 内容分支 bit 统计
        hb_c = out.get("hash_bits_clean", None)  # [B,Lc,Kc]
        ceps_hat = out.get("ceps_hat", None)     # [B,Tc,Dc]
        tokens_per_frame_c = 0.0
        if isinstance(hb_c, torch.Tensor) and isinstance(ceps_hat, torch.Tensor):
            Bc, Lc, Kc = hb_c.shape
            _, Tc, _ = ceps_hat.shape
            Tc_eff = max(1, int(Tc))
            tokens_per_frame_c = float(Lc) / float(Tc_eff)

            actual_rate_c_bps = frame_rate * tokens_per_frame_c * float(Kc)

            bits_c = hb_c.detach()
            p1_c = (bits_c > 0).float().mean(dim=(0, 1))  # [Kc]
            p1_c = torch.clamp(p1_c, 1e-6, 1.0 - 1e-6)
            Hc = -(p1_c * torch.log2(p1_c) + (1.0 - p1_c) * torch.log2(1.0 - p1_c))  # [Kc]
            Hc_total = float(Hc.sum().item())  # bits / token
            entropy_rate_c_bps = frame_rate * tokens_per_frame_c * Hc_total

            # ---- Content 熵正则 / bit 平衡（Hash + RVQ 通用）----
            # 在 Hash 模式下：使用 bit 级熵 Hc_total；
            # 在 RVQ 模式下：优先使用索引熵 rvq_c_H（若存在）。
            lam_c_entropy = float(getattr(cfg, 'lambda_c_entropy', 0.0))
            if lam_c_entropy > 0.0:
                try:
                    alpha_c = float(getattr(cfg, 'content_entropy_target_frac', 0.5))
                    alpha_c = max(0.0, min(alpha_c, 1.0))
                    H_target_c = alpha_c * float(Kc)

                    if qtype == 'rvq' and 'rvq_c_H' in out:
                        Hc_eff_val = out['rvq_c_H']
                        if isinstance(Hc_eff_val, torch.Tensor):
                            Hc_eff = float(Hc_eff_val.detach().item())
                        else:
                            Hc_eff = float(Hc_eff_val)
                    else:
                        Hc_eff = float(Hc_total)

                    L_c_entropy = torch.nn.functional.relu(
                        torch.tensor(H_target_c - Hc_eff, device=hb_c.device, dtype=hb_c.dtype)
                    )
                    total = total + lam_c_entropy * L_c_entropy
                    loss_dict["content_entropy"] = float(L_c_entropy.item())
                except Exception:
                    pass

            # Bit balance：鼓励每一 bit 的 P(1) 接近 0.5，避免比特长期饱和。
            lam_bit_balance_c = float(getattr(cfg, 'lambda_bit_balance_c', 0.0))
            if lam_bit_balance_c > 0.0:
                try:
                    L_balance_c = torch.mean((p1_c - 0.5) ** 2)
                    total = total + lam_bit_balance_c * L_balance_c
                    loss_dict["bit_balance_c"] = float(L_balance_c.item())
                except Exception:
                    pass
        else:
            actual_rate_c_bps = 0.0
            entropy_rate_c_bps = 0.0

        # F0/VUV 分支 bit 统计
        hb_f = out.get("f0_hash_bits_clean", None)  # [B,Tf,Kf]
        tokens_per_frame_f = 0.0
        if isinstance(hb_f, torch.Tensor) and isinstance(ceps_hat, torch.Tensor):
            Bf, Tf, Kf = hb_f.shape
            Tf_eff = max(1, int(Tf))
            Tc_eff = max(1, int(ceps_hat.size(1)))
            tokens_per_frame_f = float(Tf_eff) / float(Tc_eff)

            actual_rate_f_bps = frame_rate * tokens_per_frame_f * float(Kf)

            bits_f = hb_f.detach()
            p1_f = (bits_f > 0).float().mean(dim=(0, 1))  # [Kf]
            p1_f = torch.clamp(p1_f, 1e-6, 1.0 - 1e-6)
            Hf = -(p1_f * torch.log2(p1_f) + (1.0 - p1_f) * torch.log2(1.0 - p1_f))  # [Kf]
            Hf_total = float(Hf.sum().item())
            entropy_rate_f_bps = frame_rate * tokens_per_frame_f * Hf_total

            # 额外：F0 熵正则 —— Hash 模式下基于 bit 熵，
            # RVQ 模式下改为基于 RVQ codebook 索引熵（rvq_f_H），
            # 更直接反映码本利用率/困惑度。
            lam_f0_entropy = float(getattr(cfg, 'lambda_f0_entropy', 0.0))
            if lam_f0_entropy > 0.0:
                try:
                    alpha = float(getattr(cfg, 'f0_entropy_target_frac', 0.5))
                    alpha = max(0.0, min(alpha, 1.0))
                    H_target = alpha * float(Kf)

                    # 有效熵 H_eff：
                    # - Hash 模式：使用 bit 级熵 Hf_total；
                    # - RVQ 模式：优先使用索引熵 rvq_f_H（若存在），
                    #   其上界同样为 sum(stage_bits)≈Kf。
                    if qtype == 'rvq' and 'rvq_f_H' in out:
                        H_eff_val = out['rvq_f_H']
                        if isinstance(H_eff_val, torch.Tensor):
                            H_eff = float(H_eff_val.detach().item())
                        else:
                            H_eff = float(H_eff_val)
                    else:
                        H_eff = float(Hf_total)

                    L_f0_entropy = torch.nn.functional.relu(
                        torch.tensor(H_target - H_eff, device=hb_f.device, dtype=hb_f.dtype)
                    )
                    total = total + lam_f0_entropy * L_f0_entropy
                    loss_dict["f0_entropy"] = float(L_f0_entropy.item())
                except Exception:
                    pass
        else:
            actual_rate_f_bps = 0.0
            entropy_rate_f_bps = 0.0

        total_actual_kbps = (actual_rate_c_bps + actual_rate_f_bps) / 1000.0
        total_entropy_kbps = (entropy_rate_c_bps + entropy_rate_f_bps) / 1000.0

        # Hash 模式下：bit 熵/码率作为第一性指标打印
        if qtype == 'hash':
            loss_dict["hash_tokens_per_frame_c"] = float(tokens_per_frame_c)
            loss_dict["hash_tokens_per_frame_f"] = float(tokens_per_frame_f)

            loss_dict["br_c_actual_kbps"] = float(actual_rate_c_bps / 1000.0)
            loss_dict["br_c_entropy_kbps"] = float(entropy_rate_c_bps / 1000.0)
            loss_dict["br_f0_actual_kbps"] = float(actual_rate_f_bps / 1000.0)
            loss_dict["br_f0_entropy_kbps"] = float(entropy_rate_f_bps / 1000.0)
            loss_dict["br_total_actual_kbps"] = float(total_actual_kbps)
            loss_dict["br_total_entropy_kbps"] = float(total_entropy_kbps)

            # 显式 hash_* 前缀，强调这是 hash 比特流码率
            loss_dict["hash_br_total_actual_kbps"] = float(total_actual_kbps)
            loss_dict["hash_br_total_entropy_kbps"] = float(total_entropy_kbps)
    except Exception:
        pass

    # RVQ 模式下：打印 RVQ 索引级诊断指标（来自模型前向），替代 bit 级码率日志。
    if qtype == 'rvq':
        try:
            for k in ("rvq_c_H", "rvq_c_usage", "rvq_c_perp",
                      "rvq_f_H", "rvq_f_usage", "rvq_f_perp"):
                if k in out:
                    v = out[k]
                    if isinstance(v, torch.Tensor):
                        loss_dict[k] = float(v.detach().item()) if v.numel() == 1 else float(v.mean().item())
                    else:
                        loss_dict[k] = float(v)
        except Exception:
            pass

    # 共有的 STFT 幅度提取（供 PHC / 对齐等使用）
    def _stft_mag(x: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
        x32 = x.to(torch.float32)
        win_t = torch.hann_window(win, device=x32.device, dtype=torch.float32)
        X = torch.stft(x32, n_fft=n_fft, hop_length=hop, win_length=win, window=win_t,
                       center=False, return_complex=True)
        mag = X.abs()  # [B, F, T]
        # Ensure frame count matches expected audio_length // hop_length
        B_, F_, T_ = mag.shape
        expected_frames = x32.size(-1) // hop
        if T_ > expected_frames:
            mag = mag[:, :, :expected_frames]
        return mag

    # ---- Residual Texture Loss (unvoiced + boundaries, occupancy + dynamics) ----
    def _dnn_pitch_to_hz(dp: torch.Tensor) -> torch.Tensor:
        # period = 256 / 2^(dp+1.5) clamped to [32,255]; f0 = 16000/period
        period = torch.clamp(256.0 / torch.pow(2.0, dp + 1.5), 32.0, 255.0)
        return 16000.0 / period

    lam_texture = float(getattr(cfg, 'lambda_texture_protect', 0.0))
    if lam_texture > 0.0:
        try:
            # Basic STFT mags
            y_hat = out['audio_hat']
            y_ref = out['audio']
            n_fft = 1024; hop = 160; sr = 16000; n_mels = 80
            Mag_g = _stft_mag(y_hat, n_fft=n_fft, hop=hop, win=n_fft)  # [B,F,T]
            Mag_r = _stft_mag(y_ref, n_fft=n_fft, hop=hop, win=n_fft)
            B, Fbins, Ts = Mag_g.shape
            # Build mel filterbank (HTK-like)
            def _hz_to_mel(f):
                return 2595.0 * torch.log10(1.0 + f / 700.0)
            def _mel_to_hz(m):
                return 700.0 * (10.0**(m / 2595.0) - 1.0)
            def _mel_fb(n_freqs: int, sr: int, n_mels: int) -> torch.Tensor:
                f_min = 0.0; f_max = sr / 2.0
                m_min = _hz_to_mel(torch.tensor(f_min, device=Mag_g.device, dtype=Mag_g.dtype))
                m_max = _hz_to_mel(torch.tensor(f_max, device=Mag_g.device, dtype=Mag_g.dtype))
                m_pts = torch.linspace(m_min, m_max, n_mels + 2, device=Mag_g.device, dtype=Mag_g.dtype)
                f_pts = _mel_to_hz(m_pts)
                freqs = torch.linspace(0.0, f_max, n_freqs, device=Mag_g.device, dtype=Mag_g.dtype)
                fb = torch.zeros(n_freqs, n_mels, device=Mag_g.device, dtype=Mag_g.dtype)
                for i in range(n_mels):
                    f_l, f_c, f_r = f_pts[i], f_pts[i+1], f_pts[i+2]
                    left = (freqs >= f_l) & (freqs <= f_c)
                    right = (freqs >= f_c) & (freqs <= f_r)
                    fb[left, i] = (freqs[left] - f_l) / (f_c - f_l + 1e-9)
                    fb[right, i] = (f_r - freqs[right]) / (f_r - f_c + 1e-9)
                fb = fb / (fb.sum(dim=0, keepdim=True) + 1e-9)
                return fb
            fb = _mel_fb(Fbins, sr, n_mels)  # [F,M]

            # Voicing and F0 (GT) for harmonic mask
            fc_ref = out.get('frame_corr')  # [B,T,1]
            dp_ref = out.get('dnn_pitch')   # [B,T,1]
            if isinstance(fc_ref, torch.Tensor) and isinstance(dp_ref, torch.Tensor):
                T = min(Mag_g.size(2), fc_ref.size(1), dp_ref.size(1))
                Mag_g = Mag_g[:, :, :T]
                Mag_r = Mag_r[:, :, :T]
                thr = float(getattr(cfg, 'vuv_threshold', 0.3))
                voiced = (fc_ref[:, :T, :].squeeze(-1) > thr)  # [B,T]
                unvoiced = (~voiced).to(Mag_g.dtype)
                f0_hz = _dnn_pitch_to_hz(dp_ref[:, :T, :].squeeze(-1))  # [B,T]

                # Harmonic mask H [B,F,T]
                Ffreqs = torch.linspace(0, sr/2, Fbins, device=Mag_g.device, dtype=Mag_g.dtype)
                K = int(getattr(cfg, 'harmonics_max', 5))
                ks = torch.arange(1, K+1, device=Mag_g.device, dtype=Mag_g.dtype).view(1, 1, K)
                centers = f0_hz.unsqueeze(-1) * ks  # [B,T,K]
                bw = float(getattr(cfg, 'harmonic_bandwidth_hz', 30.0))
                diff = Ffreqs.view(1, 1, 1, Fbins) - centers.unsqueeze(-1)
                Hk = torch.exp(-0.5 * (diff / bw) ** 2)  # [B,T,K,F]
                Hk = Hk / (Hk.sum(dim=-1, keepdim=True) + 1e-6)
                H = Hk.sum(dim=2)  # [B,T,F]
                H = (H * voiced.to(H.dtype).unsqueeze(-1))  # zero non-voiced
                H = H.permute(0, 2, 1).contiguous()  # [B,F,T]

                # Residual spectra (mask out harmonics)
                Mag_g_res = Mag_g * (1.0 - H)
                Mag_r_res = Mag_r * (1.0 - H)

                # To log-mel
                eps = float(getattr(cfg, 'texture_eps', 1e-4))
                mel_res_g = torch.log10(torch.matmul((Mag_g_res ** 2).transpose(1, 2), fb) + eps)  # [B,T,M]
                mel_res_r = torch.log10(torch.matmul((Mag_r_res ** 2).transpose(1, 2), fb) + eps)

                # HF region
                hf_start = int(getattr(cfg, 'texture_hf_start', 40))
                Xg = mel_res_g[:, :, hf_start:]  # [B,T,H]
                Xr = mel_res_r[:, :, hf_start:]
                if Xg.numel() > 0:
                    # Occupancy (where bright): compute 0.75 quantile over (T,H) by flattening dims
                    Bq = Xr.size(0)
                    tau = torch.quantile(
                        Xr.detach().reshape(Bq, -1), 0.75, dim=1, keepdim=True
                    ).view(Bq, 1, 1)
                    occupy_ref = (Xr > tau).to(Xg.dtype)
                    logit_pred = (Xg - tau) / 0.5
                    prob_pred = torch.sigmoid(logit_pred)
                    L_occ_all = F.binary_cross_entropy(prob_pred, occupy_ref, reduction='none')  # [B,T,H]
                    L_occ_t = L_occ_all.mean(dim=-1)  # [B,T]

                    # Dynamics over time (mean HF residual)
                    mu_g = Xg.mean(dim=-1)  # [B,T]
                    mu_r = Xr.mean(dim=-1)
                    if mu_g.size(1) > 1:
                        diff_g = torch.diff(mu_g, dim=1)
                        diff_r = torch.diff(mu_r, dim=1)
                        L_dyn_t = (diff_g - diff_r).abs()  # [B,T-1]
                    else:
                        L_dyn_t = torch.zeros(mu_g.size(0), 0, device=mu_g.device, dtype=mu_g.dtype)

                    # Unvoiced + boundary masks，叠加“非静音无声段”的高权重：
                    #   - vm: 基于 frame_corr 的有声掩膜
                    #   - unv: 无声掩膜（VUV 意义上的 unvoiced）
                    vm = voiced.to(Xg.dtype)  # [B,T]
                    unv = (1.0 - vm)  # [B,T]
                    boundary = torch.zeros_like(vm)
                    if vm.size(1) > 1:
                        boundary[:, 1:] = (vm[:, 1:] != vm[:, :-1]).to(vm.dtype)

                    # 基于静音掩膜的“非静音无声”掩膜：
                    #  - silence_mask: 真正静音（能量低且HF纹理弱）；
                    #  - unv_nonsil: 无声且非静音（典型是擦音/气声、虚线F0区域）。
                    if silence_mask is not None and silence_mask.dim() == 2:
                        if silence_mask.size(1) >= T:
                            sil_tex = silence_mask[:, :T].to(unv.dtype)
                        else:
                            sil_tex = torch.zeros_like(unv, dtype=unv.dtype, device=unv.device)
                            sil_tex[:, :silence_mask.size(1)] = silence_mask.to(unv.dtype)
                        unv_nonsil = unv * (1.0 - sil_tex)
                    else:
                        unv_nonsil = unv

                    # 基础权重：
                    #   - 无声段：1.0
                    #   - 无声∧非静音：额外提升（例如 ×2），鼓励保留这些虚线F0区的高频残差纹理；
                    #   - 边界帧：再叠加额外权重。总体上限制在 [0, 3]。
                    mask_tex = unv + 2.0 * boundary
                    mask_tex = mask_tex + unv_nonsil  # 为无声∧非静音额外+1权重
                    mask_tex = torch.clamp(mask_tex, max=3.0)

                    # Aggregate
                    denom_occ = mask_tex.sum() + 1e-6
                    L_occ = (L_occ_t * mask_tex).sum() / denom_occ
                    if L_dyn_t.numel() > 0:
                        mask_dyn = mask_tex[:, 1:]
                        denom_dyn = mask_dyn.sum() + 1e-6
                        L_dyn = (L_dyn_t * mask_dyn).sum() / denom_dyn
                    else:
                        L_dyn = torch.tensor(0.0, device=Xg.device, dtype=Xg.dtype)

                    tex_grad_w = float(getattr(cfg, 'texture_grad_weight', 0.5))
                    L_tex_res = L_occ + tex_grad_w * L_dyn
                    total = total + lam_texture * L_tex_res
                    loss_dict['texture_residual'] = float(L_tex_res.detach().item())
                    loss_tensors['texture_residual'] = (L_tex_res, 'audio')
                else:
                    loss_dict['texture_residual'] = 0.0
            else:
                loss_dict['texture_residual'] = 0.0
        except Exception as _e:
            loss_dict['texture_residual'] = 0.0
            if os.environ.get('DBG_TEXTURE', '0') == '1':
                try:
                    print(f"[TEXTURE_RES] skipped due to error: {_e}")
                except Exception:
                    pass

    # ---- F0 Pattern Preservation Loss: 保护无声段F0虚线模式 ----
    lam_f0_pattern = float(getattr(cfg, 'lambda_f0_pattern', 0.0))
    if lam_f0_pattern > 0.0:
        try:
            # 获取预测和参考F0
            dp_pred = out.get('dnn_pitch_hat')  # 预测F0 [B,T,1]
            dp_ref = out.get('dnn_pitch')       # 参考F0 [B,T,1]
            fc_pred = out.get('frame_corr_hat') # 预测VUV [B,T,1]
            fc_ref = out.get('frame_corr')      # 参考VUV [B,T,1]

            if all(x is not None for x in [dp_pred, dp_ref, fc_pred, fc_ref]):
                # 对齐时间长度
                T = min(dp_pred.size(1), dp_ref.size(1), fc_pred.size(1), fc_ref.size(1))
                if os.environ.get('DBG_F0_PATTERN', '0') == '1':
                    print(f"[F0_PATTERN] Tensor shapes before align: dp_pred={dp_pred.shape}, dp_ref={dp_ref.shape}")
                    print(f"[F0_PATTERN] T={T}")

                dp_pred = dp_pred[:, :T, :].squeeze(-1)  # [B,T]
                dp_ref = dp_ref[:, :T, :].squeeze(-1)
                fc_pred = fc_pred[:, :T, :].squeeze(-1)
                fc_ref = fc_ref[:, :T, :].squeeze(-1)

                # 转换为Hz
                def _dnn_pitch_to_hz_local(dp):
                    period = torch.clamp(256.0 / torch.pow(2.0, dp + 1.5), 32.0, 255.0)
                    return 16000.0 / period

                f0_pred_hz = _dnn_pitch_to_hz_local(dp_pred)  # [B,T]
                f0_ref_hz = _dnn_pitch_to_hz_local(dp_ref)

                # VUV阈值
                vuv_thr = float(getattr(cfg, 'vuv_threshold', -0.1))

                # 无声段和有声段分别处理
                voiced_mask = (fc_ref > vuv_thr).float()
                unvoiced_mask = 1.0 - voiced_mask

                L_f0_pattern = torch.tensor(0.0, device=f0_pred_hz.device)

                # 1. 无声段F0模式匹配（核心功能）
                if T > 1 and unvoiced_mask.sum() > 1.0:
                    # F0变化模式匹配
                    f0_diff_pred = torch.diff(f0_pred_hz, dim=1)  # [B,T-1]
                    f0_diff_ref = torch.diff(f0_ref_hz, dim=1)
                    unvoiced_diff_mask = unvoiced_mask[:, :-1]

                    # 在无声段，保持F0变化的方向一致性
                    if unvoiced_diff_mask.sum() > 0:
                        # 归一化变化幅度，关注相对模式而非绝对值
                        pred_changes = f0_diff_pred * unvoiced_diff_mask
                        ref_changes = f0_diff_ref * unvoiced_diff_mask

                        # 计算变化的符号一致性
                        sign_consistency = torch.abs(torch.sign(pred_changes) - torch.sign(ref_changes)) * unvoiced_diff_mask

                        # 计算变化幅度的相对一致性（归一化后比较）
                        pred_abs = torch.abs(pred_changes)
                        ref_abs = torch.abs(ref_changes)
                        # 避免除零，只在有显著变化的地方比较
                        significant_change_mask = (ref_abs > 10.0) * unvoiced_diff_mask  # 10Hz以上的变化
                        if significant_change_mask.sum() > 0:
                            # 相对幅度比较
                            pred_norm = pred_abs / (torch.max(pred_abs) + 1e-6)
                            ref_norm = ref_abs / (torch.max(ref_abs) + 1e-6)
                            magnitude_consistency = torch.abs(pred_norm - ref_norm) * significant_change_mask

                            L_f0_pattern = L_f0_pattern + sign_consistency.sum() / (unvoiced_diff_mask.sum() + 1e-6)
                            L_f0_pattern = L_f0_pattern + magnitude_consistency.sum() / (significant_change_mask.sum() + 1e-6)

                # 2. 轻量级纹理-F0协同约束
                pattern_synergy = float(getattr(cfg, 'f0_pattern_synergy_weight', 0.3))
                if pattern_synergy > 0.0:
                    # 使用已计算的mel谱图（如果可用）
                    if 'mel_energy' in loss_dict and loss_dict['mel_energy'] > 0:
                        # 简化版：在无声段，F0活跃度应该与音频能量变化相关
                        audio_hat = out.get('audio_hat')
                        audio_ref = out.get('audio')
                        if audio_hat is not None and audio_ref is not None and T > 1:
                            # 计算帧级能量变化
                            frame_len = 160
                            audio_frames_hat = audio_hat.unfold(1, frame_len, frame_len)  # [B, N_frames, frame_len]
                            N_frames = audio_frames_hat.size(1)
                            # 对齐音频帧数和特征帧数
                            T_audio = min(T, N_frames)
                            energy_hat = torch.mean(audio_frames_hat[:, :T_audio, :] ** 2, dim=2)  # [B,T_audio]

                            if T_audio > 1:
                                energy_changes = torch.abs(torch.diff(energy_hat, dim=1))  # [B,T_audio-1]
                                # 对齐F0和能量变化的时间维度
                                f0_pred_aligned = f0_pred_hz[:, :T_audio]  # [B,T_audio]
                                f0_changes = torch.abs(torch.diff(f0_pred_aligned, dim=1))  # [B,T_audio-1]

                                # 在无声段，能量变化和F0变化应该有一定相关性
                                # 对齐unvoiced_mask到T_audio-1的维度
                                unvoiced_audio_aligned = unvoiced_mask[:, :T_audio]  # [B,T_audio]
                                unvoiced_energy_mask = unvoiced_audio_aligned[:, :-1]  # [B,T_audio-1]
                                if unvoiced_energy_mask.sum() > 0:
                                    # 归一化后计算相关性
                                    energy_norm = F.normalize(energy_changes * unvoiced_energy_mask, dim=1)
                                    f0_norm = F.normalize(f0_changes * unvoiced_energy_mask, dim=1)
                                    synergy_loss = F.mse_loss(energy_norm, f0_norm) * pattern_synergy
                                    L_f0_pattern = L_f0_pattern + synergy_loss

                # 添加到总损失
                if L_f0_pattern.numel() > 0 and not torch.isnan(L_f0_pattern):
                    total = total + lam_f0_pattern * L_f0_pattern
                    loss_dict['f0_pattern'] = float(L_f0_pattern.detach().item())
                    loss_tensors['f0_pattern'] = (L_f0_pattern, 'f0')
                else:
                    loss_dict['f0_pattern'] = 0.0
            else:
                loss_dict['f0_pattern'] = 0.0

        except Exception as _e:
            loss_dict['f0_pattern'] = 0.0
            try:
                print(f"[F0_PATTERN] skipped due to error: {_e}")
            except Exception:
                pass

    # ---- F0 center continuity loss (inside 10%-90% of each voiced run) ----
    lam_f0_center = float(getattr(cfg, 'lambda_f0_center', 0.0))
    if lam_f0_center > 0.0:
        try:
            dp_pred = out.get('dnn_pitch_hat')  # [B,T,1]
            fc_ref = out.get('frame_corr')      # [B,T,1] (use GT to avoid shifting boundaries)
            if isinstance(dp_pred, torch.Tensor) and isinstance(fc_ref, torch.Tensor):
                B, T, _ = dp_pred.shape
                th = float(getattr(cfg, 'vuv_threshold', 0.3))
                voiced = (fc_ref[:, :T, 0] > th)  # [B,T] bool

                # map to Hz -> cents for scale invariance
                f0_hz = (16000.0 / torch.clamp(256.0 / torch.pow(2.0, dp_pred[:, :T, 0] + 1.5), 32.0, 255.0))
                # cents scale (avoid dependency on outer helper order)
                cent = 1200.0 * torch.log2(torch.clamp(f0_hz, min=1e-3) / 55.0)  # [B,T]

                # build center mask per voiced segment (keep only 10%-90% core)
                center = torch.zeros_like(voiced, dtype=torch.bool)
                dbg = os.environ.get('DBG_F0_CENTER', '0') == '1'
                seg_total = 0
                seg_with_core = 0
                if dbg:
                    try:
                        print(f"[F0_CENTER] B={B} T={T} thr={th} voiced_sum={int(voiced.sum().item())}")
                    except Exception:
                        pass
                for b in range(B):
                    v = voiced[b]
                    i = 0
                    while i < T:
                        if v[i]:
                            j = i
                            while j < T and v[j]:
                                j += 1
                            L = j - i
                            if L >= 5:
                                lk = int(max(1, math.floor(L * 0.10)))
                                rk = int(max(1, math.floor(L * 0.10)))
                                s = i + lk
                                e = j - rk
                                if e - s >= 3:
                                    center[b, s:e] = True
                                    seg_with_core += 1
                                seg_total += 1
                            i = j
                        else:
                            i += 1

                if center.any():
                    # second-order continuity on center frames (needs neighbors)
                    d2 = cent[:, 2:] - 2.0 * cent[:, 1:-1] + cent[:, :-2]  # [B,T-2]
                    m = center[:, 2:] & center[:, 1:-1] & center[:, :-2]
                    # Robust hinge normalization in cents: penalize curvature only if > tau
                    tau = float(getattr(cfg, 'f0_tv_delta_cents', 40.0))
                    ad2 = d2.abs()
                    norm = torch.relu(ad2 - tau) / (tau + 1e-6)
                    denom = m.float().sum() + 1e-6
                    l_cont = (norm * m.float()).sum() / denom
                    total = total + lam_f0_center * l_cont
                    loss_dict['f0_center'] = float(l_cont.item())
                    loss_tensors['f0_center'] = (l_cont, 'f0')
                    if dbg:
                        try:
                            frac = float(((ad2 > tau) & m).float().sum().item() / max(1.0, float(m.float().sum().item())))
                            print(f"[F0_CENTER] seg_total={seg_total} seg_with_core={seg_with_core} center_sum={int(center.sum().item())} denom={float(denom.item()):.1f} tau={tau:.1f} frac>|tau|={frac:.2f} loss={float(l_cont.item()):.6f}")
                        except Exception:
                            pass
                else:
                    loss_dict['f0_center'] = 0.0
                    if dbg:
                        try:
                            print(f"[F0_CENTER] skipped: no center frames (seg_total={seg_total}, seg_with_core={seg_with_core}, voiced_sum={int(voiced.sum().item())})")
                        except Exception:
                            pass
            else:
                loss_dict['f0_center'] = 0.0
                if os.environ.get('DBG_F0_CENTER', '0') == '1':
                    print(f"[F0_CENTER] skipped: missing tensors dp_pred={isinstance(dp_pred, torch.Tensor)} fc_ref={isinstance(fc_ref, torch.Tensor)}")
        except Exception as _e:
            loss_dict['f0_center'] = 0.0
            if os.environ.get('DBG_F0_CENTER', '0') == '1':
                print(f"[F0_CENTER] skipped due to error: {_e}")

    # ===== 额外：音频级F0一致性（soft、可导）+ 谐波对齐 =====
    

    # （已移除）波形域 F0 损失：f0_audio / f0_slope
    # 仅保留谐波对齐 harm_align，避免 mel 被抹平。

    def _harmonic_alignment(y_hat: torch.Tensor, y: torch.Tensor, f0_hat_hz: torch.Tensor,
                             voiced: torch.Tensor, sr: int = 16000, n_fft: int = 1024, hop: int = 160,
                             K: int = 5, bw_hz: float = 30.0) -> torch.Tensor:
        # y_hat,y: [B,L]; f0_hat_hz, voiced: [B,T]
        Mag_h = _stft_mag(y_hat, n_fft=n_fft, hop=hop, win=n_fft)  # [B,F,Ts]
        Mag_t = _stft_mag(y,     n_fft=n_fft, hop=hop, win=n_fft)
        B, F, Ts = Mag_h.shape
        # 对齐时间长度
        Ttrim = min(Ts, f0_hat_hz.size(1), voiced.size(1))
        if Ttrim <= 1:
            return torch.zeros((), device=y_hat.device)
        Mag_h = Mag_h[:, :, :Ttrim]
        Mag_t = Mag_t[:, :, :Ttrim]
        f0_hat_hz = f0_hat_hz[:, :Ttrim]
        voiced = voiced[:, :Ttrim]

        freqs = torch.linspace(0, sr / 2, F, device=Mag_h.device, dtype=Mag_h.dtype)  # [F]
        # Harmonic centers [B,T,K]
        ks = torch.arange(1, K + 1, device=Mag_h.device, dtype=Mag_h.dtype).view(1, 1, K)
        centers = f0_hat_hz.unsqueeze(-1) * ks  # [B,T,K]
        # Gaussian weights over freq bins -> [B,T,K,F]
        sigma = bw_hz
        diff = freqs.view(1, 1, 1, F) - centers.unsqueeze(-1)
        w = torch.exp(-0.5 * (diff / sigma) ** 2)
        # Normalize per (B,T,K)
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-6)
        # Energy around harmonics
        Eh = (w * Mag_h.permute(0, 2, 1).unsqueeze(2)).sum(dim=-1)  # [B,T,K]
        Et = (w * Mag_t.permute(0, 2, 1).unsqueeze(2)).sum(dim=-1)  # [B,T,K]
        # L1 diff masked by voiced
        mask = voiced.unsqueeze(-1)  # [B,T,1]
        denom = (mask.sum() * K + 1e-6)
        loss = (mask * (Eh - Et).abs()).sum() / denom
        return loss

    # 3.x) 高频加权 STFT（弱约束，强调 >hf_start 的细节，带 unvoiced/non-silence 门控）
    try:
        lam_hf = float(getattr(cfg, 'lambda_hf_stft', 0.0))
        if lam_hf > 0.0 and isinstance(out.get('audio_hat'), torch.Tensor) and isinstance(out.get('audio'), torch.Tensor):
            y_hat = out['audio_hat'].to(device)
            y_tgt = out['audio'].to(device)
            n_fft = 1024
            hop = 160
            Mag_h = _stft_mag(y_hat, n_fft=n_fft, hop=hop, win=n_fft)
            Mag_t = _stft_mag(y_tgt, n_fft=n_fft, hop=hop, win=n_fft)
            B, Fbins, Tm = Mag_h.shape
            sr = 16000.0
            freqs = torch.linspace(0, sr / 2.0, Fbins, device=Mag_h.device, dtype=Mag_h.dtype)
            f0 = float(getattr(cfg, 'hf_start_hz', 4000))
            p = float(getattr(cfg, 'hf_power', 2.0))
            w = torch.clamp(freqs / max(f0, 1.0), min=0.0, max=10.0).pow(p)
            w = w * (freqs >= f0).to(w.dtype)
            wn = w / (w.sum() + 1e-6)  # 频向归一化权重

            diff = torch.abs(Mag_h - Mag_t)

            # 帧级 gate：优先在“无声但非静音”的帧上约束（典型擦音/气声区域），
            # 若缺少 VUV 或静音掩膜，则退化为全时段约束。
            gate_t: Optional[torch.Tensor] = None  # [B,Tm]
            try:
                fc_ref = out.get('frame_corr', None)
                if isinstance(fc_ref, torch.Tensor) and fc_ref.dim() >= 2:
                    Tv = fc_ref.size(1)
                    T_use = min(Tm, Tv)
                    vuv_thr = float(getattr(cfg, 'vuv_threshold', 0.3))
                    v = (fc_ref[:, :T_use, 0] > vuv_thr)  # True=voiced
                    unv = (~v)
                    if silence_mask is not None and silence_mask.dim() == 2:
                        Ts = silence_mask.size(1)
                        Tm_use = min(T_use, Ts)
                        # 无声且非静音：典型对应擦音/气声、虚线 F0 区域
                        unv_nonsil = unv[:, :Tm_use] & (~silence_mask[:, :Tm_use])
                        gate_t = unv_nonsil.to(diff.dtype)
                    else:
                        gate_t = unv[:, :T_use].to(diff.dtype)
                elif silence_mask is not None and silence_mask.dim() == 2:
                    Ts = silence_mask.size(1)
                    T_use = min(Tm, Ts)
                    # 无静音门控时退化为“非静音”区域
                    gate_t = (~silence_mask[:, :T_use]).to(diff.dtype)
            except Exception:
                gate_t = None

            if gate_t is None:
                gate_t = torch.ones(B, Tm, device=diff.device, dtype=diff.dtype)
            else:
                # 若 STFT 帧数多于 gate_t，则简单截取；若更少则裁剪 gate
                if gate_t.size(1) < Tm:
                    pad = Tm - gate_t.size(1)
                    gate_t = torch.nn.functional.pad(gate_t, (0, pad))
                elif gate_t.size(1) > Tm:
                    gate_t = gate_t[:, :Tm]

            w_f = wn.view(1, Fbins, 1)
            w_t = gate_t.view(B, 1, Tm)
            wt = w_f * w_t
            denom = wt.sum() + 1e-6
            l_hf = (diff * wt).sum() / denom
            total = total + lam_hf * l_hf
            loss_dict['hf_stft'] = float(l_hf.item())
            loss_tensors['hf_stft'] = (l_hf, 'audio')
    except Exception:
        pass

    # ---- crepe-guided F0 envelope & presence losses (optional) ----
    def _extract_f0_batch(y: torch.Tensor, sr: int = 16000, hop: int = 160,
                          estimator: str = "auto", model: str = "tiny"):
        """Return (f0_hz, periodicity, f0_fb, fb_mask) as torch tensors on CPU for y: [B,L].
        Uses torchcrepe if available; falls back to librosa.pyin + yin fallback.
        """
        B, L = y.shape
        f0_list, p_list, fb_list, fbmask_list = [], [], [], []
        try:
            import librosa  # local import
        except Exception:
            librosa = None  # type: ignore
        try:
            import torchcrepe  # type: ignore
            has_crepe = True
        except Exception:
            torchcrepe = None
            has_crepe = False

        use_crepe = (estimator in ("auto", "crepe")) and has_crepe
        device_y = y.device

        for b in range(B):
            wav = y[b].detach()
            if use_crepe:
                try:
                    dev = wav.device if wav.is_cuda else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
                    x = wav.unsqueeze(0).to(dev)
                    with torch.no_grad():
                        f0_t, p_t = torchcrepe.predict(x, sr, hop, 50.0, 500.0, model,
                                                       batch_size=512, device=dev, return_periodicity=True)
                        # median/mean smoothing (window=3~5)
                        try:
                            f0_t = torchcrepe.filter.median(f0_t, 3)
                            p_t = torchcrepe.filter.median(p_t, 3)
                            f0_t = torchcrepe.filter.mean(f0_t, 3)
                            p_t = torchcrepe.filter.mean(p_t, 3)
                        except Exception:
                            pass
                    f0 = f0_t.squeeze(0).float().cpu().numpy()
                    p = p_t.squeeze(0).float().cpu().numpy()
                except Exception:
                    use_crepe = False  # fall through
            if not use_crepe:
                # pYIN fallback only
                if librosa is not None:
                    try:
                        f0, vflag, _ = librosa.pyin(wav.detach().cpu().numpy(), fmin=50, fmax=500,
                                                    sr=sr, hop_length=hop, frame_length=hop*4, fill_na=0.0)
                        f0 = np.where((f0 > 0) & (f0 >= 50) & (f0 <= 500), f0, 0.0)
                        p = vflag.astype(np.float32)
                    except Exception:
                        T = int(np.ceil(L / hop))
                        f0 = np.zeros((T,), dtype=np.float32)
                        p = np.zeros((T,), dtype=np.float32)
                else:
                    T = int(np.ceil(L / hop))
                    f0 = np.zeros((T,), dtype=np.float32)
                    p = np.zeros((T,), dtype=np.float32)
            # Fallback (YIN) based on librosa if available
            if librosa is not None:
                try:
                    fb = librosa.yin(wav.detach().cpu().numpy(), fmin=50, fmax=500, sr=sr,
                                     frame_length=hop*4, hop_length=hop)
                    fb = np.where((fb > 0) & (fb >= 50) & (fb <= 500), fb, 0.0).astype(np.float32)
                except Exception:
                    fb = np.zeros_like(f0, dtype=np.float32)
            else:
                fb = np.zeros_like(f0, dtype=np.float32)
            fb_mask = (fb > 0)
            f0_list.append(torch.from_numpy(f0))
            p_list.append(torch.from_numpy(p))
            fb_list.append(torch.from_numpy(fb))
            fbmask_list.append(torch.from_numpy(fb_mask.astype(np.float32)))

        # pad to same length
        T = min(min(t.size(0) for t in f0_list), y.size(-1) // hop)
        f0 = torch.stack([t[:T] for t in f0_list], 0)
        p = torch.stack([t[:T] for t in p_list], 0)
        fb = torch.stack([t[:T] for t in fb_list], 0)
        fb_mask = torch.stack([t[:T] for t in fbmask_list], 0)
        return f0.to(device), p.to(device), fb.to(device), fb_mask.to(device)

    def _to_cents(f_hz: torch.Tensor) -> torch.Tensor:
        return 1200.0 * torch.log2(torch.clamp(f_hz, min=1e-3) / 55.0)

    def _smooth_median(x: torch.Tensor, k: int) -> torch.Tensor:
        # quick median smoothing using unfold (odd k)
        if k <= 1 or x.size(1) < k:
            return x
        pad = k // 2
        xp = F.pad(x.unsqueeze(1), (pad, pad), mode='replicate').squeeze(1)  # [B,T+2p]
        xs = xp.unfold(1, k, 1)  # [B, T, k]
        return xs.median(dim=-1).values

    # Presence loss removed by request

    def _haar_dwt_coeffs(x: torch.Tensor, levels: int = 3) -> List[torch.Tensor]:
        """
        Simple Haar DWT detail coefficients for 1D sequences.
        x: [B,T] -> returns [d1, d2, ..., dL] each [B, T/2^j]
        """
        if levels <= 0:
            return []
        B, T = x.shape
        lo = torch.tensor([0.70710678, 0.70710678], device=x.device, dtype=x.dtype).view(1, 1, 2)
        hi = torch.tensor([-0.70710678, 0.70710678], device=x.device, dtype=x.dtype).view(1, 1, 2)
        a = x.unsqueeze(1)  # [B,1,T]
        coeffs: List[torch.Tensor] = []
        for _ in range(levels):
            # same padding with one-sample pad to both sides, then stride=2
            approx = F.conv1d(a, lo, stride=2, padding=1)
            detail = F.conv1d(a, hi, stride=2, padding=1)
            coeffs.append(detail.squeeze(1))
            a = approx
        return coeffs

    def _downsample_mask_avg(mask: torch.Tensor, levels: int) -> List[torch.Tensor]:
        """Downsample mask by averaging with kernel=2,stride=2 per level.
        Returns float weights in [0,1] for weighting losses per level.
        """
        if levels <= 0:
            return []
        m = mask.to(torch.float32).unsqueeze(1)  # [B,1,T]
        ker = torch.ones(1, 1, 2, device=m.device, dtype=m.dtype) * 0.5
        outs: List[torch.Tensor] = []
        for _ in range(levels):
            m = F.conv1d(m, ker, stride=2, padding=1)
            outs.append(m.squeeze(1))  # [B, T/2]
        return outs

    def _erode_mask(m: torch.Tensor, k: int = 3) -> torch.Tensor:
        """Simple 1D morphological erosion over time axis for [B,T] boolean mask.
        Keeps only frames whose k-neighborhood is fully voiced. Zero-padding at edges.
        """
        if k <= 1:
            return m
        m_f = m.to(torch.float32)
        pad = k // 2
        mp = F.pad(m_f.unsqueeze(1), (pad, pad), mode='constant', value=0.0)  # [B,1,T+2p]
        w = torch.ones(1, 1, k, device=m.device, dtype=m_f.dtype)
        conv = F.conv1d(mp, w, stride=1)
        core = (conv.squeeze(1) >= float(k) - 1e-6)
        # Fix: ensure output matches input size
        if core.shape != m.shape:
            core = core[:, :m.size(1)]
        return core & m

    def _dilate_mask(m: torch.Tensor, r: int = 1) -> torch.Tensor:
        """1D dilation over time axis for [B,T] boolean mask with radius r."""
        if r <= 0:
            return m
        m_f = m.to(torch.float32)
        k = 2 * r + 1
        mp = F.pad(m_f.unsqueeze(1), (r, r), mode='constant', value=0.0)
        w = torch.ones(1, 1, k, device=m.device, dtype=m_f.dtype)
        conv = F.conv1d(mp, w, stride=1)
        return conv.squeeze(1) > 0.0


    # 仅在输出包含音频时计算
    try:
        if cfg.lambda_harmonic > 0:
            y_hat = out["audio_hat"].to(device)
            y_tgt = out["audio"].to(device)
            B, L = y_hat.shape
            frame_corr = out.get("frame_corr")
            if frame_corr is not None:
                vmask_full = (frame_corr.to(device) > cfg.vuv_threshold).to(y_hat.dtype).squeeze(-1)
            else:
                Tframes = y_hat.shape[-1] // 160
                vmask_full = torch.ones(B, Tframes, device=device, dtype=y_hat.dtype)
            seq_frames = min(vmask_full.size(1), y_hat.size(-1) // 160)
            vmask_full = vmask_full[:, :seq_frames]
            # 获取周期序列（dnn_pitch_hat/dnn_pitch）并裁剪
            dp_any = out.get("dnn_pitch_hat", None)
            if not isinstance(dp_any, torch.Tensor):
                dp_any = out.get("dnn_pitch", None)
            if isinstance(dp_any, torch.Tensor):
                dp_any = dp_any[:, :seq_frames, :]
                f0_pred = (16000.0 / torch.clamp(256.0 / torch.pow(2.0, dp_any + 1.5), 32.0, 255.0)).squeeze(-1)
                try:
                    loss_harm = _harmonic_alignment(
                        y_hat, y_tgt, f0_pred,
                        (vmask_full > 0.5).to(y_hat.dtype),
                        sr=16000, n_fft=1024, hop=160,
                        K=cfg.harmonics_max,
                        bw_hz=cfg.harmonic_bandwidth_hz)
                except Exception as e:
                    print("[harmonic_debug]", e)
                    print("    y_hat:", y_hat.shape,
                          "y_tgt:", y_tgt.shape,
                          "f0_pred:", f0_pred.shape,
                          "vmask:", vmask_full.shape,
                          "dp_any:", dp_any.shape,
                          "seq_frames:", seq_frames)
                    raise
                total = total + cfg.lambda_harmonic * loss_harm
                loss_dict["harm_align"] = float(loss_harm.item())
                loss_tensors['harm_align'] = (loss_harm, 'audio')
        # HF distillation from teacher audio (if provided)
        lam_teacher = float(getattr(cfg, 'lambda_teacher_hf', 0.0))
        dbg_distill = os.environ.get('DBG_DISTILL', '0') == '1'
        has_teacher_audio = isinstance(out.get('audio_teacher'), torch.Tensor)
        if dbg_distill:
            try:
                keys = list(out.keys())
            except Exception:
                keys = []
            print(f"[DISTILL] lam={lam_teacher} has_teacher_audio={has_teacher_audio} keys={keys[:8]}...")
        if lam_teacher > 0.0 and has_teacher_audio:
            try:
                y_gen = out['audio_hat'].to(device)
                y_tch = out['audio_teacher'].to(device)
                n_fft = 1024; hop = 160
                Mag_g = _stft_mag(y_gen, n_fft=n_fft, hop=hop, win=n_fft)
                Mag_t = _stft_mag(y_tch, n_fft=n_fft, hop=hop, win=n_fft)
                # Align time frames to the shorter one
                Tg = Mag_g.size(2); Tt = Mag_t.size(2)
                Tm = min(Tg, Tt)
                if Tg != Tm:
                    Mag_g = Mag_g[:, :, :Tm]
                if Tt != Tm:
                    Mag_t = Mag_t[:, :, :Tm]
                B, Fbins, T = Mag_g.shape
                # Optional log-domain stabilization
                use_log = bool(getattr(cfg, 'teacher_hf_log', False))
                if use_log:
                    Mag_g_eff = torch.log1p(Mag_g)
                    Mag_t_eff = torch.log1p(Mag_t)
                else:
                    Mag_g_eff = Mag_g
                    Mag_t_eff = Mag_t
                freqs = torch.linspace(0, 16000.0/2.0, Fbins, device=Mag_g.device, dtype=Mag_g.dtype)
                hf0 = float(getattr(cfg, 'hf_start_hz', 4000.0))
                mask = (freqs >= hf0).to(Mag_g.dtype).view(1, Fbins, 1)        # [1,F,1]
                diff = (Mag_g_eff - Mag_t_eff).abs()
                norm_mode = str(getattr(cfg, 'teacher_hf_norm', 'bft_mean')).lower()
                if norm_mode == 'freq_only':
                    # Legacy (larger magnitude): normalize only by number of HF bins, sums over B and T
                    denom = mask.sum() + 1e-6
                    L_dist_hf = (mask * diff).sum() / denom
                else:
                    # Default: mean over B,F(masked),T
                    mask_bt = mask.expand(B, Fbins, T)
                    denom = mask_bt.sum() + 1e-6
                    L_dist_hf = (mask_bt * diff).sum() / denom

                # Internal gain (independent from lambda) to tune raw magnitude
                gain = float(getattr(cfg, 'teacher_hf_gain', 1.0))
                L_dist_hf = gain * L_dist_hf

                # Optional auto-balance to reach target RMS-Grad on audio_hat
                if bool(getattr(cfg, 'teacher_hf_auto_balance', False)):
                    try:
                        g = autograd.grad(L_dist_hf, y_gen, retain_graph=True, allow_unused=True)[0]
                        if isinstance(g, torch.Tensor):
                            gn = torch.sqrt(torch.mean(g.float().pow(2)) + 1e-12)
                            tgt = float(getattr(cfg, 'teacher_hf_gn_target', 5e-4))
                            # Compensate external lambda so final contribution hits target
                            lam_eff = float(getattr(cfg, 'lambda_teacher_hf', 0.0))
                            lam_eff = max(lam_eff, 1e-8)
                            tgt_internal = tgt / lam_eff
                            smin = float(getattr(cfg, 'teacher_hf_scale_min', 0.1))
                            smax = float(getattr(cfg, 'teacher_hf_scale_max', 10.0))
                            scale = torch.clamp(torch.tensor(tgt_internal, device=gn.device) / (gn + 1e-12), min=smin, max=smax)
                            L_dist_hf = (scale.detach() * L_dist_hf)
                            if dbg_distill:
                                print(f"[DISTILL] auto-balance: lam={lam_eff:.3g} gn_raw={float(gn.item()):.6e} -> target_final={tgt:.2e}, scale_int={float(scale.item()):.3f}")
                    except Exception as _e:
                        if dbg_distill:
                            print(f"[DISTILL] auto-balance error: {_e}")
                if dbg_distill:
                    try:
                        mdiff = diff.mean().item()
                        mode_note = f"{'log1p' if use_log else 'lin'}+{norm_mode}"
                        print(f"[DISTILL] y_gen={tuple(y_gen.shape)} y_tch={tuple(y_tch.shape)} Mag_g={tuple(Mag_g.shape)} Mag_t={tuple(Mag_t.shape)} (Tg={Tg}, Tt={Tt}, Tm={Tm}) hf0={hf0} mode={mode_note} mask_F={float(mask.sum().item()):.1f} denom={float(denom.item()):.1f} mean|Δ|={mdiff:.6f} gain={gain:.3f} L={float(L_dist_hf.item()):.6f}")
                    except Exception as _e:
                        print(f"[DISTILL] debug-print error: {_e}")
                total = total + lam_teacher * L_dist_hf
                loss_dict['distill_hf'] = float(L_dist_hf.item())
                loss_tensors['distill_hf'] = (L_dist_hf, 'audio')
            except Exception as _e:
                if dbg_distill:
                    print(f"[DISTILL] error: {_e}")
                loss_dict['distill_hf'] = 0.0
        elif lam_teacher > 0.0 and not has_teacher_audio:
            if dbg_distill:
                print("[DISTILL] audio_teacher missing; skip distillation")
        # crepe-guided envelope/presence losses
        need_env = (
            (getattr(cfg, 'lambda_f0_env', 0.0) > 0.0)
            or (getattr(cfg, 'lambda_f0_slope', 0.0) > 0.0)
            or (float(getattr(cfg, 'lambda_f0_tv', 0.0)) > 0.0)
            or (float(getattr(cfg, 'lambda_f0_wavelet', 0.0)) > 0.0)
        )
        dbg_slope_flag = os.environ.get('DBG_F0_SLOPE', '0') == '1'
        if (not need_env) and dbg_slope_flag:
            try:
                print(f"[F0_SLOPE] need_env=False (lam_env={getattr(cfg,'lambda_f0_env',0.0)}, lam_slope={getattr(cfg,'lambda_f0_slope',0.0)}, lam_tv={getattr(cfg,'lambda_f0_tv',0.0)}, lam_wav={getattr(cfg,'lambda_f0_wavelet',0.0)})")
            except Exception:
                pass
        if need_env:
            B, L = audio_hat.shape
            # Extract F0 from target and prediction (no_grad)
            with torch.no_grad():
                f0_t, p_t, f0_fb_t, fb_mask_t = _extract_f0_batch(audio_real, sr=16000, hop=160,
                                                                  estimator=getattr(cfg, 'f0_estimator', 'auto'),
                                                                  model=getattr(cfg, 'f0_estimator_model', 'tiny'))
                f0_h, p_h, f0_fb_h, fb_mask_h = _extract_f0_batch(audio_hat, sr=16000, hop=160,
                                                                  estimator=getattr(cfg, 'f0_estimator', 'auto'),
                                                                  model=getattr(cfg, 'f0_estimator_model', 'tiny'))
            # ----- Simplified envelope/center -----
            thr = 0.35
            # 统一参考序列长度 - include f0_h in alignment
            T_ref = min(f0_t.size(1), f0_fb_t.size(1), f0_h.size(1))
            f0_t = f0_t[:, :T_ref]
            f0_h = f0_h[:, :T_ref]
            p_t = p_t[:, :T_ref]
            p_h = p_h[:, :T_ref]
            core_full = _erode_mask((p_t > thr) & (f0_t > 60.0) & (f0_t < 450.0), k=2)
            w = int(getattr(cfg, 'f0_env_window', 5))
            k = max(3, w if (w % 2 == 1) else (w + 1))
            f0_smooth = _smooth_median(f0_t, k)
            cent_ref = _to_cents(torch.clamp(f0_smooth, min=1e-3))
            base_m = float(getattr(cfg, 'f0_env_margin_cents', 80.0))
            margin_full = cent_ref.new_full(cent_ref.shape, base_m)

            # Predicted f0 from dnn_pitch_hat / dnn_pitch（统一对齐 Tm 后再计算所有项）
            dp_any = out.get("dnn_pitch_hat", None)
            if not isinstance(dp_any, torch.Tensor):
                dp_any = out.get("dnn_pitch", None)
            if isinstance(dp_any, torch.Tensor):
                Tm = min(cent_ref.size(1), core_full.size(1), p_t.size(1), dp_any.size(1), L // 160)
                cent_env = cent_ref[:, :Tm]
                core_t = core_full[:, :Tm]
                p_conf = torch.clamp(p_t[:, :Tm], 0.0, 1.0)
                margin_t = margin_full[:, :Tm]
                dp       = dp_any[:, :Tm, :]
                # 预测 F0 → cents
                f0_pred  = (16000.0 / torch.clamp(256.0 / torch.pow(2.0, dp + 1.5), 32.0, 255.0)).squeeze(-1)
                # Defensive alignment: ensure f0_pred has expected shape
                if f0_pred.size(1) != Tm:
                    f0_pred = f0_pred[:, :Tm]
                cent_pred= _to_cents(f0_pred)
                # Additional defensive alignment: ensure cent_pred matches the trimmed length
                if cent_pred.size(1) != Tm:
                    cent_pred = cent_pred[:, :Tm]
                # 简化包络：core（CREPE voiced）与预测 V/UV 的软融合
                # 软门：m_soft = alpha*core + (1-alpha)*sigmoid((fc_hat-th)/tau)
                m_env    = core_t.to(f0_pred.dtype)
                fc_hat_any = out.get('frame_corr_hat', None)
                if isinstance(fc_hat_any, torch.Tensor):
                    fc_hat_t = fc_hat_any[:, :Tm, 0]
                    tau = float(getattr(cfg, 'vuv_ce_tau', 0.15))
                    th  = float(getattr(cfg, 'vuv_threshold', 0.3))
                    q_hat = torch.sigmoid((fc_hat_t - th) / max(tau, 1e-6))
                    alpha = float(getattr(cfg, 'vuv_ce_alpha', 0.7))
                    m_env = alpha * m_env + (1.0 - alpha) * q_hat.to(m_env.dtype)
                w_env    = torch.sqrt(p_conf) * m_env
                err_c    = torch.abs(cent_pred - cent_env)
                norm_h   = torch.relu(err_c - margin_t.to(err_c.dtype)) / (margin_t.to(err_c.dtype) + 1e-6)
                active   = (w_env > 0)
                if active.any():
                    q = torch.quantile(norm_h[active], 0.90)
                    norm_h = torch.minimum(norm_h, q)
                denom = w_env.sum() + 1e-6
                if getattr(cfg, 'lambda_f0_env', 0.0) > 0.0:
                    l_env = (norm_h * w_env).sum() / denom
                    total = total + (float(cfg.lambda_f0_env) * float(getattr(cfg, '_isolate_f0_scale', 1.0))) * l_env
                    loss_dict["f0_env"] = float(l_env.item())
                    loss_tensors['f0_env'] = (l_env, 'f0')
                # Presence loss disabled
                # 轻量斜率（归一化 L1 跟随）：仅 core 内
                if getattr(cfg, 'lambda_f0_slope', 0.0) > 0.0 and cent_env.size(1) > 1:
                    d_pred = cent_pred[:, 1:] - cent_pred[:, :-1]
                    d_env  = cent_env[:, 1:] - cent_env[:, :-1]
                    core_pair = core_t[:, 1:] & core_t[:, :-1]
                    w_s = torch.sqrt(p_conf[:, 1:]) * core_pair.to(p_conf.dtype)
                    # tau_min from TV delta: tau_min = max(5c, 0.25 * f0_tv_delta_cents)
                    tau_min = max(5.0, 0.25 * float(getattr(cfg, 'f0_tv_delta_cents', 40.0)))
                    diff_s = torch.abs(d_pred - d_env)
                    denom_s = torch.abs(d_env) + tau_min
                    norm_s = diff_s / denom_s
                    active_s = (w_s > 0)
                    if active_s.any():
                        q_s = torch.quantile(norm_s[active_s], 0.90)
                        norm_s = torch.minimum(norm_s, q_s)
                    l_slope = (norm_s * w_s).sum() / (w_s.sum() + 1e-6)
                    total = total + float(cfg.lambda_f0_slope) * l_slope
                    loss_dict["f0_slope"] = float(l_slope.item())
                    loss_tensors['f0_slope'] = (l_slope, 'f0')
                    if dbg_slope_flag:
                        try:
                            def _stat1d(t):
                                t = t.reshape(-1)
                                return float(t.min().item()), float(t.max().item()), float(t.mean().item())
                            cp = int(core_pair.sum().item())
                            ws = float(w_s.sum().item())
                            dm = diff_s[core_pair] if core_pair.any() else diff_s.reshape(-1)
                            de = denom_s[core_pair] if core_pair.any() else denom_s.reshape(-1)
                            df_min, df_max, df_mean = _stat1d(dm)
                            de_min, de_max, de_mean = _stat1d(de)
                            print(f"[F0_SLOPE] Tm={Tm} core_pair={cp} w_sum={ws:.1f} diff=[{df_min:.1f},{df_max:.1f},{df_mean:.1f}] denom=[{de_min:.1f},{de_max:.1f},{de_mean:.1f}] loss={float(l_slope.item()):.6f}")
                        except Exception:
                            pass

                # 4) small second-order TV on cent_pred (within core only)
                lambda_tv = float(getattr(cfg, 'lambda_f0_tv', 0.0))
                if lambda_tv > 0.0 and cent_pred.size(1) > 2:
                    core_tv = core_t[:, 2:] & core_t[:, 1:-1] & core_t[:, :-2]
                    if core_tv.any():
                        c = cent_pred
                        d1 = c[:, 1:] - c[:, :-1]
                        d2 = d1[:, 1:] - d1[:, :-1]
                        w_tv = torch.sqrt(p_conf[:, 2:]) * core_tv.to(c.dtype)
                        # robust, normalized hinge on |d2| in cents with delta threshold
                        delta = float(getattr(cfg, 'f0_tv_delta_cents', 40.0))
                        abs_d2 = torch.abs(d2)
                        norm_tv = torch.relu(abs_d2 - delta) / (delta + 1e-6)
                        l_tv = (norm_tv * w_tv).sum() / (w_tv.sum() + 1e-6)
                        total = total + lambda_tv * l_tv
                        loss_dict["f0_tv"] = float(l_tv.item())

                # V/UV 一致性：frame_corr_hat ↔ CREPE periodicity on generated audio
                lam_vcre = float(getattr(cfg, 'lambda_vuv_crepe', 0.0))
                if lam_vcre > 0.0 and isinstance(out.get('frame_corr_hat', None), torch.Tensor):
                    try:
                        fc_hat_t = out['frame_corr_hat'][:, :Tm, 0]
                        q_ref = torch.clamp(p_h[:, :Tm], 0.0, 1.0)
                        tau = float(getattr(cfg, 'vuv_ce_tau', 0.15))
                        th  = float(getattr(cfg, 'vuv_threshold', 0.3))
                        s = (fc_hat_t - th) / max(tau, 1e-6)
                        bce = torch.nn.functional.binary_cross_entropy_with_logits(s, q_ref)
                        total = total + lam_vcre * bce
                        loss_dict['vuv_ce'] = float(bce.item())
                        loss_tensors['vuv_ce'] = (bce, 'vuv')
                    except Exception:
                        loss_dict['vuv_ce'] = 0.0

                # F0 Presence Loss: 强制网络预测有声段存在
                lam_presence = float(getattr(cfg, 'lambda_f0_presence', 0.0))
                if lam_presence > 0.0:
                    try:
                        # 获取模型预测的frame_corr (VUV)
                        fc_hat = out.get('frame_corr_hat')  # [B,T,1]
                        if fc_hat is not None:
                            fc_hat_flat = fc_hat[:, :Tm, 0]  # [B,T]

                            # 从CREPE periodicity获取真实的有声段标签
                            vuv_target = (p_t[:, :Tm] > 0.5).float()  # [B,T] 1=有声，0=无声

                            # 使用focal loss来处理类别不平衡
                            gamma = float(getattr(cfg, 'f0_presence_gamma', 2.0))

                            # 将frame_corr转换为概率 (sigmoid)
                            vuv_prob = torch.sigmoid(fc_hat_flat)  # [B,T]

                            # Focal loss计算
                            ce = -vuv_target * torch.log(vuv_prob + 1e-8) - (1 - vuv_target) * torch.log(1 - vuv_prob + 1e-8)
                            pt = torch.where(vuv_target == 1, vuv_prob, 1 - vuv_prob)
                            focal_weight = (1 - pt) ** gamma
                            focal_loss = focal_weight * ce

                            # 只对高置信度的CREPE检测结果计算损失
                            conf_mask = (p_conf[:, :Tm] > 0.3).float()
                            weighted_loss = focal_loss * conf_mask

                            l_presence = weighted_loss.sum() / (conf_mask.sum() + 1e-6)
                            total = total + lam_presence * l_presence
                            loss_dict["f0_presence"] = float(l_presence.item())
                            loss_tensors['f0_presence'] = (l_presence, 'vuv')
                        else:
                            loss_dict["f0_presence"] = 0.0
                    except Exception as e:
                        loss_dict["f0_presence"] = 0.0
                        print(f"[F0_PRESENCE] Error: {e}")

                # 5) Cross-voicing Wavelet constraint: handle voicing misclassification
                lam_wav = float(getattr(cfg, 'lambda_f0_wavelet', 0.0))
                if lam_wav > 0.0:
                    cp = cent_pred  # Model prediction
                    ct = cent_env   # CREPE reference (ground truth voicing)

                    # Determine ground truth voicing from CREPE reference
                    # Assume CREPE periodicity p_t indicates true voicing
                    true_voiced = p_t > 0.5    # Ground truth voicing from CREPE
                    pred_voiced = m_env > 0.5  # Predicted voicing from model

                    # Cross-voicing analysis: different prediction vs ground truth combinations
                    correct_voiced = true_voiced & pred_voiced      # Both agree: voiced
                    correct_unvoiced = (~true_voiced) & (~pred_voiced)  # Both agree: unvoiced
                    false_negative = true_voiced & (~pred_voiced)   # Model missed voiced (重要情况!)
                    false_positive = (~true_voiced) & pred_voiced  # Model hallucinated voiced

                    # Statistics for understanding misclassification
                    correct_v_count = correct_voiced.sum() if correct_voiced.any() else 0
                    false_neg_count = false_negative.sum() if false_negative.any() else 0
                    false_pos_count = false_positive.sum() if false_positive.any() else 0

                    # print(f"[wavelet_debug] Voicing: correct_voiced={correct_v_count}, false_neg={false_neg_count}, false_pos={false_pos_count}")

                    # Compute errors for different misclassification cases
                    e_total = cp - ct

                    # Adaptive weighting based on voicing classification quality
                    region_weights = torch.zeros_like(m_env, dtype=torch.float32)

                    # Correct voiced predictions: high confidence, full weight
                    region_weights = torch.where(correct_voiced, 1.0, region_weights)

                    # False negatives: model thinks unvoiced but CREPE says voiced
                    # This is the key insight - we should constrain the "unvoiced" prediction to match CREPE
                    region_weights = torch.where(false_negative, 0.8, region_weights)  # Strong weight to learn from missed voiced

                    # Connection regions where both show some voicing tendency
                    weak_voiced_regions = ((p_t > 0.2) | (m_env > 0.2)) & (~correct_voiced) & (~false_negative)
                    region_weights = torch.where(weak_voiced_regions, 0.3, region_weights)  # Moderate weight for transitions

                    # Apply region-weighted error
                    e_weighted = e_total * region_weights

                    # Optional: tiny bias penalty to reduce constant offset drift
                    lam_bias = float(getattr(cfg, 'lambda_f0_bias', 0.0))
                    if lam_bias > 0.0:
                        clip_c = float(getattr(cfg, 'f0_wav_clip_cents', 120.0))
                        # per-sample weighted mean error (in cents)
                        denom_b = region_weights.sum(dim=1) + 1e-6
                        mu = (e_total * region_weights).sum(dim=1) / denom_b  # [B]
                        loss_bias = (mu.abs() / (clip_c + 1e-6)).mean()
                        total = total + lam_bias * loss_bias
                        loss_dict['f0_bias'] = float(loss_bias.item())

                    # Detailed analysis for different error types
                    if correct_voiced.any():
                        corr_v_mean = torch.abs(e_total[correct_voiced]).mean()
                        # print(f"[wavelet_debug] Correct voiced: mean_error={corr_v_mean:.2f} cents, frames={correct_v_count}")

                    if false_negative.any():
                        fn_mean = torch.abs(e_total[false_negative]).mean()
                        fn_max = torch.abs(e_total[false_negative]).max()
                        # print(f"[wavelet_debug] False negative (missed voiced): mean={fn_mean:.2f}, max={fn_max:.2f} cents, frames={false_neg_count}")
                        # print(f"[wavelet_debug] -> Constraining 'unvoiced' prediction to match CREPE voiced reference")

                    if false_positive.any():
                        fp_mean = torch.abs(e_total[false_positive]).mean()
                        # print(f"[wavelet_debug] False positive (hallucinated): mean_error={fp_mean:.2f} cents, frames={false_pos_count}")

                    # Check if we have sufficient constraints
                    total_weight = region_weights.sum()
                    if total_weight > 1e-5:
                        # Moderate clipping - allow natural F0 variations
                        # Now configurable via cfg.f0_wav_clip_cents (default 120c)
                        clip_c = float(getattr(cfg, 'f0_wav_clip_cents', 120.0))
                        e_clipped = torch.clamp(e_weighted, min=-clip_c, max=clip_c)
                        e_normalized = e_clipped / 60.0  # Scale by moderate F0 error

                        levels = int(getattr(cfg, 'f0_wav_levels', 3))
                        coeffs = _haar_dwt_coeffs(e_normalized, levels=max(1, levels))
                        masks = _downsample_mask_avg(region_weights, levels=max(1, levels))

                        # Frequency-selective alphas: emphasize smoothness over absolute accuracy
                        alphas = getattr(cfg, 'f0_wav_alphas', None)
                        if not alphas:
                            # Focus on reducing jitter while allowing natural contour
                            alphas = [1.2, 0.7, 0.3]  # Balanced weights for cross-voicing scenarios

                        # ensure length
                        if len(alphas) < len(coeffs):
                            last = alphas[-1]
                            alphas = alphas + [last] * (len(coeffs) - len(alphas))

                        loss_w = 0.0
                        for j, d in enumerate(coeffs):
                            wj = masks[j].to(d.dtype)
                            denom = wj.sum() + 1e-6
                            if denom > 1e-5:
                                level_loss = ((d ** 2) * wj).sum() / denom
                                # print(f"[wavelet_debug] Level {j}: loss={level_loss:.4f}, alpha={alphas[j]}, active_ratio={wj.mean():.3f}")
                                loss_w = loss_w + float(alphas[j]) * level_loss

                        total = total + (lam_wav * float(getattr(cfg, '_isolate_f0_scale', 1.0))) * loss_w
                        loss_dict['f0_wavelet'] = float(loss_w.item())
                        loss_tensors['f0_wavelet'] = (loss_w, 'f0')
                    else:
                        # print("[wavelet_debug] Insufficient constraint regions detected")
                        loss_dict['f0_wavelet'] = 0.0
            else:
                # No predicted F0 available; skip envelope-related losses gracefully
                if getattr(cfg, 'lambda_f0_env', 0.0) > 0.0:
                    loss_dict["f0_env"] = 0.0
                if getattr(cfg, 'lambda_f0_slope', 0.0) > 0.0:
                    loss_dict["f0_slope"] = 0.0
                if float(getattr(cfg, 'lambda_f0_tv', 0.0)) > 0.0:
                    loss_dict["f0_tv"] = 0.0
                if float(getattr(cfg, 'lambda_f0_presence', 0.0)) > 0.0:
                    loss_dict["f0_presence"] = 0.0
        # 5) F0 peakiness (prevent over-flattening on voiced frames)
        lam_peak = float(getattr(cfg, 'lambda_f0_peak', 0.0))
        if lam_peak > 0.0:
            dp_any = out.get('dnn_pitch_hat', None)
            if not isinstance(dp_any, torch.Tensor):
                dp_any = out.get('dnn_pitch', None)
            vuv_fc = out.get('frame_corr', None)
            if isinstance(dp_any, torch.Tensor) and isinstance(vuv_fc, torch.Tensor):
                Bp, Tp, _ = dp_any.shape
                Tv = vuv_fc.shape[1]
                Tpk = min(Tp, Tv)
                dp_p = dp_any[:, :Tpk, :]
                # map to Hz and then to cents
                f0_p = (16000.0 / torch.clamp(256.0 / torch.pow(2.0, dp_p + 1.5), 32.0, 255.0)).squeeze(-1)
                cent_p = _to_cents(f0_p)
                mask_v = (vuv_fc.to(device) > cfg.vuv_threshold).squeeze(-1)[:, :Tpk]
                if mask_v.any():
                    vals = cent_p[mask_v]
                    mu = vals.mean()
                    var = ((vals - mu) ** 2).mean()
                    target_var = (30.0 ** 2)
                    loss_peak = torch.relu(target_var - var) / (target_var + 1e-6)
                    total = total + (lam_peak * float(getattr(cfg, '_isolate_f0_scale', 1.0))) * loss_peak
                    loss_dict['f0_peak'] = float(loss_peak.item())
                    loss_tensors['f0_peak'] = (loss_peak, 'f0')
    except Exception as e:
        def _shape(x):
            try:
                return tuple(x.shape)
            except Exception:
                return None
        dp_debug = out.get('dnn_pitch_hat', None)
        if not isinstance(dp_debug, torch.Tensor):
            dp_debug = out.get('dnn_pitch', None)
        print(f"[ExtraLoss] Skip f0/harmonic losses due to error: {e}")
        print("    Shapes => y_hat:", _shape(out.get('audio_hat')),
              "frame_corr:", _shape(out.get('frame_corr')),
              "vmask:", _shape(locals().get('vmask')),
              "f0_t:", _shape(locals().get('f0_t')),
              "core0:", _shape(locals().get('core0')),
              "cent_ref:", _shape(locals().get('cent_ref')),
              "dp_any:", _shape(dp_debug))

    # 参考项梯度尺度（对 refined mel 的最后层梯度）
    if bool(getattr(cfg, 'adaptive_hf', False)):
        try:
            gref = autograd.grad(loss_mel_struct, mel_hat_refined, retain_graph=True, allow_unused=True)[0]
            if isinstance(gref, torch.Tensor):
                gv = torch.nan_to_num(gref.detach(), nan=0.0, posinf=0.0, neginf=0.0)
                grad_info['g_ref'] = float(gv.abs().mean().item())
        except Exception:
            pass

    # 梯度范围调查（统一度量：RMS-Grad；按域使用统一掩膜规则）
    if bool(getattr(cfg, 'grad_survey', False)):
        anchors: Dict[str, torch.Tensor] = {}
        anchors['mel'] = mel_hat_refined
        if isinstance(out.get('ceps_hat'), torch.Tensor):
            anchors['ceps'] = out['ceps_hat']
        if isinstance(out.get('audio_hat'), torch.Tensor):
            anchors['audio'] = out['audio_hat']
        if isinstance(out.get('dnn_pitch_hat'), torch.Tensor):
            anchors['f0'] = out['dnn_pitch_hat']
        if isinstance(out.get('frame_corr_hat'), torch.Tensor):
            anchors['vuv'] = out['frame_corr_hat']

        def _build_mask(domain: str, anchor: torch.Tensor) -> torch.Tensor:
            # 统一掩膜（简化）：
            # - mel: 仅高频（F>mel_hp_low_bins），不使用有声/软门控
            # - f0: 有声掩膜
            # - 其余：全1
            if domain == 'mel':
                Bm, Tm, Fm = anchor.shape
                lb = int(getattr(cfg, 'mel_hp_low_bins', 16))
                lb = max(1, min(lb, Fm - 1))
                Mf = torch.zeros(Bm, Tm, Fm, dtype=anchor.dtype, device=anchor.device)
                Mf[:, :, lb:] = 1.0
                return Mf
            elif domain == 'f0':
                Bf, Tf, _ = anchor.shape
                th = float(getattr(cfg, 'vuv_threshold', 0.3))
                fc = out.get('frame_corr', None)
                if isinstance(fc, torch.Tensor):
                    M = (fc > th).squeeze(-1).to(anchor.dtype)
                else:
                    M = torch.ones(Bf, Tf, dtype=anchor.dtype, device=anchor.device)
                return M.unsqueeze(-1)
            else:
                return torch.ones_like(anchor)

        for name, (lt, domain) in list(loss_tensors.items()):
            try:
                anchor = anchors.get(domain, None)
                if isinstance(lt, torch.Tensor) and isinstance(anchor, torch.Tensor):
                    g = autograd.grad(lt, anchor, retain_graph=True, allow_unused=True)[0]
                    if isinstance(g, torch.Tensor):
                        gv = torch.nan_to_num(g.detach(), nan=0.0, posinf=0.0, neginf=0.0)
                        M = _build_mask(domain, anchor)
                        # RMS 统一度量
                        num = (gv.pow(2) * M).sum()
                        den = M.sum() + 1e-6
                        gn_val = torch.sqrt(num / den).item()
                        grad_info[f'gn_{name}'] = float(gn_val)
            except Exception:
                pass

    return total, loss_dict, grad_info

def save_checkpoint(
    cfg: SupportConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    mpd: Optional[torch.nn.Module] = None,
    msd: Optional[torch.nn.Module] = None,
    optimizer_hifi_disc: Optional[torch.optim.Optimizer] = None,
) -> str:
    """
    保存当前训练状态：
      - 模型权重
      - optimizer 状态
      - epoch / global_step
      - 训练配置 cfg
    返回保存的文件路径，方便打印。
    """
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(
        cfg.ckpt_dir,
        f"checkpoint_step_{global_step:06d}_epoch_{epoch:02d}.pth"
    )

    state = {
        "cfg": asdict(cfg),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }

    # Optional: persist HiFi-GAN style discriminators when enabled
    if mpd is not None:
        state["mpd"] = mpd.state_dict()
    if msd is not None:
        state["msd"] = msd.state_dict()
    if optimizer_hifi_disc is not None:
        state["optimizer_hifi_disc"] = optimizer_hifi_disc.state_dict()
    torch.save(state, ckpt_path)
    return ckpt_path


def _aggregate_jscc_fsk_metrics(out_dir: Path) -> Dict[str, float]:
    """Aggregate metrics from *_metrics.json files under out_dir.

    Expected keys (if present): STOI/PESQ/f0_mse/mel_mse. The function is
    defensive: it will average whatever subset of these keys it finds.
    """

    metric_names = ["STOI", "stoi", "PESQ", "pesq", "f0_mse", "mel_mse"]
    accum: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    for json_path in out_dir.glob("*metrics.json"):
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        for key in metric_names:
            if key in data and isinstance(data[key], (int, float)):
                v = float(data[key])
                accum[key] = accum.get(key, 0.0) + v
                counts[key] = counts.get(key, 0) + 1

    if not accum:
        return {}

    summary: Dict[str, float] = {}
    for key, total in accum.items():
        n = counts.get(key, 0)
        if n > 0:
            summary[key] = total / float(n)

    def _pick(*names: str) -> Optional[float]:
        for n in names:
            if n in summary:
                return summary[n]
        return None

    result: Dict[str, float] = {}
    stoi = _pick("STOI", "stoi")
    pesq = _pick("PESQ", "pesq")
    if stoi is not None:
        result["stoi"] = stoi
    if pesq is not None:
        result["pesq"] = pesq
    if "f0_mse" in summary:
        result["f0_mse"] = summary["f0_mse"]
    if "mel_mse" in summary:
        result["mel_mse"] = summary["mel_mse"]

    return result


def _append_jscc_fsk_row(
    csv_path: Path,
    ckpt_path: Path,
    step: int,
    epoch: int,
    metrics: Dict[str, float],
) -> None:
    is_new = not csv_path.is_file()
    fieldnames = ["ckpt_path", "step", "epoch", "stoi", "pesq", "f0_mse", "mel_mse"]

    row = {
        "ckpt_path": str(ckpt_path),
        "step": str(step),
        "epoch": str(epoch),
        "stoi": f"{metrics.get('stoi', float('nan')):.6f}" if metrics.get("stoi") is not None else "",
        "pesq": f"{metrics.get('pesq', float('nan')):.6f}" if metrics.get("pesq") is not None else "",
        "f0_mse": f"{metrics.get('f0_mse', float('nan')):.6f}" if metrics.get("f0_mse") is not None else "",
        "mel_mse": f"{metrics.get('mel_mse', float('nan')):.6f}" if metrics.get("mel_mse") is not None else "",
    }

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow(row)

    print(
        f"[JSCC-FSK] Appended metrics for step={step}, epoch={epoch} "
        f"to {csv_path} → {row}"
    )


def _compute_pesq_stoi(
    ref: np.ndarray,
    deg: np.ndarray,
    sample_rate: int = 16000,
) -> Tuple[Optional[float], Optional[float]]:
    """Compute PESQ and STOI between reference and degraded audio.

    Best-effort helper copied from scripts/jscc_single_sample_decode_from_bits.py:
    if third-party packages (pesq, pystoi) are missing, it prints a warning
    and returns None for the corresponding metric instead of raising.
    """

    pesq_score: Optional[float] = None
    stoi_score: Optional[float] = None

    ref_f = np.asarray(ref, dtype=np.float32).reshape(-1)
    deg_f = np.asarray(deg, dtype=np.float32).reshape(-1)
    if ref_f.size == 0 or deg_f.size == 0:
        return pesq_score, stoi_score

    n = min(ref_f.size, deg_f.size)
    ref_f = ref_f[:n]
    deg_f = deg_f[:n]
    ref_f = np.nan_to_num(ref_f, nan=0.0).clip(-1.0, 1.0)
    deg_f = np.nan_to_num(deg_f, nan=0.0).clip(-1.0, 1.0)

    try:
        from pesq import pesq as pesq_fn  # type: ignore

        pesq_score = float(pesq_fn(sample_rate, ref_f, deg_f, "wb"))
    except Exception as exc:
        print(f"[bit_only_eval] WARNING: PESQ computation failed or pesq package missing: {exc}")

    try:
        from pystoi import stoi as stoi_fn  # type: ignore

        stoi_val = stoi_fn(ref_f, deg_f, sample_rate, extended=False)
        try:
            stoi_score = float(stoi_val)
        except Exception:
            stoi_score = None
    except Exception as exc:
        print(f"[bit_only_eval] WARNING: STOI computation failed or pystoi package missing: {exc}")

    return pesq_score, stoi_score


def _compute_visqol(
    ref: np.ndarray,
    deg: np.ndarray,
    sample_rate: int = 16000,
) -> Optional[float]:
    """Best-effort VISQOL (MOS-LQO) between reference and degraded audio.

    实现基于 ``pyvisqol`` 包的 ``Visqol`` 封装：在本地环境满足
    依赖（libstdc++/modelscope 等）时会返回一个 float 分数；如果
    依赖缺失或运行失败，则打印告警并返回 None，不会中断训练。
    """

    try:
        # pyvisqol 的 Python 封装目前通过 ``Visqol.measure(ref_path, deg_path)``
        # 接收 wav 路径，因此这里需要临时写入磁盘再调用。
        import soundfile as sf  # type: ignore
        from pyvisqol.pyvisqol import Visqol  # type: ignore
    except Exception as exc:
        # ImportError 或底层 so 加载失败（例如 GLIBCXX 版本不匹配）
        print(f"[bit_only_eval] WARNING: VISQOL import failed or pyvisqol missing: {exc}")
        return None

    ref_f = np.asarray(ref, dtype=np.float32).reshape(-1)
    deg_f = np.asarray(deg, dtype=np.float32).reshape(-1)
    if ref_f.size == 0 or deg_f.size == 0:
        return None

    n = min(ref_f.size, deg_f.size)
    ref_f = np.nan_to_num(ref_f[:n], nan=0.0).clip(-1.0, 1.0)
    deg_f = np.nan_to_num(deg_f[:n], nan=0.0).clip(-1.0, 1.0)

    import tempfile
    import os as _os

    tmp_dir = tempfile.mkdtemp(prefix="visqol_tmp_")
    ref_path = _os.path.join(tmp_dir, "ref.wav")
    deg_path = _os.path.join(tmp_dir, "deg.wav")

    try:
        # pyvisqol 内部通过 librosa.load 读入为 float64；soundfile 这里
        # 直接写 float32 即可，由 librosa 负责类型转换。
        sf.write(ref_path, ref_f, sample_rate)
        sf.write(deg_path, deg_f, sample_rate)

        try:
            engine = Visqol(mode="speech" if sample_rate == 16000 else "audio")
        except Exception as exc:
            print(f"[bit_only_eval] WARNING: VISQOL engine init failed: {exc}")
            return None

        try:
            score = float(engine.measure(ref_path, deg_path))
        except Exception as exc:
            print(f"[bit_only_eval] WARNING: VISQOL computation failed: {exc}")
            return None

        return score
    finally:
        # 尽量清理临时文件；失败时静默忽略
        try:
            for p in (ref_path, deg_path):
                if _os.path.isfile(p):
                    _os.remove(p)
            _os.rmdir(tmp_dir)
        except Exception:
            pass


def _run_jscc_fsk_eval_if_enabled(
    cfg: SupportConfig,
    ckpt_path: str,
    epoch: int,
    global_step: int,
) -> None:
    """If configured, run pcm_segment_infer_jscc_fsk.py on the latest ckpt.

    该函数在 save_checkpoint 后同步调用，会略微增加保存 checkpoint
    时的耗时。所有配置通过 cfg.jscc_fsk_* 字段传入，保持在 Python
    侧与 Fargan_sim 侧完全解耦。
    """

    if not bool(getattr(cfg, "jscc_fsk_eval", False)):
        return

    script = getattr(cfg, "jscc_fsk_pcm_infer_script", None)
    pcm_path = getattr(cfg, "jscc_fsk_pcm_path", None)
    noise_csv = getattr(cfg, "jscc_fsk_noise_csv", None)
    output_root = getattr(cfg, "jscc_fsk_output_root", None)

    if not (script and pcm_path and noise_csv and output_root):
        print("[JSCC-FSK] jscc_fsk_eval is True but required paths are missing; skip eval.")
        return

    script_path = Path(script).expanduser().resolve()
    if not script_path.is_file():
        print(f"[JSCC-FSK] pcm_segment_infer_jscc_fsk.py not found at {script_path}; skip eval.")
        return

    ckpt = Path(ckpt_path).expanduser().resolve()
    out_root = Path(output_root).expanduser().resolve()
    out_dir = out_root / f"step_{global_step:06d}_epoch_{epoch:02d}"

    cmd = [
        sys.executable,
        str(script_path),
        "--pcm_path", str(pcm_path),
        "--output_dir", str(out_dir),
        "--sample_rate", str(getattr(cfg, "jscc_fsk_sample_rate", 16000)),
        "--pcm_dtype", str(getattr(cfg, "jscc_fsk_pcm_dtype", "int16")),
        "--segment_sec", str(getattr(cfg, "jscc_fsk_segment_sec", 4.0)),
        "--num_segments", str(getattr(cfg, "jscc_fsk_num_segments", 50)),
        "--seed", str(getattr(cfg, "jscc_fsk_seed", 123)),
        "--snr_db", str(getattr(cfg, "jscc_fsk_snr_db", 3.0)),
        "--ckpt_stage2_5", str(ckpt),
        "--noise_csv", str(noise_csv),
    ]

    try:
        print(f"[JSCC-FSK] Running JSCC+FSK eval for ckpt {ckpt.name} at step={global_step}, epoch={epoch}")
        print("[JSCC-FSK] CMD:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"[JSCC-FSK] ERROR: pcm_segment_infer_jscc_fsk.py failed: {exc}")
        return

    metrics = _aggregate_jscc_fsk_metrics(out_dir)
    if not metrics:
        print(f"[JSCC-FSK] WARNING: no *_metrics.json found under {out_dir}; skip CSV append.")
        return

    metrics_csv = getattr(cfg, "jscc_fsk_metrics_csv", None)
    if metrics_csv:
        csv_path = Path(metrics_csv).expanduser().resolve()
    else:
        csv_path = Path(cfg.ckpt_dir).expanduser().resolve() / "jscc_fsk_metrics.csv"

    _append_jscc_fsk_row(csv_path, ckpt, global_step, epoch, metrics)

def run_training_support(cfg: SupportConfig) -> None:
    device = torch.device(cfg.device)

    # Fix random seed for reproducibility
    seed = getattr(cfg, 'seed', None)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print(f"[Seed] Fixed random seed to {seed} for reproducibility")
    else:
        # Enable cudnn benchmark for faster training when not requiring reproducibility
        torch.backends.cudnn.benchmark = True

    # Initialize AMP GradScaler
    use_amp = getattr(cfg, 'use_amp', True) and device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)
    if use_amp:
        print("[AMP] Mixed precision training enabled (faster)")

    if os.environ.get("DBG_STAGE25_ANOMALY", "0") == "1":
        torch.autograd.set_detect_anomaly(True)
    try:
        cfg.lambda_wave = float(os.environ.get("STAGE25_LAMBDA_WAVE", cfg.lambda_wave))
    except Exception:
        pass
    dataloader = build_dataloader(cfg)
    channel_sim = ChannelSimulator(sample_rate=16000, frame_hz=100)

    model = build_model(cfg).to(device)
    # 将禁用 c0 校准的开关同步到模型上，供前向路径内部判断使用。
    try:
        if hasattr(cfg, 'disable_ceps_c0_calib'):
            setattr(model, 'disable_ceps_c0_calib', bool(getattr(cfg, 'disable_ceps_c0_calib', False)))
    except Exception:
        pass

    # 简要打印 SSL 内容 loss 配置，便于确认是否启用/参数是否按预期生效。
    try:
        if float(getattr(cfg, 'lambda_ssl', 0.0)) > 0.0 and getattr(cfg, 'ssl_model_name', None):
            print(
                f"[SSL] Content loss enabled: lambda_ssl={float(getattr(cfg, 'lambda_ssl', 0.0))}, "
                f"model='{getattr(cfg, 'ssl_model_name')}', "
                f"layers={getattr(cfg, 'ssl_layers', None)}, "
                f"warmup={int(getattr(cfg, 'ssl_warmup_steps', 0))}"
            )
        else:
            print("[SSL] Content loss disabled (lambda_ssl=0 or no ssl_model_name)")
    except Exception:
        # 如果早期 cfg 没有这些字段，静默跳过，不影响训练
        pass

    # Optional: initialize wandb logging on main process
    use_wandb = False
    try:
        if bool(getattr(cfg, "use_wandb", False)) and wandb is not None:
            project = getattr(cfg, "wandb_project", None) or "DBP-JSCC"
            run_name = getattr(cfg, "wandb_run_name", None)
            wandb.init(project=project, name=run_name, config=asdict(cfg))
            use_wandb = True
            print(f"[wandb] Initialized: project='{project}', run='{run_name}'")
        elif bool(getattr(cfg, "use_wandb", False)) and wandb is None:
            print("[wandb] WARNING: wandb is not installed, disabling wandb logging")
    except Exception as _w:
        print(f"[wandb] WARNING: failed to initialize wandb: {_w}")
        use_wandb = False

    # CSV for bit-only objective metrics (STOI / PESQ / F0 / Mel) during training。
    # 始终为其选择一个落盘路径（默认 ckpt_dir/bit_only_metrics.csv），避免因
    # 配置缺失导致指标不写入。
    bit_csv_str = getattr(cfg, 'bit_only_metrics_csv', None)
    if bit_csv_str:
        bit_csv_path = Path(bit_csv_str).expanduser().resolve()
    else:
        bit_csv_path = Path(cfg.ckpt_dir).expanduser().resolve() / 'bit_only_metrics.csv'

    # Log content-only mode status
    content_only = getattr(cfg, 'content_only', False)
    if content_only:
        if cfg.with_hash:
            print("[Mode] CONTENT-ONLY (with hash): training mel+VMamba+HashBottleneck, skipping F0/ceps/vocoder")
        else:
            print("[Mode] CONTENT-ONLY (no hash): training mel+VMamba only, skipping F0/ceps/vocoder/hash")

    # Build a frozen teacher vocoder for HF distillation when enabled
    teacher_needed = float(getattr(cfg, 'lambda_teacher_hf', 0.0)) > 0.0
    if teacher_needed:
        try:
            voc_t = FARGANDecoder().to(device)
            # Prefer loading explicit FARGAN ckpt for teacher; fallback to student weights
            loaded = False
            if isinstance(getattr(cfg, 'vocoder_ckpt', None), str) and os.path.isfile(cfg.vocoder_ckpt):
                try:
                    print(f"[Teacher] Loading vocoder ckpt for teacher: {cfg.vocoder_ckpt}")
                    try:
                        ck = torch.load(cfg.vocoder_ckpt, map_location='cpu', weights_only=True)
                    except TypeError:
                        ck = torch.load(cfg.vocoder_ckpt, map_location='cpu')
                    sd = ck['state_dict'] if isinstance(ck, dict) and 'state_dict' in ck else ck
                    voc_t.fargan_core.load_state_dict(sd, strict=False)
                    loaded = True
                except Exception as _e:
                    print(f"[Teacher] WARNING: failed to load teacher ckpt: {_e}")
            if not loaded:
                try:
                    print("[Teacher] Falling back to student vocoder weights for teacher init")
                    voc_t.load_state_dict(model.vocoder.state_dict(), strict=False)
                except Exception:
                    pass
            for p in voc_t.parameters():
                p.requires_grad = False
            model.vocoder_teacher = voc_t
            print("[Teacher] Frozen teacher vocoder initialized")
        except Exception as _e:
            print(f"[Teacher] WARNING: teacher init failed: {_e}")

    # 可选：仅做 Mel 频带趋势/关联性的快速验证并退出
    if bool(int(os.environ.get('VERIFY_MEL_BAND_STATS', '0'))):
        try:
            import torchaudio  # local import for mel
        except Exception as _e:
            print('[verify_mel_band] torchaudio not available:', _e)
            return
        mel_tf = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=400, hop_length=160,
            n_mels=32, power=2.0, center=True, norm=None, mel_scale='htk'
        ).to(device)
        to_db = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80).to(device)
        print('[verify_mel_band] running quick stats on a few batches ...')
        n_batches = 5
        low_bins = int(getattr(cfg, 'mel_low_bins', 10))
        edges = getattr(cfg, 'mel_band_edges', None)
        if not edges:
            edges = [16, 24, 32]
        r_means = []
        corr_means = []
        it = 0
        for batch in dataloader:
            audio = batch['audio'].to(device)
            P = mel_tf(audio)
            P_db = to_db(P) / 10.0   # [-8,0]
            mel = P_db.transpose(1, 2)  # [B,T,32]
            B, T, F_dim = mel.shape
            low = mel[:, :, :min(low_bins, F_dim)].mean(dim=2)
            prev = min(low_bins, F_dim - 1)
            for hi in edges:
                hi = min(hi, F_dim)
                if hi <= prev:
                    continue
                band = mel[:, :, prev:hi].mean(dim=2)
                # 比值与相关性
                ratio = (band - low).mean().item()  # log域差，≈ dB/10
                r_means.append(ratio)
                if T > 1:
                    d_low = low[:, 1:] - low[:, :-1]
                    d_hi = band[:, 1:] - band[:, :-1]
                    # 余弦相关
                    dl = d_low.reshape(B, -1)
                    dh = d_hi.reshape(B, -1)
                    num = (dl * dh).sum(dim=1)
                    den = (dl.norm(dim=1) * dh.norm(dim=1) + 1e-6)
                    corr = (num / den).mean().item()
                    corr_means.append(corr)
                prev = hi
            it += 1
            if it >= n_batches:
                break
        if len(r_means) > 0:
            print(f"[verify_mel_band] mean(log-mel band-low) over bands: {sum(r_means)/len(r_means):+.4f}")
        if len(corr_means) > 0:
            print(f"[verify_mel_band] mean temporal corr(dHigh,dLow): {sum(corr_means)/len(corr_means):+.4f}")
        return

    # 冻结声码器热身1-2k步，减少端到端训练难度
    vocoder_warmup_steps = cfg.stage1_steps // 2  # 前半段冻结vocoder
    if cfg.use_two_stage:
        # F0-only warmup：vocoder 从 step0 就参与训练，跳过冻结逻辑
        if bool(getattr(cfg, 'f0_only', False)):
            print("[Vocoder] F0-only warmup: vocoder trainable from step 0 (no warmup freeze)")
            for p in model.vocoder.parameters():
                p.requires_grad = True
        # If freeze_vocoder_all is set, keep vocoder frozen throughout training
        elif cfg.freeze_vocoder_all:
            print("[Vocoder] Freezing vocoder for ALL steps (freeze_vocoder_all=True)")
            for p in model.vocoder.parameters():
                p.requires_grad = False
        else:
            print(f"[Vocoder] Freezing vocoder for first {vocoder_warmup_steps} steps")
            for p in model.vocoder.parameters():
                p.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)

    # ------------------------------------------------------------------
    # Gradient / weight sanitization helpers（来自旧版v3，增加一次性详细日志）。
    # ------------------------------------------------------------------

    _seen_bad_grad_ids: set[int] = set()
    _seen_bad_module_names: set[str] = set()
    _last_grad_exploded: bool = False

    def _sanitize_gradients_(model: nn.Module, step: int) -> int:
        """Replace NaN/Inf gradients with zeros and lazily log offenders.

        仅当检测到“疑似梯度爆炸”时，才在 DBG_SANITIZE=1 下打印
        详细 dbg 信息；否则只保留首次出现时的简要 param 名称。
        """
        nonlocal _seen_bad_grad_ids, _last_grad_exploded
        _last_grad_exploded = False
        bad = 0
        for p in model.parameters():
            if p.grad is None:
                continue
            g = p.grad
            if not torch.isfinite(g).all():
                with torch.no_grad():
                    g_clone = g.detach().clone()
                    nonfinite_mask = ~torch.isfinite(g_clone)
                    num_nonfinite = int(nonfinite_mask.sum().item())
                    num_total = int(g_clone.numel())
                    cleaned = torch.nan_to_num(g_clone, nan=0.0, posinf=0.0, neginf=0.0)
                    g_abs = cleaned.abs()
                    grad_max = float(g_abs.max().item()) if num_total > 0 else 0.0
                    grad_mean = float(g_abs.mean().item()) if num_total > 0 else 0.0
                    torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0, out=g)

                    # 简单的“梯度爆炸”判据：
                    #  - 非有限占比超过 frac_thresh，或
                    #  - |g|_max 超过 grad_thresh。
                    try:
                        grad_thresh = float(os.environ.get("STAGE25_GRAD_EXP_THRESH", "1e3"))
                    except Exception:
                        grad_thresh = 1e3
                    try:
                        frac_thresh = float(os.environ.get("STAGE25_GRAD_FRAC_THRESH", "1e-3"))
                    except Exception:
                        frac_thresh = 1e-3
                    frac = float(num_nonfinite) / float(max(1, num_total))
                    is_exploded = (grad_max > grad_thresh) or (frac > frac_thresh)
                    if is_exploded:
                        _last_grad_exploded = True

                bad += 1
                pid = id(p)
                dbg_all = os.environ.get("DBG_SANITIZE", "0") == "1"
                # 仅在两种情况下打印：
                #  - 该参数第一次出现非有限梯度（简要定位）；
                #  - 或在 dbg_all 模式下且被判定为“梯度爆炸”。
                should_log = (pid not in _seen_bad_grad_ids) or (dbg_all and is_exploded)
                if should_log:
                    if pid not in _seen_bad_grad_ids:
                        _seen_bad_grad_ids.add(pid)
                    try:
                        pname = "<unknown>"
                        for n, pp in model.named_parameters():
                            if pp is p:
                                pname = n
                                break
                        print(f"[Sanitize] First non-finite grad on param: {pname} (shape={tuple(p.shape)})")
                        if dbg_all and is_exploded:
                            print(
                                f"[Sanitize-debug] step={step} param={pname} "
                                f"nonfinite={num_nonfinite}/{num_total} frac={frac:.3e} "
                                f"|g|_mean={grad_mean:.3e} |g|_max={grad_max:.3e}"
                            )
                    except Exception:
                        pass
        return bad

    def _module_has_nonfinite(m: nn.Module) -> bool:
        for p in m.parameters(recurse=True):
            if not torch.isfinite(p.data).all():
                return True
        return False

    def _reinit_module_(m: nn.Module, verbose_name: str) -> None:
        with torch.no_grad():
            for p in m.parameters(recurse=True):
                if p.dim() >= 2:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.zeros_(p)
        print(f"[Sanitize] Reinitialized non-finite module: {verbose_name}")

    def _scan_and_repair_weights_(model: DualBranchBarkJSCC) -> int:
        """Scan critical submodules for non-finite weights and reinit if needed.

        仅在发现非有限权重时打印模块名，并返回修复的模块数量。
        """
        nonlocal _seen_bad_module_names
        repaired = 0
        # Vocoder core（即使在 content_only 下通常不会更新，也做一次安全检查）
        try:
            if hasattr(model, 'vocoder') and hasattr(model.vocoder, 'fargan_core'):
                if _module_has_nonfinite(model.vocoder.fargan_core):
                    name = 'vocoder.fargan_core'
                    if name not in _seen_bad_module_names:
                        _seen_bad_module_names.add(name)
                        print(f"[Sanitize] Detected non-finite weights in module: {name}")
                    _reinit_module_(model.vocoder.fargan_core, name)
                    repaired += 1
        except Exception:
            pass

        # F0 分支：如果存在这些子模块，也一并检查
        for attr_name in ['f0vuv_enc', 'f0vuv_jscc_enc', 'f0vuv_jscc_dec', 'f0vuv_dec']:
            if hasattr(model, attr_name):
                mod = getattr(model, attr_name)
                if _module_has_nonfinite(mod):
                    if attr_name not in _seen_bad_module_names:
                        _seen_bad_module_names.add(attr_name)
                        print(f"[Sanitize] Detected non-finite weights in module: {attr_name}")
                    _reinit_module_(mod, attr_name)
                    repaired += 1

        # 内容分支 VMamba：若 encoder/decoder 中出现非有限权重，直接重置对应子模块，
        # 避免在第一次 forward 时就产生 NaN（尤其是 decoder-side SelectiveScan2D）。
        try:
            if hasattr(model, 'content_vmamba'):
                for sub_name in ['encoder', 'decoder']:
                    if hasattr(model.content_vmamba, sub_name):
                        sub_mod = getattr(model.content_vmamba, sub_name)
                        if _module_has_nonfinite(sub_mod):
                            full_name = f'content_vmamba.{sub_name}'
                            if full_name not in _seen_bad_module_names:
                                _seen_bad_module_names.add(full_name)
                                print(f"[Sanitize] Detected non-finite weights in module: {full_name}")
                            _reinit_module_(sub_mod, full_name)
                            repaired += 1
        except Exception:
            pass

        # Mel→ceps 可学习映射：若映射参数已出现 NaN/Inf，同样重置以避免倒谱直接爆掉。
        try:
            if hasattr(model, 'mel18_to_ceps') and _module_has_nonfinite(model.mel18_to_ceps):
                name = 'mel18_to_ceps'
                if name not in _seen_bad_module_names:
                    _seen_bad_module_names.add(name)
                    print(f"[Sanitize] Detected non-finite weights in module: {name}")
                _reinit_module_(model.mel18_to_ceps, name)
                repaired += 1
        except Exception:
            pass

        return repaired
    # Optional adversarial discriminator (decoder-only regularization)
    class PatchDisc2D(torch.nn.Module):
        def __init__(self, in_ch: int = 1, base: int = 32) -> None:
            super().__init__()
            C = base
            self.conv1 = torch.nn.Conv2d(in_ch, C, 3, 2, 1)
            self.conv2 = torch.nn.Conv2d(C, C*2, 3, 2, 1)
            self.conv3 = torch.nn.Conv2d(C*2, C*4, 3, 2, 1)
            self.conv4 = torch.nn.Conv2d(C*4, C*8, 3, 2, 1)
            self.head  = torch.nn.Conv2d(C*8, 1, 3, 1, 1)
            self.act = torch.nn.LeakyReLU(0.2, inplace=True)
        def forward(self, x: torch.Tensor):
            # x: [B,1,F,T]
            feats = []
            h = self.act(self.conv1(x)); feats.append(h)
            h = self.act(self.conv2(h)); feats.append(h)
            h = self.act(self.conv3(h)); feats.append(h)
            h = self.act(self.conv4(h)); feats.append(h)
            logit = self.head(h)
            # mimic fargan interface: list of layers + last logit per scale
            return feats + [logit]

    class MultiScaleSpecDisc(torch.nn.Module):
        def __init__(self, fft_sizes: list[int], sr: int = 16000,
                     roi: tuple[int, int] = (4000, 8000), base: int = 32) -> None:
            super().__init__()
            self.ffts = fft_sizes
            self.sr = sr
            self.roi = roi
            self.discs = torch.nn.ModuleList([PatchDisc2D(1, base) for _ in self.ffts])
        @staticmethod
        def _stft_mag(x: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
            x32 = x.to(torch.float32)
            win_t = torch.hann_window(win, device=x32.device, dtype=torch.float32)
            X = torch.stft(x32, n_fft=n_fft, hop_length=hop, win_length=win, window=win_t,
                           center=False, return_complex=True)
            mag = X.abs()  # [B, F, T]
            B, F, T = mag.shape
            expected_frames = x32.size(-1) // hop
            if T > expected_frames:
                mag = mag[:, :, :expected_frames]
            return mag
        def forward(self, y: torch.Tensor):
            # y: [B,L]
            outs = []
            for i, n_fft in enumerate(self.ffts):
                hop = n_fft // 4
                Mag = self._stft_mag(y, n_fft=n_fft, hop=hop, win=n_fft)
                B, F, T = Mag.shape
                # ROI crop [freq_low, freq_high]
                fl, fh = self.roi
                freqs = torch.linspace(0, self.sr/2, F, device=Mag.device, dtype=Mag.dtype)
                keep = (freqs >= float(fl)) & (freqs <= float(fh))
                Mag = Mag[:, keep, :]
                x = (Mag + 1e-4).log().unsqueeze(1)  # [B,1,F',T]
                outs.append(self.discs[i](x))
            return outs

    class FeaturePatchDisc(torch.nn.Module):
        def __init__(self, in_ch: int = 1, base: int = 32) -> None:
            super().__init__()
            C = base
            self.conv1 = torch.nn.Conv2d(in_ch, C, 3, 2, 1)
            self.conv2 = torch.nn.Conv2d(C, C*2, 3, 2, 1)
            self.conv3 = torch.nn.Conv2d(C*2, C*4, 3, 2, 1)
            self.conv4 = torch.nn.Conv2d(C*4, C*8, 3, 2, 1)
            self.head  = torch.nn.Conv2d(C*8, 1, 3, 1, 1)
            self.act = torch.nn.LeakyReLU(0.2, inplace=True)
        def forward(self, x: torch.Tensor):
            feats = []
            h = self.act(self.conv1(x)); feats.append(h)
            h = self.act(self.conv2(h)); feats.append(h)
            h = self.act(self.conv3(h)); feats.append(h)
            h = self.act(self.conv4(h)); feats.append(h)
            logit = self.head(h)
            return feats + [logit]

    # HiFi-GAN style raw-waveform discriminators (MPD + MSD, optional)
    mpd: Optional[torch.nn.Module] = None
    msd: Optional[torch.nn.Module] = None
    optimizer_hifi_disc: Optional[torch.optim.Optimizer] = None
    try:
        lam_hifi_adv = float(getattr(cfg, 'lambda_hifi_adv', 0.0))
        lam_hifi_fm = float(getattr(cfg, 'lambda_hifi_fm', 0.0))
        if lam_hifi_adv > 0.0 or lam_hifi_fm > 0.0:
            mpd = MultiPeriodDiscriminator().to(device)
            msd = MultiScaleDiscriminator().to(device)
            lr_hifi = float(getattr(cfg, 'hifi_disc_lr', cfg.lr))
            optimizer_hifi_disc = torch.optim.Adam(
                list(mpd.parameters()) + list(msd.parameters()),
                lr=lr_hifi,
                betas=(0.8, 0.99),
            )
            print(f"[HiFi-ADV] Enabled MPD+MSD discriminators (lr={lr_hifi})")
    except Exception as _e:
        print(f"[HiFi-ADV] WARNING: failed to init MPD/MSD discriminators: {_e}")
        mpd = None
        msd = None
        optimizer_hifi_disc = None

    # HF adversarial discriminator (4–8kHz STFT, optional)
    hf_disc: Optional[torch.nn.Module] = None
    optimizer_hf_disc: Optional[torch.optim.Optimizer] = None
    try:
        lam_hf_adv = float(getattr(cfg, 'lambda_hf_adv', 0.0))
        lam_hf_fm = float(getattr(cfg, 'lambda_hf_fm', 0.0))
        if lam_hf_adv > 0.0 or lam_hf_fm > 0.0:
            roi_lo = int(getattr(cfg, 'hf_adv_roi_low_hz', 4000))
            roi_hi = int(getattr(cfg, 'hf_adv_roi_high_hz', 8000))
            fft_sizes = [1024, 512]
            hf_disc = MultiScaleSpecDisc(fft_sizes=fft_sizes, sr=16000, roi=(roi_lo, roi_hi), base=32).to(device)
            lr_d = float(getattr(cfg, 'hf_adv_disc_lr', 1e-4))
            optimizer_hf_disc = torch.optim.Adam(hf_disc.parameters(), lr=lr_d, betas=(0.5, 0.9))
            print(f"[HF-ADV] Enabled HF discriminator roi={roi_lo}-{roi_hi}Hz, fft_sizes={fft_sizes}, lr={lr_d}")
    except Exception as _e:
        print(f"[HF-ADV] WARNING: failed to init HF discriminator: {_e}")
        hf_disc = None
        optimizer_hf_disc = None

    # Optional OSCE FD-based adversarial discriminator on audio_hat.
    osce_disc: Optional[torch.nn.Module] = None
    optimizer_osce_disc: Optional[torch.optim.Optimizer] = None

    def _build_osce_disc(device: torch.device):
        """Construct FD-MResDisc from ``../osce/models/fd_discriminator.py``.

        通过 importlib 直接按文件路径加载 OSCE 的 fd_discriminator，
        并临时调整 sys.path 以确保其内部的 ``from utils.spec``
        命中 OSCE 仓库自带的 ``utils/spec.py``，而非当前仓库的
        ``utils`` 包。
        """
        if _OSCE_DIR_GLOBAL is None:
            return None
        try:
            import importlib.util as _ilu

            fd_path = os.path.join(_OSCE_DIR_GLOBAL, "models", "fd_discriminator.py")
            if not os.path.isfile(fd_path):
                raise FileNotFoundError(f"fd_discriminator.py not found at {fd_path}")

            spec = _ilu.spec_from_file_location("osce_fd_discriminator", fd_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"spec_from_file_location failed for {fd_path}")

            fd_mod = _ilu.module_from_spec(spec)  # type: ignore

            orig_sys_path = list(sys.path)
            orig_utils = sys.modules.get("utils", None)

            try:
                tmp_sys_path: list[str] = []
                for p in orig_sys_path:
                    try:
                        if '_ROOT_DIR' in globals() and _ROOT_DIR and os.path.samefile(os.path.abspath(p), _ROOT_DIR):
                            continue
                    except Exception:
                        pass
                    tmp_sys_path.append(p)

                if _OSCE_DIR_GLOBAL in tmp_sys_path:
                    tmp_sys_path.remove(_OSCE_DIR_GLOBAL)
                tmp_sys_path.insert(0, _OSCE_DIR_GLOBAL)
                sys.path[:] = tmp_sys_path

                if "utils" in sys.modules:
                    del sys.modules["utils"]

                spec.loader.exec_module(fd_mod)  # type: ignore[arg-type]
            finally:
                sys.path[:] = orig_sys_path
                if orig_utils is not None:
                    sys.modules["utils"] = orig_utils
                elif "utils" in sys.modules:
                    del sys.modules["utils"]

            DiscClass = getattr(fd_mod, "TFDMultiResolutionDiscriminator", None)
            if DiscClass is None:
                raise AttributeError("TFDMultiResolutionDiscriminator not found in fd_discriminator")

            d = DiscClass(
                architecture="free",
                design="f_down",
                fft_sizes_16k=[2 ** n for n in range(6, 12)],
                freq_roi=[0, 7400],
                max_channels=256,
                noise_gain=0.0,
            )
            d.to(device)
            return d
        except Exception as _e:
            print(f"[OSCE-GAN] WARNING: failed to build discriminator: {_e}")
            return None

    if bool(getattr(cfg, 'bfcc_gan', False)):
        osce_disc = _build_osce_disc(device)
        if osce_disc is not None:
            optimizer_osce_disc = torch.optim.AdamW(
                [p for p in osce_disc.parameters() if p.requires_grad],
                lr=cfg.lr,
                betas=(0.8, 0.99),
                eps=1e-8,
            )
            print("[OSCE-GAN] Adversarial discriminator enabled on audio_hat")

    class HFMelDiscriminator(torch.nn.Module):
        """高频 Bark/BFCC 频谱判别器（时间轴 1D 卷积）。"""

        def __init__(self, hf_bins: int = 22, hidden: int = 64) -> None:
            super().__init__()
            self.hf_bins = hf_bins
            self.net = torch.nn.Sequential(
                torch.nn.Conv1d(hf_bins, hidden, kernel_size=5, stride=2, padding=2),
                torch.nn.LeakyReLU(0.2),
                torch.nn.Dropout(0.1),
                torch.nn.Conv1d(hidden, hidden * 2, kernel_size=5, stride=2, padding=2),
                torch.nn.LeakyReLU(0.2),
                torch.nn.Dropout(0.1),
                torch.nn.Conv1d(hidden * 2, 1, kernel_size=5, stride=1, padding=2),
            )

        def forward(self, mel_hf: torch.Tensor) -> torch.Tensor:
            # mel_hf: [B, T, HF] -> [B, HF, T] -> [B, T']
            x = mel_hf.transpose(1, 2)
            return self.net(x).squeeze(1)

        def get_features(self, mel_hf: torch.Tensor) -> List[torch.Tensor]:
            features: List[torch.Tensor] = []
            x = mel_hf.transpose(1, 2)
            for layer in self.net:
                x = layer(x)
                if isinstance(layer, torch.nn.Conv1d):
                    features.append(x)
            return features

    # Mel-domain HF adversarial discriminator (optional)
    hf_mel_disc: Optional[torch.nn.Module] = None
    optimizer_hf_mel_disc: Optional[torch.optim.Optimizer] = None
    try:
        lam_hf_mel_adv = float(getattr(cfg, 'lambda_hf_mel_adv', 0.0))
        lam_hf_mel_fm = float(getattr(cfg, 'lambda_hf_mel_fm', 0.0))
        if lam_hf_mel_adv > 0.0 or lam_hf_mel_fm > 0.0:
            # high-frequency mel 区域的频带数在运行时根据 mel 维度确定
            hf_start = int(getattr(cfg, 'hf_mel_low_bins', int(getattr(cfg, 'l2h_low_bins', 10))))
            # 先占位，具体 HF 维度在首次使用时根据 mel.shape 进行裁剪
            hf_mel_disc = HFMelDiscriminator(hf_bins=max(8, 32 - hf_start), hidden=64).to(device)
            optimizer_hf_mel_disc = torch.optim.Adam(hf_mel_disc.parameters(), lr=1e-4, betas=(0.5, 0.9))
            print("[HF-MEL-ADV] Enabled mel HF discriminator")
    except Exception as _e:
        print(f"[HF-MEL-ADV] WARNING: failed to init mel HF discriminator: {_e}")
        hf_mel_disc = None
        optimizer_hf_mel_disc = None
    # 记录基线 LR 和权重，便于两阶段调度
    base_lr = cfg.lr
    base_wave = cfg.lambda_wave
    base_mel = cfg.lambda_mel
    base_hash_reg = float(getattr(cfg, 'lambda_hash_reg', 0.0))

    global_step = 0
    start_epoch = 0
    # ★ 新增：如果指定了 --resume，则从 checkpoint 恢复
    resume_step_marker: Optional[int] = None

    if cfg.resume is not None and cfg.resume != "":
        if not os.path.isfile(cfg.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {cfg.resume}")
        print(f"[Resume] Loading checkpoint from {cfg.resume}")
        ckpt = torch.load(cfg.resume, map_location=device)

        # 恢复模型与优化器
        if "model" in ckpt:
            # 形状兼容加载：遇到结构变更（如上/下采样stride调整）时丢弃尺寸不匹配的权重
            def _shape_compatible_state_dict(model: torch.nn.Module, sd: dict) -> dict:
                msd = model.state_dict()
                out = {}
                dropped = []
                for k, v in sd.items():
                    if k in msd and isinstance(v, torch.Tensor) and v.shape == msd[k].shape:
                        out[k] = v
                    else:
                        dropped.append(k)
                if len(dropped) > 0:
                    print(f"[Resume] Dropped {len(dropped)} key(s) due to shape mismatch; e.g., {dropped[:5]}")
                return out

            safe_sd = _shape_compatible_state_dict(model, ckpt["model"])
            load_result = model.load_state_dict(safe_sd, strict=False)
            try:
                missing = list(load_result.missing_keys) if hasattr(load_result, 'missing_keys') else []
                unexpected = list(load_result.unexpected_keys) if hasattr(load_result, 'unexpected_keys') else []
            except Exception:
                missing, unexpected = [], []
            if len(missing) > 0 or len(unexpected) > 0:
                print(f"[Resume] Non-strict load: missing={len(missing)}, unexpected={len(unexpected)}")
                if len(missing) > 0:
                    print(f"[Resume] Missing (first 10): {missing[:10]}")
                if len(unexpected) > 0:
                    print(f"[Resume] Unexpected (first 10): {unexpected[:10]}")
        else:
            print("[Resume] WARNING: no 'model' key in checkpoint, skipping model load")

        # Optimizer: if model structure changed (e.g., --with_hash adds params),
        # always reinitialize optimizer to include new parameters.
        try:
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            # Rebuild to ensure new params (e.g., hash modules) are optimized
            optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0)
        except Exception as e:
            print(f"[Resume] WARNING: optimizer load/rebuild issue: {e}; using fresh optimizer")
            optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0)

        # Optional: resume HiFi-GAN style discriminators when enabled
        try:
            if mpd is not None and "mpd" in ckpt:
                mpd.load_state_dict(ckpt["mpd"])
            if msd is not None and "msd" in ckpt:
                msd.load_state_dict(ckpt["msd"])
            if optimizer_hifi_disc is not None and "optimizer_hifi_disc" in ckpt:
                optimizer_hifi_disc.load_state_dict(ckpt["optimizer_hifi_disc"])
        except Exception as e:
            print(f"[Resume] WARNING: failed to load MPD/MSD discriminator states: {e}")

        global_step = int(ckpt.get("global_step", 0))
        resume_step_marker = int(global_step)
        start_epoch = int(ckpt.get("epoch", 0))
        print(f"[Resume] start_epoch={start_epoch}, global_step={global_step}")

        # —— 参数有限性检查与自愈 ——
        # 在 resume 之后立即扫描关键子模块（vocoder/F0/内容分支/mel18_to_ceps），
        # 若发现已有 NaN/Inf 权重，则当场重置，避免第一次 forward 就产生 NaN。
        try:
            repaired_at_resume = _scan_and_repair_weights_(model)
            if repaired_at_resume > 0:
                print(f"[Sanitize] Repaired {repaired_at_resume} module(s) with non-finite weights at resume")
        except Exception as e:
            print(f"[Sanitize] WARNING: non-finite scan failed at resume: {e}")

        # 根据已经走过的 step 决定 vocoder 是否应该解冻
        if cfg.use_two_stage and not bool(getattr(cfg, 'f0_only', False)):
            if (not cfg.freeze_vocoder_all) and (global_step < vocoder_warmup_steps):
                print(f"[Resume] global_step<{vocoder_warmup_steps}, keep vocoder frozen")
                for p in model.vocoder.parameters():
                    p.requires_grad = False
            elif not cfg.freeze_vocoder_all:
                print(f"[Resume] global_step>={vocoder_warmup_steps}, unfreeze vocoder")
                for p in model.vocoder.parameters():
                    p.requires_grad = True

        # If a FARGAN ckpt is provided, optionally re-load it AFTER resume to override
        # the student's vocoder core weights. 默认关闭（reload_fargan_after_resume=False）：
        # 此时学生 vocoder 完全沿用 checkpoint 中的权重，仅 HF-teacher 使用 fargan_ckpt。
        if cfg.vocoder_ckpt and bool(getattr(cfg, 'reload_vocoder_after_resume', False)):
            try:
                if os.path.isfile(cfg.vocoder_ckpt):
                    print(f"[Vocoder] Re-loading vocoder ckpt after resume: {cfg.vocoder_ckpt}")
                    try:
                        ck = torch.load(cfg.vocoder_ckpt, map_location='cpu', weights_only=True)
                    except TypeError:
                        ck = torch.load(cfg.vocoder_ckpt, map_location='cpu')
                    sd = ck['state_dict'] if isinstance(ck, dict) and 'state_dict' in ck else ck
                    model.vocoder.fargan_core.load_state_dict(sd, strict=False)
                else:
                    print(f"[Vocoder] WARNING: vocoder_ckpt not found: {cfg.vocoder_ckpt}")
            except Exception as e:
                print(f"[Vocoder] WARNING: failed re-load FARGAN after resume: {e}")

    else:
        # ★ 不 resume 的正常路径：从头开始训练，按原计划冻结 vocoder
        if cfg.use_two_stage:
            if bool(getattr(cfg, 'f0_only', False)):
                print("[Vocoder] F0-only warmup: vocoder trainable from step 0 (no warmup freeze)")
                for p in model.vocoder.parameters():
                    p.requires_grad = True
            elif cfg.freeze_vocoder_all:
                print("[Vocoder] Freezing vocoder for ALL steps (freeze_vocoder_all=True)")
                for p in model.vocoder.parameters():
                    p.requires_grad = False
            else:
                print(f"[Vocoder] Freezing vocoder for first {vocoder_warmup_steps} steps")
                for p in model.vocoder.parameters():
                    p.requires_grad = False
    for epoch in range(start_epoch, cfg.num_epochs):
        model.train()
        for batch in dataloader:
            # ---- Optional: adversarial discriminator update happens before generator backward ----
            # Optional: vocoder-only baseline evaluation using GT FARGAN features
            if cfg.vocoder_eval_every_steps and cfg.vocoder_eval_every_steps > 0 and (global_step % cfg.vocoder_eval_every_steps == 0):
                try:
                    feats_36 = batch.get("x")
                    aud_ref = batch.get("audio")
                    if feats_36 is not None and aud_ref is not None:
                        feats_36 = feats_36.to(device)
                        aud_ref = aud_ref.to(device)
                        with torch.no_grad():
                            _per, aud_voc = model.vocoder(feats_36, target_len=aud_ref.size(-1))
                            aud_voc = aud_voc.squeeze(1)
                            voc_stft = multi_resolution_stft_loss(aud_voc, aud_ref, device=device,
                                                                   fft_sizes=[1024,512,256,128],
                                                                   hop_sizes=[256,128,64,32],
                                                                   win_lengths=[1024,512,256,128])
                        print(f"[vocoder_only] MR-STFT={float(voc_stft.item()):.4f} (step {global_step})")
                except Exception as e:
                    print(f"[vocoder_only] eval failed at step {global_step}: {e}")
            # 两阶段调度：前期弱 STFT、较小 LR，后期恢复
            if cfg.use_two_stage and not bool(getattr(cfg, 'f0_only', False)):
                if global_step < cfg.stage1_steps:
                    for pg in optimizer.param_groups:
                        pg["lr"] = min(base_lr, 5e-5)
                    cfg.lambda_wave = 0.3
                    cfg.lambda_mel = base_mel

                    # 在中间点解冻vocoder并逐步增加lambda_wave
                    if (not cfg.freeze_vocoder_all) and (global_step == vocoder_warmup_steps):
                        print(f"[Vocoder] Unfreezing vocoder at step {global_step}")
                        for p in model.vocoder.parameters():
                            p.requires_grad = True
                        # 重新初始化optimizer以包含vocoder参数
                        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0)
                else:
                    for pg in optimizer.param_groups:
                        pg["lr"] = base_lr
                    cfg.lambda_wave = base_wave
                    cfg.lambda_mel = base_mel

            # Hash 正则调度：训练早期减弱 lambda_hash_reg，缓解离散瓶颈收敛难度
            try:
                warm_steps = int(getattr(cfg, 'hash_reg_warmup_steps', 0))
                lam_start = float(getattr(cfg, 'hash_reg_start', 0.0))
                if warm_steps > 0 and base_hash_reg > 0.0:
                    if global_step < warm_steps:
                        # 线性从 lam_start 过渡到 base_hash_reg
                        alpha = float(global_step) / float(max(1, warm_steps))
                        cfg.lambda_hash_reg = lam_start + (base_hash_reg - lam_start) * alpha
                    else:
                        cfg.lambda_hash_reg = base_hash_reg
            except Exception:
                cfg.lambda_hash_reg = base_hash_reg

            # Isolation window after resume: freeze vocoder and scale down F0 extras
            isolate_steps = int(os.environ.get("STAGE25_ISOLATE_STEPS", "200"))
            in_isolation = False
            if resume_step_marker is not None:
                in_isolation = (global_step - resume_step_marker) < isolate_steps
            if in_isolation:
                for p in model.vocoder.parameters():
                    p.requires_grad = False
                setattr(cfg, '_isolate_f0_scale', 0.0)
                # 降低恢复初期学习率，避免刚刚重置的分支跳变过大
                for pg in optimizer.param_groups:
                    pg["lr"] = min(pg.get("lr", base_lr), max(5e-5, base_lr * 0.25))
            else:
                setattr(cfg, '_isolate_f0_scale', 1.0)

            # 动态设定 L2H 融合因子：支持按绝对 step 或 resume 后相对 step 调度。
            try:
                warm = int(getattr(cfg, 'l2h_warmup_steps', 400))
                blend_steps = int(getattr(cfg, 'l2h_blend_steps', 800))

                # 调度模式优先由环境变量 L2H_SCHEDULE_MODE 控制，其次使用 cfg.l2h_schedule_mode。
                import os as _os
                mode = str(_os.environ.get('L2H_SCHEDULE_MODE', getattr(cfg, 'l2h_schedule_mode', 'abs')))
                mode = 'resume_rel' if mode.lower() in ('resume_rel', 'rel') else 'abs'

                if mode == 'resume_rel' and resume_step_marker is not None:
                    base = int(resume_step_marker)
                    rel = max(0, int(global_step - base))
                else:
                    # 绝对调度：基于 global_step，适合频繁 resume 的长训
                    rel = max(0, int(global_step))
                if rel < warm:
                    l2h_blend = 0.0
                else:
                    l2h_blend = min(1.0, float(rel - warm) / max(1, blend_steps))
                if hasattr(model, 'l2h_blend'):
                    model.l2h_blend = float(l2h_blend)
            except Exception:
                pass

            # Cepstral delta quality schedule (cosine ramp for smooth introduction)，
            # 与 L2H 调度共享相同的 warmup/blend 模式。
            # p(s) = clip((s - warm) / blend_window, 0, 1)
            # p_smooth(s) = 0.5 * (1 - cos(pi * p(s)))  (cosine ramp)
            # alpha_c(s) = alpha_max * p_smooth(s)^2   (quadratic for extra smoothness)
            try:
                ceps_warm = int(getattr(cfg, 'l2h_warmup_steps', 400))
                ceps_blend = int(getattr(cfg, 'l2h_blend_steps', 800))
                ceps_alpha_max = 0.2  # max strength for cepstral delta

                import os as _os
                mode = str(_os.environ.get('L2H_SCHEDULE_MODE', getattr(cfg, 'l2h_schedule_mode', 'abs')))
                mode = 'resume_rel' if mode.lower() in ('resume_rel', 'rel') else 'abs'
                if mode == 'resume_rel' and resume_step_marker is not None:
                    base = int(resume_step_marker)
                    rel = max(0, int(global_step - base))
                else:
                    rel = max(0, int(global_step))

                # Linear progress
                p = max(0.0, min(1.0, float(rel - ceps_warm) / max(1, ceps_blend)))
                # Cosine ramp for smooth introduction
                p_smooth = 0.5 * (1.0 - math.cos(math.pi * p))
                # Quadratic for extra stability (slower early ramp)
                alpha_c = ceps_alpha_max * (p_smooth ** 2)

                if hasattr(model, 'ceps_delta_alpha'):
                    model.ceps_delta_alpha = float(alpha_c)
            except Exception:
                pass

            # F0 decoder content-attention schedule：通过外部 cfg 逐步放大
            # F0VUVDecoder 内部的 cross-attention 贡献，以避免从旧
            # checkpoint 直接启用新结构导致训练初期震荡。
            try:
                warm_f0 = int(getattr(cfg, 'f0_cond_attn_warmup_steps', 0))
                alpha_max = float(getattr(cfg, 'f0_cond_attn_max_alpha', 1.0))
                if warm_f0 > 0:
                    pf = max(0.0, min(1.0, float(global_step) / float(max(1, warm_f0))))
                    alpha_f0 = alpha_max * pf
                else:
                    alpha_f0 = alpha_max
                if hasattr(model, 'f0vuv_dec') and hasattr(model.f0vuv_dec, 'attn_alpha'):
                    model.f0vuv_dec.attn_alpha = float(alpha_f0)
            except Exception:
                pass

            # Teacher-forcing anneal weights (F0 period + frame_corr gate)
            try:
                def _lin_anneal(start: Optional[int], end: Optional[int], step: int) -> float:
                    if start is None or end is None or end <= start:
                        return 1.0
                    if step <= start:
                        return 1.0
                    if step >= end:
                        return 0.0
                    return 1.0 - float(step - start) / float(end - start)

                w_tf = _lin_anneal(getattr(cfg, 'tf_start_step', None),
                                   getattr(cfg, 'tf_end_step', None),
                                   int(global_step))
                try:
                    setattr(model, 'tf_period_w', float(w_tf))
                    setattr(model, 'tf_union_w', float(w_tf))
                except Exception:
                    pass
            except Exception:
                pass

            # Forward pass with optional AMP autocast
            with autocast(enabled=use_amp):
                out = model_forward(model, batch, channel_sim, cfg, device)
                total_loss, loss_dict, grad_info = compute_losses(model, out, cfg, device, global_step=global_step)

                # 显式从前向输出中提取 content/F0 的 RVQ VQ loss，
                # 以便在日志中单独观察 vq_loss_c / vq_loss_f0。
                try:
                    vq_c = out.get("vq_loss_content", None)
                    if isinstance(vq_c, torch.Tensor):
                        loss_dict.setdefault("vq_loss_c", float(vq_c.detach().item()))
                    vq_f = out.get("vq_loss_f0", None)
                    if isinstance(vq_f, torch.Tensor):
                        loss_dict.setdefault("vq_loss_f0", float(vq_f.detach().item()))
                except Exception:
                    pass

                # 额外：bit-only 路径约束 / 诊断
                # - 静音 RMS 约束：鼓励纯 bits→decode_from_bits_offline 路径上静音更干净
                # - Teacher distillation：让 decode_from_bits_offline 输出模仿
                #   decode_hash_codec/forward_with_hash 产生的 teacher 音频。
                # - bit_only_eval：即使不启用额外 loss，也可在可视化步
                #   上跑一遍 bits→音频路径，用于 PESQ/STOI/F0/Bark 统计和
                #   DBG_BITS_FSK 相关调试，并将结果写入 CSV。
                #
                # 三者共用同一批 bits，避免重复 encode。
                lam_bit = float(getattr(cfg, 'lambda_bit_only_silence', 0.0))
                lam_bit_distill = float(getattr(cfg, 'lambda_bit_only_distill', 0.0))
                # 在仅开启 bit_only_eval（不加 loss）时，只在可视化步
                # 触发 bit-only 路径，以避免对训练主循环增加过多开销。
                viz_every = int(getattr(cfg, 'viz_every_steps', 0) or 0)
                do_bit_eval_step = (
                    bool(getattr(cfg, 'bit_only_eval', False))
                    and viz_every > 0
                    and int(global_step) % viz_every == 0
                )
                enable_bit_path = (lam_bit > 0.0 or lam_bit_distill > 0.0 or do_bit_eval_step)
                if enable_bit_path and getattr(model, 'with_hash', False):
                    # 默认记录为 0.0，便于日志观察该项是否启用
                    loss_dict.setdefault('bit_only_silence', 0.0)
                    if lam_bit_distill > 0.0:
                        loss_dict.setdefault('bit_only_distill_sc', 0.0)
                    try:
                        B_cur = out["audio"].size(0)
                        B_eval = min(B_cur, int(getattr(cfg, 'bit_only_eval_max_samples', 2)))
                        if B_eval > 0:
                            audio_eval = out["audio"][:B_eval].to(device)  # [B_eval,L]
                            # 取 FARGAN 特征：优先 batch['x']，其次 batch['features']
                            feats_src = None
                            if "x" in batch:
                                feats_src = batch["x"]
                            elif "features" in batch:
                                feats_src = batch["features"]
                            if feats_src is None:
                                raise KeyError("bit_only_silence_loss requires batch['x'] or batch['features']")
                            feats_eval = feats_src[:B_eval].to(device)

                            # 通过 encode_hash_codec 导出 bits 与 CSI/meta。
                            # 为了让 bit_only_eval 更贴近 JSCC+FSK 脚本的部署路径，
                            # 这里使用 clean bits（use_noisy_bits=False），将“信道噪声”
                            # 的模拟完全交给外部 FSK 仿真脚本；bit_only_eval 只关注
                            # "给定比特流 → decode_quant_codec" 这段解码行为。
                            bits_c, bits_f, bits_s, meta_eval = model.encode_hash_codec(
                                audio=audio_eval,
                                fargan_feats=feats_eval,
                                channel_sim=channel_sim,
                                snr_min_db=cfg.snr_min_db,
                                snr_max_db=cfg.snr_max_db,
                                use_noisy_bits=False,
                            )
                            if os.getenv("DBG_BITS","0") == "1":
                                csi = meta_eval.get("csi_vec", None)
                                print(f"[bit_only_eval] csi_vec={csi}")
                                # 同时导出 clean bits 用来估计 BER（不改变你现有 decode，只做统计）
                                bits_c_cl, bits_f_cl, bits_s_cl, meta2 = model.encode_hash_codec(
                                    audio=audio_eval, fargan_feats=feats_eval,
                                    channel_sim=channel_sim, snr_min_db=args.snr_min_db, snr_max_db=args.snr_max_db,
                                    return_meta=True, use_noisy_bits=False
                                )
                                if bits_f_cl is not None and bits_f is not None:
                                    ber_f = float((torch.as_tensor(bits_f_cl)*torch.as_tensor(bits_f) < 0).float().mean().item())
                                    print(f"[bit_only_eval] f0_BER_est={ber_f:.4f}")
                                if bits_c_cl is not None and bits_c is not None:
                                    ber_c = float((torch.as_tensor(bits_c_cl)*torch.as_tensor(bits_c) < 0).float().mean().item())
                                    print(f"[bit_only_eval] content_BER_est={ber_c:.4f}")

                            # bits 视作常量，避免将该损失反向到 encoder 端
                            bits_c = bits_c.detach() if isinstance(bits_c, torch.Tensor) else bits_c
                            bits_f = bits_f.detach() if isinstance(bits_f, torch.Tensor) else bits_f
                            bits_s = bits_s.detach() if isinstance(bits_s, torch.Tensor) else bits_s

                            # 在 bit_only_eval 中额外模拟一次 FSK 调制/解调带来的 bit 错误，
                            # 以便更贴近 pcm_segment_infer_jscc_fsk.py 的整体行为：
                            #   JSCC bits -> FSK modem + AWGN -> 硬判决 bits。
                            # 这里采用一个简单的 BSC 近似：
                            #   - 根据 *本批次训练信道的 SNR* 估计 BPSK+AWGN 理论 BER；
                            #   - 以该 BER 在 ±1 比特上做随机翻转，并统计经验 BER。
                            def _simulate_fsk_bit_errors(bits, snr_db: Optional[float]):
                                import math as _math
                                if bits is None or snr_db is None:
                                    return bits, None, None
                                b = torch.as_tensor(bits, device=device, dtype=torch.float32)
                                if b.numel() == 0:
                                    return b, None, None
                                # 允许 {0,1} 或 {-1,+1} 或软值；这里统一视作 ±1 来翻转。
                                bmin = float(b.min().item())
                                bmax = float(b.max().item())
                                if bmin >= 0.0 and bmax <= 1.0:
                                    b = b * 2.0 - 1.0
                                # BPSK in AWGN 近似 BER：0.5*erfc(sqrt(SNR_lin))
                                snr_lin = 10.0 ** (float(snr_db) / 10.0)
                                ber_th = 0.5 * _math.erfc(_math.sqrt(max(snr_lin, 1e-8)))
                                ber_th = max(0.0, min(ber_th, 0.5))
                                if ber_th <= 0.0:
                                    return b, ber_th, 0.0
                                flip = (torch.rand_like(b) < ber_th)
                                b_noisy = torch.where(flip, -b, b)
                                ber_emp = float(flip.float().mean().item())
                                if os.environ.get('DBG_BITS_FSK', '0') == '1':
                                    try:
                                        xs = b_noisy.detach().flatten()
                                        print(
                                            f"[bit_only][FSK] snr_db={snr_db:.2f} ber~{ber_th:.4e} "
                                            f"shape={tuple(b_noisy.shape)} min={xs.min().item():+.3f} "
                                            f"max={xs.max().item():+.3f} mean={xs.mean().item():+.3f} std={xs.std().item():+.3f}"
                                        )
                                    except Exception:
                                        pass
                                return b_noisy, ber_th, ber_emp

                            # JSCC+FSK 部署路径在解码端通常只依赖一个简化的 SNR
                            #（pcm_segment_infer_jscc_fsk.py 通过 --snr_db 传入），
                            # 因此在 bit_only_eval 中，我们不再复用训练时的逐帧 CSI，
                            # 而是将 snr_db 显式传给 decode_from_bits_offline，
                            # 让其构造形如 [snr_db,0,0,1] 的简化 CSI 向量。
                            # 从 encode_hash_codec meta 中提取训练时采样的 CSI，
                            # 使用其中的 snr_proxy 作为本次 bit-only 仿真的 SNR，
                            # 以确保 bit 翻转分布与 decode 端 CSI 基本对齐。
                            csi_vec = None
                            snr_db_bits = None
                            if isinstance(meta_eval, dict) and 'csi_vec' in meta_eval:
                                try:
                                    csi_val = torch.as_tensor(meta_eval['csi_vec'], device=device, dtype=torch.float32)
                                    # csi_vec: [B,4]，第 0 维为 snr_proxy
                                    if csi_val.dim() == 2 and csi_val.size(1) >= 1:
                                        snr_db_bits = float(csi_val[:, 0].mean().item())
                                        csi_vec = csi_val
                                except Exception:
                                    snr_db_bits = None

                            # 若能拿到 snr_db_bits，则在 decode 前对 bits 注入一次
                            # FSK 风格的 bit 错误；否则保持 bits 完全干净。
                            snr_db_bits_val: Optional[float] = snr_db_bits
                            ber_th_val: Optional[float] = None
                            ber_emp_val: Optional[float] = None
                            if snr_db_bits is not None:
                                bits_c, ber_th_c, ber_emp_c = _simulate_fsk_bit_errors(bits_c, snr_db_bits)
                                bits_f, _, _ = _simulate_fsk_bit_errors(bits_f, snr_db_bits)
                                bits_s, _, _ = _simulate_fsk_bit_errors(bits_s, snr_db_bits)
                                ber_th_val = ber_th_c
                                ber_emp_val = ber_emp_c
                            def _dbg_bits(name, x, max_print=0):
                                if x is None:
                                    print(f"[bit_only][{name}] = None")
                                    return
                                x = torch.as_tensor(x)
                                xs = x.detach().flatten()
                                # 注意：不要 torch.unique，太慢；用比例足够
                                frac_pos = (xs > 0).float().mean().item()
                                frac_neg = (xs < 0).float().mean().item()
                                frac_zero = (xs == 0).float().mean().item()
                                print(f"[bit_only][{name}] shape={tuple(x.shape)} dtype={x.dtype} "
                                    f"min={xs.min().item():+.3f} max={xs.max().item():+.3f} "
                                    f"mean={xs.mean().item():+.3f} std={xs.std().item():+.3f} "
                                    f"pos={frac_pos:.3f} neg={frac_neg:.3f} zero={frac_zero:.3f}")
                                if max_print > 0:
                                    print(f"  sample={xs[:max_print].cpu().tolist()}")
                            _dbg_bits("bits_c", bits_c)
                            _dbg_bits("bits_f", bits_f)
                            _dbg_bits("bits_s", bits_s)

                            print(f"[bit_only][meta] keys={list(meta_eval.keys())}")
                            print(f"[bit_only][meta] T={meta_eval.get('T')} F_mel={meta_eval.get('F_mel')} hw={meta_eval.get('hw')}")
                            cv = meta_eval.get("csi_vec", None)
                            if cv is None:
                                print("[bit_only][meta] csi_vec=None")
                            else:
                                cvt = torch.as_tensor(cv)
                                print(f"[bit_only][meta] csi_vec shape={tuple(cvt.shape)} min={cvt.min().item():+.3f} max={cvt.max().item():+.3f} mean={cvt.mean().item():+.3f}")
                            if os.getenv("DBG_BITS_ONLY", "0") == "1":
                                content_vm = getattr(model, 'content_vmamba', None)
                                has_dec = hasattr(content_vm, "decoder") if content_vm is not None else False
                                has_de  = hasattr(content_vm, "decode") if content_vm is not None else False
                                print(f"[vmamba] has decoder={has_dec} has decode={has_de} using="
                                    f"{'decoder' if has_dec else ('decode' if has_de else 'none')}")

                            out_bits = model.decode_from_bits_offline(
                                bits_content=bits_c,
                                bits_f0=bits_f,
                                bits_stats=bits_s,
                                f0_T=int(meta_eval.get('T', feats_eval.size(1))),
                                target_len=int(audio_eval.size(1)),
                                csi_vec=csi_vec,
                                snr_db=None,
                                content_hw=meta_eval.get('hw', None),
                            )

                            audio_bits = out_bits.get('audio_hat')  # [B_eval,L]
                            if isinstance(audio_bits, torch.Tensor):
                                # 使用与 compute_losses 相同的帧 RMS + 静音判定逻辑
                                def _frame_rms_local(x: torch.Tensor, frame_len: int = 160, hop: int = 160) -> torch.Tensor:
                                    Bf, Lf = x.shape
                                    if Lf < frame_len:
                                        pad = frame_len - Lf
                                        x = torch.nn.functional.pad(x, (0, pad))
                                        Lf = frame_len
                                    x_frames = x.unfold(dimension=1, size=frame_len, step=hop)
                                    return torch.sqrt(x_frames.pow(2).mean(dim=-1) + 1e-8)

                                # 1) 静音 RMS 约束（可选）
                                if lam_bit > 0.0:
                                    rms_real = _frame_rms_local(audio_eval)   # [B_eval,Tr]
                                    eps = 1e-8
                                    rms_max = rms_real.max(dim=1, keepdim=True).values.clamp_min(eps)
                                    rms_norm = rms_real / rms_max
                                    rms_db = 20.0 * torch.log10(rms_norm + eps)
                                    rms_thr_db = float(getattr(cfg, 'silence_rms_thr_db', -35.0))
                                    sil_mask = (rms_db <= rms_thr_db).to(audio_bits.dtype)  # [B_eval,Tr]

                                    rms_bits = _frame_rms_local(audio_bits)   # [B_eval,Tr']
                                    T_use = min(rms_bits.size(1), sil_mask.size(1))
                                    rms_bits = rms_bits[:, :T_use]
                                    sil_mask = sil_mask[:, :T_use]

                                    loss_bit = (rms_bits * sil_mask).sum() / (sil_mask.sum() + 1e-6)
                                    total_loss = total_loss + lam_bit * loss_bit
                                    loss_dict['bit_only_silence'] = float(loss_bit.item())

                                # 3) Optional objective metrics (PESQ/STOI) on bit-only path, and
                                #    append to CSV for later analysis. 仅在：
                                #       - 启用了 bit-only loss（lambda_bit_only_* > 0）；或
                                #       - 当前为 bit_only_eval 可视化步
                                #     时计算，以控制额外开销。
                                try:
                                    if bit_csv_path is not None and (lam_bit > 0.0 or lam_bit_distill > 0.0 or do_bit_eval_step):
                                        idx0 = 0
                                        ref_np = audio_eval[idx0].detach().cpu().numpy()
                                        deg_np = audio_bits[idx0].detach().cpu().numpy()
                                        pesq_score, stoi_score = _compute_pesq_stoi(ref_np, deg_np, sample_rate=16000)
                                        visqol_score = _compute_visqol(ref_np, deg_np, sample_rate=16000)

                                        # estimate F0/Bark MSE from comparison plots (same as viz)
                                        f0_mse = float(out.get('viz_f0_mse_bits', 0.0)) if isinstance(out.get('viz_f0_mse_bits', None), (float, int)) else 0.0
                                        mel_mse = float(out.get('viz_mel_mse_bits', 0.0)) if isinstance(out.get('viz_mel_mse_bits', None), (float, int)) else 0.0

                                        bit_csv_path.parent.mkdir(parents=True, exist_ok=True)
                                        is_new_csv = not bit_csv_path.is_file()
                                        fieldnames = ['step', 'stoi', 'pesq', 'f0_mse', 'mel_mse', 'snr_db', 'ber_th', 'ber_emp', 'visqol']
                                        row = {
                                            'step': int(global_step),
                                            'stoi': f"{stoi_score:.6f}" if stoi_score is not None else '',
                                            'pesq': f"{pesq_score:.6f}" if pesq_score is not None else '',
                                            'f0_mse': f"{f0_mse:.6f}",
                                            'mel_mse': f"{mel_mse:.6f}",
                                            'snr_db': f"{snr_db_bits_val:.4f}" if snr_db_bits_val is not None else '',
                                            'ber_th': f"{ber_th_val:.6e}" if ber_th_val is not None else '',
                                            'ber_emp': f"{ber_emp_val:.6e}" if ber_emp_val is not None else '',
                                            'visqol': f"{visqol_score:.6f}" if visqol_score is not None else '',
                                        }
                                        with bit_csv_path.open('a', newline='', encoding='utf-8') as _fcsv:
                                            _writer = csv.DictWriter(_fcsv, fieldnames=fieldnames)
                                            if is_new_csv:
                                                _writer.writeheader()
                                            _writer.writerow(row)
                                except Exception as _metric_e:
                                    if os.environ.get('DBG_STAGE25', '0') == '1':
                                        print(f"[bit_only_eval] metrics CSV append failed: {_metric_e}")

                                # 2) Teacher distillation（可选）：
                                #    使用 decode_hash_codec/forward_with_hash 输出作为 teacher，
                                #    对同一批 bits 解码得到的 audio_bits 做 MR-STFT 谱收敛对齐。
                                if lam_bit_distill > 0.0:
                                    try:
                                        with torch.no_grad():
                                            teacher_out = model.decode_hash_codec(
                                                bits_content=bits_c,
                                                bits_f0=bits_f,
                                                audio=audio_eval,
                                                fargan_feats=feats_eval,
                                                channel_sim=channel_sim,
                                                snr_min_db=cfg.snr_min_db,
                                                snr_max_db=cfg.snr_max_db,
                                                target_len=int(audio_eval.size(1)),
                                            )
                                        audio_teacher = teacher_out.get('audio_hat')
                                        if isinstance(audio_teacher, torch.Tensor):
                                            # 对齐长度，防御性裁剪
                                            L_use = min(audio_teacher.size(1), audio_bits.size(1))
                                            aud_t = audio_teacher[:, :L_use]
                                            aud_s = audio_bits[:, :L_use]

                                            # 复用 compute_losses 中的 MR-STFT 配置
                                            if getattr(cfg, 'stft_preset', 'aether') == 'fargan':
                                                fs = [2560, 1280, 640, 320, 160, 80]
                                                hs = [640, 320, 160, 80, 40, 20]
                                                wl = [2560, 1280, 640, 320, 160, 80]
                                            else:
                                                fs = [1024, 512, 256, 128]
                                                hs = [256, 128, 64, 32]
                                                wl = [1024, 512, 256, 128]

                                            loss_dist_sc = multi_resolution_sc_loss(
                                                aud_s,
                                                aud_t,
                                                device=device,
                                                fft_sizes=fs,
                                                hop_sizes=hs,
                                                win_lengths=wl,
                                            )
                                            total_loss = total_loss + lam_bit_distill * loss_dist_sc
                                            loss_dict['bit_only_distill_sc'] = float(loss_dist_sc.item())
                                    except Exception:
                                        if os.environ.get('DBG_STAGE25', '0') == '1':
                                            print('[bit_only_distill] skipped due to error', flush=True)
                    except Exception as _bit_sil_e:
                        if os.environ.get('DBG_STAGE25', '0') == '1':
                            print(f"[bit_only_silence] skipped due to error: {_bit_sil_e}")

            # 记录当前 teacher-forcing 权重（便于确认 tf_start_step/tf_end_step 是否生效）
            try:
                if hasattr(model, 'tf_period_w'):
                    loss_dict['tf_w'] = float(getattr(model, 'tf_period_w'))
            except Exception:
                pass

            # wandb 标量日志：每 cfg.wandb_log_freq 步记录一次所有 loss_dict
            try:
                if use_wandb and wandb is not None:
                    log_freq = int(getattr(cfg, 'wandb_log_freq', 10))
                    if log_freq > 0 and (global_step % log_freq == 0):
                        log_data = {"total": float(total_loss.detach().item())}
                        log_data.update(loss_dict)
                        wandb.log(log_data, step=int(global_step))
            except Exception as _we:
                if os.environ.get('DBG_STAGE25', '0') == '1':
                    print(f"[wandb] log failed at step {global_step}: {_we}")

            # OSCE FD-based adversarial training on audio_hat（LSGAN+feature matching）。
            # 注意：判别器更新使用 detach 的 fake/real，生成器更新必须使用带梯度的 fake，
            # 否则 osce_gan 对生成器没有任何约束。
            if osce_disc is not None and bool(getattr(cfg, 'bfcc_gan', False)):
                try:
                    # --- 判别器更新：使用 detach，避免反向传播到生成器 ---
                    with torch.no_grad():
                        x_fake_full_d = out["audio_hat"].detach().unsqueeze(1)
                        x_real_full_d = out["audio"].detach().unsqueeze(1)
                        min_len = min(x_fake_full_d.size(-1), x_real_full_d.size(-1))
                        if x_fake_full_d.size(-1) != min_len:
                            x_fake_full_d = x_fake_full_d[..., :min_len]
                        if x_real_full_d.size(-1) != min_len:
                            x_real_full_d = x_real_full_d[..., :min_len]

                    optimizer_osce_disc.zero_grad(set_to_none=True)
                    with autocast(enabled=use_amp):
                        scores_fake = osce_disc(x_fake_full_d)
                        scores_real = osce_disc(x_real_full_d)

                        d_loss = 0.0
                        num_scales = len(scores_fake)
                        for sf, sr in zip(scores_fake, scores_real):
                            d_loss = d_loss + (sf[-1] ** 2).mean()
                            d_loss = d_loss + ((1.0 - sr[-1]) ** 2).mean()
                        d_loss = 0.5 * d_loss / float(max(1, num_scales))

                    d_loss.backward()
                    optimizer_osce_disc.step()
                    loss_dict["osce_disc"] = float(d_loss.item())

                    # --- 生成器更新：重新计算带梯度的 fake，判别器参数不需要梯度 ---
                    for p in osce_disc.parameters():
                        p.requires_grad_(False)
                    try:
                        x_fake_full_g = out["audio_hat"].unsqueeze(1)
                        x_real_full_g = out["audio"].detach().unsqueeze(1)
                        min_len_g = min(x_fake_full_g.size(-1), x_real_full_g.size(-1))
                        if x_fake_full_g.size(-1) != min_len_g:
                            x_fake_full_g = x_fake_full_g[..., :min_len_g]
                        if x_real_full_g.size(-1) != min_len_g:
                            x_real_full_g = x_real_full_g[..., :min_len_g]

                        with autocast(enabled=use_amp):
                            scores_fake_g = osce_disc(x_fake_full_g)
                            adv_loss = 0.0
                            num_scales_g = len(scores_fake_g)
                            for sf in scores_fake_g:
                                adv_loss = adv_loss + ((1.0 - sf[-1]) ** 2).mean() / float(max(1, num_scales_g))

                            fmap_w = float(getattr(cfg, 'bfcc_gan_fmap_weight', 1.0))
                            fmap_loss_val = 0.0
                            if fmap_w > 0.0:
                                scores_real_g = osce_disc(x_real_full_g)
                                num_discs = len(scores_real_g)
                                for k in range(num_discs):
                                    num_layers = len(scores_fake_g[k]) - 1
                                    if num_layers <= 0:
                                        continue
                                    f = 4.0 / float(num_discs * num_layers)
                                    for l in range(num_layers):
                                        fmap_loss_val = fmap_loss_val + f * F.l1_loss(
                                            scores_fake_g[k][l], scores_real_g[k][l].detach()
                                        )

                            lam_gan = float(getattr(cfg, 'bfcc_gan_lambda', 1.0))
                            gen_adv = lam_gan * adv_loss + fmap_w * fmap_loss_val
                            total_loss = total_loss + gen_adv
                            loss_dict["osce_gan"] = float(gen_adv.item())
                    finally:
                        for p in osce_disc.parameters():
                            p.requires_grad_(True)
                except Exception as _gan_e:
                    print(f"[OSCE-GAN] skipped due to error at step {global_step}: {_gan_e}")

            # HiFi-GAN style MPD+MSD adversarial training on raw waveform
            if mpd is not None and msd is not None and optimizer_hifi_disc is not None:
                lam_adv_hifi = float(getattr(cfg, 'lambda_hifi_adv', 0.0))
                lam_fm_hifi = float(getattr(cfg, 'lambda_hifi_fm', 0.0))
                warm_hifi = int(getattr(cfg, 'hifi_adv_warmup_steps', 0))
                if (lam_adv_hifi > 0.0 or lam_fm_hifi > 0.0) and global_step >= warm_hifi:
                    # --- D update (LSGAN): real→1, fake→0 ---
                    try:
                        y_real = out['audio'].detach()
                        y_fake = out['audio_hat'].detach()
                        min_len = min(y_real.size(-1), y_fake.size(-1))

                        # 为减轻显存压力，仅在 HiFi 判别器上使用较短片段
                        crop_len = int(getattr(cfg, 'hifi_adv_crop_len', 0))
                        if crop_len > 0 and min_len > crop_len:
                            start = (min_len - crop_len) // 2
                            end = start + crop_len
                            y_real = y_real[:, start:end]
                            y_fake = y_fake[:, start:end]
                        else:
                            if y_real.size(-1) != min_len:
                                y_real = y_real[:, :min_len]
                            if y_fake.size(-1) != min_len:
                                y_fake = y_fake[:, :min_len]

                        y_real_d = y_real.unsqueeze(1)
                        y_fake_d = y_fake.unsqueeze(1)

                        for p in mpd.parameters():
                            p.requires_grad_(True)
                        for p in msd.parameters():
                            p.requires_grad_(True)

                        optimizer_hifi_disc.zero_grad(set_to_none=True)
                        with autocast(enabled=use_amp):
                            y_df_hat_r, y_df_hat_g, _, _ = mpd(y_real_d, y_fake_d)
                            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y_real_d, y_fake_d)
                            loss_disc_f, _, _ = hifi_discriminator_loss(y_df_hat_r, y_df_hat_g)
                            loss_disc_s, _, _ = hifi_discriminator_loss(y_ds_hat_r, y_ds_hat_g)
                            loss_disc_all = loss_disc_f + loss_disc_s

                        loss_disc_all.backward()
                        optimizer_hifi_disc.step()
                        loss_dict['hifi_disc'] = float(loss_disc_all.detach().item())
                    except Exception as _e:
                        loss_dict['hifi_disc'] = 0.0
                        if os.environ.get('DBG_HIFI_ADV', '0') == '1':
                            print(f"[HiFi-ADV] D update skipped: {_e}")

                    # --- G adversarial + feature matching ---
                    try:
                        for p in mpd.parameters():
                            p.requires_grad_(False)
                        for p in msd.parameters():
                            p.requires_grad_(False)

                        y_real_g = out['audio']
                        y_fake_g = out['audio_hat']
                        min_len_g = min(y_real_g.size(-1), y_fake_g.size(-1))

                        crop_len = int(getattr(cfg, 'hifi_adv_crop_len', 0))
                        if crop_len > 0 and min_len_g > crop_len:
                            start = (min_len_g - crop_len) // 2
                            end = start + crop_len
                            y_real_g = y_real_g[:, start:end]
                            y_fake_g = y_fake_g[:, start:end]
                        else:
                            if y_real_g.size(-1) != min_len_g:
                                y_real_g = y_real_g[:, :min_len_g]
                            if y_fake_g.size(-1) != min_len_g:
                                y_fake_g = y_fake_g[:, :min_len_g]

                        y_real_g1 = y_real_g.unsqueeze(1)
                        y_fake_g1 = y_fake_g.unsqueeze(1)

                        with autocast(enabled=use_amp):
                            y_df_hat_r_g, y_df_hat_g_g, fmap_f_r, fmap_f_g = mpd(y_real_g1, y_fake_g1)
                            y_ds_hat_r_g, y_ds_hat_g_g, fmap_s_r, fmap_s_g = msd(y_real_g1, y_fake_g1)

                            loss_gen_f, _ = hifi_generator_loss(y_df_hat_g_g)
                            loss_gen_s, _ = hifi_generator_loss(y_ds_hat_g_g)
                            loss_g_adv = loss_gen_f + loss_gen_s

                            loss_fm_f = hifi_feature_loss(fmap_f_r, fmap_f_g)
                            loss_fm_s = hifi_feature_loss(fmap_s_r, fmap_s_g)
                            loss_g_fm = loss_fm_f + loss_fm_s

                            total_loss = total_loss + lam_adv_hifi * loss_g_adv + lam_fm_hifi * loss_g_fm

                        if lam_adv_hifi > 0.0:
                            loss_dict['hifi_adv_g'] = float((lam_adv_hifi * loss_g_adv).detach().item())
                        if lam_fm_hifi > 0.0:
                            loss_dict['hifi_adv_fm'] = float((lam_fm_hifi * loss_g_fm).detach().item())
                    except Exception as _e:
                        if os.environ.get('DBG_HIFI_ADV', '0') == '1':
                            print(f"[HiFi-ADV] G update skipped: {_e}")
                        loss_dict.setdefault('hifi_adv_g', 0.0)
                        loss_dict.setdefault('hifi_adv_fm', 0.0)

            # HF adversarial: 4–8kHz STFT discriminator (MultiScaleSpecDisc)
            if hf_disc is not None and optimizer_hf_disc is not None:
                def _as_scales(obj):
                    if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], list):
                        return obj
                    else:
                        return [obj]

                warm = int(getattr(cfg, 'hf_adv_warmup_steps', 10000))
                if global_step >= warm:
                    # --- D update (LSGAN): real→1, fake→0 ---
                    try:
                        for p in hf_disc.parameters():
                            p.requires_grad_(True)
                        y_real = out['audio'].detach()
                        y_fake = out['audio_hat'].detach()
                        # 对齐时长，避免 STFT 帧数不同导致 feature matching 形状不一致
                        min_len = min(y_real.size(-1), y_fake.size(-1))
                        if y_real.size(-1) != min_len:
                            y_real = y_real[:, :min_len]
                        if y_fake.size(-1) != min_len:
                            y_fake = y_fake[:, :min_len]
                        scores_fake = hf_disc(y_fake)
                        scores_real = hf_disc(y_real)
                        scales_fake = _as_scales(scores_fake)
                        scales_real = _as_scales(scores_real)
                        d_loss = 0.0
                        for sc in scales_fake:
                            d_loss = d_loss + (sc[-1] ** 2).mean()
                        for sc in scales_real:
                            d_loss = d_loss + ((1.0 - sc[-1]) ** 2).mean()
                        d_loss = 0.5 * d_loss / max(1, len(scales_fake))
                        optimizer_hf_disc.zero_grad(set_to_none=True)
                        d_loss.backward()
                        optimizer_hf_disc.step()
                        loss_dict['hf_adv_d'] = float(d_loss.detach().item())
                    except Exception as _e:
                        loss_dict['hf_adv_d'] = 0.0
                        if os.environ.get('DBG_HF_ADV', '0') == '1':
                            print(f"[HF-ADV] D update skipped: {_e}")

                    # --- G adversarial + feature matching ---
                    lam_adv = float(getattr(cfg, 'lambda_hf_adv', 0.0))
                    lam_fm = float(getattr(cfg, 'lambda_hf_fm', 0.0))
                    if lam_adv > 0.0 or lam_fm > 0.0:
                        try:
                            for p in hf_disc.parameters():
                                p.requires_grad_(False)
                            y_fake_g = out['audio_hat']
                            y_real_g = out['audio']
                            # 与 D 更新保持一致：先在时域对齐长度再送入判别器
                            min_len_g = min(y_real_g.size(-1), y_fake_g.size(-1))
                            if y_real_g.size(-1) != min_len_g:
                                y_real_g = y_real_g[:, :min_len_g]
                            if y_fake_g.size(-1) != min_len_g:
                                y_fake_g = y_fake_g[:, :min_len_g]
                            scores_fake_g = hf_disc(y_fake_g)
                            with torch.no_grad():
                                scores_real_g = hf_disc(y_real_g)
                            scales_fake_g = _as_scales(scores_fake_g)
                            scales_real_g = _as_scales(scores_real_g)
                            # GAN (LSGAN): (1-D(G))^2
                            g_loss = 0.0
                            for sc in scales_fake_g:
                                g_loss = g_loss + ((1.0 - sc[-1]) ** 2).mean() / max(1, len(scales_fake_g))
                            # Feature matching
                            fm_loss = 0.0
                            for k in range(len(scales_fake_g)):
                                num_layers = len(scales_fake_g[k]) - 1
                                if num_layers <= 0:
                                    continue
                                f = 4.0 / max(1, len(scales_fake_g)) / num_layers
                                for l in range(num_layers):
                                    fm_loss = fm_loss + f * torch.nn.functional.l1_loss(
                                        scales_fake_g[k][l], scales_real_g[k][l]
                                    )
                            total_loss = total_loss + lam_adv * g_loss + lam_fm * fm_loss
                            val_g = float((lam_adv * g_loss).detach().item())
                            val_fm = float((lam_fm * fm_loss).detach().item())
                            loss_dict['hf_adv_g'] = val_g
                            loss_dict['hf_adv_fm'] = val_fm

                            if os.environ.get('DBG_HF_ADV', '0') == '1':
                                try:
                                    print(
                                        f"[HF-ADV] step={global_step} lam_adv={lam_adv:.3g} lam_fm={lam_fm:.3g} "
                                        f"raw_g={float(g_loss.item()):.6f} raw_fm={float(fm_loss.item()):.6f} "
                                        f"w_g={val_g:.6f} w_fm={val_fm:.6f}"
                                    )
                                except Exception:
                                    pass
                        except Exception as _e:
                            loss_dict['hf_adv_g'] = 0.0
                            loss_dict['hf_adv_fm'] = 0.0
                            if os.environ.get('DBG_HF_ADV', '0') == '1':
                                print(f"[HF-ADV] G update skipped: {_e}")
                        finally:
                            for p in hf_disc.parameters():
                                p.requires_grad_(True)

            # Mel-domain HF adversarial: 判别高频 mel 纹理
            if hf_mel_disc is not None and optimizer_hf_mel_disc is not None:
                warm_m = int(getattr(cfg, 'hf_mel_adv_warmup_steps', 5000))
                lam_m_adv = float(getattr(cfg, 'lambda_hf_mel_adv', 0.0))
                lam_m_fm = float(getattr(cfg, 'lambda_hf_mel_fm', 0.0))
                if global_step >= warm_m and (lam_m_adv > 0.0 or lam_m_fm > 0.0):
                    try:
                        # D update
                        for p in hf_mel_disc.parameters():
                            p.requires_grad_(True)
                        mel_ref = out.get('mel')
                        mel_gen = out.get('mel_hat_refined', out.get('mel_hat'))
                        if isinstance(mel_ref, torch.Tensor) and isinstance(mel_gen, torch.Tensor):
                            Bm, Tm, Fm = mel_ref.shape
                            hf_start = int(getattr(cfg, 'hf_mel_low_bins', int(getattr(cfg, 'l2h_low_bins', 10))))
                            hf_start = max(1, min(hf_start, Fm - 1))
                            mel_hf_real = mel_ref[:, :, hf_start:]
                            mel_hf_fake = mel_gen[:, :, hf_start:]
                            T_use = min(mel_hf_real.size(1), mel_hf_fake.size(1))
                            mel_hf_real = mel_hf_real[:, :T_use, :]
                            mel_hf_fake = mel_hf_fake[:, :T_use, :]
                            d_real = hf_mel_disc(mel_hf_real.detach())
                            d_fake = hf_mel_disc(mel_hf_fake.detach())
                            loss_d_real = ((d_real - 1.0) ** 2).mean()
                            loss_d_fake = (d_fake ** 2).mean()
                            loss_d_mel = 0.5 * (loss_d_real + loss_d_fake)
                            optimizer_hf_mel_disc.zero_grad(set_to_none=True)
                            loss_d_mel.backward()
                            optimizer_hf_mel_disc.step()
                            loss_dict['hf_mel_adv_d'] = float(loss_d_mel.detach().item())
                        else:
                            loss_dict['hf_mel_adv_d'] = 0.0
                    except Exception:
                        loss_dict['hf_mel_adv_d'] = 0.0

                    # G update (adv + feature matching)
                    try:
                        mel_ref = out.get('mel')
                        mel_gen = out.get('mel_hat_refined', out.get('mel_hat'))
                        if isinstance(mel_ref, torch.Tensor) and isinstance(mel_gen, torch.Tensor):
                            Bm, Tm, Fm = mel_ref.shape
                            hf_start = int(getattr(cfg, 'hf_mel_low_bins', int(getattr(cfg, 'l2h_low_bins', 10))))
                            hf_start = max(1, min(hf_start, Fm - 1))
                            mel_hf_real = mel_ref[:, :, hf_start:]
                            mel_hf_fake = mel_gen[:, :, hf_start:]
                            T_use = min(mel_hf_real.size(1), mel_hf_fake.size(1))
                            mel_hf_real = mel_hf_real[:, :T_use, :]
                            mel_hf_fake = mel_hf_fake[:, :T_use, :]

                            for p in hf_mel_disc.parameters():
                                p.requires_grad_(False)

                            d_fake_g = hf_mel_disc(mel_hf_fake)
                            with torch.no_grad():
                                d_real_g = hf_mel_disc(mel_hf_real)

                            # LSGAN G loss
                            g_loss_mel = ((d_fake_g - 1.0) ** 2).mean()

                            fm_loss_mel = 0.0
                            if lam_m_fm > 0.0:
                                with torch.no_grad():
                                    feat_real = hf_mel_disc.get_features(mel_hf_real)
                                feat_fake = hf_mel_disc.get_features(mel_hf_fake)
                                if len(feat_real) > 0:
                                    fm_terms = [torch.nn.functional.l1_loss(fr.detach(), ff) for fr, ff in zip(feat_real, feat_fake)]
                                    fm_loss_mel = sum(fm_terms) / float(len(fm_terms))

                            total_loss = total_loss + lam_m_adv * g_loss_mel + lam_m_fm * fm_loss_mel
                            loss_dict['hf_mel_adv_g'] = float((lam_m_adv * g_loss_mel).detach().item())
                            if lam_m_fm > 0.0:
                                loss_dict['hf_mel_adv_fm'] = float((lam_m_fm * fm_loss_mel).detach().item())
                        else:
                            loss_dict['hf_mel_adv_g'] = 0.0
                            loss_dict['hf_mel_adv_fm'] = 0.0
                    except Exception:
                        loss_dict['hf_mel_adv_g'] = 0.0
                        loss_dict['hf_mel_adv_fm'] = 0.0

            # Adversarial: update discriminator first (LSGAN style), then add generator adv losses
            if False and disc is not None and (global_step >= int(getattr(cfg, 'adv_warmup_steps', 5000))):
                try:
                    dom = str(getattr(cfg, 'adv_domain', 'audio')).lower()
                    if dom == 'audio':
                        y_real = out['audio'].detach()
                        y_fake = out['audio_hat'].detach()
                        scores_gen = disc(y_fake)
                        scores_real = disc(y_real)
                    elif dom == 'mel_high':
                        mel_ref = out.get('mel'); mel_gen = out.get('mel_hat_refined', out.get('mel_hat'))
                        if not (isinstance(mel_ref, torch.Tensor) and isinstance(mel_gen, torch.Tensor)):
                            raise RuntimeError('no mel for adv')
                        lb = int(getattr(cfg, 'adv_mel_low_bins', 16))
                        lb = max(1, min(lb, mel_ref.size(-1) - 1))
                        Xr = mel_ref[:, :, lb:].detach().unsqueeze(1)  # [B,1,T,Fh]
                        Xg = mel_gen[:, :, lb:].detach().unsqueeze(1)
                        if bool(getattr(cfg, 'adv_gate_voiced', True)):
                            fc = out.get('frame_corr')
                            if isinstance(fc, torch.Tensor):
                                th = float(getattr(cfg, 'vuv_threshold', 0.3))
                                mask = (fc > th).to(Xr.dtype)[:, :, None]
                                Xr = Xr * mask
                                Xg = Xg * mask
                        scores_gen = disc(Xg)
                        scores_real = disc(Xr)
                    else:  # ceps_hi
                        ceps_ref = out.get('ceps'); ceps_gen = out.get('ceps_hat')
                        if not (isinstance(ceps_ref, torch.Tensor) and isinstance(ceps_gen, torch.Tensor)):
                            raise RuntimeError('no ceps for adv')
                        s0 = int(getattr(cfg, 'adv_ceps_hi_start', 12))
                        s0 = max(1, min(s0, ceps_ref.size(-1) - 1))
                        # reshape to [B,1,T,Fh] to reuse 2D patch disc
                        Xr = ceps_ref[:, :, s0:].detach().unsqueeze(1)
                        Xg = ceps_gen[:, :, s0:].detach().unsqueeze(1)
                        scores_gen = disc(Xg)
                        scores_real = disc(Xr)
                    # Normalize to a list of scales: each element is [feat1,...,featK, logit]
                    def _as_scales(obj):
                        if isinstance(obj, list) and len(obj)>0 and isinstance(obj[0], list):
                            return obj
                        else:
                            return [obj]
                    scales_gen = _as_scales(scores_gen)
                    scales_real = _as_scales(scores_real)
                    d_loss = 0.0
                    for sc in scales_gen:
                        d_loss = d_loss + (sc[-1] ** 2).mean()
                    for sc in scales_real:
                        d_loss = d_loss + ((1.0 - sc[-1]) ** 2).mean()
                    d_loss = 0.5 * d_loss / max(1, len(scales_gen))
                    optimizer_disc.zero_grad(set_to_none=True)
                    d_loss.backward()
                    optimizer_disc.step()
                    loss_dict['adv_d'] = float(d_loss.detach().item())
                except Exception as _e:
                    loss_dict['adv_d'] = 0.0

                # G adversarial + feature matching
                try:
                    dom = str(getattr(cfg, 'adv_domain', 'audio')).lower()
                    if dom == 'audio':
                        y_real = out['audio']
                        y_fake = out['audio_hat']
                        scores_gen = disc(y_fake)
                        with torch.no_grad():
                            scores_real = disc(y_real)
                    elif dom == 'mel_high':
                        mel_ref = out.get('mel'); mel_gen = out.get('mel_hat_refined', out.get('mel_hat'))
                        lb = int(getattr(cfg, 'adv_mel_low_bins', 16))
                        lb = max(1, min(lb, mel_ref.size(-1) - 1))
                        Xr = mel_ref[:, :, lb:].unsqueeze(1)
                        Xg = mel_gen[:, :, lb:].unsqueeze(1)
                        if bool(getattr(cfg, 'adv_gate_voiced', True)):
                            fc = out.get('frame_corr')
                            if isinstance(fc, torch.Tensor):
                                th = float(getattr(cfg, 'vuv_threshold', 0.3))
                                mask = (fc > th).to(Xr.dtype)[:, :, None]
                                Xr = Xr * mask
                                Xg = Xg * mask
                        scores_gen = disc(Xg)
                        with torch.no_grad():
                            scores_real = disc(Xr)
                    else:
                        ceps_ref = out.get('ceps'); ceps_gen = out.get('ceps_hat')
                        s0 = int(getattr(cfg, 'adv_ceps_hi_start', 12))
                        s0 = max(1, min(s0, ceps_ref.size(-1) - 1))
                        Xr = ceps_ref[:, :, s0:].unsqueeze(1)
                        Xg = ceps_gen[:, :, s0:].unsqueeze(1)
                        scores_gen = disc(Xg)
                        with torch.no_grad():
                            scores_real = disc(Xr)
                    def _as_scales(obj):
                        if isinstance(obj, list) and len(obj)>0 and isinstance(obj[0], list):
                            return obj
                        else:
                            return [obj]
                    scales_gen = _as_scales(scores_gen)
                    scales_real = _as_scales(scores_real)
                    # GAN (LSGAN): (1-D(G))^2
                    g_loss = 0.0
                    for sc in scales_gen:
                        g_loss = g_loss + ((1.0 - sc[-1]) ** 2).mean() / max(1, len(scales_gen))
                    # Feature matching
                    fm_loss = 0.0
                    for k in range(len(scales_gen)):
                        num_layers = len(scales_gen[k]) - 1
                        f = 4.0 / max(1, len(scales_gen)) / max(1, num_layers)
                        for l in range(num_layers):
                            fm_loss = fm_loss + f * torch.nn.functional.l1_loss(scales_gen[k][l], scales_real[k][l])
                    lam_adv = float(getattr(cfg, 'lambda_adv', 0.0))
                    lam_fm  = float(getattr(cfg, 'lambda_fm', 0.0))
                    total_loss = total_loss + lam_adv * g_loss + lam_fm * fm_loss
                    loss_dict['adv_g'] = float((lam_adv * g_loss).detach().item())
                    loss_dict['adv_fm'] = float((lam_fm * fm_loss).detach().item())
                except Exception:
                    loss_dict['adv_g'] = 0.0
                    loss_dict['adv_fm'] = 0.0

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total_loss).backward()
            bad_grads = _sanitize_gradients_(model, step=global_step)
            if bad_grads > 0 and (global_step % 50 == 0):
                print(f"[Sanitize] Cleared non-finite grads on {bad_grads} tensors (step {global_step})")
                # 只有在判定为“梯度爆炸”且 DBG_SANITIZE=1 时，才打印详细 loss/F0/VUV 统计，避免日志过载。
                if os.environ.get("DBG_SANITIZE", "0") == "1" and _last_grad_exploded:
                    try:
                        keys = [
                            "wave",
                            "ceps",
                            "ceps_hi",
                            "ceps_weighted",
                            "f0",
                            "f0_smooth",
                            "vuv",
                            "vuv_sil",
                            "vuv_bce",
                            "vuv_ratio",
                            "harm_align",
                            "f0_entropy",
                            # RVQ VQ 损失：方便在梯度爆炸时检查是否数值异常
                            "vq",
                        ]
                        parts = []
                        for k in keys:
                            if k in loss_dict:
                                parts.append(f"{k}={loss_dict[k]:.4f}")
                        msg = " ".join(parts)
                        print(
                            f"[Sanitize] loss snapshot (step {global_step}): "
                            f"total={float(total_loss.item()):.4f} {msg}"
                        )

                        def _stat_tensor(name: str, t: torch.Tensor | None) -> None:
                            if not isinstance(t, torch.Tensor):
                                return
                            v = t.detach().to(torch.float32)
                            if v.numel() == 0:
                                return
                            v = v.reshape(-1)
                            print(
                                f"[Sanitize] {name}: "
                                f"min={v.min().item():+.4f} max={v.max().item():+.4f} "
                                f"mean={v.mean().item():+.4f} std={v.std().item():.4f}"
                            )

                        _stat_tensor("dnn_pitch_hat", out.get("dnn_pitch_hat"))
                        _stat_tensor("dnn_pitch", out.get("dnn_pitch"))
                        _stat_tensor("frame_corr_hat", out.get("frame_corr_hat"))
                        _stat_tensor("frame_corr", out.get("frame_corr"))
                        hb_f = out.get("f0_hash_bits_clean", out.get("f0_hash_bits", None))
                        _stat_tensor("f0_hash_bits_clean", hb_f)
                    except Exception as _san_e:
                        print(f"[Sanitize] DEBUG snapshot failed at step {global_step}: {_san_e}")
            # Unscale gradients before clipping
            scaler.unscale_(optimizer)

            # Optional: CA-specific grad-norm diagnostics for VMambaJSCC2D
            # Enable by setting DBG_CA_GRAD=1 (and optionally CA_GRAD_EVERY=N).
            try:
                if os.environ.get('DBG_CA_GRAD', '0') == '1':
                    try:
                        every = int(os.environ.get('CA_GRAD_EVERY', '200'))
                    except Exception:
                        every = 200
                    if every <= 0:
                        every = 200
                    if global_step % every == 0:
                        for name, p in model.named_parameters():
                            if (
                                'content_vmamba.SNR_embedding' in name
                                or 'content_vmamba.proj_list_enc' in name
                                or 'content_vmamba.proj_list_dec' in name
                            ):
                                if p.grad is not None:
                                    try:
                                        gnorm = float(p.grad.detach().norm().item())
                                    except Exception:
                                        continue
                                    print(f"[CA-GRAD] step={global_step} {name} grad_norm={gnorm:.4e}")
            except Exception as _ca_grad_e:
                if os.environ.get('DBG_STAGE25', '0') == '1':
                    print(f"[CA-GRAD] diagnostics failed: {_ca_grad_e}")

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            # 可选：调试 vocoder 是否被反向更新。
            if os.environ.get('DBG_VOCODER_GRAD', '0') == '1' and (global_step % 100 == 0):
                has_vocoder_grad = any(
                    (p.grad is not None) and torch.isfinite(p.grad).any() for p in model.vocoder.parameters()
                )
                print(f"[DBG] step={global_step} vocoder_has_grad={has_vocoder_grad}")
            scaler.step(optimizer)
            scaler.update()
            # 步后扫描：若某些模块参数出现非有限，立即重置以避免下一个forward崩溃
            repaired = _scan_and_repair_weights_(model)
            if repaired > 0 and (global_step % 50 == 0):
                print(f"[Sanitize] Repaired {repaired} module(s) with non-finite weights (step {global_step})")

            # ===== 新增：定期保存 checkpoint =====
            if cfg.save_every_steps > 0 and global_step > 0:
                if global_step % cfg.save_every_steps == 0:
                    ckpt_path = save_checkpoint(
                        cfg=cfg,
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        global_step=global_step,
                        mpd=mpd,
                        msd=msd,
                        optimizer_hifi_disc=optimizer_hifi_disc,
                    )
                    print(f"[checkpoint] saved to {ckpt_path}")
                    # Optional: run JSCC+FSK offline evaluation for this checkpoint
                    _run_jscc_fsk_eval_if_enabled(cfg, ckpt_path, epoch=epoch, global_step=global_step)

            # 自适应高频加权（数量级对齐）：
            # 目标：使 HF 相关 gn_* 大致落在“参考量级”的同一数量级（无需严控精确值）。
            # 策略：
            #  - 参考量级 base = 所有 gn_* 中 >=1e-4 的均值（若无，则用 1e-4）。
            #  - 若某项 gi < base/3 → 适度增大其 λ；若 gi > 3*base → 适度减小其 λ；否则不动。
            if bool(getattr(cfg, 'adaptive_hf', False)) and (global_step % int(getattr(cfg, 'adaptive_every', 20)) == 0):
                try:
                    # 1) 统计参考量级：仅考虑非 F0 相关的 gn_* 且 >=1e-4，避免 f0_env/f0_wavelet 的大值支配
                    ref_vals = []
                    for k, v in grad_info.items():
                        if not k.startswith('gn_'):
                            continue
                        if k.startswith('gn_f0'):
                            continue
                        try:
                            fv = float(v)
                        except Exception:
                            continue
                        if np.isfinite(fv) and fv >= 1e-4:
                            ref_vals.append(fv)
                    base = float(np.mean(ref_vals)) if len(ref_vals) > 0 else 1e-4
                    # 将 base 限定在 [1e-4, 5e-4]，对齐“同数量级即可”的目标
                    base = max(1e-4, min(base, 5e-4))

                    # 2) 使用 gn_*（RMS-Grad）作为各项当前量级
                    #    初始只针对“真正高频/结构细化”类损失做自适应，避免动到主干重建项。
                    entries = [
                        ('gn_mel_tex',          'lambda_mel_texture',       0.005, 0.15),  # 高频纹理梯度一致性
                        ('gn_mel_band',         'lambda_mel_band_prior',    0.003, 0.12),  # 旧版频带先验（如启用）
                        ('gn_mel_mod',          'lambda_mel_modulation',    0.003, 0.10),  # 高频调制先验（如启用）
                        # L2H 高频细化与纹理保护也纳入自适应
                        ('gn_l2h_high_l1',      'lambda_l2h',               0.003, 0.12),
                        ('gn_texture_residual', 'lambda_texture_protect',   0.001, 0.05),
                    ]
                    alpha = float(getattr(cfg, 'adaptive_alpha', 0.5))
                    beta  = float(getattr(cfg, 'adaptive_beta', 0.1))
                    low_band = base / 2.0
                    high_band = base * 2.0
                    # 追加新的高频替代项作为可调对象（在原 entries 基础上扩展）：
                    # - 频率加权 STFT / 高频倒谱 / 高频 patch 结构
                    entries.extend([
                        ('gn_hf_stft',       'lambda_hf_stft',       0.01,  0.15),
                        ('gn_ceps_hi',       'lambda_ceps_hi',       0.005, 0.10),
                        ('gn_mel_hp',        'lambda_mel_hp',        0.005, 0.10),
                        # 无声非静音高阶倒谱（虚线 F0 区域保护）
                        ('gn_ceps_hi_unv',   'lambda_ceps_hi_unv',   0.003, 0.08),
                        # 倒谱 manifold 自一致性
                        ('gn_ceps_manifold', 'lambda_feature_manifold', 0.02, 0.20),
                        # Mel 亮度/对比度/分频带锚点（避免亮度漂移与过度塌缩）
                        ('gn_mel_energy',    'lambda_mel_energy',    0.005, 0.06),
                        ('gn_mel_contrast',  'lambda_mel_contrast',  0.005, 0.06),
                        ('gn_mel_bandE',     'lambda_mel_bandE',     0.005, 0.06),
                        # 高频时间边缘 + 倾斜约束（边界摩擦/擦音高度）
                        ('gn_hf_time_edge',  'lambda_hf_time_edge',  0.003, 0.06),
                        ('gn_hf_tilt',       'lambda_hf_tilt',       0.003, 0.06),
                    ])
                    # Include teacher HF distillation into adaptive balancing when enabled
                    try:
                        tmode = str(getattr(cfg, 'teacher_hf_norm', 'bft_mean')).lower()
                        if float(getattr(cfg, 'lambda_teacher_hf', 0.0)) > 0.0:
                            if tmode == 'freq_only':
                                entries.append(('gn_distill_hf', 'lambda_teacher_hf', 0.001, 2.0))
                            else:  # bft_mean
                                entries.append(('gn_distill_hf', 'lambda_teacher_hf', 0.05, 2.0))
                    except Exception:
                        pass
                    for gk, lam_name, lo, hi in entries:
                        gi = float(grad_info.get(gk, 0.0))
                        if not np.isfinite(gi) or gi <= 0.0:
                            continue
                        cur = float(getattr(cfg, lam_name))
                        new = cur
                        if gi < low_band:
                            # 适度放大：按 (base/gi)^alpha 比例更新
                            scale = (base / (gi + 1e-12)) ** alpha
                            target = cur * scale
                            new = max(lo, min(hi, (1.0 - beta) * cur + beta * target))
                        elif gi > high_band:
                            # 适度缩小：按 (gi/base)^alpha 比例更新
                            scale = (gi / (base + 1e-12)) ** alpha
                            target = cur / scale
                            new = max(lo, min(hi, (1.0 - beta) * cur + beta * target))
                        if new != cur:
                            setattr(cfg, lam_name, float(new))
                            loss_dict[f'{lam_name}'] = float(new)
                except Exception:
                    pass

            # 将梯度survey结果注入打印（仅在开启grad_survey时）
            if bool(getattr(cfg, 'grad_survey', False)):
                try:
                    should_log_g = (global_step % 10 == 0) or (global_step % int(getattr(cfg, 'adaptive_every', 20)) == 0)
                    if should_log_g:
                        for k, v in grad_info.items():
                            if k.startswith('gn_') or k in ('g_ref', 'g_tex', 'g_l2h', 'g_band', 'g_mod'):
                                # 记录梯度尺度（保持原始精度，打印时用高精度/科学计数法）
                                loss_dict[k] = float(v)
                except Exception:
                    pass

            if global_step % 10 == 0:
                msg = f"[epoch {epoch} step {global_step}] total={total_loss.item():.4f}"
                for k, v in loss_dict.items():
                    # 对梯度survey相关的 key（gn_* 以及 g_ref/g_tex 等）使用更高精度/科学计数法，
                    # 便于观察 1e-6 级别的小梯度；其余保持原有4位小数格式。
                    if k.startswith('gn_') or k in ('g_ref', 'g_tex', 'g_l2h', 'g_band', 'g_mod'):
                        msg += f" {k}={v:.3e}"
                    else:
                        msg += f" {k}={v:.4f}"
                print(msg)

            # 简单的 vocoder IO 诊断：每隔 100 步打印一次，
            # 直接从 out 中拆分 fargan_feats_hat 的 ceps / F0 / VUV 三块，以及 audio_hat。
            # 为避免环境变量判断带来的混淆，这里不再依赖 DBG_* 开关。
            try:
                if global_step % 100 == 0:
                    fhat = out.get("fargan_feats_hat")
                    if isinstance(fhat, torch.Tensor) and fhat.dim() == 3 and fhat.size(-1) >= 20:
                        x = fhat.detach().to(torch.float32)
                        ceps_hat = x[..., :18]
                        f0_hat = x[..., 18:19]
                        vuv_hat = x[..., 19:20]

                        def _stat_simple(name: str, t: torch.Tensor) -> None:
                            if t.numel() == 0:
                                return
                            # 使用 reshape 而不是 view，以避免在非连续张量上触发 RuntimeError。
                            v = t.reshape(-1)
                            print(
                                f"[dbg_voc] {name}: min={v.min().item():+.4f} max={v.max().item():+.4f} "
                                f"mean={v.mean().item():+.4f} std={v.std().item():.4f}"
                            )

                        _stat_simple("ceps_hat_in", ceps_hat)
                        _stat_simple("f0_hat_in", f0_hat)
                        _stat_simple("vuv_hat_in", vuv_hat)

                    audio_hat = out.get("audio_hat")
                    if isinstance(audio_hat, torch.Tensor):
                        a = audio_hat.detach().to(torch.float32)
                        if a.numel() > 0:
                            # 同样使用 reshape 以兼容非连续张量（例如经过 permute/slice 的张量）。
                            v = a.reshape(-1)
                            print(
                                f"[dbg_voc] audio_hat: min={v.min().item():+.4f} max={v.max().item():+.4f} "
                                f"mean={v.mean().item():+.4f} std={v.std().item():.4f}"
                            )
            except Exception as e:
                print("[dbg_voc] exception:", repr(e))

            # 诊断打印：每个可视化周期输出特征域关键统计，便于定位“亮度漂移/voiced塌缩”
            try:
                if cfg.viz_every_steps and cfg.viz_every_steps > 0 and (global_step % cfg.viz_every_steps == 0):
                    # ceps 逐维偏差（均值）与 c0 偏差
                    ceps = out.get("ceps"); ceps_hat = out.get("ceps_hat")
                    if isinstance(ceps, torch.Tensor) and isinstance(ceps_hat, torch.Tensor):
                        diff = (ceps_hat - ceps).detach()
                        c0_bias = diff[..., 0].mean().item()
                        max_bias = diff.mean(dim=(0,1)).abs().max().item()
                        print(f"[diag] ceps_bias: c0_mean_diff={c0_bias:+.4f}, max_dim_mean_abs_diff={max_bias:.4f}")
                    # mel 能量均值差
                    mel = out.get("mel"); mel_hat = out.get("mel_hat")
                    if isinstance(mel, torch.Tensor) and isinstance(mel_hat, torch.Tensor):
                        mdiff = (mel_hat.mean(dim=(1,2)) - mel.mean(dim=(1,2))).mean().item()
                        print(f"[diag] mel_mean_log_energy_diff={mdiff:+.4f}")
                    # V/UV 比例 (基于 frame_corr 与 vuv_threshold)
                    fc = out.get("frame_corr"); fc_hat = out.get("frame_corr_hat")
                    th = float(getattr(cfg, 'vuv_threshold', 0.3))
                    if isinstance(fc, torch.Tensor) and isinstance(fc_hat, torch.Tensor):
                        v_t = (fc > th).float().mean().item()
                        v_h = (fc_hat > th).float().mean().item()
                        print(f"[diag] voiced_ratio target={v_t*100:.1f}% hat={v_h*100:.1f}% (thr={th})")

                        # 可选 F0/VUV 详细统计：由 DBG_F0=1 控制
                        if os.environ.get('DBG_F0', '0') == '1':
                            fc_t = fc.detach().view(-1)
                            fc_h = fc_hat.detach().view(-1)
                            print(
                                "[diag] frame_corr target: min={:.3f} max={:.3f} mean={:.3f} std={:.3f}".format(
                                    float(fc_t.min().item()), float(fc_t.max().item()),
                                    float(fc_t.mean().item()), float(fc_t.std().item()),
                                )
                            )
                            print(
                                "[diag] frame_corr hat   : min={:.3f} max={:.3f} mean={:.3f} std={:.3f}".format(
                                    float(fc_h.min().item()), float(fc_h.max().item()),
                                    float(fc_h.mean().item()), float(fc_h.std().item()),
                                )
                            )

                    # dnn_pitch Hz 统计
                    dp = out.get("dnn_pitch"); dp_hat = out.get("dnn_pitch_hat")
                    if isinstance(dp, torch.Tensor) and isinstance(dp_hat, torch.Tensor):
                        def to_hz(x: torch.Tensor) -> torch.Tensor:
                            # period = 256 / 2^(x+1.5)  (clamped to [32,255]);  f0_hz = 16000 / period
                            period = torch.clamp(256.0 / torch.pow(2.0, x + 1.5), 32.0, 255.0)
                            return 16000.0 / period
                        hz_t = to_hz(dp)
                        hz_h = to_hz(dp_hat)
                        print(f"[diag] dnn_pitch mean Hz: target={hz_t.mean().item():.2f}, hat={hz_h.mean().item():.2f}")

                        if os.environ.get('DBG_F0', '0') == '1':
                            ht = hz_t.detach().view(-1)
                            hh = hz_h.detach().view(-1)
                            print(
                                "[diag] F0 target (Hz): min={:.1f} max={:.1f} mean={:.1f} std={:.1f}".format(
                                    float(ht.min().item()), float(ht.max().item()),
                                    float(ht.mean().item()), float(ht.std().item()),
                                )
                            )
                            print(
                                "[diag] F0 hat    (Hz): min={:.1f} max={:.1f} mean={:.1f} std={:.1f}".format(
                                    float(hh.min().item()), float(hh.max().item()),
                                    float(hh.mean().item()), float(hh.std().item()),
                                )
                            )

                    # 额外：检查声码器输入的 20 维核心特征分布（ceps + F0 + VUV），
                    # 便于判断 F0 维度是否被压缩或被 ceps 能量“淹没”。
                    # 由 DBG_F0=1 或 DBG_VOCODER_IO=1 控制。
                    dbg_voc = (
                        os.environ.get('DBG_F0', '0') == '1'
                        or os.environ.get('DBG_VOCODER_IO', '0') == '1'
                    )
                    if dbg_voc:
                        try:
                            fhat = out.get("fargan_feats_hat")
                            if isinstance(fhat, torch.Tensor):
                                x = fhat.detach().to(torch.float32)
                                if x.dim() == 3 and x.size(-1) >= 20:
                                    ceps_hat = x[..., :18]
                                    f0_hat = x[..., 18:19]
                                    vuv_hat = x[..., 19:20]

                                    def _stat(name: str, t: torch.Tensor) -> None:
                                        if t.numel() == 0:
                                            return
                                        v = t.view(-1)
                                        print(
                                            f"[diag] {name}: min={v.min().item():+.4f} max={v.max().item():+.4f} "
                                            f"mean={v.mean().item():+.4f} std={v.std().item():.4f}"
                                        )

                                    _stat("ceps_hat_in", ceps_hat)
                                    _stat("f0_hat_in", f0_hat)
                                    _stat("vuv_hat_in", vuv_hat)
                        except Exception:
                            pass
            except Exception as _e:
                pass

            # 可视化与音频样本导出（周期性执行）
            if cfg.viz_every_steps and cfg.viz_every_steps > 0 and (global_step % cfg.viz_every_steps == 0):
                try:
                    # Skip audio-related visualization in content-only mode (no audio_hat available)
                    is_content_only = getattr(cfg, 'content_only', False)
                    if not is_content_only and "audio_hat" in out:
                        # 1) 常规前向路径的对比图（训练前向）
                        audio_real_b = out["audio"].detach().cpu()
                        audio_gen_b = out["audio_hat"].detach().cpu()
                        create_batch_comparison_plots(
                            audio_real_batch=audio_real_b,
                            audio_gen_batch=audio_gen_b,
                            save_dir=cfg.viz_dir,
                            step=global_step,
                            max_samples=cfg.viz_max_samples,
                            sr=16000,
                        )

                        # 2) 可选：bit-only eval，对同一 batch 的前若干条样本
                        # 使用 encode_hash_codec + decode_from_bits_offline 跑一遍
                        # 纯 bits→音频 的路径，并额外保存对比图。
                        try:
                            # 仅当配置开启且模型具备 hash 编解码接口时，执行 bit-only eval。
                            if bool(getattr(cfg, 'bit_only_eval', False)) \
                               and getattr(model, 'with_hash', False) \
                               and hasattr(model, 'encode_hash_codec') \
                               and hasattr(model, 'decode_from_bits_offline'):

                                # 取前 bit_only_eval_max_samples 条样本
                                max_samples = int(getattr(cfg, 'bit_only_eval_max_samples', 2))
                                B_eval = min(int(audio_real_b.size(0)), max_samples)
                                if B_eval > 0:
                                    audio_eval = audio_real_b[:B_eval].to(device)

                                    # 训练 batch 中的 FARGAN 特征：优先使用 batch['x']，
                                    # 若不存在则回退到 batch['features']。
                                    feats_src = None
                                    if isinstance(batch, dict):
                                        if "x" in batch:
                                            feats_src = batch["x"]
                                        elif "features" in batch:
                                            feats_src = batch["features"]
                                    if feats_src is None:
                                        raise KeyError("bit_only_eval requires batch['x'] or batch['features'] with 36-dim FARGAN features")

                                    feats_eval = feats_src[:B_eval].to(device)

                                    # 使用与训练一致的 ChannelSimulator 采样 CSI
                                    chan_eval = channel_sim
                                    bits_c, bits_f, bits_s, meta = model.encode_hash_codec(
                                        audio=audio_eval,
                                        fargan_feats=feats_eval,
                                        channel_sim=chan_eval,
                                        snr_min_db=float(getattr(cfg, 'snr_min_db', cfg.snr_min_db)),
                                        snr_max_db=float(getattr(cfg, 'snr_max_db', cfg.snr_max_db)),
                                        use_noisy_bits=True,
                                    )

                                    csi_vec = None
                                    if isinstance(meta, dict) and 'csi_vec' in meta:
                                        csi_val = meta['csi_vec']
                                        if isinstance(csi_val, torch.Tensor):
                                            csi_vec = csi_val.to(device)
                                        else:
                                            csi_vec = torch.from_numpy(csi_val).to(device)

                                    out_bits = model.decode_from_bits_offline(
                                        bits_content=bits_c,
                                        bits_f0=bits_f,
                                        bits_stats=bits_s,
                                        f0_T=int(meta.get('T', feats_eval.size(1))),
                                        target_len=int(audio_eval.size(1)),
                                        csi_vec=csi_vec,
                                        snr_db=None,
                                        content_hw=meta.get('hw', None),
                                    )

                                    if "audio_hat" in out_bits:
                                        audio_gen_bits = out_bits["audio_hat"].detach().cpu()
                                        # 单独子目录：bit_only 子路径
                                        viz_dir_bits = os.path.join(cfg.viz_dir, "bit_only")
                                        os.makedirs(viz_dir_bits, exist_ok=True)
                                        create_batch_comparison_plots(
                                            audio_real_batch=audio_real_b[:B_eval],
                                            audio_gen_batch=audio_gen_bits,
                                            save_dir=viz_dir_bits,
                                            step=global_step,
                                            max_samples=B_eval,
                                            sr=16000,
                                        )

                                        # Debug: 当 DBG_STAGE25=1 时，对比 teacher 解码
                                        # (forward_with_hash / decode_hash_codec) 与
                                        # 纯 bits→decode_from_bits_offline 在中间特征
                                        # （mel/ceps/F0/VUV）上的统计差异。
                                        try:
                                            if os.environ.get("DBG_STAGE25", "0") == "1":
                                                # teacher 端特征（当前 batch 的 out）
                                                mel_t = out.get("mel_hat_refined", out.get("mel_hat", None))
                                                ceps_t = out.get("ceps_hat", None)
                                                dp_t = out.get("dnn_pitch_hat", None)
                                                fc_t = out.get("frame_corr_hat", None)

                                                # bits 端特征（decode_from_bits_offline 输出）
                                                mel_b = out_bits.get("mel_hat_refined", out_bits.get("mel_hat", None))
                                                ceps_b = out_bits.get("ceps_hat", None)
                                                dp_b = out_bits.get("dnn_pitch_hat", None)
                                                fc_b = out_bits.get("frame_corr_hat", None)

                                                def _stat(name: str, t: Optional[torch.Tensor]) -> None:
                                                    if not isinstance(t, torch.Tensor):
                                                        print(f"[DBG_STAGE25] {name}: None")
                                                        return
                                                    x = t.detach().float()
                                                    if x.numel() == 0:
                                                        print(f"[DBG_STAGE25] {name}: empty")
                                                        return
                                                    x_flat = x.view(-1)
                                                    print(
                                                        f"[DBG_STAGE25] {name}: "
                                                        f"shape={tuple(x.shape)} "
                                                        f"min={x_flat.min().item():+.4f} "
                                                        f"max={x_flat.max().item():+.4f} "
                                                        f"mean={x_flat.mean().item():+.4f} "
                                                        f"std={x_flat.std().item():.4f}"
                                                    )

                                                print("[DBG_STAGE25] === Teacher decode features ===")
                                                _stat("mel_hat_refined_T", mel_t)
                                                _stat("ceps_hat_T", ceps_t)
                                                _stat("dnn_pitch_hat_T", dp_t)
                                                _stat("frame_corr_hat_T", fc_t)

                                                print("[DBG_STAGE25] === Bit-only decode features ===")
                                                _stat("mel_hat_refined_B", mel_b)
                                                _stat("ceps_hat_B", ceps_b)
                                                _stat("dnn_pitch_hat_B", dp_b)
                                                _stat("frame_corr_hat_B", fc_b)

                                                # 可选：将特征快照保存为 .npy，便于离线可视化。
                                                try:
                                                    import numpy as _np

                                                    snap_dir = os.path.join(viz_dir_bits, "debug_feats")
                                                    os.makedirs(snap_dir, exist_ok=True)
                                                    idx0 = 0
                                                    if isinstance(mel_t, torch.Tensor) and isinstance(mel_b, torch.Tensor):
                                                        _np.save(
                                                            os.path.join(snap_dir, f"mel_T_step{global_step:06d}.npy"),
                                                            mel_t[idx0].detach().cpu().numpy(),
                                                        )
                                                        _np.save(
                                                            os.path.join(snap_dir, f"mel_B_step{global_step:06d}.npy"),
                                                            mel_b[idx0].detach().cpu().numpy(),
                                                        )
                                                    if isinstance(ceps_t, torch.Tensor) and isinstance(ceps_b, torch.Tensor):
                                                        _np.save(
                                                            os.path.join(snap_dir, f"ceps_T_step{global_step:06d}.npy"),
                                                            ceps_t[idx0].detach().cpu().numpy(),
                                                        )
                                                        _np.save(
                                                            os.path.join(snap_dir, f"ceps_B_step{global_step:06d}.npy"),
                                                            ceps_b[idx0].detach().cpu().numpy(),
                                                        )
                                                except Exception as _snap_e:
                                                    print(f"[DBG_STAGE25] saving feature snapshots failed: {_snap_e}")
                                        except Exception as _dbg_e:
                                            print(f"[DBG_STAGE25] feature debug failed: {_dbg_e}")

                                        # 额外：在 bit-only eval 下打印静音掩膜与 RMS 统计，
                                        # 用真实音频的帧 RMS 定义静音掩膜，分别统计：
                                        #   - real / teacher(audio_gen_b) / bit-only(audio_gen_bits)
                                        #     在静音帧上的平均 RMS，用于定量对比底噪水平。
                                        try:
                                            import math as _math

                                            def _frame_rms_local(x: torch.Tensor, frame_len: int = 160, hop: int = 160) -> torch.Tensor:
                                                """10 ms 帧 RMS 计算，返回 [B,Tf]."""
                                                if x.dim() != 2:
                                                    x = x.view(x.size(0), -1)
                                                Bf, Lf = x.shape
                                                if Lf < frame_len:
                                                    pad = frame_len - Lf
                                                    x = torch.nn.functional.pad(x, (0, pad))
                                                    Lf = frame_len
                                                x_frames = x.unfold(dimension=1, size=frame_len, step=hop)  # [Bf,Tf,frame_len]
                                                rms = torch.sqrt(x_frames.pow(2).mean(dim=-1) + 1e-8)
                                                return rms

                                            real_eval = audio_real_b[:B_eval]
                                            teacher_eval = audio_gen_b[:B_eval]
                                            bits_eval = audio_gen_bits

                                            rms_real = _frame_rms_local(real_eval)  # [B,T]
                                            rms_teacher = _frame_rms_local(teacher_eval)
                                            rms_bits = _frame_rms_local(bits_eval)

                                            eps = 1e-8
                                            rms_max = rms_real.max(dim=1, keepdim=True).values.clamp_min(eps)
                                            rms_norm = rms_real / rms_max
                                            rms_db = 20.0 * torch.log10(rms_norm + eps)

                                            thr_db = float(getattr(cfg, "silence_rms_thr_db", -35.0))
                                            silence_mask = (rms_db <= thr_db)  # [B,T]

                                            # 对齐时间维长度
                                            T_use = int(min(rms_real.size(1), rms_teacher.size(1), rms_bits.size(1), silence_mask.size(1)))
                                            rms_real = rms_real[:, :T_use]
                                            rms_teacher = rms_teacher[:, :T_use]
                                            rms_bits = rms_bits[:, :T_use]
                                            silence_mask = silence_mask[:, :T_use]

                                            if silence_mask.any():
                                                real_sil = rms_real[silence_mask]
                                                teacher_sil = rms_teacher[silence_mask]
                                                bits_sil = rms_bits[silence_mask]

                                                sil_ratio = float(silence_mask.float().mean().item())
                                                print(
                                                    "[bit_only_eval] silence_ratio={:.3f}, "
                                                    "RMS_silence(real/teacher/bits)={:.6f}/{:.6f}/{:.6f}".format(
                                                        sil_ratio,
                                                        float(real_sil.mean().item()),
                                                        float(teacher_sil.mean().item()),
                                                        float(bits_sil.mean().item()),
                                                    )
                                                )
                                            else:
                                                print("[bit_only_eval] WARNING: no silent frames detected for RMS stats")
                                        except Exception as _rms_e:
                                            if os.environ.get('DBG_STAGE25', '0') == '1':
                                                print(f"[bit_only_eval] RMS stats failed: {_rms_e}")
                        except Exception as _bit_e:
                            if os.environ.get('DBG_STAGE25', '0') == '1':
                                print(f"[viz] bit-only eval failed at step {global_step}: {_bit_e}")

                        # 可选：绘制 F0 三轨对齐图，方便排查
                        # （dnn_pitch_hat → Hz, vocoder period → Hz, audio_hat F0）。
                        try:
                            if os.environ.get("DBG_F0_ALIGN", "0") == "1":
                                dp_hat = out.get("dnn_pitch_hat")
                                period_vocoder = out.get("period_vocoder")
                                audio_hat = out.get("audio_hat")
                                if (
                                    isinstance(dp_hat, torch.Tensor)
                                    and isinstance(period_vocoder, torch.Tensor)
                                    and isinstance(audio_hat, torch.Tensor)
                                ):
                                    f0_dir = os.path.join(cfg.viz_dir, "f0_align")
                                    os.makedirs(f0_dir, exist_ok=True)
                                    batch_size = min(
                                        int(audio_hat.size(0)), int(getattr(cfg, "viz_max_samples", 3))
                                    )
                                    for i in range(batch_size):
                                        save_path = os.path.join(
                                            f0_dir,
                                            f"f0_align_step_{global_step:06d}_sample_{i:02d}.png",
                                        )
                                        create_f0_alignment_plot(
                                            audio_gen=audio_hat[i],
                                            dnn_pitch_hat=dp_hat[i],
                                            period_vocoder=period_vocoder[i],
                                            save_path=save_path,
                                            sr=16000,
                                            hop_length=160,
                                            title=(
                                                f"F0 Alignment - Step {global_step} - "
                                                f"Sample {i}"
                                            ),
                                        )
                        except Exception as _f0e:
                            print(f"[viz] F0 alignment plot failed: {_f0e}")
                        # 可选：保存音频样本（与图配套）
                        save_comparison_audio_samples(
                            audio_real_batch=out["audio"].detach().cpu(),
                            audio_gen_batch=out["audio_hat"].detach().cpu(),
                            save_dir=cfg.viz_dir,
                            step=global_step,
                            max_samples=cfg.viz_max_samples,
                            sr=16000,
                        )
                    # 额外：保存内容分支 BFCC（32 维 Bark log 能量）可视化，便于观察 JSCC 端恢复的谱层次
                    try:
                        mel = out.get("mel")
                        mel_hat = out.get("mel_hat")
                        if isinstance(mel, torch.Tensor) and isinstance(mel_hat, torch.Tensor):
                            bfcc_dir = os.path.join(cfg.viz_dir, "bfcc")
                            os.makedirs(bfcc_dir, exist_ok=True)
                            # 取第一个样本做可视化
                            mel_real = mel[0].detach().cpu().numpy()  # [T,32]
                            mel_gen = mel_hat[0].detach().cpu().numpy()
                            # 转为 [freq, time]
                            mel_real_T = mel_real.T
                            mel_gen_T = mel_gen.T
                            n_frames = mel_real_T.shape[1]
                            duration_sec = n_frames * cfg.sequence_length * 0.0  # 占位，下面重新按 hop 估算
                            # 使用 10ms 帧移近似（与 WaveToBFCC 设置一致）
                            hop = 160
                            sr_vis = 16000
                            duration_sec = mel_real_T.shape[1] * hop / float(sr_vis)

                            def _save_bfcc_img(arr_T: np.ndarray, path: str, title: str) -> None:
                                """Save BFCC image; tolerate extra channel dims.

                                期望输入形状约为 [F,T]。若传入 [F,T,C] 或
                                更高维，则在最后一维取平均以构造 2D 图像，
                                避免 Matplotlib 对非常规通道数报错。
                                """
                                if arr_T.ndim > 2:
                                    arr_T = arr_T.mean(axis=-1)
                                elif arr_T.ndim < 2:
                                    # 退化情形：拉成单行
                                    arr_T = np.reshape(arr_T, (1, -1))

                                vmin = float(np.percentile(arr_T, 1))
                                vmax = float(np.percentile(arr_T, 99))
                                plt.figure(figsize=(8, 3))
                                extent = [0.0, duration_sec, 0, arr_T.shape[0]]
                                img = plt.imshow(
                                    arr_T,
                                    origin="lower",
                                    aspect="auto",
                                    interpolation="nearest",
                                    cmap="magma",
                                    vmin=vmin,
                                    vmax=vmax,
                                    extent=extent,
                                )
                                plt.colorbar(img, label="log10 energy")
                                plt.xlabel("Time (s)")
                                plt.ylabel("Bark band index (0-31)")
                                plt.title(title)
                                plt.tight_layout()
                                plt.savefig(path, dpi=150)
                                plt.close()

                            base_name = f"step{global_step:06d}_sample0"
                            real_path = os.path.join(bfcc_dir, base_name + "_bfcc_gt.png")
                            gen_path = os.path.join(bfcc_dir, base_name + "_bfcc_jscc.png")
                            _save_bfcc_img(mel_real_T, real_path, "Content BFCC (GT)")
                            _save_bfcc_img(mel_gen_T, gen_path, "Content BFCC (JSCC decoded)")

                        # 额外：在 content_only 预训练时，周期性输出 ceps 各维分布对比图，
                        # 便于监控 ceps_hat 是否逐渐对齐 FARGAN 目标分布。
                        try:
                            if getattr(cfg, 'content_only', False) and os.environ.get('DBG_CEPS_HIST', '0') == '1':
                                ceps = out.get("ceps")
                                ceps_hat = out.get("ceps_hat")
                                if isinstance(ceps, torch.Tensor) and isinstance(ceps_hat, torch.Tensor):
                                    ceps_dir = os.path.join(cfg.viz_dir, "ceps_hist")
                                    os.makedirs(ceps_dir, exist_ok=True)
                                    save_path = os.path.join(
                                        ceps_dir,
                                        f"ceps_hist_step_{global_step:06d}.png",
                                    )
                                    create_ceps_hist_comparison(ceps, ceps_hat, save_path)
                        except Exception as _ceps_e:
                            if os.environ.get('DBG_STAGE25', '0') == '1':
                                print(f"[viz] ceps histogram failed: {_ceps_e}")

                        # 额外：内容分支 token 多通道彩色可视化（去除旧的能量热力图，改为 RGB token 图）
                        tok_map = out.get("tokens_map")
                        tok_hat_map = out.get("tokens_hat_map")
                        if isinstance(tok_map, torch.Tensor) and isinstance(tok_hat_map, torch.Tensor):
                            tok_dir = os.path.join(cfg.viz_dir, "tokens")
                            os.makedirs(tok_dir, exist_ok=True)

                            pre = tok_map[0].detach().cpu().numpy()      # [C,H,W]
                            post = tok_hat_map[0].detach().cpu().numpy()  # [C,H,W]

                            def _tokens_to_rgb(arr_chw: np.ndarray) -> np.ndarray:
                                """将 [C,H,W] token 映射为 [H,W,3] RGB 图像，用前 3 个通道做伪彩色。

                                为避免 LayerNorm 带来的均匀能量误导，这里对每个通道分别做
                                1/99 分位归一化，再拼成 RGB。
                                """

                                if arr_chw.ndim != 3:
                                    raise ValueError(f"tokens_to_rgb expects [C,H,W], got shape={arr_chw.shape}")
                                C_t, H_t, W_t = arr_chw.shape
                                if C_t == 0:
                                    return np.zeros((H_t, W_t, 3), dtype=np.float32)

                                c_sel = min(C_t, 3)
                                x = arr_chw[:c_sel].astype(np.float32)

                                # 每个通道单独做 1/99 分位裁剪和归一化
                                vmin = np.percentile(x, 1, axis=(1, 2), keepdims=True)
                                vmax = np.percentile(x, 99, axis=(1, 2), keepdims=True)
                                denom = np.maximum(vmax - vmin, 1e-6)
                                x_norm = (x - vmin) / denom
                                x_norm = np.clip(x_norm, 0.0, 1.0)

                                # 通道数不足 3 时，用最后一个通道重复填充
                                if c_sel < 3:
                                    pad = np.repeat(x_norm[-1:], 3 - c_sel, axis=0)
                                    x_norm = np.concatenate([x_norm, pad], axis=0)

                                rgb = np.transpose(x_norm[:3], (1, 2, 0))  # [H,W,3]
                                return rgb

                            def _save_tok_rgb(arr_chw: np.ndarray, path: str, title: str) -> None:
                                rgb = _tokens_to_rgb(arr_chw)
                                plt.figure(figsize=(4, 4))
                                plt.imshow(rgb, origin="lower", aspect="auto")
                                plt.axis("off")
                                plt.title(title)
                                plt.tight_layout()
                                plt.savefig(path, dpi=150)
                                plt.close()

                            base_tok = f"step{global_step:06d}_sample0"
                            pre_path = os.path.join(tok_dir, base_tok + "_tokens_pre_channel.png")
                            post_path = os.path.join(tok_dir, base_tok + "_tokens_post_channel.png")
                            _save_tok_rgb(pre, pre_path, "Content tokens (pre-channel, RGB)")
                            _save_tok_rgb(post, post_path, "Content tokens (post-channel, RGB)")
                    except Exception as _viz_bfcc_err:
                        print(f"[Viz] Failed to export BFCC/token images at step {global_step}: {_viz_bfcc_err}")
                except Exception as e:
                    print(f"[Viz] Failed to generate comparison plots/audio at step {global_step}: {e}")

            global_step += 1



def parse_args() -> SupportConfig:
    import argparse


    parser = argparse.ArgumentParser(description="DBP-JSCC training support")
    parser.add_argument("--data_root", type=str,
                        default="./data")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--sequence_length", type=int, default=200)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_amp", action="store_true", default=True,
                        help="Enable automatic mixed precision for faster training (default: True)")
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable automatic mixed precision")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (None = no fix)")
    parser.add_argument("--snr_min_db", type=float, default=-5.0)
    parser.add_argument("--snr_max_db", type=float, default=15.0)
    # Extra losses and schedule
    parser.add_argument("--lambda_mel", type=float, default=0.5,
                        help="Weight for mel MS-SSIM loss")
    parser.add_argument("--lambda_mel_l1", type=float, default=0.0,
                        help="Extra mel L1 weight (added alongside MS-SSIM)")
    parser.add_argument("--lambda_mel_delta", type=float, default=0.0,
                        help="Weight for mel Δ/ΔΔ (delta/delta-delta) loss (0 to disable)")
    parser.add_argument("--lambda_delta", type=float, default=0.0,
                        help="Weight for Δ/ΔΔ ceps loss (0 to disable)")
    parser.add_argument("--lambda_wave", type=float, default=1.0,
                        help="Weight for STFT spectral convergence (overrides default)")
    parser.add_argument("--lambda_wave_mag", type=float, default=0.0,
                        help="Weight for MR-STFT magnitude L1 (added alongside spectral convergence)")
    parser.add_argument("--lambda_ceps", type=float, default=0.5,
                        help="Weight for ceps L1 loss (overrides default)")
    parser.add_argument("--lambda_ceps_weighted", type=float, default=0.0,
                        help="Weight for cepstral weighted loss (higher weight on low-order coeffs)")
    parser.add_argument("--lambda_ceps_map_gt", type=float, default=0.0,
                        help="Weight for BFCC->ceps mapping loss using GT mel and FARGAN ceps")
    parser.add_argument("--lambda_f0", type=float, default=0.5,
                        help="Weight for f0 MSE loss (overrides default)")
    parser.add_argument("--lambda_f0_base", type=float, default=0.0,
                        help="Weight for SR base-branch f0 MSE loss (using only first k JSCC dims)")
    parser.add_argument("--lambda_f0_smooth", type=float, default=0.0,
                        help="Weight for F0 smoothness loss (2nd-order diff, prevents jitter)")
    parser.add_argument("--lambda_f0_base_smooth", type=float, default=0.0,
                        help="Weight for SR base-branch F0 smoothness loss")
    parser.add_argument("--lambda_vuv", type=float, default=0.5,
                        help="Weight for vuv MSE loss (overrides default)")
    parser.add_argument("--lambda_f0_std", type=float, default=0.0,
                        help="Weight for F0 std matching loss in Hz domain (voiced frames only)")
    parser.add_argument("--lambda_hash_recon", type=float, default=0.1,
                        help="Weight for hash reconstruction loss")
    parser.add_argument("--lambda_hash_reg", type=float, default=0.1,
                        help="Weight for hash regularization loss")
    parser.add_argument("--lambda_vq", type=float, default=0.0,
                        help="Global weight for RVQ VQ loss (legacy, when quantizer_type='rvq')")
    parser.add_argument("--lambda_vq_c", type=float, default=0.0,
                        help="Weight for content-branch RVQ VQ loss (overrides lambda_vq when >0)")
    parser.add_argument("--lambda_vq_f", type=float, default=0.0,
                        help="Weight for F0-branch RVQ VQ loss (overrides lambda_vq when >0)")
    parser.add_argument("--quantizer_type", type=str, default="hash", choices=["hash", "rvq"],
                        help="Quantizer type for content/F0 bottlenecks: 'hash' (default) or 'rvq'")
    parser.add_argument("--lambda_f0_entropy", type=float, default=0.0,
                        help="Weight for F0-specific bit/index entropy lower-bound loss (0 to disable)")
    parser.add_argument("--f0_entropy_target_frac", type=float, default=0.5,
                        help="Target entropy fraction for F0 bits/codebooks (0..1; 0.5≈Kf/2 bits per token)")
    parser.add_argument("--lambda_c_entropy", type=float, default=0.0,
                        help="Weight for content-branch bit/index entropy lower-bound loss (0 to disable)")
    parser.add_argument("--content_entropy_target_frac", type=float, default=0.5,
                        help="Target entropy fraction for content bits/codebooks (0..1; 0.5≈Kc/2 bits per token)")
    parser.add_argument("--lambda_bit_balance_c", type=float, default=0.0,
                        help="Weight for content bit-balance loss (encourage P(bit=1)≈0.5 for each bit)")
    # SSL 语音内容一致性（HuBERT / Wav2Vec2 / WavLM 等）：
    parser.add_argument("--lambda_ssl", type=float, default=0.0,
                        help="Weight for SSL speech-content consistency loss (0 to disable)")
    parser.add_argument("--ssl_model_name", type=str, default=None,
                        help="HuggingFace model name or local path for SSL encoder (e.g., 'facebook/hubert-base-ls960')")
    parser.add_argument("--ssl_layers", type=str, default=None,
                        help="Comma-separated SSL hidden layer indices (e.g., '6,9,12'; empty for auto)")
    parser.add_argument("--ssl_warmup_steps", type=int, default=0,
                        help="Warmup steps for SSL loss (linear ramp from 0 to lambda_ssl)")
    # Eval/diagnostic flags
    parser.add_argument("--bit_only_eval", action="store_true",
                        help="At viz steps, also run bit-only encode+decode and save extra comparison plots")
    parser.add_argument("--bit_only_eval_max_samples", type=int, default=2,
                        help="Max samples per batch for bit-only eval visualization")
    parser.add_argument("--disable_ceps_c0_calib", action="store_true",
                        help="Disable GT ceps-based c0 calibration during training (energy_calib uses bits_stats only)")
    # JSCC+FSK offline evaluation options
    parser.add_argument("--jscc_fsk_eval", action="store_true",
                        help="After each checkpoint save, run pcm_segment_infer_jscc_fsk.py for JSCC+FSK metrics")
    parser.add_argument("--jscc_fsk_pcm_path", type=str, default=None,
                        help="Path to long PCM file for JSCC+FSK eval (e.g., merged_high_energy.pcm)")
    parser.add_argument("--jscc_fsk_output_root", type=str, default=None,
                        help="Root output directory for pcm_segment_infer_jscc_fsk.py")
    parser.add_argument("--jscc_fsk_pcm_infer_script", type=str, default=None,
                        help="Path to pcm_segment_infer_jscc_fsk.py (in Fargan_sim)")
    parser.add_argument("--jscc_fsk_noise_csv", type=str, default=None,
                        help="Path to noise_voltage_*.csv for FSK simulation")
    parser.add_argument("--jscc_fsk_sample_rate", type=int, default=16000,
                        help="Sample rate passed to pcm_segment_infer_jscc_fsk.py (default: 16000)")
    parser.add_argument("--jscc_fsk_pcm_dtype", type=str, default="int16",
                        help="PCM dtype passed to pcm_segment_infer_jscc_fsk.py (default: int16)")
    parser.add_argument("--jscc_fsk_segment_sec", type=float, default=4.0,
                        help="Segment length in seconds for pcm_segment_infer_jscc_fsk.py (default: 4.0)")
    parser.add_argument("--jscc_fsk_num_segments", type=int, default=50,
                        help="Number of segments for pcm_segment_infer_jscc_fsk.py (default: 50)")
    parser.add_argument("--jscc_fsk_seed", type=int, default=123,
                        help="Seed used by pcm_segment_infer_jscc_fsk.py (default: 123)")
    parser.add_argument("--jscc_fsk_snr_db", type=float, default=3.0,
                        help="SNR (dB) for pcm_segment_infer_jscc_fsk.py (default: 3.0)")
    parser.add_argument("--jscc_fsk_metrics_csv", type=str, default=None,
                        help="CSV file to append JSCC+FSK metrics; default ckpt_dir/jscc_fsk_metrics.csv")
    # wandb logging options
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable wandb logging (if wandb is installed)")
    parser.add_argument("--wandb_project", type=str, default="DBP-JSCC",
                        help="wandb project name (default: DBP-JSCC)")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Optional wandb run name")
    parser.add_argument("--wandb_log_freq", type=int, default=10,
                        help="Log losses to wandb every N steps")
    # Audio-level F0 and harmonic alignment losses
    # 已移除波形域 F0 损失（f0_audio / f0_slope）；不再提供对应 CLI。
    parser.add_argument("--lambda_harmonic", type=float, default=None,
                        help="Weight for harmonic alignment loss (overrides default)")
    parser.add_argument("--harmonics_max", type=int, default=None,
                        help="Number of harmonics to align in harmonic loss")
    parser.add_argument("--harmonic_bandwidth_hz", type=float, default=None,
                        help="Bandwidth (Hz) of Gaussian around each harmonic center")
    parser.add_argument("--vuv_threshold", type=float, default=None,
                        help="Threshold on frame_corr to consider voiced frames")
    parser.add_argument("--use_two_stage", action="store_true",
                        help="Enable two-stage schedule (weak STFT + low LR in stage 1)")
    parser.add_argument("--stage1_steps", type=int, default=5000,
                        help="Steps for stage 1 before restoring base LR/weights")
    parser.add_argument("--with_hash", action="store_true",
                        help="Enable hash bottleneck for content branch")
    parser.add_argument("--content_only", action="store_true",
                        help="Content-only mode: train only Bark/BFCC reconstruction, skip F0/VUV/ceps/vocoder")
    parser.add_argument("--f0_only", action="store_true",
                        help="F0-only warmup: freeze content/JSCC, train only F0/VUV branch + vocoder")
    parser.add_argument("--content_cnn_baseline", action="store_true",
                        help="Use simple BFCC CNN JSCC baseline for content branch (replaces VMamba in content-only mode)")
    parser.add_argument("--content_cnn_latent_channels", type=int, default=1,
                        help="Latent channels for BFCC CNN baseline (1≈1/128 DOF, 2≈1/64)")
    parser.add_argument("--hash_bits_content", type=int, default=16,
                        help="Bits per content token for hash bottleneck")
    parser.add_argument("--hash_bits_f0", type=int, default=None,
                        help="Bits per F0/VUV token for hash bottleneck (default: max(4, hash_bits_content//2))")
    parser.add_argument("--rvq_nq_content", type=int, default=2,
                        help="Number of RVQ codebooks for content bottleneck (when quantizer_type='rvq')")
    parser.add_argument("--rvq_nq_f0", type=int, default=None,
                        help="Number of RVQ codebooks for F0 bottleneck (default: same as rvq_nq_content)")
    parser.add_argument("--rvq_beta", type=float, default=0.25,
                        help="Commitment weight beta for RVQ bottlenecks")
    parser.add_argument(
        "--content_time_downsample",
        type=int,
        default=1,
        help="Downsample factor on time axis before content vSSM encoder (2 => 50Hz->25Hz token-rate at 100Hz mel)",
    )
    parser.add_argument(
        "--vm_channel_adaptive",
        type=str,
        default="no",
        choices=["no", "ca", "ssm"],
        help=(
            "VMamba content-branch SNR fusion: "
            "'no' (disable), 'ca' (per-stage bias CA), or 'ssm' (SSM-level SNR modulation)"
        ),
    )
    parser.add_argument(
        "--vm_lightweight_config",
        type=str,
        default="all_native",
        choices=["all_native", "progressive", "all_lightweight"],
        help="CSI gating pattern inside VMamba blocks (all_native/progressive/all_lightweight)",
    )
    # 架构细粒度参数改由 --arch 预设控制；以下 CLI 隐藏
    parser.add_argument("--eq_fading", action="store_true",
                        help="Enable simple Rayleigh fading equalization on the decoder side")
    parser.add_argument("--ckpt_dir", type=str,
                        default="./outputs/checkpoints",
                        help="Directory to save training checkpoints")
    parser.add_argument("--save_every_steps", type=int, default=500,
                        help="Save checkpoint every N training steps")
    # FARGAN / vocoder settings
    parser.add_argument("--vocoder_ckpt", type=str, default=None,
                        help="Path to pretrained vocoder checkpoint (.pth)")
    parser.add_argument("--vocoder_eval_every_steps", type=int, default=0,
                        help="If >0, periodically evaluate vocoder-only baseline STFT using GT features")
    parser.add_argument("--freeze_vocoder_all", action="store_true",
                        help="Freeze FARGAN vocoder for the entire training (no unfreeze)")
    parser.add_argument("--reload_vocoder_after_resume", action="store_true",
                        help="After --resume, re-load the vocoder ckpt into the student vocoder core (default: keep vocoder from checkpoint)")
    parser.add_argument("--freeze_codec_all", action="store_true",
                        help="Freeze all codec (encoder/decoder) weights and train only hash bottlenecks")
    parser.add_argument("--train_only_hash", action="store_true",
                        help="Alias for --freeze_codec_all; only HashBottleneck enc/dec remain trainable")
    parser.add_argument("--freeze_content_jscc", action="store_true",
                        help="Freeze BFCC content JSCC (wave_to_mel + content_vmamba + hash_content[_stats])")
    parser.add_argument("--lambda_teacher_hf", type=float, default=0.0,
                        help="Weight for HF distillation from teacher (GT features vocoder path)")
    # Teacher HF distillation uses stable defaults (log1p + bft_mean + auto-balance)
    # Only --lambda_teacher_hf is exposed; no extra CLI needed
    # Anti-flatten knobs (FARGAN-style)
    parser.add_argument("--stft_preset", type=str, default="aether",
                        choices=["aether", "fargan"], help="STFT window set: aether(default) or fargan (long windows)")
    # 高频替代项（可调；在 hash-only 场景默认关闭，需要时显式打开）
    parser.add_argument("--lambda_hf_stft", type=float, default=0.0,
                        help="Weight for HF-emphasized STFT difference")
    # hf_start_hz/hf_power 固定为实现内部默认，不再暴露 CLI
    parser.add_argument("--lambda_ceps_hi", type=float, default=0.03,
                        help="Weight for high-order cepstra supervision (e.g., c10..)")
    parser.add_argument("--lambda_ceps_hi_unv", type=float, default=0.0,
                        help="Weight for high-order cepstra supervision on unvoiced/non-silent frames")
    parser.add_argument("--ceps_hi_start", type=int, default=10,
                        help="Start index for high-order cepstra supervision")
    parser.add_argument("--lambda_mel_hp", type=float, default=0.0,
                        help="Weight for mel Laplacian (frequency 2nd-derivative) match on high bins")
    parser.add_argument("--mel_hp_low_bins", type=int, default=16,
                        help="Low-bin cutoff for mel HP loss (only bins >= cutoff are used)")
    parser.add_argument("--lambda_mel_valley", type=float, default=0.0,
                        help="Weight for mel valley clamp loss (prevents filling spectral valleys in non-silence frames)")
    parser.add_argument("--lambda_hf_tilt", type=float, default=0.0,
                        help="Weight for boundary HF tilt loss (high vs low band energy on boundary/unvoiced frames)")
    parser.add_argument("--hf_tilt_split_bin", type=int, default=16,
                        help="Split bin between low/high bands for HF tilt (0..split-1 vs split..Fm-1)")
    parser.add_argument("--hf_tilt_extra_push", type=float, default=0.0,
                        help="Extra hinge margin (log-mel) to push HF energy above LF on boundaries; 0.5≈5dB, 1.0≈10dB")
    parser.add_argument("--lambda_hf_time_edge", type=float, default=0.0,
                        help="Weight for HF time-edge loss (Δt on high mel bins)")
    parser.add_argument("--hf_time_edge_start", type=int, default=32,
                        help="Start mel bin for HF time-edge loss")
    parser.add_argument("--hf_time_edge_ref_thr", type=float, default=0.03,
                        help="Reference Δt threshold in log-mel units (1.0≈10dB)")
    parser.add_argument("--hf_time_edge_boundary_boost", type=float, default=2.0,
                        help="Extra weight on VUV boundaries for HF time-edge")
    parser.add_argument("--hf_time_edge_weight_clip", type=float, default=5.0,
                        help="Clip normalized edge weights")
    # Teacher-forcing schedule for F0/VUV (period + frame_corr gate)
    parser.add_argument("--tf_start_step", type=int, default=0,
                        help="Global step to start annealing teacher-forcing (None disables)")
    parser.add_argument("--tf_end_step", type=int, default=1000,
                        help="Global step to end annealing teacher-forcing (w from 1->0)")
    parser.add_argument("--lambda_mel_texture", type=float, default=0.0,
                        help="Weight for mel texture (time/freq gradient consistency) on bright voiced regions")
    # 低→高细化与频带先验 + 条件 flow 高频建模
    parser.add_argument("--with_l2h", action="store_true", help="Enable low->high mel refinement head")
    parser.add_argument("--lambda_l2h", type=float, default=0.00,
                        help="Weight for high-band envelope supervision of L2H refiner")
    parser.add_argument("--lambda_l2h_direct", type=float, default=0.0,
                        help="Weight for direct L1 on refined HF mel (no temporal smoothing)")
    parser.add_argument("--lambda_l2h_resid", type=float, default=0.00,
                        help="Weight for L2H high-bin residual improvement loss (harm+noise)")
    parser.add_argument("--lambda_l2h_decor", type=float, default=0.00,
                        help="Weight for l2h decor")
    parser.add_argument("--l2h_improve_margin", type=float, default=0.0,
                        help="Margin in log-mel for L2H HF improvement over baseline (0=only prevent degradation)")
    parser.add_argument("--l2h_low_bins", type=int, default=10, help="Low-bin split for L2H head")
    # L2H 穿透损失：让优化目标对齐到 vocoder 输入 (18-band/ceps)
    parser.add_argument("--lambda_l2h_band18", type=float, default=0.0,
                        help="Weight for L2H pass-through loss in 18-band energy domain")
    parser.add_argument("--lambda_l2h_ceps", type=float, default=0.0,
                        help="Weight for L2H pass-through loss in 18-dim cepstral domain")
    parser.add_argument("--l2h_band18_hi_start", type=int, default=8,
                        help="Start bin for HF region in 18-band pass-through loss (default 8, i.e. bin 8-17)")
    parser.add_argument("--use_l2h_flow", action="store_true", help="Enable conditional flow modeling for GT mel high band (NLL regularizer)")
    parser.add_argument("--l2h_flow_hidden", type=int, default=128, help="Hidden size of conditional HF flow encoder")
    parser.add_argument("--l2h_flow_n_flows", type=int, default=4, help="Number of affine coupling flow layers for HF generator")
    parser.add_argument("--lambda_l2h_flow_nll", type=float, default=0.0, help="Weight for HF flow negative log-likelihood on GT mel high band")
    # 旧版频带先验相关 CLI 隐藏（实现保留，默认关闭）
    parser.add_argument("--freeze_band_agg", action="store_true",
                        help="Freeze 32->18 band aggregation weights to prevent low-band collapse")
    parser.add_argument("--energy_calib_alpha", type=float, default=0.8,
                        help="Strength of mean energy calibration (0..1) for mel/ceps c0")
    parser.add_argument("--l2h_warmup_steps", type=int, default=400,
                        help="Warmup steps for L2H (no fusion to vocoder during warmup)")
    parser.add_argument("--l2h_blend_steps", type=int, default=800,
                        help="Blend-in steps for L2H (linearly ramp fusion from 0->1)")
    parser.add_argument("--l2h_schedule_mode", type=str, default="abs",
                        choices=["abs", "resume_rel"],
                        help=(
                            "Schedule mode for L2H blend: 'abs' uses absolute global_step (resume-safe), "
                            "'resume_rel' restarts warmup after each resume. Can also be overridden by "
                            "env L2H_SCHEDULE_MODE."
                        ))
    parser.add_argument("--deco_l2h", action="store_true",
                        help="Use DeCo-style conditional L2H head instead of residual L2H")
    parser.add_argument("--deco_l2h_hidden", type=int, default=64,
                        help="Hidden size of DeCo L2H semantic encoder")
    parser.add_argument("--deco_l2h_blocks", type=int, default=3,
                        help="Number of AdaLN blocks in DeCo L2H head")
    # HF 侧通道：将高频残差特征直接传给 FARGAN
    parser.add_argument("--with_hf_sideband", action="store_true",
                        help="Enable HF sideband: pass HF residual features directly to FARGAN")
    parser.add_argument("--hf_sideband_dim", type=int, default=6,
                        help="Dimension of HF sideband features (4-8 typically sufficient)")
    parser.add_argument("--hf_sideband_type", type=str, default="learnable",
                        choices=["learnable", "dct", "linear"],
                        help="HF sideband encoder type: learnable MLP, fixed DCT, or simple linear")
    parser.add_argument("--hf2ceps_dim", type=int, default=8,
                        help="Dimension of HF→ceps correction head (high-order cepstral bins)")
    parser.add_argument("--hf2ceps_scale", type=float, default=0.5,
                        help="Global scale for HF→ceps correction (0..1)")
    # （精简）移除 FARGAN 风格附加项的 CLI：lambda_fargan_sc/signal/continuity/pitch_consistency/subframe_align
    parser.add_argument("--lambda_mel_energy", type=float, default=0.0,
                        help="Weight for global mel energy/brightness anchor |mean(logmel_hat)-mean(logmel)|")
    parser.add_argument("--lambda_mel_energy_t", type=float, default=0.0,
                        help="Weight for per-frame mel energy curve anchor (non-silence frames)")
    parser.add_argument("--lambda_mel_contrast", type=float, default=0.0,
                        help="Weight for per-frame mel contrast (freq-std) anchor")
    parser.add_argument("--lambda_mel_bandE", type=float, default=0.0,
                        help="Weight for 3-band (low/mid/high) mel energy anchor")
    parser.add_argument("--mel_loss_crop_time", type=int, default=0,
                        help="Crop this many frames from start/end when computing mel/BFCC losses (0 = no crop)")
    parser.add_argument("--mel_loss_crop_freq", type=int, default=0,
                        help="Crop this many Bark bands from top/bottom when computing mel/BFCC losses (0 = no crop)")
    # BFCC（32-Bark log 能量图）图像域能量/纹理约束（仿照 BFCC CNN baseline）
    parser.add_argument("--lambda_energy_t", type=float, default=0.0,
                        help="Weight for per-frame BFCC energy |E_hat(t)-E(t)| (Bark log-energy)")
    parser.add_argument("--lambda_energy_f", type=float, default=0.0,
                        help="Weight for per-band BFCC energy |E_hat(f)-E(f)| (Bark log-energy)")
    parser.add_argument("--lambda_tex_t", type=float, default=0.0,
                        help="Weight for BFCC time-axis gradient consistency |∂_t BFCC_hat-∂_t BFCC|")
    parser.add_argument("--lambda_freq_aware_mel", type=float, default=0.0,
                        help="Weight for DCT-based frequency-aware mel loss (emphasize low frequencies)")
    parser.add_argument("--jpeg_quality_factor", type=int, default=85,
                        help="JPEG quality factor (1-100) for freq-aware loss weighting")
    parser.add_argument("--lambda_f0_peak", type=float, default=0.0,
                        help="Weight for F0 peakiness (mean normalized entropy on voiced frames)")
    parser.add_argument("--lambda_f0_slope", type=float, default=0.0,
                        help="Weight for ΔF0 slope consistency (in cents)")
    # Mel band prior (low→high guidance)
    # 旧 mel 高频先验/细化相关 CLI 已移除
    # 简洁高频替代项（不暴露 CLI；使用 SupportConfig 默认值）
    # Gradient survey (log last-layer grad norms per loss)
    parser.add_argument('--grad_survey', action='store_true', help='Log last-layer grad norms of each loss term w.r.t. its domain anchor')
    # Adaptive HF weighting (match last-layer gradient scales)
    parser.add_argument('--adaptive_hf', action='store_true', help='Enable adaptive weighting for HF losses (texture/l2h/band/mod)')
    parser.add_argument('--adaptive_every', type=int, default=20, help='Steps between adaptive weight updates')
    parser.add_argument('--adaptive_alpha', type=float, default=0.5, help='Exponent for ratio scaling (0.25~0.75)')
    parser.add_argument('--adaptive_beta', type=float, default=0.1, help='EMA smoothing for lambda updates')
    # crepe-guided F0 envelope/presence losses
    parser.add_argument("--lambda_f0_env", type=float, default=0.0,
                        help="Weight for crepe/YIN envelope hinge in cents")
    parser.add_argument("--f0_env_margin_cents", type=float, default=80.0,
                        help="Base tolerance margin in cents for envelope hinge")
    parser.add_argument("--f0_env_alpha", type=float, default=0.5,
                        help="Extra margin factor scaled by crepe-vs-fallback disagreement (in cents)")
    parser.add_argument("--f0_env_window", type=int, default=5,
                        help="Median smoothing window for envelope center (odd) in frames")
    parser.add_argument("--f0_env_k_sigma", type=float, default=2.0,
                        help="Gaussian band gate: keep frames within k*std around local mean (core only)")
    parser.add_argument("--lambda_f0_tv", type=float, default=0.0,
                        help="Small second-order TV on cent_pred within core")
    parser.add_argument("--f0_tv_delta_cents", type=float, default=40.0,
                        help="Robust TV threshold (cents) for normalized hinge on |Δ²cent|")
    # wavelet differential constraint
    parser.add_argument("--lambda_f0_wavelet", type=float, default=0.0,
                        help="Weight for wavelet differential loss between pred and smoothed ref (cents)")
    parser.add_argument("--f0_wav_levels", type=int, default=3,
                        help="Number of Haar DWT levels for F0 wavelet loss")
    parser.add_argument("--f0_wav_alphas", type=str, default=None,
                        help="Comma-separated weights per wavelet level, e.g., 1.0,0.5,0.25")
    parser.add_argument("--f0_wav_clip_cents", type=float, default=120.0,
                        help="Clipping threshold in cents for wavelet error before normalization")
    parser.add_argument("--lambda_f0_bias", type=float, default=0.0,
                        help="Tiny penalty on region-mean F0 error (cents) to avoid constant offset drift")
    parser.add_argument("--f0_cond_attn_warmup_steps", type=int, default=0,
                        help="Warmup steps for content-conditioned F0 decoder attention (0=disabled)")
    parser.add_argument("--f0_cond_attn_max_alpha", type=float, default=1.0,
                        help="Max alpha for F0 decoder content-attention (scaled by internal gate)")
    # (presence/vuv_crepe removed)
    parser.add_argument("--f0_estimator", type=str, default="auto", choices=["auto","crepe","pyin"],
                        help="Estimator used inside loss extraction (visualizer remains independent)")
    parser.add_argument("--f0_estimator_model", type=str, default="tiny", choices=["tiny","full"],
                        help="torchcrepe model variant used for loss extraction")
    # Silence refinement: optional HF-variance gate for silence mask
    parser.add_argument("--silence_use_hf_var", action="store_true",
                        help="Use high-frequency variance in addition to energy/RMS to classify silence frames")
    parser.add_argument("--silence_hf_var_thr", type=float, default=0.02,
                        help="Variance threshold on HF Bark log-energy used when --silence_use_hf_var is enabled")
    # Train-only-hash mode
    # 已废弃：train_only_hash（训练循环未使用）
    # 预设：架构与 F0 正则器
    # 默认 'auto' 不覆盖原有模型构型，防止与旧 checkpoint 形状不一致
    parser.add_argument("--arch", type=str, default="auto", choices=["auto","small","base","large"],
                        help="Architecture preset for VMamba/JSCC")
    parser.add_argument("--f0_regularizer", type=str, default="none", choices=["none","light","strong"],
                        help="Preset for F0 regularization bundle (env/wavelet/peak)")
    # HF adversarial CLI（4–8kHz STFT 判别器，仅在 Stage2.5 中启用）
    parser.add_argument("--hf_adv_disc_lr", type=float, default=1e-4,
                        help="Learning rate for HF adversarial discriminator")
    parser.add_argument("--hf_adv_roi_low_hz", type=int, default=4000,
                        help="Lower frequency bound (Hz) for HF adversarial STFT ROI")
    parser.add_argument("--hf_adv_roi_high_hz", type=int, default=8000,
                        help="Upper frequency bound (Hz) for HF adversarial STFT ROI")

    # OSCE BFCC-GAN（基于 OSCE fd_discriminator 的波形判别器，对 audio_hat 做 LSGAN+FM）
    parser.add_argument("--bfcc_gan", action="store_true",
                        help="Enable OSCE FD-based adversarial loss on audio_hat")
    parser.add_argument("--bfcc_gan_lambda", type=float, default=1.0,
                        help="Generator adversarial loss weight for OSCE BFCC-GAN (LSGAN)")
    parser.add_argument("--bfcc_gan_fmap_weight", type=float, default=1.0,
                        help="Feature-matching loss weight for OSCE BFCC-GAN")

    # Pitch-harmonic contrast loss
    # High-frequency texture protection loss (replaces PHC)
    parser.add_argument("--lambda_texture_protect", type=float, default=0.0,
                        help="Weight for high-frequency texture protection loss (replaces PHC)")
    parser.add_argument("--texture_hf_start", type=int, default=30,
                        help="Starting mel bin for high-frequency region (default: 40 ≈ 4kHz)")
    parser.add_argument("--texture_grad_weight", type=float, default=0.5,
                        help="Weight for gradient term in texture loss")
    parser.add_argument("--texture_var_weight", type=float, default=0.3,
                        help="Weight for variance term in texture loss")
    parser.add_argument("--texture_contrast_weight", type=float, default=0.4,
                        help="Weight for frequency contrast term in texture loss")
    # Feature manifold self-consistency (ceps_from(audio_hat) ≈ ceps_hat)
    parser.add_argument("--lambda_feature_manifold", type=float, default=0.0,
                        help="Weight for feature self-consistency loss in ceps space")
    parser.add_argument("--texture_eps", type=float, default=1e-4,
                        help="Numerical epsilon for texture protection loss")
    # HiFi-GAN style adversarial regularization (raw waveform MPD + MSD)
    parser.add_argument("--lambda_hifi_adv", type=float, default=0.0,
                        help="Weight for HiFi-GAN style adversarial loss (MPD+MSD, LSGAN)")
    parser.add_argument("--lambda_hifi_fm", type=float, default=0.0,
                        help="Weight for HiFi-GAN style feature-matching loss from MPD/MSD feature maps")
    parser.add_argument("--hifi_adv_warmup_steps", type=int, default=0,
                        help="Warmup steps before enabling MPD/MSD discriminator updates")
    parser.add_argument("--hifi_disc_lr", type=float, default=1e-4,
                        help="Learning rate for HiFi-GAN MPD/MSD discriminators")
    parser.add_argument("--hifi_adv_crop_len", type=int, default=16000,
                        help="Crop length in samples for HiFi-GAN MPD/MSD (0=use full clip)")
    # HF adversarial regularization (4–8kHz STFT discriminator)
    parser.add_argument("--lambda_hf_adv", type=float, default=0.0,
                        help="Weight for HF adversarial loss on 4–8kHz STFT (LSGAN)")
    parser.add_argument("--lambda_hf_fm", type=float, default=0.0,
                        help="Weight for HF feature-matching loss from discriminator features")
    parser.add_argument("--hf_adv_warmup_steps", type=int, default=10000,
                        help="Warmup steps before enabling HF discriminator updates")
    # Mel-domain HF adversarial (high mel bins)
    parser.add_argument("--lambda_hf_mel_adv", type=float, default=0.0,
                        help="Weight for HF adversarial loss on mel high-frequency bins")
    parser.add_argument("--lambda_hf_mel_fm", type=float, default=0.0,
                        help="Weight for HF feature-matching loss on mel high-frequency bins")
    parser.add_argument("--hf_mel_adv_warmup_steps", type=int, default=5000,
                        help="Warmup steps before enabling mel HF discriminator updates")
    parser.add_argument("--hf_mel_low_bins", type=int, default=10,
                        help="Low mel bin index from which mel HF adversarial region starts")
    # F0 pattern preservation loss
    parser.add_argument("--lambda_f0_pattern", type=float, default=0.0,
                        help="Weight for F0 pattern preservation loss (preserves unvoiced F0 variations)")
    parser.add_argument("--lambda_f0_center", type=float, default=0.0,
                        help="Weight for F0 center continuity loss (voiced segments 10%-90% core)")
    parser.add_argument("--f0_pattern_synergy_weight", type=float, default=0.3,
                        help="Weight for F0-texture synergy component in pattern loss")
    # F0 presence loss (forces network to predict voiced segments)
    parser.add_argument("--lambda_f0_presence", type=float, default=0.0,
                        help="Weight for F0 presence loss (forces network to predict voiced segments)")
    parser.add_argument("--f0_presence_gamma", type=float, default=2.0,
                        help="Focal loss gamma parameter for F0 presence loss")
    # Keep legacy PHC parameter for backward compatibility
    parser.add_argument("--lambda_pitch_harm", type=float, default=0.0,
                        help="DEPRECATED: Use --lambda_texture_protect instead")
    # VUV extra losses: silence suppression + global ratio + BCE
    parser.add_argument("--lambda_vuv_sil", type=float, default=0.0,
                        help="Extra VUV loss on silent frames (penalize high frame_corr_hat when GT unvoiced)")
    parser.add_argument("--lambda_vuv_ratio", type=float, default=0.0,
                        help="Loss on global voiced frame ratio (hat vs target)")
    parser.add_argument("--lambda_vuv_bce", type=float, default=0.0,
                        help="BCE loss on voiced/unvoiced prediction from frame_corr_hat")
    parser.add_argument("--lambda_vuv_base", type=float, default=0.0,
                        help="Weight for SR base-branch VUV/frame_corr reconstruction loss")
    # Hash 正则调度：早期减弱、后期恢复到 lambda_hash_reg
    parser.add_argument("--hash_reg_warmup_steps", type=int, default=20000,
                        help="Warmup steps for hash regularization; 0 disables scheduling")
    parser.add_argument("--hash_reg_start", type=float, default=0.02,
                        help="Initial lambda for hash regularization during warmup")
    # Silence shaping (wave/mel energy in silent frames)
    parser.add_argument("--lambda_silence_wave", type=float, default=0.0,
                        help="Weight for wave-domain silence energy loss (RMS on silent frames)")
    parser.add_argument("--lambda_silence_mel", type=float, default=0.0,
                        help="Weight for mel-domain silence high-frequency loss (currently unused)")
    parser.add_argument("--lambda_bit_only_silence", type=float, default=0.0,
                        help="Weight for bit-only path silence loss (decode_from_bits_offline RMS on silent frames)")
    parser.add_argument("--lambda_bit_only_distill", type=float, default=0.0,
                        help="Weight for bit-only distillation loss (decode_from_bits_offline vs decode_hash_codec MR-STFT)")
    parser.add_argument("--silence_energy_thr_db", type=float, default=-35.0,
                        help="Relative dB threshold on HF log-mel energy below which frames are treated as silence")
    parser.add_argument("--silence_dilate_frames", type=int, default=0,
                        help="Dilate non-silence mask by +/- N frames to protect voiced/unvoiced boundaries")
    parser.add_argument("--silence_hf_low_bins", type=int, default=16,
                        help="Low mel bin index from which HF energy is used for silence detection")
    # 可视化导出
    parser.add_argument("--viz_dir", type=str, default="./outputs/visualizations",
                        help="Directory to save mel/F0 comparison plots and audio")
    parser.add_argument("--viz_every_steps", type=int, default=1000,
                        help="Export visualization every N training steps (0 to disable)")
    parser.add_argument("--viz_max_samples", type=int, default=2,
                        help="Max samples per batch to visualize")
    # Base output dir (convenience): if set, overrides ckpt_dir and viz_dir
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Base output directory; will write checkpoints to <out_dir>/checkpoints and visuals to <out_dir>/viz")
        # 新增：resume 功能
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")


    args = parser.parse_args()

    def _parse_int_list(val: Optional[str]) -> Optional[List[int]]:
        if val is None:
            return None
        parts = [p.strip() for p in val.split(",")]
        ints = [int(p) for p in parts if p]
        return ints or None

    def _parse_float_list(val: Optional[str]) -> Optional[List[float]]:
        if val is None:
            return None
        parts = [p.strip() for p in val.split(",")]
        floats = []
        for p in parts:
            try:
                floats.append(float(p))
            except Exception:
                pass
        return floats or None

    cfg = SupportConfig(
        data_root=args.data_root,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        num_epochs=args.num_epochs,
        lr=args.lr,
        device=args.device,
        use_amp=args.use_amp and not args.no_amp,  # --no_amp overrides --use_amp
        seed=args.seed,
        snr_min_db=args.snr_min_db,
        snr_max_db=args.snr_max_db,
        lambda_mel=args.lambda_mel,
        lambda_mel_l1=args.lambda_mel_l1,
        lambda_mel_delta=float(getattr(args, 'lambda_mel_delta', 0.0)),
        lambda_delta=args.lambda_delta,
        lambda_wave=args.lambda_wave,
        lambda_wave_mag=args.lambda_wave_mag,
        lambda_ceps=args.lambda_ceps,
        lambda_ceps_hi=float(args.lambda_ceps_hi),
        lambda_ceps_hi_unv=float(getattr(args, 'lambda_ceps_hi_unv', 0.0)),
        lambda_ceps_weighted=float(getattr(args, 'lambda_ceps_weighted', 0.0)),
        lambda_ceps_map_gt=float(getattr(args, 'lambda_ceps_map_gt', 0.0)),
        lambda_f0=args.lambda_f0,
        lambda_f0_base=float(getattr(args, 'lambda_f0_base', 0.0)),
        lambda_f0_base_smooth=float(getattr(args, 'lambda_f0_base_smooth', 0.0)),
        lambda_vuv=args.lambda_vuv,
        lambda_vuv_base=float(getattr(args, 'lambda_vuv_base', 0.0)),
        lambda_f0_smooth=float(getattr(args, 'lambda_f0_smooth', 0.3)),
        lambda_hash_recon=args.lambda_hash_recon,
        lambda_hash_reg=args.lambda_hash_reg,
        lambda_vq=float(getattr(args, 'lambda_vq', 0.0)),
        lambda_vq_c=float(getattr(args, 'lambda_vq_c', 0.0)),
        lambda_vq_f=float(getattr(args, 'lambda_vq_f', 0.0)),
        hash_reg_warmup_steps=int(getattr(args, 'hash_reg_warmup_steps', 20000)),
        hash_reg_start=float(getattr(args, 'hash_reg_start', 0.02)),
        lambda_ssl=float(getattr(args, 'lambda_ssl', 0.0)),
        ssl_model_name=getattr(args, 'ssl_model_name', None),
        ssl_layers=_parse_int_list(getattr(args, 'ssl_layers', None)),
        ssl_warmup_steps=int(getattr(args, 'ssl_warmup_steps', 0)),
        lambda_silence_wave=float(getattr(args, 'lambda_silence_wave', 0.0)),
        lambda_silence_mel=float(getattr(args, 'lambda_silence_mel', 0.0)),
        lambda_bit_only_silence=float(getattr(args, 'lambda_bit_only_silence', 0.0)),
        lambda_bit_only_distill=float(getattr(args, 'lambda_bit_only_distill', 0.0)),
        silence_energy_thr_db=float(getattr(args, 'silence_energy_thr_db', -35.0)),
        silence_dilate_frames=int(getattr(args, 'silence_dilate_frames', 0)),
        silence_hf_low_bins=int(getattr(args, 'silence_hf_low_bins', 16)),
        silence_use_hf_var=bool(getattr(args, 'silence_use_hf_var', False)),
        silence_hf_var_thr=float(getattr(args, 'silence_hf_var_thr', 0.02)),
        use_two_stage=args.use_two_stage,
        stage1_steps=args.stage1_steps,
        with_hash=args.with_hash,
        content_only=getattr(args, 'content_only', False),
        f0_only=bool(getattr(args, 'f0_only', False)),
        content_cnn_baseline=bool(getattr(args, 'content_cnn_baseline', False)),
        content_cnn_latent_channels=int(getattr(args, 'content_cnn_latent_channels', 1)),
        hash_bits_content=args.hash_bits_content,
        hash_bits_f0=getattr(args, 'hash_bits_f0', None),
        quantizer_type=str(getattr(args, 'quantizer_type', 'hash')),
        rvq_nq_content=int(getattr(args, 'rvq_nq_content', 2)),
        rvq_nq_f0=getattr(args, 'rvq_nq_f0', None),
        rvq_beta=float(getattr(args, 'rvq_beta', 0.25)),
        content_time_downsample=int(getattr(args, 'content_time_downsample', 1)),
        vm_channel_adaptive=str(getattr(args, 'vm_channel_adaptive', 'no')),
        vm_lightweight_config=str(getattr(args, 'vm_lightweight_config', 'all_native')),
        # VMamba 细粒度参数由 --arch 预设设置（见下）
        eq_fading=args.eq_fading,
        viz_dir=args.viz_dir,
        viz_every_steps=args.viz_every_steps,
        viz_max_samples=args.viz_max_samples,
        ckpt_dir=args.ckpt_dir,                    
        save_every_steps=args.save_every_steps,    
        resume=args.resume,
        vocoder_ckpt=args.vocoder_ckpt,
        vocoder_eval_every_steps=int(args.vocoder_eval_every_steps),
        freeze_vocoder_all=bool(args.freeze_vocoder_all),
        freeze_codec_all=bool(args.freeze_codec_all or getattr(args, 'train_only_hash', False)),
        freeze_content_jscc=bool(getattr(args, 'freeze_content_jscc', False)),
        stft_preset=str(args.stft_preset),
        # HF alternatives (explicitly configurable via CLI)
        lambda_hf_stft=float(args.lambda_hf_stft),
        ceps_hi_start=int(args.ceps_hi_start),
        lambda_mel_hp=float(args.lambda_mel_hp),
        mel_hp_low_bins=int(args.mel_hp_low_bins),
        lambda_hf_tilt=float(getattr(args, 'lambda_hf_tilt', 0.0)),
        hf_tilt_split_bin=int(getattr(args, 'hf_tilt_split_bin', 16)),
        hf_tilt_extra_push=float(getattr(args, 'hf_tilt_extra_push', 0.0)),
        lambda_hf_time_edge=float(getattr(args, 'lambda_hf_time_edge', 0.0)),
        hf_time_edge_start=int(getattr(args, 'hf_time_edge_start', 32)),
        hf_time_edge_ref_thr=float(getattr(args, 'hf_time_edge_ref_thr', 0.03)),
        hf_time_edge_boundary_boost=float(getattr(args, 'hf_time_edge_boundary_boost', 2.0)),
        hf_time_edge_weight_clip=float(getattr(args, 'hf_time_edge_weight_clip', 5.0)),
        lambda_mel_texture=float(args.lambda_mel_texture),
        mel_loss_crop_time=int(getattr(args, 'mel_loss_crop_time', 0)),
        mel_loss_crop_freq=int(getattr(args, 'mel_loss_crop_freq', 0)),
        # L2H + band/mod priors (all default off)
        with_l2h=bool(args.with_l2h),
        # lambda_l2h=float(args.lambda_l2h),
        # lambda_l2h_direct=float(getattr(args, 'lambda_l2h_direct', 0.0)),
        l2h_low_bins=int(args.l2h_low_bins),
        # L2H 穿透损失 + 高频 residual 改善
        lambda_l2h_band18=float(getattr(args, 'lambda_l2h_band18', 0.0)),
        lambda_l2h_ceps=float(getattr(args, 'lambda_l2h_ceps', 0.0)),
        lambda_l2h_resid=float(getattr(args, 'lambda_l2h_resid', 0.0)),
        lambda_l2h_decor=float(getattr(args, 'lambda_l2h_decor', 0.0)),
        l2h_improve_margin=float(getattr(args, 'l2h_improve_margin', 0.0)),
        l2h_band18_hi_start=int(getattr(args, 'l2h_band18_hi_start', 8)),
        freeze_band_agg=bool(args.freeze_band_agg),
        energy_calib_alpha=float(args.energy_calib_alpha),
        l2h_warmup_steps=int(args.l2h_warmup_steps),
        l2h_blend_steps=int(args.l2h_blend_steps),
        l2h_schedule_mode=str(getattr(args, 'l2h_schedule_mode', 'abs')),
        # HF 侧通道
        with_hf_sideband=bool(getattr(args, 'with_hf_sideband', False)),
        hf_sideband_dim=int(getattr(args, 'hf_sideband_dim', 6)),
        hf_sideband_type=str(getattr(args, 'hf_sideband_type', 'learnable')),
        hf2ceps_dim=int(getattr(args, 'hf2ceps_dim', 8)),
        hf2ceps_scale=float(getattr(args, 'hf2ceps_scale', 0.5)),
        # （精简）不再从 CLI 注入 FARGAN 附加项的权重，保持为默认 0
        lambda_mel_energy=float(args.lambda_mel_energy),
        lambda_mel_energy_t=float(getattr(args, 'lambda_mel_energy_t', 0.0)),
        lambda_mel_contrast=float(getattr(args, 'lambda_mel_contrast', 0.0)),
        lambda_mel_bandE=float(getattr(args, 'lambda_mel_bandE', 0.0)),
        lambda_freq_aware_mel=float(getattr(args, 'lambda_freq_aware_mel', 0.0)),
        jpeg_quality_factor=int(getattr(args, 'jpeg_quality_factor', 85)),
        lambda_mel_valley=float(getattr(args, 'lambda_mel_valley', 0.0)),
        lambda_energy_t=float(getattr(args, 'lambda_energy_t', 0.0)),
        lambda_energy_f=float(getattr(args, 'lambda_energy_f', 0.0)),
        lambda_tex_t=float(getattr(args, 'lambda_tex_t', 0.0)),
        lambda_f0_peak=float(args.lambda_f0_peak),
        lambda_f0_slope=float(args.lambda_f0_slope),
        lambda_f0_std=float(getattr(args, 'lambda_f0_std', 0.0)),
        lambda_f0_env=float(args.lambda_f0_env),
        f0_env_margin_cents=float(args.f0_env_margin_cents),
        f0_env_alpha=float(args.f0_env_alpha),
        f0_env_window=int(args.f0_env_window),
        
        f0_env_k_sigma=float(args.f0_env_k_sigma),
        lambda_f0_tv=float(args.lambda_f0_tv),
        f0_tv_delta_cents=float(args.f0_tv_delta_cents),
        lambda_f0_wavelet=float(args.lambda_f0_wavelet),
        f0_wav_levels=int(args.f0_wav_levels),
        f0_wav_alphas=_parse_float_list(args.f0_wav_alphas),
        f0_wav_clip_cents=float(args.f0_wav_clip_cents),
        lambda_f0_bias=float(args.lambda_f0_bias),
        f0_cond_attn_warmup_steps=int(getattr(args, 'f0_cond_attn_warmup_steps', 0)),
        f0_cond_attn_max_alpha=float(getattr(args, 'f0_cond_attn_max_alpha', 1.0)),
        lambda_f0_entropy=float(getattr(args, 'lambda_f0_entropy', 0.0)),
        f0_entropy_target_frac=float(getattr(args, 'f0_entropy_target_frac', 0.5)),
        lambda_c_entropy=float(getattr(args, 'lambda_c_entropy', 0.0)),
        content_entropy_target_frac=float(getattr(args, 'content_entropy_target_frac', 0.5)),
        lambda_bit_balance_c=float(getattr(args, 'lambda_bit_balance_c', 0.0)),
        lambda_feature_manifold=float(getattr(args, 'lambda_feature_manifold', 0.0)),
        grad_survey=bool(args.grad_survey),
        adaptive_hf=bool(args.adaptive_hf),
        adaptive_every=int(args.adaptive_every),
        adaptive_alpha=float(args.adaptive_alpha),
        adaptive_beta=float(args.adaptive_beta),
        tf_start_step=getattr(args, 'tf_start_step', None),
        tf_end_step=getattr(args, 'tf_end_step', None),
        f0_estimator=str(args.f0_estimator),
        f0_estimator_model=str(args.f0_estimator_model),
        # train_only_hash removed
        
        # Texture protection parameters (replaces PHC)
        lambda_texture_protect=float(args.lambda_texture_protect),
        texture_hf_start=int(args.texture_hf_start),
        texture_grad_weight=float(args.texture_grad_weight),
        texture_var_weight=float(args.texture_var_weight),
        texture_contrast_weight=float(args.texture_contrast_weight),
        texture_eps=float(args.texture_eps),
        # OSCE BFCC-GAN
        bfcc_gan=bool(getattr(args, 'bfcc_gan', False)),
        bfcc_gan_lambda=float(getattr(args, 'bfcc_gan_lambda', 1.0)),
        bfcc_gan_fmap_weight=float(getattr(args, 'bfcc_gan_fmap_weight', 1.0)),
        # HiFi-GAN style adversarial (raw waveform MPD + MSD)
        lambda_hifi_adv=float(getattr(args, 'lambda_hifi_adv', 0.0)),
        lambda_hifi_fm=float(getattr(args, 'lambda_hifi_fm', 0.0)),
        hifi_adv_warmup_steps=int(getattr(args, 'hifi_adv_warmup_steps', 0)),
        hifi_disc_lr=float(getattr(args, 'hifi_disc_lr', 1e-4)),
        hifi_adv_crop_len=int(getattr(args, 'hifi_adv_crop_len', 16000)),
        # HF adversarial (audio STFT 4–8kHz)
        lambda_hf_adv=float(getattr(args, 'lambda_hf_adv', 0.0)),
        lambda_hf_fm=float(getattr(args, 'lambda_hf_fm', 0.0)),
        hf_adv_warmup_steps=int(getattr(args, 'hf_adv_warmup_steps', 10000)),
        hf_adv_roi_low_hz=int(getattr(args, 'hf_adv_roi_low_hz', 4000)),
        hf_adv_roi_high_hz=int(getattr(args, 'hf_adv_roi_high_hz', 8000)),
        lambda_hf_mel_adv=float(getattr(args, 'lambda_hf_mel_adv', 0.0)),
        lambda_hf_mel_fm=float(getattr(args, 'lambda_hf_mel_fm', 0.0)),
        hf_mel_adv_warmup_steps=int(getattr(args, 'hf_mel_adv_warmup_steps', 5000)),
        hf_mel_low_bins=int(getattr(args, 'hf_mel_low_bins', 10)),
        # F0 pattern preservation parameters
        lambda_f0_pattern=float(args.lambda_f0_pattern),
        lambda_f0_center=float(args.lambda_f0_center),
        f0_pattern_synergy_weight=float(args.f0_pattern_synergy_weight),
        lambda_teacher_hf=float(args.lambda_teacher_hf),
        # F0 presence parameters
        lambda_f0_presence=float(args.lambda_f0_presence),
        f0_presence_gamma=float(args.f0_presence_gamma),
        # Keep legacy PHC for backward compatibility
        lambda_pitch_harm=float(args.lambda_pitch_harm),
        lambda_vuv_sil=float(getattr(args, 'lambda_vuv_sil', 0.0)),
        lambda_vuv_ratio=float(getattr(args, 'lambda_vuv_ratio', 0.0)),
        lambda_vuv_bce=float(getattr(args, 'lambda_vuv_bce', 0.0)),
        reload_vocoder_after_resume=bool(getattr(args, 'reload_vocoder_after_resume', False)),
        use_l2h_flow=bool(getattr(args, 'use_l2h_flow', False)),
        l2h_flow_hidden=int(getattr(args, 'l2h_flow_hidden', 128)),
        l2h_flow_n_flows=int(getattr(args, 'l2h_flow_n_flows', 4)),
        lambda_l2h_flow_nll=float(getattr(args, 'lambda_l2h_flow_nll', 0.0)),
        deco_l2h=bool(getattr(args, 'deco_l2h', False)),
        deco_l2h_hidden=int(getattr(args, 'deco_l2h_hidden', 64)),
        deco_l2h_blocks=int(getattr(args, 'deco_l2h_blocks', 3)),
        # Eval/diagnostic switches
        bit_only_eval=bool(getattr(args, 'bit_only_eval', False)),
        bit_only_eval_max_samples=int(getattr(args, 'bit_only_eval_max_samples', 2)),
        disable_ceps_c0_calib=bool(getattr(args, 'disable_ceps_c0_calib', False)),
        # wandb
        use_wandb=bool(getattr(args, 'use_wandb', False)),
        wandb_project=str(getattr(args, 'wandb_project', 'DBP-JSCC')),
        wandb_run_name=getattr(args, 'wandb_run_name', None),
        wandb_log_freq=int(getattr(args, 'wandb_log_freq', 10)),
        # JSCC+FSK offline evaluation
        jscc_fsk_eval=bool(getattr(args, 'jscc_fsk_eval', False)),
        jscc_fsk_pcm_path=getattr(args, 'jscc_fsk_pcm_path', None),
        jscc_fsk_output_root=getattr(args, 'jscc_fsk_output_root', None),
        jscc_fsk_pcm_infer_script=getattr(args, 'jscc_fsk_pcm_infer_script', None),
        jscc_fsk_noise_csv=getattr(args, 'jscc_fsk_noise_csv', None),
        jscc_fsk_sample_rate=int(getattr(args, 'jscc_fsk_sample_rate', 16000)),
        jscc_fsk_pcm_dtype=str(getattr(args, 'jscc_fsk_pcm_dtype', 'int16')),
        jscc_fsk_segment_sec=float(getattr(args, 'jscc_fsk_segment_sec', 4.0)),
        jscc_fsk_num_segments=int(getattr(args, 'jscc_fsk_num_segments', 50)),
        jscc_fsk_seed=int(getattr(args, 'jscc_fsk_seed', 123)),
        jscc_fsk_snr_db=float(getattr(args, 'jscc_fsk_snr_db', 3.0)),
        jscc_fsk_metrics_csv=getattr(args, 'jscc_fsk_metrics_csv', None),
    )

    # If --out_dir is provided, override ckpt_dir and viz_dir for convenience
    if args.out_dir:
        base = os.path.abspath(args.out_dir)
        ckpt_dir = os.path.join(base, "checkpoints")
        viz_dir = os.path.join(base, "viz")
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(viz_dir, exist_ok=True)
        cfg.ckpt_dir = ckpt_dir
        cfg.viz_dir = viz_dir

    # ---- Apply presets: architecture + F0 regularizer ----
    arch = getattr(args, 'arch', 'auto')
    try:
        if arch == 'small':
            cfg.vm_channels = [32, 48, 64, 96]
            cfg.vm_depths = [2, 2, 2, 2]
            cfg.d_s_content = 8
            cfg.freq_downsample_stages = 1
        elif arch == 'large':
            cfg.vm_channels = [64, 96, 128, 160]
            cfg.vm_depths = [3, 3, 3, 3]
            cfg.d_s_content = 8
            cfg.freq_downsample_stages = 1
        elif arch == 'base':
            cfg.vm_channels = [48, 64, 96, 128]
            cfg.vm_depths = [2, 2, 2, 2]
            cfg.d_s_content = 8
            cfg.freq_downsample_stages = 1
        else:
            # auto: 不做覆盖，保持现有 cfg 或模型默认，以兼容旧 checkpoint
            pass
    except Exception:
        pass

    reg = getattr(args, 'f0_regularizer', 'none')
    try:
        # New behavior:
        # - 'light'/'strong' apply presets.
        # - 'none' means NO override (respect CLI-provided values).
        if reg == 'light':
            cfg.lambda_f0_env = 0.00001; cfg.f0_env_window = 3; cfg.f0_env_margin_cents = 30.0; cfg.f0_env_alpha = 1.0
            cfg.lambda_f0_wavelet = 0.015; cfg.f0_wav_levels = 3; cfg.f0_wav_alphas = [1.0, 0.35, 0.15]
            cfg.lambda_f0_tv = 0.0
            cfg.lambda_f0_peak = 0.00
            cfg.lambda_f0_bias = 0.001
        elif reg == 'strong':
            cfg.lambda_f0_env = 0.05; cfg.f0_env_window = 3; cfg.f0_env_margin_cents = 80.0; cfg.f0_env_alpha = 1.2
            cfg.lambda_f0_wavelet = 0.010; cfg.f0_wav_levels = 3; cfg.f0_wav_alphas = [1.0, 0.55, 0.25]
            cfg.lambda_f0_tv = 0.0
            cfg.lambda_f0_peak = 0.00
            cfg.lambda_f0_bias = 0.005
        else:
            # reg == 'none' (or unknown): keep current cfg values as set by CLI/defaults
            pass
    except Exception:
        pass

    # Apply CLI overrides for optional audio-F0/harmonic settings when provided
    # 波形域 F0 损失已移除；不再从 CLI 覆盖相关权重。
    if args.lambda_harmonic is not None:
        cfg.lambda_harmonic = float(args.lambda_harmonic)
    if args.harmonics_max is not None:
        cfg.harmonics_max = int(args.harmonics_max)
    if args.harmonic_bandwidth_hz is not None:
        cfg.harmonic_bandwidth_hz = float(args.harmonic_bandwidth_hz)
    if args.vuv_threshold is not None:
        cfg.vuv_threshold = float(args.vuv_threshold)

    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    run_training_support(cfg)
