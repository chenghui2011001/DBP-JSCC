#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频可视化工具：生成F0和Mel谱图对比图
"""

import torch
import torch.nn.functional as F
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import os
from typing import Tuple, Optional
import torchaudio
import warnings

warnings.filterwarnings('ignore')

try:
    import torchcrepe  # Optional but preferred F0 estimator
    _HAS_TORCHCREPE = True
except Exception:
    _HAS_TORCHCREPE = False


def _fallback_pitch_from_centroid(audio_np: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    """基于谱质心的粗略 F0 估计（裁剪到 [80, 400] Hz），仅用于可视化回退。"""
    try:
        S = np.abs(librosa.stft(audio_np, n_fft=hop_length * 4, hop_length=hop_length))
        cent = librosa.feature.spectral_centroid(S=S, sr=sr).squeeze(0)
        cent = np.clip(cent, 80.0, 400.0).astype(np.float32)
        return cent
    except Exception:
        n_frames = max(1, int(np.ceil(len(audio_np) / float(hop_length))))
        return np.zeros((n_frames,), dtype=np.float32)


def extract_f0(audio: torch.Tensor, sr: int = 16000, hop_length: int = 160) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """提取F0（基频），返回 (f0_hz, voiced_mask, f0_fallback_hz, fallback_mask)。

    主估计器：torchcrepe（若可用）；否则回退到 librosa.pyin。
    始终返回一个“辅助回退轨迹”（librosa.yin 或谱质心），用于可视化参考。
    """
    audio_np = audio.detach().cpu().numpy()

    # 能量掩码（避免静音段）
    rms = librosa.feature.rms(y=audio_np, frame_length=hop_length * 4, hop_length=hop_length).squeeze(0)
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-7) / (np.max(rms) + 1e-12) + 1e-12)
    energy_mask = rms_db > -35.0

    # ---- 主估计：torchcrepe（若可用） ----
    if _HAS_TORCHCREPE:
        try:
            device = audio.device if audio.is_cuda else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
            x = audio.detach().unsqueeze(0).to(device)  # [1, L]
            with torch.no_grad():
                # torchcrepe API expects positional sample_rate (no 'sr' kwarg)
                # predict(audio, sample_rate, hop_length, fmin, fmax, model, ...)
                f0_t, p_t = torchcrepe.predict(
                    x, sr, hop_length, 50.0, 500.0, 'full',
                    batch_size=1024, device=device, return_periodicity=True)
                # 平滑周期性，提高鲁棒性
                p_t = torchcrepe.filter.median(p_t, 3)
                p_t = torchcrepe.filter.mean(p_t, 3)
                # 平滑F0
                f0_t = torchcrepe.filter.median(f0_t, 3)
                f0 = f0_t.squeeze(0).float().cpu().numpy()
                periodicity = p_t.squeeze(0).float().cpu().numpy()
            # 有声判定：周期性阈值 + 合法频段 + 能量掩码
            vmask = (periodicity > 0.45) & (f0 >= 50.0) & (f0 <= 500.0)
            vmask = vmask & energy_mask[:len(vmask)]
            estimator = 'torchcrepe'
        except Exception as e:
            print(f"torchcrepe failed, falling back to pyin: {e}")
            f0, vmask, f0_fb, fb_mask, estimator = _extract_f0_with_pyin(audio_np, sr, hop_length, energy_mask)
            return f0, vmask, f0_fb, fb_mask, estimator
    else:
        f0, vmask, f0_fb, fb_mask, estimator = _extract_f0_with_pyin(audio_np, sr, hop_length, energy_mask)
        return f0, vmask, f0_fb, fb_mask, estimator

    # ---- 回退估计：优先 librosa.yin，其次谱质心 ----
    try:
        f0_fb = librosa.yin(audio_np, fmin=50, fmax=500, sr=sr,
                             frame_length=hop_length * 4, hop_length=hop_length)
        f0_fb = np.where((f0_fb > 0) & (f0_fb >= 50) & (f0_fb <= 500), f0_fb, 0.0).astype(np.float32)
    except Exception:
        f0_fb = _fallback_pitch_from_centroid(audio_np, sr, hop_length)
    fb_mask = (~vmask) & energy_mask[:len(vmask)] & (f0_fb[:len(vmask)] > 0)
    return f0, vmask, f0_fb, fb_mask, estimator


def create_ceps_hist_comparison(
    ceps_orig: torch.Tensor,
    ceps_hat: torch.Tensor,
    save_path: str,
    max_frames: int = 200_000,
) -> None:
    """Compare per-dimension ceps distributions and save a histogram figure.

    This is meant for debugging JSCC/FARGAN alignment. It flattens [B,T,D]
    into frames and plots orig vs hat histograms for the first up-to-18 dims.
    """

    if ceps_orig.dim() != 3 or ceps_hat.dim() != 3:
        return

    with torch.no_grad():
        B, T, D = ceps_orig.shape
        Dm = int(min(D, ceps_hat.size(2)))
        D_use = int(min(Dm, 18))
        if D_use <= 0:
            return

        orig_np = ceps_orig[:, :T, :Dm].detach().cpu().numpy().reshape(-1, Dm)
        hat_np = ceps_hat[:, :T, :Dm].detach().cpu().numpy().reshape(-1, Dm)

        N = min(orig_np.shape[0], hat_np.shape[0], int(max_frames))
        if N <= 0:
            return
        orig_np = orig_np[:N]
        hat_np = hat_np[:N]

        n_dims = D_use
        n_cols = min(6, n_dims)
        n_rows = int(np.ceil(n_dims / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.5 * n_rows))
        axes = np.asarray(axes).reshape(-1)

        for ax, d in zip(axes, range(n_dims)):
            b = orig_np[:, d]
            r = hat_np[:, d]
            ax.hist(b, bins=50, alpha=0.5, label="orig", density=True)
            ax.hist(r, bins=50, alpha=0.5, label="hat", density=True)
            ax.set_title(f"dim {d}")
            ax.grid(True, alpha=0.3)

        for ax in axes[n_dims:]:
            ax.axis("off")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        fig.suptitle("FARGAN ceps dims 0-17", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 0.98, 0.92))

        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Ceps histogram plot saved: {save_path}")


def create_f0_alignment_plot(
    audio_gen: torch.Tensor,
    dnn_pitch_hat: torch.Tensor,
    period_vocoder: torch.Tensor,
    save_path: str,
    sr: int = 16000,
    hop_length: int = 160,
    title: Optional[str] = None,
) -> None:
    """绘制 F0 三轨对齐图：

    - 蓝色: 由 dnn_pitch_hat 映射得到的 F0 (Hz)
    - 橙色: FARGAN 解码时实际使用的 period 映射到的 F0 (Hz)
    - 红色: 从生成 audio_hat 上用 torchcrepe/pyin 估计的 F0 (Hz)

    通过对齐这三条曲线，可以直观判断 F0 在
    "特征 -> 声码器内部 -> 最终音频" 三个阶段
    是从哪一环开始偏离或崩塌的。
    """

    # 将张量搬到 CPU 并展平时间维
    audio_gen = audio_gen.detach().cpu()
    dp_hat = dnn_pitch_hat.detach().cpu().view(-1).to(torch.float32)
    period_v = period_vocoder.detach().cpu().view(-1).to(torch.float32)

    # 1) dnn_pitch_hat -> Hz
    #    period = 256 / 2^(x+1.5)  (clamp 到 [32,255])，f0_hz = 16000 / period
    period_from_dp = 256.0 / torch.pow(2.0, dp_hat + 1.5)
    period_from_dp = torch.clamp(period_from_dp, 32.0, 255.0)
    f0_dp_hz = (16000.0 / period_from_dp).numpy()

    # 2) vocoder 内部使用的 period -> Hz
    period_v = torch.clamp(period_v, 32.0, 255.0)
    f0_period_hz = (16000.0 / period_v).numpy()

    # 3) 从生成音频估计 F0（优先 torchcrepe）
    f0_audio_hz, vmask_audio, _fb, _fb_mask, est_name = extract_f0(
        audio_gen, sr=sr, hop_length=hop_length
    )

    # 对齐长度：按最短长度裁剪，并使用统一时间轴
    T = min(len(f0_dp_hz), len(f0_period_hz), len(f0_audio_hz))
    if T <= 0:
        return

    f0_dp_hz = f0_dp_hz[:T]
    f0_period_hz = f0_period_hz[:T]
    f0_audio_hz = f0_audio_hz[:T]
    vmask_audio = vmask_audio[:T]

    t = np.arange(T) * hop_length / float(sr)

    plt.figure(figsize=(12, 4))
    plt.plot(t, f0_dp_hz, label="dnn_pitch_hat → Hz", color="tab:blue", linewidth=1.5)
    plt.plot(t, f0_period_hz, label="vocoder period → Hz", color="tab:orange", linewidth=1.5)
    plt.plot(
        t,
        np.where(vmask_audio, f0_audio_hz, np.nan),
        label=f"audio_hat F0 ({est_name})",
        color="tab:red",
        linewidth=1.5,
    )

    plt.xlabel("Time (s)")
    plt.ylabel("F0 (Hz)")
    plt.ylim(50, 500)
    plt.grid(True, alpha=0.3)
    if title is None:
        title = "F0 Alignment (feature → period → audio)"
    plt.title(title, fontweight="bold")
    plt.legend(loc="upper right")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _extract_f0_with_pyin(audio_np: np.ndarray, sr: int, hop_length: int, energy_mask: np.ndarray):
    """备用路径：librosa.pyin + 回退。"""
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_np, fmin=50, fmax=500, sr=sr,
            hop_length=hop_length, frame_length=hop_length * 4, fill_na=0.0,
        )
        f0 = np.where((f0 > 0) & (f0 >= 50) & (f0 <= 500), f0, 0.0)
        vmask = (voiced_flag.astype(np.bool_) & (f0 > 0))
    except Exception as e:
        print(f"pYIN failed: {e}")
        f0 = np.zeros((max(1, int(np.ceil(len(audio_np) / float(hop_length))))), dtype=np.float32)
        vmask = np.zeros_like(f0, dtype=bool)

    try:
        f0_fb = librosa.yin(audio_np, fmin=50, fmax=500, sr=sr,
                             frame_length=hop_length * 4, hop_length=hop_length)
        f0_fb = np.where((f0_fb > 0) & (f0_fb >= 50) & (f0_fb <= 500), f0_fb, 0.0).astype(np.float32)
    except Exception:
        f0_fb = _fallback_pitch_from_centroid(audio_np, sr, hop_length)

    fb_mask = (~vmask) & energy_mask[:len(vmask)] & (f0_fb[:len(vmask)] > 0)
    return f0, vmask, f0_fb, fb_mask, 'pyin'


def extract_mel_spectrogram(audio: torch.Tensor, sr: int = 16000,
                          n_fft: int = 1024, hop_length: int = 160,
                          n_mels: int = 80) -> np.ndarray:
    """提取Mel谱图"""
    audio_np = audio.detach().cpu().numpy()

    # 计算Mel谱图
    mel = librosa.feature.melspectrogram(
        y=audio_np,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=0,
        fmax=sr // 2
    )

    # 使用固定参考刻度，避免每条音频各自按峰值归一化后掩盖真实能量差异。
    mel_db = librosa.power_to_db(mel, ref=1.0, top_db=120.0)

    return mel_db


def create_audio_comparison_plot(
    audio_real: torch.Tensor,
    audio_gen: torch.Tensor,
    save_path: str,
    sr: int = 16000,
    title: str = "Audio Comparison",
    show_waveform: bool = True,
    hop_length: int = 160
) -> None:
    """
    创建音频对比图，包含波形、F0和Mel谱图

    Args:
        audio_real: 真实音频 [L]
        audio_gen: 生成音频 [L]
        save_path: 保存路径
        sr: 采样率
        title: 图标题
        show_waveform: 是否显示波形
    """
    # 确保Tensor已detach并移到CPU
    audio_real = audio_real.detach().cpu()
    audio_gen = audio_gen.detach().cpu()

    # 确保音频长度一致
    min_len = min(audio_real.size(0), audio_gen.size(0))
    audio_real = audio_real[:min_len]
    audio_gen = audio_gen[:min_len]

    # 提取特征（返回 f0/voiced 与 fallback）
    f0_real, vmask_real, f0_real_fb, vmask_real_fb, est_real = extract_f0(audio_real, sr, hop_length)
    f0_gen, vmask_gen, f0_gen_fb, vmask_gen_fb, est_gen = extract_f0(audio_gen, sr, hop_length)

    mel_real = extract_mel_spectrogram(audio_real, sr, hop_length=hop_length)
    mel_gen = extract_mel_spectrogram(audio_gen, sr, hop_length=hop_length)

    # 创建图形
    if show_waveform:
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 8))

    fig.suptitle(title, fontsize=16, fontweight='bold')

    # 确保F0长度一致，使用较短的长度
    min_f0_len = min(len(f0_real), len(f0_gen))
    f0_real = f0_real[:min_f0_len]
    f0_gen = f0_gen[:min_f0_len]
    vmask_real = vmask_real[:min_f0_len]
    vmask_gen = vmask_gen[:min_f0_len]
    f0_real_fb = f0_real_fb[:min_f0_len]
    f0_gen_fb = f0_gen_fb[:min_f0_len]
    vmask_real_fb = vmask_real_fb[:min_f0_len]
    vmask_gen_fb = vmask_gen_fb[:min_f0_len]

    # 确保Mel长度一致
    min_mel_frames = min(mel_real.shape[1], mel_gen.shape[1])
    mel_real = mel_real[:, :min_mel_frames]
    mel_gen = mel_gen[:, :min_mel_frames]

    # 时间轴计算
    time_audio = np.arange(audio_real.size(0)) / sr
    time_f0 = np.arange(min_f0_len) * hop_length / sr if min_f0_len > 0 else np.array([0])
    time_mel = np.arange(min_mel_frames) * hop_length / sr if min_mel_frames > 0 else np.array([0])
    wave_peak = float(
        max(
            audio_real.abs().max().item() if audio_real.numel() > 0 else 0.0,
            audio_gen.abs().max().item() if audio_gen.numel() > 0 else 0.0,
            1e-4,
        )
    )
    wave_lim = 1.05 * wave_peak

    mel_join = np.concatenate([mel_real.reshape(-1), mel_gen.reshape(-1)], axis=0)
    mel_vmax = float(np.percentile(mel_join, 99.5)) if mel_join.size > 0 else 0.0
    mel_vmin = float(np.percentile(mel_join, 0.5)) if mel_join.size > 0 else -120.0
    if mel_vmax - mel_vmin < 20.0:
        mel_vmin = mel_vmax - 20.0

    row_idx = 0

    # 1. 波形对比（如果启用）
    if show_waveform:
        axes[row_idx, 0].plot(time_audio, audio_real.numpy(), 'b-', alpha=0.7, linewidth=0.5)
        axes[row_idx, 0].set_title('Real Audio Waveform', fontweight='bold')
        axes[row_idx, 0].set_xlabel('Time (s)')
        axes[row_idx, 0].set_ylabel('Amplitude')
        axes[row_idx, 0].grid(True, alpha=0.3)
        axes[row_idx, 0].set_ylim(-wave_lim, wave_lim)

        axes[row_idx, 1].plot(time_audio, audio_gen.numpy(), 'r-', alpha=0.7, linewidth=0.5)
        axes[row_idx, 1].set_title('Generated Audio Waveform', fontweight='bold')
        axes[row_idx, 1].set_xlabel('Time (s)')
        axes[row_idx, 1].set_ylabel('Amplitude')
        axes[row_idx, 1].grid(True, alpha=0.3)
        axes[row_idx, 1].set_ylim(-wave_lim, wave_lim)

        row_idx += 1

    # 2. F0对比（以有声掩码显示；无声段用灰色遮罩；若全部无声则用质心虚线显示）
    # 先计算真实端统计（仍以 pYIN 为主）
    f0_real_valid = f0_real[vmask_real]
    f0_real_mean = np.mean(f0_real_valid) if f0_real_valid.size > 0 else 0
    f0_real_std = np.std(f0_real_valid) if f0_real_valid.size > 0 else 0
    f0_real_ratio = float(vmask_real.mean()) if vmask_real.size > 0 else 0

    fr_plot = np.where(vmask_real, f0_real, np.nan)
    axes[row_idx, 0].plot(time_f0, fr_plot, 'b-', linewidth=2, label='Real F0')
    # 对于 pyin 判无声但能量足够的帧，叠加回退估计（灰色虚线）
    if np.any(vmask_real_fb):
        fb_plot_r = np.where(vmask_real_fb, f0_real_fb, np.nan)
        axes[row_idx, 0].plot(time_f0, fb_plot_r, color='gray', linestyle='--', linewidth=1.0, label='Fallback (YIN/centroid)')
    if (~vmask_real).any():
        axes[row_idx, 0].fill_between(time_f0, 50, 500, where=(~vmask_real), color='lightgray', alpha=0.2, step='pre')
    axes[row_idx, 0].set_title(f'Real Audio F0 ({f0_real_ratio:.1%} voiced, est={est_real})\nMean: {f0_real_mean:.1f}Hz, Std: {f0_real_std:.1f}Hz', fontweight='bold')
    axes[row_idx, 0].set_xlabel('Time (s)')
    axes[row_idx, 0].set_ylabel('F0 (Hz)')
    axes[row_idx, 0].grid(True, alpha=0.3)
    axes[row_idx, 0].set_ylim(50, 500)
    # 固定横轴范围，避免全NaN导致 [-0.05, 0.05] 的默认范围
    if time_f0.size > 0:
        axes[row_idx, 0].set_xlim(0.0, float(time_f0[-1]) if time_f0[-1] > 0 else 0.05)

    # 生成端：主曲线固定为“主估计器”（优先 torchcrepe），辅曲线为回退估计
    vratio_gen = float(vmask_gen.mean()) if vmask_gen.size > 0 else 0.0
    vratio_gen_fb = float(vmask_gen_fb.mean()) if vmask_gen_fb.size > 0 else 0.0
    fg_main = np.where(vmask_gen, f0_gen, np.nan)
    fg_aux  = np.where(vmask_gen_fb, f0_gen_fb, np.nan)
    axes[row_idx, 1].plot(time_f0, fg_main, 'r-', linewidth=2, label='Generated F0 (primary)')
    if np.any(vmask_gen_fb):
        axes[row_idx, 1].plot(time_f0, fg_aux, 'r--', linewidth=1.0, label='fallback')
    if (~vmask_gen).any():
        axes[row_idx, 1].fill_between(time_f0, 50, 500, where=(~vmask_gen), color='lightgray', alpha=0.2, step='pre')
    f0_gen_valid = f0_gen[vmask_gen]
    f0_gen_mean_plot = float(np.mean(f0_gen_valid)) if f0_gen_valid.size > 0 else 0.0
    f0_gen_std_plot = float(np.std(f0_gen_valid)) if f0_gen_valid.size > 0 else 0.0
    f0_gen_ratio_plot = vratio_gen
    est_note = est_gen

    axes[row_idx, 1].set_title(
        f'Generated Audio F0 ({f0_gen_ratio_plot:.1%} voiced, {est_note})\nMean: {f0_gen_mean_plot:.1f}Hz, Std: {f0_gen_std_plot:.1f}Hz',
        fontweight='bold'
    )
    axes[row_idx, 1].set_xlabel('Time (s)')
    axes[row_idx, 1].set_ylabel('F0 (Hz)')
    axes[row_idx, 1].grid(True, alpha=0.3)
    axes[row_idx, 1].set_ylim(50, 500)
    if time_f0.size > 0:
        axes[row_idx, 1].set_xlim(0.0, float(time_f0[-1]) if time_f0[-1] > 0 else 0.05)

    row_idx += 1

    # 3. Mel谱图对比
    im1 = axes[row_idx, 0].imshow(mel_real, aspect='auto', origin='lower',
                                  extent=[0, time_mel[-1], 0, mel_real.shape[0]],
                                  cmap='viridis', vmin=mel_vmin, vmax=mel_vmax)
    axes[row_idx, 0].set_title('Real Audio Mel Spectrogram', fontweight='bold')
    axes[row_idx, 0].set_xlabel('Time (s)')
    axes[row_idx, 0].set_ylabel('Mel Bin')

    im2 = axes[row_idx, 1].imshow(mel_gen, aspect='auto', origin='lower',
                                  extent=[0, time_mel[-1], 0, mel_gen.shape[0]],
                                  cmap='viridis', vmin=mel_vmin, vmax=mel_vmax)
    axes[row_idx, 1].set_title('Generated Audio Mel Spectrogram', fontweight='bold')
    axes[row_idx, 1].set_xlabel('Time (s)')
    axes[row_idx, 1].set_ylabel('Mel Bin')

    # 添加颜色条
    plt.colorbar(im1, ax=axes[row_idx, 0], format='%+2.0f dB', shrink=0.8)
    plt.colorbar(im2, ax=axes[row_idx, 1], format='%+2.0f dB', shrink=0.8)

    # 计算并显示统计信息（仅在有声交集上计算 F0 MSE；若无交集则尝试回退掩码）
    Tm_len = min(len(f0_real), len(f0_gen))
    f0_r = f0_real[:Tm_len]
    f0_g = f0_gen[:Tm_len]
    vm_r = vmask_real[:Tm_len]
    vm_g = vmask_gen[:Tm_len]
    inter = vm_r & vm_g
    if np.any(inter):
        f0_mse = float(np.mean((f0_r[inter] - f0_g[inter])**2))
    else:
        # 回退到能量有效的 YIN/centroid 估计的交集
        fb_r = f0_real_fb[:Tm_len]
        fb_g = f0_gen_fb[:Tm_len]
        vm_r_fb = vmask_real_fb[:Tm_len]
        vm_g_fb = vmask_gen_fb[:Tm_len]
        inter_fb = vm_r_fb & vm_g_fb
        f0_mse = float(np.mean((fb_r[inter_fb] - fb_g[inter_fb])**2)) if np.any(inter_fb) else float('nan')
    mel_mse = np.mean((mel_real - mel_gen)**2)

    # 在图上添加统计信息
    stats_text = f'F0 MSE: {f0_mse:.2f} Hz²\nMel MSE: {mel_mse:.3f} dB²'
    fig.text(0.02, 0.02, stats_text, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.1)

    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Audio comparison plot saved: {save_path}")
    print(f"F0 MSE: {f0_mse:.2f} Hz², Mel MSE: {mel_mse:.3f} dB²")


def create_batch_comparison_plots(
    audio_real_batch: torch.Tensor,
    audio_gen_batch: torch.Tensor,
    save_dir: str,
    step: int,
    max_samples: int = 3,
    sr: int = 16000
) -> None:
    """
    为batch中的音频样本创建对比图

    Args:
        audio_real_batch: 真实音频批 [B, L]
        audio_gen_batch: 生成音频批 [B, L]
        save_dir: 保存目录
        step: 训练步数
        max_samples: 最大生成样本数
        sr: 采样率
    """
    batch_size = min(audio_real_batch.size(0), max_samples)

    for i in range(batch_size):
        audio_real = audio_real_batch[i]  # [L]
        audio_gen = audio_gen_batch[i]    # [L]

        save_path = os.path.join(save_dir, f"comparison_step_{step:06d}_sample_{i:02d}.png")
        title = f"Audio Comparison - Step {step} - Sample {i}"

        try:
            create_audio_comparison_plot(
                audio_real=audio_real,
                audio_gen=audio_gen,
                save_path=save_path,
                sr=sr,
                title=title,
                show_waveform=True,
                hop_length=160
            )
        except Exception as e:
            print(f"Failed to create comparison plot for sample {i}: {e}")


def create_source_control_plot(
    dnn_pitch_ref: torch.Tensor,
    dnn_pitch_hat: torch.Tensor,
    frame_corr_ref: torch.Tensor,
    frame_corr_hat: torch.Tensor,
    period_vocoder: torch.Tensor,
    gain_ref: torch.Tensor,
    gain_hat: torch.Tensor,
    save_path: str,
    sr: int = 16000,
    hop_length: int = 160,
    title: str = "Source Controls",
) -> None:
    """Visualise the source-control chain used by the vocoder for one sample."""

    def _flat_np(x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().to(torch.float32).view(-1).numpy()

    def _stats_line(name: str, arr: np.ndarray) -> str:
        if arr.size <= 0:
            return f"{name}: empty"
        return (
            f"{name}: mean={float(np.mean(arr)):+.4f} std={float(np.std(arr)):.4f} "
            f"min={float(np.min(arr)):+.4f} max={float(np.max(arr)):+.4f}"
        )

    dp_ref = _flat_np(dnn_pitch_ref)
    dp_hat = _flat_np(dnn_pitch_hat)
    fc_ref = _flat_np(frame_corr_ref)
    fc_hat = _flat_np(frame_corr_hat)
    period_hat = _flat_np(period_vocoder)
    gain_r = _flat_np(gain_ref)
    gain_h = _flat_np(gain_hat)

    T = min(len(dp_ref), len(dp_hat), len(fc_ref), len(fc_hat), len(period_hat), len(gain_r), len(gain_h))
    if T <= 0:
        return

    dp_ref = dp_ref[:T]
    dp_hat = dp_hat[:T]
    fc_ref = fc_ref[:T]
    fc_hat = fc_hat[:T]
    period_hat = np.clip(period_hat[:T], 32.0, 255.0)
    gain_r = gain_r[:T]
    gain_h = gain_h[:T]

    period_ref = np.clip(256.0 / np.power(2.0, dp_ref + 1.5), 32.0, 255.0)
    f0_ref = sr / period_ref
    f0_hat = sr / np.clip(256.0 / np.power(2.0, dp_hat + 1.5), 32.0, 255.0)
    t = np.arange(T, dtype=np.float32) * (hop_length / float(sr))

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    fig.suptitle(title, fontsize=15, fontweight='bold')

    axes[0].plot(t, f0_ref, color='tab:blue', linewidth=1.6, label='ref pitch -> Hz')
    axes[0].plot(t, f0_hat, color='tab:red', linewidth=1.2, label='hat pitch -> Hz')
    axes[0].set_ylabel('F0 (Hz)')
    axes[0].set_ylim(50, 500)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')

    axes[1].plot(t, fc_ref, color='tab:blue', linewidth=1.6, label='ref frame_corr')
    axes[1].plot(t, fc_hat, color='tab:red', linewidth=1.2, label='hat frame_corr')
    axes[1].axhline(0.25, color='gray', linestyle='--', linewidth=0.8, alpha=0.8)
    axes[1].set_ylabel('frame_corr')
    axes[1].set_ylim(-0.9, 0.6)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')

    axes[2].plot(t, period_ref, color='tab:blue', linewidth=1.6, label='ref period')
    axes[2].plot(t, period_hat, color='tab:orange', linewidth=1.2, label='vocoder period')
    axes[2].set_ylabel('Period')
    axes[2].set_ylim(30, 260)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper right')

    axes[3].plot(t, gain_r, color='tab:blue', linewidth=1.6, label='ref gain')
    axes[3].plot(t, gain_h, color='tab:red', linewidth=1.2, label='hat gain')
    axes[3].set_ylabel('Gain')
    axes[3].set_xlabel('Time (s)')
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc='upper right')

    voiced_ref = float((fc_ref > 0.25).mean()) if fc_ref.size > 0 else 0.0
    voiced_hat = float((fc_hat > 0.25).mean()) if fc_hat.size > 0 else 0.0
    stats_text = "\n".join(
        [
            _stats_line("dnn_pitch_ref", dp_ref),
            _stats_line("dnn_pitch_hat", dp_hat),
            _stats_line("frame_corr_ref", fc_ref),
            _stats_line("frame_corr_hat", fc_hat),
            _stats_line("period_ref", period_ref),
            _stats_line("period_vocoder", period_hat),
            _stats_line("gain_ref", gain_r),
            _stats_line("gain_hat", gain_h),
            f"voiced_ratio_ref(>0.25)={voiced_ref:.1%} | voiced_ratio_hat(>0.25)={voiced_hat:.1%}",
        ]
    )
    fig.text(
        0.012,
        0.012,
        stats_text,
        fontsize=9,
        family='monospace',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.85),
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.2)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Source control plot saved: {save_path}")


def create_vocoder_internal_plot(
    pitch_gain_mean: torch.Tensor,
    fwc0_rms: torch.Tensor,
    skip_rms: torch.Tensor,
    sig_core_rms: torch.Tensor,
    sig_out_rms: torch.Tensor,
    save_path: str,
    sr: int = 16000,
    hop_length: int = 160,
    title: str = "Vocoder Internals",
) -> None:
    """Plot key vocoder internal traces aggregated at frame rate."""

    def _flat_np(x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().to(torch.float32).view(-1).numpy()

    def _stats_line(name: str, arr: np.ndarray) -> str:
        if arr.size <= 0:
            return f"{name}: empty"
        return (
            f"{name}: mean={float(np.mean(arr)):+.4f} std={float(np.std(arr)):.4f} "
            f"min={float(np.min(arr)):+.4f} max={float(np.max(arr)):+.4f}"
        )

    pg = _flat_np(pitch_gain_mean)
    fwc = _flat_np(fwc0_rms)
    sk = _flat_np(skip_rms)
    sc = _flat_np(sig_core_rms)
    so = _flat_np(sig_out_rms)
    T = min(len(pg), len(fwc), len(sk), len(sc), len(so))
    if T <= 0:
        return
    pg = pg[:T]
    fwc = fwc[:T]
    sk = sk[:T]
    sc = sc[:T]
    so = so[:T]
    t = np.arange(T, dtype=np.float32) * (hop_length / float(sr))

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    fig.suptitle(title, fontsize=15, fontweight='bold')

    axes[0].plot(t, pg, color='tab:purple', linewidth=1.6)
    axes[0].set_ylabel('pitch_gain')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, fwc, color='tab:orange', linewidth=1.6, label='fwc0_rms')
    axes[1].plot(t, sk, color='tab:green', linewidth=1.2, label='skip_rms')
    axes[1].set_ylabel('state rms')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')

    axes[2].plot(t, sc, color='tab:red', linewidth=1.6)
    axes[2].set_ylabel('sig_core_rms')
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(t, so, color='tab:brown', linewidth=1.6, label='sig_out_rms')
    axes[3].plot(t, sc, color='tab:red', linewidth=1.0, alpha=0.6, label='sig_core_rms')
    axes[3].set_ylabel('output rms')
    axes[3].set_xlabel('Time (s)')
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc='upper right')

    stats_text = "\n".join(
        [
            _stats_line("pitch_gain_mean", pg),
            _stats_line("fwc0_rms", fwc),
            _stats_line("skip_rms", sk),
            _stats_line("sig_core_rms", sc),
            _stats_line("sig_out_rms", so),
            f"sig_out/sig_core mean ratio={float(np.mean(so / np.maximum(sc, 1e-8))):.4f}",
        ]
    )
    fig.text(
        0.012,
        0.012,
        stats_text,
        fontsize=9,
        family='monospace',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.85),
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.18)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Vocoder internal plot saved: {save_path}")


def save_comparison_audio_samples(
    audio_real_batch: torch.Tensor,
    audio_gen_batch: torch.Tensor,
    save_dir: str,
    step: int,
    max_samples: int = 3,
    sr: int = 16000
) -> None:
    """
    保存音频样本文件（配合可视化使用）

    Args:
        audio_real_batch: 真实音频批 [B, L]
        audio_gen_batch: 生成音频批 [B, L]
        save_dir: 保存目录
        step: 训练步数
        max_samples: 最大保存样本数
        sr: 采样率
    """
    os.makedirs(save_dir, exist_ok=True)
    batch_size = min(audio_real_batch.size(0), max_samples)

    for i in range(batch_size):
        audio_real = audio_real_batch[i].detach().cpu()  # [L]
        audio_gen = audio_gen_batch[i].detach().cpu()    # [L]

        # 归一化
        audio_real = audio_real / (audio_real.abs().max() + 1e-8)
        audio_gen = audio_gen / (audio_gen.abs().max() + 1e-8)

        # 保存音频文件
        real_path = os.path.join(save_dir, f"step_{step:06d}_sample_{i:02d}_real.wav")
        gen_path = os.path.join(save_dir, f"step_{step:06d}_sample_{i:02d}_gen.wav")

        try:
            torchaudio.save(real_path, audio_real.unsqueeze(0), sr)
            torchaudio.save(gen_path, audio_gen.unsqueeze(0), sr)
        except Exception as e:
            print(f"Failed to save audio sample {i}: {e}")
