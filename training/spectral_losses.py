# -*- coding: utf-8 -*-
"""
FARGAN专用损失函数
适配原始FARGAN训练中的信号损失和连续性损失
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math


def fargan_signal_loss(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    FARGAN信号损失 (对应原始fargan.py中的sig_loss)
    计算归一化信号之间的余弦距离

    Args:
        y_true: [B, T] 真实信号
        y_pred: [B, T] 预测信号
        eps: 数值稳定性常数

    Returns:
        loss: 标量损失值
    """
    # 归一化信号
    t_norm = y_true / (eps + torch.norm(y_true, dim=-1, p=2, keepdim=True))
    p_norm = y_pred / (eps + torch.norm(y_pred, dim=-1, p=2, keepdim=True))

    # 余弦相似度损失
    cosine_sim = torch.sum(p_norm * t_norm, dim=-1)  # [B]
    loss = torch.mean(1.0 - cosine_sim)

    return loss


def fargan_l1_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    FARGAN相对L1损失 (对应原始fargan.py中的sig_l1)

    Args:
        y_true: [B, T] 真实信号
        y_pred: [B, T] 预测信号

    Returns:
        loss: 标量损失值
    """
    abs_diff = torch.abs(y_true - y_pred)
    abs_true = torch.abs(y_true)
    return torch.mean(abs_diff) / torch.mean(abs_true)


def fargan_continuity_loss(
    y_pred: torch.Tensor,
    frame_size: int = 160,
    weight: float = 1.0
) -> torch.Tensor:
    """
    FARGAN连续性损失
    确保生成信号在帧边界处的连续性

    Args:
        y_pred: [B, T] 预测信号
        frame_size: 帧大小 (默认160样本)
        weight: 损失权重

    Returns:
        loss: 连续性损失
    """
    if y_pred.size(-1) < frame_size * 2:
        return torch.zeros((), device=y_pred.device)

    # 提取帧边界附近的样本
    boundary_size = 8  # 边界附近的样本数
    total_frames = y_pred.size(-1) // frame_size

    continuity_loss = torch.zeros((), device=y_pred.device)
    count = 0

    for i in range(1, total_frames):
        boundary_pos = i * frame_size
        if boundary_pos >= boundary_size and boundary_pos + boundary_size < y_pred.size(-1):
            # 边界前后的信号段
            before = y_pred[..., boundary_pos - boundary_size:boundary_pos]
            after = y_pred[..., boundary_pos:boundary_pos + boundary_size]

            # 计算一阶差分的连续性
            diff_before = torch.diff(before, dim=-1)
            diff_after = torch.diff(after, dim=-1)

            # 边界处的跳跃
            boundary_jump = torch.abs(before[..., -1] - after[..., 0])
            gradient_jump = torch.abs(diff_before[..., -1] - diff_after[..., 0])

            continuity_loss = continuity_loss + boundary_jump.mean() + 0.5 * gradient_jump.mean()
            count += 1

    if count > 0:
        continuity_loss = continuity_loss / count

    return weight * continuity_loss


def fargan_pitch_consistency_loss(
    y_pred: torch.Tensor,
    period: torch.Tensor,
    frame_size: int = 160,
    subframe_size: int = 40,
    weight: float = 0.1
) -> torch.Tensor:
    """
    FARGAN基频一致性损失
    确保生成的信号与给定周期的基频一致

    Args:
        y_pred: [B, T] 预测信号
        period: [B, T_frames] 周期序列
        frame_size: 帧大小
        subframe_size: 子帧大小
        weight: 损失权重

    Returns:
        loss: 基频一致性损失
    """
    B, T = y_pred.shape
    nb_frames = T // frame_size
    nb_subframes = frame_size // subframe_size

    # 检查period是否为None（Stage5兼容性）
    if period is None or nb_frames == 0 or period.size(1) < nb_frames:
        return torch.zeros((), device=y_pred.device)

    pitch_loss = torch.zeros((), device=y_pred.device)
    count = 0

    for frame_idx in range(nb_frames):
        if frame_idx + 3 >= period.size(1):  # FARGAN uses period[:, 3+n]
            break

        frame_period = period[:, frame_idx + 3]  # [B]
        frame_start = frame_idx * frame_size
        frame_end = frame_start + frame_size

        if frame_end > T:
            break

        frame_signal = y_pred[:, frame_start:frame_end]  # [B, frame_size]

        for b in range(B):
            p = int(frame_period[b].item())
            if p < 32 or p > 255:
                continue

            signal = frame_signal[b]  # [frame_size]

            # 计算自相关
            if len(signal) >= p * 2:
                autocorr = F.conv1d(
                    signal.unsqueeze(0).unsqueeze(0),
                    signal[:p].flip(0).unsqueeze(0).unsqueeze(0),
                    padding=0
                ).squeeze()

                if len(autocorr) > 0:
                    # 期望在周期位置有较高的自相关
                    target_pos = len(autocorr) - p
                    if target_pos >= 0 and target_pos < len(autocorr):
                        max_corr = autocorr.max()
                        target_corr = autocorr[target_pos]
                        pitch_loss = pitch_loss + F.relu(max_corr - target_corr) / (max_corr + 1e-6)
                        count += 1

    if count > 0:
        pitch_loss = pitch_loss / count

    return weight * pitch_loss


def fargan_subframe_alignment_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    subframe_size: int = 40,
    weight: float = 0.2
) -> torch.Tensor:
    """
    FARGAN子帧对齐损失
    确保在子帧级别的精确对齐

    Args:
        y_pred: [B, T] 预测信号
        y_true: [B, T] 真实信号
        subframe_size: 子帧大小
        weight: 损失权重

    Returns:
        loss: 子帧对齐损失
    """
    T = min(y_pred.size(-1), y_true.size(-1))
    nb_subframes = T // subframe_size

    if nb_subframes == 0:
        return torch.zeros((), device=y_pred.device)

    subframe_loss = torch.zeros((), device=y_pred.device)

    for i in range(nb_subframes):
        start = i * subframe_size
        end = start + subframe_size

        pred_subframe = y_pred[..., start:end]
        true_subframe = y_true[..., start:end]

        # 子帧级MSE损失
        mse_loss = F.mse_loss(pred_subframe, true_subframe)

        # 子帧能量对齐
        pred_energy = torch.mean(pred_subframe.pow(2), dim=-1)
        true_energy = torch.mean(true_subframe.pow(2), dim=-1)
        energy_loss = F.mse_loss(pred_energy, true_energy)

        subframe_loss = subframe_loss + mse_loss + 0.1 * energy_loss

    subframe_loss = subframe_loss / nb_subframes
    return weight * subframe_loss


def compute_fargan_comprehensive_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    period: torch.Tensor,
    frame_size: int = 160,
    subframe_size: int = 40,
    weights: Optional[Dict[str, float]] = None
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    计算FARGAN综合损失

    Args:
        y_pred: [B, T] 预测信号
        y_true: [B, T] 真实信号
        period: [B, T_frames] 周期序列
        frame_size: 帧大小
        subframe_size: 子帧大小
        weights: 各损失项的权重

    Returns:
        total_loss: 总损失
        loss_dict: 各损失项的详细信息
    """
    if weights is None:
        weights = {
            'signal': 1.0,
            'l1': 0.1,
            'continuity': 0.03,
            'pitch_consistency': 0.1,
            'subframe_alignment': 0.2
        }

    # 确保信号长度一致
    min_len = min(y_pred.size(-1), y_true.size(-1))
    y_pred = y_pred[..., :min_len]
    y_true = y_true[..., :min_len]

    # 各损失项
    signal_loss = fargan_signal_loss(y_true, y_pred)
    l1_loss = fargan_l1_loss(y_true, y_pred)
    continuity_loss = fargan_continuity_loss(y_pred, frame_size)
    pitch_loss = fargan_pitch_consistency_loss(y_pred, period, frame_size, subframe_size)
    subframe_loss = fargan_subframe_alignment_loss(y_pred, y_true, subframe_size)

    # 数值清洗
    def _clean_loss(loss_tensor):
        return torch.nan_to_num(loss_tensor, nan=0.0, posinf=1e4, neginf=-1e4)

    signal_loss = _clean_loss(signal_loss)
    l1_loss = _clean_loss(l1_loss)
    continuity_loss = _clean_loss(continuity_loss)
    pitch_loss = _clean_loss(pitch_loss)
    subframe_loss = _clean_loss(subframe_loss)

    # 加权组合
    total_loss = (
        weights['signal'] * signal_loss +
        weights['l1'] * l1_loss +
        weights['continuity'] * continuity_loss +
        weights['pitch_consistency'] * pitch_loss +
        weights['subframe_alignment'] * subframe_loss
    )

    loss_dict = {
        'fargan_signal': signal_loss,
        'fargan_l1': l1_loss,
        'fargan_continuity': continuity_loss,
        'fargan_pitch_consistency': pitch_loss,
        'fargan_subframe_alignment': subframe_loss,
        'fargan_total': total_loss
    }

    return total_loss, loss_dict


# ------------------------------
# 训练期：加入多分辨率STFT损失与基础项
# ------------------------------

_WINDOW_CACHE = {}

def _get_hann_window(length: int, device: torch.device) -> torch.Tensor:
    """Cache and return a Hann window tensor on the given device."""
    key = (int(length), str(device))
    win = _WINDOW_CACHE.get(key)
    if win is None or win.device != device:
        win = torch.hann_window(length, device=device)
        _WINDOW_CACHE[key] = win
    return win

def _stft_mag(x: torch.Tensor, fft_size: int, hop_size: int, win_length: int, window: torch.Tensor) -> torch.Tensor:
    x_stft = torch.stft(x, n_fft=fft_size, hop_length=hop_size, win_length=win_length,
                        window=window, return_complex=True)
    # 使用log1p幅度以增强感知相关性，提高数值稳定性
    return torch.log1p(torch.clamp(torch.abs(x_stft), min=1e-4))


def multi_resolution_stft_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    device: torch.device,
    fft_sizes: Optional[List[int]] = None,
    hop_sizes: Optional[List[int]] = None,
    win_lengths: Optional[List[int]] = None,
) -> torch.Tensor:
    """轻量级多分辨率STFT幅度L1损失（与原版思想一致，计算更简洁）。

    输入为 [B, T]，返回标量损失。
    """
    if fft_sizes is None:
        fft_sizes = [1024, 512, 256]
    if hop_sizes is None:
        hop_sizes = [256, 128, 64]
    if win_lengths is None:
        win_lengths = [1024, 512, 256]

    # 对齐长度
    min_len = min(y_pred.size(-1), y_true.size(-1))
    y_pred = y_pred[..., :min_len]
    y_true = y_true[..., :min_len]

    total = 0.0
    for fs, hs, wl in zip(fft_sizes, hop_sizes, win_lengths):
        window = _get_hann_window(wl, device)
        mag_p = _stft_mag(y_pred, fs, hs, wl, window)
        mag_t = _stft_mag(y_true, fs, hs, wl, window)
        total = total + F.l1_loss(mag_p, mag_t)

    return total / len(fft_sizes)


# ------------------------------
# 原版风格：谱收敛 + 单帧信号余弦
# ------------------------------

def _spectral_convergence(x_mag: torch.Tensor, y_mag: torch.Tensor) -> torch.Tensor:
    # 与原仓库类似：先开方，再按L1比值做谱收敛
    x_mag = torch.sqrt(x_mag)
    y_mag = torch.sqrt(y_mag)
    num = torch.norm(y_mag - x_mag, p=1)
    den = torch.norm(y_mag, p=1) + 1e-12
    return num / den


def multi_resolution_sc_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    fft_sizes: Optional[List[int]] = None,
    hop_sizes: Optional[List[int]] = None,
    win_lengths: Optional[List[int]] = None,
) -> torch.Tensor:
    """多分辨率谱收敛损失，仿 dnn/torch/fargan/stft_loss.py。"""
    if fft_sizes is None:
        fft_sizes = [2560, 1280, 640, 320, 160, 80]
    if hop_sizes is None:
        hop_sizes = [640, 320, 160, 80, 40, 20]
    if win_lengths is None:
        win_lengths = [2560, 1280, 640, 320, 160, 80]

    min_len = min(x.size(-1), y.size(-1))
    x = x[..., :min_len]
    y = y[..., :min_len]

    sc_total = 0.0
    for fs, hs, wl in zip(fft_sizes, hop_sizes, win_lengths):
        window = _get_hann_window(wl, device)
        x_mag = torch.stft(x, n_fft=fs, hop_length=hs, win_length=wl, window=window, return_complex=True).abs().clamp_min(1e-7)
        y_mag = torch.stft(y, n_fft=fs, hop_length=hs, win_length=wl, window=window, return_complex=True).abs().clamp_min(1e-7)
        sc_total = sc_total + _spectral_convergence(x_mag, y_mag)
    return sc_total / len(fft_sizes)


def compute_fargan_original_style_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    device: torch.device,
    frame_size: int = 160,
    focus_start: int = 0,
    sc_weights: Optional[Dict[str, float]] = None,
    sig_weight: float = 0.03,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """原版风格损失：多分辨率谱收敛 + 单帧信号余弦（仅聚焦一帧）。

    Args:
        y_pred: [B, T]
        y_true: [B, T]
        device: torch.device
        frame_size: 聚焦帧长度（默认160）
        focus_start: 聚焦帧起始位置（通常是 pre 之后的首帧）
        sc_weights: 可选权重配置（当前未使用，预留）
        sig_weight: 单帧信号余弦权重（默认0.03）
    """
    min_len = min(y_pred.size(-1), y_true.size(-1))
    y_pred = y_pred[..., :min_len]
    y_true = y_true[..., :min_len]

    # 谱收敛
    sc = multi_resolution_sc_loss(y_pred, y_true, device)

    # 单帧信号余弦：对齐 focus 帧
    end = min(focus_start + frame_size, min_len)
    if end - focus_start >= frame_size // 2:
        sig = fargan_signal_loss(y_true[..., focus_start:end], y_pred[..., focus_start:end])
    else:
        sig = torch.zeros((), device=device)

    total = sc + sig_weight * sig
    return total, {
        'orig_sc': sc,
        'orig_sig': sig,
        'orig_total': total,
    }


def compute_fargan_training_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    period: torch.Tensor,
    frame_size: int = 160,
    subframe_size: int = 40,
    weights: Optional[Dict[str, float]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """训练期综合损失：在验证用综合损失基础上加入多分辨率STFT、L2与能量约束。

    默认权重更倾向快速收敛与一定感知质量约束。
    """
    if weights is None:
        weights = {
            'l1': 1.0,
            'l2': 0.2,
            'energy': 0.3,
            'mr_stft': 1.0,
            'signal': 0.2,
            'continuity': 0.02,
            'pitch_consistency': 0.05,
            'subframe_alignment': 0.05,
        }

    # 对齐长度
    min_len = min(y_pred.size(-1), y_true.size(-1))
    y_pred = y_pred[..., :min_len]
    y_true = y_true[..., :min_len]

    # 基础重建项
    l1_loss = F.l1_loss(y_pred, y_true)
    l2_loss = F.mse_loss(y_pred, y_true)
    pred_energy = torch.mean(y_pred ** 2, dim=-1)
    true_energy = torch.mean(y_true ** 2, dim=-1)
    energy_loss = F.l1_loss(pred_energy, true_energy)

    # FARGAN结构相关项
    signal_loss = fargan_signal_loss(y_true, y_pred)
    continuity_loss = fargan_continuity_loss(y_pred, frame_size)
    pitch_loss = fargan_pitch_consistency_loss(y_pred, period, frame_size, subframe_size)
    subframe_loss = fargan_subframe_alignment_loss(y_pred, y_true, subframe_size)

    # 多分辨率STFT
    dev = device if device is not None else y_pred.device
    mr_stft = multi_resolution_stft_loss(y_pred, y_true, dev)

    # 组合
    total = (
        weights['l1'] * l1_loss +
        weights['l2'] * l2_loss +
        weights['energy'] * energy_loss +
        weights['mr_stft'] * mr_stft +
        weights['signal'] * signal_loss +
        weights['continuity'] * continuity_loss +
        weights['pitch_consistency'] * pitch_loss +
        weights['subframe_alignment'] * subframe_loss
    )

    def clean(t):
        # Accept float or Tensor; always return Tensor on the correct device
        tt = t if isinstance(t, torch.Tensor) else torch.tensor(t, device=dev)
        return torch.nan_to_num(tt, nan=0.0, posinf=1e4, neginf=-1e4)
    loss_dict = {
        'train_l1': clean(l1_loss),
        'train_l2': clean(l2_loss),
        'train_energy': clean(energy_loss),
        'train_mr_stft': clean(mr_stft),
        'fargan_signal': clean(signal_loss),
        'fargan_continuity': clean(continuity_loss),
        'fargan_pitch_consistency': clean(pitch_loss),
        'fargan_subframe_alignment': clean(subframe_loss),
        'fargan_train_total': clean(total),
    }

    return loss_dict['fargan_train_total'], loss_dict


## 测试函数移除：避免模块内执行与冗余输出
