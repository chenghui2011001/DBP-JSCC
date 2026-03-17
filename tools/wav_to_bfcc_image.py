#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将单个音频文件转换为 32 维对数 BFCC 图像并保存为 PNG，
同时导出原始波形图像。

用法示例：

    python tools/wav_to_bfcc_image.py \
        --input path/to/audio.wav \
        --out_dir path/to/output_dir

默认使用 16 kHz 采样率、25 ms 窗长 (n_fft=400)、10 ms 帧移 (hop=160)、32 个 Bark 频带，
内部复用公开模型中的 WaveToBFCC，保证与训练流程一致。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")  # 非交互式后端，适合脚本保存图像
import matplotlib.pyplot as plt

import torch
from scipy.io import wavfile
from scipy import signal


def _add_repo_root_to_path() -> None:
    """将仓库根目录加入 sys.path 以便导入本地模块。"""

    repo_root = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


_add_repo_root_to_path()

# 为避免导入 torchaudio（当前环境下会报错），
# 这里复制公开模型入口中的 WaveToBFCC 实现，
# 只保留与 BFCC 计算相关的逻辑。
from torch.cuda.amp import autocast as amp_autocast  # noqa: E402

try:  # 可选：复用 utils.audio_visualizer 中的 F0 提取逻辑
    from utils.audio_visualizer import extract_f0 as _viz_extract_f0  # type: ignore[attr-defined]

    _HAS_F0_VIS = True
except Exception:
    _HAS_F0_VIS = False


class WaveToBFCC(torch.nn.Module):
    """波形 → Bark 滤波器组对数能量图 [B,T,n_bands]。

    该实现与公开训练模型中的 WaveToBFCC 保持一致，
    但去掉了对 torchaudio 等依赖，方便在轻量环境下独立使用。
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_bands: int = 32,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop = hop_length
        self.n_bands = n_bands

        win = torch.hann_window(n_fft, periodic=True)
        self.register_buffer("win", win)

        # 构建 Bark 三角滤波器组权重 [Fbins, n_bands]
        Fbins = n_fft // 2 + 1
        freqs = torch.linspace(0, sample_rate / 2, Fbins)

        def hz_to_bark(f: torch.Tensor) -> torch.Tensor:
            return 13.0 * torch.atan(0.00076 * f) + 3.5 * torch.atan((f / 7500.0) ** 2)

        z = hz_to_bark(freqs)
        z_min, z_max = float(z.min()), float(z.max())
        centers = torch.linspace(z_min, z_max, steps=n_bands)
        # 边界使用相邻中心的中点，首尾用全域边界
        edges = torch.zeros(n_bands + 1)
        if n_bands > 1:
            edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
        edges[0] = z_min
        edges[-1] = z_max
        W = torch.zeros(Fbins, n_bands)
        for j in range(n_bands):
            left = edges[j]
            c = centers[j]
            right = edges[j + 1]
            lj = (z >= left) & (z <= c)
            rj = (z > c) & (z <= right)
            if (c - left) > 1e-6:
                W[lj, j] = (z[lj] - left) / (c - left)
            if (right - c) > 1e-6:
                W[rj, j] = (right - z[rj]) / (right - c)
        W = W / (W.sum(dim=0, keepdim=True) + 1e-8)
        self.register_buffer("bark_w", W)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Waveform → Bark log-energy [B,T,n_bands] in float32。"""

        with amp_autocast(enabled=False):
            audio_f32 = audio.to(dtype=torch.float32)
            # 去除直流分量，避免 DC/极低频漂移在第 0 Bark 带上持续堆积
            if audio_f32.dim() >= 2:
                audio_f32 = audio_f32 - audio_f32.mean(dim=-1, keepdim=True)
            win = self.win.to(device=audio_f32.device, dtype=torch.float32)
            bark_w = self.bark_w.to(device=audio_f32.device, dtype=torch.float32)

            # STFT: [B,F,T]
            X = torch.stft(
                audio_f32,
                n_fft=self.n_fft,
                hop_length=self.hop,
                win_length=self.n_fft,
                window=win,
                center=False,
                return_complex=True,
            )
            mag2 = X.real.pow(2) + X.imag.pow(2)  # [B,F,T]
            E = torch.matmul(mag2.transpose(1, 2), bark_w)  # [B,T,n_bands]
            logE = torch.log10(E + 1e-10)
        return logE


def load_audio_mono_16k(path: str | Path, target_sr: int = 16000) -> np.ndarray:
    """读取 WAV 音频并转换为 mono 16 kHz 的 numpy 数组。

    仅依赖 scipy.io.wavfile，避免额外第三方依赖。

    Args:
        path: 输入音频路径（.wav）。
        target_sr: 目标采样率，默认 16 kHz。

    Returns:
        形状为 [L] 的 float32 numpy 数组，范围约 [-1, 1]。
    """

    path = str(path)
    sr, audio = wavfile.read(path)

    # 转为 float32
    if np.issubdtype(audio.dtype, np.integer):
        max_int = np.iinfo(audio.dtype).max
        audio = audio.astype(np.float32) / max(max_int, 1)
    else:
        audio = audio.astype(np.float32)

    # 多通道转单声道
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    # 需要重采样时，使用 scipy.signal.resample
    if sr != target_sr:
        num_samples = int(round(len(audio) * float(target_sr) / float(sr)))
        if num_samples <= 0:
            raise ValueError(f"无效的重采样长度: {num_samples} (sr={sr}, target_sr={target_sr})")
        audio = signal.resample(audio, num_samples).astype(np.float32)
        sr = target_sr

    # 再做一次安全归一化，避免极端值
    max_abs = float(np.max(np.abs(audio)) + 1e-9)
    audio = audio / max_abs
    return audio


def compute_bfcc_32(audio_np: np.ndarray,
                    sample_rate: int = 16000,
                    n_fft: int = 400,
                    hop_length: int = 160,
                    n_bands: int = 32) -> np.ndarray:
    """使用 WaveToBFCC 计算 32 维对数 BFCC（Bark 滤波器组能量）。

    Args:
        audio_np: [L] 的单声道波形 (numpy 数组)。
        sample_rate: 采样率（默认 16 kHz）。
        n_fft: STFT 窗长。
        hop_length: 帧移。
        n_bands: Bark 频带数，默认 32。

    Returns:
        [T, 32] 的 numpy 数组，对应时间帧 × 频带的对数能量。
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # [1, L]
    audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).to(device)

    model = WaveToBFCC(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_bands=n_bands,
    ).to(device)

    model.eval()
    with torch.no_grad():
        bfcc = model(audio_tensor)  # [1, T, 32]

    bfcc_np = bfcc.squeeze(0).cpu().numpy()  # [T, 32]
    return bfcc_np


def compute_f0_and_vuv(
    audio_np: np.ndarray,
    sample_rate: int = 16000,
    hop_length: int = 160,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """使用项目内的 F0 提取器计算 F0 轮廓和 VUV 掩膜。

    若 ``utils.audio_visualizer.extract_f0`` 无法导入，则抛出 RuntimeError。
    """

    if not _HAS_F0_VIS:
        raise RuntimeError("F0 extractor not available (utils.audio_visualizer import failed)")

    # audio_visualizer.extract_f0 期望 [L] 的 torch.Tensor
    audio_tensor = torch.from_numpy(audio_np.astype(np.float32))
    f0_hz, vmask, _f0_fb, _fb_mask, estimator = _viz_extract_f0(
        audio_tensor,
        sr=sample_rate,
        hop_length=hop_length,
    )

    # 强制为 1D numpy 数组和布尔掩膜
    f0_hz = np.asarray(f0_hz, dtype=np.float32).reshape(-1)
    vuv_mask = np.asarray(vmask, dtype=bool).reshape(-1)
    # 对齐长度
    T_use = min(len(f0_hz), len(vuv_mask))
    return f0_hz[:T_use], vuv_mask[:T_use], str(estimator)


def downsample_time(bfcc: np.ndarray, factor: int = 2, mode: str = "average") -> np.ndarray:
    """沿时间轴对 BFCC 进行降采样。

    Args:
        bfcc: [T, n_bands] 的 BFCC 数组。
        factor: 降采样因子（默认 2）。
        mode: 降采样模式，"average" 取相邻帧平均，"skip" 直接跳帧。

    Returns:
        [T//factor, n_bands] 的降采样 BFCC 数组。
    """
    if factor <= 1:
        return bfcc

    T, n_bands = bfcc.shape
    T_new = T // factor

    if mode == "average":
        # 相邻 factor 帧取平均
        bfcc_truncated = bfcc[: T_new * factor, :]  # 截断到整除长度
        bfcc_reshaped = bfcc_truncated.reshape(T_new, factor, n_bands)
        bfcc_ds = bfcc_reshaped.mean(axis=1)
    else:
        # 跳帧：每隔 factor 帧取一帧
        bfcc_ds = bfcc[::factor, :]

    return bfcc_ds


def save_bfcc_image(bfcc: np.ndarray,
                    output_path: str | Path,
                    sample_rate: int = 16000,
                    hop_length: int = 160,
                    cmap: str = "magma",
                    title_suffix: str = "") -> None:
    """将 [T, 32] 的对数 BFCC 保存为 PNG 图像。

    时间轴在横向，频带在纵向。
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # [freq, time] 以便更直观：频率向上，时间向右
    bfcc_T = bfcc.T  # [32, T]

    n_frames = bfcc_T.shape[1]
    duration_sec = n_frames * hop_length / float(sample_rate)

    vmin = float(np.percentile(bfcc_T, 1))
    vmax = float(np.percentile(bfcc_T, 99))

    plt.figure(figsize=(10, 4))
    extent = [0.0, duration_sec, 0, bfcc_T.shape[0]]
    img = plt.imshow(
        bfcc_T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
    )
    plt.colorbar(img, label="log10 energy")
    plt.xlabel("Time (s)")
    plt.ylabel("Bark band index (0-31)")
    title = "32-dim log BFCC (Bark log-energy)"
    if title_suffix:
        title += f" {title_suffix}"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_f0_image(
    f0_hz: np.ndarray,
    vuv_mask: np.ndarray,
    output_path: str | Path,
    sample_rate: int = 16000,
    hop_length: int = 160,
    estimator: str | None = None,
) -> None:
    """保存 F0 轮廓图。

    仅在有声帧上绘制 F0 曲线，无声帧以灰色区域表示。
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if f0_hz.ndim != 1:
        f0_hz = f0_hz.reshape(-1)
    if vuv_mask.ndim != 1:
        vuv_mask = vuv_mask.reshape(-1)

    T = min(len(f0_hz), len(vuv_mask))
    if T == 0:
        raise ValueError("F0 序列长度为 0，无法绘制图像。")

    f0_hz = f0_hz[:T]
    vuv_mask = vuv_mask[:T]

    time_f0 = np.arange(T, dtype=np.float32) * float(hop_length) / float(sample_rate)
    f0_plot = np.where(vuv_mask, f0_hz, np.nan)

    voiced_ratio = float(vuv_mask.mean()) if T > 0 else 0.0
    f0_valid = f0_hz[vuv_mask]
    f0_mean = float(f0_valid.mean()) if f0_valid.size > 0 else 0.0
    f0_std = float(f0_valid.std()) if f0_valid.size > 0 else 0.0

    plt.figure(figsize=(10, 4))
    plt.plot(time_f0, f0_plot, "b-", linewidth=1.5, label="F0 (voiced)")
    if (~vuv_mask).any():
        plt.fill_between(
            time_f0,
            f0_hz.min(initial=50.0),
            f0_hz.max(initial=500.0),
            where=(~vuv_mask),
            color="lightgray",
            alpha=0.2,
            step="pre",
            label="unvoiced",
        )

    est_note = f", est={estimator}" if estimator else ""
    plt.title(
        f"F0 contour{est_note} (voiced={voiced_ratio:.1%}, "
        f"mean={f0_mean:.1f}Hz, std={f0_std:.1f}Hz",
    )
    plt.xlabel("Time (s)")
    plt.ylabel("F0 (Hz)")
    plt.ylim(50, 500)
    plt.xlim(float(time_f0[0]), float(time_f0[-1]) if time_f0[-1] > 0 else 0.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_vuv_image(
    vuv_mask: np.ndarray,
    output_path: str | Path,
    sample_rate: int = 16000,
    hop_length: int = 160,
) -> None:
    """保存 VUV（二值有声/无声）图像。"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if vuv_mask.ndim != 1:
        vuv_mask = vuv_mask.reshape(-1)

    T = len(vuv_mask)
    if T == 0:
        raise ValueError("VUV 序列长度为 0，无法绘制图像。")

    time_v = np.arange(T, dtype=np.float32) * float(hop_length) / float(sample_rate)
    vuv_float = vuv_mask.astype(np.float32)

    plt.figure(figsize=(10, 2.5))
    plt.step(time_v, vuv_float, where="post", linewidth=1.5)
    plt.ylim(-0.1, 1.1)
    plt.yticks([0.0, 1.0], ["unvoiced", "voiced"])
    plt.xlabel("Time (s)")
    plt.title("Voiced/Unvoiced (VUV) mask")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_waveform_image(audio: np.ndarray,
                        output_path: str | Path,
                        sample_rate: int = 16000) -> None:
    """将原始波形保存为 PNG 图像。

    时间轴为横坐标（秒），幅度为纵坐标。
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if audio.ndim != 1:
        audio = audio.reshape(-1)

    num_samples = len(audio)
    if num_samples == 0:
        raise ValueError("音频长度为 0，无法绘制波形图。")

    t = np.arange(num_samples, dtype=np.float32) / float(sample_rate)

    plt.figure(figsize=(10, 3))
    plt.plot(t, audio, linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform")
    plt.xlim(t[0], t[-1])
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将音频转换为 32 维对数 BFCC 图像",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="输入音频文件路径（例如 WAV）",
    )
    parser.add_argument(
        "--out_dir",
        "-o",
        type=str,
        default=None,
        help="输出图像目录；若省略，则与输入音频同目录",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=16000,
        help="目标采样率（默认 16000）",
    )
    parser.add_argument(
        "--n_fft",
        type=int,
        default=400,
        help="STFT 窗长 n_fft（默认 400，即 25 ms@16kHz）",
    )
    parser.add_argument(
        "--hop",
        type=int,
        default=160,
        help="帧移 hop_length（默认 160，即 10 ms@16kHz）",
    )
    parser.add_argument(
        "--n_bands",
        type=int,
        default=32,
        help="Bark 频带数量（默认 32）",
    )
    parser.add_argument(
        "--wave-output",
        "-w",
        type=str,
        default=None,
        help="原始波形图输出路径；若省略，则在输出目录中生成 *_waveform.png",
    )
    parser.add_argument(
        "--no_f0_vuv",
        action="store_true",
        help="不额外生成 F0 和 VUV 图像（默认生成，若依赖可用）",
    )
    parser.add_argument(
        "--time_downsample",
        "-t",
        type=int,
        default=1,
        help="时间轴降采样因子（默认 1 表示不降采样，2 表示两倍降采样）",
    )
    parser.add_argument(
        "--ds_mode",
        type=str,
        default="average",
        choices=["average", "skip"],
        help="降采样模式：average（相邻帧平均）或 skip（跳帧）",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"输入音频不存在: {input_path}")

    # 输出目录
    if args.out_dir is None:
        out_dir = input_path.parent
    else:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    # 默认文件名基于输入音频名
    bfcc_filename = input_path.stem + "_bfcc32.png"
    wave_filename = input_path.stem + "_waveform.png"
    f0_filename = input_path.stem + "_f0.png"
    vuv_filename = input_path.stem + "_vuv.png"

    output_path = out_dir / bfcc_filename

    if args.wave_output is None:
        wave_output_path = out_dir / wave_filename
    else:
        wave_output_path = Path(args.wave_output)

    print(f"[BFCC] 读取音频: {input_path}")
    audio_np = load_audio_mono_16k(input_path, target_sr=args.sr)

    print(f"[WAVE] 保存波形图到: {wave_output_path}")
    save_waveform_image(
        audio_np,
        output_path=wave_output_path,
        sample_rate=args.sr,
    )

    print(f"[BFCC] 计算 32 维对数 BFCC (Bark log-energy)...")
    bfcc = compute_bfcc_32(
        audio_np,
        sample_rate=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop,
        n_bands=args.n_bands,
    )

    print(f"[BFCC] 保存图像到: {output_path}")
    save_bfcc_image(
        bfcc,
        output_path=output_path,
        sample_rate=args.sr,
        hop_length=args.hop,
    )

    # 如果指定了时间降采样，生成额外的降采样图像
    if args.time_downsample > 1:
        bfcc_ds = downsample_time(bfcc, factor=args.time_downsample, mode=args.ds_mode)
        ds_filename = input_path.stem + f"_bfcc32_ds{args.time_downsample}.png"
        ds_output_path = out_dir / ds_filename
        # 降采样后的等效 hop_length
        effective_hop = args.hop * args.time_downsample
        print(f"[BFCC] 时间轴 {args.time_downsample}x 降采样 ({args.ds_mode}): {bfcc.shape[0]} -> {bfcc_ds.shape[0]} 帧")
        print(f"[BFCC] 保存降采样图像到: {ds_output_path}")
        save_bfcc_image(
            bfcc_ds,
            output_path=ds_output_path,
            sample_rate=args.sr,
            hop_length=effective_hop,
            title_suffix=f"[{args.time_downsample}x downsample]",
        )

    # 可选：生成 F0 与 VUV 图像
    if not args.no_f0_vuv:
        if not _HAS_F0_VIS:
            print("[F0/VUV] utils.audio_visualizer 无法导入，跳过 F0/VUV 图像生成。")
        else:
            try:
                print("[F0/VUV] 提取 F0 与 VUV 掩膜...")
                f0_hz, vuv_mask, estimator = compute_f0_and_vuv(
                    audio_np,
                    sample_rate=args.sr,
                    hop_length=args.hop,
                )
                f0_path = out_dir / f0_filename
                vuv_path = out_dir / vuv_filename
                print(f"[F0] 保存 F0 图像到: {f0_path}")
                save_f0_image(
                    f0_hz,
                    vuv_mask,
                    output_path=f0_path,
                    sample_rate=args.sr,
                    hop_length=args.hop,
                    estimator=estimator,
                )
                print(f"[VUV] 保存 VUV 图像到: {vuv_path}")
                save_vuv_image(
                    vuv_mask,
                    output_path=vuv_path,
                    sample_rate=args.sr,
                    hop_length=args.hop,
                )
            except Exception as exc:
                print(f"[F0/VUV] 生成 F0/VUV 图像失败: {exc}")

    print("[BFCC] 完成。")


if __name__ == "__main__":
    main()
