#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export JSCC HashBottleneck bits for a single PCM+feature pair.

This script is designed to be called from the Fargan_sim GUI and assumes
that:

- ``--pcm`` points to a 16 kHz mono 16-bit PCM file (e.g. Fargan_sim/input.pcm)
- ``--features`` points to the matching 36-dim FARGAN features (.f32)

It runs a single forward pass through the Stage2.5 model (with_hash=True)
and exports either the content-branch hash bits (``hash_bits_clean``) or
the F0-branch hash bits (``f0_hash_bits_clean``) as a 1D numpy array. The
array elements are ±1 (float32) and can be further processed by the
Fargan_sim FSK modem.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import time

import numpy as np
import torch
import subprocess
import shutil
import json


class ExternalSnrChannelSim:
    """Lightweight CSI adapter driven only by external SNR.

    This class mimics the ``ChannelSimulator`` interface used during
    training, but it:

    - derives CSI scalars directly from the provided SNR range
      (snr_min_db / snr_max_db);
    - returns a constant per-frame SNR trajectory and unit amplitude
      envelope;
    - does *not* apply any additional fading/noise in ``apply``.

    It is intended for offline JSCC+FSK inference where the physical
    channel is modelled explicitly outside Aether-lite, and the JSCC
    model should only see a CSI proxy derived from that external SNR.
    """

    def __init__(
        self,
        default_time_selectivity: float = 0.5,
        default_freq_selectivity: float = 0.5,
        default_los_ratio: float = 1.0,
    ) -> None:
        self.default_time_selectivity = float(default_time_selectivity)
        self.default_freq_selectivity = float(default_freq_selectivity)
        self.default_los_ratio = float(default_los_ratio)

    def sample_csi(
        self,
        B: int,
        T: int,
        channel: str = "fading",
        snr_min_db: float | None = None,
        snr_max_db: float | None = None,
    ):
        import torch as _torch

        device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
        dtype = _torch.float32

        if snr_min_db is None and snr_max_db is None:
            snr_db = 0.0
        elif snr_min_db is None:
            snr_db = float(snr_max_db)
        elif snr_max_db is None:
            snr_db = float(snr_min_db)
        else:
            lo = float(snr_min_db)
            hi = float(snr_max_db)
            if hi < lo:
                lo, hi = hi, lo
            snr_db = 0.5 * (lo + hi)

        snr_proxy = _torch.full((B,), snr_db, device=device, dtype=dtype)
        time_selectivity = _torch.full(
            (B,), self.default_time_selectivity, device=device, dtype=dtype
        )
        freq_selectivity = _torch.full(
            (B,), self.default_freq_selectivity, device=device, dtype=dtype
        )
        los_ratio = _torch.full(
            (B,), self.default_los_ratio, device=device, dtype=dtype
        )

        csi = {
            "snr_proxy": snr_proxy,
            "time_selectivity": time_selectivity,
            "freq_selectivity": freq_selectivity,
            "los_ratio": los_ratio,
        }

        amp_t = _torch.ones(B, T, device=device, dtype=dtype)
        snr_db_t = _torch.full((B, T), snr_db, device=device, dtype=dtype)

        return csi, amp_t, snr_db_t

    def apply(self, z, amp_t, snr_db_t, **kwargs):  # noqa: D401
        """Identity mapping: do not add any additional noise/fading."""

        return z


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export JSCC HashBottleneck bits for a single sample")

    p.add_argument("--ckpt", type=str, required=True, help="Stage2.5 checkpoint (.pth)")
    p.add_argument(
        "--pcm",
        type=str,
        default="../Fargan_sim/input.pcm",
        help="Path to 16 kHz mono 16-bit PCM file or WAV file",
    )
    p.add_argument(
        "--features",
        type=str,
        default=None,
        help=(
            "Optional path to precomputed 36-dim vocoder features (.f32). "
            "If omitted, features are extracted from --pcm via the external extractor "
            "and saved to --out_features (or next to --out_bits)."
        ),
    )
    p.add_argument(
        "--out_bits",
        type=str,
        default="../Fargan_sim/jscc_bits.npy",
        help="Output .npy path for exported hash bits (1D ±1 float32)",
    )
    p.add_argument(
        "--feature_bin",
        type=str,
        default=None,
        help=(
            "Optional path to an external feature extractor. "
            "Compatible with dump_data -train and fargan_demo -features."
        ),
    )
    p.add_argument(
        "--out_features",
        type=str,
        default=None,
        help=(
            "Optional path to save/copy 36-dim vocoder features (.f32). "
            "If not set, a copy is written next to --out_bits as 'features.f32'."
        ),
    )
    p.add_argument(
        "--out_pcm_aligned",
        type=str,
        default=None,
        help=(
            "Optional path to save the aligned 16 kHz mono 16-bit PCM used for JSCC "
            "(after padding/truncation to a multiple of sequence_length*160). "
            "If not set, a file named 'input_aligned.pcm' is written next to --out_bits."
        ),
    )
    p.add_argument(
        "--branch",
        type=str,
        default="both",
        choices=["content", "f0", "both"],
        help=(
            "Which hash branch to export: 'content', 'f0', or 'both'. "
            "When 'both', content and F0/VUV bits are concatenated into one "
            "stream and metadata is saved alongside."
        ),
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (e.g. 'cpu' or 'cuda'); default: auto",
    )
    p.add_argument("--snr_min_db", type=float, default=-5.0, help="SNR min (dB) for ChannelSimulator")
    p.add_argument("--snr_max_db", type=float, default=15.0, help="SNR max (dB) for ChannelSimulator")
    p.add_argument(
        "--csi_mode",
        type=str,
        default="channel_sim",
        choices=["channel_sim", "external_snr", "real_noise"],
        help=(
            "CSI generation mode: 'channel_sim' uses the training-time "
            "ChannelSimulator, 'external_snr' derives CSI directly from "
            "the provided SNR without additional channel simulation, "
            "and 'real_noise' uses RealNoiseChannelSimulator driven by "
            "an external noise CSV."
        ),
    )
    p.add_argument(
        "--noise_csv",
        type=str,
        default=None,
        help=(
            "Noise CSV path used when --csi_mode=real_noise to drive the "
            "RealNoiseChannelSimulator. Ignored otherwise."
        ),
    )
    p.add_argument(
        "--seq_length",
        type=int,
        default=None,
        help=(
            "Optional max sequence length in frames (10 ms/frame). "
            "If set, only the first seq_length frames (and corresponding audio) "
            "are processed."
        ),
    )
    p.add_argument(
        "--bit_only",
        action="store_true",
        help=(
            "Use training-style bits-only path: encode_hash_codec with noisy bits "
            "and decode_from_bits_offline-compatible bitstreams. This ignores "
            "--csi_mode/--snr_min_db/--snr_max_db and uses cfg.snr_min_db/max_db."
        ),
    )

    p.add_argument(
        "--pcm_list",
        type=str,
        default=None,
        help=(
            "Optional text file with one PCM/WAV path per line. When provided, "
            "this script runs in batch mode: it invokes itself once per input "
            "and concatenates all exported bitstreams into a single --out_bits "
            "file. Individual per-sample jscc_bits*.npy files are treated as "
            "temporary."
        ),
    )

    return p.parse_args()


def _load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict):
        raise RuntimeError("Unexpected checkpoint format: expected dict")
    return ckpt


def _safe_load_model_state(model: torch.nn.Module, state_dict: Dict[str, Any]) -> None:
    msd = model.state_dict()
    safe_sd: Dict[str, torch.Tensor] = {}
    dropped = []
    for k, v in state_dict.items():
        if k in msd and isinstance(v, torch.Tensor) and v.shape == msd[k].shape:
            safe_sd[k] = v
        else:
            dropped.append(k)
    if dropped:
        print(f"[ExportBits] Dropped {len(dropped)} key(s) due to shape mismatch; examples: {dropped[:5]}")
    load_ret = model.load_state_dict(safe_sd, strict=False)
    try:
        missing = list(getattr(load_ret, "missing_keys", []))
        unexpected = list(getattr(load_ret, "unexpected_keys", []))
    except Exception:
        missing, unexpected = [], []
    if missing or unexpected:
        print(
            f"[ExportBits] Non-strict load: missing={len(missing)}, "
            f"unexpected={len(unexpected)}",
        )


def _load_pcm_or_wav_as_int16(path: str, expect_sr: int = 16000) -> np.ndarray:
    """Load 16-bit mono audio as int16 from .pcm or .wav.

    - For ``.pcm``: interpret as raw 16-bit little-endian mono.
    - For ``.wav``/``.wave``: use Python ``wave`` module, require mono
      16-bit, and ``framerate == expect_sr``.
    """

    ext = os.path.splitext(path)[1].lower()
    if ext in (".wav", ".wave"):
        import wave

        with wave.open(path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()

            if n_channels != 1:
                raise RuntimeError(
                    f"WAV must be mono (1 channel), got {n_channels} channels: {path}"
                )
            if sampwidth != 2:
                raise RuntimeError(
                    f"WAV must be 16-bit PCM (sampwidth=2), got {sampwidth}: {path}"
                )
            if framerate != expect_sr:
                raise RuntimeError(
                    f"WAV sample rate must be {expect_sr} Hz, got {framerate} Hz: {path}"
                )

            raw = wf.readframes(n_frames)
            pcm = np.frombuffer(raw, dtype=np.int16)
        return pcm

    # 默认按裸 PCM 读取
    return np.fromfile(path, dtype=np.int16)


def _run_batch(args: argparse.Namespace) -> int:
    """Batch mode: process multiple PCM paths and concatenate bits.

    This mode is activated when ``--pcm_list`` is provided. The list
    file should contain one PCM/WAV path per line (empty lines and
    lines starting with ``#`` are ignored). For each entry we invoke
    this script as a subprocess (single-sample mode) to produce a
    temporary ``.npy`` file, then load and concatenate all bitstreams
    into ``args.out_bits``.

    The rationale for using subprocesses (instead of refactoring the
    entire single-sample path) is to keep the existing, battle-tested
    logic intact while still offering a convenient CLI wrapper for
    generating a long JSCC bitstream from multiple utterances.
    """

    import subprocess as _sub
    import tempfile as _tmp

    list_path = Path(args.pcm_list).expanduser()
    if not list_path.is_file():
        raise FileNotFoundError(f"pcm_list file not found: {list_path}")

    pcm_paths: list[Path] = []
    with list_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            pcm_paths.append(Path(s).expanduser())

    if not pcm_paths:
        raise RuntimeError(f"pcm_list contains no valid paths: {list_path}")

    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent

    # Temporary directory to hold per-sample jscc_bits .npy files
    tmp_dir = Path(_tmp.mkdtemp(prefix="jscc_bits_batch_"))
    bits_all: list[np.ndarray] = []

    print(
        f"[ExportBits-Batch] Processing {len(pcm_paths)} PCM file(s) from {list_path} "
        f"→ concatenated bits at {args.out_bits}"
    )

    for idx, pcm_path in enumerate(pcm_paths):
        tmp_bits = tmp_dir / f"jscc_bits_{idx:04d}.npy"

        cmd: list[str] = [
            sys.executable,
            str(script_path),
            "--ckpt",
            args.ckpt,
            "--pcm",
            str(pcm_path),
            "--out_bits",
            str(tmp_bits),
        ]

        # Forward relevant options from the batch invocation.
        if args.features is not None:
            cmd.extend(["--features", args.features])
        if args.out_features is not None:
            cmd.extend(["--out_features", args.out_features])
        if args.out_pcm_aligned is not None:
            cmd.extend(["--out_pcm_aligned", args.out_pcm_aligned])
        if args.branch is not None:
            cmd.extend(["--branch", args.branch])
        if args.device is not None:
            cmd.extend(["--device", args.device])
        cmd.extend(["--snr_min_db", str(args.snr_min_db)])
        cmd.extend(["--snr_max_db", str(args.snr_max_db)])
        if getattr(args, "csi_mode", None) is not None:
            cmd.extend(["--csi_mode", args.csi_mode])
        if getattr(args, "noise_csv", None):
            cmd.extend(["--noise_csv", args.noise_csv])
        if args.seq_length is not None:
            cmd.extend(["--seq_length", str(int(args.seq_length))])
        if getattr(args, "bit_only", False):
            cmd.append("--bit_only")

        print(f"[ExportBits-Batch] ({idx+1}/{len(pcm_paths)}) CMD: {' '.join(cmd)}")
        _sub.run(cmd, check=True, cwd=str(repo_root))

        if not tmp_bits.is_file():
            raise RuntimeError(f"Child export did not produce bits file: {tmp_bits}")
        arr = np.load(str(tmp_bits)).astype(np.float32).reshape(-1)
        bits_all.append(arr)

    if not bits_all:
        raise RuntimeError("No bits produced in batch mode")

    concat = np.concatenate(bits_all, axis=0).astype(np.float32)
    out_dir = os.path.dirname(args.out_bits) or "."
    os.makedirs(out_dir, exist_ok=True)
    np.save(args.out_bits, concat)
    print(
        f"[ExportBits-Batch] Saved concatenated JSCC bits to {args.out_bits} "
        f"total_bits={concat.size} from {len(bits_all)} file(s)"
    )

    # Best-effort cleanup of temporary directory
    try:
        import shutil as _sh

        _sh.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass

    return 0


def main() -> int:
    args = _parse_args()

    # Batch mode: when --pcm_list is provided, run a light-weight wrapper
    # that invokes this script once per PCM and concatenates all
    # bitstreams into a single jscc_bits.npy specified by --out_bits.
    if getattr(args, "pcm_list", None):
        return _run_batch(args)

    t_total_start = time.time()

    t_total_start = time.time()
    t_model = t_io = t_forward = t_post = 0.0

    # 计时：模型构建阶段
    t_model_start = time.time()

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from training.train_support import build_model  # type: ignore
    from utils.channel_sim import ChannelSimulator, RealNoiseChannelSimulator  # type: ignore
    from utils.feature_extraction import resolve_feature_extractor, run_feature_extractor  # type: ignore

    auto_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(args.device) if args.device is not None else auto_device

    ckpt = _load_checkpoint(args.ckpt, device=device)
    if "cfg" not in ckpt:
        raise RuntimeError("Checkpoint does not contain 'cfg'; please use Stage2.5 checkpoint")
    cfg_dict: Dict[str, Any] = ckpt["cfg"]
    if not isinstance(cfg_dict, dict):
        raise RuntimeError("Checkpoint field 'cfg' must be a dict")

    class _ShimCfg:
        pass

    cfg = _ShimCfg()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)

    qtype = str(getattr(cfg, "quantizer_type", "hash"))
    if qtype not in ("hash", "rvq"):
        raise RuntimeError(f"Unsupported quantizer_type={qtype}; expected 'hash' or 'rvq'")

    cfg.device = str(device)

    print(f"[ExportBits] Building model on device={device} ...")
    model = build_model(cfg).to(device)
    model.eval()
    state_dict = None
    for key in ("model_state_dict", "model", "state_dict"):
        if key in ckpt:
            state_dict = ckpt[key]
            break
    if state_dict is None:
        raise RuntimeError("Checkpoint missing model weights")
    _safe_load_model_state(model, state_dict)

    if getattr(args, "bit_only", False):
        # bits-only 路径：使用原始 ChannelSimulator，并允许外层通过
        # --snr_min_db/--snr_max_db 控制信道 SNR（默认仍回退到 cfg）。
        channel_sim = ChannelSimulator(sample_rate=16000, frame_hz=100)
        print(
            f"[ExportBits] bit_only mode: ChannelSimulator with snr_min_db={args.snr_min_db} "
            f"snr_max_db={args.snr_max_db} (cfg.snr=[{getattr(cfg, 'snr_min_db', -5.0)}, {getattr(cfg, 'snr_max_db', 15.0)}])"
        )
    elif getattr(args, "csi_mode", "channel_sim") == "external_snr":
        channel_sim = ExternalSnrChannelSim()
        print("[ExportBits] Using ExternalSnrChannelSim (CSI from external SNR)")
    elif getattr(args, "csi_mode", "channel_sim") == "real_noise":
        if not args.noise_csv:
            raise ValueError(
                "--noise_csv must be provided when --csi_mode=real_noise"
            )
        noise_csv_path = args.noise_csv
        if not os.path.isabs(noise_csv_path):
            noise_csv_path = os.path.join(Path(repo_root).parent, noise_csv_path)
        if not os.path.exists(noise_csv_path):
            raise FileNotFoundError(
                f"Noise CSV not found for RealNoiseChannelSimulator: {noise_csv_path}"
            )
        channel_sim = RealNoiseChannelSimulator(
            noise_csv=noise_csv_path,
            sample_rate=16000,
            frame_hz=100,
        )
        print(
            f"[ExportBits] Using RealNoiseChannelSimulator with noise_csv={noise_csv_path}"
        )
    else:
        channel_sim = ChannelSimulator(sample_rate=16000, frame_hz=100)
        print("[ExportBits] Using training-style ChannelSimulator for CSI")

    t_model = time.time() - t_model_start

    # 计时：I/O + 特征处理
    t_io_start = time.time()

    # Load PCM/WAV (16 kHz, 16-bit mono)
    if not os.path.isfile(args.pcm):
        raise FileNotFoundError(f"PCM/WAV file not found: {args.pcm}")
    pcm = _load_pcm_or_wav_as_int16(args.pcm, expect_sr=16000)
    if pcm.size == 0:
        raise RuntimeError("PCM file is empty")
    audio = torch.from_numpy(pcm.astype(np.float32) / 32768.0).unsqueeze(0).to(device)  # [1,L]

    # 若输入为 WAV，则为 fargan_demo 生成一个临时 PCM 文件；
    # 若本身已经是 .pcm，则直接使用原路径。
    pcm_path_for_fargan = args.pcm
    ext = os.path.splitext(args.pcm)[1].lower()
    if ext in (".wav", ".wave"):
        pcm_dir = os.path.dirname(args.pcm) or "."
        pcm_base = os.path.splitext(os.path.basename(args.pcm))[0] + ".pcm"
        pcm_tmp_path = os.path.join(pcm_dir, pcm_base)
        try:
            pcm.tofile(pcm_tmp_path)
        except Exception as exc:  # pragma: no cover - best-effort IO
            raise RuntimeError(
                f"Failed to write temporary PCM for fargan_demo: {pcm_tmp_path} ({exc})"
            ) from exc
        pcm_path_for_fargan = pcm_tmp_path
        print(f"[ExportBits] Wrote temporary PCM for fargan_demo: {pcm_path_for_fargan}")

    # 决定主特征文件路径：
    #   1) 优先使用 --features（外部已准备好的 .f32）
    #   2) 其次使用 --out_features（自动提取并保存到这里）
    #   3) 否则默认写在 --out_bits 同目录下的 features.f32
    if args.features is not None:
        features_path = args.features
    elif args.out_features is not None:
        features_path = args.out_features
    else:
        out_dir_for_bits = os.path.dirname(args.out_bits) or "."
        features_path = os.path.join(out_dir_for_bits, "features.f32")

    aligned_pcm_path = args.out_pcm_aligned
    if not aligned_pcm_path:
        out_dir_for_bits = os.path.dirname(args.out_bits) or "."
        aligned_pcm_path = os.path.join(out_dir_for_bits, "input_aligned.pcm")

    # 是否需要通过外部特征提取器提取特征：
    #   - 若用户显式提供 --features，则只在文件缺失时尝试自动提取；
    #   - 若未提供 --features（即让脚本托管特征路径），则每次都从 PCM 重新提取，
    #     覆盖已有的 features_path，以保证始终是当前语音对应的“原始特征”。
    need_extract = False
    if args.features is None:
        need_extract = True
    elif not os.path.isfile(features_path):
        need_extract = True

    # Load features (36-dim vocoder features)
    if need_extract:
        # 若特征文件不存在或需要强制刷新，调用兼容提取器自动提取：
        print(
            f"[ExportBits] Extracting 36D vocoder features via external tool → {features_path}"
        )
        extractor_bin = resolve_feature_extractor(args.feature_bin)
        extract_meta = run_feature_extractor(
            input_pcm=Path(pcm_path_for_fargan),
            output_features=Path(features_path),
            output_pcm=Path(aligned_pcm_path),
            extractor_bin=extractor_bin,
        )
        print(
            "[ExportBits] Feature extraction finished: "
            f"mode={extract_meta['extractor_mode']}, frames={extract_meta['feature_frames']}, "
            f"duration={extract_meta['duration_s']:.2f}s",
        )
        pcm = np.fromfile(aligned_pcm_path, dtype=np.int16)
        audio = torch.from_numpy(pcm.astype(np.float32) / 32768.0).unsqueeze(0).to(device)
    elif not os.path.isfile(features_path):
        # 用户显式指定了 --features 但文件不存在，且未启用自动提取
        raise FileNotFoundError(f"Features file not found: {features_path}")

    feats_flat = np.fromfile(features_path, dtype=np.float32)
    if feats_flat.size % 36 != 0:
        raise RuntimeError(f"features.f32 size {feats_flat.size} is not divisible by 36")
    T_total = feats_flat.size // 36

    # 训练使用的 4s 片段长度（帧数）
    SEG_T = int(getattr(cfg, "sequence_length", 400))
    frames_per_sample = 160  # 16 kHz, 10 ms / frame

    import math

    n_segments = max(1, math.ceil(T_total / SEG_T))
    T_padded = n_segments * SEG_T
    pad_frames = T_padded - T_total

    if pad_frames > 0:
        pad_flat = np.zeros((pad_frames * 36,), dtype=np.float32)
        feats_flat = np.concatenate([feats_flat, pad_flat], axis=0)

    print(
        f"[ExportBits] Features: T_raw={T_total}, T_padded={T_padded}, "
        f"pad_frames={pad_frames}, SEG_T={SEG_T}, n_segments={n_segments}"
    )

    feats = torch.from_numpy(feats_flat.reshape(T_padded, 36)).unsqueeze(0).to(device)  # [1,T_padded,36]

    # 可选：在 --out_bits 指定的目录旁边额外保存一份 .f32 特征
    try:
        if args.out_bits:
            if getattr(args, "out_features", None):
                features_out_path = args.out_features
            else:
                out_dir_for_bits = os.path.dirname(args.out_bits) or "."
                features_out_path = os.path.join(out_dir_for_bits, "features.f32")

            os.makedirs(os.path.dirname(features_out_path) or ".", exist_ok=True)
            feats_flat.astype(np.float32).tofile(features_out_path)
            print(
                f"[ExportBits] Saved features copy to {features_out_path} "
                f"(T={T_padded}, dim=36)"
            )
    except Exception as exc:  # pragma: no cover - best-effort side effect
        print(f"[ExportBits] WARNING: failed to save features copy alongside bits: {exc}")

    # 对齐 PCM 长度到 T_padded 帧（10 ms / frame at 16 kHz -> 160 samples/frame）
    target_len_total = int(T_padded * frames_per_sample)
    pcm_len_raw = int(audio.size(1))
    if audio.size(1) < target_len_total:
        pad = target_len_total - audio.size(1)
        audio = torch.nn.functional.pad(audio, (0, pad))
    elif audio.size(1) > target_len_total:
        audio = audio[:, :target_len_total]

    pcm_len_aligned = int(audio.size(1))
    print(
        f"[ExportBits] PCM: len_raw={pcm_len_raw}, len_aligned={pcm_len_aligned}, "
        f"frames_aligned={pcm_len_aligned // frames_per_sample}, T_padded={T_padded}"
    )

    # 保存对齐后的 PCM（供解码端/调试统一使用）
    try:
        pcm_out_path = aligned_pcm_path

        os.makedirs(os.path.dirname(pcm_out_path) or ".", exist_ok=True)

        audio_np = audio.squeeze(0).detach().cpu().numpy()
        audio_np = np.nan_to_num(audio_np, nan=0.0).clip(-1.0, 1.0)
        pcm_aligned = (audio_np * 32767.0).astype(np.int16)
        pcm_aligned.tofile(pcm_out_path)
        print(
            f"[ExportBits] Saved aligned PCM to {pcm_out_path} "
            f"(len={pcm_aligned.size}, frames={pcm_aligned.size // frames_per_sample})"
        )
    except Exception as exc:  # pragma: no cover - 最好努力写文件
        print(f"[ExportBits] WARNING: failed to save aligned PCM: {exc}")

    t_io = time.time() - t_io_start

    # 逐 4s 片段前向 JSCC 编码（量化器可为 Hash 或 RVQ）
    bits_segments: list[np.ndarray] = []
    ber_segments: list[float] = []
    hw_list: list[np.ndarray] = []
    csi_list: list[np.ndarray] = []
    content_shape: Optional[list[int]] = None
    f0_shape: Optional[list[int]] = None
    stats_shape: Optional[list[int]] = None
    with torch.no_grad():
        t_forward_start = time.time()

        for seg_idx in range(n_segments):
            t0 = seg_idx * SEG_T
            t1 = (seg_idx + 1) * SEG_T
            s0 = t0 * frames_per_sample
            s1 = t1 * frames_per_sample

            audio_seg = audio[:, s0:s1]
            feats_seg = feats[:, t0:t1, :]

            if getattr(args, "bit_only", False):
                # bits-only：分别导出 clean bits 与 noisy bits，用于内部 BER
                # 统计，并与 decode_from_bits_offline 对齐。
                bits_c_clean, bits_f_clean, bits_s_clean, _ = model.encode_hash_codec(
                    audio=audio_seg,
                    fargan_feats=feats_seg,
                    channel_sim=channel_sim,
                    snr_min_db=float(args.snr_min_db),
                    snr_max_db=float(args.snr_max_db),
                    return_meta=True,
                    use_noisy_bits=False,
                )
                bits_c_t, bits_f_t, bits_s_t, meta_t = model.encode_hash_codec(
                    audio=audio_seg,
                    fargan_feats=feats_seg,
                    channel_sim=channel_sim,
                    snr_min_db=float(args.snr_min_db),
                    snr_max_db=float(args.snr_max_db),
                    return_meta=True,
                    use_noisy_bits=True,
                )
            else:
                # 默认路径：encode_quant_codec，兼容 Hash/RVQ + 外部 FSK
                use_noisy_bits = bool(
                    getattr(args, "csi_mode", "channel_sim") == "real_noise"
                )
                bits_c_t, bits_f_t, bits_s_t, meta_t = model.encode_quant_codec(
                    audio=audio_seg,
                    fargan_feats=feats_seg,
                    channel_sim=channel_sim,
                    snr_min_db=float(args.snr_min_db),
                    snr_max_db=float(args.snr_max_db),
                    use_noisy_bits=use_noisy_bits,
                )
            print(f"[DEBUG] encode_quant_codec后，模型hb的quantizer_type: {model.hash_bottleneck.quantizer_type if hasattr(model, 'hash_bottleneck') else '无hash_bottleneck属性'}")
            # 记录每段 bitshape（仅记录第一段，假设后续一致）
            if content_shape is None and isinstance(bits_c_t, torch.Tensor):
                content_shape = list(bits_c_t.shape)
            if f0_shape is None and isinstance(bits_f_t, torch.Tensor):
                f0_shape = list(bits_f_t.shape)
            if stats_shape is None and isinstance(bits_s_t, torch.Tensor):
                stats_shape = list(bits_s_t.shape)
            # 保存每个 segment 的符号图尺寸和 CSI 向量作为“信道固有信息”
            if isinstance(meta_t, dict) and "hw" in meta_t:
                hw_list.append(np.array(meta_t["hw"], dtype=np.int32))              # (2,)
            if isinstance(meta_t, dict) and "csi_vec" in meta_t:
                csi_list.append(meta_t["csi_vec"].astype(np.float32).reshape(-1))   # (4,)

            seg_bits_list = []
            if getattr(args, "bit_only", False):
                # 保留软值 / noisy bits（与 decode_from_bits_offline 对齐）
                bits_c_noisy = bits_c_t.detach().cpu().numpy().astype(np.float32).reshape(-1)
                bits_f_noisy = bits_f_t.detach().cpu().numpy().astype(np.float32).reshape(-1)
                bits_s_noisy = bits_s_t.detach().cpu().numpy().astype(np.float32).reshape(-1)

                bits_c_clean = bits_c_clean.detach().cpu().numpy().astype(np.float32).reshape(-1)
                bits_f_clean = bits_f_clean.detach().cpu().numpy().astype(np.float32).reshape(-1)
                bits_s_clean = bits_s_clean.detach().cpu().numpy().astype(np.float32).reshape(-1)

                bits_noisy_all = np.concatenate(
                    [bits_c_noisy, bits_f_noisy, bits_s_noisy], axis=0
                )
                bits_clean_all = np.concatenate(
                    [bits_c_clean, bits_f_clean, bits_s_clean], axis=0
                )

                # 内部 BER：基于 sign(b) 的 0/1 判决
                bits0 = (bits_clean_all > 0).astype(np.uint8)
                bits1 = (bits_noisy_all > 0).astype(np.uint8)
                n = bits0.size
                bit_err = int(np.sum(bits0 != bits1))
                ber_seg = float(bit_err / max(1, n))
                ber_segments.append(ber_seg)

                seg_bits_list += [bits_c_noisy, bits_f_noisy, bits_s_noisy]
                bits_segments.append(bits_noisy_all)
            else:
                bits_c = (
                    bits_c_t.detach().sign().cpu().numpy().astype(np.float32).reshape(-1)
                )
                bits_f = (
                    bits_f_t.detach().sign().cpu().numpy().astype(np.float32).reshape(-1)
                )
                bits_s = (
                    bits_s_t.detach().sign().cpu().numpy().astype(np.float32).reshape(-1)
                )

                seg_bits_list += [bits_c, bits_f, bits_s]
                bits_segments.append(np.concatenate(seg_bits_list, axis=0))

        t_forward = time.time() - t_forward_start

    # 拼接所有片段的比特流：按 [seg0_content|seg0_f0|seg0_stats][seg1_content|...] 顺序
    bits_all = np.concatenate(bits_segments, axis=0)

    # 将 bits 作为 .npy 保存（与现有解码脚本兼容）
    out_dir = os.path.dirname(args.out_bits) or "."
    os.makedirs(out_dir, exist_ok=True)
    np.save(args.out_bits, bits_all.astype(np.float32))

    # 其余 meta（hw/csi/shape/quantizer_type 等）以文本 JSON 形式保存为 sidecar 文件，
    # 视作“信道固有信息”与 bitstream 布局描述，不占用比特预算。
    meta: dict[str, object] = {
        "layout_version": 1,
        "quantizer_type": str(getattr(cfg, "quantizer_type", "hash")),
        "branch": args.branch,
        "n_segments": int(n_segments),
        "seg_frames": int(SEG_T),
    }
    if content_shape is not None:
        meta["content_shape"] = content_shape
    if f0_shape is not None:
        meta["f0_shape"] = f0_shape
    if stats_shape is not None:
        meta["stats_shape"] = stats_shape
    if hw_list:
        meta["hw"] = [hw.tolist() for hw in hw_list]
    if csi_list:
        meta["csi"] = [c.astype(float).tolist() for c in csi_list]
    if ber_segments:
        meta["ber_internal"] = [float(b) for b in ber_segments]

    meta_path = os.path.splitext(args.out_bits)[0] + "_meta.json"
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(
            f"[ExportBits] Exported total {bits_all.size} bits "
            f"(segments={n_segments}, branch={args.branch}) to {args.out_bits}\n"
            f"[ExportBits] Saved sidecar meta (hw/csi) to {meta_path}"
        )
    except Exception as exc:  # pragma: no cover - best-effort meta save
        print(f"[ExportBits] WARNING: failed to save meta to {meta_path}: {exc}")

    t_post_start = time.time()
    t_post = time.time() - t_post_start

    total_elapsed = time.time() - t_total_start
    print(
        f"[Timing-Export] model={t_model:.3f}s, io={t_io:.3f}s, "
        f"forward={t_forward:.3f}s, post={t_post:.3f}s, total={total_elapsed:.3f}s",
    )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
