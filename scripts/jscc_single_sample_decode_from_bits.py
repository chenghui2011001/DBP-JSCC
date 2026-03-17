#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Decode JSCC audio from external HashBottleneck bits (single sample).

本版本使用 DualBranchMelJSCC 内部的 ``decode_hash_codec`` +
``forward_with_hash`` 覆盖路径：

- 从外部传入的 Hash bits 覆盖模型内部的 HashBottleneck 输出；
- 其余流水线保持与训练阶段 ``forward_with_hash`` 完全一致
  （含内容分支 VMamba、F0/VUV 分支和 FARGAN 声码器）。

为保证与训练路径对齐，解码阶段需要：

- 一段 PCM（与编码端对齐，通常为 input_aligned.pcm）；
- 对应的 36 维 FARGAN 特征 (.f32)，通常由编码脚本
  ``jscc_single_sample_export_bits.py`` 生成的 features.f32。

接收端只需持有与编码端一致的 PCM+features+bits 组合即可
重现训练阶段的前向行为；--pcm_ref 仍用于可视化对比。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict
import wave
import time

import numpy as np
import torch
import json
import torch.nn.functional as F


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-sample JSCC decode from external hash bits")

    p.add_argument("--ckpt", type=str, required=True, help="Stage2.5 checkpoint (.pth)")
    p.add_argument(
        "--pcm",
        type=str,
        default=None,
        help=(
            "Path to 16 kHz mono 16-bit PCM/WAV used for JSCC decode. "
            "If omitted, tries '<dir(bits_rx)>/input_aligned.pcm' (export script output), "
            "then falls back to --pcm_ref."
        ),
    )
    p.add_argument(
        "--pcm_ref",
        type=str,
        default=None,
        help=(
            "Optional reference PCM path for visualization. "
            "If not set, --pcm is used as the reference audio."
        ),
    )
    p.add_argument(
        "--features",
        type=str,
        default=None,
        help=(
            "Path to matching 36-dim vocoder features (.f32). "
            "If omitted, defaults to '<dir(bits_rx)>/features.f32' (export script output)."
        ),
    )
    p.add_argument(
        "--bits_rx",
        type=str,
        default="../Fargan_sim/jscc_bits_rx.npy",
        help="Path to received JSCC bits .npy (0/1 or ±1)",
    )
    p.add_argument(
        "--out_pcm",
        type=str,
        default="../Fargan_sim/output_jscc_fsk.pcm",
        help="Output PCM path for decoded audio",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (e.g. 'cpu' or 'cuda'); default: auto",
    )
    p.add_argument(
        "--no_metrics",
        action="store_true",
        help="Skip PESQ/STOI/F0/Mel MSE style objective metrics computation",
    )
    p.add_argument(
        "--no_plot",
        action="store_true",
        help="Skip generating comparison plot PNG (still writes PCM/WAV)",
    )
    p.add_argument("--snr_min_db", type=float, default=-5.0, help="SNR min (dB) for ChannelSimulator")
    p.add_argument("--snr_max_db", type=float, default=15.0, help="SNR max (dB) for ChannelSimulator")
    p.add_argument(
        "--csi_mode",
        type=str,
        default="channel_sim",
        choices=["channel_sim", "external_snr", "real_noise"],
        help=(
            "CSI generation mode used for dummy shape inference / fallback: "
            "'channel_sim' (training-style), 'external_snr' (external SNR proxy), "
            "or 'real_noise' (RealNoiseChannelSimulator driven by noise CSV)."
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
        "--snr_db",
        type=float,
        default=None,
        help=(
            "Optional SNR (dB) injected into CSI FiLM for offline decode; "
            "若未提供，则使用模型默认的全零 CSI。"
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

    # Optional post-processing on decoded audio
    p.add_argument(
        "--crossfade_ms",
        type=float,
        default=20.0,
        help=(
            "Crossfade duration (in ms) between consecutive segments to reduce boundary artifacts "
            "(0 disables crossfade)."
        ),
    )


    # DEBUG: optionally bypass stats bits and use GT mel mean/std
    # 仅用于离线对比 rumble 是否由能量标尺错误引起，不建议训练/生产长期开启。
    p.add_argument(
        "--dbg_force_gt_stats",
        action="store_true",
        help=(
            "DEBUG ONLY: bypass stats bits and feed GT mel mean/std per segment "
            "into decode_from_bits_offline (requires PCM/features)."
        ),
    )

    # 当前分段实现假设导出端使用 'both' 分支（内容+F0）。
    # 如需支持其它分支，可以在此扩展并在推导片段形状时分别处理。
    p.add_argument(
        "--branch",
        type=str,
        default="both",
        choices=["both"],
        help="Branch configuration used at export (currently only 'both' is supported)",
    )
    p.add_argument(
        "--bit_only",
        action="store_true",
        help=(
            "Use training-style bits-only decode path: decode_from_bits_offline "
            "with noisy bits, matching the public DBP-JSCC bit-only evaluation path."
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
        print(f"[JSCC-Decode] Dropped {len(dropped)} key(s) due to shape mismatch; examples: {dropped[:5]}")
    load_ret = model.load_state_dict(safe_sd, strict=False)
    try:
        missing = list(getattr(load_ret, "missing_keys", []))
        unexpected = list(getattr(load_ret, "unexpected_keys", []))
    except Exception:
        missing, unexpected = [], []
    if missing or unexpected:
        print(
            f"[JSCC-Decode] Non-strict load: missing={len(missing)}, "
            f"unexpected={len(unexpected)}",
        )


def _load_pcm_or_wav_as_int16(path: str, expect_sr: int = 16000) -> np.ndarray:
    """Load 16-bit mono audio as int16 from .pcm or .wav for visualization.

    解码本身不依赖原始 PCM，该函数仅用于可视化对比；
    若为 WAV，则要求 16-bit mono 且采样率为 ``expect_sr``。
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

    return np.fromfile(path, dtype=np.int16)


def _crossfade_segments(chunks, crossfade_samples: int, device: torch.device) -> torch.Tensor:
    """Linearly crossfade a list of [1,L] tensors along time.

    Args:
        chunks: list of tensors [1, L_i]
        crossfade_samples: number of samples for overlap region
    Returns:
        Single tensor [1, L_total] with overlaps blended.
    """
    if not chunks:
        return torch.zeros(1, 0, device=device)
    if len(chunks) == 1 or crossfade_samples <= 0:
        return chunks[0]

    out = chunks[0]
    for seg in chunks[1:]:
        a = out
        b = seg
        L1 = int(a.size(-1))
        L2 = int(b.size(-1))
        N = min(crossfade_samples, L1, L2)
        if N <= 0:
            out = torch.cat([a, b], dim=-1)
            continue
        fade_out = torch.linspace(1.0, 0.0, N, device=device, dtype=a.dtype).unsqueeze(0)
        fade_in = torch.linspace(0.0, 1.0, N, device=device, dtype=b.dtype).unsqueeze(0)
        overlap = a[..., -N:] * fade_out + b[..., :N] * fade_in
        out = torch.cat([a[..., :-N], overlap, b[..., N:]], dim=-1)
    return out





def _compute_pesq_stoi(
    ref: np.ndarray,
    deg: np.ndarray,
    sample_rate: int = 16000,
) -> tuple[Optional[float], Optional[float]]:
    """Compute PESQ and STOI between reference and degraded audio.

    This helper is best-effort: if the required third-party packages
    (``pesq``, ``pystoi``) are not available, it prints a warning and
    returns ``None`` for the corresponding metric instead of raising.
    """

    pesq_score: Optional[float] = None
    stoi_score: Optional[float] = None

    # Ensure 1D float32 in [-1, 1] and aligned length
    ref_f = np.asarray(ref, dtype=np.float32).reshape(-1)
    deg_f = np.asarray(deg, dtype=np.float32).reshape(-1)
    if ref_f.size == 0 or deg_f.size == 0:
        return pesq_score, stoi_score

    n = min(ref_f.size, deg_f.size)
    ref_f = ref_f[:n]
    deg_f = deg_f[:n]
    ref_f = np.nan_to_num(ref_f, nan=0.0).clip(-1.0, 1.0)
    deg_f = np.nan_to_num(deg_f, nan=0.0).clip(-1.0, 1.0)

    # PESQ (ITU-T P.862 wideband mode at 16 kHz)
    try:
        from pesq import pesq as pesq_fn  # type: ignore

        pesq_score = float(pesq_fn(sample_rate, ref_f, deg_f, "wb"))
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[JSCC-Decode] WARNING: PESQ computation failed or pesq package missing: {exc}")

    # STOI
    try:
        from pystoi import stoi as stoi_fn  # type: ignore

        stoi_val = stoi_fn(ref_f, deg_f, sample_rate, extended=False)
        try:
            stoi_score = float(stoi_val)
        except Exception:
            stoi_score = None
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[JSCC-Decode] WARNING: STOI computation failed or pystoi package missing: {exc}")

    return pesq_score, stoi_score


def main() -> int:
    args = _parse_args()
    t_total_start = time.time()
    t_model = t_io = t_bits = t_forward = t_post = 0.0

    t_model_start = time.time()

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from training.train_support import build_model  # type: ignore
    from utils.audio_visualizer import create_audio_comparison_plot  # type: ignore
    from utils.channel_sim import ChannelSimulator, RealNoiseChannelSimulator  # type: ignore

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
    print(f"quantizer_type={qtype}")
    print(f"[DEBUG] 模型cfg中的quantizer_type: {qtype}")
    print(f"[DEBUG] cfg完整quantizer相关配置: {cfg.quantizer_type if hasattr(cfg, 'quantizer_type') else '无此字段'}")
    if qtype not in ("hash", "rvq"):
        raise RuntimeError(f"Unsupported quantizer_type={qtype}; expected 'hash' or 'rvq'")
    cfg.device = str(device)

    print(f"[JSCC-Decode] Building model on device={device} ...")
    model = build_model(cfg).to(device)
    # Ensure offline decode uses stats-based energy calibration path.
    # This only relies on bits_stats (mel_mean_hat) and does not leak GT.
    try:
        if hasattr(model, "enable_energy_calib"):
            model.enable_energy_calib = True
    except Exception:
        pass
    model.eval()
    state_dict = None
    for key in ("model_state_dict", "model", "state_dict"):
        if key in ckpt:
            state_dict = ckpt[key]
            break
    if state_dict is None:
        raise RuntimeError("Checkpoint missing model weights")
    _safe_load_model_state(model, state_dict)

    t_model = time.time() - t_model_start
    t_io = 0.0

    # Load received bits
    t_bits_start = time.time()

    data = np.load(args.bits_rx, allow_pickle=True)
    bits_rx = data["bits"] if isinstance(data, np.lib.npyio.NpzFile) else data

    # 侧信道与布局信息（来自 sidecar JSON）：
    hw_arr = None
    csi_arr = None
    meta_content_shape: Optional[Tuple[int, int, int]] = None
    meta_f0_shape: Optional[Tuple[int, int, int]] = None
    meta_stats_shape: Optional[Tuple[int, int, int]] = None
    meta_layout_version: int = 0
    meta_qtype: Optional[str] = None

    # 若 bits_rx 为普通 .npy，比特本身不包含 hw/csi/shape，这些被视为“信道固有信息”
    # 由导出脚本写入 sidecar 文本（JSON）。这里尝试从同名 *_meta.json 读取。
    try:
        base = os.path.splitext(args.bits_rx)[0]
        meta_path = base + "_meta.json"
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if isinstance(meta, dict):
                if "hw" in meta:
                    hw_arr = np.asarray(meta["hw"], dtype=np.int32)
                if "csi" in meta:
                    csi_arr = np.asarray(meta["csi"], dtype=np.float32)
                if "content_shape" in meta:
                    cs = [int(x) for x in meta["content_shape"]]
                    if len(cs) == 3:
                        meta_content_shape = (cs[0], cs[1], cs[2])
                if "f0_shape" in meta:
                    fs = [int(x) for x in meta["f0_shape"]]
                    if len(fs) == 3:
                        meta_f0_shape = (fs[0], fs[1], fs[2])
                if "stats_shape" in meta:
                    ss = [int(x) for x in meta["stats_shape"]]
                    if len(ss) == 3:
                        meta_stats_shape = (ss[0], ss[1], ss[2])
                if "layout_version" in meta:
                    try:
                        meta_layout_version = int(meta["layout_version"])
                    except Exception:
                        meta_layout_version = 0
                if "quantizer_type" in meta:
                    meta_qtype = str(meta["quantizer_type"])
            print(f"[JSCC-Decode] Loaded sidecar meta from {meta_path} (layout_v={meta_layout_version})")
    except Exception as exc:
        print(f"[JSCC-Decode] WARNING: failed to load sidecar meta: {exc}")


    # bits-only 路径：保留 encode_hash_codec 输出的 noisy bits（软值）；
    # 默认路径仍对 0/1 或 ±1 bitstream 做归一化。
    if getattr(args, "bit_only", False):
        bits_sign = bits_rx.astype(np.float32)
    else:
        if np.any((bits_rx < 0) | (bits_rx > 1)):
            bits01 = (bits_rx > 0).astype(np.uint8)
        else:
            bits01 = bits_rx.astype(np.uint8)
        bits_sign = (2.0 * bits01.astype(np.float32) - 1.0).astype(np.float32)

    # ---- 推断单个片段的 bit 形状（content/F0/Stats） ----
    # 默认用 ChannelSimulator；bit_only 模式下仍与训练保持一致。
    chan = ChannelSimulator(sample_rate=16000, frame_hz=100)
    # 训练时使用的目标帧数（10 ms / frame），作为每段音频的“理想长度”。
    # 若 CLI 提供 --seq_length，则优先使用该值，否则退回到 cfg.sequence_length。
    SEG_T_cfg = int(args.seq_length or int(getattr(cfg, "sequence_length", 400)))
    frames_per_sample = 160

    if meta_content_shape is not None and meta_f0_shape is not None and meta_stats_shape is not None:
        Bc_m, Lc_seg, Kc = meta_content_shape
        Bf_m, Tf_seg, Kf = meta_f0_shape
        Bs_m, Sf_seg, Ks = meta_stats_shape
        B_seg_c = int(Bc_m * Lc_seg * Kc)
        B_seg_f = int(Bf_m * Tf_seg * Kf)
        B_seg_s = int(Bs_m * Sf_seg * Ks)
        content_hw_seg = None
        print(
            "[JSCC-Decode] Using meta shapes: "
            f"content_shape={meta_content_shape}, f0_shape={meta_f0_shape}, stats_shape={meta_stats_shape}"
        )
    else:
        # 兼容旧版：使用 dummy encode 推断形状
        dummy_audio = torch.zeros(1, SEG_T_cfg * frames_per_sample, device=device)
        dummy_feats = torch.zeros(1, SEG_T_cfg, 36, device=device)

        with torch.no_grad():
            bits_c_t, bits_f_t, bits_s_t, meta_dummy = model.encode_quant_codec(
                audio=dummy_audio,
                fargan_feats=dummy_feats,
                channel_sim=chan,
                snr_min_db=float(args.snr_min_db),
                snr_max_db=float(args.snr_max_db),
            )
            content_hw_seg = meta_dummy.get("hw", None)
            print(
                "[DBG] content_hw_seg=",
                content_hw_seg,
                "Lc_seg=",
                bits_c_t.shape[1],
                "prod=",
                (content_hw_seg[0] * content_hw_seg[1]) if content_hw_seg else None,
            )
        if not isinstance(bits_c_t, torch.Tensor) or not isinstance(bits_f_t, torch.Tensor):
            raise RuntimeError("encode_quant_codec did not produce both content and F0 bits; branch='both' required")

        _, Lc_seg, Kc = bits_c_t.shape  # [1,Lc_seg,Kc]
        _, Tf_seg, Kf = bits_f_t.shape  # [1,Tf_seg,Kf]
        _, Sf_seg, Ks = bits_s_t.shape
        B_seg_c = int(bits_c_t.numel())
        B_seg_f = int(bits_f_t.numel())
        B_seg_s = int(bits_s_t.numel())
    B_seg_total = B_seg_c + B_seg_f + B_seg_s

    if Tf_seg != SEG_T_cfg:
        print(
            f"[JSCC-Decode] WARNING: inferred Tf_seg={Tf_seg} != cfg.sequence_length={SEG_T_cfg}; "
            "using cfg.sequence_length (or --seq_length) for target duration, "
            "but Tf_seg for bit layout."
        )

    if B_seg_total <= 0:
        raise RuntimeError("Segment bit-length is zero; cannot decode")

    if bits_sign.size % B_seg_total != 0:
        raise RuntimeError(
            f"Total bits ({bits_sign.size}) not divisible by per-segment bits ({B_seg_total}); "
            "please ensure export and decode use the same configuration."
        )

    n_segments = bits_sign.size // B_seg_total
    print(
        f"[JSCC-Decode] Decoding {n_segments} segment(s), "
        f"SEG_T_cfg={SEG_T_cfg} frames, Tf_seg={Tf_seg}"
    )

    t_bits = time.time() - t_bits_start

    # ---- I/O: load aligned PCM + FARGAN features for decode ----
    t_io_start = time.time()

    # 1) 解析 PCM 路径：优先显式 --pcm，其次 bits 目录下的 input_aligned.pcm，
    #    最后回退到 --pcm_ref（仅用于没有对齐 PCM 的场景）。
    bits_dir = os.path.dirname(args.bits_rx) or "."
    pcm_path = args.pcm
    if pcm_path is None:
        cand_aligned = os.path.join(bits_dir, "input_aligned.pcm")
        if os.path.isfile(cand_aligned):
            pcm_path = cand_aligned
        elif args.pcm_ref is not None:
            pcm_path = args.pcm_ref
    if pcm_path is None or not os.path.isfile(pcm_path):
        raise FileNotFoundError(
            "Decode PCM not found. Please provide --pcm, or ensure "
            "'<dir(bits_rx)>/input_aligned.pcm' exists, or pass --pcm_ref."
        )

    pcm = _load_pcm_or_wav_as_int16(pcm_path, expect_sr=16000)
    if pcm.size == 0:
        raise RuntimeError(f"Decode PCM file is empty: {pcm_path}")
    audio = torch.from_numpy(pcm.astype(np.float32) / 32768.0).unsqueeze(0).to(device)  # [1,L]

    # 2) 解析 FARGAN 特征路径：优先显式 --features，其次 bits 目录下的 features.f32。
    features_path = args.features
    if features_path is None:
        features_path = os.path.join(bits_dir, "features.f32")
    if not os.path.isfile(features_path):
        raise FileNotFoundError(
            f"FARGAN features file not found: {features_path}.\n"
            f"Please pass --features explicitly, or ensure the export script wrote "
            f"features.f32 next to bits_rx."
        )

    feats_flat = np.fromfile(features_path, dtype=np.float32)
    if feats_flat.size % 36 != 0:
        raise RuntimeError(f"features.f32 size {feats_flat.size} is not divisible by 36")
    T_total = feats_flat.size // 36
    feats = torch.from_numpy(feats_flat.reshape(T_total, 36)).unsqueeze(0).to(device)  # [1,T_total,36]

    # 对齐 PCM 长度到特征帧数（10 ms / frame at 16 kHz -> 160 samples/frame）
    target_len_total = int(T_total * frames_per_sample)
    if audio.size(1) < target_len_total:
        pad = target_len_total - audio.size(1)
        audio = torch.nn.functional.pad(audio, (0, pad))
    elif audio.size(1) > target_len_total:
        audio = audio[:, :target_len_total]

    print(
        f"[JSCC-Decode] Decode PCM len={audio.size(1)}, features T={T_total} "
        f"(frames={audio.size(1) // frames_per_sample})"
    )

    # 根据特征帧数推导分段数，并与 bits 推导的 n_segments 交叉检查
    import math

    n_segments_feat = max(1, math.ceil(T_total / SEG_T_cfg))
    T_padded = n_segments_feat * SEG_T_cfg
    if T_padded != T_total:
        pad_frames = T_padded - T_total
        pad_flat = np.zeros((pad_frames * 36,), dtype=np.float32)
        feats_flat = np.concatenate([feats_flat, pad_flat], axis=0)
        feats = torch.from_numpy(feats_flat.reshape(T_padded, 36)).unsqueeze(0).to(device)
        T_total = T_padded
        target_len_total = int(T_total * frames_per_sample)
        if audio.size(1) < target_len_total:
            pad = target_len_total - audio.size(1)
            audio = torch.nn.functional.pad(audio, (0, pad))
        elif audio.size(1) > target_len_total:
            audio = audio[:, :target_len_total]
        print(
            f"[JSCC-Decode] Padded features to T={T_total} frames (segments={n_segments_feat}) "
            f"to match SEG_T_cfg={SEG_T_cfg}"
        )

    if n_segments_feat != n_segments:
        raise RuntimeError(
            f"Segment count mismatch: bits imply {n_segments} segment(s) but "
            f"features/PCM imply {n_segments_feat} segment(s)."
        )

    t_io = time.time() - t_io_start

    # ---- 按 4s 片段逐段解码并拼接音频（使用 decode_hash_codec + forward_with_hash 覆盖路径） ----
    audio_chunks: list[torch.Tensor] = []
    offset = 0

    channel_sim = chan  # 复用前面构建的 ChannelSimulator

    t_forward_start = time.time()

    for seg_idx in range(n_segments):
        seg_bits = bits_sign[offset : offset + B_seg_total]
        offset += B_seg_total

        # seg_bits: length = B_seg_c + B_seg_f + B_seg_s
        o0 = 0
        o1 = o0 + B_seg_c
        o2 = o1 + B_seg_f
        o3 = o2 + B_seg_s

        bits_c_flat = seg_bits[o0:o1]
        bits_f_flat = seg_bits[o1:o2]
        bits_s_flat = seg_bits[o2:o3]

        assert bits_c_flat.size == B_seg_c
        assert bits_f_flat.size == B_seg_f
        assert bits_s_flat.size == B_seg_s

        bits_tensor_c = torch.from_numpy(bits_c_flat.reshape(1, Lc_seg, Kc)).to(device)
        bits_tensor_f = torch.from_numpy(bits_f_flat.reshape(1, Tf_seg, Kf)).to(device)
        bits_tensor_s = torch.from_numpy(bits_s_flat.reshape(1, Sf_seg, Ks)).to(device)

        t0 = seg_idx * SEG_T_cfg
        t1 = (seg_idx + 1) * SEG_T_cfg
        s0 = t0 * frames_per_sample
        s1 = t1 * frames_per_sample
        # 该段目标音频长度（采样点）
        target_len_seg = int(SEG_T_cfg * frames_per_sample)
        content_hw_seg = tuple(hw_arr[seg_idx].tolist()) if hw_arr is not None else None

        if csi_arr is not None:
            csi_np = csi_arr[seg_idx].astype(np.float32).reshape(-1)   # (4,)
            csi_vec_t = torch.from_numpy(csi_np).to(device).view(1, -1)  # [1,4]
        else:
            # 没有保存 CSI 的情况下：给一个固定 CSI，避免随机 sample 导致解码条件错乱
            snr = float(args.snr_db) if args.snr_db is not None else float(args.snr_min_db)
            csi_vec_t = torch.tensor([[snr, 0.5, 0.5, 0.5]], device=device, dtype=torch.float32)  # [1,4]

        # 可选 DEBUG：直接使用该段 PCM 估计 GT mel 的 mean/std，
        # 作为 decode_from_bits_offline 的统计量侧信息，旁路 bits_stats，
        # 用于对比 rumble 是否来自 stats hash 的偏差。
        gt_mel_mean = None
        gt_mel_std = None
        if args.dbg_force_gt_stats:
            with torch.no_grad():
                audio_seg = audio[:, s0:s1]  # [1,L_seg]
                mel_seg = model.wave_to_mel(audio_seg)  # [1,T_mel,32]
                T_mel = int(mel_seg.size(1))
                T_seg = min(T_mel, SEG_T_cfg)
                mel_seg = mel_seg[:, :T_seg, :]
                gt_mel_mean = mel_seg.mean(dim=(1, 2), keepdim=True)  # [1,1,1]
                gt_mel_std = mel_seg.std(dim=(1, 2), keepdim=True).clamp(min=0.1)  # [1,1,1]

        if getattr(args, "bit_only", False):
            out_seg = model.decode_from_bits_offline(
                bits_content=bits_tensor_c,
                bits_f0=bits_tensor_f,
                bits_stats=bits_tensor_s,
                f0_T=SEG_T_cfg,
                target_len=target_len_seg,
                csi_vec=csi_vec_t,
                snr_db=None,
                content_hw=content_hw_seg,
            )
        else:
            out_seg = model.decode_quant_codec(
                bits_content=bits_tensor_c,
                bits_f0=bits_tensor_f,
                bits_stats=None if args.dbg_force_gt_stats else bits_tensor_s,
                f0_T=SEG_T_cfg,
                target_len=target_len_seg,
                csi_vec=csi_vec_t,
                snr_db=None,
                content_hw=content_hw_seg,
                gt_mel_mean=gt_mel_mean,
                gt_mel_std=gt_mel_std,
            )

        if "audio_hat" not in out_seg:
            raise RuntimeError("Model output missing 'audio_hat' for segment decode")

        audio_chunks.append(out_seg["audio_hat"])  # [1,L_seg]

    # 拼接所有片段音频，并在段间做简单淡入淡出
    crossfade_samples = int(max(0.0, float(args.crossfade_ms)) * 16.0)  # 16 samples/ms @16kHz
    audio_hat_t = _crossfade_segments(audio_chunks, crossfade_samples, device=device)  # [1,L_total]
    t_forward = time.time() - t_forward_start

    t_post_start = time.time()


    

    audio_hat = audio_hat_t.detach().cpu().numpy().reshape(-1)
    audio_hat = np.nan_to_num(audio_hat, nan=0.0).clip(-1.0, 1.0)
    pcm_out = (audio_hat * 32767.0).astype(np.int16)
    out_dir = os.path.dirname(args.out_pcm) or "."
    os.makedirs(out_dir, exist_ok=True)
    pcm_out.tofile(args.out_pcm)
    print(f"[JSCC-Decode] Wrote decoded PCM to: {args.out_pcm}")

    # 1) 写出放大后的 WAV，便于试听
    try:
        wav_path = os.path.splitext(args.out_pcm)[0] + ".wav"

        pcm_for_wav = pcm_out
        pcm_float = pcm_out.astype(np.float32)
        peak = float(np.max(np.abs(pcm_float))) if pcm_float.size > 0 else 0.0
        target_peak = 0.98 * 32767.0
        max_gain = 8.0
        applied_gain = 1.0
        if peak > 0.0 and peak < target_peak:
            gain = target_peak / peak
            gain = min(gain, max_gain)
            if gain > 1.0:
                pcm_float = np.clip(pcm_float * gain, -32768.0, 32767.0)
                pcm_for_wav = pcm_float.astype(np.int16)
                applied_gain = gain

        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(pcm_for_wav.tobytes())
        print(
            f"[JSCC-Decode] Wrote decoded WAV to: {wav_path} "
            f"(gain x{applied_gain:.2f})"
        )
    except Exception as exc:  # pragma: no cover - best-effort side effect
        print(f"[JSCC-Decode] WARNING: failed to write WAV: {exc}")

    # 2) 可选：生成与训练阶段类似的对比图像 + 输入 WAV（若提供参考 PCM）。
    #    若传入 --no_plot，则跳过整段可视化逻辑，仅保留 PCM/WAV 写出。
    if not getattr(args, "no_plot", False):
        try:
            sr = 16000
            # 读回参考 PCM（原始输入），优先使用 --pcm_ref，其次回退到 --pcm；若均未提供则跳过。
            pcm_ref_path = args.pcm_ref or args.pcm
            if pcm_ref_path is None or not os.path.isfile(pcm_ref_path):
                raise FileNotFoundError("No reference PCM provided for visualization (--pcm/--pcm_ref)")

            pcm_in = _load_pcm_or_wav_as_int16(pcm_ref_path, expect_sr=16000)
            if pcm_in.size > 0:
                audio_in = torch.from_numpy(pcm_in.astype(np.float32) / 32768.0)
                audio_out = torch.from_numpy(audio_hat.astype(np.float32))

            # 截齐长度
            min_len = min(audio_in.numel(), audio_out.numel())
            audio_in = audio_in[:min_len]
            audio_out = audio_out[:min_len]

            # 计算 PESQ / STOI 评价指标（若可用），除非显式请求 --no_metrics。
            if not getattr(args, "no_metrics", False):
                try:
                    pesq_score, stoi_score = _compute_pesq_stoi(
                        audio_in.detach().cpu().numpy(),
                        audio_out.detach().cpu().numpy(),
                        sample_rate=sr,
                    )
                    if pesq_score is not None or stoi_score is not None:
                        print(
                            "[JSCC-Decode] Objective metrics: "
                            f"PESQ={pesq_score if pesq_score is not None else float('nan'):.3f}, "
                            f"STOI={stoi_score if stoi_score is not None else float('nan'):.3f}"
                        )

                        # 额外写出 JSON 方便脚本化分析
                        try:
                            base = os.path.splitext(os.path.basename(args.out_pcm))[0]
                            metrics_path = os.path.join(out_dir, f"{base}_metrics.json")
                            metrics = {
                                "pesq": pesq_score,
                                "stoi": stoi_score,
                                "sample_rate": sr,
                                "ref_path": pcm_ref_path,
                                "out_pcm": args.out_pcm,
                            }
                            with open(metrics_path, "w", encoding="utf-8") as f:
                                json.dump(metrics, f, ensure_ascii=False, indent=2)
                            print(f"[JSCC-Decode] Wrote objective metrics JSON to: {metrics_path}")
                        except Exception as exc_metrics:  # pragma: no cover
                            print(f"[JSCC-Decode] WARNING: failed to write metrics JSON: {exc_metrics}")
                except Exception as exc_metrics:  # pragma: no cover
                    print(f"[JSCC-Decode] WARNING: failed to compute PESQ/STOI: {exc_metrics}")

                # 对比图保存路径（和 out_pcm 同目录）
                base = os.path.splitext(os.path.basename(args.out_pcm))[0]
                viz_path = os.path.join(out_dir, f"{base}_comparison.png")

                create_audio_comparison_plot(
                    audio_real=audio_in,
                    audio_gen=audio_out,
                    save_path=viz_path,
                    sr=sr,
                    title=f"JSCC Decode Comparison - {base}",
                    show_waveform=True,
                    hop_length=160,
                )

                # 额外保存输入端 WAV，方便 A/B 对比（仅在提供参考 PCM 时启用）
                try:
                    wav_in_path = os.path.join(out_dir, f"{base}_input.wav")
                    with wave.open(wav_in_path, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(sr)
                        wf.writeframes(pcm_in.tobytes())
                    print(f"[JSCC-Decode] Wrote input WAV to: {wav_in_path}")
                except Exception as exc2:
                    print(f"[JSCC-Decode] WARNING: failed to write input WAV: {exc2}")
        except Exception as exc:  # pragma: no cover
            print(f"[JSCC-Decode] WARNING: failed to generate comparison plot: {exc}")

    t_post = time.time() - t_post_start

    total_elapsed = time.time() - t_total_start
    print(
        f"[Timing-Decode] model={t_model:.3f}s, io={t_io:.3f}s, "
        f"bits={t_bits:.3f}s, forward={t_forward:.3f}s, "
        f"post={t_post:.3f}s, total={total_elapsed:.3f}s",
    )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
