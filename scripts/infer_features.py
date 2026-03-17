#!/usr/bin/env python3
"""DBP-JSCC inference from prepared feature dumps.

This script assumes you already have an aligned 36-dim feature stream
plus PCM audio, for example from ``scripts/prepare_dataset.py``.

Given ``out_features.f32`` and ``out_speech.pcm``, the script:

1. Loads the DBP-JSCC checkpoint and builds the
   :class:`DualBranchBarkJSCC` model (same as training).
2. Crops the sequence to a configurable maximum length (default 400
   frames ≈ 4s) to keep memory usage bounded.
3. Runs one forward pass (no hash if ``with_hash=False`` in the
   checkpoint config).
4. Uses :mod:`utils.audio_visualizer` to create a waveform/F0/Bark
   comparison plot and exports CSVs for external plotting.

The path to the external feature extractor is **not** used here;
feature dumping should be done beforehand.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torchaudio

ROOT_DIR = Path(__file__).resolve().parent.parent
import sys as _sys

if str(ROOT_DIR) not in map(str, _sys.path):
    _sys.path.insert(0, str(ROOT_DIR))

from models.dual_branch_bark_jscc import DualBranchBarkJSCC  # type: ignore
from training.train_support import build_model  # type: ignore
from utils.channel_sim import ChannelSimulator  # type: ignore
from utils.audio_visualizer import create_audio_comparison_plot  # type: ignore
from utils.feature_extraction import load_feature_pcm_pair  # type: ignore


def _load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(str(path), map_location=device)
    if not isinstance(ckpt, dict):
        raise RuntimeError("Checkpoint must be a dict")
    return ckpt


def _build_model_from_ckpt_cfg(ckpt: Dict[str, Any], device: torch.device) -> Tuple[DualBranchBarkJSCC, Any]:
    cfg_dict = ckpt.get("cfg")
    if not isinstance(cfg_dict, dict):
        raise RuntimeError("Checkpoint field 'cfg' must be a dict containing the training config")

    class _ShimCfg:
        pass

    cfg = _ShimCfg()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)

    setattr(cfg, "device", str(device))

    model = build_model(cfg).to(device)

    # Restore model weights (shape-safe).
    sd = None
    for key in ("model_state_dict", "model", "state_dict"):
        if key in ckpt:
            sd = ckpt[key]
            break
    if sd is None:
        raise RuntimeError("Checkpoint does not contain model weights (model_state_dict/model/state_dict)")

    cur_sd = model.state_dict()
    safe_sd: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k in cur_sd and isinstance(v, torch.Tensor) and v.shape == cur_sd[k].shape:
            safe_sd[k] = v
    model.load_state_dict(safe_sd, strict=False)
    model.eval()

    return model, cfg


def _load_feature_outputs(
    features_path: Path,
    pcm_path: Path,
    device: torch.device,
    max_frames: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load 36-dim vocoder features and aligned PCM from prepared files.

    Crops both features and audio to at most ``max_frames`` (frames,
    10 ms/帧) for memory safety.
    """

    pcm_i16, feats = load_feature_pcm_pair(
        features_path=features_path,
        pcm_path=pcm_path,
        max_frames=max_frames,
    )
    feats36 = torch.from_numpy(feats).to(device=device, dtype=torch.float32)
    feats36 = feats36.unsqueeze(0)  # [1,T,36]
    audio = torch.from_numpy(pcm_i16.astype(np.float32) / 32768.0).to(device=device)
    audio = audio.unsqueeze(0)  # [1,L]

    return audio, feats36


def run_inference_from_dump(args: argparse.Namespace) -> None:
    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"),
    )

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    feat_path = Path(args.features).expanduser().resolve()
    pcm_path = Path(args.pcm).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Infer-Features] Using device: {device}")
    print(f"[Infer-Features] Loading checkpoint: {ckpt_path}")

    ckpt = _load_checkpoint(ckpt_path, device=device)
    model, cfg = _build_model_from_ckpt_cfg(ckpt, device=device)

    print(
        f"[Infer-Features] with_hash={getattr(model, 'with_hash', False)}, "
        f"quantizer_type={getattr(model, 'quantizer_type', 'unknown')}",
    )

    max_frames = int(args.max_frames)
    print(
        f"[Infer-Features] Loading prepared features: features={feat_path}, pcm={pcm_path}, "
        f"max_frames={max_frames}",
    )
    audio_b, feats_b = _load_feature_outputs(feat_path, pcm_path, device, max_frames)

    # Channel simulator on the same device as the model.
    channel_sim = ChannelSimulator(
        sample_rate=16000,
        frame_hz=100,
        snr_step_db=float(getattr(cfg, "snr_step_db", 1.0)),
        device=device,
    )

    # Choose forward path based on model.with_hash / cfg.content_only.
    content_only = bool(getattr(cfg, "content_only", False))
    if content_only:
        if getattr(model, "with_hash", False):
            forward_fn = model.forward_content_only  # type: ignore[attr-defined]
        else:
            forward_fn = model.forward_content_only_no_hash  # type: ignore[attr-defined]
    elif getattr(model, "with_hash", False):
        forward_fn = model.forward_with_hash
    else:
        forward_fn = model

    with torch.no_grad():
        out = forward_fn(
            audio=audio_b,
            fargan_feats=feats_b,
            channel_sim=channel_sim,
            snr_min_db=float(getattr(cfg, "snr_min_db", -5.0)),
            snr_max_db=float(getattr(cfg, "snr_max_db", 15.0)),
            target_len=audio_b.size(-1),
        )

    audio_hat = out["audio_hat"].detach().cpu().squeeze(0)
    audio_real = audio_b.detach().cpu().squeeze(0)

    # Align waveform lengths defensively before any further analysis or CSV
    # export. Depending on vocoder internals, ``audio_hat`` may be a few
    # samples shorter than ``audio_real``.
    L = min(audio_real.numel(), audio_hat.numel())
    if audio_real.numel() != audio_hat.numel():
        print(
            f"[Infer-Features][DBG] Length mismatch before align: "
            f"real={audio_real.numel()}, gen={audio_hat.numel()} -> using L={L}",
        )
    audio_real = audio_real[:L]
    audio_hat = audio_hat[:L]

    # Save waveforms.
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
    torchaudio.save(str(audio_dir / "real.wav"), audio_real.unsqueeze(0), 16000)
    torchaudio.save(str(audio_dir / "gen.wav"), audio_hat.unsqueeze(0), 16000)

    # Visualisation.
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    plot_path = plot_dir / "comparison.png"
    create_audio_comparison_plot(
        audio_real=audio_real,
        audio_gen=audio_hat,
        save_path=str(plot_path),
        sr=16000,
        title=f"Audio Comparison (from prepared features) - {ckpt_path.name}",
        show_waveform=True,
        hop_length=160,
    )

    # Optional CSV exports: waveform, F0, Bark/BFCC (real vs gen).
    try:
        from utils.audio_visualizer import extract_f0, extract_mel_spectrogram  # type: ignore

        f0_r, vmask_r, _fb_r, _fbm_r, _ = extract_f0(audio_real, sr=16000, hop_length=160)
        f0_g, vmask_g, _fb_g, _fbm_g, _ = extract_f0(audio_hat, sr=16000, hop_length=160)

        # Align F0 lengths defensively.
        len_f0 = min(len(f0_r), len(f0_g))
        if len(f0_r) != len(f0_g):
            print(
                f"[Infer-Features][DBG] F0 length mismatch: real={len(f0_r)}, "
                f"gen={len(f0_g)} -> using len_f0={len_f0}",
            )
        f0_r = f0_r[:len_f0]
        f0_g = f0_g[:len_f0]
        vmask_r = vmask_r[:len_f0]
        vmask_g = vmask_g[:len_f0]

        mel_r = extract_mel_spectrogram(audio_real, sr=16000, hop_length=160)
        mel_g = extract_mel_spectrogram(audio_hat, sr=16000, hop_length=160)

        # Align Bark/BFCC time length defensively.
        T_mel = min(mel_r.shape[1], mel_g.shape[1])
        mel_r = mel_r[:, :T_mel]
        mel_g = mel_g[:, :T_mel]

        csv_dir = out_dir / "csv"
        csv_dir.mkdir(exist_ok=True)

        # 1) Waveform CSV
        t_audio = np.arange(audio_real.numel()) / 16000.0
        np.savetxt(
            csv_dir / "waveform_real.csv",
            np.stack([t_audio, audio_real.numpy()], axis=1),
            delimiter=",",
            header="time_s,amplitude",
        )
        np.savetxt(
            csv_dir / "waveform_gen.csv",
            np.stack([t_audio, audio_hat.numpy()], axis=1),
            delimiter=",",
            header="time_s,amplitude",
        )

        # 2) F0 CSV
        t_f0 = np.arange(len_f0) * 0.01
        f0_real_out = np.stack([t_f0, f0_r, vmask_r.astype(np.float32)], axis=1)
        f0_gen_out = np.stack([t_f0, f0_g, vmask_g.astype(np.float32)], axis=1)
        np.savetxt(csv_dir / "f0_real.csv", f0_real_out, delimiter=",", header="time_s,f0_hz,voiced_mask")
        np.savetxt(csv_dir / "f0_gen.csv", f0_gen_out, delimiter=",", header="time_s,f0_hz,voiced_mask")

        # 3) Bark/BFCC CSV
        t_mel = np.arange(T_mel) * 0.01
        mel_real_out = np.concatenate([t_mel[None, :], mel_r], axis=0).T
        mel_gen_out = np.concatenate([t_mel[None, :], mel_g], axis=0).T
        header_mel = ",".join(["time_s"] + [f"bark_{i}" for i in range(mel_r.shape[0])])
        np.savetxt(csv_dir / "bark_real.csv", mel_real_out, delimiter=",", header=header_mel)
        np.savetxt(csv_dir / "bark_gen.csv", mel_gen_out, delimiter=",", header=header_mel)

        print(f"[Infer-Features] CSVs saved under {csv_dir}")
    except Exception as exc:  # pragma: no cover - best effort
        print(f"[Infer-Features] WARNING: failed to export CSV diagnostics: {exc}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DBP-JSCC inference from prepared feature/PCM dumps")
    p.add_argument("--ckpt", type=str, required=True, help="Path to a DBP-JSCC checkpoint")
    p.add_argument("--features", type=str, required=True, help="Path to prepared 36D vocoder feature .f32 file")
    p.add_argument("--pcm", type=str, required=True, help="Path to aligned PCM .pcm file")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for audio/plots/CSVs")
    p.add_argument("--device", type=str, default=None, help="Device override, e.g. 'cuda:0' or 'cpu'")
    p.add_argument("--max_frames", type=int, default=400, help="Maximum number of frames to use (default 400 ≈ 4s)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_inference_from_dump(args)


if __name__ == "__main__":
    main()
