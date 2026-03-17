#!/usr/bin/env python3
"""Offline inference helper for DBP-JSCC checkpoints.

This script loads a :class:`DualBranchBarkJSCC` model from a
DBP-JSCC checkpoint and runs one full forward pass
on a single WAV file, then visualises real vs generated audio
using :mod:`utils.audio_visualizer`.

Key properties
--------------
- Uses the *checkpoint's* saved config (``ckpt["cfg"]``) as the
  base for all model hyper‑parameters, including whether hash/RVQ
  is enabled (``with_hash``, ``quantizer_type`` etc.).
- If the checkpoint was trained with hash/RVQ enabled, the model
  forward path will also go through the hash/RVQ bottlenecks.
  If not, it will fall back to the non‑hash forward path.
- 36-dim vocoder features are extracted from the input audio with a
  compatible external feature extractor, without relying on any
  on-disk dataset layout.
- Visualisation is done via
  :func:`utils.audio_visualizer.create_audio_comparison_plot`.
  In addition, three CSV files are exported with the raw time
  series / F0 / Bark/BFCC data for external plotting.

Usage example
-------------

.. code-block:: bash

    python scripts/infer_wav.py \
        --ckpt ./outputs/checkpoints/checkpoint_step_010000_epoch_00.pth \
        --wav ./examples/test.wav \
        --feature_bin /home/bluestar/fargan_demo/fargan_demo \
        --out_dir ./outputs/infer_wav

The script will create ``audio/`` and ``plots/`` sub‑folders under
``out_dir`` and save both ``.wav`` and ``.png`` files for the
comparison.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torchaudio

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in map(str, __import__("sys").path):
    __import__("sys").path.insert(0, str(ROOT_DIR))

from models.dual_branch_bark_jscc import DualBranchBarkJSCC  # type: ignore
from training.train_support import build_model  # type: ignore
from utils.channel_sim import ChannelSimulator  # type: ignore
from utils.audio_visualizer import create_audio_comparison_plot  # type: ignore
from utils.feature_extraction import (  # type: ignore
    concatenate_inputs_to_pcm,
    load_feature_pcm_pair,
    resolve_feature_extractor,
    run_feature_extractor,
)


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

    # Build a lightweight shim config object that exposes the same
    # attribute interface expected by ``build_model``. This mirrors
    # the approach used in ``training/train.py`` so that
    # checkpoints trained with the simplified config can be reused
    # without depending on the exact support-config signature.
    class _ShimCfg:
        pass

    cfg = _ShimCfg()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)

    # Ensure device is set to the runtime device.
    setattr(cfg, "device", str(device))

    model = build_model(cfg).to(device)

    # Restore model weights; support multiple common keys.
    sd = None
    for key in ("model_state_dict", "model", "state_dict"):
        if key in ckpt:
            sd = ckpt[key]
            break
    if sd is None:
        raise RuntimeError("Checkpoint does not contain model weights (model_state_dict/model/state_dict)")

    cur_sd = model.state_dict()
    safe_sd = {}
    for k, v in sd.items():
        if k in cur_sd and isinstance(v, torch.Tensor) and v.shape == cur_sd[k].shape:
            safe_sd[k] = v
    model.load_state_dict(safe_sd, strict=False)
    model.eval()

    return model, cfg


def _run_feature_extractor_on_wav(
    wav_path: Path,
    feature_bin: Path,
    work_dir: Path,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Use a compatible external feature extractor to build 36-dim vocoder features."""

    work_dir.mkdir(parents=True, exist_ok=True)

    in_pcm = work_dir / "input.s16"
    feat_path = work_dir / "features.f32"
    pcm_out_path = work_dir / "out_speech.pcm"

    concatenate_inputs_to_pcm([wav_path], in_pcm)
    meta = run_feature_extractor(
        input_pcm=in_pcm,
        output_features=feat_path,
        output_pcm=pcm_out_path,
        extractor_bin=feature_bin,
    )
    print(
        "[Infer] Feature extraction finished: "
        f"mode={meta['extractor_mode']}, frames={meta['feature_frames']}, "
        f"duration={meta['duration_s']:.2f}s",
    )

    audio_out_i16, feat_np = load_feature_pcm_pair(feat_path, pcm_out_path)
    audio_real = torch.from_numpy(audio_out_i16.astype(np.float32) / 32768.0).to(device=device).unsqueeze(0)
    feats36 = torch.from_numpy(feat_np).to(device=device, dtype=torch.float32).unsqueeze(0)

    return audio_real, feats36


def _extract_fargan_feats_from_audio(*args: Any, **kwargs: Any) -> torch.Tensor:  # pragma: no cover - legacy stub
    """Legacy stub kept for backward compatibility.

    The current inference path uses an external feature extractor to
    obtain aligned 36-dim vocoder features, so this function is no
    longer used.
    """

    raise RuntimeError(
        "_extract_fargan_feats_from_audio is deprecated; use the feature-extractor path instead",
    )


def run_single_wav_inference(args: argparse.Namespace) -> None:
    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    wav_path = Path(args.wav).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Infer] Using device: {device}")
    print(f"[Infer] Loading checkpoint: {ckpt_path}")

    ckpt = _load_checkpoint(ckpt_path, device=device)
    model, cfg = _build_model_from_ckpt_cfg(ckpt, device=device)

    print(
        f"[Infer] with_hash={getattr(model, 'with_hash', False)}, "
        f"quantizer_type={getattr(model, 'quantizer_type', 'unknown')}"
    )

    print(f"[Infer] Loading WAV & extracting 36D vocoder features: {wav_path}")
    feature_bin = resolve_feature_extractor(args.feature_bin or os.environ.get("VOCODER_FEATURE_BIN"))

    tmp_dir = out_dir / "dump_tmp"
    audio_b, feats_b = _run_feature_extractor_on_wav(wav_path, feature_bin, tmp_dir, device)

    # Channel simulator; SNR range from cfg (used by forward_with_hash).
    # Pass the runtime device explicitly so that CSI tensors live on
    # the same device as the model (important for CPU-only inference
    # when CUDA is available).
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

    # Save audio.
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
        title=f"Audio Comparison - {ckpt_path.name}",
        show_waveform=True,
        hop_length=160,
    )

    # Optional CSV exports: waveform, F0, Bark/BFCC (real vs gen).
    try:
        from utils.audio_visualizer import extract_f0, extract_mel_spectrogram  # type: ignore

        f0_r, vmask_r, _fb_r, _fbm_r, _ = extract_f0(audio_real, sr=16000, hop_length=160)
        f0_g, vmask_g, _fb_g, _fbm_g, _ = extract_f0(audio_hat, sr=16000, hop_length=160)

        mel_r = extract_mel_spectrogram(audio_real, sr=16000, hop_length=160)
        mel_g = extract_mel_spectrogram(audio_hat, sr=16000, hop_length=160)

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
        t_f0 = np.arange(len(f0_r)) * 0.01
        f0_real_out = np.stack([t_f0, f0_r, vmask_r.astype(np.float32)], axis=1)
        f0_gen_out = np.stack([t_f0, f0_g, vmask_g.astype(np.float32)], axis=1)
        np.savetxt(csv_dir / "f0_real.csv", f0_real_out, delimiter=",", header="time_s,f0_hz,voiced_mask")
        np.savetxt(csv_dir / "f0_gen.csv", f0_gen_out, delimiter=",", header="time_s,f0_hz,voiced_mask")

        # 3) Bark/BFCC CSV (flattened time axis as rows).
        # Each row: t, mel_bin_0, ..., mel_bin_{F-1}
        T_mel = mel_r.shape[1]
        t_mel = np.arange(T_mel) * 0.01
        mel_real_out = np.concatenate([t_mel[None, :], mel_r], axis=0).T
        mel_gen_out = np.concatenate([t_mel[None, :], mel_g], axis=0).T
        header_mel = ",".join(["time_s"] + [f"mel_{i}" for i in range(mel_r.shape[0])])
        np.savetxt(csv_dir / "mel_real.csv", mel_real_out, delimiter=",", header=header_mel)
        np.savetxt(csv_dir / "mel_gen.csv", mel_gen_out, delimiter=",", header=header_mel)

        print(f"[Infer] CSVs saved under {csv_dir}")
    except Exception as exc:  # pragma: no cover - best effort
        print(f"[Infer] WARNING: failed to export CSV diagnostics: {exc}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DBP-JSCC single-WAV inference")
    p.add_argument("--ckpt", type=str, required=True, help="Path to a DBP-JSCC checkpoint")
    p.add_argument("--wav", type=str, required=True, help="Input WAV file (16 kHz recommended)")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for audio/plots/CSVs")
    p.add_argument("--feature_bin", type=str, default=None, help="Path to external feature extractor. Compatible with dump_data -train and fargan_demo -features")
    p.add_argument("--device", type=str, default=None, help="Device override, e.g. 'cuda:0' or 'cpu'")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_single_wav_inference(args)


if __name__ == "__main__":
    main()
