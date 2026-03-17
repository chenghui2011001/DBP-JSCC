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
- FARGAN 36‑dim features are derived directly from the input audio
  via :class:`Feature48To36Adapter`, without relying on any on‑disk
  dataset layout.
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
        --out_dir ./outputs/infer_wav

The script will create ``audio/`` and ``plots/`` sub‑folders under
``out_dir`` and save both ``.wav`` and ``.png`` files for the
comparison.
"""

from __future__ import annotations

import argparse
import os
import subprocess
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


def _run_dump_data_on_wav(
    wav_path: Path,
    dump_bin: Path,
    work_dir: Path,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Use external ``dump_data`` to extract FARGAN 36‑dim features.

    The binary is expected to follow the LPCNet/FARGAN interface::

        dump_data -train input.s16 output_features.f32 output_pcm.pcm

    where ``output_features.f32`` contains 36‑dim features at 100 Hz
    and ``output_pcm.pcm`` is the aligned training waveform.

    Returns
    -------
    audio_real : torch.Tensor
        Real audio waveform [1, L] reconstructed by ``dump_data``.
    feats36 : torch.Tensor
        FARGAN features [1, T, 36] for JSCC input.
    """

    work_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load WAV, convert to 16 kHz mono and then to int16 PCM.
    wav, sr = torchaudio.load(str(wav_path))
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    audio_f = wav.squeeze(0).to(torch.float32)
    audio_clamped = torch.clamp(audio_f, -1.0, 1.0)
    audio_i16 = (audio_clamped * 32767.0).round().to(torch.int16).cpu().numpy()

    in_pcm = work_dir / "input.s16"
    with open(in_pcm, "wb") as f_pcm:
        f_pcm.write(audio_i16.tobytes())

    # 2) Run dump_data in training mode to get aligned PCM + features.
    feat_path = work_dir / "features.f32"
    pcm_out_path = work_dir / "out_speech.pcm"

    cmd = [str(dump_bin), "-train", str(in_pcm), str(feat_path), str(pcm_out_path)]
    print(f"[Infer] Running dump_data: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # 3) Load aligned PCM as real audio.
    audio_out_i16 = np.fromfile(str(pcm_out_path), dtype=np.int16)
    if audio_out_i16.size == 0:
        raise RuntimeError(f"dump_data produced empty PCM: {pcm_out_path}")
    audio_real = torch.from_numpy(audio_out_i16.astype(np.float32) / 32768.0).to(device=device)
    audio_real = audio_real.unsqueeze(0)  # [1, L]

    # 4) Load 36‑dim FARGAN features.
    feat_np = np.fromfile(str(feat_path), dtype=np.float32)
    if feat_np.size == 0:
        raise RuntimeError(f"dump_data produced empty feature file: {feat_path}")
    if feat_np.size % 36 != 0:
        raise RuntimeError(
            f"Unexpected feature length {feat_np.size}; expected multiple of 36 (got {feat_np.size/36:.3f} frames)"
        )
    T = feat_np.size // 36
    feats36 = torch.from_numpy(feat_np.reshape(T, 36)).to(device=device, dtype=torch.float32)
    feats36 = feats36.unsqueeze(0)  # [1, T, 36]

    return audio_real, feats36


def _extract_fargan_feats_from_audio(*args: Any, **kwargs: Any) -> torch.Tensor:  # pragma: no cover - legacy stub
    """Legacy stub kept for backward compatibility.

    The current inference path uses ``dump_data`` to obtain FARGAN
    36‑dim features, so this function is no longer used.
    """

    raise RuntimeError(
        "_extract_fargan_feats_from_audio is deprecated; use dump_data-based path instead",
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

    print(f"[Infer] Loading WAV & extracting features via dump_data: {wav_path}")

    dump_bin = Path(os.environ.get("DUMP_DATA_BIN", "/home/bluestar/FARGAN/opus/dump_data")).expanduser()
    if not dump_bin.is_file():
        raise FileNotFoundError(
            f"dump_data binary not found: {dump_bin}. Set DUMP_DATA_BIN env to override.",
        )

    tmp_dir = out_dir / "dump_tmp"
    audio_b, feats_b = _run_dump_data_on_wav(wav_path, dump_bin, tmp_dir, device)

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
    p.add_argument("--device", type=str, default=None, help="Device override, e.g. 'cuda:0' or 'cpu'")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_single_wav_inference(args)


if __name__ == "__main__":
    main()
