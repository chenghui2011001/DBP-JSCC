#!/usr/bin/env python3
"""DBP-JSCC bits-only inference.

This script runs the deployment-style path:

1. waveform/features -> quantized bitstreams via ``encode_quant_codec``
2. bitstreams -> audio via ``decode_from_bits_offline``

It intentionally keeps only inference-time CLI arguments and avoids
training-only options.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in map(str, sys.path):
    sys.path.insert(0, str(ROOT_DIR))

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


def _build_model_from_ckpt_cfg(ckpt: Dict[str, Any], device: torch.device) -> Tuple[Any, Any]:
    from training.train_support import build_model  # type: ignore

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


def _run_feature_extractor_on_wav(
    wav_path: Path,
    feature_bin: Path,
    work_dir: Path,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
        "[BitsOnly] Feature extraction finished: "
        f"mode={meta['extractor_mode']}, frames={meta['feature_frames']}, "
        f"duration={meta['duration_s']:.2f}s",
    )

    return _load_feature_outputs(feat_path, pcm_out_path, device=device, max_frames=None)


def _load_feature_outputs(
    features_path: Path,
    pcm_path: Path,
    device: torch.device,
    max_frames: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    pcm_i16, feats = load_feature_pcm_pair(
        features_path=features_path,
        pcm_path=pcm_path,
        max_frames=max_frames,
    )
    audio = torch.from_numpy(pcm_i16.astype(np.float32) / 32768.0).to(device=device).unsqueeze(0)
    feats36 = torch.from_numpy(feats).to(device=device, dtype=torch.float32).unsqueeze(0)
    return audio, feats36


def _save_diagnostics(
    out_dir: Path,
    ckpt_name: str,
    audio_real: torch.Tensor,
    audio_gen: torch.Tensor,
) -> None:
    import torchaudio
    from utils.audio_visualizer import create_audio_comparison_plot  # type: ignore

    audio_dir = out_dir / "audio"
    plot_dir = out_dir / "plots"
    csv_dir = out_dir / "csv"
    audio_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    torchaudio.save(str(audio_dir / "real.wav"), audio_real.unsqueeze(0), 16000)
    torchaudio.save(str(audio_dir / "gen.wav"), audio_gen.unsqueeze(0), 16000)

    create_audio_comparison_plot(
        audio_real=audio_real,
        audio_gen=audio_gen,
        save_path=str(plot_dir / "comparison.png"),
        sr=16000,
        title=f"Bits-only Audio Comparison - {ckpt_name}",
        show_waveform=True,
        hop_length=160,
    )

    try:
        from utils.audio_visualizer import extract_f0, extract_mel_spectrogram  # type: ignore

        f0_r, vmask_r, *_ = extract_f0(audio_real, sr=16000, hop_length=160)
        f0_g, vmask_g, *_ = extract_f0(audio_gen, sr=16000, hop_length=160)
        len_f0 = min(len(f0_r), len(f0_g))
        f0_r = f0_r[:len_f0]
        f0_g = f0_g[:len_f0]
        vmask_r = vmask_r[:len_f0]
        vmask_g = vmask_g[:len_f0]

        bark_r = extract_mel_spectrogram(audio_real, sr=16000, hop_length=160)
        bark_g = extract_mel_spectrogram(audio_gen, sr=16000, hop_length=160)
        t_bark = min(bark_r.shape[1], bark_g.shape[1])
        bark_r = bark_r[:, :t_bark]
        bark_g = bark_g[:, :t_bark]

        t_audio = np.arange(audio_real.numel()) / 16000.0
        np.savetxt(
            csv_dir / "waveform_real.csv",
            np.stack([t_audio, audio_real.numpy()], axis=1),
            delimiter=",",
            header="time_s,amplitude",
        )
        np.savetxt(
            csv_dir / "waveform_gen.csv",
            np.stack([t_audio, audio_gen.numpy()], axis=1),
            delimiter=",",
            header="time_s,amplitude",
        )

        t_f0 = np.arange(len_f0) * 0.01
        np.savetxt(
            csv_dir / "f0_real.csv",
            np.stack([t_f0, f0_r, vmask_r.astype(np.float32)], axis=1),
            delimiter=",",
            header="time_s,f0_hz,voiced_mask",
        )
        np.savetxt(
            csv_dir / "f0_gen.csv",
            np.stack([t_f0, f0_g, vmask_g.astype(np.float32)], axis=1),
            delimiter=",",
            header="time_s,f0_hz,voiced_mask",
        )

        t_spec = np.arange(t_bark) * 0.01
        bark_real_out = np.concatenate([t_spec[None, :], bark_r], axis=0).T
        bark_gen_out = np.concatenate([t_spec[None, :], bark_g], axis=0).T
        header = ",".join(["time_s"] + [f"bark_{i}" for i in range(bark_r.shape[0])])
        np.savetxt(csv_dir / "bark_real.csv", bark_real_out, delimiter=",", header=header)
        np.savetxt(csv_dir / "bark_gen.csv", bark_gen_out, delimiter=",", header=header)
    except Exception as exc:  # pragma: no cover - best effort
        print(f"[BitsOnly] WARNING: failed to export CSV diagnostics: {exc}")


def _serialize_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, tuple):
            out[k] = list(v)
        else:
            out[k] = v
    return out


def run_bits_only_inference(args: argparse.Namespace) -> None:
    from utils.channel_sim import ChannelSimulator  # type: ignore

    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = _load_checkpoint(ckpt_path, device=device)
    model, cfg = _build_model_from_ckpt_cfg(ckpt, device=device)

    if args.wav is not None:
        feature_bin = resolve_feature_extractor(args.feature_bin)
        audio_b, feats_b = _run_feature_extractor_on_wav(
            wav_path=Path(args.wav).expanduser().resolve(),
            feature_bin=feature_bin,
            work_dir=out_dir / "dump_tmp",
            device=device,
        )
    else:
        if args.features is None or args.pcm is None:
            raise RuntimeError("Either --wav or --features/--pcm must be provided")
        audio_b, feats_b = _load_feature_outputs(
            features_path=Path(args.features).expanduser().resolve(),
            pcm_path=Path(args.pcm).expanduser().resolve(),
            device=device,
            max_frames=args.max_frames,
        )

    channel_sim = ChannelSimulator(
        sample_rate=16000,
        frame_hz=100,
        snr_step_db=float(getattr(cfg, "snr_step_db", 1.0)),
        device=device,
    )

    snr_db = float(args.snr_db)
    with torch.no_grad():
        bits_c, bits_f, bits_s, meta = model.encode_quant_codec(
            audio=audio_b,
            fargan_feats=feats_b,
            channel_sim=channel_sim,
            snr_min_db=snr_db,
            snr_max_db=snr_db,
            return_meta=True,
            use_noisy_bits=bool(args.use_noisy_bits),
        )

        csi_vec = meta.get("csi_vec", None)
        if isinstance(csi_vec, np.ndarray):
            csi_vec = torch.from_numpy(csi_vec).to(device=device, dtype=torch.float32)
        content_hw = meta.get("hw", None)
        if isinstance(content_hw, list):
            content_hw = tuple(content_hw)

        out_bits = model.decode_from_bits_offline(
            bits_content=bits_c,
            bits_f0=bits_f,
            bits_stats=bits_s,
            target_len=audio_b.size(-1),
            csi_vec=csi_vec,
            snr_db=snr_db,
            content_hw=content_hw,
        )

    audio_real = audio_b.detach().cpu().squeeze(0)
    audio_hat = out_bits["audio_hat"].detach().cpu().squeeze(0)
    l = min(audio_real.numel(), audio_hat.numel())
    audio_real = audio_real[:l]
    audio_hat = audio_hat[:l]

    _save_diagnostics(out_dir=out_dir, ckpt_name=ckpt_path.name, audio_real=audio_real, audio_gen=audio_hat)

    meta_out = {
        "ckpt": str(ckpt_path),
        "snr_db": snr_db,
        "use_noisy_bits": bool(args.use_noisy_bits),
        "quantizer_type": str(getattr(model, "quantizer_type", "unknown")),
        "with_hash": bool(getattr(model, "with_hash", False)),
        "bit_shapes": {
            "content": None if bits_c is None else list(bits_c.shape),
            "f0": None if bits_f is None else list(bits_f.shape),
            "stats": None if bits_s is None else list(bits_s.shape),
        },
        "meta": _serialize_meta(meta),
    }
    (out_dir / "bits_only_summary.json").write_text(
        json.dumps(meta_out, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    if args.save_bits:
        torch.save(
            {
                "bits_content": None if bits_c is None else bits_c.detach().cpu(),
                "bits_f0": None if bits_f is None else bits_f.detach().cpu(),
                "bits_stats": None if bits_s is None else bits_s.detach().cpu(),
                "meta": _serialize_meta(meta),
                "snr_db": snr_db,
                "use_noisy_bits": bool(args.use_noisy_bits),
            },
            out_dir / "bitstream.pt",
        )

    print(f"[BitsOnly] Finished. Outputs saved under {out_dir}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DBP-JSCC bits-only inference")
    p.add_argument("--ckpt", type=str, required=True, help="Path to a DBP-JSCC checkpoint")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for audio/plots/CSVs")
    p.add_argument("--snr_db", type=float, default=0.0, help="Receiver-side SNR used for bits-only decode")
    p.add_argument("--device", type=str, default=None, help="Device override, e.g. 'cuda:0' or 'cpu'")
    p.add_argument("--wav", type=str, default=None, help="Input WAV/FLAC file. If set, a compatible feature extractor will be used internally")
    p.add_argument("--feature_bin", type=str, default=None, help="Path to external feature extractor. Compatible with dump_data -train and fargan_demo -features")
    p.add_argument("--dump_data_bin", dest="feature_bin", type=str, help=argparse.SUPPRESS)
    p.add_argument("--features", type=str, default=None, help="Path to a precomputed 36D vocoder feature .f32 file")
    p.add_argument("--pcm", type=str, default=None, help="Path to aligned PCM .pcm file for --features mode")
    p.add_argument("--max_frames", type=int, default=400, help="Maximum number of frames for --features/--pcm mode")
    p.add_argument("--use_noisy_bits", action="store_true", help="Use noisy bits from encode path instead of clean hard bits")
    p.add_argument("--save_bits", action="store_true", help="Save exported bitstreams to bitstream.pt")
    args = p.parse_args()

    if args.wav is None and (args.features is None or args.pcm is None):
        p.error("Provide either --wav, or both --features and --pcm")
    if args.wav is not None and (args.features is not None or args.pcm is not None):
        p.error("Use either --wav mode or --features/--pcm mode, not both")
    return args


def main() -> None:
    args = _parse_args()
    run_bits_only_inference(args)


if __name__ == "__main__":
    main()
