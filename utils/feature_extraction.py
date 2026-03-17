#!/usr/bin/env python3
"""Feature extraction helpers for DBP-JSCC."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

FEATURE_DIM = 36
FRAME_SIZE = 160
TARGET_SAMPLE_RATE = 16000


def resolve_feature_extractor(explicit_path: Optional[str] = None) -> Path:
    """Resolve a compatible external feature extractor binary."""

    candidates: List[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())

    for key in ("VOCODER_FEATURE_BIN", "FEATURE_EXTRACT_BIN", "FARGAN_DEMO_BIN", "DUMP_DATA_BIN"):
        value = os.environ.get(key)
        if value:
            candidates.append(Path(value).expanduser())

    for raw in (
        "/home/bluestar/fargan_demo/fargan_demo",
        "/home/bluestar/FARGAN/opus/dump_data",
    ):
        candidates.append(Path(raw))

    for name in ("fargan_demo", "dump_data"):
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate

    raise FileNotFoundError(
        "No compatible feature extractor found. Set --feature_bin or "
        "VOCODER_FEATURE_BIN / FEATURE_EXTRACT_BIN.",
    )


def detect_feature_extractor_mode(bin_path: Path) -> str:
    """Return ``dump_data`` or ``vocoder_demo`` for the given binary."""

    name = bin_path.name.lower()
    if "dump_data" in name:
        return "dump_data"
    if "fargan_demo" in name or "lpcnet_demo" in name:
        return "vocoder_demo"

    try:
        proc = subprocess.run(
            [str(bin_path), "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        text = f"{proc.stdout}\n{proc.stderr}".lower()
    except Exception:
        return "vocoder_demo"

    if "-train" in text:
        return "dump_data"
    if "-features" in text:
        return "vocoder_demo"
    return "vocoder_demo"


def load_audio_as_pcm16(audio_path: Path, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Load WAV/FLAC or raw PCM and return mono int16 samples."""

    if audio_path.suffix.lower() in {".pcm", ".s16", ".raw"}:
        pcm = np.fromfile(str(audio_path), dtype=np.int16)
        if pcm.size == 0:
            raise RuntimeError(f"Empty PCM file: {audio_path}")
        return pcm.astype(np.int16, copy=False)

    try:
        import soundfile as sf
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("soundfile is required to load WAV/FLAC inputs") from exc

    wav, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    wav = np.asarray(wav, dtype=np.float32)
    if sr != target_sr:
        try:
            from scipy.signal import resample_poly
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("scipy is required to resample non-16k audio") from exc

        gcd = math.gcd(int(sr), int(target_sr))
        wav = resample_poly(wav, target_sr // gcd, sr // gcd).astype(np.float32, copy=False)

    wav = np.clip(wav, -1.0, 1.0)
    return np.round(wav * 32767.0).astype(np.int16)


def concatenate_inputs_to_pcm(inputs: Sequence[Path], output_pcm: Path) -> Dict[str, object]:
    """Convert and concatenate input audio files into one PCM stream."""

    if not inputs:
        raise ValueError("At least one input file is required")

    output_pcm.parent.mkdir(parents=True, exist_ok=True)
    sample_counts: List[int] = []
    with open(output_pcm, "wb") as handle:
        for path in inputs:
            pcm = load_audio_as_pcm16(path)
            sample_counts.append(int(pcm.size))
            handle.write(pcm.tobytes())

    total_samples = int(sum(sample_counts))
    return {
        "inputs": [str(path) for path in inputs],
        "input_sample_counts": sample_counts,
        "total_samples": total_samples,
        "duration_s": float(total_samples / TARGET_SAMPLE_RATE),
    }


def _count_feature_frames(features_path: Path, feature_dim: int = FEATURE_DIM) -> int:
    feat_np = np.fromfile(str(features_path), dtype=np.float32)
    if feat_np.size == 0:
        raise RuntimeError(f"Feature extractor produced an empty file: {features_path}")
    if feat_np.size % feature_dim != 0:
        raise RuntimeError(
            f"Invalid feature file {features_path}: {feat_np.size} floats is not divisible by {feature_dim}",
        )
    return int(feat_np.size // feature_dim)


def run_feature_extractor(
    input_pcm: Path,
    output_features: Path,
    output_pcm: Path,
    extractor_bin: Path,
    extractor_mode: str = "auto",
    feature_dim: int = FEATURE_DIM,
    frame_size: int = FRAME_SIZE,
) -> Dict[str, object]:
    """Run a compatible feature extractor and return alignment metadata."""

    mode = extractor_mode if extractor_mode != "auto" else detect_feature_extractor_mode(extractor_bin)
    if mode not in {"dump_data", "vocoder_demo"}:
        raise ValueError(f"Unsupported extractor mode: {mode}")

    output_features.parent.mkdir(parents=True, exist_ok=True)
    output_pcm.parent.mkdir(parents=True, exist_ok=True)

    if mode == "dump_data":
        subprocess.run(
            [str(extractor_bin), "-train", str(input_pcm), str(output_features), str(output_pcm)],
            check=True,
        )
    else:
        subprocess.run(
            [str(extractor_bin), "-features", str(input_pcm), str(output_features)],
            check=True,
        )
        pcm_i16 = np.fromfile(str(input_pcm), dtype=np.int16)
        if pcm_i16.size == 0:
            raise RuntimeError(f"Input PCM is empty: {input_pcm}")
        feature_frames = _count_feature_frames(output_features, feature_dim=feature_dim)
        expected_pcm_samples = feature_frames * frame_size
        if pcm_i16.size < expected_pcm_samples:
            raise RuntimeError(
                "Feature stream is longer than the input PCM supports: "
                f"{feature_frames} frames -> {expected_pcm_samples} samples, "
                f"but PCM has only {pcm_i16.size} samples",
            )
        output_pcm.write_bytes(pcm_i16[:expected_pcm_samples].tobytes())

    feature_frames = _count_feature_frames(output_features, feature_dim=feature_dim)
    pcm_i16 = np.fromfile(str(output_pcm), dtype=np.int16)
    expected_pcm_samples = feature_frames * frame_size
    if pcm_i16.size < expected_pcm_samples:
        raise RuntimeError(
            f"Aligned PCM is too short: {pcm_i16.size} samples for {feature_frames} frames",
        )
    if pcm_i16.size != expected_pcm_samples:
        pcm_i16 = pcm_i16[:expected_pcm_samples]
        output_pcm.write_bytes(pcm_i16.tobytes())

    return {
        "extractor_bin": str(extractor_bin),
        "extractor_mode": mode,
        "feature_frames": feature_frames,
        "pcm_samples": int(pcm_i16.size),
        "duration_s": float(pcm_i16.size / TARGET_SAMPLE_RATE),
    }


def load_feature_pcm_pair(
    features_path: Path,
    pcm_path: Path,
    feature_dim: int = FEATURE_DIM,
    frame_size: int = FRAME_SIZE,
    max_frames: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load an aligned feature/PCM pair as numpy arrays."""

    feat_np = np.fromfile(str(features_path), dtype=np.float32)
    if feat_np.size == 0 or feat_np.size % feature_dim != 0:
        raise RuntimeError(f"Invalid features file: {features_path}")

    total_frames = feat_np.size // feature_dim
    use_frames = total_frames if max_frames is None else min(total_frames, int(max_frames))
    features = feat_np.reshape(total_frames, feature_dim)[:use_frames]

    pcm_i16 = np.fromfile(str(pcm_path), dtype=np.int16)
    if pcm_i16.size == 0:
        raise RuntimeError(f"Empty PCM file: {pcm_path}")

    target_samples = use_frames * frame_size
    if pcm_i16.size < target_samples:
        raise RuntimeError(
            f"PCM file {pcm_path} has {pcm_i16.size} samples but {target_samples} are required",
        )
    pcm_i16 = pcm_i16[:target_samples]
    return pcm_i16, features
