#!/usr/bin/env python3
"""Prepare DBP-JSCC training data from public audio files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.feature_extraction import (  # type: ignore
    concatenate_inputs_to_pcm,
    resolve_feature_extractor,
    run_feature_extractor,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare aligned DBP-JSCC training data from audio files")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input audio files. WAV/FLAC and raw 16-bit PCM (.pcm/.s16) are supported.",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="./data",
        help="Output dataset root. Writes out_speech.pcm and lmr_export/features_36_vocoder_baseline.f32.",
    )
    parser.add_argument(
        "--feature_bin",
        type=str,
        default=None,
        help="Path to an external feature extractor. Compatible with dump_data -train and fargan_demo -features.",
    )
    parser.add_argument(
        "--extractor_mode",
        type=str,
        choices=["auto", "dump_data", "vocoder_demo"],
        default="auto",
        help="Force the extractor interface instead of auto-detecting it.",
    )
    parser.add_argument(
        "--write_legacy_name",
        action="store_true",
        help="Also write lmr_export/features_36_fargan_baseline.f32 for backward compatibility.",
    )
    parser.add_argument(
        "--keep_tmp",
        action="store_true",
        help="Keep the intermediate merged_input.s16 file under out_root/tmp_prepare.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    out_root = Path(args.out_root).expanduser().resolve()
    lmr_dir = out_root / "lmr_export"
    tmp_dir = out_root / "tmp_prepare"
    lmr_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    input_paths = [Path(item).expanduser().resolve() for item in args.inputs]
    merged_input = tmp_dir / "merged_input.s16"
    input_meta = concatenate_inputs_to_pcm(input_paths, merged_input)

    extractor_bin = resolve_feature_extractor(args.feature_bin)
    feature_path = lmr_dir / "features_36_vocoder_baseline.f32"
    pcm_path = out_root / "out_speech.pcm"

    extract_meta = run_feature_extractor(
        input_pcm=merged_input,
        output_features=feature_path,
        output_pcm=pcm_path,
        extractor_bin=extractor_bin,
        extractor_mode=args.extractor_mode,
    )

    if args.write_legacy_name:
        legacy_path = lmr_dir / "features_36_fargan_baseline.f32"
        legacy_path.write_bytes(feature_path.read_bytes())

    manifest = {
        "dataset_root": str(out_root),
        "feature_file": str(feature_path),
        "pcm_file": str(pcm_path),
        "input": input_meta,
        "extractor": extract_meta,
    }
    (out_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    if not args.keep_tmp and merged_input.exists():
        merged_input.unlink()
        if not any(tmp_dir.iterdir()):
            tmp_dir.rmdir()

    print(f"[PrepareDataset] Dataset root: {out_root}")
    print(f"[PrepareDataset] PCM: {pcm_path}")
    print(f"[PrepareDataset] Features: {feature_path}")
    print(f"[PrepareDataset] Frames: {extract_meta['feature_frames']}")
    print(f"[PrepareDataset] Extractor: {extract_meta['extractor_bin']} ({extract_meta['extractor_mode']})")


if __name__ == "__main__":
    main()
