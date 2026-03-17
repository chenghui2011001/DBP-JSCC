#!/usr/bin/env python3
"""Export lightweight release assets from full training checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Any

import torch


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _checkpoint_basename(global_step: Any, epoch: Any) -> str:
    step_str = "unknown" if global_step is None else str(global_step)
    epoch_str = "unknown" if epoch is None else str(epoch)
    return f"step{step_str}_epoch{epoch_str}"


def export_assets(
    full_ckpt_path: Path,
    vocoder_ckpt_path: Path,
    output_dir: Path,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    full_ckpt = torch.load(full_ckpt_path, map_location="cpu")
    if not isinstance(full_ckpt, dict):
        raise RuntimeError(f"Expected dict checkpoint, got {type(full_ckpt).__name__}")
    if "model_state_dict" not in full_ckpt:
        raise RuntimeError("Checkpoint does not contain 'model_state_dict'")

    basename = _checkpoint_basename(full_ckpt.get("global_step"), full_ckpt.get("epoch"))

    model_only_path = output_dir / f"dbp_jscc_model_only_{basename}.pth"
    resume_full_path = output_dir / f"dbp_jscc_resume_full_{basename}.pth"
    vocoder_out_path = output_dir / "dbp_jscc_vocoder_pretrained.pth"
    metadata_path = output_dir / "release_metadata.json"
    sha_path = output_dir / "SHA256SUMS.txt"

    model_only = {
        "checkpoint_type": "model_only",
        "format_version": 1,
        "model_state_dict": full_ckpt["model_state_dict"],
        "cfg": full_ckpt.get("cfg", {}),
        "global_step": full_ckpt.get("global_step"),
        "epoch": full_ckpt.get("epoch"),
    }
    torch.save(model_only, model_only_path)

    shutil.copy2(full_ckpt_path, resume_full_path)
    shutil.copy2(vocoder_ckpt_path, vocoder_out_path)

    metadata = {
        "model_only": model_only_path.name,
        "resume_full": resume_full_path.name,
        "vocoder": vocoder_out_path.name,
        "source_full_checkpoint": str(full_ckpt_path),
        "source_vocoder_checkpoint": str(vocoder_ckpt_path),
        "global_step": full_ckpt.get("global_step"),
        "epoch": full_ckpt.get("epoch"),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    assets = [model_only_path, resume_full_path, vocoder_out_path, metadata_path]
    sha_lines = [f"{_sha256(path)}  {path.name}" for path in assets]
    sha_path.write_text("\n".join(sha_lines) + "\n", encoding="utf-8")

    return {
        "model_only": model_only_path,
        "resume_full": resume_full_path,
        "vocoder": vocoder_out_path,
        "metadata": metadata_path,
        "sha256": sha_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export DBP-JSCC release assets")
    parser.add_argument(
        "--full_ckpt",
        type=Path,
        required=True,
        help="Full training checkpoint used for resume",
    )
    parser.add_argument(
        "--vocoder_ckpt",
        type=Path,
        required=True,
        help="Standalone vocoder checkpoint to ship in releases",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("release_assets"),
        help="Directory where release assets will be written",
    )
    args = parser.parse_args()

    assets = export_assets(args.full_ckpt, args.vocoder_ckpt, args.output_dir)
    for name, path in assets.items():
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"{name}: {path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
