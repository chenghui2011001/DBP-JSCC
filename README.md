# DBP-JSCC

DBP-JSCC is a speech joint source-channel coding system built around a dual-branch architecture:

- a content branch that transmits a 32-bin Bark/BFCC image with a 2D JSCC backbone
- an F0/voicing branch that preserves pitch and voicing cues
- a lightweight vocoder that reconstructs waveform from decoded acoustic features

This repository contains the public training code, inference scripts, multistage training pipeline, and release asset export tools.

## Environment

Reference environment:

```bash
mamba activate farGan-sota
python -V
pip install -r requirements.txt
```

Reference Python version:

- `3.10.18`

## Repository Layout

```text
training/
  train.py
  train_pipeline.py
  train_support.py
  spectral_losses.py

scripts/
  prepare_dataset.py
  build_feature_tools.sh
  infer_bits_only.py
  infer_wav.py
  infer_features.py
  jscc_single_sample_export_bits.py
  jscc_single_sample_decode_from_bits.py
  export_release_assets.py

models/
  dual_branch_bark_jscc.py
  vmamba_jscc2d.py
  vocoder_decoder.py
  vocoder_components.py
  bfcc_vocoder.py
  hash_bottleneck.py
  rvq_bottleneck.py
  hifi_discriminators.py
  feature_adapter.py
  lite_speech_jscc.py

configs/
  multistage_training.json
  resume_run.json

docs/
  dataset_preparation.md
  training_guide.md
  upload_manifest.txt
```

## Weights

Prebuilt weights are intended to be distributed through GitHub Releases:

- `dbp_jscc_model_only_step703800_epoch42.pth`
- `dbp_jscc_vocoder_pretrained.pth`
- `dbp_jscc_resume_full_step703800_epoch42.pth`

Recommended local placement:

```text
artifacts/
  jscc_ber_table.json

checkpoints/
  dbp_jscc_reference/
    dbp_jscc_model_only_step703800_epoch42.pth
    dbp_jscc_resume_full_step703800_epoch42.pth

vocoder_pt/
  dbp_jscc_vocoder_pretrained.pth
```

Usage:

- `dbp_jscc_model_only_step703800_epoch42.pth`: inference
- `dbp_jscc_vocoder_pretrained.pth`: vocoder initialization
- `dbp_jscc_resume_full_step703800_epoch42.pth`: exact training resume

## Training

Single-run training:

```bash
python training/train.py --help
```

Five-stage automatic training:

```bash
python training/train_pipeline.py --config configs/multistage_training.json
```

Dry run of the multistage pipeline:

```bash
python training/train_pipeline.py --config configs/multistage_training.json --dry_run
```

Resume from a later stage:

```bash
python training/train_pipeline.py --config configs/multistage_training.json --start_stage 3
```

The default public pipeline follows five stages:

1. content pretraining
2. F0/VUV alignment
3. RVQ training
4. L2H branch tuning
5. full vocoder finetuning

Detailed stage descriptions are in [docs/training_guide.md](/home/bluestar/FARGAN/opus/dnn/torch/Aether-lite/Aether-lite-bak/docs/training_guide.md).

## Inference

### Bits-only deployment path

This is the recommended public inference path. It explicitly runs:

1. feature extraction
2. bitstream export
3. `decode_from_bits_offline`

This path does not require precomputed `.f32` features if you provide `--wav`.
The script accepts either a `dump_data -train` extractor or a
`fargan_demo -features` style extractor through `--feature_bin`.

From WAV:

```bash
python scripts/infer_bits_only.py \
  --ckpt ./checkpoints/dbp_jscc_reference/dbp_jscc_model_only_step703800_epoch42.pth \
  --wav ./examples/test.wav \
  --feature_bin /home/bluestar/fargan_demo/fargan_demo \
  --out_dir ./outputs/infer_bits_only \
  --snr_db 0
```

From precomputed feature dumps:

```bash
python scripts/infer_bits_only.py \
  --ckpt ./checkpoints/dbp_jscc_reference/dbp_jscc_model_only_step703800_epoch42.pth \
  --features ./examples/features.f32 \
  --pcm ./examples/audio.pcm \
  --out_dir ./outputs/infer_bits_only \
  --snr_db 0
```

Optional:

- `--use_noisy_bits`: decode with noisy bits exported by the encoder path
- `--save_bits`: save the exported bitstream as `bitstream.pt`

### Forward-path inference

These scripts run the model forward path directly and are useful for diagnostics:

```bash
python scripts/infer_wav.py \
  --ckpt ./checkpoints/dbp_jscc_reference/dbp_jscc_model_only_step703800_epoch42.pth \
  --wav ./examples/test.wav \
  --feature_bin /home/bluestar/fargan_demo/fargan_demo \
  --out_dir ./outputs/infer_wav
```

```bash
python scripts/infer_features.py \
  --ckpt ./checkpoints/dbp_jscc_reference/dbp_jscc_model_only_step703800_epoch42.pth \
  --features ./examples/features.f32 \
  --pcm ./examples/audio.pcm \
  --out_dir ./outputs/infer_features
```

`infer_features.py` is optional. It is intended for users who already have aligned feature dumps and want a low-level diagnostic path.

### Split transmitter / receiver path

For communication-style experiments, the repository also keeps a
separated encoder-side bit export script and a receiver-side offline
decode script.

Export bits on the transmitter side:

```bash
python scripts/jscc_single_sample_export_bits.py \
  --ckpt ./checkpoints/dbp_jscc_reference/dbp_jscc_model_only_step703800_epoch42.pth \
  --pcm ./examples/test.wav \
  --feature_bin /home/bluestar/fargan_demo/fargan_demo \
  --out_bits ./outputs/tx/jscc_bits.npy
```

Decode on the receiver side:

```bash
python scripts/jscc_single_sample_decode_from_bits.py \
  --ckpt ./checkpoints/dbp_jscc_reference/dbp_jscc_model_only_step703800_epoch42.pth \
  --bits_rx ./outputs/tx/jscc_bits.npy \
  --pcm ./outputs/tx/input_aligned.pcm \
  --features ./outputs/tx/features.f32 \
  --out_pcm ./outputs/rx/output_jscc_fsk.pcm
```

## Release Assets

To export lightweight release files from a full training checkpoint:

```bash
python scripts/export_release_assets.py \
  --full_ckpt ./checkpoints/dbp_jscc_reference/checkpoint_step_703800_epoch_42.pth \
  --vocoder_ckpt ./vocoder_pt/vocoder_sq1Ab_adv_50.pth \
  --output_dir ./release_assets
```

This generates:

- a model-only checkpoint for inference
- a full resume checkpoint
- a standalone vocoder checkpoint copy
- `release_metadata.json`
- `SHA256SUMS.txt`

## External Assets

Prepare these locally before training or inference:

- dataset root such as `./data`
- BER lookup table such as `./artifacts/jscc_ber_table.json`
- vocoder checkpoint such as `./vocoder_pt/dbp_jscc_vocoder_pretrained.pth`
- external feature extractor binary when using WAV-based feature extraction
- or build the feature extractor from the modified Opus source tree

For other researchers, the most practical delivery is:

1. ship prebuilt `dump_data` / `fargan_demo` binaries in GitHub Releases for the common Linux environment
2. also ship the exact modified Opus source or a pinned submodule so users can rebuild the tools themselves
3. keep `scripts/prepare_dataset.py` and the TX/RX scripts as the stable public entrypoints

### Feature Tools Download

Prebuilt Linux x86_64 binaries are provided in GitHub Releases:

- Release page: https://github.com/chenghui2011001/DBP-JSCC/releases/tag/v0.1.0-feature-tools
- Asset: `dbp_jscc_feature_tools_linux_x86_64.tar.gz`

Download and extract:

```bash
mkdir -p ./tools
tar -xzf dbp_jscc_feature_tools_linux_x86_64.tar.gz -C ./tools
```

After extraction, the binaries are:

```text
./tools/dbp_jscc_feature_tools_linux_x86_64/fargan_demo
./tools/dbp_jscc_feature_tools_linux_x86_64/dump_data
```

Recommended choice:

- `fargan_demo`: ordinary inference and simple feature extraction
- `dump_data`: full `-train` style aligned feature/PCM generation

Examples with `--feature_bin`:

```bash
python scripts/prepare_dataset.py \
  input.wav \
  --out_root ./data \
  --feature_bin ./tools/dbp_jscc_feature_tools_linux_x86_64/fargan_demo
```

```bash
python scripts/infer_bits_only.py \
  --ckpt ./checkpoints/dbp_jscc_reference/dbp_jscc_model_only_step703800_epoch42.pth \
  --wav ./examples/test.wav \
  --feature_bin ./tools/dbp_jscc_feature_tools_linux_x86_64/fargan_demo \
  --out_dir ./outputs/infer_bits_only \
  --snr_db 0
```

```bash
python scripts/jscc_single_sample_export_bits.py \
  --ckpt ./checkpoints/dbp_jscc_reference/dbp_jscc_model_only_step703800_epoch42.pth \
  --pcm ./examples/test.wav \
  --feature_bin ./tools/dbp_jscc_feature_tools_linux_x86_64/fargan_demo \
  --out_bits ./outputs/tx/jscc_bits.npy
```

Example with `dump_data`:

```bash
python scripts/prepare_dataset.py \
  input.wav \
  --out_root ./data \
  --feature_bin ./tools/dbp_jscc_feature_tools_linux_x86_64/dump_data
```

Optional environment variable:

```bash
export VOCODER_FEATURE_BIN=./tools/dbp_jscc_feature_tools_linux_x86_64/fargan_demo
```

If you have the modified Opus source tree locally, you can build the
feature tools with:

```bash
bash scripts/build_feature_tools.sh /path/to/modified_opus
```

## Training Data Preparation

The current public training loader expects aligned files on disk, not raw WAV manifests.

Public speech data can be obtained from:

- https://media.xiph.org/lpcnet/speech/tts_speech_negative_16k.sw

The corresponding `in_speech.pcm` input is expected to be:

- raw 16-bit PCM
- mono
- sampled at `16 kHz`

Minimal dataset layout:

```text
data/
  out_speech.pcm
  lmr_export/
    features_36_vocoder_baseline.f32
```

Use the public preparation script to build that layout directly from one
or more audio files:

```bash
python scripts/prepare_dataset.py \
  input.wav \
  --out_root ./data \
  --feature_bin /home/bluestar/fargan_demo/fargan_demo
```

The repository now owns the orchestration step end-to-end: audio loading,
resampling, concatenation, frame alignment, dataset layout, and manifest
generation. The 36-dim vocoder feature definition itself still comes from
the external extractor binary, so you must provide a compatible build of
`fargan_demo` or `dump_data`.

For multi-file corpora, pass multiple inputs and the script will
resample, concatenate, extract the 36-dim feature stream, and trim
`out_speech.pcm` to exact frame alignment automatically:

```bash
python scripts/prepare_dataset.py \
  /path/to/wavs/*.wav \
  --out_root ./data \
  --feature_bin /home/bluestar/fargan_demo/fargan_demo
```

The script writes:

- `data/out_speech.pcm`
- `data/lmr_export/features_36_vocoder_baseline.f32`
- `data/dataset_manifest.json`

Full details are in [docs/dataset_preparation.md](/home/bluestar/FARGAN/opus/dnn/torch/Aether-lite/Aether-lite-bak/docs/dataset_preparation.md).

## Notes

- `mamba/` and `mamba_ssm/` may require local CUDA compilation depending on your environment
- `torch` and `torchaudio` must come from compatible builds
- generated outputs should stay outside Git history; keep them under `runs/`, `outputs/`, or `release_assets/`

For additional details, see:

- [docs/dataset_preparation.md](/home/bluestar/FARGAN/opus/dnn/torch/Aether-lite/Aether-lite-bak/docs/dataset_preparation.md)
- [docs/training_guide.md](/home/bluestar/FARGAN/opus/dnn/torch/Aether-lite/Aether-lite-bak/docs/training_guide.md)
- [docs/upload_manifest.txt](/home/bluestar/FARGAN/opus/dnn/torch/Aether-lite/Aether-lite-bak/docs/upload_manifest.txt)
