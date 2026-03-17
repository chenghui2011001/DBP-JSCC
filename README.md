# DBP-JSCC

Public training and inference code for the DBP-JSCC speech JSCC system.

## Environment

Reference environment:

```bash
mamba activate farGan-sota
python -V
```

Reference Python version:

- `3.10.18`

Install Python packages with:

```bash
pip install -r requirements.txt
```

## Minimal Public Layout

```text
training/
  train.py
  train_pipeline.py
  train_support.py
  spectral_losses.py

scripts/
  infer_wav.py
  infer_features.py

configs/
  multistage_training.json
  resume_run.json

docs/
  training_guide.md
  upload_manifest.txt
```

## Training

Single-run training:

```bash
python training/train.py --help
```

Five-stage automatic training:

```bash
python training/train_pipeline.py --config configs/multistage_training.json
```

Dry run:

```bash
python training/train_pipeline.py --config configs/multistage_training.json --dry_run
```

Resume from a later stage:

```bash
python training/train_pipeline.py --config configs/multistage_training.json --start_stage 3
```

## Inference

From WAV:

```bash
python scripts/infer_wav.py \
  --ckpt ./outputs/checkpoints/checkpoint_step_010000_epoch_00.pth \
  --wav ./examples/test.wav \
  --out_dir ./outputs/infer_wav
```

From pre-dumped features:

```bash
python scripts/infer_features.py \
  --ckpt ./outputs/checkpoints/checkpoint_step_010000_epoch_00.pth \
  --features ./examples/features.f32 \
  --pcm ./examples/audio.pcm \
  --out_dir ./outputs/infer_features
```

## External Assets

Prepare these locally:

- dataset directory such as `./data`
- vocoder checkpoint such as `./vocoder_pt/vocoder_sq1Ab_adv_50.pth`
- BER table such as `./artifacts/jscc_ber_table.json`
- external `dump_data` binary for WAV feature extraction

## Weights From Releases

Prebuilt weights are published on GitHub Releases:

- `https://github.com/chenghui2011001/DBP-JSCC/releases`

Download and place the files as follows:

- `dbp_jscc_model_only_step703800_epoch42.pth`
  Place at `./checkpoints/dbp_jscc_reference/dbp_jscc_model_only_step703800_epoch42.pth`
- `dbp_jscc_vocoder_pretrained.pth`
  Place at `./vocoder_pt/dbp_jscc_vocoder_pretrained.pth`
- `dbp_jscc_resume_full_step703800_epoch42.pth`
  Optional. Place at `./checkpoints/dbp_jscc_reference/dbp_jscc_resume_full_step703800_epoch42.pth` when you need full training resume state

Recommended local layout:

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

Usage notes:

- For inference, use `dbp_jscc_model_only_step703800_epoch42.pth` as `--ckpt`
- For standalone vocoder initialization, point `vocoder_ckpt` to `./vocoder_pt/dbp_jscc_vocoder_pretrained.pth`
- For exact training resume, use `dbp_jscc_resume_full_step703800_epoch42.pth`

## Notes

- `mamba/` and `mamba_ssm/` are used by the model code and may require local CUDA compilation
- `torch` and `torchaudio` must come from matching builds
- training outputs are written under `./runs` or `./outputs` by default

Detailed instructions are in `docs/training_guide.md`.
