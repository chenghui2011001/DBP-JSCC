# Training Guide

## Files

- training entrypoint: `training/train.py` for DBP-JSCC
- multistage pipeline: `training/train_pipeline.py`
- training helpers: `training/train_support.py`
- spectral losses: `training/spectral_losses.py`
- single-run config snapshot: `configs/resume_run.json`
- five-stage pipeline config: `configs/multistage_training.json`
- inference from WAV: `scripts/infer_wav.py`
- inference from dumped features: `scripts/infer_features.py`

## Five-Stage Schedule

1. `content_pretrain`
   content-only warmup for `5000` steps
2. `f0_vuv_alignment`
   align F0/VUV and related branches while content and vocoder stay frozen for `10000` steps
3. `rvq_tuning`
   update RVQ codebooks only for `10000` steps
4. `l2h_branch_tuning`
   train the L2H-related branch while content and vocoder remain mostly frozen for `10000` steps
5. `full_vocoder_finetune`
   unfreeze the vocoder and enable final GAN losses for `20000` steps

## Commands

Run the full pipeline:

```bash
python training/train_pipeline.py --config configs/multistage_training.json
```

Dry run:

```bash
python training/train_pipeline.py --config configs/multistage_training.json --dry_run
```

Start from a later stage:

```bash
python training/train_pipeline.py --config configs/multistage_training.json --start_stage 3
```

Single-run training:

```bash
python training/train.py --help
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

## Paths To Edit Before Running

- `configs/multistage_training.json`
  - `data_root`
  - `vocoder_ckpt`
  - `fsk_ber_table`
  - `run_root`
- `configs/resume_run.json`
  - `data_root`
  - `resume`
  - `fsk_ber_table`
  - `out_dir`

## Local Assets Not Shipped In Git

- dataset directory
- pretrained vocoder weights
- BER lookup table
- external `dump_data` executable
- compiled selective-scan CUDA dependency
