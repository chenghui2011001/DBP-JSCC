# Dataset Preparation

## Overview

Public inference does not require `.f32` feature files if you use:

- `scripts/infer_bits_only.py --wav ...`
- `scripts/infer_wav.py --wav ...`

Those paths call a compatible external feature extractor internally to
extract the required 36-dim vocoder features.

Training is different. The current public training loader expects an aligned PCM file plus an aligned feature file on disk.

Public speech data can be obtained from:

- https://media.xiph.org/lpcnet/speech/tts_speech_negative_16k.sw

The corresponding `in_speech.pcm` input is expected to be:

- raw 16-bit PCM
- mono
- sampled at `16 kHz`

## Minimal Training Dataset Layout

The simplest supported layout is:

```text
data/
  out_speech.pcm
  lmr_export/
    features_36_vocoder_baseline.f32
```

Required files:

- `out_speech.pcm`
  A mono `int16` PCM stream at `16 kHz`
- `lmr_export/features_36_vocoder_baseline.f32`
  A float32 feature stream with shape `[T, 36]`

The loader aligns them with:

- `160` PCM samples per frame
- `1` feature frame per `10 ms`

So the expected relation is:

- `num_pcm_samples ~= num_feature_frames * 160`

## How To Build The Training Files

The public repository ships a dataset preparation script that wraps a
compatible external feature extractor. Supported extractor interfaces are:

```bash
dump_data -train input.s16 output_features.f32 output_pcm.pcm
fargan_demo -features input.pcm output_features.f32
```

For a single long recording, the practical command is:

```bash
python scripts/prepare_dataset.py \
  input.wav \
  --out_root ./data \
  --feature_bin /home/bluestar/fargan_demo/fargan_demo
```

What the script does:

1. converts WAV/FLAC inputs to 16 kHz mono signed 16-bit PCM
2. concatenates all inputs into one long PCM stream
3. runs the external feature extractor
4. trims `out_speech.pcm` to exact `160 samples / frame` alignment
5. writes a dataset manifest with frame counts and extractor metadata

## Building A Dataset From Many WAV Files

The current public loader is stream-based, not manifest-based. It expects one long aligned PCM stream and one long aligned feature stream.

Example:

```bash
python scripts/prepare_dataset.py \
  /path/to/wavs/*.wav \
  --out_root ./data \
  --feature_bin /home/bluestar/fargan_demo/fargan_demo
```

## Alternative Expert-Mixed Layout

The loader also supports an expert-mixed layout under `data_root` with files such as:

```text
harmonic_200k_36.f32
harmonic_200k.pcm
transient_200k_36.f32
transient_200k.pcm
burst_inpaint_200k_36.f32
burst_inpaint_200k.pcm
low_snr_200k_36.f32
low_snr_200k.pcm
```

If these files exist, training automatically switches to the mixed-data loader.

## Notes

- `scripts/infer_features.py` is optional and mainly useful for debugging or offline comparison when you already have aligned `.f32 + .pcm` dumps
- for ordinary public inference, prefer `scripts/infer_bits_only.py --wav ...`
- if you want a future public version to train directly from WAV manifests instead of `.f32 + .pcm`, that requires a loader refactor; the current codebase does not do that yet
- `utils/real_data_loader.py` now prefers `features_36_vocoder_baseline.f32`, and still accepts the old `features_36_fargan_baseline.f32` name for backward compatibility
