# Feature Extraction Tools

## Recommendation

For public release, do not rely only on a machine-local binary path such
as `/home/bluestar/fargan_demo/fargan_demo` or `/home/bluestar/FARGAN/opus/dump_data`.

The practical public setup is:

1. release prebuilt `dump_data` and `fargan_demo` binaries for the common Linux environment
2. publish the exact modified Opus source tree that contains `dnn/dump_data.c` and `dnn/fargan_demo.c`
3. keep the Python entrypoints in this repository stable:
   - `scripts/prepare_dataset.py`
   - `scripts/infer_wav.py`
   - `scripts/infer_bits_only.py`
   - `scripts/jscc_single_sample_export_bits.py`
   - `scripts/jscc_single_sample_decode_from_bits.py`

## Download Release Binaries

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

## Why Not Depend On Upstream Xiph/Opus Alone

This repository expects feature extractors that are compatible with the
modified Opus tree already used in the project.

Local evidence from the current source tree:

- `Makefile.am` builds both `dump_data` and `fargan_demo`
- `dnn/README.md` documents `./dump_data -train ...`
- `dnn/torch/fargan/README.md` documents `./fargan_demo -features ...`

Depending directly on upstream `https://gitlab.xiph.org/xiph/opus.git`
is risky unless you also pin the exact fork/commit that contains the
same feature pipeline as your experiments.

## Build From Source

If you already have the modified Opus tree, build the tools with:

```bash
bash scripts/build_feature_tools.sh /path/to/modified_opus
```

This runs:

```bash
./configure --enable-deep-plc
make -j"$(nproc)" dump_data fargan_demo
```

If `configure` is not present, the helper first runs `./autogen.sh`
or `autoreconf -fi`.

## Public User Workflow

1. clone `DBP-JSCC`
2. obtain the exact modified Opus source or the prebuilt release binaries
3. build `dump_data` and `fargan_demo`, or download them from Releases
4. run `scripts/prepare_dataset.py` for training data
5. run `scripts/jscc_single_sample_export_bits.py` and `scripts/jscc_single_sample_decode_from_bits.py` for separated TX/RX experiments
