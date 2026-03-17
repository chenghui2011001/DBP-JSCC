#!/usr/bin/env bash
set -euo pipefail

OPUS_ROOT="${1:-./third_party/opus}"

if [[ ! -d "${OPUS_ROOT}" ]]; then
  echo "Opus source tree not found: ${OPUS_ROOT}" >&2
  echo "Pass the modified Opus root as the first argument." >&2
  exit 1
fi

cd "${OPUS_ROOT}"

if [[ ! -x ./configure ]]; then
  if [[ -x ./autogen.sh ]]; then
    ./autogen.sh
  else
    autoreconf -fi
  fi
fi

./configure --enable-deep-plc
make -j"$(nproc)" dump_data fargan_demo

echo "Built feature tools under: ${OPUS_ROOT}"
echo "  ${OPUS_ROOT}/dump_data"
echo "  ${OPUS_ROOT}/fargan_demo"
