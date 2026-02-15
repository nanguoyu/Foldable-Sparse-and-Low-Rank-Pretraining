#!/bin/bash

set -euo pipefail

GPUS="3,4"

SEED=42
LR_RATE="5e-5"
TASKS=(mnli qqp)
RANKS=(4 8)
ALPHAS=(32 64 128)
ACT=False
MIX=True
NUM_EPOCHS="15"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for task in "${TASKS[@]}"; do
  for rank in "${RANKS[@]}"; do
    for alpha in "${ALPHAS[@]}"; do
      echo "[RUN] task=${task} rank=${rank} alpha=${alpha} act=${ACT} mix=${MIX} seed=${SEED} gpus=${GPUS}"
      "${SCRIPT_DIR}/glue.sh" \
        -g "${GPUS}" \
        -t "${task}" \
        -r "${rank}" \
        -l "${alpha}" \
        -a "${ACT}" \
        -m "${MIX}" \
        -lr "${LR_RATE}" \
        -e "${NUM_EPOCHS}" \
        -s "${SEED}"
    done
  done
done


