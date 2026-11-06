#!/usr/bin/env bash
# Copyright (c) 2026 Wuhan University, Binglin Hao. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Evaluate an existing segmentation checkpoint on one physical GPU.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 CONFIG CHECKPOINT [VISUALIZATION_DIR]" >&2
  echo "Set GPU_ID (default: 0) to select a physical GPU." >&2
  exit 2
fi

repository_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$repository_dir"

config_path=$1
checkpoint_path=$2
gpu_id=${GPU_ID:-0}

if [[ ! -f "$config_path" ]]; then
  echo "Config not found: $config_path" >&2
  exit 2
fi
if [[ ! -f "$checkpoint_path" ]]; then
  echo "Checkpoint not found: $checkpoint_path" >&2
  exit 2
fi

test_command=(python tools/test.py "$config_path" "$checkpoint_path" --eval mIoU)
if [[ $# -eq 3 ]]; then
  test_command+=(--show-dir "$3")
fi

CUDA_VISIBLE_DEVICES="$gpu_id" "${test_command[@]}"
