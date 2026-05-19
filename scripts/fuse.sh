#!/usr/bin/env bash
# 将 LoRA adapter 合并回基座，导出独立模型到 fused-model/
set -euo pipefail
cd "$(dirname "$0")/.."

mlx_lm.fuse \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter-path adapters \
  --save-path fused-model
