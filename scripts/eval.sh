#!/usr/bin/env bash
# 在 test.jsonl 上跑测试集 loss
set -euo pipefail
cd "$(dirname "$0")/.."

mlx_lm.lora \
  --model ./models/Qwen2.5-1.5B-Instruct \
  --adapter-path adapters \
  --data data \
  --test
