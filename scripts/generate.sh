#!/usr/bin/env bash
# 单条推理 — 用法: ./scripts/generate.sh "请将以下白话文翻译为文言文：今天的咖啡很苦"
set -euo pipefail
cd "$(dirname "$0")/.."

PROMPT="${1:-请将以下白话文翻译为文言文：今天的咖啡味道有点苦，但是很提神。}"

mlx_lm.generate \
  --model ./models/Qwen2.5-1.5B-Instruct \
  --adapter-path adapters \
  --max-tokens 256 \
  --temp 0.3 \
  --prompt "$PROMPT"
