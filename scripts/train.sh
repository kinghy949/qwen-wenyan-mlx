#!/usr/bin/env bash
# LoRA 训练 — 读取 lora_config.yaml
set -euo pipefail
cd "$(dirname "$0")/.."

# 确保已激活 venv
if [ -z "${VIRTUAL_ENV:-}" ]; then
  echo "[!] 请先: source mlx-env/bin/activate"
  exit 1
fi

mlx_lm.lora --config lora_config.yaml
