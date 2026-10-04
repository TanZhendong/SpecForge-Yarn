#!/usr/bin/env bash
# Create an isolated training env for SpecForge-Yarn (Apertus EAGLE3 yarn run).
#
# Deliberately DIVERGES from requirements.txt in two places:
#   1. transformers >=4.56 is required for ApertusForCausalLM (4.55.2 in
#      requirements.txt predates it); 4.57.6 is what the SpecPV venv uses.
#   2. sglang / qwen-vl-utils / openai-harmony are NOT installed: the
#      train_eagle3_online_yarn.py path is pure HF + torch and does not need
#      them (they are for data regeneration and the sglang-online trainer).
#
# Usage: bash setup_env.sh          (creates .venv inside this repo)

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

uv venv --python 3.11 .venv
source .venv/bin/activate

# torch 2.8.0 per requirements.txt (flex_attention included); cu128 works on
# driver >= 525 (this machine: 595.71.05)
uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# training stack with the Apertus-capable transformers override
uv pip install \
    "transformers==4.57.6" \
    accelerate \
    datasets \
    tqdm \
    wandb \
    psutil \
    numpy \
    pydantic \
    setuptools

uv pip install -e . --no-deps

echo "SpecForge-Yarn env ready: $ROOT_DIR/.venv"
echo "Sanity check before training:"
echo "  .venv/bin/python -c \"import torch; from transformers import ApertusForCausalLM; print(torch.__version__)\""
