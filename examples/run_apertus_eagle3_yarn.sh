#!/usr/bin/env bash
# Fine-tune the EAGLE3-Apertus draft with YaRN rope on PG-19 long context.
#
# Recipe follows examples/run_llama3_eagle3_yarn.sh:
#   - resume from the author checkpoint placed in outputs/apertus-eagle3-yarn/epoch_9
#     (config.json swapped to the yarn version, training_state.pt removed so the
#     optimizer/epoch counter start fresh)
#   - PG-19 preformatted text (is_preformatted -> loss_mask = ones), max-length 32768
#   - aux hidden-state layers pinned via eagle_config in the draft config so the
#     training extraction ({2,16,29}) matches SpecPV inference
#
# Usage: bash examples/run_apertus_eagle3_yarn.sh [NUM_GPUS=2]

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

NUM_GPUS=${1:-2}

# train eagle3 yarn for Apertus-8B-Instruct-2509 (target ~16GB bf16 per rank,
# draft 513M; 80GB A100s fit target + 32k activations + logits comfortably)
torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3_online_yarn.py \
    --target-model-path /home/tanzhendong/models/Apertus-8B-Instruct-2509 \
    --draft-model-config $ROOT_DIR/configs/apertus-8B-eagle3-yarn.json \
    --train-data-path $ROOT_DIR/cache/dataset/train_pg19_64k_6400.json \
    --output-dir $ROOT_DIR/outputs/apertus-eagle3-yarn \
    --num-epochs 1 \
    --learning-rate 2e-5 \
    --warmup-ratio 0.05 \
    --max-length 32768 \
    --log-steps 4 \
    --chat-template llama3 \
    --is-preformatted \
    --ttt-length 4 \
    --save-interval 1 \
    --tp-size 1 \
    --dp-size $NUM_GPUS \
    --draft-micro-batch-size 1 \
    --draft-global-batch-size 16 \
    --cache-dir $ROOT_DIR/cache \
    --attention-backend flex_attention \
    --resume \
    --report-to none
