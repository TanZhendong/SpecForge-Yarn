SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

export WANDB_MODE=offline

# train eagle3 for llama3.1-8b
NUM_GPUS=${1:-4}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3_online_yarn.py \
    --target-model-path /home/lthpc/nvmessd/zhendong/models/LLAMA3.1-8B-Instruct \
    --draft-model-config $ROOT_DIR/configs/llama3-8B-eagle3-yarn.json \
    --train-data-path $ROOT_DIR/cache/dataset/train_pg19_64k_6400.json \
    --output-dir $ROOT_DIR/outputs/llama3-8b-eagle3-32k-steps400-bsz16 \
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
    --dp-size 4 \
    --draft-micro-batch-size 1 \
    --draft-global-batch-size 16 \
    --draft-accumulation-steps 4 \
    --cache-dir $ROOT_DIR/cache \
    --attention-backend flex_attention \
    --resume \
    --report-to wandb \
    --wandb-project llama3-eagle3-yarn \
    --wandb-name llama3-8b-eagle3-32k-steps400-bsz16 \
    --wandb-key 9aa2ce0f18736ca9c640852efb65aa854fedaa1e

