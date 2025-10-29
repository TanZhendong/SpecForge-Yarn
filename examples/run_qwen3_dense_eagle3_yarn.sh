SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

export WANDB_MODE=offline
# support tp8 train eagle3 for Qwen3-4B/8B/32B up to tp_size = 8
NUM_GPUS=${1:-4}

# Qwen3-4B Eagle3 Yarn Training
torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3_online_yarn.py \
    --target-model-path /home/lthpc/nvmessd/zhendong/models/Qwen3-4B\
    --draft-model-config $ROOT_DIR/configs/qwen3-4b-eagle3-yarn.json \
    --train-data-path $ROOT_DIR/cache/dataset/train_pg19_64k_6400.json \
    --output-dir $ROOT_DIR/outputs/Qwen3-4B-eagle3-32k-steps400-bsz16 \
    --num-epochs 1 \
    --learning-rate 2e-5 \
    --warmup-ratio 0.05 \
    --max-length 32768 \
    --log-steps 4 \
    --chat-template qwen \
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
    --wandb-project qwen3-eagle3-yarn \
    --wandb-name qwen3-4b-eagle3-32k-steps400-bsz16 \
    --wandb-key 9aa2ce0f18736ca9c640852efb65aa854fedaa1e

# Qwen3-8B Eagle3 Yarn Training
torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3_online_yarn.py \
    --target-model-path /home/lthpc/nvmessd/zhendong/models/Qwen3-8B\
    --draft-model-config $ROOT_DIR/configs/qwen3-8b-eagle3-yarn.json \
    --train-data-path $ROOT_DIR/cache/dataset/train_pg19_64k_6400.json \
    --output-dir $ROOT_DIR/outputs/Qwen3-8B-eagle3-32k-steps400-bsz16 \
    --num-epochs 1 \
    --learning-rate 2e-5 \
    --warmup-ratio 0.05 \
    --max-length 32768 \
    --log-steps 4 \
    --chat-template qwen \
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
    --wandb-project qwen3-eagle3-yarn \
    --wandb-name qwen3-8b-eagle3-32k-steps400-bsz16 \
    --wandb-key 9aa2ce0f18736ca9c640852efb65aa854fedaa1e
