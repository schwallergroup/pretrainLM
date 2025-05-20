#!/bin/bash
#SBATCH --job-name=qwen-convert
#SBATCH --container-writable
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=4
#SBATCH --partition=h100
#SBATCH --no-requeue
#SBATCH --account=liac

export NCCL_NSOCKS_PERTHREAD=4
module restore dev_env
source /home/senthilk/torch_env/bin/activate
export CUDA_DEVICE_MAX_CONNECTIONS=1 # Important for Nanotron
export OMP_NUM_THREADS=16

export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_API_KEY=d25f82312adf534313b41219af6dfb51175458da
export WANDB_DIR=/home/senthilk/wandb_dir
export MASTER_ADDR=$(hostname)
export HF_TOKEN=None
export MASTER_ADDR=$(hostname)
export MASTER_PORT=25678
export TOKENIZERS_PARALLELISM=true
export NCCL_SOCKET_IFNAME=bond0
export NCCL_DEBUG=INFO

export TORCHRUN_ARGS="--nproc-per-node=4 \
--start-method=forkserver \
--monitor-interval=10000 "

python delta.py
python pft.py
#huggingface-cli whoami
#python finetune.py
#python fine1.py
#python -u -m torch.distributed.run --nnodes=1 --nproc_per_node=4 finefull.py
#python -u -m torch.distributed.run --nproc_per_node=4 mcq_fine_sft.py
#python delta.py

LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node 4 \
    --nnodes 8 \
    --node_rank \$SLURM_PROCID \
    --rdzv_id $SLURM_JOB_ID \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --role \$(hostname -s|tr -dc '0-9'): \
    --tee 3 \
    "
#PROGRAM="finetune.py"

export CMD="${LAUNCHER} ${PROGRAM}"

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --jobid $SLURM_JOB_ID \
    "

srun $SRUN_ARGS bash -c "
cd  /home/senthilk/dev_env
module restore dev_env
source /home/senthilk/torch_env/bin/activate
cd /home/senthilk/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/qwen/
export PYTHONPATH=\"/home/senthilk/Pai-Megatron-Patch/:/home/senthilk/Pai-Megatron-Patch/PAI-Megatron-LM-240718:\$PYTHONPATH\"
set -e
export CUDA_VISIBLE_DEVICES=0,1,2,3
MASTER_ADDR=localhost
MASTER_PORT=\$(shuf -n 1 -i 10000-65535)
START_TIME=\$SECONDS

MODEL_SIZE=3B
SOURCE_CKPT_PATH=/work/liac/shai/qwen/ 
TARGET_CKPT_PATH=/work/liac/shai/qwen_base/
TP=2
PP=1
PR=bf16
USE_TE=true
MG2HF=false
HF_CKPT_PATH=\${9}

CURRENT_DIR=\"\$( cd \"\$( dirname \"\$0\" )\" && pwd )\"
MEGATRON_PATH=\$( dirname \$(dirname \$( dirname \${CURRENT_DIR})))


if [ \$MODEL_SIZE = 0.5B ]; then

NUM_LAYERS=24
HIDDEN_SIZE=896
NUM_ATTN_HEADS=14
INTERMEDIATE_SIZE=4864
NUM_KEY_VALUE_HEADS=2
MAX_POSITION_EMBEDDINGS=32768
EXTRA_VOCAB_SIZE=293
RMS_NORM_EPS=1e-6
gqa_options=\" \
                    --group-query-attention \
                    --num-query-groups \${NUM_KEY_VALUE_HEADS}\"


tie_option=\"\"
cpu_options=\"\"

elif [ \$MODEL_SIZE = 1.5B ]; then

NUM_LAYERS=28
HIDDEN_SIZE=1536
NUM_ATTN_HEADS=12
INTERMEDIATE_SIZE=8960
NUM_KEY_VALUE_HEADS=2
MAX_POSITION_EMBEDDINGS=32768
EXTRA_VOCAB_SIZE=293
RMS_NORM_EPS=1e-6
gqa_options=\" \
                    --group-query-attention \
                    --num-query-groups \${NUM_KEY_VALUE_HEADS}\"

tie_option=\"\"
cpu_options=\"\"

elif [ \$MODEL_SIZE = 3B ]; then

NUM_LAYERS=36
HIDDEN_SIZE=2048
NUM_ATTN_HEADS=16
INTERMEDIATE_SIZE=11008
NUM_KEY_VALUE_HEADS=2
MAX_POSITION_EMBEDDINGS=32768
EXTRA_VOCAB_SIZE=293
RMS_NORM_EPS=1e-6
gqa_options=\" \
                    --group-query-attention \
                    --num-query-groups \${NUM_KEY_VALUE_HEADS}\"

tie_option=\"\"
cpu_options=\"\"

elif [ \$MODEL_SIZE = 7B ]; then

NUM_LAYERS=28
HIDDEN_SIZE=3584
NUM_ATTN_HEADS=28
INTERMEDIATE_SIZE=18944
NUM_KEY_VALUE_HEADS=4
MAX_POSITION_EMBEDDINGS=131072
EXTRA_VOCAB_SIZE=421
RMS_NORM_EPS=1e-6
gqa_options=\" \
                    --group-query-attention \
                    --num-query-groups \${NUM_KEY_VALUE_HEADS}\"

tie_option=\" \
        --untie-embeddings-and-output-weights \
        \"

cpu_options=\"\"

elif [ \$MODEL_SIZE = 14B ]; then
NUM_LAYERS=48
HIDDEN_SIZE=5120
NUM_ATTN_HEADS=40
INTERMEDIATE_SIZE=13824
NUM_KEY_VALUE_HEADS=8
MAX_POSITION_EMBEDDINGS=131072
EXTRA_VOCAB_SIZE=421
RMS_NORM_EPS=1e-5
gqa_options=\" \
                    --group-query-attention \
                    --num-query-groups \${NUM_KEY_VALUE_HEADS}\"

tie_option=\" \
        --untie-embeddings-and-output-weights \
        \"
cpu_options=\"\"

elif [ \$MODEL_SIZE = 32B ]; then

NUM_LAYERS=64
HIDDEN_SIZE=5120
NUM_ATTN_HEADS=40
INTERMEDIATE_SIZE=27648
NUM_KEY_VALUE_HEADS=8
MAX_POSITION_EMBEDDINGS=131072
EXTRA_VOCAB_SIZE=421
RMS_NORM_EPS=1e-5
gqa_options=\" \
                    --group-query-attention \
                    --num-query-groups \${NUM_KEY_VALUE_HEADS}\"

tie_option=\" \
        --untie-embeddings-and-output-weights \
        \"

cpu_options=\"\"

elif [ \$MODEL_SIZE = 72B ]; then

NUM_LAYERS=80
HIDDEN_SIZE=8192
NUM_ATTN_HEADS=64
INTERMEDIATE_SIZE=29568
NUM_KEY_VALUE_HEADS=8
MAX_POSITION_EMBEDDINGS=131072
EXTRA_VOCAB_SIZE=421
RMS_NORM_EPS=1e-5
gqa_options=\" \
                    --group-query-attention \
                    --num-query-groups \${NUM_KEY_VALUE_HEADS}\"

tie_option=\" \
        --untie-embeddings-and-output-weights \
        \"

cpu_options=\" \
            --use-cpu-initialization\"

fi

if [ \$MG2HF = true ]; then
    convert_options=\" \
                --convert-checkpoint-from-megatron-to-transformers \
                --hf-ckpt-path \${HF_CKPT_PATH}\"

elif [ \$MG2HF = false ]; then
    convert_options=""
fi

if [ \$USE_TE = true ]; then
    te_options=\" \
                --transformer-impl transformer_engine \
                \"

elif [ \$USE_TE = false ]; then
    te_options=\" \
                --transformer-impl local \
                \"
fi

if [ \$PR = fp16 ]; then
    pr_options=\" \
                    --fp16\"

elif [ \$PR = bf16 ]; then
    pr_options=\" \
        --bf16\"

fi

if [ \$PP -gt 1 ]; then
    tie_option=\" \
        --untie-embeddings-and-output-weights \
        \"
fi

DISTRIBUTED_ARGS=\"--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr \$MASTER_ADDR --master_port \$MASTER_PORT\"

unset TORCH_CUDA_ARCH_LIST

python -m torch.distributed.launch \${DISTRIBUTED_ARGS} hf2mcore_qwen2_dense_and_moe_gqa.py \
    --load \${SOURCE_CKPT_PATH} \
    --save \${TARGET_CKPT_PATH} \
    --target-tensor-model-parallel-size \${TP} \
    --target-pipeline-model-parallel-size \${PP} \
    --micro-batch-size 1 \
    --save-interval 1 \
    --swiglu \
    --num-layers \${NUM_LAYERS} \
    --hidden-size \${HIDDEN_SIZE} \
    --ffn-hidden-size \${INTERMEDIATE_SIZE} \
    --num-attention-heads \${NUM_ATTN_HEADS} \
    --max-position-embeddings \${MAX_POSITION_EMBEDDINGS} \
    --seq-length 1 \
    --no-async-tensor-model-parallel-allreduce \
    --patch-tokenizer-type Qwen2Tokenizer \
    --extra-vocab-size \${EXTRA_VOCAB_SIZE} \
    --no-bias-swiglu-fusion \
    --no-rope-fusion \
    --use-rotary-position-embeddings \
    --disable-bias-linear \
    --add-qkv-bias \
    --normalization RMSNorm \
    --norm-epsilon \${RMS_NORM_EPS} \
    --use-mcore-models \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --rotary-base 1000000 \
    --save-safetensors \
    \${te_options} \
    \${convert_options} \
    \${pr_options} \
    \${cpu_options} \
    \${tie_option} \
    \${gqa_options}


ELAPSED_TIME=\$((\$SECONDS - \$START_TIME))
echo \"\$((\$ELAPSED_TIME/60)) min \$((\$ELAPSED_TIME%60)) sec\"
" 2>&1
