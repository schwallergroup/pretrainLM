#!/bin/bash
#SBATCH --job-name=qwen-convert-back
#SBATCH --container-writable
#SBATCH --time=01:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=2
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
    --nproc_per_node 2 \
    --nnodes 16 \
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
cd /home/senthilk/Pai-Megatron-Patch/
export PYTHONPATH=\"/home/senthilk/Pai-Megatron-Patch/:/home/senthilk/Pai-Megatron-Patch/PAI-Megatron-LM-240718:\$PYTHONPATH\"

set -e

START_TIME=\$SECONDS

CURRENT_DIR=\"\$( cd \"\$( dirname \"\$0\" )\" && pwd )\"
MEGATRON_PATH=\$( dirname \$( dirname \${CURRENT_DIR}))

cd /home/senthilk/Pai-Megatron-Patch/toolkits/sft_data_preprocessing
python sample_stats.py /work/liac/combined.jsonl

input_data_path=/work/liac/combined.jsonl
tokenizer=Qwen2Tokenizer
seq_len=8192
output_data_path=/work/liac/sft1/
load_dir=/work/liac/shai/saves/checkpoint/-pr-bf16-tp-2-pp-1-cp-1-ac-false-do-true-sp-true-ti--wi-
default_packing=\$6

if [ -z \${default_packing} ]; then
  default_packing=false
fi

if [ \$default_packing = true ]; then
  packing_option=\"\
    --sequence-packing 
  \"
else
  packing_option=\"\"
fi

packing_option=\"\
    --sequence-packing
\"

cmd=\"python build_idxmap_sft_dataset.py \
  --input \${input_data_path} \
  --output-prefix \${output_data_path} \
  --patch-tokenizer-type \${tokenizer} \
  --load \${load_dir} \
  --seq-length \${seq_len} \
  --workers 12 \
  --partitions 1 \${packing_option}\"

echo \$cmd
eval \$cmd

ELAPSED_TIME=\$((\$SECONDS - \$START_TIME))
echo \"\$((\$ELAPSED_TIME/60)) min \$((\$ELAPSED_TIME%60)) sec\"
"
