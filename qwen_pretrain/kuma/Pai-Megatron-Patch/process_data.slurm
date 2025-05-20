#!/bin/bash
#SBATCH --job-name=qwen-pretrain-convert
#SBATCH --container-writable
#SBATCH --time=00:30:00
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

#python delta.py
#python pft.py
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
export PYTHONPATH=\"/home/senthilk/Pai-Megatron-Patch/:/home/senthilk/Pai-Megatron-Patch/PAI-Megatron-LM-240718:\$PYTHONPATH\"
cd /home/senthilk/Pai-Megatron-Patch
START_TIME=\$SECONDS

CURRENT_DIR=\"\$( cd \"\$( dirname \"\$0\" )\" && pwd )\"
MEGATRON_PATH=\$( dirname \$( dirname \${CURRENT_DIR}))
export PYTHONPATH=\"/home/senthilk/Pai-Megatron-Patch/:/home/senthilk/Pai-Megatron-Patch/Megatron-LM-240405:\$PYTHONPATH\"

input_data_dir=/home/senthilk/shuffled_pretrain.jsonl
tokenizer=Qwen2Tokenizer
json_keys=text
output_data_dir=/home/senthilk/chemrxiv_megatron
load_dir=/work/liac/shai/qwen

INPUT=\"\${input_data_dir}\"
python -m pip uninstall -y setuptools
python -m pip install python-dotenv 
python -m pip install setuptools==67.2.0
python -m pip install distribute
python -m pip install tabular
if [ \$tokenizer = \"Qwen2Tokenizer\" ]; then
  python /home/senthilk/Pai-Megatron-Patch/toolkits/pretrain_data_preprocessing/preprocess_data_megatron.py \
  --input \${INPUT} \
  --output-prefix \${output_data_dir}/mmap_qwen2_datasets \
  --patch-tokenizer-type Qwen2Tokenizer \
  --json-keys \${json_keys} \
  --load \${load_dir} \
  --workers 8 \
  --partitions 1 \
  --keep-sequential-samples \
  --append-eod

elif [ \$tokenizer = \"DeepSeekV2Tokenizer\" ]; then
  python preprocess_data_megatron.py \
  --input \${INPUT} \
  --output-prefix \${output_data_dir}/mmap_deepseekv2_datasets \
  --patch-tokenizer-type DeepSeekV2Tokenizer \
  --json-keys \${json_keys} \
  --load \${load_dir} \
  --workers 8 \
  --partitions 1 \
  --keep-sequential-samples \
  --append-eod

elif [ \$tokenizer = \"LLamaTokenizer\" ]; then
  python preprocess_data_megatron.py \
  --input \${INPUT} \
  --output-prefix \${output_data_dir}/mmap_llama_datasets \
  --patch-tokenizer-type LLamaTokenizer \
  --json-keys \${json_keys} \
  --load \${load_dir} \
  --workers 16 \
  --partitions 1 \
  --keep-sequential-samples \
  --append-eod

elif [ \$tokenizer = \"LLama2Tokenizer\" ]; then
  python preprocess_data_megatron.py \
  --input \${INPUT} \
  --output-prefix \${output_data_dir}/mmap_llama2_datasets \
  --patch-tokenizer-type LLama2Tokenizer \
  --json-keys \${json_keys} \
  --load \${load_dir} \
  --workers 16 \
  --partitions 1 \
  --keep-sequential-samples \
  --append-eod

elif [ \$tokenizer = \"LLama3Tokenizer\" ]; then
  python preprocess_data_megatron.py \
  --input \${INPUT} \
  --output-prefix \${output_data_dir}/mmap_llama3_datasets \
  --patch-tokenizer-type LLama3Tokenizer \
  --load \${load_dir} \
  --workers 16 \
  --partitions 1 \
  --keep-sequential-samples \
  --append-eod

fi

ELAPSED_TIME=\$((\$SECONDS - \$START_TIME))
echo \"\$((\$ELAPSED_TIME/60)) min \$((\$ELAPSED_TIME%60)) sec\"
"
" 2>&1
