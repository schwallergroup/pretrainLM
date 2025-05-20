# pretrainLM
One stop instructions for continued pretraining for LMs

In this repo, there will instructions on continuing the pretraining for Qwen-2.5 and Llama-3 models

Pretraining and SFT for Qwen-2.5
================================
For the CP of Qwen-2.5 models we will be using the modified Megatron pipeline from Alibaba - https://github.com/alibaba/Pai-Megatron-Patch.
There were multiple versions of the pipeline, but in this repo I will be adding the further modified repo to make it work on CSCS and Kuma (found in their respective folders.)

Step 1: Convert HF to megatron format:
--------------------------------------
Megatron uses its own custom framework of weights for pretraining. To convert them, use the `convert.slurm` script for the appropriate cluster. You will have to change the following before running the script: (Might feel like a lot to change, but if you open the slurm script it will be the first few lines after the srun bash command)
1. Load your own environment (I had modules in dev_env and torch_env)
2. Change the path to your toolkit folder correctly
3. PYTHONPATH (change to the appropriate Pai-Megatron-Patch and PAI-Megatron-LM-240718 folder paths)
4. Change TP, PP appropriately. (PP is always 1 unless you want pipeline parallelism. And TP should be 2 for models lesser than equal to 3b, and can be bigger for other models). Make sure to change nnodes and GPUS_PER_NODE appropriately after changing TP.
5. Change source and target path. source is wehere you have stored your HF models and target will be the path where your megatron checkpoints will be saved.

Submit the slurm scrip using sbatch to launch the script.


Step 2: prepare data for pretraining and pretrain:
--------------------------------------------------
Similarly we have to convert the pretraining data (right now in jsonl format) to the megatron format. Use the script `process_data.slurm` for this. Change the same set of parameters as above and run the script. In addition to this, you can change the hyperparameters relating to sequence_len, pad_len, batch_size, etc.

Submit the slurm script using sbatch to launch the script.

Use pretrain.slurm file for launching your pretraining jobs using the processed qwen weights and data. Change the number of steps, seq_len appropriately.

Make sure total GPUs (world size) = TP*DP*PP

And for calculating the steps. example:
- number of sequences: global batch size * batch accumulation steps * dp = 8 * 4 * 1 = 32
- full bsz: seqlen * global batch size * batch accumulation steps * dp = 8192 * 32 = 262144
- one epoch in steps: 1000688526(total tokens) / 262144 = 3817 (number of steps in 1 epoch)


Step 3: convert back and SFT
----------------------------
Use the convert_back.slurm script to convert megatron to HF back again. 
You have a pretrained model now that you can use for benchmarks and for further SFT.
Note: You will have to convert the pretrained model to HF, and then to megatron for SFT, and you cannot use the pretrained megatron weights directly.
Use the prepare_sft.slurm and run_sft.sbatch script to process the data and to run the SFT using megatron. Make sure to change the appropriate parameters for your run.

Right now there is no support for wandb in these runs, but all your model diagnostics will show up in the out file.

