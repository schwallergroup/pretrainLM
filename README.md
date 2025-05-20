# pretrainLM

A one-stop repository containing comprehensive instructions, scripts, and resources for continued pretraining and supervised fine-tuning (SFT) of large language models (LLMs), specifically tailored for chemical domain models including **Qwen-2.5** and **Llama-3.1**.

## Overview

This repository provides:

* Complete setup guides for continued pretraining using Megatron and Nanotron.
* Conversion scripts for seamless transitions between Hugging Face and Megatron checkpoints.
* Preprocessing pipelines for chemical data integration.
* Resources for generating synthetic chemical data.

---

## Pretraining and SFT for Qwen-2.5

### Prerequisites

* Modified Megatron pipeline from Alibaba: [Pai-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch)

### Directory Structure

* **Pretraining data**: `/work/liac/pretrain_data`
* **SFT data**: `/work/liac/sft_data`

### Step-by-step Instructions

#### Step 1: Convert HF Checkpoints to Megatron Format

Use `convert.slurm`:

Modify the following variables at the top of the script:

* Environment modules (`dev_env`, `torch_env`)
* Toolkit path
* `PYTHONPATH` to `Pai-Megatron-Patch` and `PAI-Megatron-LM-240718`
* Tensor Parallelism (`TP`) and Pipeline Parallelism (`PP`). Use:

  * TP = 2 for models ≤ 3B
  * PP = 1 (default unless pipeline parallelism is required)
* Node and GPU count (`nnodes`, `GPUS_PER_NODE`)
* Source path (HF checkpoints)
* Target path (Megatron checkpoints)

Launch script:

```bash
sbatch convert.slurm
```

#### Step 2: Prepare Data & Run Pretraining

Use `process_data.slurm`:

* Modify paths, sequence length, padding, batch size as needed.

Launch preprocessing:

```bash
sbatch process_data.slurm
```

Pretraining with `pretrain.slurm`:

* Adjust `number_of_steps`, `seq_len`, and GPU configurations.

Calculate total steps:

```bash
# Example:
# global_batch_size * batch_accumulation_steps * dp = 8 * 4 * 1 = 32
# full_batch_size = seq_len * global_batch_size * accumulation_steps * dp
# steps_per_epoch = total_tokens / full_batch_size
```

Launch pretraining:

```bash
sbatch pretrain.slurm
```

#### Step 3: Convert Back & Run SFT

Convert Megatron checkpoints back to Hugging Face format using `convert_back.slurm`. Note that for SFT, convert the HF checkpoint again back to Megatron format using `prepare_sft.slurm` and then launch SFT with `run_sft.sbatch`.

Diagnostic logs will appear in the output files (currently no W\&B integration).

---

## Pretraining Llama-3.1

Minimal modifications required for Megatron-based pretraining. Refer to:

* Llama checkpoint conversion: `qwen_pretrain/kuma/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/llama`

Nanotron pretraining setup (CSCS specific) available here:

* [Nanotron Llama CP Setup](https://docs.google.com/document/d/1YBC5pFcrwh2Mo98vL1xQXSgUCTGyzX3bEKBpciRm-Lw/edit?usp=sharing)

---

## Preprocessing Data with SMILES

Scripts to integrate chemical structures (SMILES) into textual datasets:

1. **Extract chemical entities**:

   * Run `chem_extract.py` (requires `chemdataextractor2`)
   * Outputs pickle with entities' positions

2. **Interleave SMILES into text**:

   * Run `fineweb_smiles.py` (requires `empty4.pkl` and `string2smiles3.pkl` for entity handling)

---

## Synthetic Data Generation

Use [SMILESbench](https://github.com/schwallergroup/SMILESbench/tree/main) for generating synthetic chemical data (used previously for benchmarking and exam purposes).

---

### Contributing

Feel free to open issues, submit PRs, or contact maintainers for any questions or suggestions.

