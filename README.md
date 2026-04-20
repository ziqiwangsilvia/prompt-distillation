# Codebase for Prompt Distillation

## Overview
This repository contains the code necessary to reproduce the main results of our Prompt Distillation submission. It supports same-tokenizer and cross-tokenizer distillation (e.g., Qwen → Llama) with character-span aligned KL divergence.

## Repository Structure

```
├── models/            # LLM class, model configs, messages, tool call format
├── data/              # Dataset classes, dataloaders, file naming utilities
├── curriculum/        # Lesson/exercise data structures and curriculum generation scripts
├── training/          # Training loop, loss functions, projection, alignment
│   ├── train.py       # Entry point for training
│   ├── trainer.py     # Trainer class with multi-GPU teacher support
│   ├── loss.py        # Token loss, logit loss (flat + aligned KL), distributed teacher forward
│   ├── projection.py  # VocabProjection + character-span alignment
│   ├── params.py      # All training hyperparameters
│   └── utils.py       # Tokenization, generation, data loading helpers
├── evaluation/        # Evaluation and grading scripts
├── config/            # Pipeline YAML config, accelerate configs, and DeepSpeed configs
├── scripts/           # Pipeline runner and evaluation scripts
├── datasets/          # Evaluation datasets
├── context/           # System prompts and tool schemas
└── .env               # Environment variables (HF_HOME, HF_TOKEN)
```

## Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Set up environment
cp .env.example .env  # Edit with your HF_HOME and HF_TOKEN
```

## Pipeline

The main entry point is `scripts/run_pipeline.sh`, configured via `config/pipeline.yaml`.

```bash
# Full pipeline: data generation + training
bash scripts/run_pipeline.sh

# Full pipeline with custom config
bash scripts/run_pipeline.sh config/my_config.yaml

# Training only (set train_only: true in pipeline.yaml)
bash scripts/run_pipeline.sh
```

### Pipeline Steps

1. **Generate questions** — uses vLLM server with the teacher model
2. **Convert to lesson format** — structures questions for training
3. **Generate exam** — creates evaluation set
4. **Generate teacher answers (train)** — teacher answers training questions
5. **Generate teacher answers (eval)** — teacher answers eval questions
6. **Train** — kills vLLM, loads teacher + student on GPUs, runs training

### Configuration (`config/pipeline.yaml`)

```yaml
project:
  run_name: my_run          # Experiment name
  train_only: false          # Skip data generation, go straight to training
  use_wandb: false
  system_prompt_path: ...    # System prompt for the teacher
  tools_schema_path: ...     # Tool schema for tool-calling tasks

models:
  teacher: qwen2.5-72b-instruct
  student: llama3.1-8b-instruct

dataset:
  datapath: output/teacher_answers   # Where generated data is saved/read
  custom_train_data: ""              # Absolute path to override auto-generated train data
  custom_val_data: ""                # Absolute path to override auto-generated val data

gpu:
  vllm: "0,1"              # GPUs for vLLM (auto tensor-parallel)
  train: "0,1,2,3,4"       # All GPUs visible during training
  teacher: "2,3,4"          # GPUs for teacher model (must not overlap with student DDP ranks)
  vllm_host: localhost

training:
  token_loss_weight: 1.0    # Cross-entropy on answer tokens
  logit_loss_weight: 0.5    # KL divergence from teacher
  closed_book: true         # Student sees question only (no material)
  accelerate_config: config/accelerate_ddp_5gpu.yaml
  # ... see pipeline.yaml for all options
```

## Training Modes

### SFT only (no distillation)

Standard supervised fine-tuning on teacher-generated answers:

```yaml
training:
  token_loss_weight: 1.0
  logit_loss_weight: 0.0
  accelerate_config: config/accelerate_ddp_2gpu.yaml
```

### Logit distillation

KL divergence from teacher logits, optionally combined with SFT:

```yaml
training:
  token_loss_weight: 1.0    # Set to 0.0 for logit-only
  logit_loss_weight: 0.5
  train_temperature: 2.0    # Temperature for KL softmax
  reverse_kl: false          # Forward KL (default) or reverse KL
  accelerate_config: config/accelerate_ddp_5gpu.yaml
```

### Training only with custom data

```yaml
project:
  train_only: true
dataset:
  custom_train_data: /path/to/train.json
  custom_val_data: /path/to/val.json
```

## Cross-Tokenizer Distillation

When the teacher and student use different tokenizers (e.g., Qwen → Llama), token-level KL divergence requires alignment since the same text produces different token sequences.

We use character-span alignment: each tokenizer provides character offsets for its tokens via `return_offsets_mapping=True`. For each student token, we find overlapping teacher tokens and compute weights proportional to the number of shared characters.

For example, given the text `"unhappiness"`:
- Student tokens: `["unhapp", "iness"]` → spans `(0,6)`, `(6,11)`
- Teacher tokens: `["un", "happi", "ness"]` → spans `(0,2)`, `(2,7)`, `(7,11)`

The alignment weight for student token `"unhapp"` (0-6):
- `"un"` (0-2): overlap = 2 → weight 2/6
- `"happi"` (2-7): overlap = 4 → weight 4/6
- `"ness"` (7-11): overlap = 0

The aligned teacher distribution at each student position is a weighted average of overlapping teacher token probabilities (in probability space, not logit space), then KL divergence is computed on the result.

When vocab sizes differ, a learned bottleneck projection (`VocabProjection`) maps teacher logits to the student vocab space before alignment. The projection is trained alongside the student LoRA.

### Key functions

- `build_alignment_weights()` — builds a sparse alignment matrix from character spans
- `build_shared_alignment()` — alignment for tool-call tokens with different formatting
- `VocabProjection` — bottleneck linear layer mapping teacher vocab → student vocab
- `_aligned_kl()` — per-sample KL with character-span alignment
- `_flat_kl()` — batched KL for same-tokenizer distillation

## Multi-GPU Setup

### GPU assignment

DDP always assigns rank N → CUDA device N. Accelerate sets `CUDA_VISIBLE_DEVICES` from the `gpu_ids` field in the accelerate config, which controls device remapping. The teacher's `teacher_gpus` must refer to remapped indices within that list and must not overlap with the first `num_processes` devices (claimed by student DDP).

Example with 5 GPUs — 2 for student, 3 for teacher:
- `accelerate_ddp_5gpu.yaml`: `gpu_ids: "0,1,2,3,4"`, `num_processes: 2`
- `pipeline.yaml`: `teacher: "2,3,4"`
- Student DDP rank 0 → device 0, rank 1 → device 1
- Teacher loaded on devices 2, 3, 4 via `device_map="auto"` with `max_memory`

### Accelerate configs

- `config/accelerate_ddp_2gpu.yaml` — SFT-only, 2 DDP processes on GPUs 2,3
- `config/accelerate_ddp_5gpu.yaml` — Logit distillation, 2 DDP student processes + 3 teacher GPUs

### Distributed teacher forward

When using logit loss with multiple DDP processes, the teacher runs only on rank 0. Teacher logits are distributed to other ranks via:

1. All ranks `gather` their inputs to rank 0
2. Rank 0 runs teacher forward on each rank's inputs
3. Rank 0 `send`s each rank its logits via point-to-point communication

This avoids broadcasting all logits to all ranks, keeping memory usage proportional to each rank's own batch.

### Training flow

- Student (with LoRA) is distributed across DDP ranks via `accelerator.prepare()`
- Teacher (frozen) is loaded only on rank 0, split across its assigned GPUs via `dispatch_model`
- `VocabProjection` (if needed) is trained alongside the student LoRA
- Each training step: logit loss (if enabled) → token loss (if enabled) → gradient accumulation → optimizer step

## Datasets

We use the four variants (Amazon, New Wiki, NYT, Reddit) of SquadShifts from HuggingFace, and the distractor-dataset from HotpotQA. Our evaluation set (1000 questions / dataset) can be found in `datasets/`.

## Other Repositories

For MMLU-Pro evaluation, we used:
https://github.com/TIGER-AI-Lab/MMLU-Pro

For generation of training questions with Bonito, we used:
https://github.com/BatsResearch/bonito with the following prompt:

```
<|tasktype|>
extractive question answering
<|context|>
{context}
<|task|>
```
