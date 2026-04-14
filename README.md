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
│   ├── loss.py        # Token loss, logit loss (flat + aligned KL)
│   ├── projection.py  # VocabProjection + character-span alignment
│   ├── params.py      # All training hyperparameters
│   └── utils.py       # Tokenization, generation, data loading helpers
├── evaluation/        # Evaluation and grading scripts
├── config/            # Pipeline YAML config and DeepSpeed configs
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
  student: llama3-8b-instruct

dataset:
  datapath: output/teacher_answers   # Where generated data is saved/read
  custom_train_data: ""              # Absolute path to override auto-generated train data
  custom_val_data: ""                # Absolute path to override auto-generated val data

gpu:
  vllm: "0,1"              # GPUs for vLLM (auto tensor-parallel)
  train: "0,1"             # GPUs for training

training:
  token_loss_weight: 1.0    # Cross-entropy on answer tokens
  logit_loss_weight: 0.5    # KL divergence from teacher
  closed_book: true         # Student sees question only (no material)
  # ... see pipeline.yaml for all options
```

## Usage Examples

### Same-tokenizer distillation (Llama → Llama)

```bash
# Launch vLLM server
python3 -u -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct --dtype auto --api-key token-abc123

# Generate data, then train
bash scripts/run_pipeline.sh
```

### Cross-tokenizer distillation (Qwen → Llama)

Set in `pipeline.yaml`:
```yaml
models:
  teacher: qwen2.5-72b-instruct
  student: llama3-8b-instruct
```

The pipeline automatically:
- Detects vocab mismatch and creates a `VocabProjection` (bottleneck linear layer)
- Uses character-span alignment for token-level KL
- Splits the teacher across multiple GPUs via `dispatch_model`

### Training only with custom data

```yaml
project:
  train_only: true
dataset:
  custom_train_data: /path/to/train.json
  custom_val_data: /path/to/val.json
```

### SFT only (no distillation)

```yaml
training:
  token_loss_weight: 1.0
  logit_loss_weight: 0.0
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

When vocab sizes differ, a learned bottleneck projection (`VocabProjection`) maps teacher logits to the student vocab space before alignment.

## Multi-GPU Setup

- **Data generation**: vLLM uses tensor parallelism across all `gpu.vllm` GPUs
- **Training**: vLLM is shut down, then:
  - Student (with LoRA) is placed on one GPU via `accelerator.prepare()`
  - Teacher (frozen) is split across all GPUs via `dispatch_model` with an explicit layer-based device map
  - `VocabProjection` is trained alongside the student LoRA

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
