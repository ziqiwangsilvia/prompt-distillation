# Codebase for Prompt Distillation

## Overview
This repository contains the code necessary to reproduce the main results of our Prompt Distillation submission.

## Repository Structure

```
├── models/            # LLM class, model configs, messages, tool call format
├── data/              # File naming and dataset loading utilities
├── curriculum/        # Lesson/exercise data structures and curriculum generation scripts
├── training/          # Code for performing training runs 
├── evaluation/        # Miscellaneous scripts for evaluation 
├── config/            # DeepSpeed Configs 
├── datasets/          # Evaluation datasets 
├── paths.py           # Path constants and markup tags (TIPS, DELIMITER)
├── utils.py           # General utilities (seeding, GPU, sampling params, etc.)
├── requirements.txt   # Dependencies
├── README.md          # This file
└── setup.py           # Setup file 
```

## Installation

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Usage
Basic instructions for reproducing the main Prompt Distillation result on NYT with Llama-3-8B-Instruct.

```bash
# Launch vLLM server
python3 -u -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --dtype auto --api-key token-abc123 --tensor-parallel-size <N_GPUS>

# Generate questions
python3 evaluation/sample_questions.py --vllm_hostname <VLLM_HOST>

# Convert generated questions into lesson format
python3 curriculum/csv_to_lesson.py

# Generate test set questions in lesson format
python3 curriculum/questions_to_exam.py --max_items 20

# Generate answers to training questions
python3 curriculum/generate_teacher_answers.py --generate_lesson True

# Generate answers to test set questions
python3 curriculum/generate_teacher_answers.py --generate_exam True --max_items 20

# Run training
python3 training/train.py --run_name example --use_wandb True

# Run final evaluation
python3 evaluation/evaluate.py --adapter_id example

# Run final evaluation with RAG
python3 evaluation/evaluate.py --adapter_id example --output_filename output_rewritten_rag.csv --openai_rag True

# Run grading
python3 evaluation/grade_answers_llm.py --input_path "./outputs/squadshifts_nyt/**/output*" --vllm_hostname <VLLM_HOST>
python3 evaluation/grade_answers_match.py --input_path "./outputs/squadshifts_nyt/**/output*"
```

To reproduce SFT experiments on NYT with Llama-3-8B-Instruct, generate answers to training questions at a lower temperature and run training using a different command:

```bash
# Generate answers to training questions
python3 curriculum/generate_teacher_answers.py --generate_lesson True --lesson_temp 0.25

# Run training
python3 training/train.py --run_name example_sft --use_wandb True --logit_loss_weight 0.0 --token_loss_weight 1.0 --lesson_temp 0.25
```

Example, to get the PD results for Qwen2.5-14B-Instruct on Amazon, we re-generate the questions and answers with qwen, and run the training using DeepSpeed
```bash
# Launch vLLM server
python3 -u -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-14B-Instruct --dtype auto --api-key token-abc123 --tensor-parallel-size <N_GPUS>

# Generate questions
python3 evaluation/sample_questions.py --vllm_hostname <VLLM_HOST> --base qwen2.5-14b-instruct --dataset amazon

# Convert generated questions into lesson format
python3 curriculum/csv_to_lesson.py --dataset amazon --model qwen2.5-14b-instruct

# Generate answers to training questions
python3 curriculum/generate_teacher_answers.py --generate_lesson True --base qwen2.5-14b-instruct --dataset amazon --question_model qwen2.5-14b-instruct

# Run training
accelerate launch training/train.py --deepspeed_path config/ds2.json --run_name example_qwen_amazon --use_wandb True --base qwen2.5-14b-instruct --dataset amazon --question_model qwen2.5-14b-instruct --lesson_model qwen2.5-14b-instruct --n_epochs 1 --warmup_steps 100

# Run final evaluation
python3 evaluation/evaluate.py --adapter_id example_qwen_amazon --dataset amazon
```

For all HotpotQA experiments, use
```bash
--dataset_family hotpotqa --dataset distractor
```

## Datasets

We use the the four variants (Amazon, New Wiki, NYT, Reddit) of Squadshift from HuggingFace, and the distractor-dataset from HotpotQA. Our evaluation set (1000 questions / dataset) can be found in datasets/

## Other repositories

For MMLU-Pro evaluation, we used the following repo:
https://github.com/TIGER-AI-Lab/MMLU-Pro

For generation of training questions with Bonito, we used the official repo:
https://github.com/BatsResearch/bonito with the following prompt

```
<|tasktype|>
extractive question answering
<|context|>
{context}
<|task|>
```
