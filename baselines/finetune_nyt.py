import json
import numpy as np
import random
import torch
import wandb
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, Dataset
from peft import get_peft_model, LoraConfig, TaskType

from core.utils import set_seed, load_squadshifts
from core.model_configs import MODEL_CONFIGS
from training.utils import (
    extract_primitive_config,
    setup_wandb,
    setup_tokenizer_and_model,
    save_base_model_config,
)

SUPPORTED_DATASETS = ["nyt", "reddit", "amazon", "new_wiki"]


def load_and_prepare_dataset(dataset: str, n_items: int) -> Dataset:
    """Load and select subset of dataset."""
    if dataset not in SUPPORTED_DATASETS:
        raise NotImplementedError(f"Unknown dataset {dataset}. Supported: {SUPPORTED_DATASETS}")

    full_dataset = load_squadshifts(dataset)
    return full_dataset.select(range(n_items))


def chunk_and_tokenize(examples: Dict[str, List[str]], tokenizer, max_length: int, overlap: int) -> Dict[str, List[List[int]]]:
    """Chunk and tokenize text examples."""
    chunks = []

    for text in examples['context']:
        # Tokenize without padding or truncation
        tokens = tokenizer(text, truncation=False, padding=False)['input_ids']

        # Chunk the tokens
        for i in range(0, len(tokens), max_length - overlap):
            chunk = tokens[i:i + max_length]
            # Only add EOS token if this is the last chunk
            if i + max_length >= len(tokens):
                chunk = chunk + [tokenizer.eos_token_id]
            chunks.append(chunk)

    # Pad or truncate all chunks to max_length
    padded_chunks = []
    for chunk in chunks:
        if len(chunk) > max_length:
            padded_chunks.append(chunk[:max_length])
        else:
            padded_chunks.append(chunk + [tokenizer.pad_token_id] * (max_length - len(chunk)))

    return {"input_ids": padded_chunks}


def remove_duplicate_contexts(examples: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Remove duplicate contexts from examples."""
    unique_contexts = list(dict.fromkeys(examples['context']))
    return {'context': unique_contexts}


def prepare_dataset_for_training(dataset: Dataset, tokenizer, max_length: int, overlap: int) -> Dataset:
    """Prepare dataset for training by removing duplicates and tokenizing."""
    # Remove duplicates
    unique_context_dataset = dataset.map(
        remove_duplicate_contexts,
        batched=True,
        remove_columns=[col for col in dataset.column_names if col != 'context']
    )

    # Tokenize and chunk
    tokenized_datasets = unique_context_dataset.map(
        lambda examples: chunk_and_tokenize(examples, tokenizer, max_length, overlap),
        batched=True,
        remove_columns=['context']
    )

    return tokenized_datasets.shuffle()


def create_training_arguments(
    output_dir: str,
    n_epochs: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    warmup_steps: int,
    weight_decay: float,
    use_wandb: bool
) -> TrainingArguments:
    """Create training arguments."""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=n_epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_steps=1,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=5000,
        save_only_model=True,
        bf16=True,
        report_to=["wandb"] if use_wandb else [],
        save_safetensors=True,
        push_to_hub=False,
    )


def save_base_model_config_uft(output_dir: str, model_name: str) -> None:
    """Save base model configuration."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model '{model_name}' not supported. Available models: {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[model_name]
    base_model_config = {
        "model_path": config.vllm_model,
        "adapter_ids": [],
    }
    save_base_model_config(base_model_config, output_dir, verbose=False)


def create_lora_config(model_name: str, lora_r: int) -> LoraConfig:
    """Create LoRA configuration."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model '{model_name}' not supported. Available models: {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[model_name]
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_r * 2,
        lora_dropout=0.05,
        target_modules=config.lora_targets
    )


def main(
    dataset: str = "nyt",
    n_items: int = 1000,
    model_name: str = "llama3-8b-instruct",
    lr: float = 1e-5,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 2,
    n_epochs: int = 3,
    lora_r: int = 1024,
    weight_decay: float = 0.1,
    warmup_steps: int = 100,
    seed: int = None,
    project: str = "huggingface",
    group: str = "default",
    run_name: str = "default",
    use_wandb: bool = False,
    max_length: int = 256,
    overlap: int = 64,
    debug: bool = False,
) -> None:
    # Set seed for reproducibility
    if seed:
        set_seed(seed)

    # Load dataset
    hf_dataset = load_and_prepare_dataset(dataset, n_items)

    # Setup model and tokenizer
    tokenizer, base_model = setup_tokenizer_and_model(model_name)

    # Create LoRA configuration
    peft_config = create_lora_config(model_name, lora_r)

    # Prepare dataset for training
    train_dataset = prepare_dataset_for_training(hf_dataset, tokenizer, max_length, overlap)

    # Setup data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Wrap the model with PEFT
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    # Setup output directory and config
    output_dir = Path(f"./checkpoints/{project}/{run_name}")
    config = extract_primitive_config(locals())

    # Setup wandb
    setup_wandb(use_wandb, project, group, run_name, config)

    # Create training arguments
    training_args = create_training_arguments(
        output_dir, n_epochs, per_device_batch_size, gradient_accumulation_steps,
        warmup_steps, weight_decay, use_wandb
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Save base model config
    save_base_model_config_uft(output_dir, model_name)

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

    # Cleanup wandb
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
