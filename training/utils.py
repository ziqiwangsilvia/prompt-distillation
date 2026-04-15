from collections import Counter
import json
import numpy as np
import os
import re
from pathlib import Path
import random
from typing import List, Dict, Any, Union, Tuple

from peft import get_peft_model, LoraConfig, TaskType
import torch
from torch.utils.data import Sampler
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel

from models.llm import LLM
from models.messages import Message, Role
from models.configs import MODEL_CONFIGS
from data.paths import DELIMITER
from curriculum.exercise_with_answers import ExerciseWithAnswers


def warn(msg: str) -> None:
    print(f"[WARNING] {msg}", flush=True)


def ensure_path_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")


def read_exercises(filepath: Path) -> List[ExerciseWithAnswers]:
    """
    Read a JSON file and parse all exercises into ExerciseWithAnswers objects.
    """
    ensure_path_exists(filepath)
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    lesson_id = filepath.stem
    if 'LOCAL_RANK' not in os.environ or os.environ['LOCAL_RANK'] == "0":
        print(f"lesson_id {lesson_id}")
    return [
        ExerciseWithAnswers.from_dict(ex, lesson_id=lesson_id)
        for ex in data["exercises_with_answers"]
    ]


def generate_answers(
    base_llm: LLM,
    generation_samples: List[Dict[str, Any]],
    accelerator,
    *,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
) -> List[str]:
    """
    Generate answers for a batch of samples using the LLM's generate method.
    """
    base_llm.model.eval()
    answers: List[str] = []

    with torch.no_grad():
        for sample in generation_samples:
            input_ids = sample.get("prompt_tokens")
            if input_ids is None:
                input_ids = sample.get("student_prompt_tokens")
            if input_ids is None:
                continue
            attention_mask = torch.ones_like(input_ids)

            input_ids = input_ids.to(accelerator.device)
            attention_mask = attention_mask.to(accelerator.device)

            output, _ = base_llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
            print(f"[Generated answer, {accelerator.process_index}]:\n", output, "\n", flush=True)
            answers.append(output)

    base_llm.model.train()
    return answers


def tokenize_teacher_student(material: str, question: str, llm: LLM, teacher_llm: LLM = None, tools: list = None, student_tools: list = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize teacher and student prompts using chat template.
    Teacher: material as system message + question (uses teacher_llm tokenizer if provided).
    Student open-book: material as system message + question (student tokenizer).
    Student closed-book: default system message + question (student tokenizer).
    Returns (student_closed_tokens, student_open_tokens, teacher_tokens).
    """
    t_llm = teacher_llm or llm

    # Teacher: use material as system message
    saved = t_llm.opening_message
    t_llm.opening_message = Message(Role.SYSTEM, material) if material else None
    teacher_tokens = t_llm.tokenize(t_llm.messages_to_prompt([Message(Role.USER, question)], tools=tools))
    t_llm.opening_message = saved

    # Student open-book: material as system message (student tokenizer)
    saved = llm.opening_message
    llm.opening_message = Message(Role.SYSTEM, material) if material else None
    student_open_tokens = llm.tokenize(llm.messages_to_prompt([Message(Role.USER, question)], tools=student_tools))
    llm.opening_message = saved

    # Student closed-book: default system message
    student_closed_tokens = llm.tokenize(llm.messages_to_prompt([Message(Role.USER, question)], tools=student_tools))

    return student_closed_tokens, student_open_tokens, teacher_tokens


def extract_question(exercise: ExerciseWithAnswers) -> str:
    """Extract the question (last user message content)."""
    for msg in reversed(exercise.messages):
        if msg.role.value == "user":
            return msg.content.strip()
    return exercise.messages[-1].content.strip()


def extract_material_and_question(exercise: ExerciseWithAnswers) -> Tuple[str, str]:
    """Extract (material, question) from exercise messages."""
    material = ""
    question = ""
    for msg in exercise.messages:
        if msg.role.value == "system":
            material = msg.content.replace(DELIMITER + "\n\n", "").strip()
        elif msg.role.value == "user":
            question = msg.content.strip()
    return material, question


class InfiniteSampler(Sampler):
    def __init__(self, data_source_length: int) -> None:
        self.data_source_length = data_source_length

    def __iter__(self):
        while True:
            yield from torch.randperm(self.data_source_length)

    def __len__(self) -> int:
        return float('inf')


def save_base_model_config(base_llm_config: Dict[str, Any], run_project_dir: Path, verbose: bool = True) -> None:
    with open(run_project_dir / "base_model_config.json", 'w', encoding='utf-8') as f:
        json.dump(base_llm_config, f, ensure_ascii=False, indent=4)
    if verbose:
        print(f"Saved to {run_project_dir}")


def save_with_base_model_config(model: PreTrainedModel, base_llm: LLM, run_project_dir: Path) -> None:
    model.save_pretrained(run_project_dir)
    save_base_model_config(base_llm.get_config(), run_project_dir)


def save_with_deepspeed(model: PreTrainedModel, accelerator, base_llm: LLM, run_project_dir: Path) -> None:
    unwrapped_model = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        unwrapped_model.save_pretrained(run_project_dir)
        save_base_model_config(base_llm.get_config(), run_project_dir, verbose=True)


def extract_primitive_config(local_vars: Dict[str, Any]) -> Dict[str, Any]:
    primitive_types = (int, float, str, bool)
    return {k: v for k, v in local_vars.items()
            if isinstance(v, primitive_types) and not k.startswith('_')}


def setup_wandb(use_wandb: bool, project: str, group: str, run_name: str, config: Dict[str, Any]) -> None:
    if use_wandb:
        import wandb
        wandb.init(
            project=project,
            group=group,
            name=run_name,
            config=config
        )


def setup_tokenizer_and_model(model_name: str) -> Union[AutoTokenizer, PreTrainedModel]:
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model '{model_name}' not supported. Available models: {list(MODEL_CONFIGS.keys())}")
    config = MODEL_CONFIGS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(config.vllm_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        config.vllm_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        base_model.config.pad_token_id = base_model.config.eos_token_id
    return tokenizer, base_model


def print_token_tensor(tsr: torch.Tensor, base_llm: LLM) -> None:
    tsr = torch.where(tsr < 0, torch.zeros_like(tsr), tsr)
    if tsr.ndim == 1:
        tsr = tsr[None]
    print(base_llm.decode(tsr))
