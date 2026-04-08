from collections import Counter
import json
import numpy as np
import os
import re
from pathlib import Path
import random
from typing import List, Dict, Any, Union, Tuple
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

from peft import get_peft_model, LoraConfig, TaskType
import torch
from torch.utils.data import Sampler
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel

from models.llm import LLM
from models.messages import Message
from models.configs import MODEL_CONFIGS
from data.paths import TIPS_START, TIPS_END, DELIMITER
from curriculum.exercise_with_answers import ExerciseWithAnswers
from training.metrics import Aggregator


def remove_non_xml_chars(text: str) -> str:
    """
    Remove any character not allowed by XML 1.0.
    """
    disallowed_chars_pattern = re.compile(
        '[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFDCF\uFDF0-\uFFFD]'
        '|[\uD800-\uDFFF]|[\U00010000-\U0010FFFF]', re.UNICODE)
    return disallowed_chars_pattern.sub('', text)


def clean_xml_content(filename: str) -> str:
    """
    Remove invalid XML characters from file, saving to a new file with .cleaned suffix.
    Returns the new filename.
    """
    with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()
    cleaned_content = remove_non_xml_chars(content)
    temp_filename = filename + '.cleaned'
    with open(temp_filename, 'w', encoding='utf-8') as file:
        file.write(cleaned_content)
    return temp_filename


def warn(msg: str) -> None:
    print(f"[WARNING] {msg}", flush=True)


def ensure_path_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")


def read_exercises(filepath: Path) -> List[ExerciseWithAnswers]:
    """
    Read an XML file and parse all exercises into ExerciseWithAnswers objects.
    """
    ensure_path_exists(filepath)
    with open(filepath, 'r', encoding='utf-8') as file:
        xml_content = file.read()
    xml_content = xml_content.replace(TIPS_START, escape(TIPS_START))
    xml_content = xml_content.replace(TIPS_END, escape(TIPS_END))
    xml_content = remove_non_xml_chars(xml_content)
    try:
        root = ET.fromstring(xml_content)
    except Exception as e:
        warn(f"Failed to parse XML: {filepath}")
        raise
    lesson_id = filepath.stem
    if 'LOCAL_RANK' not in os.environ or os.environ['LOCAL_RANK'] == "0":
        print(f"lesson_id {lesson_id}")
    return [
        ExerciseWithAnswers.from_xml(ex, lesson_id=lesson_id)
        for ex in root.findall("exercise_with_answers")
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
    The LLM should already be on the correct device.
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

            # Move to correct device
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


def substring_locations(s: str, sub: str) -> List[int]:
    return [match.start() for match in re.finditer(re.escape(sub), s)]


def tip_split(s: str) -> Tuple[List[str], List[bool]]:
    begin_locations = substring_locations(s, TIPS_START)
    end_locations = substring_locations(s, TIPS_END)
    if len(begin_locations) != len(end_locations):
        raise ValueError(f"Mismatch between {TIPS_START} and {TIPS_END} markers.")
    for i in range(len(begin_locations)-1):
        if begin_locations[i] > end_locations[i] or (i < len(end_locations) - 1 and begin_locations[i+1] < end_locations[i]):
            raise ValueError(f"Misplaced {TIPS_START} or {TIPS_END} marker.")
    parts: List[str] = []
    tip: List[bool] = []
    last_index = 0
    for begin, end in zip(begin_locations, end_locations, strict=True):
        parts.append(s[last_index:begin])
        tip.append(False)
        parts.append(s[begin+len(TIPS_START):end])
        tip.append(True)
        last_index = end + len(TIPS_END)
    if last_index < len(s):
        parts.append(s[last_index:])
        tip.append(False)
    return parts, tip


def tokenize(prompt_with_tips: str, llm: LLM) -> Tuple[torch.Tensor, torch.Tensor]:
    parts, tip = tip_split(prompt_with_tips)
    teacher_prompt = "".join(parts)
    teacher_tokens = llm.tokenize(teacher_prompt)
    teacher_tokens = llm.add_bos(teacher_tokens)
    student_prompt = "".join([parts[i] for i in range(len(parts)) if not tip[i]])
    student_tokens = llm.tokenize(student_prompt)
    student_tokens = llm.add_bos(student_tokens)
    return student_tokens, teacher_tokens


def extract_question(message: Message) -> str:
    content = message.content
    last_tips_end = content.rfind(TIPS_END)
    if last_tips_end == -1:
        return content.strip()
    return content[last_tips_end + len(TIPS_END):].strip()


def extract_material_and_question(message: Message) -> Tuple[str, str]:
    """
    Split the message content into (material, question) using <TIPS> markers.
    """
    content = message.content
    parts, tip_flags = tip_split(content)
    # All consecutive True parts are material/context/tips.
    # The last False part is the question (after the last </TIPS>)
    # All text up to and including the last True part is material.
    if not parts:
        return "", ""
    # Find last True
    last_tip_idx = None
    for i in reversed(range(len(tip_flags))):
        if tip_flags[i]:
            last_tip_idx = i
            break
    if last_tip_idx is None:
        # No <TIPS> at all, everything is question
        return "", content.strip()
    # Material: everything up to last_tip_idx (inclusive)
    material = "".join(parts[:last_tip_idx+1])
    material = material.replace(DELIMITER + "\n\n", "").strip()
    # Question: everything after
    question = "".join(parts[last_tip_idx+1:]).strip()
    return material.strip(), question


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
    """
    Extract primitive configuration values for logging.
    """
    primitive_types = (int, float, str, bool)
    return {k: v for k, v in local_vars.items()
            if isinstance(v, primitive_types) and not k.startswith('_')}


def setup_wandb(use_wandb: bool, project: str, group: str, run_name: str, config: Dict[str, Any]) -> None:
    """
    Initialize wandb if requested.
    """
    if use_wandb:
        import wandb
        wandb.init(
            project=project,
            group=group,
            name=run_name,
            config=config
        )


def setup_tokenizer_and_model(model_name: str) -> Union[AutoTokenizer, PreTrainedModel]:
    """
    Initialize tokenizer and base model.
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model '{model_name}' not supported. Available models: {list(MODEL_CONFIGS.keys())}")
    config = MODEL_CONFIGS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(config.vllm_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        config.vllm_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    # Ensure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        base_model.config.pad_token_id = base_model.config.eos_token_id
    return tokenizer, base_model


def print_token_tensor(tsr: torch.Tensor, base_llm: LLM) -> None:
    tsr = torch.where(tsr < 0, torch.zeros_like(tsr), tsr)
    if tsr.ndim == 1:
        tsr = tsr[None]
    print(base_llm.decode(tsr))
