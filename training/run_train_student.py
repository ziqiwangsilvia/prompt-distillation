import math
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, Union, Optional

from peft import LoraConfig
import randomname

from core import BASE_PATH
from core.file_naming import generate_lesson_name, generate_exam_name, generate_augmented_filename 
from core.llm import LLM
from core.messages import Message, Role
from core.model_configs import MODEL_CONFIGS, create_model_flags
from core.utils import set_seed
from training.train_student import train

def main(
    # This is the student model being trained 
    base: str = "llama3-8b-instruct",
    project_name: str = "huggingface",
    group_name: str = None,
    eval_interval: int = -1,
    save_interval: float = float("inf"),
    log_interval: int = 1,
    generation_interval: int = 0, # -1 every epoch, 0 never
    # Hyperparameters
    learning_rate: float = 1e-5,
    # Note: batch_size and micro_batch_size per device
    batch_size: int = 4,
    micro_batch_size: int = 4,
    n_epochs: int = 2,
    train_temperature: float = 2.0,
    kl_div_loss: bool = True,
    token_loss_weight: float = 0.0,
    logit_loss_weight: float = 1.0,
    teacher: Union[Literal["student", "student_base"], str] = "student_base",
    lora_r: int = 1024,
    lora_type: str = "full",
    weight_decay: float = 0.1,
    warmup_steps: int = None,
    warmup_ratio: float = 0.1,
    run_name: str = None,
    # dataset specification
    lesson_model: str = "llama3-8b-instruct",
    exam_model: str = "llama3-8b-instruct",
    dataset: str = "nyt",
    dataset_family: str = "squadshifts",
    variant: str = "default",
    lesson_temp: float = 1.5,
    exam_temp: float = 0.25,
    lesson_num_choices: int = 1,
    exam_num_choices: int = 1,
    use_wandb: bool = False,
    validate: bool = True,
    seed: int = 0,
    mixed_precision: str = "bf16",
    max_grad_norm: float = 1.0,
    decay: bool = True,
    save: bool = True,
    save_during_training: bool = False,
    checkpoint_interval: int = 0,
    checkpoint_interval_seconds: int = 0,
    datapath: Path = Path("data"),
    # question generating model
    question_model: str = "llama3-8b-instruct",
    question_temperature: float = 1.5,
    train_questions: int = 30,
    max_items_train: int = 1000,
    max_items_test: int = 20,
    max_length: int = 0,
    max_total_length: int = 0,
    closed_book_token_loss: bool = True,
    distractor_dataset: str = "",
    deepspeed_path: str = "",
    deepspeed_path_teacher: str = "",
    reverse_kl: bool = False,
    partition_idx: Optional[int] = None,
    partition_type: Optional[str] = None,
    tulu: bool = False,
    tulu_batch_size: int = 2,
):
    if seed:
        set_seed(seed)
    project_path = BASE_PATH / "checkpoints" / project_name

    # Train lesson
    base_lesson_id = generate_lesson_name(
        dataset_family=dataset_family,
        dataset=dataset,
        variant=variant,
        model=question_model,
        questions=train_questions,
        temperature=question_temperature,
        max_items=max_items_train
    )

    # Exam
    base_exam_id = generate_exam_name(
        dataset_family=dataset_family,
        dataset=dataset,
        variant=variant,
        max_items=max_items_test
    )

    lesson_model_flags = create_model_flags(lesson_model)
    exam_model_flags = create_model_flags(exam_model)

    train_file = generate_augmented_filename(
        lesson_filename=base_lesson_id,
        n_choices=lesson_num_choices,
        temperature=lesson_temp,
        model_flags=lesson_model_flags,
        partition_idx=partition_idx,
        partition_type=partition_type,
        suffix="xml",
    )

    val_file = generate_augmented_filename(
        lesson_filename=base_exam_id,
        n_choices=exam_num_choices,
        temperature=exam_temp,
        model_flags=exam_model_flags,
        suffix="xml",
    )

    if validate:
        data = [[train_file], [val_file]]
    else:
        data = [[train_file], []]

    # --- System message ---
    if base not in MODEL_CONFIGS:
        raise ValueError(f"Unknown base model: {base}")
    model_config = MODEL_CONFIGS[base]
    opening_message = Message(Role.SYSTEM, model_config.system_message)
    base_llm = LLM(base, opening_message=opening_message)

    if not run_name:
        run_name = randomname.get_name()
    if not group_name:
        group_name = "initial_runs"

    # --- LoRA config (from model config) ---
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,
        target_modules=model_config.lora_targets,
        lora_dropout=0.05,
    )

    # Derived params for loss balancing
    logit_loss_micro_batch_size = micro_batch_size
    token_loss_micro_batch_size = micro_batch_size
    n_logit_micro_batches_per_batch = math.ceil(batch_size / micro_batch_size)
    n_token_micro_batches_per_batch = math.ceil(batch_size / micro_batch_size)

    # --- Collect all hyperparameters into a namespace ---
    hparams = {
        k: v for k, v in locals().items()
        if (isinstance(v, (int, float, str, bool, dict, LoraConfig, Path, Message)) or v is None)
        and not k.startswith("_")
        and not k.isupper()
    }
    hparams = SimpleNamespace(**hparams)

    # Validate
    if distractor_dataset and "llama3" not in base:
        raise NotImplementedError("Distractors only tested for LLama3 models. If you want to use distractors for Qwen, comment out this line and make sure that the outputs are reasonable.")

    train(
        project_path=project_path,
        base_llm=base_llm,
        data=data,
        hp=hparams,
    )

if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
