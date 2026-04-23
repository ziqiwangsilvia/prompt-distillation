import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from peft import LoraConfig

from paths import BASE_PATH
from models.configs import MODEL_CONFIGS, ModelConfig
from models.messages import Message, Role


@dataclass
class StudentArgs:
    base: str = "llama3-8b-instruct"
    lora_r: int = 1024
    mixed_precision: str = "bf16"
    closed_book: bool = True
    knowledge_cutoff: str = "default"  # "default" keeps original, null removes it, any string overrides it
    # Computed in __post_init__
    peft_config: LoraConfig = field(init=False, default=None)
    opening_message: Message = field(init=False, default=None)

    def __post_init__(self):
        if self.base not in MODEL_CONFIGS:
            raise ValueError(f"Unknown base model: {self.base}")
        model_config = MODEL_CONFIGS[self.base]
        self.opening_message = Message(Role.SYSTEM, model_config.system_message)
        self.peft_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_r * 2,
            target_modules=model_config.lora_targets,
            lora_dropout=0.05,
        )


@dataclass
class TeacherArgs:
    teacher: str = "qwen2.5-7b-instruct"
    teacher_gpus: str = "0,1"
    train_temperature: float = 2.0
    token_loss_weight: float = 0.0
    logit_loss_weight: float = 1.0
    reverse_kl: bool = False
    vocab_mapping: str = "svd"  # "svd" (from shared tokens), "bottleneck" (random init), or "topk" (top-K revert, no learned params)
    n_topk: int = 100  # number of top-K teacher tokens to use when vocab_mapping="topk"


@dataclass
class DataArgs:
    dataset: str = "nyt"
    dataset_family: str = "squadshifts"
    variant: str = "default"
    lesson_model: str = "llama3-8b-instruct"
    exam_model: str = "llama3-8b-instruct"
    lesson_temp: float = 1.5
    exam_temp: float = 0.25
    lesson_num_choices: int = 1
    exam_num_choices: int = 1
    question_model: str = "llama3-8b-instruct"
    question_temperature: float = 1.5
    train_questions: int = 30
    max_items_train: int = 1000
    max_items_test: int = 20
    max_length: int = 0
    max_total_length: int = 0
    datapath: Path = field(default_factory=lambda: Path("output/teacher_answers"))
    custom_train_data: str = ""
    custom_val_data: str = ""
    tools_schema_path: str = ""
    use_tool_token: bool = False
    test_mode: bool = False
    multi_turn: bool = False
    distractor_dataset: str = ""
    partition_idx: Optional[int] = None
    partition_type: Optional[str] = None


@dataclass
class TrainingArgs:
    group_name: str = None
    run_name: str = None
    learning_rate: float = 1e-5
    batch_size: int = 4  # per device
    micro_batch_size: int = 4  # per device
    n_epochs: int = 2
    weight_decay: float = 0.1
    warmup_steps: int = None
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    decay: bool = True
    seed: int = 0
    eval_interval: int = -1
    log_interval: int = 1
    generation_interval: int = 0  # -1 every epoch, 0 never
    validate: bool = True
    save: bool = True
    save_during_training: bool = False
    checkpoint_interval: int = 0
    checkpoint_interval_seconds: int = 0
    use_wandb: bool = False
    deepspeed_path: str = ""
    deepspeed_path_teacher: str = ""
    # Computed in __post_init__
    project_path: Path = field(init=False, default=None)
    logit_loss_micro_batch_size: int = field(init=False, default=0)
    token_loss_micro_batch_size: int = field(init=False, default=0)
    n_logit_micro_batches_per_batch: int = field(init=False, default=0)
    n_token_micro_batches_per_batch: int = field(init=False, default=0)

    def __post_init__(self):
        self.project_path = BASE_PATH / "output" / "checkpoints"
        self.logit_loss_micro_batch_size = self.micro_batch_size
        self.token_loss_micro_batch_size = self.micro_batch_size
        self.n_logit_micro_batches_per_batch = math.ceil(self.batch_size / self.micro_batch_size)
        self.n_token_micro_batches_per_batch = math.ceil(self.batch_size / self.micro_batch_size)
