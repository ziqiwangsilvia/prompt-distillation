from dataclasses import dataclass
from typing import List, Dict

@dataclass
class ModelConfig:
    name: str
    vllm_model: str
    system_message: str
    flag_name: str
    lora_targets: List[str] = None


# Model configurations
MODEL_CONFIGS = {
    "llama3-8b-instruct": ModelConfig(
        name="llama3-8b-instruct",
        vllm_model="meta-llama/Llama-3.1-8B-Instruct",
        system_message="You are a knowledgeable assistant trained to provide accurate and helpful information. Please respond to the user's queries promptly.",
        flag_name="llama3_8b",
        lora_targets=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
    ),
    "llama3-70b-instruct": ModelConfig(
        name="llama3-70b-instruct",
        vllm_model="meta-llama/Llama-3.1-70B-Instruct",
        system_message="You are a knowledgeable assistant trained to provide accurate and helpful information. Please respond to the user's queries promptly.",
        flag_name="llama3_70b",
        lora_targets=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
    ),
    "qwen2.5-3b-instruct": ModelConfig(
        name="qwen2.5-3b-instruct",
        vllm_model="Qwen/Qwen2.5-3B-Instruct",
        system_message="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        flag_name="qwen25_3b",
        lora_targets=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
    ),
    "qwen2.5-7b-instruct": ModelConfig(
        name="qwen2.5-7b-instruct",
        vllm_model="Qwen/Qwen2.5-7B-Instruct",
        system_message="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        flag_name="qwen25_7b",
        lora_targets=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
    ),
    "qwen2.5-14b-instruct": ModelConfig(
        name="qwen2.5-14b-instruct",
        vllm_model="Qwen/Qwen2.5-14B-Instruct",
        system_message="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        flag_name="qwen25_14b",
        lora_targets=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
    ),
    "qwen2.5-72b-instruct": ModelConfig(
        name="qwen2.5-72b-instruct",
        vllm_model="Qwen/Qwen2.5-72B-Instruct",
        system_message="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        flag_name="qwen25_72b",
        lora_targets=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
    ),
}


def create_model_flags(model_name: str) -> Dict[str, bool]:
    """Create model flags dictionary."""
    flags = {config.flag_name: False for config in MODEL_CONFIGS.values()}
    if model_name in MODEL_CONFIGS:
        flags[MODEL_CONFIGS[model_name].flag_name] = True
    return flags


def get_model_config(model_name: str) -> ModelConfig:
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]
    for k, v in MODEL_CONFIGS.items():
        if model_name in (v.name, v.vllm_model):
            return v
    raise ValueError(f"Unknown model {model_name}")
