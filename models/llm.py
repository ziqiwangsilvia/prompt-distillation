import os
import json
import time
import torch
import warnings
from pathlib import Path
from typing import Union, List, Optional, Tuple

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from .configs import get_model_config
from data.paths import MODEL_PATH
from .messages import Message, Role, merge_messages, QUESTION_PLACEHOLDER
from models.utils import get_adapter_path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_model_family(model_id: Union[str, Path]) -> str:
    model_id = str(model_id).lower()
    if "llama" in model_id:
        return "llama"
    elif "qwen" in model_id:
        return "qwen"
    raise ValueError(f"Model family not recognized for {model_id}")


def get_system_message(model_id: Union[str, Path]) -> str:
    cfg = get_model_config(str(model_id))
    return cfg.system_message


class LLM:
    def __init__(
        self,
        base_model_name_or_path: Union[str, os.PathLike],
        adapter_ids: Optional[List[Path]] = None,
        opening_message: Optional[Message] = None,
    ):
        cfg = get_model_config(str(base_model_name_or_path))
        self.model_path = cfg.vllm_model
        self.model_family = get_model_family(self.model_path)
        self.system_message = cfg.system_message

        self.model = None
        self.temperature = 0.1

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if os.environ.get('LOCAL_RANK', '0') == '0':
            print("Tokenizer loaded", flush=True)

        self.llama_eot_token = None
        if self.model_family == "llama":
            self.llama_eot_token = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

        self.adapter_ids = [get_adapter_path(aid) for aid in (adapter_ids or [])]
        self.opening_message = opening_message

    @classmethod
    def from_adapter(cls, adapter_id: str, opening_message: Optional[Message] = None):
        model_id, adapter_ids = get_adapter_chain(adapter_id)
        return cls(model_id, adapter_ids, opening_message)

    def get_config(self) -> dict:
        return {
            "model_path": str(self.model_path),
            "adapter_ids": [str(adapter_id) for adapter_id in self.adapter_ids],
        }

    def messages_to_prompt(self, messages: List[Message], placeholder: bool = False, no_template: bool = False, tools: Optional[list] = None) -> str:
        if self.opening_message and not no_template:
            messages = [self.opening_message] + messages
        if no_template:
            return " ".join([m.content for m in messages])
        if self.model_family == "llama":
            return self.llama_messages_to_prompt(messages, placeholder=placeholder, tools=tools)
        elif self.model_family == "qwen":
            return self.qwen_messages_to_prompt(messages, placeholder=placeholder, tools=tools)
        raise ValueError(f"Unknown model family: {self.model_family}")

    def qwen_messages_to_prompt(self, messages: List[Message], placeholder: bool = False, tools: Optional[list] = None) -> str:
        new_messages = []
        for i, msg in enumerate(messages):
            if msg.role not in {Role.SYSTEM, Role.USER}:
                raise ValueError(f"Wrong message role {msg.role}.")
            content = QUESTION_PLACEHOLDER if (placeholder and i == len(messages) - 1) else msg.content
            new_messages.append({"role": msg.role.value, "content": content})
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if tools:
            kwargs["tools"] = tools
        return self.tokenizer.apply_chat_template(new_messages, **kwargs)

    def llama_messages_to_prompt(self, messages: List[Message], placeholder: bool = False, tools: Optional[list] = None) -> str:
        new_messages = []
        for i, msg in enumerate(messages):
            if msg.role not in {Role.SYSTEM, Role.USER}:
                raise ValueError(f"Wrong message role {msg.role}.")
            content = QUESTION_PLACEHOLDER if (placeholder and i == len(messages) - 1) else msg.content
            new_messages.append({"role": msg.role.value, "content": content})
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if tools:
            kwargs["tools"] = tools
        return self.tokenizer.apply_chat_template(new_messages, **kwargs)

    def tokenize(self, seq: str) -> torch.Tensor:
        return self.tokenizer.encode(seq, add_special_tokens=False, return_tensors="pt")

    def add_bos(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.tokenizer.bos_token_id is not None:
            if tokens[0, 0] == self.tokenizer.bos_token_id:
                return tokens  # already has BOS
            bos = torch.tensor([[self.tokenizer.bos_token_id]])
            return torch.cat([bos, tokens], dim=1)
        return tokens

    def add_eos(self, tokens: torch.Tensor) -> torch.Tensor:
        eos_id = self.llama_eot_token if self.model_family == "llama" else self.tokenizer.eos_token_id
        eos = torch.tensor([[eos_id]])
        return torch.cat([tokens, eos], dim=1)

    def decode(self, tokens: torch.Tensor) -> str:
        return self.tokenizer.batch_decode(tokens)[0]

    def get_terminators(self) -> List[int]:
        terminators = [self.tokenizer.eos_token_id]
        if self.model_family == "llama" and self.llama_eot_token:
            terminators.append(self.llama_eot_token)
            eom_id = self.tokenizer.convert_tokens_to_ids("<|eom_id|>")
            if eom_id is not None:
                terminators.append(eom_id)
        return terminators

    def load_model(
        self, training: bool = False, deepspeed: bool = False, device_map=None
    ):
        torch_dtype = torch.bfloat16
        if device_map is None:
            device_map = None if training else "auto"

        t0 = time.perf_counter()
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=device_map if not deepspeed else None,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        print(f"Time to load the model: {time.perf_counter() - t0:.02f} sec", flush=True)

        if not self.adapter_ids:
            self.model = base_model
            return self.model

        assert len(self.adapter_ids) == 1, "Only one adapter is supported"
        for adapter_id in self.adapter_ids:
            model = base_model
            print(f"Load and merge adapter {adapter_id}")
            model = PeftModel.from_pretrained(
                model=model, model_id=adapter_id, adapter_name="lora", is_trainable=False,
            )
            model = model.merge_and_unload()
            print(f"Adapter {adapter_id} loaded and merged")
        self.model = model
        return self.model

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        max_new_tokens: int = 2000,
        do_sample: bool = True,
    ) -> tuple[str, bool]:
        input_ids = input_ids.to(DEVICE)
        t0 = time.perf_counter()
        terminators = self.get_terminators()
        tokens = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature or self.temperature,
            do_sample=do_sample,
            eos_token_id=terminators,
        )
        truncated = bool(tokens[0, -1] not in terminators)
        prompt_length = input_ids.size(1)
        output_tokens = tokens[:, prompt_length:]
        t = time.perf_counter() - t0
        print(f"Generated {output_tokens.size(1)} tokens, time: {t:.02f} sec total, {output_tokens.size(1)/t:.02f} tokens/sec")
        output = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        return output, truncated

    def call(
        self,
        messages: Optional[List[Message]] = None,
        temperature: Optional[float] = None,
        max_new_tokens: int = 2000,
        merge_messages_by_role: bool = True,
    ) -> tuple[str, bool]:
        if merge_messages_by_role:
            messages = merge_messages(messages)
        prompt = self.messages_to_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        content, truncated = self.generate(
            **inputs, temperature=temperature, max_new_tokens=max_new_tokens,
        )
        return content, truncated


def get_adapter_chain(adapter_id: str) -> Tuple[str, List[str]]:
    """Get the model ID and adapter chain for a given adapter."""
    adapter_path = Path(get_adapter_path(adapter_id))

    base_model_config_file = adapter_path / "base_model_config.json"
    if os.path.exists(base_model_config_file):
        with open(base_model_config_file, 'r') as f:
            base_model_config = json.load(f)
        model_id = base_model_config["model_path"]
        adapter_ids = base_model_config["adapter_ids"] + [adapter_path]
        return model_id, adapter_ids

    warnings.warn(f"Adapter {adapter_id} does not have base_model_config.json", stacklevel=2)
    adapter_config_file = adapter_path / "adapter_config.json"
    with open(adapter_config_file, 'r') as f:
        adapter_config = json.load(f)
    model_id = adapter_config["base_model_name_or_path"]
    adapter_ids = [adapter_path]
    return model_id, adapter_ids
