from datetime import datetime
import git
import glob
import importlib
import json
import os
from pathlib import Path
import pickle
import random
import re
import string
import sys
import traceback
from types import SimpleNamespace, FunctionType
from typing import Any, Dict
import warnings
from xml.sax.saxutils import escape as sx_escape

import torch
from vllm import SamplingParams

from . import BASE_PATH, ADAPTER_PATH


def generate_extra_body(base: str) -> Dict[str, Any]:
    extra_body = {
        "top_k": 50,
        "include_stop_str_in_output": True,
        "skip_special_tokens": False,
    }
    if 'llama3' in base.lower():
        extra_body["stop_token_ids"] = [128009]
    return extra_body


def generate_sampling_params(max_total_tokens: int, temperature: float) -> SamplingParams:
    return SamplingParams(
        include_stop_str_in_output=True,
        top_k=50,
        skip_special_tokens=False,
        max_tokens=max_total_tokens,
        temperature=temperature,
    )


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def num_parameters(module: torch.nn.Module, requires_grad: bool = None) -> int:
    """Count the number of parameters in a module, optionally filtering by requires_grad."""
    total = 0
    for p in module.parameters():
        if requires_grad is None or p.requires_grad == requires_grad:
            total += p.numel()
    return total


def find_runs(path: os.PathLike, pattern: str) -> list[Path]:
    """Find runs/folders matching a pattern under a given path."""
    full_pattern = os.path.join(path, f'*/*{pattern}')
    matching_folders = glob.glob(full_pattern, recursive=True)
    return matching_folders


class DualOutput:
    """Write output to both terminal and a log file."""
    def __init__(self, filename, mode='a'):
        self.terminal = sys.stdout
        self.log = open(filename, mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # This flush method is needed for the file and terminal to handle the buffer
        self.terminal.flush()
        self.log.flush()


def get_adapter_path(adapter_id: str) -> str:
    """Resolve an adapter ID to a full path if needed."""
    if not adapter_id or os.path.exists(adapter_id):
        return adapter_id

    matching_folders = find_runs(BASE_PATH / "checkpoints", adapter_id)
    matching_folders.extend(find_runs(ADAPTER_PATH, adapter_id))

    if len(matching_folders) > 1:
        raise ValueError(f"Multiple adapters found: {matching_folders}")

    elif len(matching_folders) == 0:
        raise ValueError(f"Adapter not found: {adapter_id}")

    else:
        print(f"Adapter found: {matching_folders[0]}")
        return matching_folders[0]


def remove_empty(lst: list) -> list:
    """Remove empty items from a list."""
    return [item for item in lst if item]


def dict_to_simplenamespace(d: dict) -> SimpleNamespace:
    """Recursively convert dict (and sub-dicts/lists) to SimpleNamespace."""
    if isinstance(d, dict):
        for key, value in d.items():
            d[key] = dict_to_simplenamespace(value)
        return SimpleNamespace(**d)
    if isinstance(d, list):
        return [dict_to_simplenamespace(item) for item in d]
    return d


def random_id(length: int) -> str:
    """Generate a random alphanumeric ID of a given length."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


_SQUADSHIFTS_URLS = {
    "new_wiki": "https://raw.githubusercontent.com/modestyachts/squadshifts-website/master/datasets/new_wiki_v1.0.json",
    "nyt": "https://raw.githubusercontent.com/modestyachts/squadshifts-website/master/datasets/nyt_v1.0.json",
    "reddit": "https://raw.githubusercontent.com/modestyachts/squadshifts-website/master/datasets/reddit_v1.0.json",
    "amazon": "https://raw.githubusercontent.com/modestyachts/squadshifts-website/master/datasets/amazon_reviews_v1.0.json",
}


def load_squadshifts(subset: str):
    """Load a SquadShifts subset directly from source JSON, bypassing the deprecated HF loading script."""
    import urllib.request
    from datasets import Dataset

    url = _SQUADSHIFTS_URLS[subset]
    with urllib.request.urlopen(url) as resp:
        squad = json.loads(resp.read().decode("utf-8"))

    rows = []
    for article in squad["data"]:
        title = article.get("title", "").strip()
        for paragraph in article["paragraphs"]:
            context = paragraph["context"].strip()
            for qa in paragraph["qas"]:
                rows.append({
                    "id": qa["id"],
                    "title": title,
                    "context": context,
                    "question": qa["question"].strip(),
                    "answers": {
                        "answer_start": [a["answer_start"] for a in qa["answers"]],
                        "text": [a["text"].strip() for a in qa["answers"]],
                    },
                })
    return Dataset.from_list(rows)
