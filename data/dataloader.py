from functools import partial
from itertools import cycle
from typing import List, Optional

from torch.utils.data import DataLoader, Sampler

from data.dataset import StudentTeacherDataset, StudentDataset
from models.llm import LLM
from training import PADDING_VALUE
from training.utils import InfiniteSampler


def _make_loader(dataset, batch_size, collate_fn, sampler=None, shuffle=False):
    if dataset is None or batch_size == 0:
        return None
    return DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn,
        sampler=sampler, shuffle=shuffle if sampler is None else False,
    )


def build_dataloaders(base_llm: LLM, data, hp, teacher_llm: LLM = None, tools: list = None):
    """Build all train/val dataloaders from file lists and hyperparams.

    Returns (logit_train_ds, token_train_ds,
             logit_loader, token_loader, logit_val_loader, token_val_loader)
    """
    train_files, val_files = data

    # Datasets
    needs_logit_data = hp.distillation_type == "on_policy" or hp.logit_loss_weight
    logit_train_ds = _init_logit_dataset(base_llm, train_files, hp, teacher_llm=teacher_llm, tools=tools) if needs_logit_data else None
    token_train_ds = _init_token_dataset(base_llm, train_files, hp, tools=tools) if hp.token_loss_weight and not needs_logit_data else None
    max_samples = 7 if hp.test_mode else 0
    logit_val_ds = StudentTeacherDataset(base_llm, val_files, datapath=hp.datapath, teacher_llm=teacher_llm, tools=tools, use_tool_token=hp.use_tool_token, max_samples=max_samples, multi_turn=hp.multi_turn) if needs_logit_data and val_files else None
    token_val_ds = StudentDataset(base_llm, val_files, datapath=hp.datapath, tools=tools, use_tool_token=hp.use_tool_token, max_samples=max_samples, multi_turn=hp.multi_turn) if hp.token_loss_weight and not needs_logit_data and val_files else None

    # Train loaders — when distillation is active, token loss reuses the logit dataset
    logit_collate = partial(logit_train_ds.collate_fn, padding_value=PADDING_VALUE, llm=base_llm) if logit_train_ds else None

    logit_loader = _make_loader(
        logit_train_ds, hp.logit_loss_micro_batch_size, logit_collate, shuffle=True,
    ) if logit_train_ds else None

    if token_train_ds:
        token_loader = _make_loader(
            token_train_ds, hp.token_loss_micro_batch_size,
            partial(token_train_ds.collate_fn, padding_value=PADDING_VALUE, llm=base_llm, max_total_length=hp.max_total_length),
            sampler=InfiniteSampler(len(token_train_ds)),
        )
    elif hp.token_loss_weight and logit_train_ds:
        token_loader = _make_loader(
            logit_train_ds, hp.token_loss_micro_batch_size, logit_collate,
            sampler=InfiniteSampler(len(logit_train_ds)),
        )
    else:
        token_loader = None

    # Val loaders
    logit_val_loader = _make_loader(
        logit_val_ds, hp.logit_loss_micro_batch_size,
        partial(StudentTeacherDataset.collate_fn, padding_value=PADDING_VALUE, llm=base_llm),
    ) if logit_val_ds else None

    token_val_loader = _make_loader(
        token_val_ds, hp.token_loss_micro_batch_size,
        partial(token_val_ds.collate_fn, padding_value=PADDING_VALUE, llm=base_llm),
    ) if token_val_ds else None

    return logit_train_ds, token_train_ds, logit_loader, token_loader, logit_val_loader, token_val_loader


def _init_logit_dataset(base_llm, filenames, hp, teacher_llm=None, tools=None):
    max_samples = 7 if hp.test_mode else 0
    ds = StudentTeacherDataset(base_llm, filenames, verbose=hp.verbose, datapath=hp.datapath, max_length=hp.max_length, teacher_llm=teacher_llm, tools=tools, use_tool_token=hp.use_tool_token, max_samples=max_samples, multi_turn=hp.multi_turn)
    if len(ds) == 0:
        raise ValueError("No logit training data found.")
    return ds


def _init_token_dataset(base_llm, filenames, hp, tools=None):
    max_samples = 7 if hp.test_mode else 0
    return StudentDataset(base_llm, filenames, verbose=hp.verbose, datapath=hp.datapath, max_length=hp.max_length, distractor_dataset=hp.distractor_dataset, tools=tools, use_tool_token=hp.use_tool_token, max_samples=max_samples, multi_turn=hp.multi_turn)
