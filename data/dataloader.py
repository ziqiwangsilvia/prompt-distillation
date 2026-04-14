from functools import partial
from itertools import cycle
from typing import List, Optional

from torch.utils.data import DataLoader, Sampler

from data.dataset import StudentTeacherDataset, TeacherDataset
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


def build_dataloaders(base_llm: LLM, data, hp, teacher_llm: LLM = None):
    """Build all train/val dataloaders from file lists and hyperparams.

    Returns (logit_train_ds, token_train_ds,
             logit_loader, token_loader, logit_val_loader, token_val_loader)
    """
    train_files, val_files = data

    # Datasets
    logit_train_ds = _init_logit_dataset(base_llm, train_files, hp, teacher_llm=teacher_llm) if hp.logit_loss_weight else None
    token_train_ds = _init_token_dataset(base_llm, train_files, hp) if hp.token_loss_weight else None
    logit_val_ds = StudentTeacherDataset(base_llm, val_files, datapath=hp.datapath, teacher_llm=teacher_llm) if hp.logit_loss_weight and val_files else None
    token_val_ds = TeacherDataset(base_llm, val_files, datapath=hp.datapath) if hp.token_loss_weight and val_files else None

    # Train loaders
    logit_loader = _make_loader(
        logit_train_ds, hp.logit_loss_micro_batch_size,
        partial(logit_train_ds.collate_fn, padding_value=PADDING_VALUE, llm=base_llm),
        shuffle=True,
    ) if logit_train_ds else None

    token_loader = _make_loader(
        token_train_ds, hp.token_loss_micro_batch_size,
        partial(token_train_ds.collate_fn, padding_value=PADDING_VALUE, llm=base_llm, max_total_length=hp.max_total_length),
        sampler=InfiniteSampler(len(token_train_ds)),
    ) if token_train_ds else None

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


def _init_logit_dataset(base_llm, filenames, hp, teacher_llm=None):
    ds = StudentTeacherDataset(base_llm, filenames, verbose=hp.verbose, datapath=hp.datapath, max_length=hp.max_length, teacher_llm=teacher_llm)
    if hp.logit_loss_weight and len(ds) == 0:
        raise ValueError("No logit training data found.")
    return ds


def _init_token_dataset(base_llm, filenames, hp):
    return TeacherDataset(base_llm, filenames, verbose=hp.verbose, datapath=hp.datapath, max_length=hp.max_length, distractor_dataset=hp.distractor_dataset)
