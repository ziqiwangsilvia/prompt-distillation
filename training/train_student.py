import json
import math
import os
import pprint
import sys
import time
from contextlib import nullcontext
from functools import partial
from itertools import cycle
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import ProjectConfiguration
from datasets import load_dataset
from peft import get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, PreTrainedModel
import wandb

from models.llm import LLM
from models.utils import DualOutput, num_parameters
from training import PADDING_VALUE
from training.metrics import Aggregator
from training.student_teacher_dataset import (
    IGNORE_INDEX,
    StudentTeacherDataset,
    TeacherDataset,
)
from training.utils import (
    InfiniteSampler,
    generate_answers,
    save_with_base_model_config,
    save_with_deepspeed,
    print_token_tensor,
)


def _get_ds_plugin(ds_config_path: str) -> DeepSpeedPlugin:
    with open(ds_config_path, "r", encoding="utf-8") as f:
        return DeepSpeedPlugin(hf_ds_config=json.load(f))


def _model_is_bf16(model: PreTrainedModel) -> bool:
    return all(p.dtype == torch.bfloat16 for p in model.parameters())


def _init_logit_train_dataset(
    llm: LLM, filenames: List[str], hp: SimpleNamespace
) -> StudentTeacherDataset:
    ds = StudentTeacherDataset(
        llm,
        filenames,
        verbose=hp.verbose,
        datapath=hp.datapath,
        max_length=hp.max_length,
    )
    if hp.logit_loss_weight and len(ds) == 0:
        raise ValueError("No logit training data found.")
    return ds


def _init_token_train_dataset(
    llm: LLM, filenames: List[str], hp: SimpleNamespace
) -> TeacherDataset:
    return TeacherDataset(
        llm,
        filenames,
        verbose=hp.verbose,
        datapath=hp.datapath,
        max_length=hp.max_length,
        distractor_dataset=hp.distractor_dataset,
    )


def _make_loader(
    dataset,
    batch_size: int,
    collate_fn: Callable,
    sampler=None,
    shuffle: bool = False,
):
    if dataset is None or batch_size == 0:
        return None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
    )


def _init_run(hp: SimpleNamespace) -> Accelerator:
    hp.run_project_dir = hp.project_path / hp.run_name
    hp.run_project_dir.mkdir(parents=True, exist_ok=True)

    project_cfg = ProjectConfiguration(automatic_checkpoint_naming=True)

    # Deepspeed plugin(s)
    if hp.deepspeed_path and hp.teacher in {"student", "student_base"}:
        ds_plugin: DeepSpeedPlugin | Dict[str, DeepSpeedPlugin] = _get_ds_plugin(
            hp.deepspeed_path
        )
    elif hp.deepspeed_path:
        ds_plugin = {
            "student": DeepSpeedPlugin(hf_ds_config=hp.deepspeed_path),
            "teacher": DeepSpeedPlugin(hf_ds_config=hp.deepspeed_path_teacher),
        }
    else:
        ds_plugin = None

    accelerator = Accelerator(
        mixed_precision=hp.mixed_precision,
        project_dir=hp.run_project_dir,
        project_config=project_cfg,
        deepspeed_plugin=ds_plugin,
    )

    hp.devices = (
        accelerator.state.num_processes if hp.deepspeed_path else 1
    )
    hp.verbose = accelerator.is_main_process

    # Tee stdout / stderr
    sys.stdout = DualOutput(hp.run_project_dir / f"output_{accelerator.process_index}.log")

    if hp.verbose:
        print("Run name:", hp.run_name)
        pprint.pprint(hp)

    # WandB
    hp.log_to_wandb = hp.use_wandb and accelerator.is_main_process
    if hp.log_to_wandb:
        wandb.init(
            project=hp.project_path.name,
            name=hp.run_name,
            group=hp.group_name,
            allow_val_change=True,
            config=vars(hp),
        )

    # Convenience flags
    hp.generate = bool(hp.generation_interval)
    hp.checkpoint = bool(hp.checkpoint_interval or hp.checkpoint_interval_seconds)
    return accelerator


def _init_models(
    base_llm: LLM, accelerator: Accelerator, hp: SimpleNamespace
) -> Tuple[PreTrainedModel, Union[str, PreTrainedModel], torch.optim.Optimizer]:
    if hp.verbose:
        print("Loading student model", flush=True)

    student: PreTrainedModel = base_llm.load_model(
        training=True, deepspeed=bool(hp.deepspeed_path)
    )

    if hp.verbose:
        print(
            f"Trainable params: {num_parameters(student, True):,},  "
            f"Frozen params: {num_parameters(student, False):,}",
            flush=True
        )
    
    if hp.verbose:
        print("Preparing student LoRA", flush=True)

    # PEFT / LoRA
    student = get_peft_model(student, hp.peft_config)
    if hp.mixed_precision == "bf16" and not hp.deepspeed_path:
        student = student.to(torch.bfloat16)

    if hp.verbose:
        print("PEFT student built.")
        print(student)
        print(
            f"Trainable params: {num_parameters(student, True):,},  "
            f"Frozen params: {num_parameters(student, False):,}",
            flush=True
        )

    # Teacher
    if hp.teacher in {"student", "student_base"}:
        teacher: Union[str, PreTrainedModel] = hp.teacher
        teacher_llm = None
    else:
        teacher_llm = LLM(hp.teacher, opening_message=hp.opening_message)

    optimizer = torch.optim.AdamW(
        student.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay
    )

    accelerator.register_for_checkpointing(student)
    student, optimizer = accelerator.prepare(student, optimizer)

    if teacher_llm:
        accelerator.state.select_deepspeed_plugin("teacher")
        teacher = teacher_llm.load_model(training=False, deepspeed=True)
        teacher = accelerator.prepare(teacher)
        teacher.eval()
    else:
        teacher = teacher_llm or teacher  # keep type

    if hp.verbose:
        print("Student bf16:", _model_is_bf16(student))

    return student, teacher, optimizer


def train(
    project_path: Path,
    base_llm: LLM,
    data: Tuple[List[str], List[str]],
    hp: SimpleNamespace,
) -> None:
    accelerator = _init_run(hp)

    # Data
    logit_train_files, logit_val_files = data
    token_train_files, token_val_files = data

    logit_train_ds = _init_logit_train_dataset(base_llm, logit_train_files, hp) if hp.logit_loss_weight else None
    token_train_ds = _init_token_train_dataset(base_llm, token_train_files, hp) if hp.token_loss_weight else None
    logit_val_ds = StudentTeacherDataset(base_llm, logit_val_files, datapath=hp.datapath) if hp.logit_loss_weight and logit_val_files else None
    token_val_ds = TeacherDataset(base_llm, token_val_files, datapath=hp.datapath) if hp.token_loss_weight and token_val_files else None

    logit_loader = _make_loader(
        logit_train_ds,
        hp.logit_loss_micro_batch_size,
        partial(logit_train_ds.collate_fn, padding_value=PADDING_VALUE, llm=base_llm),
        shuffle=True,
    ) if hp.logit_loss_weight else None

    token_loader = _make_loader(
        token_train_ds,
        hp.token_loss_micro_batch_size,
        partial(
            token_train_ds.collate_fn,
            padding_value=PADDING_VALUE,
            llm=base_llm,
            max_total_length=hp.max_total_length,
        ),
        sampler=InfiniteSampler(len(token_train_ds))
    ) if hp.token_loss_weight else None

    # Validation loaders
    logit_val_loader = (
        _make_loader(
            logit_val_ds,
            hp.logit_loss_micro_batch_size,
            partial(
                StudentTeacherDataset.collate_fn,
                padding_value=PADDING_VALUE,
                llm=base_llm,
            ),
        ) if logit_val_ds else None
    )
    token_val_loader = (
        _make_loader(
            token_val_ds,
            hp.token_loss_micro_batch_size,
            partial(
                token_val_ds.collate_fn,
                padding_value=PADDING_VALUE,
                llm=base_llm,
            ),
        ) if token_val_ds else None
    )

    if hp.verbose:
        token_count = len(token_train_ds) if token_train_ds else 0
        logit_count = len(logit_train_ds) if logit_train_ds else 0
        print(f"Training data: token loss: {token_count} examples, logit loss: {logit_count} examples\n\n")

    # Prepare with accelerator
    logit_loader, token_loader, logit_val_loader, token_val_loader = accelerator.prepare(
        logit_loader, token_loader, logit_val_loader, token_val_loader
    )
    if logit_loader:
        logit_loader = cycle(logit_loader)

    # Models & optimiser 
    student, teacher, optimizer = _init_models(base_llm, accelerator, hp)
    
    ##### Set up training #####
    if hp.logit_loss_weight:
        n_batches = math.ceil(len(logit_train_ds) / hp.n_logit_micro_batches_per_batch / hp.logit_loss_micro_batch_size / hp.devices)
    else:
        n_batches = math.ceil(len(token_train_ds) / hp.n_token_micro_batches_per_batch / hp.token_loss_micro_batch_size / hp.devices)
    max_steps = hp.n_epochs * n_batches
    if hp.verbose:
        print(f"Training for {hp.n_epochs} epochs, {max_steps} iterations", flush=True)

    if hp.tulu:
        tulu_dataset = load_dataset("allenai/tulu-3-sft-mixture", split="train")
        tulu_loader = DataLoader(
            tulu_dataset,
            batch_size=hp.tulu_batch_size*2, # We sample extra to discard invalid ones
            shuffle=True,
            collate_fn=partial(tulu_collate_fn, padding_value=PADDING_VALUE, llm=base_llm, max_length=hp.max_total_length,
                               system_msg=hp.opening_message,
                               lesson_ix=len(logit_train_files),
                               logit_collate_fn=partial(logit_train_ds.collate_fn,
                                                        padding_value=PADDING_VALUE,
                                                        llm=base_llm),
                               use_batch_size=hp.tulu_batch_size,
                               ),
        )
        tulu_loader = accelerator.prepare(tulu_loader)
        iter_tulu_loader = cycle(iter(tulu_loader))
    else:
        iter_tulu_loader = None

    warmup_steps = 0
    if hasattr(hp, 'warmup_steps') and hp.warmup_steps is not None:
        hp.warmup_steps = hp.warmup_steps
    elif hasattr(hp, 'warmup_ratio'):
        hp.warmup_steps = int(hp.warmup_ratio * max_steps)
    if hp.verbose:
        print(f"Learning rate warmup: {hp.warmup_steps} steps", flush=True)

    train_t0 = time.perf_counter()
    student.train()

    iter_logit_loader = iter(logit_loader) if logit_loader is not None else None
    iter_token_loader = iter(token_loader) if token_loader is not None else None

    if hp.eval_interval < 0:
        hp.eval_interval = n_batches
        print(f"Setting evaluation interval to {hp.eval_interval}", flush=True)
    if hp.generate and hp.generation_interval < 0:
        hp.generation_interval = n_batches
        print(f"Setting generation interval to {hp.generation_interval}", flush=True)

    last_checkpoint_time = time.time()
    n_saves = 0
    all_losses = []
    step_times = []

    # +1 for validation at the end of training
    for step in range(max_steps + 1):
        # We start with validation to log the untrained model's performance
        if hp.validate and step % hp.eval_interval == 0:
            t0 = time.perf_counter()
            metrics_total, metrics_by_group = validate(student, teacher, token_val_loader, logit_val_loader, accelerator,
                                                       hp.closed_book_token_loss, hp, base_llm)

            t1 = time.perf_counter() - t0
            if hp.verbose:
                print("Validation results:", metrics_total, flush=True)

            log_to_wandb(accelerator, metrics_total, metrics_by_group, step, hp)

        if hp.generate and step % hp.generation_interval == 0:
            # Select 1 random sample from logit_train_ds
            if hp.logit_loss_weight:
                ix = torch.randint(high=len(logit_train_ds), size=(1,)).item()
                generation_samples = [logit_train_ds[ix]]
            else:
                # token loss
                ix = torch.randint(high=len(token_train_ds), size=(1,)).item()
                generation_samples = [token_train_ds[ix]]
            generate_answers(base_llm, generation_samples, accelerator)

        if step == max_steps:
            print("Finishing training, saving model")
            break

        is_logging = step % hp.log_interval == 0
        train_metrics = {}

        update_lr(step, optimizer, train_metrics, hp, is_logging, max_steps)

        step_t0 = time.perf_counter()
        # Process logit loss
        if hp.logit_loss_weight:
            for _ in range(hp.n_logit_micro_batches_per_batch):
                batch = next(iter_logit_loader)
                if iter_tulu_loader:
                    batch_tulu = next(iter_tulu_loader)
                    batch = merge_with_tulu_batch(batch, batch_tulu, padding_value=PADDING_VALUE)
                logit_loss = compute_logit_loss(batch, student, teacher, temperature=hp.train_temperature, reverse_kl=hp.reverse_kl, base_llm=base_llm)
                logit_loss = logit_loss.mean()
                loss = hp.logit_loss_weight * logit_loss / hp.n_logit_micro_batches_per_batch
                accelerator.backward(loss)

            if is_logging:
                train_metrics['logit_loss'] = logit_loss.item()
            
        # Process token loss
        if hp.token_loss_weight:
            for _ in range(hp.n_token_micro_batches_per_batch):
                batch = next(iter_token_loader)
                token_loss = compute_token_loss(batch, student, reduction="batch",
                                                closed_book_token_loss=hp.closed_book_token_loss,
                                                base_llm=base_llm)
                loss = hp.token_loss_weight * token_loss / hp.n_token_micro_batches_per_batch
                accelerator.backward(loss)

            if is_logging and hp.n_token_micro_batches_per_batch > 0:
                train_metrics['token_loss'] = token_loss.item()

        if hp.max_grad_norm:
            accelerator.clip_grad_norm_(student.parameters(), hp.max_grad_norm)
       
        torch.cuda.empty_cache()
        optimizer.step()
        optimizer.zero_grad()
        log_step(step, is_logging, accelerator, train_metrics, hp, step_t0, max_steps, all_losses=all_losses, step_times=step_times)

        if hp.save_during_training:
            current_time = time.time()
            if hp.checkpoint_interval and (step + 1) % hp.checkpoint_interval == 0:
                if hp.deepspeed_path:
                    save_with_deepspeed(student, accelerator, base_llm, hp.run_project_dir)
                else:
                    save_with_base_model_config(student, base_llm, hp.run_project_dir)

            if hp.checkpoint_interval_seconds and current_time - last_checkpoint_time >= hp.checkpoint_interval_seconds:
                n_saves += 1
                save_idx = n_saves * hp.checkpoint_interval_seconds
                checkpoint_subfolder = Path(os.path.join(hp.run_project_dir, f"checkpoint_time_{save_idx}"))
                if hp.deepspeed_path:
                    print("Saving deepspeed model")
                    save_with_deepspeed(student, accelerator, base_llm, checkpoint_subfolder)
                else:
                    save_with_base_model_config(student, base_llm, checkpoint_subfolder)
                last_checkpoint_time = current_time

    if hp.save:
        if hp.deepspeed_path:
            save_with_deepspeed(student, accelerator, base_llm, hp.run_project_dir)
        else:
            save_with_base_model_config(student, base_llm, hp.run_project_dir)

    print(f"Training time: {(time.perf_counter()-train_t0):.2f}s", flush=True)


def compute_token_loss(
    batch: Dict[str, torch.Tensor],
    model: PreTrainedModel,
    reduction: Literal["batch", "sample"] = "batch",  # "batch" for training, "sample" for validation
    closed_book_token_loss: bool = False,
    base_llm: Optional[LLM] = None,
) -> torch.Tensor:

    assert reduction in ["batch", "sample"]

    if closed_book_token_loss:
        inputs = batch['closed_book_seqs'][..., :-1]  # (batch_size, seq_length)
        labels = batch['closed_book_labels'][..., 1:]  # (batch_size, seq_length)
    else:
        inputs = batch['open_book_seqs'][..., :-1]  # (batch_size, seq_length)
        labels = batch['open_book_labels'][..., 1:]  # (batch_size, seq_length)

    batch_size, seq_length = inputs.shape
    output = model.forward(inputs)
    output_logits = output.logits
    token_loss = F.cross_entropy(
        output_logits.flatten(0, 1),
        labels.flatten(0, 1),
        ignore_index=IGNORE_INDEX,
        reduction="mean" if reduction == "batch" else "none"
    )
    if reduction == "sample":
        token_loss = token_loss.reshape(batch_size, seq_length).mean(-1)

    return token_loss


def compute_logit_loss(
    batch: Dict[str, torch.Tensor],
    student: PreTrainedModel,
    teacher: Union[str, PreTrainedModel],
    temperature: float,
    reverse_kl: bool = False,
    base_llm: Optional[LLM] = None,
) -> torch.Tensor:
    """
    Compute a KL-divergence-based distillation loss from teacher to student.
    """
    student_inputs = batch['student_seqs'][..., :-1]  # (batch_size, seq_length)
    student_labels = batch['student_labels'][..., 1:]  # (batch_size, seq_length)
    # The mask determines which outputs should be used for loss calculation
    student_masks = student_labels != IGNORE_INDEX

    batch_size, seq_length = student_inputs.shape

    if teacher == "student_base":
        teacher_context = student.disable_adapter()
        teacher = student
    elif teacher == "student":
        teacher_context = nullcontext()
        teacher = student
    else:
        teacher_context = nullcontext()

    with torch.no_grad(), teacher_context:
        teacher.eval()
        teacher_inputs = batch['teacher_seqs'][..., :-1]
        # The mask determines which outputs should be used for loss calculation
        teacher_masks = batch['teacher_masks'][..., 1:]
    
        teacher_output = teacher.forward(teacher_inputs)
        teacher_logits = teacher_output.logits[teacher_masks].detach()

        t_logits = teacher_logits / temperature  # (n_tokens, vocab_size)

        t_log_probs = F.log_softmax(t_logits, dim=-1).detach()

    student_output = student.forward(student_inputs)
    student_logits = student_output.logits
    student_logits = student_logits[student_masks]

    s_logits = student_logits / temperature  # (n_tokens, vocab_size)
    s_log_probs = F.log_softmax(s_logits, dim=-1)

    if reverse_kl:
        logit_loss = F.kl_div(
            t_log_probs, s_log_probs, log_target=True,
            reduction="none",
        )
    else:
        logit_loss = F.kl_div(
            s_log_probs, t_log_probs, log_target=True,
            reduction="none",
        )

    logit_loss_t = logit_loss.sum(-1)  # (n_tokens,)

    logit_loss_mx = torch.zeros(batch_size, seq_length, device=student_inputs.device, dtype=logit_loss_t.dtype)
    logit_loss_mx[student_masks] = logit_loss_t
    logit_loss = logit_loss_mx.sum(-1) / student_masks.sum(-1)  # (batch_size,)

    return logit_loss


def validate(
    student: PreTrainedModel,
    teacher: Union[str, PreTrainedModel, None],
    token_loader: Optional[DataLoader],
    logit_loader: Optional[DataLoader],
    accelerator: Accelerator,
    closed_book_token_loss: bool,
    hp: SimpleNamespace,
    base_llm: LLM,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    student.eval()
    metrics_total = {}
    metrics_by_group = {}
    with torch.no_grad():
        if token_loader is not None:
            group_names = token_loader.dataset.lesson_names
            aggregator = Aggregator(group_names, accelerator.device)
            for _, batch in enumerate(token_loader, start=1):
                token_loss = compute_token_loss(batch, student, reduction="sample",
                                                closed_book_token_loss=closed_book_token_loss, base_llm=base_llm)
                metrics = {
                    "val_token_loss": token_loss,
                }
                aggregator.add_batch(batch["lesson_ixs"], metrics, accelerator)
            token_metrics_total, token_metrics_by_group = aggregator.get_average()
            metrics_total.update(token_metrics_total)
            metrics_by_group.update(token_metrics_by_group)

        if logit_loader is not None:
            group_names = logit_loader.dataset.lesson_names
            aggregator = Aggregator(group_names, accelerator.device)
            for _, batch in enumerate(logit_loader, start=1):
                logit_loss = compute_logit_loss(batch, student, teacher, temperature=1, reverse_kl=hp.reverse_kl,
                                                base_llm=base_llm)
                metrics = {
                    "val_logit_loss": logit_loss,
                }
                aggregator.add_batch(batch["lesson_ixs"], metrics, accelerator)
            logit_metrics_total, logit_metrics_by_group = aggregator.get_average()
            metrics_total.update(logit_metrics_total)
            metrics_by_group.update(logit_metrics_by_group)

    student.train()
    return metrics_total, metrics_by_group


def log_to_wandb(
    accelerator: Accelerator,
    metrics_total: Dict[str, Any],
    metrics_by_group: Dict[str, Any],
    step: int,
    hp: SimpleNamespace,
) -> None:
    if hp.log_to_wandb and accelerator.is_main_process:
        # Log aggregated metrics
        wandb.log(metrics_total, step=step)
        wandb.log(metrics_by_group, step=step)


def update_lr(
    step: int,
    optimizer: torch.optim.Optimizer,
    train_metrics: Dict[str, Any],
    hp: SimpleNamespace,
    is_logging: bool,
    max_steps: int,
) -> None:
    if step <= hp.warmup_steps and hp.warmup_steps:
        # linear warmup
        lr = hp.learning_rate * step / hp.warmup_steps
    else:
        if hp.decay:
            multiplier = max(0.0, float(max_steps - step) / float(max(1, max_steps - hp.warmup_steps)))
            lr = hp.learning_rate * multiplier
        else:
            lr = hp.learning_rate

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    if is_logging:
        train_metrics['lr'] = lr


def log_step(
    step: int,
    is_logging: bool,
    accelerator: Accelerator,
    train_metrics: Dict[str, Any],
    hp: SimpleNamespace,
    step_t0: float,
    max_steps: int,
    chunk: Optional[int] = None,
    n_chunks: Optional[int] = None,
    total_step: Optional[int] = None,
    total_max_steps: Optional[int] = None,
    all_losses: Optional[List[float]] = None,
    step_times: Optional[List[float]] = None,
) -> None:
    if is_logging and accelerator.is_main_process:
        t1 = time.perf_counter()
        if step_times is not None:
            step_times.append((t1 - step_t0) * 1000)
        loss_to_print = train_metrics.get('logit_loss', train_metrics.get('token_loss', None))
        if all_losses is not None:
            all_losses.append(loss_to_print)
            avg_loss = sum(all_losses)/len(all_losses)
        else:
            avg_loss = None
        loss_type = 'logit' if 'logit_loss' in train_metrics else 'token'

        print(
            (f"Total step {total_step}/{total_max_steps}, " if n_chunks else "") +
            (f"Chunk {chunk}/{n_chunks}, " if n_chunks else "") +
            f"Step {step+1}/{max_steps}: {loss_type} loss {loss_to_print:.8f}, "
            f"iter time: {(t1 - step_t0) * 1000:.2f}ms" +
            (f", avg iter time: {np.mean(step_times):.2f}ms" if step_times else "") +
            (f", Total avg loss {avg_loss:.4f} " if avg_loss else ""),
            flush=True
        )

        if hp.log_to_wandb:
            train_metrics['step_time'] = (t1 - step_t0) * 1000
            if total_step is not None:
                wandb.log(train_metrics, step=total_step)
            else:
                wandb.log(train_metrics, step=step)
