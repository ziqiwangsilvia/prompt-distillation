import json
import math
import os
import pprint
import sys
import time
from functools import partial
from itertools import cycle
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import ProjectConfiguration
from peft import get_peft_model
from torch.utils.data import DataLoader
from transformers import PreTrainedModel
import wandb

from models.llm import LLM
from models.utils import DualOutput, num_parameters
from training import PADDING_VALUE
from evaluation.metrics import Aggregator
from data.dataset import StudentTeacherDataset, TeacherDataset
from training.loss import compute_token_loss, compute_logit_loss
from training.utils import (
    InfiniteSampler,
    generate_answers,
    save_with_base_model_config,
    save_with_deepspeed,
    print_token_tensor,
)


class Trainer:
    def __init__(self, base_llm: LLM, data: Tuple[List[str], List[str]], hp: SimpleNamespace):
        self.base_llm = base_llm
        self.hp = hp
        self.accelerator = self._init_run()

        # Data
        logit_train_files, logit_val_files = data
        token_train_files, token_val_files = data

        self.logit_train_ds = self._init_logit_dataset(logit_train_files) if hp.logit_loss_weight else None
        self.token_train_ds = self._init_token_dataset(token_train_files) if hp.token_loss_weight else None
        logit_val_ds = StudentTeacherDataset(base_llm, logit_val_files, datapath=hp.datapath) if hp.logit_loss_weight and logit_val_files else None
        token_val_ds = TeacherDataset(base_llm, token_val_files, datapath=hp.datapath) if hp.token_loss_weight and token_val_files else None

        # Train loaders
        self.logit_loader = self._make_loader(
            self.logit_train_ds, hp.logit_loss_micro_batch_size,
            partial(self.logit_train_ds.collate_fn, padding_value=PADDING_VALUE, llm=base_llm),
            shuffle=True,
        ) if self.logit_train_ds else None

        self.token_loader = self._make_loader(
            self.token_train_ds, hp.token_loss_micro_batch_size,
            partial(self.token_train_ds.collate_fn, padding_value=PADDING_VALUE, llm=base_llm, max_total_length=hp.max_total_length),
            sampler=InfiniteSampler(len(self.token_train_ds)),
        ) if self.token_train_ds else None

        # Val loaders
        self.logit_val_loader = self._make_loader(
            logit_val_ds, hp.logit_loss_micro_batch_size,
            partial(StudentTeacherDataset.collate_fn, padding_value=PADDING_VALUE, llm=base_llm),
        ) if logit_val_ds else None

        self.token_val_loader = self._make_loader(
            token_val_ds, hp.token_loss_micro_batch_size,
            partial(token_val_ds.collate_fn, padding_value=PADDING_VALUE, llm=base_llm),
        ) if token_val_ds else None

        if hp.verbose:
            token_count = len(self.token_train_ds) if self.token_train_ds else 0
            logit_count = len(self.logit_train_ds) if self.logit_train_ds else 0
            print(f"Training data: token loss: {token_count} examples, logit loss: {logit_count} examples\n\n")

        # Prepare with accelerator
        self.logit_loader, self.token_loader, self.logit_val_loader, self.token_val_loader = self.accelerator.prepare(
            self.logit_loader, self.token_loader, self.logit_val_loader, self.token_val_loader
        )
        if self.logit_loader:
            self.logit_loader = cycle(self.logit_loader)

        # Models
        self.student, self.teacher, self.optimizer = self._init_models()

        # Steps
        if hp.logit_loss_weight:
            self.n_batches = math.ceil(len(self.logit_train_ds) / hp.n_logit_micro_batches_per_batch / hp.logit_loss_micro_batch_size / hp.devices)
        else:
            self.n_batches = math.ceil(len(self.token_train_ds) / hp.n_token_micro_batches_per_batch / hp.token_loss_micro_batch_size / hp.devices)
        self.max_steps = hp.n_epochs * self.n_batches

    # ── Setup ──────────────────────────────────────────────

    def _init_run(self) -> Accelerator:
        hp = self.hp
        hp.run_project_dir = hp.project_path / hp.run_name
        hp.run_project_dir.mkdir(parents=True, exist_ok=True)

        project_cfg = ProjectConfiguration(automatic_checkpoint_naming=True)

        if hp.deepspeed_path and hp.teacher in {"student", "student_base"}:
            ds_plugin = _get_ds_plugin(hp.deepspeed_path)
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

        hp.devices = accelerator.state.num_processes if hp.deepspeed_path else 1
        hp.verbose = accelerator.is_main_process

        sys.stdout = DualOutput(hp.run_project_dir / f"output_{accelerator.process_index}.log")

        if hp.verbose:
            print("Run name:", hp.run_name)
            pprint.pprint(hp)

        hp.log_to_wandb = hp.use_wandb and accelerator.is_main_process
        if hp.log_to_wandb:
            wandb.init(
                project=hp.project_path.name,
                name=hp.run_name,
                group=hp.group_name,
                allow_val_change=True,
                config=vars(hp),
            )

        hp.generate = bool(hp.generation_interval)
        hp.checkpoint = bool(hp.checkpoint_interval or hp.checkpoint_interval_seconds)
        return accelerator

    def _init_models(self) -> Tuple[PreTrainedModel, Union[str, PreTrainedModel], torch.optim.Optimizer]:
        hp = self.hp
        base_llm = self.base_llm
        accelerator = self.accelerator

        if hp.verbose:
            print("Loading student model", flush=True)

        student = base_llm.load_model(training=True, deepspeed=bool(hp.deepspeed_path))

        if hp.verbose:
            print(f"Trainable params: {num_parameters(student, True):,},  Frozen params: {num_parameters(student, False):,}", flush=True)
            print("Preparing student LoRA", flush=True)

        student = get_peft_model(student, hp.peft_config)
        if hp.mixed_precision == "bf16" and not hp.deepspeed_path:
            student = student.to(torch.bfloat16)

        if hp.verbose:
            print("PEFT student built.")
            print(student)
            print(f"Trainable params: {num_parameters(student, True):,},  Frozen params: {num_parameters(student, False):,}", flush=True)

        if hp.teacher in {"student", "student_base"}:
            teacher = hp.teacher
            teacher_llm = None
        else:
            teacher_llm = LLM(hp.teacher, opening_message=hp.opening_message)

        optimizer = torch.optim.AdamW(student.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay)

        accelerator.register_for_checkpointing(student)
        student, optimizer = accelerator.prepare(student, optimizer)

        if teacher_llm:
            accelerator.state.select_deepspeed_plugin("teacher")
            teacher = teacher_llm.load_model(training=False, deepspeed=True)
            teacher = accelerator.prepare(teacher)
            teacher.eval()
        else:
            teacher = teacher_llm or teacher

        if hp.verbose:
            print("Student bf16:", _model_is_bf16(student))

        return student, teacher, optimizer

    def _init_logit_dataset(self, filenames):
        hp = self.hp
        ds = StudentTeacherDataset(self.base_llm, filenames, verbose=hp.verbose, datapath=hp.datapath, max_length=hp.max_length)
        if hp.logit_loss_weight and len(ds) == 0:
            raise ValueError("No logit training data found.")
        return ds

    def _init_token_dataset(self, filenames):
        hp = self.hp
        return TeacherDataset(self.base_llm, filenames, verbose=hp.verbose, datapath=hp.datapath, max_length=hp.max_length, distractor_dataset=hp.distractor_dataset)

    @staticmethod
    def _make_loader(dataset, batch_size, collate_fn, sampler=None, shuffle=False):
        if dataset is None or batch_size == 0:
            return None
        return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, sampler=sampler, shuffle=shuffle if sampler is None else False)

    # ── Training ───────────────────────────────────────────

    def train(self):
        hp = self.hp
        accelerator = self.accelerator

        if hp.verbose:
            print(f"Training for {hp.n_epochs} epochs, {self.max_steps} iterations", flush=True)

        # Warmup
        if hasattr(hp, 'warmup_steps') and hp.warmup_steps is not None:
            pass
        elif hasattr(hp, 'warmup_ratio'):
            hp.warmup_steps = int(hp.warmup_ratio * self.max_steps)
        if hp.verbose:
            print(f"Learning rate warmup: {hp.warmup_steps} steps", flush=True)

        train_t0 = time.perf_counter()
        self.student.train()

        iter_logit = iter(self.logit_loader) if self.logit_loader is not None else None
        iter_token = iter(self.token_loader) if self.token_loader is not None else None

        if hp.eval_interval < 0:
            hp.eval_interval = self.n_batches
        if hp.generate and hp.generation_interval < 0:
            hp.generation_interval = self.n_batches

        last_checkpoint_time = time.time()
        n_saves = 0
        all_losses = []
        step_times = []

        for step in range(self.max_steps + 1):
            # Validate
            if hp.validate and step % hp.eval_interval == 0:
                t0 = time.perf_counter()
                metrics_total, metrics_by_group = self.validate()
                if hp.verbose:
                    print("Validation results:", metrics_total, flush=True)
                self._log_to_wandb(metrics_total, metrics_by_group, step)

            # Generate
            if hp.generate and step % hp.generation_interval == 0:
                if hp.logit_loss_weight and self.logit_train_ds:
                    ix = torch.randint(high=len(self.logit_train_ds), size=(1,)).item()
                    generation_samples = [self.logit_train_ds[ix]]
                elif self.token_train_ds:
                    ix = torch.randint(high=len(self.token_train_ds), size=(1,)).item()
                    generation_samples = [self.token_train_ds[ix]]
                else:
                    generation_samples = []
                if generation_samples:
                    generate_answers(self.base_llm, generation_samples, accelerator)

            if step == self.max_steps:
                print("Finishing training, saving model")
                break

            is_logging = step % hp.log_interval == 0
            train_metrics = {}
            self._update_lr(step, train_metrics)

            step_t0 = time.perf_counter()

            # Logit loss
            if hp.logit_loss_weight:
                for _ in range(hp.n_logit_micro_batches_per_batch):
                    batch = next(iter_logit)
                    logit_loss = self._compute_logit_loss(batch)
                    logit_loss = logit_loss.mean()
                    loss = hp.logit_loss_weight * logit_loss / hp.n_logit_micro_batches_per_batch
                    accelerator.backward(loss)
                if is_logging:
                    train_metrics['logit_loss'] = logit_loss.item()

            # Token loss
            if hp.token_loss_weight:
                for _ in range(hp.n_token_micro_batches_per_batch):
                    batch = next(iter_token)
                    token_loss = self._compute_token_loss(batch)
                    loss = hp.token_loss_weight * token_loss / hp.n_token_micro_batches_per_batch
                    accelerator.backward(loss)
                if is_logging:
                    train_metrics['token_loss'] = token_loss.item()

            if hp.max_grad_norm:
                accelerator.clip_grad_norm_(self.student.parameters(), hp.max_grad_norm)

            torch.cuda.empty_cache()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self._log_step(step, is_logging, train_metrics, step_t0, all_losses, step_times)

            # Checkpointing
            if hp.save_during_training:
                current_time = time.time()
                if hp.checkpoint_interval and (step + 1) % hp.checkpoint_interval == 0:
                    self.save()
                if hp.checkpoint_interval_seconds and current_time - last_checkpoint_time >= hp.checkpoint_interval_seconds:
                    n_saves += 1
                    save_idx = n_saves * hp.checkpoint_interval_seconds
                    subfolder = Path(os.path.join(hp.run_project_dir, f"checkpoint_time_{save_idx}"))
                    self.save(subfolder)
                    last_checkpoint_time = current_time

        if hp.save:
            self.save()

        print(f"Training time: {(time.perf_counter() - train_t0):.2f}s", flush=True)

    # ── Loss ───────────────────────────────────────────────

    def _compute_token_loss(self, batch, reduction="batch"):
        return compute_token_loss(batch, self.student, closed_book=self.hp.closed_book_token_loss, reduction=reduction)

    def _compute_logit_loss(self, batch):
        return compute_logit_loss(batch, self.student, self.teacher, temperature=self.hp.train_temperature, reverse_kl=self.hp.reverse_kl)

    # ── Validation ─────────────────────────────────────────

    def validate(self):
        self.student.eval()
        metrics_total = {}
        metrics_by_group = {}
        with torch.no_grad():
            if self.token_val_loader is not None:
                group_names = self.token_val_loader.dataset.lesson_names
                aggregator = Aggregator(group_names, self.accelerator.device)
                for _, batch in enumerate(self.token_val_loader, start=1):
                    token_loss = self._compute_token_loss(batch, reduction="sample")
                    aggregator.add_batch(batch["lesson_ixs"], {"val_token_loss": token_loss}, self.accelerator)
                total, by_group = aggregator.get_average()
                metrics_total.update(total)
                metrics_by_group.update(by_group)

            if self.logit_val_loader is not None:
                group_names = self.logit_val_loader.dataset.lesson_names
                aggregator = Aggregator(group_names, self.accelerator.device)
                for _, batch in enumerate(self.logit_val_loader, start=1):
                    logit_loss = self._compute_logit_loss(batch)
                    aggregator.add_batch(batch["lesson_ixs"], {"val_logit_loss": logit_loss}, self.accelerator)
                total, by_group = aggregator.get_average()
                metrics_total.update(total)
                metrics_by_group.update(by_group)

        self.student.train()
        return metrics_total, metrics_by_group

    # ── Save ───────────────────────────────────────────────

    def save(self, path=None):
        path = path or self.hp.run_project_dir
        if self.hp.deepspeed_path:
            save_with_deepspeed(self.student, self.accelerator, self.base_llm, path)
        else:
            save_with_base_model_config(self.student, self.base_llm, path)

    # ── Helpers ────────────────────────────────────────────

    def _update_lr(self, step, train_metrics):
        hp = self.hp
        if step <= hp.warmup_steps and hp.warmup_steps:
            lr = hp.learning_rate * step / hp.warmup_steps
        elif hp.decay:
            multiplier = max(0.0, float(self.max_steps - step) / float(max(1, self.max_steps - hp.warmup_steps)))
            lr = hp.learning_rate * multiplier
        else:
            lr = hp.learning_rate
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        if step % hp.log_interval == 0:
            train_metrics['lr'] = lr

    def _log_to_wandb(self, metrics_total, metrics_by_group, step):
        if self.hp.log_to_wandb and self.accelerator.is_main_process:
            wandb.log(metrics_total, step=step)
            wandb.log(metrics_by_group, step=step)

    def _log_step(self, step, is_logging, train_metrics, step_t0, all_losses, step_times):
        if not (is_logging and self.accelerator.is_main_process):
            return
        t1 = time.perf_counter()
        step_times.append((t1 - step_t0) * 1000)
        loss_val = train_metrics.get('logit_loss', train_metrics.get('token_loss', None))
        all_losses.append(loss_val)
        avg_loss = sum(all_losses) / len(all_losses)
        loss_type = 'logit' if 'logit_loss' in train_metrics else 'token'
        print(
            f"Step {step + 1}/{self.max_steps}: {loss_type} loss {loss_val:.8f}, "
            f"iter time: {(t1 - step_t0) * 1000:.2f}ms, "
            f"avg iter time: {np.mean(step_times):.2f}ms, "
            f"Total avg loss {avg_loss:.4f}",
            flush=True,
        )
        if self.hp.log_to_wandb:
            train_metrics['step_time'] = (t1 - step_t0) * 1000
            wandb.log(train_metrics, step=step)


# ── Module-level helpers ──────────────────────────────────

def _get_ds_plugin(ds_config_path: str) -> DeepSpeedPlugin:
    with open(ds_config_path, "r", encoding="utf-8") as f:
        return DeepSpeedPlugin(hf_ds_config=json.load(f))


def _model_is_bf16(model: PreTrainedModel) -> bool:
    return all(p.dtype == torch.bfloat16 for p in model.parameters())
