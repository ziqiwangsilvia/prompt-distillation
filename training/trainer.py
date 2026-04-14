import json
import math
import os
import pprint
import sys
import time
from itertools import cycle
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import ProjectConfiguration
from peft import get_peft_model
from transformers import PreTrainedModel
import wandb

from models.llm import LLM
from models.utils import DualOutput, num_parameters
from evaluation.metrics import Aggregator
from data.dataloader import build_dataloaders
from training.loss import compute_token_loss, compute_logit_loss
from training.utils import (
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
        self.logit_train_ds, self.token_train_ds, \
            self.logit_loader, self.token_loader, \
            self.logit_val_loader, self.token_val_loader = build_dataloaders(base_llm, data, hp)

        token_count = len(self.token_train_ds) if self.token_train_ds else 0
        logit_count = len(self.logit_train_ds) if self.logit_train_ds else 0
        self.log(f"Training data: token loss: {token_count} examples, logit loss: {logit_count} examples\n")

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

    # ── Logging ────────────────────────────────────────────

    def log(self, *args, **kwargs):
        if self.hp.verbose:
            print(*args, flush=True, **kwargs)

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

        self.log("Run name:", hp.run_name)
        self.log(pprint.pformat(vars(hp) if hasattr(hp, '__dict__') else hp))

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

        self.log("Loading student model")
        student = base_llm.load_model(training=True, deepspeed=bool(hp.deepspeed_path))
        self.log(f"Trainable params: {num_parameters(student, True):,},  Frozen params: {num_parameters(student, False):,}")
        self.log("Preparing student LoRA")

        student = get_peft_model(student, hp.peft_config)
        if hp.mixed_precision == "bf16" and not hp.deepspeed_path:
            student = student.to(torch.bfloat16)

        self.log("PEFT student built.")
        self.log(student)
        self.log(f"Trainable params: {num_parameters(student, True):,},  Frozen params: {num_parameters(student, False):,}")

        optimizer = torch.optim.AdamW(student.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay)

        accelerator.register_for_checkpointing(student)
        student, optimizer = accelerator.prepare(student, optimizer)

        teacher = None
        if hp.logit_loss_weight:
            if hp.teacher in {"student", "student_base"}:
                teacher = hp.teacher
            else:
                teacher_llm = LLM(hp.teacher, opening_message=hp.opening_message)
                accelerator.state.select_deepspeed_plugin("teacher")
                teacher = teacher_llm.load_model(training=False, deepspeed=True)
                teacher = accelerator.prepare(teacher)
                teacher.eval()

        self.log("Student bf16:", _model_is_bf16(student))

        return student, teacher, optimizer

    # ── Training ───────────────────────────────────────────

    def train(self):
        hp = self.hp
        accelerator = self.accelerator

        self.log(f"Training for {hp.n_epochs} epochs, {self.max_steps} iterations")

        # Warmup
        if hasattr(hp, 'warmup_steps') and hp.warmup_steps is not None:
            pass
        elif hasattr(hp, 'warmup_ratio'):
            hp.warmup_steps = int(hp.warmup_ratio * self.max_steps)
        self.log(f"Learning rate warmup: {hp.warmup_steps} steps")

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
                metrics_total, metrics_by_group = self.validate()
                self.log("Validation results:", metrics_total)
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
                self.log("Finishing training, saving model")
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

        self.log(f"Training time: {(time.perf_counter() - train_t0):.2f}s")

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
        self.log(
            f"Step {step + 1}/{self.max_steps}: {loss_type} loss {loss_val:.8f}, "
            f"iter time: {(t1 - step_t0) * 1000:.2f}ms, "
            f"avg iter time: {np.mean(step_times):.2f}ms, "
            f"Total avg loss {avg_loss:.4f}",
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
