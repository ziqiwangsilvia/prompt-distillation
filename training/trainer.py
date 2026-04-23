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
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from peft import get_peft_model
from transformers import PreTrainedModel
import wandb

from models.llm import LLM
from models.utils import DualOutput, num_parameters
from evaluation.metrics import Aggregator
from data.dataloader import build_dataloaders
from training.projection import VocabProjection, TopKProjection
from training.loss import compute_token_loss, compute_logit_loss
from training import PADDING_VALUE
from training.utils import (
    generate_answers,
    extract_primitive_config,
    save_with_base_model_config,
    print_token_tensor,
)


class Trainer:
    def __init__(self, base_llm: LLM, data: Tuple[List[str], List[str]], hp: SimpleNamespace, teacher_llm: LLM = None, tools: list = None):
        self.base_llm = base_llm
        self.teacher_llm = teacher_llm
        self.teacher_tokenizer = teacher_llm.tokenizer if teacher_llm else None
        self.hp = hp
        self.accelerator = self._init_run()

        # Data
        self.logit_train_ds, self.token_train_ds, \
            self.logit_loader, self.token_loader, \
            self.logit_val_loader, self.token_val_loader = build_dataloaders(base_llm, data, hp, teacher_llm=teacher_llm, tools=tools)

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
        self.student, self.teacher, self.optimizer, self.projection = self._init_models()

        # Steps
        if hp.distillation_type == "on_policy" or hp.logit_loss_weight:
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

        accelerator = Accelerator(
            mixed_precision=hp.mixed_precision,
            project_dir=hp.run_project_dir,
            project_config=project_cfg,
        )

        hp.devices = accelerator.state.num_processes
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

    def _init_models(self):
        hp = self.hp
        base_llm = self.base_llm
        accelerator = self.accelerator

        self.log("Loading student model")
        student = base_llm.load_model(training=True)
        self.log(f"Trainable params: {num_parameters(student, True):,},  Frozen params: {num_parameters(student, False):,}")
        self.log("Preparing student LoRA")

        student = get_peft_model(student, hp.peft_config)
        if hp.mixed_precision == "bf16":
            student = student.to(torch.bfloat16)

        self.log("PEFT student built.")
        self.log(student)
        self.log(f"Trainable params: {num_parameters(student, True):,},  Frozen params: {num_parameters(student, False):,}")

        optimizer = torch.optim.AdamW(student.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay)

        accelerator.register_for_checkpointing(student)
        student, optimizer = accelerator.prepare(student, optimizer)

        teacher = None
        projection = None
        if hp.distillation_type == "on_policy" or hp.logit_loss_weight:
            if hp.teacher in {"student", "student_base"}:
                teacher = hp.teacher
            else:
                # Only main process loads teacher
                if accelerator.is_main_process:
                    teacher_gpu_ids = [int(g) for g in hp.teacher_gpus.split(",")]
                    max_memory = {g: "140GiB" for g in teacher_gpu_ids}
                    self.log(f"Loading teacher on GPUs {teacher_gpu_ids}...")
                    teacher = self.teacher_llm.load_model(training=False, device_map="auto", max_memory=max_memory)
                    teacher.eval()
                    self.log("Teacher loaded")
                else:
                    teacher = None

                # Projection for cross-family distillation (different vocab sizes)
                import torch.distributed as dist
                if teacher is not None:
                    teacher_cfg = teacher.module.config if hasattr(teacher, 'module') else teacher.config
                    tv = torch.tensor([teacher_cfg.vocab_size], device=accelerator.device)
                else:
                    tv = torch.tensor([0], device=accelerator.device)
                if accelerator.num_processes > 1:
                    dist.broadcast(tv, src=0)
                teacher_vocab = tv.item()
                student_vocab = student.module.config.vocab_size if hasattr(student, 'module') else student.config.vocab_size
                if teacher_vocab != student_vocab:
                    self.log(f"Vocab mismatch: teacher={teacher_vocab}, student={student_vocab}. Using vocab_mapping={hp.vocab_mapping}.")
                    if hp.vocab_mapping == "topk":
                        projection = TopKProjection(
                            self.teacher_llm.tokenizer, base_llm.tokenizer,
                            k=hp.n_topk, temperature=hp.train_temperature,
                            student_vocab_size=student_vocab)
                    else:
                        projection = VocabProjection(teacher_vocab, student_vocab)
                        if hp.vocab_mapping == "svd" and accelerator.is_main_process:
                            from training.projection import init_projection_from_tokenizers
                            self.log("Initializing projection from shared tokens (SVD)...")
                            init_projection_from_tokenizers(projection, self.teacher_llm.tokenizer, base_llm.tokenizer)
                        if hp.mixed_precision == "bf16":
                            projection = projection.to(torch.bfloat16)
                        projection = accelerator.prepare(projection)
                        accelerator.register_for_checkpointing(projection)
                        optimizer.add_param_group({"params": projection.parameters()})

        self.log("Student bf16:", _model_is_bf16(student))

        return student, teacher, optimizer, projection

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
                if self.logit_train_ds:
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

            # Distillation
            if hp.distillation_type == "on_policy":
                batch = next(iter_logit)
                on_policy_loss_val = self._on_policy_step(batch)
                if is_logging:
                    train_metrics['on_policy_loss'] = on_policy_loss_val
            elif hp.logit_loss_weight:
                for _ in range(hp.n_logit_micro_batches_per_batch):
                    batch = next(iter_logit)
                    logit_loss = self._compute_logit_loss(batch)
                    logit_loss = logit_loss.mean()
                    loss = hp.logit_loss_weight * logit_loss / hp.n_logit_micro_batches_per_batch
                    accelerator.backward(loss)
                if is_logging:
                    train_metrics['logit_loss'] = logit_loss.item()

            # Token loss (SFT)
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
        return compute_token_loss(batch, self.student, closed_book=self.hp.closed_book, reduction=reduction)

    def _compute_logit_loss(self, batch):
        return compute_logit_loss(
            batch, self.student, self.teacher,
            temperature=self.hp.train_temperature, reverse_kl=self.hp.reverse_kl,
            closed_book=self.hp.closed_book, projection=self.projection,
            student_tokenizer=self.base_llm.tokenizer,
            teacher_tokenizer=self.teacher_tokenizer,
            use_tool_token=self.hp.use_tool_token,
            accelerator=self.accelerator,
        )

    def _on_policy_step(self, batch):
        """On-policy DPO: student generates N responses, teacher ranks by perplexity.

        Returns scalar loss value (float) for logging.
        """
        from training.dpo import dpo_loss
        from training import IGNORE_INDEX
        from data.tool_call_format import prepare_answer_tokens
        from data.tool_call_format import extract_tool_call, format_tool_call

        accelerator = self.accelerator
        hp = self.hp
        device = accelerator.device
        student_family = self.base_llm.model_family
        teacher_family = self.teacher_llm.model_family if self.teacher_llm else student_family
        n_candidates = hp.n_candidates
        unwrapped_student = accelerator.unwrap_model(self.student)

        prompt_key = 'student_closed_seqs' if hp.closed_book else 'student_open_seqs'
        label_key = 'student_closed_labels' if hp.closed_book else 'student_open_labels'
        seqs = batch[prompt_key]
        labels = batch[label_key]
        teacher_prompts = batch['teacher_seqs']
        teacher_masks = batch['teacher_masks']
        batch_size = seqs.size(0)

        # Extract prompt-only tokens: first non-IGNORE_INDEX position marks answer start
        prompt_list = []
        for i in range(batch_size):
            answer_positions = (labels[i] != IGNORE_INDEX).nonzero(as_tuple=False)
            prompt_len = answer_positions[0].item() if len(answer_positions) > 0 else seqs.size(1)
            p = seqs[i:i+1, :prompt_len]
            mask = (p[0] != PADDING_VALUE)
            p = p[:, mask]
            prompt_list.append(p)

        # Each rank generates N candidates for its own prompts
        unwrapped_student.eval()
        all_candidates = []
        with torch.no_grad():
            terminators = self.base_llm.get_terminators()
            for i in range(batch_size):
                candidates = []
                for _ in range(n_candidates):
                    out = unwrapped_student.generate(
                        input_ids=prompt_list[i], max_new_tokens=hp.dpo_max_new_tokens,
                        temperature=hp.dpo_generation_temperature,
                        do_sample=True, eos_token_id=terminators)
                    new_tokens = out[0, prompt_list[i].size(1):]
                    candidates.append(self.base_llm.tokenizer.decode(new_tokens, skip_special_tokens=False))
                all_candidates.append(candidates)
        unwrapped_student.train()

        # Teacher ranks candidates — gather to rank 0, score, scatter back
        chosen_idxs, rejected_idxs = self._teacher_rank_candidates(
            all_candidates, teacher_prompts, teacher_masks,
            student_family, teacher_family, batch_size, n_candidates)

        torch.cuda.empty_cache()

        # Build DPO pairs and compute loss
        def _to_student_tokens(raw_text):
            tc = extract_tool_call(raw_text, student_family)
            if tc is not None:
                text = format_tool_call(tc, student_family)
                return prepare_answer_tokens(self.base_llm, text, 0, truncated=False, use_tool_token=hp.use_tool_token).to(device)
            text = raw_text
            for tag in ['<|python_tag|>', '<|eom_id|>', '<|eot_id|>']:
                text = text.replace(tag, '')
            return prepare_answer_tokens(self.base_llm, text.strip(), 0, truncated=False, use_tool_token=False).to(device)

        n_valid = 0
        total_loss_val = 0.0
        for i in range(batch_size):
            if chosen_idxs[i] == rejected_idxs[i]:
                continue
            p = prompt_list[i].to(device)
            chosen_tokens = _to_student_tokens(all_candidates[i][chosen_idxs[i]])
            rejected_tokens = _to_student_tokens(all_candidates[i][rejected_idxs[i]])

            # dpo_loss calls backward internally for memory efficiency
            loss_val = dpo_loss(unwrapped_student, p, chosen_tokens, rejected_tokens, beta=hp.dpo_beta)
            total_loss_val += loss_val
            n_valid += 1

        if n_valid > 0:
            # Average gradients over batch (backward was called per sample)
            for param in unwrapped_student.parameters():
                if param.grad is not None:
                    param.grad.div_(batch_size)
            if accelerator.num_processes > 1:
                import torch.distributed as dist
                for param in unwrapped_student.parameters():
                    if param.grad is not None:
                        dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

        return total_loss_val / batch_size if n_valid > 0 else 0.0

    def _teacher_rank_candidates(self, all_candidates, teacher_prompts, teacher_masks,
                                 student_family, teacher_family, batch_size, n_candidates):
        """Score candidates with teacher on rank 0, return chosen/rejected indices per sample.

        In multi-GPU: gather candidates + teacher prompts from all ranks to rank 0,
        score, scatter indices back.
        """
        accelerator = self.accelerator
        device = accelerator.device
        use_distributed = accelerator.num_processes > 1

        if use_distributed:
            import torch.distributed as dist
            import json

            # Gather candidate texts from all ranks
            payload = json.dumps(all_candidates).encode()
            local_size = torch.tensor([len(payload)], dtype=torch.long, device=device)
            all_sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(accelerator.num_processes)]
            dist.all_gather(all_sizes, local_size)

            max_size = max(s.item() for s in all_sizes)
            padded = torch.zeros(max_size, dtype=torch.uint8, device=device)
            padded[:len(payload)] = torch.tensor(list(payload), dtype=torch.uint8, device=device)
            gathered_cands = [torch.zeros(max_size, dtype=torch.uint8, device=device) for _ in range(accelerator.num_processes)]
            dist.all_gather(gathered_cands, padded)

            # Gather teacher prompts/masks (different samples per rank)
            local_seq_len = torch.tensor([teacher_prompts.shape[1]], device=device)
            all_seq_lens = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(accelerator.num_processes)]
            dist.all_gather(all_seq_lens, local_seq_len)
            max_seq = max(s.item() for s in all_seq_lens)

            padded_tp = torch.zeros(batch_size, max_seq, dtype=teacher_prompts.dtype, device=device)
            padded_tp[:, :teacher_prompts.shape[1]] = teacher_prompts
            gathered_tp = [torch.zeros_like(padded_tp) for _ in range(accelerator.num_processes)]
            dist.all_gather(gathered_tp, padded_tp)

            padded_tm = torch.zeros(batch_size, max_seq, dtype=teacher_masks.dtype, device=device)
            padded_tm[:, :teacher_masks.shape[1]] = teacher_masks
            gathered_tm = [torch.zeros_like(padded_tm) for _ in range(accelerator.num_processes)]
            dist.all_gather(gathered_tm, padded_tm)

            # Rank 0 scores all ranks' candidates with their corresponding teacher prompts
            if accelerator.is_main_process:
                all_ranks_chosen = []
                all_ranks_rejected = []
                for r in range(accelerator.num_processes):
                    rank_cands = json.loads(bytes(gathered_cands[r][:all_sizes[r].item()].cpu().tolist()))
                    sl = all_seq_lens[r].item()
                    rank_tp = gathered_tp[r][:, :sl]
                    rank_tm = gathered_tm[r][:, :sl]
                    c, rej = self._score_with_teacher(
                        rank_cands, rank_tp, rank_tm,
                        student_family, teacher_family, batch_size, n_candidates)
                    all_ranks_chosen.extend(c)
                    all_ranks_rejected.extend(rej)
                idx_tensor = torch.tensor(all_ranks_chosen + all_ranks_rejected, device=device)
            else:
                total = batch_size * accelerator.num_processes
                idx_tensor = torch.zeros(total * 2, dtype=torch.long, device=device)

            dist.broadcast(idx_tensor, src=0)
            total = batch_size * accelerator.num_processes
            my_rank = accelerator.process_index
            chosen_idxs = idx_tensor[my_rank * batch_size:(my_rank + 1) * batch_size].tolist()
            rejected_idxs = idx_tensor[total + my_rank * batch_size:total + (my_rank + 1) * batch_size].tolist()
        else:
            chosen_idxs, rejected_idxs = self._score_with_teacher(
                all_candidates, teacher_prompts, teacher_masks,
                student_family, teacher_family, batch_size, n_candidates)

        return chosen_idxs, rejected_idxs

    def _score_with_teacher(self, candidates, teacher_prompts, teacher_masks,
                            student_family, teacher_family, batch_size, n_candidates):
        """Score a single rank's candidates with the teacher. Returns (chosen_idxs, rejected_idxs)."""
        from data.tool_call_format import extract_tool_call, format_tool_call

        chosen_idxs = [0] * batch_size
        rejected_idxs = [min(1, n_candidates - 1)] * batch_size

        if self.teacher is None or isinstance(self.teacher, str):
            return chosen_idxs, rejected_idxs

        teacher_device = next(self.teacher.parameters()).device
        # Find prompt length: index of first True (answer token) in mask
        # Mask is [False...(prompt) | True...(answer) | False...(padding)]
        with torch.no_grad():
            for i in range(batch_size):
                # First True position = prompt length
                true_positions = teacher_masks[i].nonzero(as_tuple=False)
                if len(true_positions) == 0:
                    continue  # no answer tokens, skip
                plen = true_positions[0].item()
                t_prompt = teacher_prompts[i:i+1, :plen].to(teacher_device)
                ppls = []
                for raw_text in candidates[i]:
                    tc = extract_tool_call(raw_text, student_family)
                    if tc is not None:
                        text = format_tool_call(tc, teacher_family)
                    else:
                        text = raw_text
                        for tag in ['<|python_tag|>', '<|eom_id|>', '<|eot_id|>']:
                            text = text.replace(tag, '')
                        text = text.strip()
                    c_tokens = self.teacher_tokenizer.encode(text, add_special_tokens=False, return_tensors="pt").to(teacher_device)
                    full_seq = torch.cat([t_prompt, c_tokens], dim=1)
                    logits = self.teacher.forward(full_seq[:, :-1]).logits
                    resp_logits = logits[:, plen-1:]
                    resp_targets = full_seq[:, plen:]
                    ce = torch.nn.functional.cross_entropy(
                        resp_logits.flatten(0, 1), resp_targets.flatten(), reduction='mean')
                    ppls.append(ce.item())
                chosen_idxs[i] = ppls.index(min(ppls))
                rejected_idxs[i] = ppls.index(max(ppls))

        return chosen_idxs, rejected_idxs

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

            if self.logit_val_loader is not None and self.hp.distillation_type != "on_policy":
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
        if not self.accelerator.is_main_process:
            return
        path = path or self.hp.run_project_dir
        model = self.student.module if hasattr(self.student, 'module') else self.student
        save_with_base_model_config(model, self.base_llm, path)
        # Save training config
        import json
        config = extract_primitive_config(vars(self.hp) if hasattr(self.hp, '__dict__') else self.hp)
        with open(Path(path) / "training_config.json", "w") as f:
            json.dump(config, f, indent=2)

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

def _model_is_bf16(model: PreTrainedModel) -> bool:
    return all(p.dtype == torch.bfloat16 for p in model.parameters())
