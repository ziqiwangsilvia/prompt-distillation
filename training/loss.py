from contextlib import nullcontext
from typing import Dict, List, Literal, Optional, Union

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

from training import IGNORE_INDEX
from training.projection import build_alignment_weights


def compute_token_loss(
    batch: Dict[str, torch.Tensor],
    model: PreTrainedModel,
    closed_book: bool = False,
    reduction: Literal["batch", "sample"] = "batch",
) -> torch.Tensor:
    if closed_book:
        inputs = batch['closed_book_seqs'][..., :-1]
        labels = batch['closed_book_labels'][..., 1:]
    else:
        inputs = batch['open_book_seqs'][..., :-1]
        labels = batch['open_book_labels'][..., 1:]

    batch_size, seq_length = inputs.shape
    output_logits = model.forward(inputs).logits
    token_loss = F.cross_entropy(
        output_logits.flatten(0, 1), labels.flatten(0, 1),
        ignore_index=IGNORE_INDEX,
        reduction="mean" if reduction == "batch" else "none",
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
    closed_book: bool = True,
    projection=None,
    student_tokenizer=None,
    teacher_tokenizer=None,
) -> torch.Tensor:
    student_inputs, student_masks, batch_size, seq_length = _get_student_masks(batch, closed_book)
    teacher, teacher_context, is_self = _resolve_teacher(student, teacher)

    teacher_logits, teacher_masks = _forward_teacher(teacher, teacher_context, batch)
    if is_self:
        student.train()

    student_logits = student.forward(student_inputs).logits
    student_device = student_logits.device
    teacher_logits = teacher_logits.to(student_device)

    cross_tokenizer = student_tokenizer is not None and teacher_tokenizer is not None
    if cross_tokenizer:
        return _aligned_kl(
            student_logits, teacher_logits, student_masks, teacher_masks,
            batch['teacher_answers'], student_tokenizer, teacher_tokenizer,
            projection, temperature, reverse_kl,
        )

    return _flat_kl(
        student_logits, teacher_logits, student_masks, teacher_masks,
        projection, temperature, reverse_kl, batch_size, seq_length,
    )


# ── Helpers ───────────────────────────────────────────────


def _get_student_masks(batch, closed_book):
    if closed_book:
        inputs = batch['student_closed_seqs'][..., :-1]
        labels = batch['student_closed_labels'][..., 1:]
    else:
        inputs = batch['student_open_seqs'][..., :-1]
        labels = batch['student_open_labels'][..., 1:]
    masks = labels != IGNORE_INDEX
    batch_size, seq_length = inputs.shape
    return inputs, masks, batch_size, seq_length


def _resolve_teacher(student, teacher):
    if teacher == "student_base":
        return student, student.disable_adapter(), True
    if teacher == "student":
        return student, nullcontext(), True
    return teacher, nullcontext(), False


def _forward_teacher(teacher, teacher_context, batch):
    with torch.no_grad(), teacher_context:
        teacher.eval()
        inputs = batch['teacher_seqs'][..., :-1]
        masks = batch['teacher_masks'][..., 1:]
        teacher_device = next(teacher.parameters()).device
        logits = teacher.forward(inputs.to(teacher_device)).logits.detach()
    return logits, masks


def _apply_projection(logits, projection, temperature):
    if projection is not None:
        logits = projection(logits)
    log_probs = F.log_softmax(logits / temperature, dim=-1)
    if projection is None:
        log_probs = log_probs.detach()
    return log_probs


def _kl(s_log_probs, t_log_probs, reverse_kl):
    if reverse_kl:
        return F.kl_div(t_log_probs, s_log_probs, log_target=True, reduction="none")
    return F.kl_div(s_log_probs, t_log_probs, log_target=True, reduction="none")


def _aligned_kl(student_logits, teacher_logits, student_masks, teacher_masks,
                teacher_answers, student_tokenizer, teacher_tokenizer,
                projection, temperature, reverse_kl):
    """Per-sample KL with character-span alignment for cross-tokenizer distillation."""
    losses = []
    for i in range(student_logits.size(0)):
        s_logits_i = student_logits[i][student_masks[i]]
        t_logits_i = teacher_logits[i][teacher_masks[i]]

        # Project vocab before alignment if needed
        if projection is not None:
            t_logits_i = projection(t_logits_i)

        # Align in probability space
        t_probs = F.softmax(t_logits_i / temperature, dim=-1)
        align = build_alignment_weights(student_tokenizer, teacher_tokenizer, teacher_answers[i])
        align = align.to(device=t_probs.device, dtype=t_probs.dtype)
        aligned_probs = align @ t_probs
        t_lp = aligned_probs.clamp(min=1e-8).log()

        if projection is None:
            t_lp = t_lp.detach()

        s_lp = F.log_softmax(s_logits_i / temperature, dim=-1)
        losses.append(_kl(s_lp, t_lp, reverse_kl).sum(-1).mean())
    return torch.stack(losses)


def _flat_kl(student_logits, teacher_logits, student_masks, teacher_masks,
             projection, temperature, reverse_kl, batch_size, seq_length):
    """Batched flat KL for same-tokenizer distillation."""
    t_lp = _apply_projection(teacher_logits[teacher_masks], projection, temperature)
    s_lp = F.log_softmax(student_logits[student_masks] / temperature, dim=-1)

    kl_per_token = _kl(s_lp, t_lp, reverse_kl).sum(-1)
    loss_mx = torch.zeros(batch_size, seq_length, device=student_logits.device, dtype=kl_per_token.dtype)
    loss_mx[student_masks] = kl_per_token
    return loss_mx.sum(-1) / student_masks.sum(-1)
