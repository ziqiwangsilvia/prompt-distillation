from contextlib import nullcontext
from typing import Dict, Literal, Optional, Union

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

from training import IGNORE_INDEX


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
) -> torch.Tensor:
    student_inputs = batch['student_seqs'][..., :-1]
    student_labels = batch['student_labels'][..., 1:]
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
        teacher_masks = batch['teacher_masks'][..., 1:]
        teacher_logits = teacher.forward(teacher_inputs).logits[teacher_masks].detach()
        t_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1).detach()

    student_logits = student.forward(student_inputs).logits[student_masks]
    s_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    if reverse_kl:
        kl = F.kl_div(t_log_probs, s_log_probs, log_target=True, reduction="none")
    else:
        kl = F.kl_div(s_log_probs, t_log_probs, log_target=True, reduction="none")

    kl_per_token = kl.sum(-1)
    loss_mx = torch.zeros(batch_size, seq_length, device=student_inputs.device, dtype=kl_per_token.dtype)
    loss_mx[student_masks] = kl_per_token
    return loss_mx.sum(-1) / student_masks.sum(-1)
