from contextlib import nullcontext
from typing import Dict, List, Literal, Optional, Union

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

from training import IGNORE_INDEX
from training.projection import build_alignment_weights, build_shared_alignment


def compute_token_loss(
    batch: Dict[str, torch.Tensor],
    model: PreTrainedModel,
    closed_book: bool = False,
    reduction: Literal["batch", "sample"] = "batch",
) -> torch.Tensor:
    if closed_book:
        inputs = batch.get('closed_book_seqs', batch.get('student_closed_seqs'))[..., :-1]
        labels = batch.get('closed_book_labels', batch.get('student_closed_labels'))[..., 1:]
    else:
        inputs = batch.get('open_book_seqs', batch.get('student_open_seqs'))[..., :-1]
        labels = batch.get('open_book_labels', batch.get('student_open_labels'))[..., 1:]

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
    use_tool_token: bool = False,
    accelerator=None,
) -> torch.Tensor:
    student_inputs, student_masks, batch_size, seq_length = _get_student_masks(batch, closed_book)
    teacher, teacher_context, is_self = _resolve_teacher(student, teacher)

    teacher_logits, teacher_masks = _forward_teacher(teacher, teacher_context, batch, accelerator=accelerator)
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
            projection, temperature, reverse_kl, use_tool_token=use_tool_token,
            student_answer_texts=batch.get('student_answer_texts'),
            teacher_answer_texts=batch.get('teacher_answer_texts'),
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


def _forward_teacher(teacher, teacher_context, batch, accelerator=None):
    inputs = batch['teacher_seqs'][..., :-1]
    masks = batch['teacher_masks'][..., 1:]
    use_distributed = accelerator is not None and accelerator.num_processes > 1

    if use_distributed:
        import torch.distributed as dist
        device = inputs.device

        # All ranks send their seq_len to rank 0
        local_seq_len = torch.tensor([inputs.shape[1]], device=device)
        all_seq_lens = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(accelerator.num_processes)]
        dist.all_gather(all_seq_lens, local_seq_len)
        max_seq = max(s.item() for s in all_seq_lens)

        # Pad and gather inputs on rank 0
        padded = torch.zeros(inputs.shape[0], max_seq, dtype=inputs.dtype, device=device)
        padded[:, :inputs.shape[1]] = inputs
        gathered = [torch.zeros_like(padded) for _ in range(accelerator.num_processes)]
        dist.all_gather(gathered, padded)

        # Rank 0 runs teacher forward on each rank's inputs
        if teacher is not None:
            all_logits = []
            with torch.no_grad(), teacher_context:
                teacher.eval()
                teacher_device = next(teacher.parameters()).device
                for rank_input, seq_len in zip(gathered, all_seq_lens):
                    sl = seq_len.item()
                    out = teacher.forward(rank_input[:, :sl].to(teacher_device))
                    lgt = (out.logits if hasattr(out, 'logits') else out[0]).detach()
                    all_logits.append(lgt.to(device))
            vocab_size = all_logits[0].shape[-1]
        else:
            all_logits = None
            vocab_size = 0

        # Broadcast vocab size so non-main ranks can allocate
        vs = torch.tensor([vocab_size], device=device)
        dist.broadcast(vs, src=0)
        vocab_size = vs.item()

        # Scatter: rank 0 sends each rank its logits
        my_rank = accelerator.process_index
        my_seq_len = inputs.shape[1]
        logit_dtype = all_logits[0].dtype if all_logits is not None else torch.bfloat16
        my_logits = torch.zeros(inputs.shape[0], my_seq_len, vocab_size, dtype=logit_dtype, device=device)

        for r in range(accelerator.num_processes):
            sl = all_seq_lens[r].item()
            buf = torch.zeros(inputs.shape[0], sl, vocab_size, dtype=logit_dtype, device=device)
            if all_logits is not None:
                buf.copy_(all_logits[r])
            dist.broadcast(buf, src=0)
            if r == my_rank:
                my_logits = buf

        logits = my_logits
    else:
        with torch.no_grad(), teacher_context:
            teacher.eval()
            teacher_device = next(teacher.parameters()).device
            output = teacher.forward(inputs.to(teacher_device))
            logits = output.logits if hasattr(output, 'logits') else output[0]
            logits = logits.detach().to(inputs.device)

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
                projection, temperature, reverse_kl, use_tool_token=False,
                student_answer_texts=None, teacher_answer_texts=None):
    """Per-sample KL with character-span alignment for cross-tokenizer distillation."""
    losses = []
    for i in range(student_logits.size(0)):
        s_logits_i = student_logits[i][student_masks[i]]
        t_logits_i = teacher_logits[i][teacher_masks[i]]
        align_text = teacher_answers[i]
        is_tool = use_tool_token and align_text.lstrip().startswith('{"')

        if is_tool and student_answer_texts and teacher_answer_texts:
            # Shared-content alignment: trim format tokens, align on common content
            s_text = student_answer_texts[i]
            t_text = teacher_answer_texts[i]
            align, s_idx, t_idx = build_shared_alignment(
                student_tokenizer, teacher_tokenizer, s_text, t_text, align_text)
            # +1 to skip <|python_tag|> prefix token
            s_sel = [s_logits_i[j + 1] for j in s_idx]
            t_sel = [t_logits_i[j] for j in t_idx]
        else:
            align, s_idx, t_idx = build_alignment_weights(student_tokenizer, teacher_tokenizer, align_text)
            s_sel = [s_logits_i[j] for j in s_idx]
            t_sel = [t_logits_i[j] for j in t_idx]

        s_logits_i = torch.stack(s_sel)
        t_logits_i = torch.stack(t_sel)

        # Project vocab before alignment if needed
        if projection is not None:
            t_logits_i = projection(t_logits_i)

        # Align in probability space
        t_probs = F.softmax(t_logits_i / temperature, dim=-1)
        align = align.to(device=t_probs.device, dtype=t_probs.dtype)
        if align.shape[1] != t_probs.shape[0]:
            print(f"ALIGN MISMATCH: align={align.shape}, t_probs={t_probs.shape}, s_logits={s_logits_i.shape}", flush=True)
            print(f"  is_tool={is_tool}, align_text={repr(align_text[:100])}", flush=True)
            print(f"  teacher_answer={repr(teacher_answers[i][:100])}", flush=True)
            if student_answer_texts:
                print(f"  student_text={repr(student_answer_texts[i][:100])}", flush=True)
            if teacher_answer_texts:
                print(f"  teacher_text={repr(teacher_answer_texts[i][:100])}", flush=True)
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
