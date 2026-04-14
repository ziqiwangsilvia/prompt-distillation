import torch
import torch.nn as nn


class VocabProjection(nn.Module):
    """Projects teacher logits (teacher_vocab_size) to student vocab space (student_vocab_size)."""

    def __init__(self, teacher_vocab_size: int, student_vocab_size: int, bottleneck: int = 4096):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(teacher_vocab_size, bottleneck, bias=False),
            nn.Linear(bottleneck, student_vocab_size, bias=False),
        )

    def forward(self, teacher_logits: torch.Tensor) -> torch.Tensor:
        return self.proj(teacher_logits)


def build_alignment_weights(student_tokenizer, teacher_tokenizer, text: str) -> torch.Tensor:
    """Build (student_len+1, teacher_len+1) alignment matrix from character offsets.

    Includes EOS-to-EOS mapping. Each row is a distribution over teacher tokens
    that overlap the student token's span. Weights are proportional to character overlap.
    """
    s_enc = student_tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    t_enc = teacher_tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    s_spans = s_enc["offset_mapping"]
    t_spans = t_enc["offset_mapping"]

    weights = torch.zeros(len(s_spans) + 1, len(t_spans) + 1)
    for i, (s_start, s_end) in enumerate(s_spans):
        for j, (t_start, t_end) in enumerate(t_spans):
            overlap = min(s_end, t_end) - max(s_start, t_start)
            if overlap > 0:
                weights[i, j] = overlap
        row_sum = weights[i].sum()
        if row_sum > 0:
            weights[i] /= row_sum
    weights[-1, -1] = 1.0  # EOS -> EOS
    return weights
