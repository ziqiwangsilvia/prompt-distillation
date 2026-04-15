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
    """Build (student_len, teacher_len) alignment matrix from character offsets.

    Each row is a distribution over teacher tokens that overlap the student
    token's span. Weights are proportional to character overlap.
    EOS is excluded — end tokens are learned via token loss only.
    """
    s_enc = student_tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    t_enc = teacher_tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    s_spans = s_enc["offset_mapping"]
    t_spans = t_enc["offset_mapping"]

    weights = torch.zeros(len(s_spans), len(t_spans))
    for i, (s_start, s_end) in enumerate(s_spans):
        for j, (t_start, t_end) in enumerate(t_spans):
            overlap = min(s_end, t_end) - max(s_start, t_start)
            if overlap > 0:
                weights[i, j] = overlap
        row_sum = weights[i].sum()
        if row_sum > 0:
            weights[i] /= row_sum
    return weights


def build_shared_alignment(student_tokenizer, teacher_tokenizer,
                           student_text: str, teacher_text: str, shared_text: str):
    """Build alignment on shared content only, returning alignment matrix and index masks.

    Finds where shared_text characters appear in each native text, then builds
    alignment only on tokens covering those shared character ranges.

    Returns (alignment_matrix, student_indices, teacher_indices) where indices
    are lists of token positions in the full masked logits that participate.
    """
    # Find shared_text offset within each native text
    # Shared pieces are: everything except format wrappers and "parameters"/"arguments" key
    # Use difflib to find matching character blocks
    import difflib
    s_matcher = difflib.SequenceMatcher(None, student_text, shared_text)
    t_matcher = difflib.SequenceMatcher(None, teacher_text, shared_text)

    # Build set of character positions in each native text that map to shared content
    s_shared_chars = set()
    for block in s_matcher.get_matching_blocks():
        for k in range(block.size):
            s_shared_chars.add(block.a + k)

    t_shared_chars = set()
    for block in t_matcher.get_matching_blocks():
        for k in range(block.size):
            t_shared_chars.add(block.a + k)

    # Tokenize native texts
    s_enc = student_tokenizer(student_text, return_offsets_mapping=True, add_special_tokens=False)
    t_enc = teacher_tokenizer(teacher_text, return_offsets_mapping=True, add_special_tokens=False)
    s_spans = s_enc["offset_mapping"]
    t_spans = t_enc["offset_mapping"]

    # Find tokens that overlap shared content (majority of token chars are shared)
    def _shared_indices(spans, shared_chars):
        indices = []
        for i, (start, end) in enumerate(spans):
            if end <= start:
                continue
            n_shared = sum(1 for c in range(start, end) if c in shared_chars)
            if n_shared > (end - start) / 2:
                indices.append(i)
        return indices

    s_indices = _shared_indices(s_spans, s_shared_chars)
    t_indices = _shared_indices(t_spans, t_shared_chars)

    # Map native char positions to shared_text char positions for alignment
    # Build reverse map: native char -> shared char
    s_to_shared = {}
    for block in s_matcher.get_matching_blocks():
        for k in range(block.size):
            s_to_shared[block.a + k] = block.b + k

    t_to_shared = {}
    for block in t_matcher.get_matching_blocks():
        for k in range(block.size):
            t_to_shared[block.a + k] = block.b + k

    # Build alignment: for each selected student token, find overlapping selected teacher tokens
    # using shared_text character space
    n_s = len(s_indices)
    n_t = len(t_indices)
    weights = torch.zeros(n_s + 1, n_t + 1)  # +1 for EOS

    for wi, si in enumerate(s_indices):
        s_start, s_end = s_spans[si]
        # Map to shared char range
        s_shared_range = set()
        for c in range(s_start, s_end):
            if c in s_to_shared:
                s_shared_range.add(s_to_shared[c])

        for wj, tj in enumerate(t_indices):
            t_start, t_end = t_spans[tj]
            t_shared_range = set()
            for c in range(t_start, t_end):
                if c in t_to_shared:
                    t_shared_range.add(t_to_shared[c])

            overlap = len(s_shared_range & t_shared_range)
            if overlap > 0:
                weights[wi, wj] = overlap

        row_sum = weights[wi].sum()
        if row_sum > 0:
            weights[wi] /= row_sum

    weights[-1, -1] = 1.0  # EOS -> EOS

    # Add EOS index (last position in masked logits)
    return weights, s_indices, t_indices
