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


def init_projection_from_tokenizers(projection: VocabProjection, teacher_tokenizer, student_tokenizer):
    """Initialize projection so shared tokens map correctly via SVD of the sparse mapping."""
    t_vocab = {teacher_tokenizer.decode([i]): i for i in range(teacher_tokenizer.vocab_size)}
    s_vocab = {student_tokenizer.decode([i]): i for i in range(student_tokenizer.vocab_size)}

    # Build sparse mapping matrix M[s, t] = 1 where tokens match by string
    t_ids, s_ids = [], []
    for tok_str, s_id in s_vocab.items():
        if tok_str in t_vocab:
            t_ids.append(t_vocab[tok_str])
            s_ids.append(s_id)

    bottleneck = projection.proj[0].out_features
    teacher_vs = projection.proj[0].in_features
    student_vs = projection.proj[1].out_features

    # Build dense mapping for shared tokens: M = proj[1].weight @ proj[0].weight
    # We want M[s_id, t_id] = 1 for shared tokens. Factor via truncated SVD.
    # Construct M as sparse then do randomized SVD for the bottleneck.
    t_ids_t = torch.tensor(t_ids, dtype=torch.long)
    s_ids_t = torch.tensor(s_ids, dtype=torch.long)
    values = torch.ones(len(t_ids), dtype=torch.float32)
    M = torch.sparse_coo_tensor(
        torch.stack([s_ids_t, t_ids_t]),
        values,
        size=(student_vs, teacher_vs),
    ).to_dense()

    # Truncated SVD: M ≈ U[:, :k] @ S[:k] @ V[:k, :]
    U, S, V = torch.svd_lowrank(M, q=bottleneck)
    sqrt_S = S.sqrt()
    # proj[0].weight: [bottleneck, teacher_vs] = diag(sqrt_S) @ V^T
    # proj[1].weight: [student_vs, bottleneck] = U @ diag(sqrt_S)
    with torch.no_grad():
        projection.proj[0].weight.copy_((sqrt_S.unsqueeze(1) * V.T))
        projection.proj[1].weight.copy_((U * sqrt_S.unsqueeze(0)))

    return projection


def build_alignment_weights(student_tokenizer, teacher_tokenizer, text: str):
    """Build alignment matrix from character offsets, excluding EOS.

    Returns (alignment_matrix, student_indices, teacher_indices) — same interface
    as build_shared_alignment for consistent handling in the loss function.
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
    return weights, list(range(len(s_spans))), list(range(len(t_spans)))


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
    _SKIP = {"parameters", "arguments", "}}\n", "}}"}

    def _shared_indices(spans, shared_chars, text):
        indices = []
        for i, (start, end) in enumerate(spans):
            if end <= start:
                continue
            token_text = text[start:end]
            if token_text in _SKIP or token_text.strip().strip('"') in _SKIP:
                continue
            n_shared = sum(1 for c in range(start, end) if c in shared_chars)
            if n_shared > (end - start) / 2:
                indices.append(i)
        return indices

    s_indices = _shared_indices(s_spans, s_shared_chars, student_text)
    t_indices = _shared_indices(t_spans, t_shared_chars, teacher_text)

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
    weights = torch.zeros(n_s, n_t)

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

    return weights, s_indices, t_indices
